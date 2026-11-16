"""Tool schemas and dynamic models used by the LLM orchestration layer."""

import json
import re
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, Union
from urllib.parse import urlsplit

import structlog
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    WithJsonSchema,
    create_model,
    field_validator,
    model_serializer,
    model_validator,
)

from utils.grep_tool_contract import GREP_SOURCE_FILTER_KEYS, GREP_TOOL_PROMPT
from utils.prompts import (
    PLANNER_RESPONSE_JOIN_DESCRIPTION,
    PLANNER_RESPONSE_TASKS_DESCRIPTION,
    PLANNER_RESPONSE_TITLE_DESCRIPTION,
    PLANNER_TASK_DEPENDS_ON_DESCRIPTION,
    PLANNER_TASK_GOAL_DESCRIPTION,
    PLANNER_TASK_IDX_DESCRIPTION,
    PLANNER_TASK_TITLE_DESCRIPTION,
    PLANNER_TASK_TOOL_NAME_DESCRIPTION,
    SELF_CONTAINED_GOAL_RULE,
)
from utils.response_schema_constraints import (
    bounds_from_entries,
    coerce_direct_bounds_to_entries,
)
from utils.tool_output_schema import ReadToolOutputTool
from utils.user_facing_writing_prompt import user_facing_text_description
from utils.wait_for_bounds import (
    WAIT_FOR_AGENTS_DEFAULT_TIMEOUT_SECONDS,
    WAIT_FOR_AGENTS_MAX_TIMEOUT_SECONDS,
    WAIT_FOR_AGENTS_MIN_TIMEOUT_SECONDS,
    resolve_wait_for_agents_timeout,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Phase Tool Lists (source of truth for agent phases)
# Must be defined early so PlanActionType can be used in PlannerTask
# ============================================================================

# Mirrors helpers.agent_addressing.TASK_NAME_PATTERN (utils may not
# import domains; background_agents.blackbox asserts the two stay equal).
SPAWN_TASK_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,39}$"
SPAWN_TASK_NAME_DESCRIPTION = (
    "Required. Short lowercase slug for this child (a-z, 0-9, _; e.g. 'venue_research'), unique among "
    "the agents you spawn. Its address becomes <your path>/<task_name>, e.g. /root/outreach/draft_a."
)
RUN_IN_BACKGROUND_DESCRIPTION = (
    "Default true: the call returns immediately with the agent's address and the agent keeps "
    "working while you continue. Needing its result is not a reason to set false: call wait_for with "
    "its address to wait, and the wait also ends when the user sends a new message, so you can reply "
    "or pass it to the agent with send_message_to_agent. A foreground agent (false) holds your turn, "
    "so messages the user sends meanwhile are read only after it returns."
)
# The planner reads spawn_agent's schema only as documentation (plan tasks carry
# no arguments) and cannot assign send_message_to_agent, so its copy keeps the
# tool-neutral wording.
PLANNER_RUN_IN_BACKGROUND_DESCRIPTION = (
    "Default true: the call returns immediately with the agent's address and the agent keeps "
    "working while you continue. Set false only when you need this agent's result before you can "
    "do anything else."
)

# Tools available during PLAN phase (research + create execution plan)
PLAN_PHASE_TOOLS = [
    "read_tool_output",
    "spawn_agent",  # Only type="explore" allowed in PLAN (enforced at runtime)
    "send_message_to_user",
    "set_plan",
    "attempt_completion",
]

# Tools available during EXECUTE phase (run the plan)
EXECUTE_PHASE_TOOLS = [
    "read_tool_output",
    "spawn_agent",  # Unified: type=browser|code|general_purpose|explore
    "wait_for",
    "send_message_to_user_execute_phase",
    "display",
    "call_phone_number",
    "message_person",
    "send_message_to_agent",
    "kill_agent",
    "attempt_completion",
]

# Root-only tools the workflow adds to both phases for eligible agents
# (domains/report_bug/handler.with_report_bug); never in the shared phase lists.
ROOT_ONLY_TOOLS = ["report_bug"]

# Stable union of PLAN + EXECUTE tools for LLM schema generation.
# Passing the same tool set in every call makes the Anthropic tools array
# identical across phases, enabling cross-phase cache hits on the
# system+tools prefix. The system prompt's ENABLED/DISABLED sections
# control which tools the model actually uses.
ALL_PHASE_TOOLS = list(dict.fromkeys(PLAN_PHASE_TOOLS + EXECUTE_PHASE_TOOLS))

# Tools the planner can assign to tasks (what shows up in task.tool_name)
# These use separate names so the planner doesn't need to understand spawn_agent params.
# The DAG executor converts these to spawn_agent via TOOL_ALIASES.
PLANNER_ASSIGNABLE_TOOLS = [
    "spawn_explore_agent",
    "spawn_browser_agent",
    "spawn_code_agent",
    "spawn_general_purpose_agent",
    "spawn_general_purpose_agent_for_each",
    "wait_for",
    "call_phone_number",
    "message_person",
    "send_message_to_user_execute_phase",
    "display",
    "reasoning",  # Pure thinking/analysis step (no tool call)
]

# Planner alias -> the execute tool the DAG executor actually runs.
TOOL_ALIASES = {
    "spawn_explore_agent": "spawn_agent",
    "spawn_browser_agent": "spawn_agent",
    "spawn_code_agent": "spawn_agent",
    "spawn_general_purpose_agent": "spawn_agent",
    "spawn_general_purpose_agent_for_each": "spawn_agent",
    "wait_seconds": "wait_for",  # Backward compat for old plans
}

# Execute tools the planner can assign (via TOOL_ALIASES), in EXECUTE_PHASE_TOOLS
# order. Tools a DAG task cannot be assigned (kill_agent, send_message_to_agent,
# attempt_completion) are excluded, so the planner prompt never describes them.
PLANNER_EXECUTE_TOOLS = [
    name
    for name in EXECUTE_PHASE_TOOLS
    if name in {TOOL_ALIASES.get(alias, alias) for alias in PLANNER_ASSIGNABLE_TOOLS}
]


class PlanToolName(str, Enum):
    """The tool to execute for this task, or 'reasoning' for thinking/analysis."""

    spawn_explore_agent = "spawn_explore_agent"
    spawn_browser_agent = "spawn_browser_agent"
    spawn_code_agent = "spawn_code_agent"
    spawn_general_purpose_agent = "spawn_general_purpose_agent"
    spawn_general_purpose_agent_for_each = "spawn_general_purpose_agent_for_each"
    wait_for = "wait_for"
    call_phone_number = "call_phone_number"
    message_person = "message_person"
    send_message_to_user_execute_phase = "send_message_to_user_execute_phase"
    display = "display"
    reasoning = "reasoning"


# Runtime check to ensure PlanToolName stays in sync with PLANNER_ASSIGNABLE_TOOLS
_plan_tool_values = {e.value for e in PlanToolName}
_expected_tools = set(PLANNER_ASSIGNABLE_TOOLS)
assert _plan_tool_values == _expected_tools, (
    f"PlanToolName enum out of sync with PLANNER_ASSIGNABLE_TOOLS. "
    f"Missing: {_expected_tools - _plan_tool_values}, Extra: {_plan_tool_values - _expected_tools}"
)


# ============================================================================
# Planner LLM Response Models
# ============================================================================


class _PlannerTaskBase(BaseModel):
    """Shared validation for planner task models."""

    @model_validator(mode="before")
    @classmethod
    def _untitled_legacy_task(cls, data: Any) -> Any:
        # Plans recorded before per-task titles (Fast Mode replays) have no
        # title; the schema still requires it from live planner calls.
        if isinstance(data, dict) and "title" not in data:
            return {**data, "title": ""}
        return data


class PlannerTask(_PlannerTaskBase):
    """A single task in the planner output."""

    task_idx: str = Field(description=PLANNER_TASK_IDX_DESCRIPTION)
    title: str = Field(description=PLANNER_TASK_TITLE_DESCRIPTION)
    goal: str = Field(description=PLANNER_TASK_GOAL_DESCRIPTION)
    tool_name: PlanToolName = Field(description=PLANNER_TASK_TOOL_NAME_DESCRIPTION)
    depends_on: List[str] = Field(default_factory=list, description=PLANNER_TASK_DEPENDS_ON_DESCRIPTION)


class PlannerResponse(BaseModel):
    """Structured output from the planner LLM."""

    title: str = Field(description=PLANNER_RESPONSE_TITLE_DESCRIPTION)
    tasks: List[PlannerTask] = Field(description=PLANNER_RESPONSE_TASKS_DESCRIPTION)
    join: Literal["all", "any"] = Field(description=PLANNER_RESPONSE_JOIN_DESCRIPTION)


# ============================================================================
# Individual Tool Pydantic Models
# ============================================================================


ResponseSchemaPrimitiveType = Literal[
    "str",
    "int",
    "float",
    "bool",
    "list[str]",
    "list[int]",
    "object",
    "list[object]",
    "file_pointer",
    "list[file_pointer]",
]


class ResponseSchemaBound(BaseModel):
    kind: Literal["minItems", "maxItems", "minLength", "maxLength"] = Field(
        description="minItems/maxItems bound lists; minLength/maxLength bound strings."
    )
    value: int = Field(ge=0, description="Nonnegative bound value.")


class ResponseSchemaNestedFieldConfig(BaseModel):
    """Configuration for a nested field in an object output."""

    name: str = Field(description="Field name in snake_case.")
    type: ResponseSchemaPrimitiveType = Field(description="Nested field type.")
    properties: List["ResponseSchemaNestedFieldConfig"] = Field(default_factory=list)
    description: str = Field(default="", description="Optional semantic description; not used for object shape.")
    required: bool = Field(default=True, description="False when this property may be omitted entirely.")
    constraints: List[ResponseSchemaBound] = Field(
        default_factory=list, description="Declared bounds; empty when none."
    )
    nullable: bool = Field(default=False, description="Whether null is a valid value; the field remains required.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_bounds(cls, value: Any) -> Any:
        return coerce_direct_bounds_to_entries(value)

    @model_validator(mode="after")
    def _validate_constraints(self):
        if self.type in ("object", "list[object]") and not self.properties:
            raise ValueError("object schema fields require explicit properties")
        bounds_from_entries(self.type, self.constraints)
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if not data.get("properties"):
            data.pop("properties", None)
        data.pop("constraints", None)
        data.update(bounds_from_entries(self.type, self.constraints))
        return data


class ResponseSchemaFieldConfig(BaseModel):
    """Configuration for a single field in the output schema."""

    name: str = Field(description="Field name in snake_case.")
    type: Literal[
        "str",
        "int",
        "float",
        "bool",
        "list[str]",
        "list[int]",
        "object",
        "list[object]",
        "file_pointer",
        "list[file_pointer]",
    ] = Field(
        description=(
            "Field type. For object/list[object], set properties explicitly instead of describing subfields in prose."
        )
    )
    required: bool = Field(description="Whether this field is required in the output.")
    description: str = Field(default="", description="Optional semantic description; not used for object shape.")
    properties: List["ResponseSchemaNestedFieldConfig"] = Field(
        default_factory=list,
        description="Explicit subfields for object/list[object] outputs. Required when type is object or list[object].",
    )
    constraints: List[ResponseSchemaBound] = Field(
        default_factory=list, description="Declared bounds; empty when none."
    )
    nullable: bool = Field(default=False, description="Whether null is a valid value; the field remains required.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_bounds(cls, value: Any) -> Any:
        return coerce_direct_bounds_to_entries(value)

    @field_validator("properties", mode="before")
    @classmethod
    def _coerce_null_properties(cls, value: Any) -> Any:
        return [] if value is None else value

    @model_validator(mode="after")
    def _object_fields_require_properties(self):
        if self.type in ("object", "list[object]") and not self.properties:
            raise ValueError("object and list[object] output_schema fields require explicit properties")
        bounds_from_entries(self.type, self.constraints)
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        data.pop("constraints", None)
        data.update(bounds_from_entries(self.type, self.constraints))
        return data


from utils.final_presentation import FinalPresentation, validate_presentation


class ResponseSchemaConfig(BaseModel):
    """Structured output schema defining what data the agent must return when complete."""

    presentation: FinalPresentation | None = None

    @model_serializer(mode="wrap")
    def _serialize_presentation(self, handler):
        data = handler(self)
        if self.presentation is None:
            data.pop("presentation", None)
        return data

    @model_validator(mode="after")
    def _validate_presentation(self):
        validate_presentation(self.model_dump(exclude_none=True))
        return self

    fields: List[ResponseSchemaFieldConfig] = Field(
        description=(
            "Output fields. Use file_pointer for open-ended multi-attribute collections; keep small results inline. "
            "constraints set size bounds; required=false permits omission."
        )
    )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        if handler.mode == "serialization":
            return handler(core_schema)
        from utils.response_schema_wire import wire_json_schema
        return wire_json_schema()

    @model_validator(mode="before")
    @classmethod
    def _coerce_malformed_input(cls, value: Any) -> Any:
        from utils.response_schema_wire import decode_wire
        # Models reliably emit output_schema in three malformed shapes that the
        # canonical {"fields": [...]} form does not match. Coerce each into the
        # structured form rather than failing validation -  failure forces a costly
        # LLM retry loop (cache_hit_pct ~99% repeats the same shape) and burns the
        # entire ~50s timeout on smoke tasks like spawn-sub-agents.
        #
        # 1. JSON-encoded string of the list (Claude Sonnet 4-6 tendency):
        #    output_schema: "[{\"name\": ..., \"type\": ...}]"
        # 2. Bare list of field dicts (no {"fields": ...} wrapper):
        #    output_schema: [{"name": ..., "type": ...}]
        # 3. Flat dict {field_name: description} (Gemini, GPT-4 tendency):
        #    output_schema: {"tokyo_population": "the population of Tokyo"}
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("[", "{")):
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    return value  # let pydantic raise a clearer error
        if isinstance(value, list):
            return {"fields": value}
        if isinstance(value, dict) and "fields" not in value and value:
            if all(isinstance(v, str) for v in value.values()):
                return {
                    "fields": [{"name": k, "description": v, "type": "str", "required": True} for k, v in value.items()]
                }
        return decode_wire(value)


class WaitForTool(BaseModel):
    """Pause until a timestamp, a natural-language condition, or other agents finish or message you."""

    tool: Literal["wait_for"]
    description: str = Field(
        min_length=3,
        max_length=80,
        description=(
            "Short UI phrase completing 'Waiting for ...' (3-80 characters). Required for every wait_for call."
        ),
    )
    until_timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp to wait until. Mutually exclusive with condition and agents.",
    )
    condition: Optional[str] = Field(
        default=None,
        description="Natural language condition for an explore agent to check on periodically.",
    )
    agents: Optional[List[str]] = Field(
        default=None,
        description=(
            'Agents to wait on: addresses like "/root/outreach/draft_a", "parent" or agent ids, or ["any"] '
            "for any agent you spawned. Wakes when one finishes or messages you. Mutually exclusive with "
            "until_timestamp and condition."
        ),
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        description=(
            f"Maximum seconds to wait for agents (default {WAIT_FOR_AGENTS_DEFAULT_TIMEOUT_SECONDS}, at most "
            f"{WAIT_FOR_AGENTS_MAX_TIMEOUT_SECONDS}; values below {WAIT_FOR_AGENTS_MIN_TIMEOUT_SECONDS} are raised "
            f"to {WAIT_FOR_AGENTS_MIN_TIMEOUT_SECONDS}). Only valid with agents. For a longer wait use "
            "until_timestamp: it ends early as soon as one of your background agents finishes or an agent "
            "messages you."
        ),
    )
    check_interval_seconds: int = Field(
        default=900,
        ge=60,
        le=86400,
        description="Seconds between condition checks (60-86400, default 900). Only used with condition.",
    )
    max_checks: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of condition checks before giving up (1-100, default 20). Only used with condition.",
    )

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _clamp_agents_timeout(cls, value: Any) -> Any:
        # Codex wait floor: below the minimum is clamped (the handler notes it), above the maximum rejected.
        if value is None:
            return None
        return resolve_wait_for_agents_timeout(value)[0]

    @model_validator(mode="before")
    @classmethod
    def _coerce_july_timestamp_description(cls, value: Any) -> Any:
        # July Fast Mode rows recorded timestamp waits as
        # {"tool":"wait_for","until_timestamp":"..."} before description became
        # required. Fill a UI phrase from the timestamp so replay parses as
        # AgentAction; condition-only calls still require an explicit description.
        if not isinstance(value, dict):
            return value
        desc = value.get("description")
        if isinstance(desc, str) and desc.strip():
            return value
        ts = value.get("until_timestamp")
        if ts and not value.get("condition"):
            phrase = f"until {ts}"
            out = dict(value)
            out["description"] = phrase[:80] if len(phrase) >= 3 else "scheduled wait"
            return out
        return value

    @model_validator(mode="after")
    def check_mutual_exclusion(self) -> "WaitForTool":
        modes = [name for name in ("until_timestamp", "condition") if getattr(self, name)]
        if self.agents is not None:
            modes.append("agents")
        if len(modes) > 1:
            raise ValueError(
                f"{' and '.join(modes)} are mutually exclusive: pass exactly one of until_timestamp, condition "
                "or agents. To wait on agents, call wait_for(agents=[...]) alone (no condition polling needed)."
            )
        if not modes:
            raise ValueError("One of until_timestamp, condition or agents must be provided.")
        if self.timeout_seconds is not None and not self.agents:
            raise ValueError(
                "timeout_seconds is only valid with agents. For a timed wait use until_timestamp; "
                "for condition polling use check_interval_seconds and max_checks."
            )
        if isinstance(self.agents, list) and not [agent for agent in self.agents if agent.strip()]:
            raise ValueError('agents must list at least one address, or be ["any"].')
        if isinstance(self.agents, list) and "any" in self.agents and len(self.agents) > 1:
            raise ValueError('["any"] waits on every agent you spawned; pass it alone, not with addresses.')
        return self


class WebSearchTool(BaseModel):
    """Search the web for real-time information."""

    tool: Literal["web_search"]
    query: str = Field(description="Search query or question about current information.")


class PeopleSearchTool(BaseModel):
    """Find people and retrieve source profiles through Lantern People. Use explicit title, company and location constraints; query supplies literal keywords, not semantic instructions. For compound qualification, inspect the returned profile evidence. Use grep source='network' for the user's own contacts."""

    tool: Literal["people_search"]
    query: str = Field(
        min_length=2,
        max_length=500,
        description=(
            "Literal profile keywords (2-500 characters), e.g. 'developer tools'. "
            "Put role, employer and geography in titles, companies and location. "
            "For enrich with source_record_id this is the lookup's display context."
        ),
    )
    mode: Literal["enrich", "discover"] = Field(
        description=(
            "'enrich': retrieve ONE source profile using source_record_id. "
            "'discover': return a LIST of people matching a role, "
            "company, geography, or seniority."
        )
    )
    provider: Literal["lantern", "exa", "parallel"] = Field(
        default="lantern",
        description=(
            "Use Lantern People. Legacy exa/parallel inputs are compatibility "
            "aliases that also execute through Lantern; there is no provider fallback."
        ),
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=25,
        description=(
            "mode='discover' ONLY: max number of people to return (1-25, default "
            "10). Not used for 'enrich', which resolves a single person."
        ),
    )
    fields: Optional[List[str]] = Field(
        default=None,
        description=(
            "mode='enrich' ONLY: the attributes to return for the resolved "
            "person. These DEFINE the output params, e.g. ['current_title', "
            "'current_company','location','linkedin_url','email','education']. "
            "Unavailable attributes are null. Contact reveals are separate. "
            "If omitted, returns the native source profile."
        ),
    )
    location: Optional[str] = Field(
        default=None, min_length=2, max_length=200, description="Named geography, resolved by Lantern."
    )
    location_kind: Optional[Literal["city", "state", "metro", "country"]] = Field(
        default=None, description="Geography kind; specify for ambiguous places."
    )
    country_code: Optional[str] = Field(
        default=None, pattern="^[A-Z]{2}$", description="Two-letter country code, e.g. US."
    )
    companies: Optional[List[Annotated[str, Field(min_length=1, max_length=200)]]] = Field(
        default=None,
        min_length=1,
        max_length=10,
        description="Employer names or domains; Lantern resolves their identities.",
    )
    titles: Optional[List[Annotated[str, Field(min_length=1, max_length=200)]]] = Field(
        default=None, min_length=1, max_length=20, description="Explicit job-title constraints."
    )
    cursor: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Discover only: exact next_cursor from the same request. Never invent offsets.",
    )
    source_record_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Required for enrich: exact source record ID from an earlier search.",
    )
    generation: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Source generation returned by discovery, for exact profile retrieval.",
    )

    @model_validator(mode="after")
    def _check_mode_params(self) -> "PeopleSearchTool":
        if self.mode == "enrich" and self.limit is not None:
            raise ValueError("limit is only valid for mode='discover'.")
        if self.mode == "discover" and self.fields is not None:
            raise ValueError("fields is only valid for mode='enrich'.")
        if self.mode == "discover" and self.source_record_id is not None:
            raise ValueError("source_record_id is only valid for mode='enrich'.")
        if self.mode == "enrich" and self.cursor is not None:
            raise ValueError("cursor is only valid for mode='discover'.")
        if self.mode == "enrich" and self.source_record_id is None:
            raise ValueError("source_record_id is required for mode='enrich'.")
        if self.mode == "enrich" and any(
            value is not None
            for value in (self.location, self.location_kind, self.country_code, self.companies, self.titles)
        ):
            raise ValueError(
                "Search constraints are only valid for mode='discover'; enrich retrieves the exact source profile."
            )
        if self.generation is not None and self.source_record_id is None:
            raise ValueError("generation requires source_record_id.")
        return self


class WebFetchTool(BaseModel):
    """Fetch a URL with JavaScript rendering; a small fast model filters the page to your objective."""

    tool: Literal["web_fetch"]
    url: str = Field(description="A fully-formed valid URL to fetch.")
    objective: str = Field(
        description=(
            "What you need from this page. The page is rendered, then a small "
            "fast model produces an objective-focused summary AND picks which "
            "of the page's links to download. If you need files (PDFs, CSVs, "
            "etc.), say so explicitly in the objective, e.g. 'extract the "
            "membership PDFs', 'download the spec sheet'."
        ),
    )


class GoogleOAuthDemoTool(BaseModel):
    """Run the single synthetic Google OAuth video fixture on the root agent."""

    tool: Literal["google_workspace_run"] = Field(
        description="Complete the requested Gmail, Calendar, Drive, and account tasks in order."
    )


class ApiRequestTool(BaseModel):
    """Call a public HTTP API and return the full response body."""

    tool: Literal["api_request"]
    method: Literal["GET", "HEAD"] = Field(
        description="HTTP method for the request. Only safe read-only methods are allowed."
    )
    url: str = Field(description="Absolute http(s) URL for the API endpoint.")
    query: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional query-string parameters appended to the URL.",
    )
    body: Optional[Any] = Field(
        default=None,
        description="Optional JSON-serializable request body for non-GET methods.",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Optional non-credential HTTP headers (for example Accept or User-Agent). "
            "Do not include Authorization, API keys, cookies, or other secrets; "
            "approved credentials must go through a credential-broker binding to an allowlisted origin."
        ),
    )

    @field_validator("headers")
    @classmethod
    def _reject_credential_headers(cls, value: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if not value:
            return value
        blocked = {
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "api-key",
            "x-auth-token",
            "x-access-token",
        }
        for key in value:
            lowered = key.lower().strip()
            if (
                lowered in blocked
                or "authorization" in lowered
                or lowered.endswith("api-key")
                or lowered.endswith("access-token")
                or lowered.endswith("auth-token")
            ):
                raise ValueError(
                    f"Credential header {key!r} is not allowed on api_request; "
                    "use an approved credential-broker binding instead."
                )
        return value


_GREP_FILTER_VALUE_TYPES: Dict[str, Any] = {
    "person": str,
    "since": str,
    "until": str,
    "path": str,
    "role": Literal["USER", "ASSISTANT", "TOOL"],
    "surface": Literal["WEB", "MAC_APP", "SMS", "WHATSAPP"],
    "record_kind": Literal["message", "llm_output", "llm_context", "all"],
    "account": str,
    "min_amount": float,
    "max_amount": float,
    "status": Literal["posted", "pending", "all"],
    "include_pending_in_totals": bool,
    "document_source": str,
    "tag": str,
}


class _GrepFilterObject(BaseModel):
    """Closed filter object: strict tool surfaces cannot accept an open map, and an
    unknown key is a validation error the model can correct rather than a silent drop."""

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler: Any) -> Dict[str, Any]:
        # Absent keys stay absent: a None filter value is not a filter.
        return {key: value for key, value in handler(self).items() if value is not None}


def _grep_filter_model(name: str, keys: Any) -> Type[BaseModel]:
    fields: Dict[str, Any] = {key: (Optional[_GREP_FILTER_VALUE_TYPES[key]], None) for key in keys}
    return create_model(name, __base__=_GrepFilterObject, **fields)


_GREP_FLAT_FILTER_KEYS = tuple(dict.fromkeys(key for keys in GREP_SOURCE_FILTER_KEYS.values() for key in keys))
GrepFilters = create_model(
    "GrepFilters",
    __base__=_GrepFilterObject,
    **{key: (Optional[_GREP_FILTER_VALUE_TYPES[key]], None) for key in _GREP_FLAT_FILTER_KEYS},
    **{
        source: (Optional[_grep_filter_model(f"GrepFilters_{source}", keys)], None)
        for source, keys in GREP_SOURCE_FILTER_KEYS.items()
    },
)


class GrepTool(BaseModel):
    # The docstring is the shared contract's description; it is assigned below.

    tool: Literal["grep"]
    pattern: Optional[str] = Field(
        default=None,
        description=(
            "Regular expression over record text. Omit for agent_trajectories or "
            "agent_network to list recent records using metadata filters. "
            "Omit for calendar or drive to list a day or folder, or to read the one event or file a path names."
        ),
    )
    sources: Optional[
        List[
            Literal[
                "messages",
                "email",
                "gmail",
                "imessage",
                "whatsapp",
                "network",
                "agent_network",
                "agent_trajectories",
                "transactions",
                "documents",
                "calendar",
                "drive",
                "summary",
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "The specific data source(s) to search - pass a list and pick the exact source(s) the "
            "task needs. Do NOT default to a blunt catch-all when you know the channel; name the "
            'concrete sources (e.g. ["email"], or ["email", "imessage"]). Omit entirely to '
            "search the local defaults (messages, network, agent_network). "
            "Channels of the user's communications: email searches the user's email/Gmail; "
            "imessage searches iMessage (including SMS stored in the Mac Messages DB); "
            "whatsapp searches WhatsApp. "
            "messages is the umbrella over all communication channels (use only when the channel "
            "is genuinely unknown). network searches contact records. agent_network searches all "
            "message roles in the current agent tree and returns role/location metadata. "
            "agent_trajectories searches prior user-owned agent runs across chat, SMS, and "
            "Temporal; it excludes the current agent tree. Results carry a trace path, "
            "message role, and agent title as separate fields. "
            "Trajectory record_kind=message is a stored agent message; llm_output is a model "
            "response; llm_context is text passed to a model and may replay earlier turns. "
            "Ordinary results include messages and outputs; record_kind=all includes replayed "
            "inputs. "
            "transactions searches synced Plaid transaction rows. documents searches synced "
            "ChatGPT/Claude exports, Apple Notes, Obsidian vault notes, "
            "meeting note takers (granola.ai, fireflies.ai), "
            "and generic user documents. "
            "calendar searches the user's Google Calendar events; drive searches Google Drive "
            "file names, folders, and file text. "
            "summary returns a compact, precomputed inventory with message counts, date ranges, "
            'channel totals, and threads ranked by volume. Use sources=["summary"] first for '
            "aggregate questions such as total messages, date coverage, or most-contacted threads; "
            "then use the concrete channel source to inspect exact message text. Thread identifiers "
            "in the summary are valid contact identifiers when no human-readable name is stored. "
            "If a named source's connector isn't connected, the search returns a 'source not "
            "connected' notice - do NOT retry that source; continue with the sources you can reach."
        ),
    )
    source: Optional[
        Literal[
            "messages",
            "email",
            "gmail",
            "imessage",
            "whatsapp",
            "network",
            "agent_network",
            "agent_trajectories",
            "transactions",
            "documents",
            "calendar",
            "drive",
            "summary",
        ]
    ] = Field(
        default=None,
        description="Legacy single-source alias for `sources`. Prefer `sources` (a list of the exact sources the task needs).",
    )
    owner_user_id: Optional[str] = Field(
        default=None, description="Only this owner's records (ids from the shared data map)."
    )
    team_id: Optional[str] = Field(default=None, description="Only accounts shared with this team.")
    account_id: Optional[str] = Field(
        default=None,
        description=(
            'Only this account (exact accountId from the shared data map): an inbox email, "" (iMessage), '
            '"whatsapp", "linkedin" (via messages), "granola", "fireflies", or a bank login id. '
            "Shared bank transactions need team_id or owner_user_id."
        ),
    )
    role: Optional[Literal["USER", "ASSISTANT", "TOOL"]] = Field(
        default=None,
        description="Message role in agent_trajectories or agent_network. Implies agent_trajectories when sources are omitted.",
    )
    record_kind: Optional[Literal["message", "llm_output", "llm_context", "all"]] = Field(
        default=None,
        description="Agent trajectory record kind; message is the stored conversation record.",
    )
    surface: Optional[Literal["WEB", "MAC_APP", "SMS", "WHATSAPP"]] = Field(
        default=None,
        description="Agent trajectory origin surface.",
    )
    trace_path: Optional[str] = Field(
        default=None,
        description=(
            "Agent trajectory root path/UUID searches its entire child tree; a child path narrows to that agent, "
            "and a message path to one message. An agent address in your tree (/root/outreach, as list_agents "
            "prints) scopes agent_network or agent_trajectories to that agent and every agent below it."
        ),
    )
    order: Optional[Literal["newest_first", "oldest_first"]] = Field(
        default=None,
        description=(
            "Order messages in one agent_trajectories search. Defaults to newest_first. "
            "Ignored for other sources; display_order reports the order used."
        ),
    )
    page_after: Optional[str] = Field(
        default=None,
        description="For the next page of one agent_trajectories search, pass the previous result's next_page_after path with the same query and order.",
    )
    until: Optional[str] = Field(
        default=None,
        description=(
            "Latest timestamp (ISO 8601) for each selected source whose filters accept until: "
            "agent_trajectories, transactions, documents, calendar, drive."
        ),
    )
    require_source_access: Optional[bool] = Field(
        default=None,
        description=(
            "Agent-side gating hint for documents-provider searches. Set true only when the user "
            "explicitly requires that document provider; leave unset for opportunistic searches, "
            "so missing access does not block the task."
        ),
    )
    before: Optional[int] = Field(
        default=None,
        alias="-B",
        description="Number of logical lines to show before each match.",
    )
    after: Optional[int] = Field(
        default=None,
        alias="-A",
        description="Number of logical lines to show after each match.",
    )
    context_c: Optional[int] = Field(
        default=None,
        alias="-C",
        description="Number of logical lines to show before and after each match. Alias for context.",
    )
    context: Optional[int] = Field(
        default=None,
        description="Number of logical lines to show before and after each match.",
    )
    line_numbers: Optional[bool] = Field(
        default=None,
        alias="-n",
        description="Show logical line numbers in content output. Defaults to true.",
    )
    case_insensitive: Optional[bool] = Field(
        default=None,
        alias="-i",
        description="Enable case-insensitive regex matching.",
    )
    head_limit: Optional[int] = Field(
        default=None,
        description="Maximum number of returned result snippets. Defaults to 20.",
    )
    offset: Optional[int] = Field(
        default=None,
        description="Skip this many matched snippets before applying head_limit.",
    )
    multiline: Optional[bool] = Field(
        default=None,
        description="Allow regex patterns to match across newlines within a record.",
    )
    filters: Optional[GrepFilters] = Field(
        default=None,
        description=(
            "Per-source filters. Each source accepts only its own keys:\n"
            "  messages (and email, gmail, imessage, whatsapp): person, since, path\n"
            "  network: path\n"
            "  agent_network: path, role\n"
            "  agent_trajectories: path, since, until, role, surface, record_kind\n"
            "  transactions: since, until, account, min_amount, max_amount, "
            "status ('posted', 'pending', or 'all'), include_pending_in_totals, path\n"
            "  documents: document_source, since, until, tag, path\n"
            "  calendar, drive: since, until, path\n"
            'A flat key ({"since": "7d"}) applies to every selected source that accepts it. To target one '
            'source, nest its keys under the source name: {"calendar": {"since": "2026-11-02"}}; a messages '
            "block covers email, gmail, imessage and whatsapp. "
            "documents is an umbrella corpus, so set document_source to the concrete provider "
            "(chatgpt, claude, apple_notes, obsidian, granola, or fireflies), or use meeting_notes "
            "for meeting note takers (granola.ai, fireflies.ai). "
            "granola and fireflies each cover that provider's meetings; granola_note, "
            "granola_summary, granola_transcript, fireflies_note, fireflies_summary, or "
            "fireflies_transcript searches just that layer. "
            "`path` takes a path from an earlier result "
            "(e.g. 'network/Andy Prevalsky.contact', 'messages/+15551234567.thread') "
            "to scope that path's own source to the record or thread; for agent_network and agent_trajectories "
            "it may instead be an agent address such as /root/outreach. "
            "A calendar or drive path may also be a folder of a result path, such as "
            "'calendar/<account>/2026-10-02' or 'drive/<account>/My Drive/Legal'. "
            "Calendar since and until take a date (YYYY-MM-DD) and include that whole day; "
            "with neither, calendar lists from yesterday onward."
        ),
    )
    person: Optional[str] = Field(
        default=None,
        description=(
            "messages source only: narrow to one contact by email, phone, or full name. "
            "Ignored for network and agent_network."
        ),
    )
    since: Optional[str] = Field(
        default=None,
        description=(
            "Results at or after this time, for each selected source whose filters accept since. "
            "ISO 8601 timestamp or a relative form like 7d, 24h, 30m."
        ),
    )
    model_config = ConfigDict(populate_by_name=True)


GrepTool.__doc__ = GREP_TOOL_PROMPT


class InspectCodeTool(BaseModel):
    """Ask a read-only code agent about the files in a code agent's workspace.

    It answers from that session's latest checkpoint in a separate sandbox,
    cites file:line, and cannot change the session.
    """

    tool: Literal["inspect_code"]
    code_session_id: str = Field(description="The code_session_id from that code agent's result.")
    question: str = Field(
        description="A self-contained question about the code, naming the files, features or behavior to check."
    )


class InlineViewArg(BaseModel):
    """A published rose_user_ui inline view rendered with the message."""

    view: str = Field(description="Inline view name from the user's published rose-ui.json.")
    props: Optional[Dict[str, Any]] = Field(default=None, description="Small JSON props passed to the view.")


INLINE_VIEW_DESCRIPTION = (
    "Optional inline view from the user's published Rose UI to render with this message, such as a game, "
    "app or visual a code agent published. Use it to show the user that work in the chat."
)


class SendMessageToUserTool(BaseModel):
    """Message the user during PLAN phase; optionally block for a reply."""

    tool: Literal["send_message_to_user"]
    message: str = Field(
        description=user_facing_text_description(
            "The exact user-visible message explaining the question, blocker, or FYI."
        )
    )
    is_question: bool = Field(
        default=True,
        description=("True to block workflow until the user responds. False to send a non-blocking FYI."),
    )


class SendMessageToUserRoseDesktopTool(SendMessageToUserTool):
    """Message the user during PLAN phase; optionally block for a reply."""

    inline_view: Optional[InlineViewArg] = Field(default=None, description=INLINE_VIEW_DESCRIPTION)


class SendMessageToUserExecutePhaseTool(BaseModel):
    """Send a non-blocking informational message to the user during EXECUTE phase."""

    tool: Literal["send_message_to_user_execute_phase"]
    message: str = Field(
        description=user_facing_text_description(
            "The exact user-visible non-blocking update explaining what the user needs to know."
        )
    )


class SendMessageToUserExecutePhaseRoseDesktopTool(SendMessageToUserExecutePhaseTool):
    """Send a non-blocking informational message to the user during EXECUTE phase."""

    inline_view: Optional[InlineViewArg] = Field(default=None, description=INLINE_VIEW_DESCRIPTION)


class DisplayTool(BaseModel):
    """Show images, videos or HTML to the user inline."""

    tool: Literal["display"]
    paths: Optional[List[str]] = Field(
        default=None,
        description="Files already in this agent's tree (attachments or outputs), by file name.",
    )
    urls: Optional[List[str]] = Field(default=None, description="Public http(s) URLs of images, videos or HTML pages.")
    html: Optional[str] = Field(
        default=None,
        description="A complete HTML page you write yourself (a chart or scorecard), shown without creating a file.",
    )
    caption: Optional[str] = Field(
        default=None,
        description=user_facing_text_description("Caption shown with the media."),
    )


class DisplayRoseDesktopTool(DisplayTool):
    """Show images, videos or HTML to the user inline."""

    @field_validator("paths", "urls", mode="before")
    @classmethod
    def _decode_json_array(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return decoded if isinstance(decoded, list) else value


class MessagePersonTool(BaseModel):
    """Drafts a personalized message and stages it for user approval. Approval is synchronous."""

    tool: Literal["message_person"]
    person_id: str = Field(
        description=(
            "Person identifier: full name, E.164 phone number, or email address. "
            "Full name must resolve unambiguously to one person."
        )
    )
    channel: Literal["imessage", "agent_sms", "email"] = Field(
        description=(
            "Communication channel: 'imessage' uses the user's Mac and queues while it is offline; "
            "'agent_sms' sends immediately from the agent's own AgentPhone number and never queues; "
            "'email' is reserved for connected email delivery."
        ),
    )
    from_email: Optional[str] = Field(
        default=None,
        description=(
            "Email address to send from (required when channel='email'). "
            "Must match one of the user's connected accounts."
        ),
    )
    intent: str = Field(
        description="What the message must accomplish; the tool drafts the text from this.",
    )
    context_gates: Optional[Dict[str, bool]] = Field(
        default=None,
        description=(
            "Optional gates controlling which user-context chunks are included. Allowed key: 'recent_communications'."
        ),
    )


ForEachItem = Annotated[
    JsonValue,
    WithJsonSchema(
        {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
                {"type": "object", "additionalProperties": {}},
                {"type": "array", "items": {}},
            ]
        }
    ),
]


_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\"'`\\]+")
# Punctuation that terminates prose but is never meaningful at the end of a URL.
_URL_TRAILING_PUNCTUATION = ".,;:!?"
_URL_CLOSERS = {")": "(", "]": "[", "}": "{"}


def first_url_in_text(text: Optional[str]) -> Optional[str]:
    """Return the first http(s) URL embedded in free-form text, or None.

    Models routinely put the browsing target in prose ("Navigate to
    https://example.com/checkout. Then ...") instead of a dedicated field, so
    the URL arrives glued to sentence punctuation or wrapped in brackets. Trim
    trailing punctuation and unbalanced closing brackets so the recovered value
    is the URL the model actually meant.
    """
    if not isinstance(text, str):
        return None
    match = _URL_IN_TEXT_RE.search(text)
    if not match:
        return None

    url = match.group(0)
    while url:
        last = url[-1]
        if last in _URL_TRAILING_PUNCTUATION:
            url = url[:-1]
            continue
        opener = _URL_CLOSERS.get(last)
        if opener and url.count(last) > url.count(opener):
            url = url[:-1]
            continue
        break

    # A destination needs a host: "https://", "https:///orders" and
    # "https://?next=orders" all parse but name no server to navigate to.
    if not url:
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return None
        # `.port` is what actually validates the port; urlsplit itself accepts
        # "https://a.test:99999/" and only raises when the value is read.
        parts.port
    except ValueError:
        return None
    return url


def url_origin_for_log(url: str) -> str:
    """Scheme + host (+ port) only -  never credentials, path, query, or fragment.

    A URL recovered from free-form text can carry `user:password@` userinfo and
    secrets in the query string, so only the origin is ever safe to log.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable-url>"
    if not parts.hostname:
        return "<hostless-url>"
    origin = f"{parts.scheme}://{parts.hostname}"
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{origin}:{port}" if port else origin


# spawn_agent `type` description. The 'code' clause differs by account: Rose
# Desktop accounts (utils.rose_desktop) also learn that the code sandbox holds
# their rose_user_ui and publishes inline views; every other account gets the
# stock clause, so its tool schema carries no Rose Desktop text.
_SPAWN_TYPE_BROWSER = (
    "Select by capability: 'browser' starts a browser session for stateful or authenticated web work; "
)
_SPAWN_TYPE_CODE_STOCK = (
    "'code' starts a coding agent using Claude Code or Codex CLI in a persistent sandbox, selected by "
    "available credentials and configuration; "
)
_SPAWN_TYPE_CODE_ROSE_DESKTOP = (
    "'code' starts a coding agent using Claude Code or Codex CLI in a persistent sandbox, selected by "
    "available credentials and configuration. Delegate all code writing to it except one HTML page you "
    "can write yourself, such as a chart or scorecard, which you show with display html. Its sandbox holds the "
    "user's Rose app UI (rose_user_ui): it edits and publishes changes to how Rose looks or behaves for "
    "the user. For a game or app the user asks to play or use, have it publish the result "
    "in Agent Display using rose display and return the display receipt; a file path alone does not show it. "
)
_SPAWN_TYPE_REST = (
    "'general_purpose' starts a full autonomous agent for "
    "cross-domain multi-step work - DO NOT use it or assign it a goal unless that goal is sufficiently distinct from the goal you are tasked with, and the sub-agent's task complexity is strictly less than half of your issued task complexity (otherwise do the work yourself); Be wary of infinite recursion: general_purpose sub-agents can also spawn sub-agents. When using one as an intelligent retry for a faulty tool, instruct it not to spawn more general_purpose sub-agents; otherwise it will likely spawn another general_purpose to retry the same faulty tool after the first error. 'explore' performs read-only research for web or user-data facts; "
    "'for_each' starts one general-purpose agent per item for repeated fan-out. "
    "A 'browser' agent is given an auth inventory at session start, separately from the goal: the user's "
    "saved sessions, cookies, passwords and passkeys for the target site, and a minted agent-owned "
    "identity - an email address and phone number it can read delivered messages at, including "
    "verification codes. It can only read messages sent to that minted identity or to a mailbox the "
    "user has connected as an integration; any other address, including one you compose from the "
    "user's name, is write-only to it."
)
SPAWN_AGENT_TYPE_DESCRIPTION = _SPAWN_TYPE_BROWSER + _SPAWN_TYPE_CODE_STOCK + _SPAWN_TYPE_REST
SPAWN_AGENT_TYPE_DESCRIPTION_ROSE_DESKTOP = _SPAWN_TYPE_BROWSER + _SPAWN_TYPE_CODE_ROSE_DESKTOP + _SPAWN_TYPE_REST


BROWSER_GOAL_PROFILE_NOTE = (
    "Exception: a browser agent also receives the user's profile (name, email, phone, delivery "
    "address) and the target site's sign-in options itself; do not restate the user's profile in "
    "its goal. Keep the goal and required output fields limited to the requested outcome and "
    "evidence needed to establish it. Make supplementary observations conditional in the goal "
    "and optional in output_schema; do not turn possible site details or background facts into "
    "mandatory completion evidence. Preserve explicit and standing user constraints and all "
    "steps and evidence necessary for correct execution."
)


def _require_spawn_output_schema_key(schema: Dict[str, Any]) -> None:
    required = schema.setdefault("required", [])
    if "output_schema" not in required:
        required.append("output_schema")


_GITHUB_REPO_REF_DESCRIPTION = (
    "Normally the project id or 'gh:owner/repo' ref shown in # Available projects; a GitHub URL "
    "also works. Code agents start in that project (repo checkout, env files and startup script); "
    "spawned agents keep it as their project. Explicit selection wins over inherited references "
    "and over github_repo_link. Omit it to inherit. An unknown or unconnected ref fails with an "
    "error naming it; it never falls back to another repo. Not allowed with target='local'. "
    "Ignored when code_session_id resumes a session, which keeps its own repo."
)


class CodeSandboxSize(BaseModel):
    """Initial vCPUs / memory of a code agent's cloud sandbox."""

    model_config = ConfigDict(extra="forbid")

    cpu: Optional[int] = Field(default=None, ge=1, le=64, description="vCPUs.")
    memory_gb: Optional[int] = Field(default=None, ge=1, le=256, description="Memory in GB.")

    @model_validator(mode="after")
    def check_dimension(self) -> "CodeSandboxSize":
        if self.cpu is None and self.memory_gb is None:
            raise ValueError("sandbox_size needs cpu, memory_gb or both.")
        return self


_SANDBOX_SIZE_DESCRIPTION = (
    "Optional when type='code' with target='cloud'; ignored for every other type. Initial vCPUs / "
    "memory of a NEW code session's sandbox; the closest built size that satisfies both is used. "
    "Omit it to use the project's size or the default. A resumed session keeps the size it holds. "
    "An unbuilt or over-limit size fails with the available sizes. Disk size is not selectable."
)


class SpawnAgentTool(BaseModel):
    """Spawn an autonomous agent; type selects capabilities and per-type required fields.

    general_purpose: DO NOT use this agent type or assign it a goal unless that goal is
    sufficiently distinct from the goal you are tasked with. Each general_purpose
    sub-agent's task complexity must be strictly less than half of your issued task
    complexity; otherwise do the work yourself. Be wary of infinite recursion:
    general_purpose sub-agents can also spawn sub-agents. When using one as an
    intelligent retry for a faulty tool, instruct it not to spawn more
    general_purpose sub-agents; otherwise it will likely spawn another
    general_purpose to retry the same faulty tool after the first error.
    """

    model_config = ConfigDict(json_schema_extra=_require_spawn_output_schema_key)

    tool: Literal["spawn_agent"]
    type: Literal["browser", "code", "general_purpose", "explore", "for_each"] = Field(
        description=SPAWN_AGENT_TYPE_DESCRIPTION
    )
    task_name: str = Field(pattern=SPAWN_TASK_NAME_PATTERN, description=SPAWN_TASK_NAME_DESCRIPTION)
    goal: str = Field(
        description=(
            "The self-contained task/prompt and description for the spawned agent. "
            "Spawned agents have no access to ANY of your context by default. "
            "The goal and attachments parameters are their ONLY source of information. "
            "Include all relevant pieces of information. "
            f"{BROWSER_GOAL_PROFILE_NOTE} "
            f"{SELF_CONTAINED_GOAL_RULE.strip()}"
        )
    )

    output_schema: Optional[ResponseSchemaConfig] = Field(
        default=None,
        description=(
            "Structured data the agent returns and validates against. The agent will not stop running "
            "until this output_schema is met. Required when type='browser', type='code', "
            "type='general_purpose', or type='for_each'. Ignored when type='explore'."
        ),
    )
    attachments: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional when the spawned agent needs files from the current context. "
            "Use exact `original_filename` values from file_pointer objects."
        ),
    )

    start_url: Optional[str] = Field(
        default=None,
        description=(
            "HTTPS URL the browser opens first. Required when type='browser'; ignored otherwise. "
            "Must come from the user or a prior explore/tool result - do not invent or guess deep "
            "paths, property codes, or booking URLs. If you only know the merchant/site, use that "
            "site's homepage or search entry URL (e.g. https://www.airbnb.com/). If you do not know "
            "even the site, spawn explore first, then spawn the browser with the recovered URL."
        ),
    )
    # code only - GitHub repo to clone into sandbox
    github_repo_link: Optional[str] = Field(
        default=None,
        description=(
            "Optional when type='code'. GitHub repository URL to clone and sync to. "
            "Ignored for every other type. Only provide this when the user supplied a real, "
            "accessible repository URL; do not invent placeholder repositories. Prefer "
            "github_repo_ref with a ref from # Available projects. If omitted, uses "
            "github_repo_ref, else the project you inherited, otherwise a repo created for this "
            "agent tree."
        ),
    )
    github_repo_ref: Optional[str] = Field(
        default=None,
        description="Optional when type='code', 'general_purpose' or 'for_each'; ignored for every other type. "
        + _GITHUB_REPO_REF_DESCRIPTION,
    )
    code_session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional when type='code'. Ignored for every other type. The code_session_id a "
            "previous code-agent result reported, to resume that exact session (files, commits, "
            "branch and conversation). Omit it to resume your own latest session for the repo."
        ),
    )
    target: Optional[Literal["cloud", "local"]] = Field(
        default="cloud",
        description=(
            "Optional when type='code': 'local' runs the coding agent on the user's Mac in a folder they "
            "shared (requires an owned connected Mac; multiple Macs require explicit device selection), 'cloud' (default) runs it in a sandbox."
        ),
    )
    sandbox_size: Optional[CodeSandboxSize] = Field(default=None, description=_SANDBOX_SIZE_DESCRIPTION)
    start_phase: Optional[Literal["PLAN", "EXECUTE"]] = Field(
        default=None,
        description=(
            "Required when type='general_purpose' or type='for_each'. Ignored for every other type. "
            "The phase that the agent should spawn in as, either 'PLAN' or 'EXECUTE'. "
            "For complex tasks that require planning and more than a few tool calls, use 'PLAN'. "
            "For straightforward tasks that are simple to complete, use 'EXECUTE'. "
            "Prefer 'PLAN' when uncertain."
        ),
    )
    items: Optional[List[ForEachItem]] = Field(
        default=None,
        description=(
            "Optional alternative when type='for_each': provide either items or items_csv. "
            "Ignored for every other type. For each item in the array, "
            "a general-purpose agent will be spawned with the given goal and the item attached to its context. "
            "Each array element spawns exactly one sub-agent. Items may be any JSON value: "
            "string, number, boolean, null, object, or array."
        ),
    )
    items_csv: Optional[str] = Field(
        default=None,
        description=(
            "Optional alternative when type='for_each': provide either items_csv or items. "
            "Ignored for every other type. CSV string input. "
            "For each row in the CSV, a general-purpose agent will be spawned with the given goal and "
            "the item attached to its context."
        ),
    )
    context_gates: Optional[Dict[str, bool]] = Field(
        default=None,
        description=(
            "Optional when type='explore', type='general_purpose', or type='for_each'. "
            "Ignored when type='browser' or type='code'. Missing keys default to true, "
            "which means that context flows through to the agent. For 'explore', applicable keys "
            "are 'network' and 'recent_communications'. For 'general_purpose' and 'for_each', keys are "
            "'user_profile', 'authenticated_sessions', 'integrations', 'recent_communications', "
            "'network', 'custom_system_prompt', and 'shared_files'. Set a key to false only when "
            "that context is clearly unnecessary."
        ),
    )
    run_in_background: bool = Field(default=True, description=RUN_IN_BACKGROUND_DESCRIPTION)
    depth: Optional[Literal["quick", "medium", "deep"]] = Field(
        default="quick",
        description=(
            "Optional when type='explore'; ignored for every other type. Controls research "
            "thoroughness as a TURN budget (one turn can batch multiple parallel searches): "
            "'quick' = a basic look (~5 turns), 'medium' = moderate exploration (~20 turns), "
            "'deep' = broad analysis across many sources (~50 turns). Default 'quick'; "
            "only use 'deep' when the task genuinely needs exhaustive research."
        ),
    )

    @model_validator(mode="after")
    def check_spawn_agent_constraints(self) -> "SpawnAgentTool":
        if self.type == "browser" and not self.start_url:
            # Models very often express the browsing target inside `goal`
            # ("Navigate to https://... then check out") and leave `start_url`
            # unset. Rejecting that costs a full re-generation, and because the
            # omission is a systematic model habit rather than a random glitch,
            # the blind retry reproduces it -  three strikes then kills the agent
            # before a browser session is ever created. Recover the URL instead;
            # only fail when the goal genuinely names no destination.
            # Only https is recoverable. The browser restores authenticated
            # session cookies before it navigates, so silently promoting a
            # cleartext target inferred from prose would leak that state; an
            # http:// goal still fails loudly rather than being repaired.
            recovered = first_url_in_text(self.goal)
            if recovered and urlsplit(recovered).scheme == "https":
                self.start_url = recovered
                logger.warning(
                    "spawn_agent(type='browser') omitted start_url; recovered from goal",
                    start_url_origin=url_origin_for_log(recovered),
                )
            else:
                raise ValueError("start_url is required when type='browser'.")
        if self.type in ("general_purpose", "for_each") and not self.start_phase:
            raise ValueError("start_phase is required when type='general_purpose' or 'for_each'.")
        if self.type == "for_each":
            if self.items and self.items_csv:
                raise ValueError("items and items_csv are mutually exclusive.")
            if not self.items and not self.items_csv:
                raise ValueError("type='for_each' requires either items or items_csv.")
        if self.type in ("browser", "code", "general_purpose", "for_each") and not self.output_schema:
            raise ValueError(f"output_schema is required when type='{self.type}'.")
        # Some models emit output_schema for explore type even though it has no effect
        # downstream (explore agents return free-form summary). Drop it silently rather
        # than failing validation -  rejecting forces a costly LLM retry loop.
        if self.type == "explore" and self.output_schema:
            self.output_schema = None
        return self


class SpawnAgentRoseDesktopTool(SpawnAgentTool):
    """Spawn an autonomous agent; type selects capabilities and per-type required fields.

    general_purpose: DO NOT use this agent type or assign it a goal unless that goal is
    sufficiently distinct from the goal you are tasked with. Each general_purpose
    sub-agent's task complexity must be strictly less than half of your issued task
    complexity; otherwise do the work yourself. Be wary of infinite recursion:
    general_purpose sub-agents can also spawn sub-agents. When using one as an
    intelligent retry for a faulty tool, instruct it not to spawn more
    general_purpose sub-agents; otherwise it will likely spawn another
    general_purpose to retry the same faulty tool after the first error.
    """

    type: Literal["browser", "code", "general_purpose", "explore", "for_each"] = Field(
        description=SPAWN_AGENT_TYPE_DESCRIPTION_ROSE_DESKTOP
    )


class PlannerSpawnAgentTool(SpawnAgentTool):
    run_in_background: bool = Field(default=True, description=PLANNER_RUN_IN_BACKGROUND_DESCRIPTION)


class PlannerSpawnAgentRoseDesktopTool(SpawnAgentRoseDesktopTool):
    run_in_background: bool = Field(default=True, description=PLANNER_RUN_IN_BACKGROUND_DESCRIPTION)


PLANNER_TOOL_MODEL_OVERRIDES = {"spawn_agent": PlannerSpawnAgentTool}
PLANNER_ROSE_DESKTOP_TOOL_MODEL_OVERRIDES = {"spawn_agent": PlannerSpawnAgentRoseDesktopTool}


class SpawnAgentToolNoGP(BaseModel):
    """EXECUTE-phase spawn_agent with general_purpose / for_each removed.

    Used when ROSE_DISABLE_GENERAL_PURPOSE_AGENT is set so coding-agent
    GitHub ranking benches cannot select recursive GP fan-out. Schema-level
    narrowing beats a runtime rejection loop the same way SpawnExploreAgentTool
    does for PLAN.
    """

    model_config = ConfigDict(json_schema_extra=_require_spawn_output_schema_key)

    tool: Literal["spawn_agent"]
    type: Literal["browser", "code", "explore"] = Field(
        description=(
            "Select by capability: 'browser' starts a browser session for stateful "
            "or authenticated web work; 'code' starts a coding agent (Claude Code / "
            "Codex CLI) in a persistent sandbox; 'explore' performs read-only "
            "research for web or user-data facts. "
            "general_purpose and for_each are disabled for this run -  use "
            "type='code' for GitHub discovery/ranking work."
        )
    )
    task_name: str = Field(pattern=SPAWN_TASK_NAME_PATTERN, description=SPAWN_TASK_NAME_DESCRIPTION)
    goal: str = Field(
        description=(
            "The self-contained task/prompt and description for the spawned agent. "
            "Spawned agents have no access to ANY of your context by default. "
            "The goal and attachments parameters are their ONLY source of information. "
            "Include all relevant pieces of information. "
            f"{BROWSER_GOAL_PROFILE_NOTE} "
            f"{SELF_CONTAINED_GOAL_RULE.strip()}"
        )
    )
    output_schema: Optional[ResponseSchemaConfig] = Field(
        default=None,
        description=(
            "Structured data the agent returns and validates against. Required when "
            "type='browser' or type='code'. Ignored when type='explore'."
        ),
    )
    attachments: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional when the spawned agent needs files from the current context. "
            "Use exact `original_filename` values from file_pointer objects."
        ),
    )
    start_url: Optional[str] = Field(
        default=None,
        description=("HTTPS URL the browser opens first. Required when type='browser'; ignored otherwise."),
    )
    github_repo_link: Optional[str] = Field(
        default=None,
        description=(
            "Optional when type='code'. GitHub repository URL to clone and sync to. Ignored for every other type."
        ),
    )
    github_repo_ref: Optional[str] = Field(
        default=None,
        description="Optional when type='code'; ignored for every other type. " + _GITHUB_REPO_REF_DESCRIPTION,
    )
    code_session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional when type='code'. Ignored for every other type. The code_session_id a "
            "previous code-agent result reported, to resume that exact session (files, commits, "
            "branch and conversation). Omit it to resume your own latest session for the repo."
        ),
    )
    target: Optional[Literal["cloud", "local"]] = Field(
        default="cloud",
        description=(
            "Optional when type='code': 'local' runs the coding agent on the user's Mac in a folder they "
            "shared (requires an owned connected Mac; multiple Macs require explicit device selection), 'cloud' (default) runs it in a sandbox."
        ),
    )
    sandbox_size: Optional[CodeSandboxSize] = Field(default=None, description=_SANDBOX_SIZE_DESCRIPTION)
    context_gates: Optional[Dict[str, bool]] = Field(
        default=None,
        description=(
            "Optional when type='explore'. Applicable keys are 'network' and "
            "'recent_communications'. Missing keys default to true."
        ),
    )
    run_in_background: bool = Field(default=True, description=RUN_IN_BACKGROUND_DESCRIPTION)
    depth: Optional[Literal["quick", "medium", "deep"]] = Field(
        default="quick",
        description=(
            "Optional when type='explore'. Controls research thoroughness as a TURN "
            "budget: 'quick' (~5 turns), 'medium' (~20 turns), 'deep' (~50 turns). "
            "Default 'quick'."
        ),
    )

    @model_validator(mode="after")
    def check_spawn_agent_no_gp_constraints(self) -> "SpawnAgentToolNoGP":
        if self.type == "browser" and not self.start_url:
            recovered = first_url_in_text(self.goal)
            if recovered and urlsplit(recovered).scheme == "https":
                self.start_url = recovered
                logger.warning(
                    "spawn_agent(type='browser') omitted start_url; recovered from goal",
                    start_url_origin=url_origin_for_log(recovered),
                )
            else:
                raise ValueError("start_url is required when type='browser'.")
        if self.type in ("browser", "code") and not self.output_schema:
            raise ValueError(f"output_schema is required when type='{self.type}'.")
        if self.type == "explore" and self.output_schema:
            self.output_schema = None
        return self


class SpawnExploreAgentTool(BaseModel):
    """Spawn a read-only explore agent to gather information during PLAN.

    Narrowed variant of SpawnAgentTool: during PLAN only explore agents are
    permitted. Schema-level narrowing is cheaper than a runtime rejection
    loop -  the LLM cannot pick an invalid `type` and we drop ~600 tokens of
    per-type field guidance the model would otherwise have to reason past.
    """

    tool: Literal["spawn_agent"]
    type: Literal["explore"] = Field(
        default="explore",
        description=(
            "Must be 'explore'. PLAN is read-only: only explore agents are "
            "legal here. To spawn anything else, finish PLAN with "
            "attempt_completion first."
        ),
    )
    task_name: str = Field(pattern=SPAWN_TASK_NAME_PATTERN, description=SPAWN_TASK_NAME_DESCRIPTION)
    goal: str = Field(
        description=(
            "The self-contained task/prompt and description for the spawned agent. "
            "Spawned agents have no access to ANY of your context by default. "
            "The goal and attachments parameters are their ONLY source of information. "
            "Include all relevant pieces of information. "
            f"{SELF_CONTAINED_GOAL_RULE.strip()}"
        )
    )
    attachments: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional when the spawned agent needs files from the current context. "
            "Use exact `original_filename` values from file_pointer objects."
        ),
    )
    context_gates: Optional[Dict[str, bool]] = Field(
        default=None,
        description=(
            "Optional. Applicable keys are 'network' and 'recent_communications'. "
            "Missing keys default to true (context flows through to the agent). "
            "Set a key to false only when that context is clearly unnecessary."
        ),
    )
    # No run_in_background: PLAN-phase explore agents always run in the foreground
    # (workflows/background_agents.py) so their findings are in context before set_plan.
    depth: Optional[Literal["quick", "medium", "deep"]] = Field(
        default="quick",
        description=(
            "Controls research thoroughness as a TURN budget (one turn can batch multiple "
            "parallel searches): 'quick' = a basic look (~5 turns), 'medium' = moderate (~20 turns), "
            "'deep' = broad analysis across many sources (~50 turns). Default 'quick'; only use "
            "'deep' when the task genuinely needs exhaustive research."
        ),
    )


# ============================================================================
# Plan Phase Tools
# ============================================================================


class ListAgentsTool(BaseModel):
    """Browse your agent tree like a table of contents: where you are, nearby agents, drill-down hints. With project_id, owner_user_id or team_id it also lists teammates' agents in projects shared with you (title and status only)."""

    tool: Literal["list_agents"]
    node: Optional[str] = Field(
        default=None,
        description='Address to expand (a /root/... path, "parent", or an agent id). Default: you.',
    )
    depth: Optional[int] = Field(default=None, description="Levels of children to show under node (1-3, default 1).")
    cursor: Optional[str] = Field(default=None, description="Next-page cursor from a previous list_agents result.")
    query: Optional[str] = Field(default=None, description="Search agent goals in your tree (words, any order).")
    path_prefix: Optional[str] = Field(
        default=None, description="List agents at or below this path, e.g. /root/outreach."
    )
    project_id: Optional[str] = Field(default=None, description="Only agents in this project.")
    owner_user_id: Optional[str] = Field(default=None, description="Only this owner's project agents.")
    team_id: Optional[str] = Field(default=None, description="Only projects shared with this team.")


_AGENT_ADDRESS_DESCRIPTION = (
    "Agent address: a path like /root/outreach/draft_a (agents and their background tasks), "
    "'parent' for the agent that spawned you, or '/root'."
)


class SendMessageToAgentTool(BaseModel):
    """Ask another agent in your tree for new work or give it new information, a redirect or a handoff (to learn what it already did, spawn an explore agent to grep the agent tree); a finished agent wakes with it as its next task, except completed or failed browser and code agents, which cannot be messaged."""

    tool: Literal["send_message_to_agent"]
    to: str = Field(description=_AGENT_ADDRESS_DESCRIPTION)
    message: str = Field(description="The message. Self-contained: the recipient has not seen your conversation.")
    continue_conversation: Optional[bool] = Field(
        default=None,
        description=(
            "Required when the recipient is a paused browser or code agent; omit it otherwise. "
            "true resumes its exact prior conversation with this message; false starts it in a fresh "
            "conversation with this message, its original goal and its last result. Either way it keeps "
            "its browser state (cookies, tabs, profile, proxy) or code sandbox (files, session)."
        ),
    )


class KillAgentTool(BaseModel):
    """Kill an agent below you and everything it spawned, or one of your background tasks."""

    tool: Literal["kill_agent"]
    to: str = Field(description="Address of one of your descendants or background tasks (see your environment block).")


class ReportBugTool(BaseModel):
    """Report a bug in Rose itself, the same as the user's Report Bug button: a separate engineering agent investigates your run and fixes it.

    Use it only after you have confirmed that Rose's own tools, sub-agents or harness malfunction: a tool keeps failing on valid arguments, returns a result that contradicts its inputs, a sub-agent crashes or loops, or a capability you were given does not work. First rule out your own arguments and transient errors that a retry clears. Do not use it for problems outside Rose, such as a website being down, a person not replying or missing information from the user. A run files at most one report, so file it once you have evidence and describe every problem you saw, then continue your task."""

    tool: Literal["report_bug"]
    description: str = Field(
        description=(
            "What went wrong: the tool or agent involved, what you called it with, the exact error or "
            "wrong result, and what you expected. The investigator reads only this and your run's logs."
        )
    )


class SetPlanTool(BaseModel):
    """Signal readiness for plan generation; the planner LLM produces the DAG."""

    tool: Literal["set_plan"]
    # No parameters - planner LLM generates the plan internally


# ============================================================================
# Universal Completion Tool (replaces mark_execution_complete and progress_to_autonomous_execution_phase)
# ============================================================================


class AttemptCompletionTool(BaseModel):
    """Trigger completion for verified agent loops."""

    tool: Literal["attempt_completion"]
    result: Optional[str] = Field(
        default=None,
        description=(
            "Content the task asks you to write yourself (a draft, message or summary), in full. "
            "The grader reads it as your own work; facts in it must come from earlier tool results."
        ),
    )


class AttemptCompletionWithResultTool(BaseModel):
    """Submit your final findings to the parent agent and end your loop.

     The `result` string you submit here is the ONLY payload the parent
     agent receives. There is no future turn after this call returns, so
     your `result` must be complete and self-contained when you emit it.

     Requirements:
    - Complete: do not defer the answer. There is no future turn.
    - Grounded: every claim must be backed by evidence from your prior
       tool results. If a fact cannot be found, report the gap explicitly
       ("No results for [X] after searching [keywords]") rather than
       fabricating it.
    - Token-efficient, punctual, and information-dense. Strip filler;
       every sentence should carry signal.
    - Cite sources inline with [N] tags and end `result` with a
       `Sources:` block mapping each tag to its URL or source identifier.

     Emit a single `attempt_completion_with_result` tool_use block, do
     not batch it with other tool calls.
    """

    tool: Literal["attempt_completion_with_result"]
    result: str = Field(
        description=(
            "The final findings handed to the parent agent. This is the "
            "ONLY payload the parent receives as the spawn_agent tool "
            "result. Must be self-contained and grounded in sources from "
            "prior tool calls."
        ),
    )


# SpawnSubAgentTool removed - unified into SpawnAgentTool above


class CallPhoneNumberTool(BaseModel):
    """Place an outbound voice-AI call that follows system_prompt and extracts structured data."""

    tool: Literal["call_phone_number"]
    to: str = Field(description="Phone number to call in E.164 format.")
    system_prompt: str = Field(
        description=("Instructions for the voice agent's identity, tone, and conversation goals during the call.")
    )
    output_schema: ResponseSchemaConfig = Field(
        description=("Fields to extract from the conversation when the call completes.")
    )


# Typed alias for the canonical tool registry shape: name -> Pydantic model.
# Use this whenever a function takes "a registry of tools" so callers get
# proper type-checking instead of an untyped Dict.
ToolRegistry = Dict[str, Type[BaseModel]]


# Map tool names to their Pydantic models
TOOL_MODELS: ToolRegistry = {
    "read_tool_output": ReadToolOutputTool,
    "wait_for": WaitForTool,
    "web_search": WebSearchTool,
    "people_search": PeopleSearchTool,
    "web_fetch": WebFetchTool,
    "api_request": ApiRequestTool,
    "google_workspace_run": GoogleOAuthDemoTool,
    "send_message_to_user": SendMessageToUserTool,
    "send_message_to_user_execute_phase": SendMessageToUserExecutePhaseTool,
    "display": DisplayTool,
    "spawn_agent": SpawnAgentTool,
    "message_person": MessagePersonTool,
    # Agent-to-agent
    "list_agents": ListAgentsTool,
    "send_message_to_agent": SendMessageToAgentTool,
    "kill_agent": KillAgentTool,
    # Root-only (ROOT_ONLY_TOOLS)
    "report_bug": ReportBugTool,
    # Plan phase tools
    "set_plan": SetPlanTool,
    # Universal completion tool (available in both phases)
    "attempt_completion": AttemptCompletionTool,
    # Sub-agent (skip_verification=True) completion tool -  carries the
    # final handoff payload to the parent in `result`.
    "attempt_completion_with_result": AttemptCompletionWithResultTool,
    # Phone calling tool
    "call_phone_number": CallPhoneNumberTool,
    # Local data search
    "grep": GrepTool,
    # Read-only look inside a code agent's workspace (explore)
    "inspect_code": InspectCodeTool,
}

# Phase-specific override registry. PLAN narrows spawn_agent to the explore-only
# variant; everything else inherits from TOOL_MODELS.
PLAN_TOOL_MODELS: ToolRegistry = {
    **TOOL_MODELS,
    "spawn_agent": SpawnExploreAgentTool,
}

# Rose Desktop accounts (utils.rose_desktop) swap these in for the stock models:
# inline views on send_message_to_user, pinned questions on display, and the
# rose_user_ui clause in spawn_agent's `type`. TOOL_MODELS / PLAN_TOOL_MODELS
# stay the stock registries, so any caller that does not pass the flag gets the
# stock surface.
ROSE_DESKTOP_TOOL_MODELS: ToolRegistry = {
    "send_message_to_user": SendMessageToUserRoseDesktopTool,
    "send_message_to_user_execute_phase": SendMessageToUserExecutePhaseRoseDesktopTool,
    "display": DisplayRoseDesktopTool,
    "spawn_agent": SpawnAgentRoseDesktopTool,
}


def _with_rose_desktop_models(registry: ToolRegistry) -> ToolRegistry:
    """Swap in the Rose Desktop variant of every stock model the registry holds.

    Only exact stock entries are replaced, so the PLAN explore-only and the
    no-general-purpose spawn_agent variants stay as they are.
    """
    stock = {name: TOOL_MODELS[name] for name in ROSE_DESKTOP_TOOL_MODELS}
    return {
        name: ROSE_DESKTOP_TOOL_MODELS[name] if stock.get(name) is model else model for name, model in registry.items()
    }


# Tools available inside the explore agent (subset of read-only tools).
# Full registry list -  callers that inject tools into an explore agent MUST use
# `get_explore_agent_tools()` so ROSE_DISABLE_API_REQUEST can strip api_request.
EXPLORE_AGENT_TOOLS = [
    "read_tool_output",
    "web_search",
    "people_search",
    "web_fetch",
    "grep",
    "list_agents",
    "api_request",
    "inspect_code",
    "attempt_completion_with_result",
]


def get_explore_agent_tools() -> list[str]:
    """Explore tool names for injection, honoring ROSE_DISABLE_API_REQUEST."""
    from utils.kill_switches import api_request_disabled

    if not api_request_disabled():
        return list(EXPLORE_AGENT_TOOLS)
    return [name for name in EXPLORE_AGENT_TOOLS if name != "api_request"]


# ============================================================================
# B7: Deterministic tool-list renderer
# ============================================================================


def _first_nonempty_line(text: str) -> str:
    """Return the first non-empty stripped line of `text`, or '' if there is none."""
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def render_tool_list(
    registry: ToolRegistry,
    *,
    tool_prompts: Optional[Dict[str, str]] = None,
    sort: bool = True,
) -> str:
    """Render a deterministic, single-line-per-tool listing of `registry`.

    Output format: ``- {tool_name}: {description}`` per tool, joined by
    newlines. The description is the first non-empty line of the matching
    entry in ``tool_prompts`` (when supplied); if absent, it falls back to
    the first line of the Pydantic class docstring. Empty fallbacks render
    as ``"No description"`` so every listed tool produces a non-empty line.

    Determinism guarantee: with the same registry, two calls return the
    exact same string. Tools are sorted by name (when ``sort=True``, the
    default) so registry insertion order can never change the output.

    Args:
        registry: ``{name: Pydantic model}`` map. Use ``TOOL_MODELS`` for
            the canonical agent-tool registry, or pass a subset.
        tool_prompts: Optional ``{name: PROMPT}`` map injected by the caller
            (typically ``tools.prompts.TOOL_PROMPTS``). Pass it from any
            non-utils caller so this utility stays free of ``tools/`` deps.
        sort: When True (default), output lines are alphabetically sorted by
            tool name. Set to False to preserve registry iteration order
            (callers that pass an already-ordered subset).

    Returns:
        A multiline string with one ``- name: description`` line per tool,
        joined by ``\\n``. Returns an empty string if the registry is empty.
    """
    prompts = tool_prompts or {}
    names = sorted(registry.keys()) if sort else list(registry.keys())
    lines = []
    for name in names:
        description = _first_nonempty_line(prompts.get(name, ""))
        if not description:
            model = registry[name]
            description = _first_nonempty_line(model.__doc__ or "")
        if not description:
            description = "No description"
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


# ============================================================================
# Dynamic AgentAction Model Generation
# ============================================================================


def get_tool_models(*, phase: Optional[str] = None, rose_desktop: bool = False) -> ToolRegistry:
    """Return the tool registry for ``phase``, honoring GPA kill-switch.

    PLAN keeps the explore-only spawn_agent variant. EXECUTE (and unspecified)
    swaps in :class:`SpawnAgentToolNoGP` when
    ``ROSE_DISABLE_GENERAL_PURPOSE_AGENT`` is set so the LLM cannot select
    general_purpose / for_each. ``rose_desktop`` (``utils.rose_desktop``)
    swaps in the Rose Desktop models; the default is the stock registry.
    """
    from utils.kill_switches import general_purpose_agent_disabled

    if phase == "plan":
        registry = dict(PLAN_TOOL_MODELS)
    else:
        registry = dict(TOOL_MODELS)
        if general_purpose_agent_disabled():
            registry["spawn_agent"] = SpawnAgentToolNoGP
    return _with_rose_desktop_models(registry) if rose_desktop else registry


def get_planner_assignable_tools() -> list[str]:
    """Planner-assignable tool names, stripping GPA aliases when disabled."""
    from utils.kill_switches import general_purpose_agent_disabled

    if not general_purpose_agent_disabled():
        return list(PLANNER_ASSIGNABLE_TOOLS)
    return [
        name
        for name in PLANNER_ASSIGNABLE_TOOLS
        if name
        not in {
            "spawn_general_purpose_agent",
            "spawn_general_purpose_agent_for_each",
        }
    ]


def get_planner_response_model() -> Type[BaseModel]:
    """PlannerResponse model, with GPA aliases removed when the kill-switch is on."""
    from utils.kill_switches import general_purpose_agent_disabled

    if not general_purpose_agent_disabled():
        return PlannerResponse

    allowed = get_planner_assignable_tools()
    # Dynamic enum so structured-output schema cannot emit disabled aliases.
    filtered_enum = Enum(
        "PlanToolNameNoGP",
        {name: name for name in allowed},
        type=str,
    )
    task_model = create_model(
        "PlannerTaskNoGP",
        task_idx=(str, Field(description=PLANNER_TASK_IDX_DESCRIPTION)),
        title=(str, Field(description=PLANNER_TASK_TITLE_DESCRIPTION)),
        goal=(str, Field(description=PLANNER_TASK_GOAL_DESCRIPTION)),
        tool_name=(filtered_enum, Field(description=PLANNER_TASK_TOOL_NAME_DESCRIPTION)),
        depends_on=(
            List[str],
            Field(default_factory=list, description=PLANNER_TASK_DEPENDS_ON_DESCRIPTION),
        ),
        __base__=_PlannerTaskBase,
    )
    return create_model(
        "PlannerResponseNoGP",
        title=(str, Field(description=PLANNER_RESPONSE_TITLE_DESCRIPTION)),
        tasks=(List[task_model], Field(description=PLANNER_RESPONSE_TASKS_DESCRIPTION)),
        join=(Literal["all", "any"], Field(description=PLANNER_RESPONSE_JOIN_DESCRIPTION)),
        __base__=BaseModel,
    )


def create_agent_action_model(
    available_tool_names: List[str],
    tool_registry: Optional[Dict[str, Type[BaseModel]]] = None,
) -> Type[BaseModel]:
    """
     Dynamically create an AgentAction Pydantic model based on available tools.

     This ensures:
    - Only available tools are in the schema (better LLM guidance)
    - Type-safe tool calls (Pydantic validates structure)
    - No JSON string parsing hacks needed

     NOTE: reasoning is NOT part of this model anymore. Reasoning is handled via
     OpenRouter's reasoning_details which are extracted from API responses and
     stored in message metadata.

     Args:
         available_tool_names: List of tool names this agent can use
         tool_registry: Optional custom registry (defaults to TOOL_MODELS,
             with SpawnAgentToolNoGP when ROSE_DISABLE_GENERAL_PURPOSE_AGENT is set)

     Returns:
         Pydantic model class for AgentAction with proper tool union
    """
    registry = tool_registry if tool_registry is not None else get_tool_models()

    # Build list of tool models for available tools
    tool_types = []
    for name in available_tool_names:
        if name in registry:
            tool_types.append(registry[name])
        else:
            logger.warning(f"Tool '{name}' not found in registry, skipping")

    # Compiled classes are shared definitions: instantiate them or derive a
    # JSON schema, but do not mutate their fields or force model rebuilding.
    return _agent_action_model_for_types(tuple(tool_types))


@lru_cache(maxsize=128)
def _agent_action_model_for_types(tool_types: tuple[Type[BaseModel], ...]) -> Type[BaseModel]:
    """Compile each ordered tool-model set once, including across workflow replays."""
    # Create union type for tool calls
    if not tool_types:
        # No tools available - agent can only communicate
        ToolCallUnion = type(None)  # Empty type
        tool_calls_field = (
            List[ToolCallUnion],
            Field(default_factory=list, description="No tools available for this agent"),
        )
    elif len(tool_types) == 1:
        # Single tool type - no union needed
        ToolCallUnion = tool_types[0]
        tool_calls_field = (
            List[ToolCallUnion],
            Field(default_factory=list, description="List of tools to call in sequence"),
        )
    else:
        # Multiple tools - create union
        ToolCallUnion = Union[tuple(tool_types)]
        tool_calls_field = (
            List[ToolCallUnion],
            Field(default_factory=list, description="List of tools to call in sequence"),
        )

    # Dynamically create the AgentAction model
    # NOTE: reasoning is handled externally via OpenRouter reasoning_details
    AgentAction = create_model(
        "AgentAction",
        response=(
            str,
            Field(
                description="Thinking about what to do next. This IS shown to the user as thinking text, "
                "but is NOT seen by the verifier (verifier only evaluates tool calls and their results). "
                "Write in plain prose. No markup, no bullet prefixes, no special formatting symbols."
            ),
        ),
        tool_calls=tool_calls_field,
        __base__=BaseModel,
    )

    return AgentAction


# ============================================================================
# Static Models (not tool-specific)
# ============================================================================


class ProgressScore(BaseModel):
    """Schema for progress evaluation."""

    reasoning: str = Field(description="Your internal thinking process about the progress")
    score: float = Field(ge=0.0, le=1.0, description="Progress score from 0.0 (no progress) to 1.0 (complete)")


__all__ = [
    "WaitForTool",
    "WebSearchTool",
    "PeopleSearchTool",
    "WebFetchTool",
    "SendMessageToUserTool",
    "SendMessageToUserRoseDesktopTool",
    "SendMessageToUserExecutePhaseTool",
    "SendMessageToUserExecutePhaseRoseDesktopTool",
    "DisplayTool",
    "DisplayRoseDesktopTool",
    "MessagePersonTool",
    "SpawnAgentTool",
    "SpawnAgentRoseDesktopTool",
    "SpawnAgentToolNoGP",
    "ListAgentsTool",
    "SendMessageToAgentTool",
    "SetPlanTool",
    "AttemptCompletionTool",
    "AttemptCompletionWithResultTool",
    "CallPhoneNumberTool",
    "GrepTool",
    "InspectCodeTool",
    "TOOL_MODELS",
    "PLAN_TOOL_MODELS",
    "ROSE_DESKTOP_TOOL_MODELS",
    "ToolRegistry",
    "create_agent_action_model",
    "get_tool_models",
    "get_planner_assignable_tools",
    "get_planner_response_model",
    "EXPLORE_AGENT_TOOLS",
    "get_explore_agent_tools",
    "render_tool_list",
    "ProgressScore",
]
