"""Spawn-agent domain handler: thin runtime layer dispatching to agent types.

CRITICAL: All imports MUST be at module level. This file runs inside Temporal's
workflow sandbox where lazy imports trigger threading.Lock errors.
"""

from __future__ import annotations

import asyncio
import json
import uuid as uuid_module
from datetime import timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from temporalio.exceptions import CancelledError as TemporalCancelledError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

from domains.spawn_agent.config import (
    EXPLORE_CANCEL_GRACE,
    EXPLORE_DEFAULT_MODEL,
    EXPLORE_GOAL_DISPLAY_LEN,
    EXPLORE_HEARTBEAT_TIMEOUT,
    FM_EXPLORE_INFLIGHT_CAP,
    FOR_EACH_MAX_DEPTH,
    SUB_ACTIVITY_MAX_ATTEMPTS,
    SUB_ACTIVITY_TIMEOUT,
    explore_limits_for_depth,
    is_oai_demo_mass_agent_task,
    resolve_explore_depth,
    should_skip_oai_demo_browser_spawn,
)
from domains.spawn_agent.service import (
    EXPLORE_FALLBACK_PAYLOAD_BUDGET,
    build_for_each_goal,
    coerce_for_each_items,
    compile_explore_grep_evidence,
    compile_explore_wrap_up,
    fetch_explore_tool_evidence,
    resolve_start_phase,
    spawn_explore_agent_activity,
    validate_spawn_params,
)
from helpers.attachments import extract_attachments_from_context, fetch_parent_attachments_activity
from helpers.child_results import fetch_child_terminal_result_activity
from helpers.generation_handoff import (
    DEPLOY_INTERRUPTED_TOOL_MESSAGE,
    GenerationHandoffRequested,
)
from helpers.github_repo_ref import explicit_github_repo_ref
from helpers.source_connect_gates import raise_source_connect_gates
from helpers.tool_interrupts import tool_stop_requested
from helpers.tracing import span
from helpers.wait_condition_or_handoff import wait_condition_or_generation_handoff
from helpers.wait_on_world_clock import hold_on_children, release_children
from helpers.wmh_activity_policy import wmh_activity_timeout_options
from helpers.world_model_runtime import world_model_specialized_agent_activity
from tools.base import EventType, MessageStatus, ToolEvent, ToolManager
from tools.grep import fetch_local_data_summary
from tools.read_tool_output import preview_tool_output_activity
from tools.spawn_browser_agent_v3 import get_manager as get_browser_manager
from tools.spawn_claude_code_agent import get_manager as get_code_manager
from tools.spawn_sub_agents import get_manager as get_sub_agent_manager
from tools.spawn_sub_agents import spawn_sub_agents_batch_activity
from utils.oai_demo_july import apply_oai_demo_july_for_each
from utils.simulated_clock import simulated_timestamp_ms
from utils.tool_call_ids import spawn_group_id

_FAST_MODE_USER_COPY = (
    "fast mode",
    "fast cache",
    "fast-cache",
    "cached trajectory",
    "from_fast_cache",
)
_EXPLORE_NO_EVIDENCE = "Explore stopped before completion with no tool evidence."

_WORLD_MODEL_SPECIALIZED_AGENT_TOOLS = {
    "explore": "spawn_explore_agent",
    "browser": "spawn_browser_agent",
    "code": "spawn_claude_code_agent",
}

# Retained as the ordinary activity budget; simulate uses owner cancellation
# instead of a per-attempt deadline at this WMH boundary.
_WORLD_MODEL_SPECIALIZED_ACTIVITY_TIMEOUT = timedelta(minutes=12)


_FILE_POINTER_SHAPE_KEYS = {
    "storage_path",
    "original_filename",
    "mime_type",
    "size_bytes",
    "created_at",
}


def _normalize_simulated_file_pointer_ids(value: Any) -> tuple[Any, bool]:
    """Give WMH-produced FilePointers stable API-valid UUID identifiers.

    A simulated Code/Browser result can legitimately describe a generated file,
    but model-authored identifiers such as ``fp_report_123`` are not valid for
    the message API's FilePointer schema.  Normalize only complete FilePointer-
    shaped objects at the simulation boundary; real subagent results remain
    untouched.  JSON strings are walked as well because WMH may place the
    terminal observation inside a string-valued ``content`` field.
    """

    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            normalized_item, item_changed = _normalize_simulated_file_pointer_ids(item)
            normalized[key] = normalized_item
            changed = changed or item_changed

        if _FILE_POINTER_SHAPE_KEYS.issubset(normalized):
            raw_id = str(normalized.get("id") or "").strip()
            try:
                uuid_module.UUID(raw_id)
            except (TypeError, ValueError, AttributeError):
                identity = json.dumps(
                    {key: normalized.get(key) for key in sorted(_FILE_POINTER_SHAPE_KEYS)},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                normalized["id"] = str(
                    uuid_module.uuid5(
                        uuid_module.NAMESPACE_URL,
                        f"rose-wmh-file-pointer:{identity}",
                    )
                )
                changed = True
        return normalized, changed

    if isinstance(value, list):
        normalized_items = []
        changed = False
        for item in value:
            normalized_item, item_changed = _normalize_simulated_file_pointer_ids(item)
            normalized_items.append(normalized_item)
            changed = changed or item_changed
        return normalized_items, changed

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
            try:
                decoded = json.loads(stripped)
            except (TypeError, ValueError):
                return value, False
            normalized, changed = _normalize_simulated_file_pointer_ids(decoded)
            if changed:
                return json.dumps(
                    normalized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ), True
        return value, False

    return value, False


def _normalize_world_model_result_text(text: str) -> str:
    normalized, changed = _normalize_simulated_file_pointer_ids(text)
    if changed and isinstance(normalized, str):
        return normalized
    # Real Code-agent summaries can embed a JSON Structured Response after
    # human-readable prose. Normalize that object before a parent agent copies
    # its FilePointers into the message API (which requires UUID ids).
    marker = "Structured Response:"
    marker_at = text.find(marker)
    if marker_at < 0:
        return text
    json_start = text.find("{", marker_at + len(marker))
    if json_start < 0:
        return text
    try:
        embedded, consumed = json.JSONDecoder().raw_decode(text[json_start:])
    except json.JSONDecodeError:
        return text
    normalized_embedded, embedded_changed = _normalize_simulated_file_pointer_ids(embedded)
    if not embedded_changed:
        return text
    encoded = json.dumps(normalized_embedded, ensure_ascii=False, separators=(",", ":"))
    return text[:json_start] + encoded + text[json_start + consumed :]


def _world_model_specialized_result_text(result: Dict[str, Any]) -> str:
    """Return the terminal simulated subagent result shown to the parent/UI."""

    # WMH's generic response shape carries both the decoded ``output`` field
    # and its raw JSON envelope in ``message``. Prefer the decoded value so the
    # parent agent receives the actual terminal observation, not wrapper JSON.
    for key in ("output", "message", "content", "summary", "final_response"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_world_model_result_text(value.strip())
    nested = result.get("result")
    if isinstance(nested, str) and nested.strip():
        return _normalize_world_model_result_text(nested.strip())
    if nested is not None:
        return _normalize_world_model_result_text(json.dumps(nested, ensure_ascii=False, sort_keys=True, default=str))
    observation = result.get("world_model_observation")
    if observation is not None:
        return _normalize_world_model_result_text(
            json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str)
        )
    return "The simulated subagent completed without a textual result."


def user_facing_explore_result(text: Any) -> str:
    """Return explore chip copy that would show in prod — never Fast Mode plumbing."""
    raw = str(text or "").strip()
    if not raw or raw.lower() in {"research started.", "research started"}:
        return ""
    lowered = raw.lower()
    if lowered.startswith("starting explore agent:"):
        return ""
    if any(marker in lowered for marker in _FAST_MODE_USER_COPY) and len(raw) < 400:
        return ""
    return raw


def wrap_up_payload_from_failure(exc: BaseException) -> Dict[str, Any]:
    """Recover compiled explore evidence from a timed-out activity heartbeat."""
    cur: Any = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        details = getattr(cur, "last_heartbeat_details", None)
        if details:
            # Temporal decodes heartbeat details into a list; the evidence dict
            # follows the message when an LLM helper sent the last beat.
            payloads = list(details) if isinstance(details, (list, tuple)) else [details]
            for payload in payloads:
                if isinstance(payload, dict):
                    return payload
            if isinstance(payloads[0], str) and payloads[0].strip():
                return {"result": payloads[0]}
        cur = getattr(cur, "cause", None) or getattr(cur, "__cause__", None)
    return {}


def _explore_chip_text(result: Dict[str, Any]) -> str:
    return user_facing_explore_result(
        compile_explore_wrap_up(
            result.get("result") or result.get("final_response") or result.get("summary") or "",
            result.get("tool_call_results") or [],
            budget=None,
        )
    )


_EXPLORE_SYNTHESIS_CHECKPOINT = (
    "Research checkpoint for the parent agent: you have sourced findings from "
    "multiple explore rounds. Compare them with the original user's requested "
    "facts and synthesize if those facts are covered. Continue researching a "
    "specific missing requested fact if needed; state uncertainty instead of "
    "chasing optional precision. This checkpoint is guidance, not evidence."
)


def _source_linked_explore_text(text: str) -> bool:
    return len(text.strip()) >= 200 and any(marker in text for marker in ("https://", "http://", "@ref:", "grep:"))


def _grounded_explore_text(text: str, tool_results: List[str]) -> bool:
    """A substantive, source-linked result rather than a sparse/failed chip."""
    return bool(tool_results) and _source_linked_explore_text(text)


def _has_prior_grounded_explore_batch(context: Any, current_tool_call_id: str) -> bool:
    """Find a prior parent turn with at least two grounded explore results."""
    if not isinstance(context, list):
        return False
    grounded_ids: set[str] = set()
    for message in context:
        if not isinstance(message, dict) or str(message.get("role") or "").upper() != "TOOL":
            continue
        metadata = message.get("metadata") or {}
        if metadata.get("toolName") == "spawn_agent" and _EXPLORE_SYNTHESIS_CHECKPOINT in str(
            message.get("text") or ""
        ):
            return False  # Persist the one-time choice through continue-as-new.
        if (
            metadata.get("toolName") != "spawn_agent"
            or metadata.get("agentType") != "explore"
            or metadata.get("toolStatus") != "FINISHED"
        ):
            continue
        call_id = str(metadata.get("toolCallId") or "")
        if call_id and call_id != current_tool_call_id:
            data = metadata.get("data")
            payload = data.get("result") if isinstance(data, dict) else None
            text = str(payload or message.get("text") or "")
            if _source_linked_explore_text(text):
                grounded_ids.add(call_id)
    if len(grounded_ids) < 2:
        return False
    for message in context:
        if not isinstance(message, dict) or str(message.get("role") or "").upper() != "ASSISTANT":
            continue
        calls = (message.get("metadata") or {}).get("tool_calls") or []
        batch_ids = {
            str(call.get("tool_call_id") or "")
            for call in calls
            if isinstance(call, dict) and call.get("tool") == "spawn_agent" and call.get("type") == "explore"
        }
        if current_tool_call_id not in batch_ids and len(batch_ids & grounded_ids) >= 2:
            return True
    return False


def _explore_result_after_activity_failure(exc: BaseException, *, user_paused: bool) -> tuple[Dict[str, Any], bool]:
    """Map an explore activity failure to (result, was_cancelled).

    Timeout / wait-budget cancel with heartbeat evidence FINISHES so the
    parent can synthesize. An empty cancellation is an error. Unexpected
    ActivityError without wrap-up is re-raised for the outer ERROR path.
    """
    recovered = wrap_up_payload_from_failure(exc)
    cancel_cause = not isinstance(exc, ActivityError) or isinstance(
        exc.cause, (asyncio.CancelledError, TemporalCancelledError)
    )
    if user_paused and cancel_cause:
        return {}, True
    recovered_text = _explore_chip_text(recovered) if recovered else ""
    if recovered.get("tool_call_results") and recovered_text:
        return {
            "status": "completed",
            "result": recovered_text,
            "evidence_fallback": bool(recovered.get("evidence_fallback")) or not bool(
                recovered.get("result") or recovered.get("final_response") or recovered.get("summary")
            ),
            "tool_call_results": recovered.get("tool_call_results") or [],
            "accumulated_usage": recovered.get("accumulated_usage") or [],
            "llm_iteration_usage": recovered.get("llm_iteration_usage") or [],
        }, False
    timed_out = isinstance(exc, ActivityError) and isinstance(exc.cause, (TimeoutError, TemporalTimeoutError))
    if isinstance(exc, ActivityError) and not (cancel_cause or timed_out):
        raise exc
    if (cancel_cause or timed_out) and not user_paused:
        return {
            "status": "error",
            "result": "",
            "error_reason": _EXPLORE_NO_EVIDENCE,
            "tool_call_results": [],
            "accumulated_usage": [],
            "llm_iteration_usage": [],
        }, False
    return {}, True


logger = structlog.get_logger()

_FM_EXPLORE_HANDLES_ATTR = "_fm_explore_handles"


def _fm_explore_handles(workflow_instance) -> List[Dict[str, Any]]:
    handles = getattr(workflow_instance, _FM_EXPLORE_HANDLES_ATTR, None)
    if handles is None:
        handles = []
        setattr(workflow_instance, _FM_EXPLORE_HANDLES_ATTR, handles)
    return handles


def _activity_handle_done(handle: Any) -> bool:
    done = getattr(handle, "done", None)
    return bool(done()) if callable(done) else False


def prune_fast_mode_explores(workflow_instance) -> None:
    handles = _fm_explore_handles(workflow_instance)
    handles[:] = [item for item in handles if not _activity_handle_done(item.get("handle"))]


async def wait_for_fast_mode_explore_slot(
    workflow_instance,
    cap: int = FM_EXPLORE_INFLIGHT_CAP,
) -> None:
    """Block only when this GP already has `cap` in-flight Fast Mode explores."""
    handles = _fm_explore_handles(workflow_instance)

    def _has_slot() -> bool:
        prune_fast_mode_explores(workflow_instance)
        return len(handles) < cap

    if _has_slot():
        return
    await wait_condition_or_generation_handoff(_has_slot, workflow_instance, workflow_mod=workflow)


def track_fast_mode_explore(
    workflow_instance,
    handle: Any,
    *,
    task_id: str,
    tool_call_id: str,
    ui_goal: str,
) -> None:
    _fm_explore_handles(workflow_instance).append(
        {
            "handle": handle,
            "task_id": task_id,
            "tool_call_id": tool_call_id,
            "ui_goal": ui_goal,
        }
    )


async def drain_fast_mode_explores(workflow_instance) -> None:
    """No-op unless leftover handles were tracked (legacy fire-and-track).

    Fast Mode now waits for each explore like live, so nothing is tracked.
    Safe when the list is empty.
    """
    items = list(_fm_explore_handles(workflow_instance))
    _fm_explore_handles(workflow_instance).clear()
    post_event = getattr(workflow_instance, "_handle_tool_event", None)
    for item in items:
        handle = item.get("handle")
        if handle is None:
            continue
        result: Dict[str, Any] = {}
        recovered: Dict[str, Any] = {}
        try:
            raw_result = await handle
            if isinstance(raw_result, dict):
                result = raw_result
        except (asyncio.CancelledError, ActivityError) as exc:
            recovered = wrap_up_payload_from_failure(exc)
            if isinstance(exc, ActivityError) and not isinstance(
                exc.cause, (asyncio.CancelledError, TemporalCancelledError)
            ):
                logger.error(
                    "Fast Mode parallel explore failed",
                    task_id=item.get("task_id"),
                    error=str(exc)[:200],
                )
        except Exception as exc:
            recovered = wrap_up_payload_from_failure(exc)
            logger.error(
                "Fast Mode parallel explore failed",
                task_id=item.get("task_id"),
                error=str(exc)[:200],
            )
        if not callable(post_event):
            continue
        explore_result = _explore_chip_text(result)
        if not explore_result and recovered:
            explore_result = _explore_chip_text(recovered)
        if not explore_result:
            continue
        ui_goal = str(item.get("ui_goal") or "")
        await post_event(
            ToolEvent(
                type=EventType.POST_MESSAGE,
                title=ui_goal[:EXPLORE_GOAL_DISPLAY_LEN],
                message=explore_result,
                status=MessageStatus.FINISHED,
                metadata={
                    "agentType": "explore",
                    "taskId": item.get("task_id"),
                    "fullGoal": ui_goal,
                },
                data={"result": explore_result},
            ),
            item.get("tool_call_id"),
            "spawn_agent",
        )


def _is_oai_demo_for_each_tool(metadata: dict) -> bool:
    tool_name = str(metadata.get("toolName") or "")
    if tool_name == "spawn_general_purpose_agent_for_each":
        return True
    return tool_name == "spawn_agent" and metadata.get("agentType") == "for_each"


def _prior_oai_demo_for_each_count(workflow_instance, current_tool_call_id: str | None = None) -> int:
    """Count prior root for_each calls, including IN_PROGRESS/ERROR.

    The July mass-agent root replays a second 50-GP for_each after the first
    wave. One wave satisfies T2; a second wave cannot finish inside the 900s
    harness (smoke8: first 50 done at 449s, second wave still ACTIVE at 901s).

    Smoke9: the first wave's spawn_general_purpose_agent_for_each ended ERROR
    ("42/50 children failed" after finalize activity timeouts) and never
    FINISHED. A FINISHED-only counter then allowed the second fan-out at
    ~451s. The in-flight first call must not count itself — the workflow may
    already have posted IN_PROGRESS for the current toolCallId.
    """
    context = getattr(workflow_instance, "context", [])
    if not isinstance(context, list):
        return 0
    count = 0
    seen_tool_calls: set[str] = set()
    for msg in context:
        if not isinstance(msg, dict) or (msg.get("role") or "").upper() != "TOOL":
            continue
        metadata = msg.get("metadata") or {}
        if not _is_oai_demo_for_each_tool(metadata):
            continue
        tool_call_id = metadata.get("toolCallId")
        if current_tool_call_id and tool_call_id == current_tool_call_id:
            continue
        if tool_call_id:
            if tool_call_id in seen_tool_calls:
                continue
            seen_tool_calls.add(tool_call_id)
        count += 1
    return count


def _is_for_each_finalize_timeout(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return "timed out" in text or "timeout" in text


def _oai_demo_child_result_failed(child_result: Dict[str, Any]) -> bool:
    status = str(child_result.get("status") or "").strip().lower()
    return status in {"failed", "error", "max_iterations"} or child_result.get("success") is False


async def _reconcile_for_each_child_result(workflow_instance, child_agent_id: Any) -> bool:
    """Reload a durably terminal child when the completion signal was missed or consumed."""
    results = getattr(workflow_instance, "child_results", None)
    if isinstance(results, dict) and child_agent_id in results:
        pending = getattr(workflow_instance, "pending_children", None)
        if isinstance(pending, set):
            pending.discard(child_agent_id)
        return True

    try:
        recovered = await workflow.execute_activity(
            fetch_child_terminal_result_activity,
            args=[child_agent_id],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
    except Exception as exc:
        logger.warning("Child terminal reconciliation failed", child_agent_id=child_agent_id, error=str(exc))
        return False
    if not isinstance(recovered, dict):
        return False

    if not isinstance(getattr(workflow_instance, "child_results", None), dict):
        workflow_instance.child_results = {}
    workflow_instance.child_results[child_agent_id] = recovered
    pending = getattr(workflow_instance, "pending_children", None)
    if isinstance(pending, set):
        pending.discard(child_agent_id)
    logger.info("Recovered terminal child result from durable state", child_agent_id=child_agent_id)
    return True


async def _recover_oai_demo_for_each_timeout(
    workflow_instance, child_agent_id: Any, exc: BaseException
) -> Dict[str, Any] | None:
    """Wait for a completed child after finalize activity saturation (smoke9).

    50 parallel finalize_child calls hit post_message / reconcile activity
    timeouts around 100s, then for_each ERRORed (T10) while the GPs were
    still replaying. On this task, keep waiting for the child signal instead
    of treating the activity timeout as a child failure.

    The wait must also reconcile from durable child state. finalize_child pops
    ``child_results`` before the FINISHED event is handled, and the completion
    signal is not re-delivered — parking only on the in-memory map hangs
    ``asyncio.gather`` and the parent never yields ``for_each complete``.
    """
    if not child_agent_id:
        return None
    if not getattr(workflow_instance, "fast_mode_cache_enabled", False):
        return None
    if not is_oai_demo_mass_agent_task(workflow_instance):
        return None
    if not _is_for_each_finalize_timeout(exc):
        return None
    results = getattr(workflow_instance, "child_results", None)
    if not isinstance(results, dict):
        return None
    if child_agent_id in results:
        return results.get(child_agent_id)
    # Simulate mode: blocked awaiting this child, the parent holds on its children.
    hold_id = f"for_each_recovery:{child_agent_id}"
    try:
        await hold_on_children(workflow_instance, hold_id)
        while child_agent_id not in workflow_instance.child_results:
            if await _reconcile_for_each_child_result(workflow_instance, child_agent_id):
                break
            try:
                await wait_condition_or_generation_handoff(
                    lambda cid=child_agent_id: cid in workflow_instance.child_results,
                    workflow_instance,
                    timeout=timedelta(seconds=30),
                    workflow_mod=workflow,
                )
            except GenerationHandoffRequested:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                continue
    except BaseException:  # includes cancellation and GenerationHandoffRequested
        await release_children(workflow_instance, hold_id, [])
        raise
    await release_children(workflow_instance, hold_id, [child_agent_id])
    return workflow_instance.child_results.get(child_agent_id)


def _oai_demo_first_level_gp_count(workflow_instance) -> int:
    """Distinct first-level GP spawn tool calls already in context."""
    context = getattr(workflow_instance, "context", [])
    if not isinstance(context, list):
        return 0
    seen_tool_calls: set[str] = set()
    count = 0
    for msg in context:
        if not isinstance(msg, dict) or (msg.get("role") or "").upper() != "TOOL":
            continue
        metadata = msg.get("metadata") or {}
        tool_name = str(metadata.get("toolName") or "")
        if tool_name == "spawn_general_purpose_agent_for_each":
            continue
        if tool_name == "spawn_general_purpose_agent":
            pass
        elif tool_name == "spawn_agent" and metadata.get("agentType") == "general_purpose":
            pass
        else:
            continue
        tool_call_id = metadata.get("toolCallId")
        if tool_call_id:
            if tool_call_id in seen_tool_calls:
                continue
            seen_tool_calls.add(tool_call_id)
        count += 1
    return count


def _inbox_address(workflow_instance: Any, tool_call_id: str) -> str:
    """Inbox id of this spawn's current run (workflows/background_agents.py)."""
    resolver = getattr(workflow_instance, "_background_inbox_address", None)
    return resolver(tool_call_id) if callable(resolver) else tool_call_id


class SpawnAgentManager(ToolManager):
    """Unified manager that dispatches to browser/code/general_purpose/explore based on type."""

    @property
    def tool_name(self) -> str:
        return "spawn_agent"

    async def execute_tool(
        self, workflow_instance, tool_call: Dict[str, Any], tool_call_id: str
    ) -> AsyncGenerator[ToolEvent, None]:
        """Dispatch to appropriate agent type."""
        agent_type = tool_call.get("type")

        # Validate params
        try:
            validate_spawn_params(tool_call)
        except ValueError as e:
            # for_each must stay a thin wrapper with no for_each-specific tool events.
            # Bubble validation errors to workflow-level handling.
            if agent_type == "for_each":
                raise
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title="spawn_agent failed",
                message=str(e),
                status=MessageStatus.ERROR,
            )
            return

        fm = bool(getattr(workflow_instance, "fast_mode_cache_enabled", False))
        parent_depth = int(getattr(workflow_instance, "depth", 0) or 0)
        # Scoped to test-mass-agent-run-0 only: first-level for_each GPs spawn,
        # but those children must not recurse more GPs. The root also gets
        # exactly one for_each wave — a second 50-GP fan-out misses 900s.
        if (
            fm
            and agent_type in {"general_purpose", "for_each"}
            and is_oai_demo_mass_agent_task(workflow_instance)
            and (
                parent_depth >= 1
                or (
                    parent_depth == 0
                    and agent_type == "for_each"
                    and (
                        _prior_oai_demo_for_each_count(workflow_instance, tool_call_id) >= 1
                        or _oai_demo_first_level_gp_count(workflow_instance) >= 50
                    )
                )
            )
        ):
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title="spawn_agent complete",
                message=("Research already gathered; skipping nested spawn so the main sub-agent handles it."),
                status=MessageStatus.FINISHED,
                metadata={"agentType": agent_type, "skippedNestedSpawn": True},
            )
            return

        # Scoped to test-mass-agent-run-0 only: never open a browser-agent
        # session in the root or any child. Explore/web_search stay live.
        # Omit agentType so browser_agent_session_count does not treat this
        # FINISH-skip as a spawned browser instance.
        if agent_type == "browser" and should_skip_oai_demo_browser_spawn(workflow_instance):
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title="spawn_agent complete",
                message=("Research already gathered; skipping browser spawn so the explore tree handles it."),
                status=MessageStatus.FINISHED,
                metadata={"skippedBrowserSpawn": True},
            )
            return

        if agent_type == "explore":
            async for event in self._spawn_explore(workflow_instance, tool_call, tool_call_id):
                yield event
        elif agent_type == "browser":
            async for event in self._spawn_browser(workflow_instance, tool_call, tool_call_id):
                yield event
        elif agent_type == "code":
            async for event in self._spawn_code(workflow_instance, tool_call, tool_call_id):
                yield event
        elif agent_type == "general_purpose":
            async for event in self._spawn_sub_agent(workflow_instance, tool_call, tool_call_id):
                yield event
        elif agent_type == "for_each":
            async for event in self._spawn_for_each(workflow_instance, tool_call, tool_call_id):
                yield event

    async def _spawn_explore(
        self,
        workflow_instance,
        tool_call: Dict[str, Any],
        tool_call_id: str,
        *,
        task_id: Optional[str] = None,
        local_data_summary: Optional[str] = None,
        display_goal: Optional[str] = None,
        capture_tool_results: Optional[List[str]] = None,
    ) -> AsyncGenerator[ToolEvent, None]:
        """Spawn explore agent - research sub-agent.

        Optional kwargs allow callers (e.g. wait_for) to provide a known
        task_id, enabling cross-referencing between parent tool events and
        explore messages.
        display_goal overrides the goal shown in the UI (e.g. clean condition text).
        capture_tool_results, if supplied, is extended with the sub-agent's raw
        tool-call outputs. This is out-of-band so the 100+ KB trace never gets
        persisted into message metadata or shipped across activity boundaries.

        Detailed contact discovery + message-thread lookup happen inside the
        explore agent on demand via grep. A compact aggregate data summary is
        prefetched into dynamic user context so counts, date ranges, and top
        threads do not require scanning the full message corpus.
        """
        goal = tool_call.get("goal", "")
        ui_goal = display_goal or goal
        task_id = task_id or str(workflow.uuid4())
        if getattr(workflow_instance, "world_model_mode", "real") == "simulate":
            async for event in self._spawn_world_model_specialized(
                workflow_instance,
                tool_call,
                tool_call_id,
                agent_type="explore",
                task_id=task_id,
                display_goal=ui_goal,
                capture_tool_results=capture_tool_results,
            ):
                yield event
            return
        # One classification for depth and the activity's cache identity.
        # Re-resolving inside the activity from the child goal or Fast Mode
        # root disagrees (truncated explore goals, resolver failures).
        oai_demo = is_oai_demo_mass_agent_task(workflow_instance)
        # Caller-controlled thoroughness → turn cap + activity timeout.
        # Fast Mode promotes quick/missing → deep so July OAi-demo explores can
        # reach attempt_completion (~14 turns avg; quick aborts at 5).
        depth = resolve_explore_depth(
            tool_call.get("depth"),
            fast_mode_cache_enabled=bool(getattr(workflow_instance, "fast_mode_cache_enabled", False)),
            oai_demo=oai_demo,
        )
        explore_max_iterations, explore_activity_timeout = explore_limits_for_depth(depth)

        # IN_PROGRESS then wait for the activity — same as live. Fast Mode
        # only substitutes cached LLM / web_search / web_fetch responses
        # inside the explore; it does not change spawn orchestration.
        yield ToolEvent(
            type=EventType.POST_MESSAGE,
            title="Exploring",
            message=ui_goal[:EXPLORE_GOAL_DISPLAY_LEN],
            status=MessageStatus.IN_PROGRESS,
            metadata={
                "agentType": "explore",
                "taskId": task_id,
                "fullGoal": ui_goal,
            },
        )

        try:
            context_gates = tool_call.get("context_gates") or {}
            if local_data_summary is None:
                cached_summary = getattr(workflow_instance, "_cached_local_data_summary", None)
                if isinstance(cached_summary, str):
                    # Same user + same connected-data snapshot for the life of
                    # this workflow. Re-fetching on every July explore was one
                    # extra Temporal hop per spawn (224+ on Huberman-class GPs).
                    local_data_summary = cached_summary
                elif context_gates.get("recent_communications", True):
                    local_data_summary = await workflow.execute_activity(
                        fetch_local_data_summary,
                        args=[workflow_instance.user_id],
                        start_to_close_timeout=SUB_ACTIVITY_TIMEOUT,
                        retry_policy=RetryPolicy(maximum_attempts=SUB_ACTIVITY_MAX_ATTEMPTS),
                    )
                    if isinstance(local_data_summary, str):
                        workflow_instance._cached_local_data_summary = local_data_summary
                else:
                    local_data_summary = ""
                    workflow_instance._cached_local_data_summary = ""

            # Keep pointers across Temporal; attachment bytes are resolved by the
            # explore activity with the rest of its model context.
            attachment_texts: List[str] = []
            attachment_pointers: List[Dict[str, Any]] = []
            requested_attachments = tool_call.get("attachments")
            if isinstance(requested_attachments, str):
                requested_attachments = [requested_attachments]
            if requested_attachments and hasattr(workflow_instance, "context"):
                context_attachments = extract_attachments_from_context(workflow_instance.context)
                filename_to_pointer = {fp.original_filename: fp for fp in context_attachments}

                matched_pointers = []
                for filename in requested_attachments:
                    if filename in filename_to_pointer:
                        matched_pointers.append(filename_to_pointer[filename])
                    else:
                        logger.warning(
                            "[ExploreAgent] Attachment not found in context",
                            filename=filename,
                            available_files=list(filename_to_pointer.keys()),
                        )

                if matched_pointers:
                    pointer_dicts = [
                        {
                            "id": fp.id,
                            "storage_path": fp.storage_path,
                            "original_filename": fp.original_filename,
                            "mime_type": fp.mime_type,
                            "size_bytes": fp.size_bytes,
                            "created_at": fp.created_at,
                        }
                        for fp in matched_pointers
                    ]
                    attachment_pointers = pointer_dicts

            # Explore is a bounded research worker with its own model-tier
            # contract. Inheriting the parent's Ultra tier multiplies cost for
            # every parallel child and bypasses EXPLORE_DEFAULT_MODEL entirely.
            model = EXPLORE_DEFAULT_MODEL
            claude_token = (
                workflow_instance._active_claude_token()
                if hasattr(workflow_instance, "_active_claude_token")
                else getattr(workflow_instance, "claude_max_token", None)
            )
            gpt_token = (
                workflow_instance._active_gpt_token()
                if hasattr(workflow_instance, "_active_gpt_token")
                else getattr(workflow_instance, "gpt_code_token", None)
            )
            gpt_account_id = (
                workflow_instance._active_gpt_account_id()
                if hasattr(workflow_instance, "_active_gpt_account_id")
                else getattr(workflow_instance, "gpt_code_account_id", None)
            )
            gpt_token_source = (
                workflow_instance._active_gpt_source()
                if hasattr(workflow_instance, "_active_gpt_source")
                else getattr(workflow_instance, "gpt_code_token_source", None)
            )
            gpt_token_pool_id = (
                workflow_instance._active_gpt_pool_id()
                if hasattr(workflow_instance, "_active_gpt_pool_id")
                else getattr(workflow_instance, "gpt_code_token_pool_id", None)
            )
            world_model_mode = getattr(workflow_instance, "world_model_mode", "real")
            activity_args = [
                workflow_instance.agent_id,
                workflow_instance.user_id,
                goal,
                model,
                task_id,
                tool_call_id,
                claude_token,
                gpt_token,
                gpt_account_id,
                gpt_token_source,
                gpt_token_pool_id,
                attachment_texts,
                getattr(workflow_instance, "max_context_tokens", None) or 200_000,
                depth,
                explore_max_iterations,
                world_model_mode,
                local_data_summary,
                oai_demo,
                # Agent-to-agent inbox address: this spawning tool call's id
                # (``<id>.run<n>`` for a woken background task's later run).
                _inbox_address(workflow_instance, tool_call_id),
                # list_agents names this task <owner path>/<task_name>.
                tool_call.get("task_name"),
            ]
            activity_args.append(attachment_pointers)
            activity_handle = workflow.start_activity(
                spawn_explore_agent_activity,
                args=activity_args,
                start_to_close_timeout=explore_activity_timeout,
                heartbeat_timeout=EXPLORE_HEARTBEAT_TIMEOUT,
                cancellation_type=workflow.ActivityCancellationType.TRY_CANCEL,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            # Every tree waits the depth's full activity timeout (quick/medium
            # 600s, deep 1800s); there is no shorter live cap.
            parent_wait = explore_activity_timeout

            async with span(
                "explore_activity_wait",
                agent_id=workflow_instance.agent_id,
                tool_call_id=tool_call_id,
                tool_name="spawn_agent",
                task_id=task_id,
                span_source="agents.domains.spawn_agent.handler.SpawnAgentManager._spawn_explore",
                span_key="agents.domains.spawn_agent.handler.SpawnAgentManager._spawn_explore",
            ):
                # Wait for activity completion or pause signal. If the handle is
                # still open at the activity timeout, cancel + grace instead of
                # awaiting it unbounded.
                was_cancelled = False
                try:
                    # cutover-class: activity-bound
                    await workflow.wait_condition(
                        lambda: activity_handle.done() or tool_stop_requested(workflow_instance, tool_call_id),
                        timeout=parent_wait,
                    )
                except TimeoutError:
                    if not activity_handle.done():
                        logger.info(
                            "Cancelling explore agent after parent wait budget",
                            goal=goal[:100],
                            wait_s=parent_wait.total_seconds(),
                        )
                        activity_handle.cancel()
                        try:
                            # cutover-class: bounded
                            await workflow.wait_condition(
                                lambda: activity_handle.done(),
                                timeout=EXPLORE_CANCEL_GRACE,
                            )
                        except TimeoutError:
                            logger.info(
                                "Explore activity still running after cancel grace; wrapping up",
                                goal=goal[:100],
                            )
                if tool_stop_requested(workflow_instance, tool_call_id) and not activity_handle.done():
                    # Pause, or a kill_agent of this agent (helpers/tool_interrupts.py).
                    logger.info("Cancelling explore agent due to pause or interrupt", goal=goal[:100])
                    activity_handle.cancel()
                    was_cancelled = True

                # Get result. Live trees never await a still-running handle
                # unbounded (53831ac9). Demo / Fast Mode July replays must
                # await so capture_tool_results / wait_for see the cached result.
                try:
                    if activity_handle.done() or oai_demo:
                        result = await activity_handle
                    else:
                        result = {
                            "status": "error",
                            "result": "",
                            "error_reason": _EXPLORE_NO_EVIDENCE,
                            "tool_call_results": [],
                            "accumulated_usage": [],
                            "llm_iteration_usage": [],
                        }
                except (asyncio.CancelledError, ActivityError) as e:
                    result, was_cancelled = _explore_result_after_activity_failure(
                        e, user_paused=bool(getattr(workflow_instance, "is_paused", False))
                    )

            if not was_cancelled and result.get("error_reason") == _EXPLORE_NO_EVIDENCE:
                # TRY_CANCEL can resolve before the child worker's last heartbeat.
                # Recover only completed tool text already saved under this task.
                try:
                    persisted_results = await workflow.execute_activity(
                        fetch_explore_tool_evidence,
                        args=[workflow_instance.agent_id, workflow_instance.user_id, task_id],
                        start_to_close_timeout=SUB_ACTIVITY_TIMEOUT,
                        retry_policy=RetryPolicy(maximum_attempts=SUB_ACTIVITY_MAX_ATTEMPTS),
                    )
                except Exception as recovery_error:
                    logger.warning("Explore evidence recovery failed", task_id=task_id, error=str(recovery_error))
                    persisted_results = []
                # The backend excerpts are already bounded. Keep the saved
                # tool label, URL, and original result text together.
                recovered_text = "\n\n".join(persisted_results)
                if recovered_text:
                    result = {
                        "status": "completed",
                        "result": recovered_text,
                        "evidence_fallback": True,
                        "tool_call_results": persisted_results,
                        "accumulated_usage": [],
                        "llm_iteration_usage": [],
                    }

            if was_cancelled:
                logger.info("Explore agent cancelled", goal=goal[:100])
                yield ToolEvent(
                    type=EventType.POST_MESSAGE,
                    title="Research cancelled",
                    message="Explore agent cancelled by user",
                    status=MessageStatus.ERROR,
                    metadata={"agentType": "explore", "taskId": task_id, "fullGoal": ui_goal},
                )
                return

            explore_result = _explore_chip_text(result)
            explore_tool_call_results = result.get("tool_call_results") or []
            evidence_fallback = bool(result.get("evidence_fallback")) or not bool(
                result.get("result") or result.get("final_response") or result.get("summary")
            )
            accumulated_usage = result.get("accumulated_usage", [])
            llm_iteration_usage = result.get("llm_iteration_usage", [])
            # Data sources the explore tried to grep but whose connector isn't
            # connected. Keep them on the event so the parent can raise one
            # blocking source_connect gate each — but do NOT mark the explore
            # itself ERROR; the result already includes a not-connected notice.
            unavailable_sources = result.get("unavailable_sources") or []
            explore_failed = result.get("status") not in {None, "completed"}
            explore_error_reason = result.get("error_reason")
            fast_mode = bool(getattr(workflow_instance, "fast_mode_cache_enabled", False))
            # An empty live completion is not usable research. In Fast Mode the
            # chip filter also strips replay-end stub copy, so a completed
            # activity (which already applied its evidence gate) stays FINISHED.
            if not explore_result and not (fast_mode and not explore_failed):
                explore_failed = True
                explore_error_reason = explore_error_reason or _EXPLORE_NO_EVIDENCE
            # Fast Mode may soft-complete a malformed cached LLM turn only
            # when prior tool calls produced evidence to pass to the parent.
            if explore_failed and fast_mode:
                cached_evidence = compile_explore_wrap_up("", explore_tool_call_results, budget=None)
                if cached_evidence and explore_error_reason != _EXPLORE_NO_EVIDENCE:
                    explore_result = cached_evidence
                    explore_failed = False
                    evidence_fallback = True

            if evidence_fallback and explore_result:
                if explore_failed and explore_error_reason:
                    explore_result = f"{explore_error_reason}\n\n{explore_result}"
                if len(explore_result) > EXPLORE_FALLBACK_PAYLOAD_BUDGET:
                    # Timeout/heartbeat recovery still has complete raw evidence
                    # here. Retain it before the normal parent ToolEvent is cut.
                    readback = await workflow.execute_activity(
                        preview_tool_output_activity,
                        args=[workflow_instance.user_id, workflow_instance.agent_id, "", explore_result, 512],
                        start_to_close_timeout=SUB_ACTIVITY_TIMEOUT,
                        retry_policy=RetryPolicy(maximum_attempts=SUB_ACTIVITY_MAX_ATTEMPTS),
                    )
                    presentation = compile_explore_wrap_up(
                        "", [explore_result], budget=EXPLORE_FALLBACK_PAYLOAD_BUDGET - len(readback) - 2,
                    )
                    explore_result = f"{presentation}\n\n{readback}"

            # Hand raw tool-call traces to any caller that asked for them
            # (wait_for's condition-judge). Keeping this out of the persisted
            # ToolEvent.data avoids shipping ~150–220 KB of debug trace per
            # spawn through Temporal activity payloads and message storage.
            if capture_tool_results is not None:
                capture_tool_results.extend(explore_tool_call_results)

            combined_llm = list(llm_iteration_usage)
            combined_service: List[Dict[str, Any]] = []
            for usage in accumulated_usage:
                combined_llm.extend(usage.get("llm", []))
                combined_service.extend(usage.get("service", []))

            event_data: Dict[str, Any] = {
                "usage": {
                    "llm": combined_llm,
                    "service": combined_service,
                },
                "result": explore_result,
                "unavailable_sources": unavailable_sources,
            }
            event_metadata: Dict[str, Any] = {
                "agentType": "explore",
                "taskId": task_id,
                "fullGoal": ui_goal,
            }
            if explore_error_reason:
                event_data["error_reason"] = explore_error_reason
                event_metadata["error_reason"] = explore_error_reason
            # User-visible message is the explore result (including any
            # not-connected notice). Only genuine explore failures swap in
            # error_reason / ERROR status.
            event_message = explore_error_reason if explore_failed and explore_error_reason and not (evidence_fallback and explore_result) else explore_result
            # Fallback already contains this evidence (and its full readback);
            # appending another hard-cut copy would exceed its presentation cap.
            grep_evidence = "" if evidence_fallback else compile_explore_grep_evidence(explore_tool_call_results)
            if not explore_failed and grep_evidence and grep_evidence not in explore_result:
                event_message = f"{event_message}\n\nOriginal grep results:\n{grep_evidence}"
            if (
                not oai_demo
                and not explore_failed
                and not evidence_fallback
                and _grounded_explore_text(explore_result, explore_tool_call_results)
                and not getattr(workflow_instance, "_explore_synthesis_checkpoint_sent", False)
                and _has_prior_grounded_explore_batch(getattr(workflow_instance, "context", None), tool_call_id)
            ):
                # One advisory after a grounded follow-up round. It must not
                # replace evidence or prevent another necessary targeted search.
                workflow_instance._explore_synthesis_checkpoint_sent = True
                event_message = f"{event_message}\n\n{_EXPLORE_SYNTHESIS_CHECKPOINT}"

            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title=ui_goal[:EXPLORE_GOAL_DISPLAY_LEN],
                message=event_message,
                status=MessageStatus.ERROR if explore_failed else MessageStatus.FINISHED,
                data=event_data,
                metadata=event_metadata,
            )

        except BaseException as e:
            if isinstance(e, GeneratorExit):
                raise  # Never suppress GeneratorExit
            error_msg = str(e)
            logger.error("Explore agent failed", error=error_msg, goal=goal[:100], exc_type=type(e).__name__)
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title=ui_goal[:EXPLORE_GOAL_DISPLAY_LEN],
                message=error_msg,
                status=MessageStatus.ERROR,
                metadata={"agentType": "explore", "taskId": task_id, "fullGoal": ui_goal},
            )

    async def _spawn_browser(
        self,
        workflow_instance,
        tool_call: Dict[str, Any],
        tool_call_id: str,
    ) -> AsyncGenerator[ToolEvent, None]:
        """Delegate to existing SpawnBrowserAgentManagerV3."""
        if getattr(workflow_instance, "world_model_mode", "real") == "simulate":
            async for event in self._spawn_world_model_specialized(
                workflow_instance,
                tool_call,
                tool_call_id,
                agent_type="browser",
            ):
                yield event
            return
        browser_tool_call = {
            "tool": "spawn_browser_agent",
            "start_url": tool_call.get("start_url"),
            "goal": tool_call.get("goal"),
            "output_schema": tool_call.get("output_schema"),
            # Agent-to-agent inbox address: this spawning tool call's id
            # (``<id>.run<n>`` for a woken background task's later run).
            "inbox_address": _inbox_address(workflow_instance, tool_call_id),
        }
        if tool_call.get("attachments"):
            browser_tool_call["attachments"] = tool_call["attachments"]
        if isinstance(tool_call.get("_resume_paused_conversation"), dict):
            # Set only by a continue_conversation resume (workflows/background_agents.py).
            browser_tool_call["_resume_paused_conversation"] = tool_call["_resume_paused_conversation"]

        manager = get_browser_manager()
        async for event in manager.execute_tool(workflow_instance, browser_tool_call, tool_call_id):
            yield event

    async def _spawn_code(
        self,
        workflow_instance,
        tool_call: Dict[str, Any],
        tool_call_id: str,
    ) -> AsyncGenerator[ToolEvent, None]:
        """Delegate to existing SpawnClaudeCodeAgentManager."""
        if getattr(workflow_instance, "world_model_mode", "real") == "simulate":
            async for event in self._spawn_world_model_specialized(
                workflow_instance,
                tool_call,
                tool_call_id,
                agent_type="code",
            ):
                yield event
            return
        code_tool_call = {
            "tool": "spawn_claude_code_agent",
            "goal": tool_call.get("goal"),
            "output_schema": tool_call.get("output_schema"),
        }
        if tool_call.get("attachments"):
            code_tool_call["attachments"] = tool_call["attachments"]
        if tool_call.get("github_repo_link"):
            code_tool_call["github_repo_link"] = tool_call["github_repo_link"]
        github_repo_ref = explicit_github_repo_ref(tool_call.get("github_repo_ref"))
        if github_repo_ref:
            # Explicit connected-repo selection: outranks every inherited repo (backend).
            code_tool_call["github_repo_ref"] = github_repo_ref
        if tool_call.get("code_session_id"):
            code_tool_call["code_session_id"] = tool_call["code_session_id"]
        if tool_call.get("target"):
            code_tool_call["target"] = tool_call["target"]
        if isinstance(tool_call.get("sandbox_size"), dict):
            # The root's initial sandbox size (validated by the backend).
            code_tool_call["sandbox_size"] = tool_call["sandbox_size"]
        if tool_call.get("task_name"):
            # Stored on the code task row: the agents list shows it as the title.
            code_tool_call["task_name"] = tool_call["task_name"]
        if tool_call.get("_fresh_conversation") is True:
            # Set only by a continue_conversation=false resume (workflows/background_agents.py).
            code_tool_call["_fresh_conversation"] = True
        # Set only by a person's resume of this code agent (helpers RESUME_CONVERSATION_KEYS):
        # the typed turn's files, the stopped session's CLI and the turn's model / effort / speed.
        for key in ("_user_attachments", "_resume_code_provider", "_resume_cli_settings"):
            if tool_call.get(key):
                code_tool_call[key] = tool_call[key]

        manager = get_code_manager()
        async for event in manager.execute_tool(workflow_instance, code_tool_call, tool_call_id):
            yield event

    async def _spawn_world_model_specialized(
        self,
        workflow_instance,
        tool_call: Dict[str, Any],
        tool_call_id: str,
        *,
        agent_type: str,
        task_id: Optional[str] = None,
        display_goal: Optional[str] = None,
        capture_tool_results: Optional[List[str]] = None,
    ) -> AsyncGenerator[ToolEvent, None]:
        """Mock an explore, browser, or code subagent with one terminal WMH transition.

        This runs in the unified spawn boundary before a real specialized-agent
        activity is created. General-purpose agent loops remain real in
        simulator mode.
        """

        boundary_tool = _WORLD_MODEL_SPECIALIZED_AGENT_TOOLS[agent_type]
        goal = str(tool_call.get("goal") or "")
        ui_goal = str(display_goal or goal)
        task_id = task_id or str(workflow.uuid4())
        progress_title = {
            "explore": "Exploring",
            "browser": "Browser agent",
            "code": "Code agent",
        }[agent_type]
        terminal_title = {
            "explore": ui_goal[:EXPLORE_GOAL_DISPLAY_LEN],
            "browser": "Browser automation complete",
            "code": "Code agent complete",
        }[agent_type]
        base_metadata: Dict[str, Any] = {
            "agentType": agent_type,
            "taskId": task_id,
            "fullGoal": ui_goal,
            "world_model_simulated": True,
            "world_model_boundary": boundary_tool,
        }
        if agent_type == "code":
            base_metadata.update(
                toolName=boundary_tool,
                task_id=task_id,
                code_provider="claude",
            )

        yield ToolEvent(
            type=EventType.POST_MESSAGE,
            title=progress_title,
            message=ui_goal[:EXPLORE_GOAL_DISPLAY_LEN],
            status=MessageStatus.IN_PROGRESS,
            metadata=base_metadata,
        )

        simulated_args = {
            key: value for key, value in tool_call.items() if key not in {"tool", "_phase", "_spawn_group_id"}
        }
        simulated_args.setdefault(
            "root_goal",
            str(
                getattr(workflow_instance, "demo_cache_root_goal", None) or getattr(workflow_instance, "goal", "") or ""
            ),
        )

        try:
            result = await workflow.execute_activity(
                world_model_specialized_agent_activity,
                args=[
                    "simulate",
                    str(getattr(workflow_instance, "agent_id", "") or ""),
                    boundary_tool,
                    simulated_args,
                    tool_call_id,
                    simulated_timestamp_ms(workflow_instance, workflow.now()),
                    str(
                        getattr(workflow_instance, "root_agent_id", None)
                        or getattr(workflow_instance, "agent_id", "")
                        or ""
                    ),
                ],
                **wmh_activity_timeout_options("simulate", _WORLD_MODEL_SPECIALIZED_ACTIVITY_TIMEOUT),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            result_text = _world_model_specialized_result_text(result)
            if capture_tool_results is not None:
                capture_tool_results.append(f"[{boundary_tool}] {result_text}")
            failed = bool(result.get("error") or result.get("success") is False)
            status_text = str(result.get("status") or "").strip().lower()
            failed = failed or status_text in {"error", "failed", "cancelled"}
            terminal_metadata = dict(base_metadata)
            if agent_type == "code":
                terminal_metadata.update(action_type="task_complete", success=not failed)
                terminal_title = "Task Incomplete" if failed else "✓ Task Completed"
            for key in (
                "world_model_session_id",
                "world_model_name",
                "predicted_latency_ms",
                "simulated_elapsed_ms",
                "world_model_timestamp_ms",
            ):
                if result.get(key) is not None:
                    terminal_metadata[key] = result[key]
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title=terminal_title,
                message=result_text,
                status=MessageStatus.ERROR if failed else MessageStatus.FINISHED,
                data={
                    "result": result_text,
                    "world_model_simulated": True,
                },
                metadata=terminal_metadata,
            )
        except BaseException as exc:
            if isinstance(exc, (GeneratorExit, asyncio.CancelledError)):
                raise
            terminal_metadata = dict(base_metadata)
            if agent_type == "code":
                terminal_metadata.update(action_type="task_complete", success=False)
                terminal_title = "Task Incomplete"
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title=terminal_title,
                message=str(exc) or type(exc).__name__,
                status=MessageStatus.ERROR,
                metadata=terminal_metadata,
            )

    async def _spawn_sub_agent(
        self,
        workflow_instance,
        tool_call: Dict[str, Any],
        tool_call_id: str,
        *,
        parent_attachments: Dict[str, Any] | None = None,
    ) -> AsyncGenerator[ToolEvent, None]:
        """Delegate to existing SpawnSubAgentManager."""
        start_phase = resolve_start_phase(tool_call)
        sub_agent_config = {
            "goal": tool_call.get("goal"),
            "output_schema": tool_call.get("output_schema"),
            "start_phase": start_phase,
            "attachments": tool_call.get("attachments"),
        }
        if tool_call.get("task_name"):
            sub_agent_config["task_name"] = tool_call["task_name"]
        group_id = spawn_group_id(tool_call_id, tool_call.get("_spawn_group_id"))
        if group_id:
            sub_agent_config["_spawn_group_id"] = group_id
        if tool_call.get("context_gates"):
            sub_agent_config["context_gates"] = tool_call["context_gates"]
        github_repo_ref = explicit_github_repo_ref(tool_call.get("github_repo_ref"))
        if github_repo_ref:
            # Persisted as the child's Agent.githubRepoRef; omitted, the backend
            # inherits the parent's ref (parent_agent_id).
            sub_agent_config["github_repo_ref"] = github_repo_ref

        sub_agent_tool_call = {
            "tool": "spawn_sub_agent",
            "sub_agent": sub_agent_config,
        }

        manager = get_sub_agent_manager()
        async for event in manager.execute_tool(
            workflow_instance,
            sub_agent_tool_call,
            tool_call_id,
            parent_attachments=parent_attachments,
        ):
            yield event

    async def _spawn_for_each(
        self,
        workflow_instance,
        tool_call: Dict[str, Any],
        tool_call_id: str,
    ) -> AsyncGenerator[ToolEvent, None]:
        """Fan out N sub-agents via the batch backend endpoint, then drain
        per-child completion events concurrently.

        Two-phase:
        1. BATCH SPAWN — one HTTP call to /api/agents/batch via the
           spawn_sub_agents_batch_activity. The activity mints stable
           agent_ids upfront so its own retries are idempotent.
        2. DRAIN — for each batch result, run SpawnSubAgentManager.finalize_child
           in parallel. finalize_child posts the IN_PROGRESS row, waits for
           completion, and posts FINISHED (or a failure event if the child
           failed at spawn time).

        FOR_EACH_MAX_INFLIGHT no longer throttles spawns — concurrency is now
        bounded server-side by the batch endpoint's max-200 limit.
        """
        items = coerce_for_each_items(tool_call)
        if not items:
            raise ValueError("No items to process. Provide items or items_csv.")

        if workflow_instance.depth >= FOR_EACH_MAX_DEPTH:
            raise ValueError(f"Cannot spawn for_each: maximum nesting depth ({FOR_EACH_MAX_DEPTH}) reached.")

        base_goal = tool_call.get("goal", "")
        fm = bool(getattr(workflow_instance, "fast_mode_cache_enabled", False))
        base_goal, items = apply_oai_demo_july_for_each(
            base_goal,
            items,
            enabled=fm and is_oai_demo_mass_agent_task(workflow_instance),
        )
        start_phase = resolve_start_phase(tool_call)
        attachments = tool_call.get("attachments")
        output_schema = tool_call.get("output_schema")
        context_gates = tool_call.get("context_gates")
        github_repo_ref = explicit_github_repo_ref(tool_call.get("github_repo_ref"))

        # Resolve bus_id once (matches per-child SpawnSubAgentManager logic).
        parent_bus_id = getattr(workflow_instance, "bus_id", None)
        if parent_bus_id is None:
            workflow_instance.bus_id = workflow_instance.agent_id
            await workflow_instance._update_tree_state()
            parent_bus_id = workflow_instance.bus_id

        # Pre-fetch parent attachments once for the entire batch.
        parent_attachments = await workflow.execute_activity(
            fetch_parent_attachments_activity,
            args=[workflow_instance.agent_id],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Build per-item agent_cfgs (shape used by spawn_sub_agent_activity).
        agent_cfgs: List[Dict[str, Any]] = []
        task_name = tool_call.get("task_name")
        for item_index, item in enumerate(items):
            cfg: Dict[str, Any] = {
                "goal": build_for_each_goal(base_goal, item),
                "output_schema": output_schema,
                "attachments": attachments,
                "start_phase": start_phase,
            }
            if task_name:
                # for_each children are addressed <spawner path>/<task_name>_<i>.
                cfg["task_name"] = f"{task_name}_{item_index}"
            if context_gates:
                cfg["context_gates"] = context_gates
            if github_repo_ref:
                cfg["github_repo_ref"] = github_repo_ref
            agent_cfgs.append(cfg)

        # Wait for un-pause before spawning (parents pause cascades to children).
        await wait_condition_or_generation_handoff(
            lambda: not workflow_instance.is_paused,
            workflow_instance,
            workflow_mod=workflow,
        )

        parent_model = getattr(workflow_instance, "model", "rose-labs/rose-1-flash")
        parent_source = getattr(workflow_instance, "agent_source", "WEB")
        group_id = spawn_group_id(tool_call_id, tool_call.get("_spawn_group_id"))

        child_agent_ids = [str(workflow.uuid4()) for _ in agent_cfgs]
        for idx, child_agent_id in enumerate(child_agent_ids):
            workflow_instance.child_tool_ids[child_agent_id] = f"{tool_call_id}:{idx}"
            workflow_instance.pending_children.add(child_agent_id)
            workflow_instance.child_workflow_ids[child_agent_id] = child_agent_id

        # Phase 1: one batch activity = one backend HTTP call.
        try:
            batch_result = await workflow.execute_activity(
                spawn_sub_agents_batch_activity,
                args=[
                    workflow_instance.user_id,
                    agent_cfgs,
                    workflow_instance.agent_id,
                    workflow.info().workflow_id,
                    workflow_instance.depth + 1,
                    getattr(workflow_instance, "is_paused", False),
                    parent_model,
                    parent_bus_id,
                    parent_source,
                    parent_attachments,
                    child_agent_ids,
                ],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except Exception:
            for child_agent_id in child_agent_ids:
                workflow_instance.child_tool_ids.pop(child_agent_id, None)
                workflow_instance.pending_children.discard(child_agent_id)
                workflow_instance.child_workflow_ids.pop(child_agent_id, None)
            raise

        children = batch_result.get("agents", [])
        returned_child_ids = {child.get("agent_id") for child in children if child.get("agent_id")}
        for child_agent_id in child_agent_ids:
            if child_agent_id not in returned_child_ids:
                workflow_instance.child_tool_ids.pop(child_agent_id, None)
                workflow_instance.pending_children.discard(child_agent_id)
                workflow_instance.child_workflow_ids.pop(child_agent_id, None)
        if len(children) != len(agent_cfgs):
            logger.warning(
                "Batch result length mismatch",
                expected=len(agent_cfgs),
                got=len(children),
            )

        sub_manager = get_sub_agent_manager()
        failed_count = 0
        child_outputs: List[Dict[str, Any]] = []

        async def _drain_child(idx: int, child_info: Dict[str, Any]):
            nonlocal failed_count
            tcid = f"{tool_call_id}:{idx}"
            cfg = agent_cfgs[idx] if idx < len(agent_cfgs) else {}
            sub_agent_for_finalize = {
                "goal": cfg.get("goal", ""),
                "output_schema": cfg.get("output_schema"),
                "_spawn_group_id": group_id,
            }
            saw_error = False
            try:
                async for event in sub_manager.finalize_child(
                    workflow_instance,
                    child_info,
                    sub_agent_for_finalize,
                    tcid,
                    parent_bus_id,
                    for_each_child=True,
                ):
                    await workflow_instance._handle_tool_event(event, tcid, "spawn_agent")
                    if isinstance(event, ToolEvent) and event.status == MessageStatus.ERROR:
                        saw_error = True
                        event_metadata = event.metadata or {}
                        failure_context = event_metadata.get("failure_context") or {}
                        failed_item = {
                            "index": idx,
                            "agent_id": child_info.get("agent_id"),
                            "status": "failed",
                            "summary": event.message or "",
                            "error_reason": event_metadata.get("error_reason") or event.message or "unknown",
                        }
                        partial = (
                            event_metadata.get("structured_response")
                            or failure_context.get("partial_structured_response")
                            or event_metadata.get("partial_structured_response")
                        )
                        if partial is not None:
                            failed_item["partial_structured_response"] = partial
                        if failure_context.get("verifier_reasoning"):
                            failed_item["verifier_reasoning"] = failure_context["verifier_reasoning"]
                        if failure_context.get("progress_estimate"):
                            failed_item["progress_estimate"] = failure_context["progress_estimate"]
                        child_outputs.append(failed_item)
                    if isinstance(event, ToolEvent) and event.status == MessageStatus.FINISHED:
                        event_metadata = event.metadata or {}
                        child_outputs.append(
                            {
                                "index": idx,
                                "agent_id": child_info.get("agent_id"),
                                "status": "completed",
                                "summary": event.message or "",
                                "structured_response": event_metadata.get("structured_response"),
                            }
                        )
            except GenerationHandoffRequested:
                raise
            except Exception as exc:
                recovered = await _recover_oai_demo_for_each_timeout(workflow_instance, child_info.get("agent_id"), exc)
                if recovered is not None and not _oai_demo_child_result_failed(recovered):
                    logger.warning(
                        "for_each child finalize timed out; recovered completed child",
                        tcid=tcid,
                        error=str(exc),
                    )
                    child_outputs.append(
                        {
                            "index": idx,
                            "agent_id": child_info.get("agent_id"),
                            "status": "completed",
                            "summary": recovered.get("summary") or "",
                            "structured_response": recovered.get("structured_response"),
                        }
                    )
                    return
                saw_error = True
                child_outputs.append(
                    {
                        "index": idx,
                        "agent_id": child_info.get("agent_id"),
                        "status": "failed",
                        "summary": "",
                        "error_reason": str(exc) or "child finalize failed",
                    }
                )
                logger.warning("for_each child finalize failed", tcid=tcid, error=str(exc))
            if saw_error:
                failed_count += 1

        await asyncio.gather(*(asyncio.create_task(_drain_child(i, child)) for i, child in enumerate(children)))

        child_outputs.sort(key=lambda item: item.get("index", 0))
        sub_agent_ids = [
            item["agent_id"] for item in child_outputs if isinstance(item.get("agent_id"), str) and item.get("agent_id")
        ]

        if failed_count > 0:
            message = (
                f"{failed_count}/{len(children)} children failed. See structured_response.items for per-child results."
            )
            yield ToolEvent(
                type=EventType.POST_MESSAGE,
                title="for_each failed",
                message=message,
                status=MessageStatus.ERROR,
                metadata={
                    "toolName": "spawn_general_purpose_agent_for_each",
                    "toolStatus": MessageStatus.ERROR.value,
                    "busId": parent_bus_id,
                    "subAgentIds": sub_agent_ids,
                    "spawnGroupId": group_id,
                    "forEachAggregate": True,
                    "structured_response": {"items": child_outputs},
                },
            )
            return

        # The per-child events use suffixed IDs (`parent:idx`) for UI routing.
        # Also yield one terminal event for the declared parent tool_call_id so
        # the next LLM context has a valid assistant tool_use -> tool_result pair.
        message = (
            f"Completed for_each delegation for {len(child_outputs)} items. "
            "See structured_response.items for per-child results."
        )
        yield ToolEvent(
            type=EventType.POST_MESSAGE,
            title="for_each complete",
            message=message,
            status=MessageStatus.FINISHED,
            metadata={
                "toolName": "spawn_general_purpose_agent_for_each",
                "toolStatus": MessageStatus.FINISHED.value,
                "busId": parent_bus_id,
                "subAgentIds": sub_agent_ids,
                "spawnGroupId": group_id,
                "forEachAggregate": True,
                "structured_response": {"items": child_outputs},
            },
        )


def get_manager():
    return SpawnAgentManager()


# =============================================================================
# WORKFLOW HELPER (called from TOOL_EXECUTORS)
# =============================================================================


async def spawn_agent(workflow_instance, tool_call: Dict, tool_call_id: str):
    """Workflow helper for spawn_agent tool.

    Called from the workflow's _execute_tool method for legacy (event-driven) tools.
    """
    agent_type = tool_call.get("type", "unknown")
    workflow.logger.info(f"[spawn_agent] Starting type={agent_type}, goal={str(tool_call.get('goal', ''))[:80]}")

    manager = get_manager()
    last_status = None
    last_result = ""
    unavailable_sources: List[Dict] = []
    event_count = 0
    try:
        async for event in manager.execute_tool(workflow_instance, tool_call, tool_call_id):
            event_count += 1
            await workflow_instance._handle_tool_event(event, tool_call_id, "spawn_agent")
            if isinstance(event, ToolEvent) and event.status:
                last_status = event.status
            # Capture the result from the FINISHED event
            if isinstance(event, ToolEvent) and event.data and event.data.get("result"):
                last_result = event.data["result"]
            # Capture data sources the explore couldn't reach (connector not connected)
            if isinstance(event, ToolEvent) and event.data and event.data.get("unavailable_sources"):
                unavailable_sources = event.data["unavailable_sources"]
            # Track usage from events
            if isinstance(event, ToolEvent) and event.data and event.data.get("usage"):
                await workflow_instance._track_usage(
                    event.data["usage"], tool_name="spawn_agent", call_site="explore_agent",
                    browser_completed=(
                        agent_type == "browser"
                        and event.status == MessageStatus.FINISHED
                        and (event.metadata or {}).get("action_type") == "task_complete"
                        and (event.metadata or {}).get("success") is True
                    ),
                )
    except GenerationHandoffRequested as exc:
        # A parked tool keeps its chip open for the next generation to finish.
        if not exc.parked and last_status != MessageStatus.FINISHED:
            await workflow_instance._handle_tool_event(
                ToolEvent(
                    type=EventType.POST_MESSAGE,
                    title="spawn_agent interrupted",
                    message=DEPLOY_INTERRUPTED_TOOL_MESSAGE,
                    status=MessageStatus.FINISHED,
                ),
                tool_call_id,
                "spawn_agent",
            )
        raise

    # A required data source wasn't connected: raise one blocking source_connect
    # gate per source (dedup/fan-in collapses duplicates across sub-agents). Each
    # gate posts its own card to the main chat and blocks until the user resolves
    # it. The explore itself stays FINISHED with a not-connected notice in its
    # result so partial findings are not clobbered for the user/parent.
    if unavailable_sources:
        await raise_source_connect_gates(workflow_instance, unavailable_sources)

    workflow.logger.info(f"[spawn_agent] Finished type={agent_type}, events={event_count}, last_status={last_status}")
    return last_status, last_result
