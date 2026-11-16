"""Blackbox tests: the root picks a code agent's initial sandbox size and sees the final one.

Observable surface:
  - spawn_agent(type='code') accepts ``sandbox_size`` ``{cpu?, memory_gb?}``; an empty
    size or an unknown dimension (disk) fails schema validation.
  - spawn_agent forwards it to spawn_claude_code_agent, which sends it in the
    activity's code_session_request; the create call sends it to the backend.
  - The backend's 400 ``SANDBOX_SIZE_UNSUPPORTED`` becomes a non-retryable typed error
    and the tool fails with the backend's message (it lists the available sizes).
  - A local target with a size fails before any activity starts.
  - The tool result names the final size vs the initial and requested ones, the
    size's source and the resizes, on success, pause and failure; unknown stays unknown.
"""

import asyncio
import json

from _test_utils import BlackboxResult, get_captured_requests, setup_mock, teardown_mock
from pydantic import ValidationError
from temporalio.exceptions import ActivityError, ApplicationError, RetryState

SCHEMA = {"fields": [{"name": "summary", "type": "str", "description": "What changed", "required": True}]}
UNSUPPORTED = (
    "No claude sandbox of at least 16 vCPU, 8 GB RAM is built. Available: 2 vCPU, 4 GB RAM; "
    "2 vCPU, 8 GB RAM; 8 vCPU, 4 GB RAM; 8 vCPU, 8 GB RAM."
)


class _Handle:
    def __init__(self, raise_exc):
        self._raise_exc = raise_exc

    def done(self):
        return True

    def cancel(self):
        pass

    def __await__(self):
        async def _inner():
            raise self._raise_exc

        return _inner().__await__()


class _Workflow:
    class ActivityCancellationType:
        TRY_CANCEL = "TRY_CANCEL"
        WAIT_CANCELLATION_COMPLETED = "WAIT_CANCELLATION_COMPLETED"

    def __init__(self, handle):
        self._handle = handle
        self.started = []
        self.args = []

    def start_activity(self, fn, args=None, **kwargs):
        self.started.append(fn.__name__)
        self.args.append(args)
        return self._handle

    async def wait_condition(self, predicate):
        return None

    @staticmethod
    def now():
        from datetime import datetime, timezone

        return datetime(2026, 11, 16, tzinfo=timezone.utc)

    async def execute_activity(self, fn, args=None, **kwargs):
        if fn.__name__ == "resolve_code_agent_provider_activity":
            return args[0]
        raise AssertionError(f"unexpected activity {fn.__name__}")


class _Instance:
    def __init__(self):
        self.user_id = "u-1"
        self.agent_id = "agent-sandbox-size"
        self.model = "rose-labs/rose-1-flash"
        self.is_paused = False


def _unsupported_activity_error() -> ActivityError:
    cause = ApplicationError(UNSUPPORTED, type="SANDBOX_SIZE_UNSUPPORTED", non_retryable=True)
    error = ActivityError(
        "Activity task failed",
        scheduled_event_id=1,
        started_event_id=2,
        identity="w",
        activity_type="spawn_claude_code_activity",
        activity_id="1",
        retry_state=RetryState.NON_RETRYABLE_FAILURE,
    )
    error.__cause__ = cause
    return error


def test_schema_accepts_a_size_and_refuses_empty_or_disk():
    from utils.tool_schemas import SpawnAgentTool, SpawnAgentToolNoGP

    base = {"tool": "spawn_agent", "type": "code", "task_name": "fix_login", "goal": "g", "output_schema": SCHEMA}
    for model in (SpawnAgentTool, SpawnAgentToolNoGP):
        sized = model(**base, sandbox_size={"cpu": 8})
        assert sized.model_dump()["sandbox_size"] == {"cpu": 8, "memory_gb": None}
        assert model(**base).sandbox_size is None, "omitted: backward compatible"
        for bad in ({}, {"cpu": 8, "disk_gb": 50}, {"cpu": 0}):
            try:
                model(**base, sandbox_size=bad)
                raise AssertionError(f"{model.__name__} accepted {bad}")
            except ValidationError:
                pass
    described = SpawnAgentTool.model_json_schema()["properties"]["sandbox_size"]
    assert "Disk size is not selectable" in str(described), described


async def test_spawn_agent_forwards_the_size_into_the_activity_request():
    import domains.spawn_claude_code_agent.handler as handler
    from domains.spawn_agent.handler import SpawnAgentManager

    fake = _Workflow(_Handle(_unsupported_activity_error()))
    original = handler.workflow
    handler.workflow = fake
    try:
        events = [
            e
            async for e in SpawnAgentManager()._spawn_code(
                _Instance(),
                {"goal": "fix", "output_schema": SCHEMA, "sandbox_size": {"cpu": 8, "memory_gb": None}},
                "tc-size",
            )
        ]
    finally:
        handler.workflow = original
    requests = [arg for arg in fake.args[0] if isinstance(arg, dict) and "sandbox_size" in arg]
    assert requests and requests[0]["sandbox_size"] == {"cpu": 8}, fake.args
    assert events[-1].metadata["error_code"] == "SANDBOX_SIZE_UNSUPPORTED", events[-1].metadata


async def test_create_sends_the_size_and_maps_400_to_a_typed_error():
    from domains.spawn_claude_code_agent.repo import create_e2b_session

    setup_mock(
        {
            "/api/claude-code-session/create": {
                "status": 400,
                "json": {
                    "code": "SANDBOX_SIZE_UNSUPPORTED",
                    "error": UNSUPPORTED,
                    "available": [{"cpu": 2, "memory_gb": 4}],
                },
            }
        }
    )
    try:
        await create_e2b_session(agent_id="agent-1", goal="fix", user_id="u1", sandbox_size={"cpu": 16})
        raise AssertionError("expected SANDBOX_SIZE_UNSUPPORTED")
    except ApplicationError as error:
        assert error.type == "SANDBOX_SIZE_UNSUPPORTED" and error.non_retryable is True
        assert error.message == UNSUPPORTED
    body = get_captured_requests()[-1]["requestBody"]
    body = json.loads(body) if isinstance(body, (str, bytes)) else body
    assert body["sandbox_size"] == {"cpu": 16}, body


async def test_unsupported_size_fails_the_tool_once_and_local_refuses_before_starting():
    import domains.spawn_claude_code_agent.handler as handler
    from domains.spawn_claude_code_agent.handler import SpawnClaudeCodeAgentManager
    from tools.base import MessageStatus

    fake = _Workflow(_Handle(_unsupported_activity_error()))
    original = handler.workflow
    handler.workflow = fake
    try:
        events = [
            e
            async for e in SpawnClaudeCodeAgentManager().execute_tool(
                _Instance(), {"goal": "fix", "output_schema": SCHEMA, "sandbox_size": {"cpu": 16}}, "tc-size"
            )
        ]
        terminal = events[-1]
        assert terminal.status == MessageStatus.ERROR and terminal.message == UNSUPPORTED, terminal.message
        assert terminal.metadata["error_code"] == "SANDBOX_SIZE_UNSUPPORTED"
        assert fake.started == ["spawn_claude_code_activity"], "never retried"

        local = _Workflow(_Handle(AssertionError("no activity for a refused local size")))
        handler.workflow = local
        events = [
            e
            async for e in SpawnClaudeCodeAgentManager().execute_tool(
                _Instance(),
                {"goal": "fix", "output_schema": SCHEMA, "target": "local", "sandbox_size": {"cpu": 8}},
                "tc-local",
            )
        ]
        assert events[-1].status == MessageStatus.ERROR and "target='cloud' only" in events[-1].message
        assert local.started == [], local.started
    finally:
        handler.workflow = original


def test_final_size_reports_requested_initial_resizes_and_unknown():
    from domains.spawn_claude_code_agent import service

    created = {
        "sandbox_size": {"cpu": 2, "memory_gb": 4},
        "sandbox_sizing": {
            "requested": {"cpu": 2, "memory_gb": 4},
            "initial": {"cpu": 2, "memory_gb": 4},
            "source": "request",
        },
    }
    sizing = service.initial_sandbox_sizing(created)
    assert sizing["final"] == {"cpu": 2, "memory_gb": 4} and sizing["source"] == "request", sizing
    sizing = service.record_sandbox_resize(
        sizing,
        from_size={"cpu": 2, "memory_gb": 4},
        to_size={"cpu": 8, "memory_gb": 8},
        applied=True,
        observed={"cpu": 8, "memory_gb": 8},
    )
    sizing = service.record_sandbox_resize(
        sizing,
        from_size={"cpu": 8, "memory_gb": 8},
        to_size={"cpu": 8, "memory_gb": 4},
        applied=False,
        observed={"cpu": 8, "memory_gb": 8},
        view={"request_id": "rq-2", "error": "TARGET_DISK_CAPACITY: restore needs 30 GB", "request_to_settled_ms": 61_000},
    )
    assert sizing["resizes"][-1]["request_id"] == "rq-2", sizing
    assert "last resize settled in 61s" in service.format_sandbox_sizing(sizing)
    assert "last resize error: TARGET_DISK_CAPACITY" in service.format_sandbox_sizing(sizing)
    assert sizing["final"] == {"cpu": 8, "memory_gb": 8} and sizing["initial"] == {"cpu": 2, "memory_gb": 4}
    assert sizing["resize_count"] == 1 and sizing["resize_failed_count"] == 1 and len(sizing["resizes"]) == 2
    text = service.format_code_session_summary({"code_session_id": "cs-1", "sandbox_sizing": sizing})
    assert (
        "Sandbox size: 8 vCPU / 8 GB at the end; started at 2 vCPU / 4 GB (source: request); "
        "requested 2 vCPU / 4 GB; resizes: 1 applied, 1 failed"
    ) in text, text

    for _ in range(service.SANDBOX_RESIZE_HISTORY + 5):
        sizing = service.record_sandbox_resize(sizing, from_size=None, to_size=None, applied=True, observed=None)
    assert len(sizing["resizes"]) == service.SANDBOX_RESIZE_HISTORY, "history is bounded"
    assert sizing["resize_count"] == service.SANDBOX_RESIZE_HISTORY + 6, "the count stays exact"
    assert sizing["final"] is None, "applied without a reported size: unknown, not the old size"
    assert "Sandbox size: unknown at the end" in service.format_sandbox_sizing(sizing)

    resumed = service.initial_sandbox_sizing(
        {
            "sandbox_size": {"cpu": 2, "memory_gb": 4},
            "sandbox_sizing": {
                "requested": {"cpu": 8, "memory_gb": 8},
                "initial": {"cpu": 2, "memory_gb": 4},
                "source": "session",
            },
        }
    )
    assert "not applied: a resumed session keeps the size it holds" in service.format_sandbox_sizing(resumed)
    assert service.initial_sandbox_sizing({"sandbox_size": None}) is None, "local: no size"
    legacy = service.initial_sandbox_sizing({"sandbox_size": {"cpu": 2, "memory_gb": 4}})
    assert legacy["source"] == "unknown" and legacy["requested"] is None, legacy


def test_failure_and_pause_results_carry_the_final_size():
    from domains.spawn_claude_code_agent import service

    sizing = service.initial_sandbox_sizing(
        {
            "sandbox_size": {"cpu": 8, "memory_gb": 8},
            "sandbox_sizing": {"requested": None, "initial": {"cpu": 8, "memory_gb": 8}, "source": "default"},
        }
    )
    facts = service._with_sandbox_sizing({"code_session_id": "cs-9", "branch": "rose/x"}, {"sandbox_sizing": sizing})
    error = service.code_session_failure_error(RuntimeError("ws dropped"), RuntimeError("ws dropped"), facts)
    assert "Sandbox size: 8 vCPU / 8 GB at the end; started at 8 vCPU / 8 GB (source: default)" in error.message
    assert error.details[0]["code_session"]["sandbox_sizing"] == sizing
    # A pause/cancel result returns these facts; the handler's summary renders them.
    closed = service._merge_close_code_session_facts(
        facts, {"code_session": {"code_session_id": "cs-9", "state": "PAUSED"}}
    )
    assert "Sandbox size: 8 vCPU / 8 GB" in service.format_code_session_summary(closed)


async def run_all_tests():
    print("code-agent-sandbox-size blackbox tests:")
    r = BlackboxResult()
    r.record("schema_size_field", test_schema_accepts_a_size_and_refuses_empty_or_disk)
    await r.record_async("spawn_agent_forwards_size", test_spawn_agent_forwards_the_size_into_the_activity_request)
    await r.record_async("create_sends_size_maps_400", test_create_sends_the_size_and_maps_400_to_a_typed_error)
    await r.record_async(
        "unsupported_fails_once_local_refused",
        test_unsupported_size_fails_the_tool_once_and_local_refuses_before_starting,
    )
    r.record("final_size_report", test_final_size_reports_requested_initial_resizes_and_unknown)
    r.record("failure_pause_carry_size", test_failure_and_pause_results_carry_the_final_size)
    teardown_mock()
    return r.summary()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
