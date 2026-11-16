"""Spawn-claude-code-agent domain service — business logic and Temporal activity."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
import uuid as uuid_module
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
import structlog.contextvars
from temporalio import activity
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from domains.spawn_claude_code_agent.config import (
    CODE_AGENT_DRAIN_YIELD_STATUS,
    CODE_AGENT_ERROR_RETRY_LIMIT,
    CODE_AGENT_ERRORS_HEARTBEAT_KEY,
    CODE_AGENT_PAUSED_STATUS,
    CODE_AGENT_RESIZE_YIELD_REASON,
    CODE_AGENT_SESSION_FAILED_CODE,
    CODE_AGENT_USAGE_LIMIT_STATUS,
    CODE_SESSION_HEARTBEAT_KEY,
    CODE_SESSION_IDLE_PAUSE_S,
    CODE_SESSION_KEEPALIVE_INTERVAL_S,
    CODE_SESSION_RESIZE_HEARTBEAT_S,
    DRAIN_CHECK_INTERVAL_S,
    E2B_COST_CENTS_PER_VCPU_SECOND,
    E2B_DEFAULT_SANDBOX_VCPUS,
    ECHO_SUPPRESS_S,
    MAX_UNCHANGED_OUTPUTS,
    REATTACH_PROBE_ATTEMPTS,
    REATTACH_PROBE_BACKOFF_S,
    SDK_POST_RESULT_EXIT_GRACE_S,
    SILENCE_TIMEOUT_S,
    TRANSCRIPT_FLUSH_INTERVAL_S,
    code_agent_verifier_enabled,
)
from domains.spawn_claude_code_agent.repo import (
    check_sandbox_health,
    close_e2b_session,
    create_e2b_session,
    fetch_code_credential_availability,
    keepalive_e2b_session,
    persist_terminal_output,
    report_code_session_subscription_usage,
    report_code_usage_limit,
    report_undelivered_code_turns,
    resize_e2b_session,
    save_claude_code_message,
    save_code_transcript_batch,
)
from helpers.activity_generation_drain import activity_drain_reason
from helpers.activity_heartbeat import heartbeat_resume_details, safe_heartbeat
from helpers.claude_code_ws_client import ClaudeCodeWsClient, ws_endpoint_reachable
from helpers.e2b_log_collector import SandboxLogCollector
from helpers.file_resolution import resolve_file_pointers_from_extracted, transform_schema_for_remote_session
from helpers.github_repo_ref import explicit_github_repo_ref
from helpers.tracing import traced
from providers.rose.subscription_usage import usage_from_rate_limit_event
from tools.base import MessageStatus
from utils.code_transcript import (
    CodeTranscriptBuilder,
    TranscriptRecord,
    render_transcript_text,
    to_ui_record,
)
from utils.error_utils import extract_error_message
from utils.llm_timeouts import VERIFIER_ACTIVITY_TIMEOUT_SECONDS
from utils.response_schema_constraints import explicitly_allows_empty_value, value_satisfies_output_field_constraints
from utils.terminal_filter import TerminalFilter

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def resolve_code_agent_provider(default_provider: str) -> str:
    """Resolve a local provider override outside Temporal workflow code."""
    forced = os.getenv("ROSE_CODE_AGENT_PROVIDER", "").strip().lower()
    return forced if forced in {"claude", "codex"} else default_provider


def check_terminal_fatal_error(
    *,
    code_provider: str,
    exit_code: Optional[int],
    credential_feedback: Optional[list[dict[str, str]]] = None,
) -> Optional[str]:
    """Classify a terminal credential failure from structured CLI signals only.

    Terminal text (tool output, assistant prose, injected turns) is never
    classified: an agent may print any banner string while doing healthy work.
    Claude Code credential failures (usage limit, out of credits, rejected
    login) arrive as structured events and pause the session instead (C15).
    Codex can switch to the funded API key in the sandbox and still exit
    successfully, so its structured credential feedback is terminal only when
    the complete Codex process still failed after that fallback opportunity.
    """
    if code_provider == "codex" and exit_code != 0 and credential_feedback:
        return (
            "Codex credentials are rate-limited or out of quota and the runtime "
            "fallback did not complete successfully."
        )
    return None


def build_codex_credential_feedback(
    code_events: list[Dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract at most one compact exhaustion event per credential source.

    Structured events are not bounded by the 500-row terminal display. This
    preserves the pre/post-fallback ordering even when a long successful run
    scrolls the original OAuth error and fallback marker out of the terminal.
    """
    feedback_by_source: dict[str, str] = {}
    credential_by_source: dict[str, str] = {}
    fallback_seen = False

    for event in code_events:
        if event.get("provider") != "codex":
            continue
        if event.get("type") == "auth_fallback":
            fallback_seen = True
            reason = str(event.get("reason") or "runtime_auth_fallback")
            feedback_by_source.setdefault("oauth", reason[:2000])
            continue
        if event.get("type") != "error":
            continue
        if isinstance(event.get("login_unusable"), dict):
            # The ws-server's structured unusable login (no Azure switch armed): the
            # backend marks exactly that login for reconnect, never a rate limit.
            reason = str(event["login_unusable"].get("reason") or "oauth_refresh_invalid")
            source = "funded_api_key" if fallback_seen else "initial"
            feedback_by_source.setdefault(source, reason[:2000])
            credential = event["login_unusable"].get("credential")
            if isinstance(credential, str) and credential:
                credential_by_source.setdefault(source, credential[:64])
            continue

        error = event.get("error")
        nested_message = error.get("message") if isinstance(error, dict) else None
        message = str(event.get("message") or nested_message or "")
        normalized = message.lower()
        if not any(
            marker in normalized
            for marker in (
                "you've hit your usage limit",
                "usage_limit_reached",
                "purchase more credits",
                "quota exceeded. check your plan and billing details.",
                "insufficient_quota",
                "no credits remaining",
                "add credits to continue",
                "credit_balance_exhausted",
                "credit balance is too low",
            )
        ):
            continue
        source = "funded_api_key" if fallback_seen else "initial"
        feedback_by_source.setdefault(source, message[:2000])

    if fallback_seen and "initial" in feedback_by_source:
        initial_reason = feedback_by_source.pop("initial")
        feedback_by_source["oauth"] = initial_reason
        if "initial" in credential_by_source:
            credential_by_source["oauth"] = credential_by_source.pop("initial")

    return [
        {
            "source": source,
            "reason": feedback_by_source[source],
            **({"credential": credential_by_source[source]} if source in credential_by_source else {}),
        }
        for source in ("initial", "oauth", "funded_api_key")
        if source in feedback_by_source
    ]


def build_credential_feedback(code_events: list[Dict[str, Any]], code_provider: str) -> list[dict[str, str]]:
    """The credential failures a run observed, from structured CLI events only (never terminal text).

    Claude Code: a rejected login (``authentication_failed``) with the
    fingerprint of the credential the sandbox held, so the backend tells an
    expired or since-replaced token from a revoked one. Usage limits and out
    of credits are reported through the C15 usage-limit path instead.
    """
    if code_provider == "codex":
        return build_codex_credential_feedback(code_events)
    for event in code_events:
        unusable = login_unusable_from_event(event)
        if unusable is not None and unusable["provider"] == "claude":
            return [
                {
                    "source": "initial",
                    "reason": unusable["reason"],
                    **({"credential": unusable["credential"]} if unusable.get("credential") else {}),
                }
            ]
    return []


_TIMELINE_EVENT_TYPES = frozenset(
    {
        "session",
        "agent_message",
        "assistant",
        "tool_use",
        "tool_result",
        "result",
        "error",
    }
)


def _tool_detail_from_event(event: dict[str, Any]) -> str:
    """Human-readable one-line detail for a tool_use event."""
    paths = event.get("paths")
    if isinstance(paths, list):
        path_bits = [str(p).strip() for p in paths if str(p).strip()]
        if path_bits:
            return ", ".join(path_bits[:4])

    raw_input = event.get("input")
    if isinstance(raw_input, str) and raw_input.strip():
        return raw_input.strip()[:240]
    if isinstance(raw_input, dict):
        for key in (
            "command",
            "file_path",
            "path",
            "notebook_path",
            "pattern",
            "query",
            "url",
            "content",
            "description",
        ):
            value = raw_input.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip().replace("\n", " ")
                return text[:240]
        try:
            return json.dumps(raw_input, ensure_ascii=False, default=str)[:240]
        except Exception:
            return ""
    return ""


def _project_startup_line(event: Dict[str, Any]) -> str | None:
    """One line for a finished project startup (ws-server ``project_startup`` event)."""
    status = str(event.get("status") or "").strip().lower()
    if status in {"", "running", "skipped"}:
        return None
    project = str(event.get("project") or "").strip()
    label = f"Project startup ({project})" if project else "Project startup"
    duration_ms = event.get("durationMs")
    has_duration = isinstance(duration_ms, (int, float)) and duration_ms > 0
    seconds = f" in {duration_ms / 1000:.1f}s" if has_duration else ""
    if status == "ok":
        return f"{label}: ok{seconds}"
    if status == "timeout":
        return f"{label}: timed out after {event.get('timeoutS') or '?'}s"
    return f"{label}: failed (exit {event.get('exitCode')}){seconds}"


def format_code_events_transcript(
    code_events: list[Dict[str, Any]] | None = None,
    action_summary: list[str] | None = None,
) -> str:
    """Format SDK/code-agent events into FE-parseable coherent log lines.

    Markers match ``ParseCodeAgentTranscript``:
    ``[Claude]``, ``[Claude tool]``, ``[Claude result]``.
    """
    lines: list[str] = []
    # One line per streamed message: partial snapshots superseded by a later
    # snapshot or the final block would repeat its text once per delta.
    for event in _latest_stream_snapshots([event for event in code_events or [] if isinstance(event, dict)]):
        event_type = str(event.get("type") or event.get("event_type") or "").strip().lower()
        provider = str(event.get("provider") or "claude").strip().title() or "Claude"
        if provider.lower() not in {"claude", "codex"}:
            provider = "Claude"

        if event_type == "session":
            cwd = event.get("cwd") or event.get("working_directory")
            detail = f"Session started in {cwd}" if cwd else "Session started"
            lines.append(f"[{provider}] {detail}")
            continue

        if event_type == "project_startup":
            line = _project_startup_line(event)
            if line:
                lines.append(f"[{provider}] {line}")
            continue

        if event_type in {"agent_message", "assistant"}:
            text = event.get("text") or event.get("content") or event.get("message")
            if isinstance(text, str) and text.strip():
                lines.append(f"[{provider}] {text.strip()}")
            continue

        if event_type == "tool_use":
            tool_name = (
                event.get("tool")
                or event.get("name")
                or event.get("tool_name")
                or "Tool"
            )
            detail = _tool_detail_from_event(event)
            if detail:
                lines.append(f"[{provider} tool] {tool_name} · {detail}")
            else:
                lines.append(f"[{provider} tool] {tool_name}")
            continue

        if event_type == "tool_result":
            # Keep the log tool-centric; skip noisy raw tool_result blobs unless they
            # carry a short status/error the FE can show under the next result card.
            status = event.get("status")
            error = event.get("error")
            if isinstance(error, str) and error.strip():
                lines.append(f"[{provider} result] {error.strip()[:400]}")
            elif isinstance(status, str) and status.strip() and status.strip().lower() not in {
                "completed",
                "success",
                "ok",
            }:
                lines.append(f"[{provider} result] status={status.strip()}")
            continue

        if event_type == "result":
            status = str(event.get("status") or "").strip()
            text = event.get("text") or event.get("content") or event.get("result")
            if isinstance(text, str) and text.strip():
                prefix = "failed - " if status.lower() in {"failed", "error"} else ""
                lines.append(f"[{provider} result] {prefix}{text.strip()[:800]}")
            elif status:
                lines.append(f"[{provider} result] {status}")
            continue

        if event_type == "error":
            text = event.get("text") or event.get("error") or event.get("message")
            if isinstance(text, str) and text.strip():
                lines.append(f"[{provider} result] {text.strip()[:800]}")
            continue

        if event_type == "user_turn":
            # Direct chat: the user (or another agent) typed to this CLI.
            text = event.get("text")
            who = "Agent" if event.get("sender") == "agent" else "User"
            if isinstance(text, str) and text.strip():
                lines.append(f"[{who} → {provider}] {text.strip()}")
            continue

    summary_lines = [str(item).strip() for item in (action_summary or []) if str(item).strip()]
    if summary_lines:
        if lines:
            lines.append("")
        lines.append("## Action Summary")
        lines.extend(summary_lines[-30:])

    return "\n".join(lines).strip()


def code_events_have_timeline(code_events: list[Dict[str, Any]] | None) -> bool:
    """True when events carry a real note/tool timeline (prefer over janky PTY)."""
    for event in code_events or []:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or event.get("event_type") or "").strip().lower()
        if event_type in _TIMELINE_EVENT_TYPES:
            return True
    return False


def build_durable_terminal_transcript(
    display_output: str,
    code_events: list[Dict[str, Any]] | None = None,
    action_summary: list[str] | None = None,
) -> str:
    """Prefer structured event timeline; fall back to PTY display / summary.

    ``ClaudeCodeSession.output`` is what the FE CompletedTranscriptSurface renders.
    Raw Claude Code PTY often staircase-wraps and dumps tool lines under a single
    note. Structured ``code_agent_event`` frames are the coherent source of truth
    when present. If only PTY bytes exist, keep them so the chip is never empty.
    """
    formatted_events = format_code_events_transcript(code_events, action_summary)
    if formatted_events and code_events_have_timeline(code_events):
        return formatted_events

    if (display_output or "").strip():
        return display_output

    return formatted_events


def safe_time() -> float:
    """Get current time, returning 0 if in workflow sandbox where time.time() is restricted."""
    try:
        return time.time()
    except Exception:
        return 0.0


def build_context_summary(
    success: bool,
    close_reasoning: str,
    action_summary: list[str],
    error_context: Optional[str] = None,
    verified: bool = True,
) -> str:
    """Build detailed execution context summary for main agent.

    ``verified=False`` marks a result nobody judged (the code-agent verifier is
    disabled and some declared fields have no mechanical evidence): it is
    neither a success nor a failure.
    """
    summary_parts: list[str] = []

    status = "✅ SUCCESS" if success else ("⚠️ UNVERIFIED" if not verified else "❌ FAILED")
    summary_parts.append(f"**Status**: {status}")

    if close_reasoning:
        summary_parts.append(f"\n**Outcome**: {close_reasoning}")

    if error_context:
        summary_parts.append(f"\n**Error**: {error_context}")

    if action_summary:
        last_actions = action_summary[-10:]
        summary_parts.append(f"\n**Actions Taken** ({len(action_summary)} total, showing last {len(last_actions)}):")
        for action in last_actions:
            summary_parts.append(f"  • {action}")

    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Agentic WebSocket loop
# ---------------------------------------------------------------------------


# Claude Code CLI stream-json messages that ws-server forwards verbatim as
# `cli_message` events. (log field, CLI field) pairs; values are scalars from the
# SDK schema (status enums, epoch seconds, counts, categorized error enum).
_CLI_RATE_LIMIT_FIELDS = (
    ("rate_limit_status", "status"),
    ("rate_limit_type", "rateLimitType"),
    ("utilization", "utilization"),
    ("resets_at", "resetsAt"),
    ("overage_status", "overageStatus"),
    ("overage_resets_at", "overageResetsAt"),
    ("overage_disabled_reason", "overageDisabledReason"),
    ("is_using_overage", "isUsingOverage"),
    ("surpassed_threshold", "surpassedThreshold"),
)
_CLI_API_RETRY_FIELDS = (
    ("attempt", "attempt"),
    ("max_retries", "max_retries"),
    ("retry_delay_ms", "retry_delay_ms"),
    ("error_status", "error_status"),
    ("error", "error"),
)
_CLI_LOG_VALUE_MAX_CHARS = 200


def _bounded_cli_log_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_CLI_LOG_VALUE_MAX_CHARS]


def _log_claude_cli_message(
    event: dict[str, Any],
    *,
    agent_id: str,
    task_id: str,
    sandbox_id: str,
) -> None:
    """Log CLI rate-limit status and API retries; other CLI messages stay unlogged."""
    message = event.get("cli_message")
    if not isinstance(message, dict):
        return
    ids = {"agent_id": agent_id, "task_id": task_id, "sandbox_id": sandbox_id}

    if message.get("type") == "rate_limit_event":
        info = message.get("rate_limit_info")
        info = info if isinstance(info, dict) else {}
        logger.info(
            "[ClaudeCode] CLI rate-limit status",
            **ids,
            **{key: _bounded_cli_log_value(info.get(src)) for key, src in _CLI_RATE_LIMIT_FIELDS},
        )
        return

    if message.get("type") == "system" and message.get("subtype") == "api_retry":
        logger.info(
            "[ClaudeCode] CLI API retry",
            **ids,
            **{key: _bounded_cli_log_value(message.get(src)) for key, src in _CLI_API_RETRY_FIELDS},
        )


def _log_code_agent_event(
    event: dict[str, Any],
    *,
    agent_id: str,
    task_id: str,
    sandbox_id: str,
) -> None:
    """Promote high-value sandbox events into retained worker logs.

    Codex Fast Mode events stay provider-specific. Claude (and Codex) tool /
    turn progress must also land at info so local + prod log retention keep
    tool_use / tool_result / agent_message / result visible in bench run JSON
    — debug-only lines are often dropped by prod log drains.
    """
    provider = event.get("provider")
    event_type = event.get("type")

    if provider == "codex":
        if event_type == "fast_cache_proxy_started":
            logger.info(
                "[ClaudeCode] Codex Fast Mode proxy started",
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                base_url=event.get("base_url"),
                target=event.get("target"),
            )
            return

        if event_type == "fast_cache_proxy_request":
            logger.info(
                "[ClaudeCode] Codex Fast Mode proxy request",
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                method=event.get("method"),
                path=event.get("path"),
                status=event.get("status"),
                cache=event.get("cache"),
            )
            return

        if event_type == "fast_cache_proxy_error":
            logger.warning(
                "[ClaudeCode] Codex Fast Mode proxy error",
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                method=event.get("method"),
                path=event.get("path"),
                error=event.get("error"),
            )
            return

        if event_type == "usage":
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            input_tokens = usage.get("input_tokens") or 0
            cached_input_tokens = usage.get("cached_input_tokens") or 0
            cached_pct = (
                round((cached_input_tokens / input_tokens) * 100, 1) if input_tokens else 0.0
            )
            logger.info(
                "[ClaudeCode] Codex turn usage",
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                cached_input_pct=cached_pct,
                output_tokens=usage.get("output_tokens") or 0,
                reasoning_output_tokens=usage.get("reasoning_output_tokens") or 0,
                **({"usage_scope": event["usage_scope"]} if isinstance(event.get("usage_scope"), str) else {}),
            )
            return

    if event_type == "cli_message":
        _log_claude_cli_message(event, agent_id=agent_id, task_id=task_id, sandbox_id=sandbox_id)
        return

    # Progress events for Claude Code + Codex — must survive info-level retention.
    if event_type in {"tool_use", "tool_result", "agent_message", "result", "error", "session"}:
        tool_name = event.get("name") or event.get("tool_name") or event.get("tool")
        logger.info(
            "[ClaudeCode] Code-agent event",
            agent_id=agent_id,
            task_id=task_id,
            sandbox_id=sandbox_id,
            provider=provider,
            event_type=event_type,
            tool_name=tool_name,
            status=event.get("status"),
        )


def is_codex_completed_turn_event(event: dict[str, Any]) -> bool:
    """Return True for normalized Codex events that represent a completed Codex turn."""
    if event.get("provider") != "codex":
        return False
    if event.get("raw_type") == "turn.completed":
        return True
    if event.get("type") == "turn_ended" and event.get("raw_type") == "turn/completed":
        return event.get("turn_outcome", event.get("status")) == "completed"
    return event.get("type") == "turn" and event.get("status") == "completed"


def codex_completed_turn_event_key(event: dict[str, Any], fallback_index: int) -> str:
    if isinstance(event.get("turn_id"), str):
        return f"turn:{event['turn_id']}"
    timestamp = event.get("timestamp")
    if timestamp is None:
        return f"event:{fallback_index}"
    return json.dumps(
        {
            "timestamp": timestamp,
            "raw_type": event.get("raw_type"),
            "type": event.get("type"),
            "status": event.get("status"),
            "usage": event.get("usage"),
        },
        sort_keys=True,
        default=str,
    )


def codex_completed_turn_count(code_events: list[Dict[str, Any]]) -> int:
    """Count unique completed Codex turns from normalized SDK events."""
    seen: set[str] = set()
    for index, event in enumerate(code_events):
        if not is_codex_completed_turn_event(event):
            continue
        seen.add(codex_completed_turn_event_key(event, index))
    return len(seen)


class CodeSessionDrained(Exception):
    """The worker is draining; the activity detaches while the sandbox keeps running."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class CodeSessionResizeRequested(CodeSessionDrained):
    """C16: the code agent asked for another sandbox size; detach so the activity applies it."""

    def __init__(self, request: Dict[str, Any]):
        self.request = request
        super().__init__(CODE_AGENT_RESIZE_YIELD_REASON)


_CODE_SESSION_FIELDS = (
    "task_id",
    "sandbox_id",
    "ws_url",
    "code_provider",
    "github_repo_url",
    "started_at",
    "loop_started_at",
    # Sandbox events already logged by an earlier attachment; the reattached
    # attempt logs only the ones that arrived while nobody was attached.
    "logged_events",
    # Persistent code session (C1/C2): a retry after a failed attempt resumes
    # the same session even when the sandbox itself has been paused.
    "code_session_id",
    "branch",
    "merge_target",
    # A local (Mac) run: a reattaching attempt keeps the same target, device and folder.
    "target",
    "device_id",
    "device_name",
    "folder",
    # Sandbox events whose transcript records are already persisted.
    "transcript_events",
    # vCPUs / memory of the sandbox (C16); the code_agent block reports it.
    "sandbox_size",
    # Attached to a live CLI: its events before this spawn's goal turn are history.
    "history_events",
)


def code_session_record(raw: Any) -> Optional[Dict[str, Any]]:
    """Return the reattachable session in ``raw``, or None when it lacks a sandbox identity."""
    if not isinstance(raw, dict):
        return None
    if not all(isinstance(raw.get(key), str) and raw[key] for key in ("task_id", "sandbox_id", "ws_url")):
        return None
    return {key: raw[key] for key in _CODE_SESSION_FIELDS if raw.get(key) is not None}


@activity.defn
async def close_detached_code_session_activity(agent_id: str, code_session: Dict[str, Any]) -> bool:
    """Close a sandbox a drained attempt left running when no attempt will reattach to it.

    Idempotent: closing an already-closed session is a no-op on the backend.
    """
    session = code_session_record(code_session)
    if session is None:
        return False
    logger.info(
        "[ClaudeCode] Closing detached sandbox",
        agent_id=agent_id,
        task_id=session["task_id"],
        sandbox_id=session["sandbox_id"],
    )
    await close_e2b_session(task_id=session["task_id"], code_provider=session.get("code_provider", "claude"))
    return True


def _next_attempt_reattaches(cancellation: activity.ActivityCancellationDetails) -> bool:
    """Whether Temporal will run another attempt of this activity that reattaches.

    A terminated or reset workflow (``not_found`` / ``reset``) never schedules
    one, and a timeout or worker shutdown leads to one only while the retry
    policy has attempts left. A paused activity resumes with its heartbeats.
    """
    if cancellation.not_found or cancellation.reset:
        return False
    if cancellation.paused:
        return True
    info = activity.info()
    max_attempts = info.retry_policy.maximum_attempts if info.retry_policy else 0
    return max_attempts <= 0 or info.attempt < max_attempts


def _code_session_from_heartbeat() -> Optional[Dict[str, Any]]:
    """Session the previous attempt of this activity was attached to, if any."""
    if not activity.in_activity():
        return None
    info = activity.info()
    if info.attempt <= 1:
        return None
    for detail in info.heartbeat_details or ():
        if isinstance(detail, dict):
            session = code_session_record(detail.get(CODE_SESSION_HEARTBEAT_KEY))
            if session is not None:
                return session
    return None


class ToolsInFlight:
    """Tools a code-agent session started and has not finished.

    Keyed by tool id, so the events repeated for one call (Claude: the streamed
    block start and the assistant envelope; Codex: item/started then
    item/completed) count it once, and one tool_result carrying several calls
    (``tool_results``) finishes each of them. A finished turn has no tool left.
    """

    _CODEX_ITEM_TYPES = frozenset({"tool_use", "command", "file_change"})

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._anonymous = 0  # tool_use events without an id (older sandboxes)

    def __len__(self) -> int:
        return len(self._ids) + self._anonymous

    def clear(self) -> None:
        self._ids.clear()
        self._anonymous = 0

    def observe(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        item_id = event.get("item_id")
        if (
            event.get("provider") == "codex"
            and event_type in self._CODEX_ITEM_TYPES
            and isinstance(item_id, str)
            and event.get("status") in ("started", "completed")
        ):
            if event.get("status") == "started":
                self._ids.add(item_id)
            else:
                self._ids.discard(item_id)
        elif event_type == "tool_use":
            tool_id = event.get("tool_use_id")
            if isinstance(tool_id, str) and tool_id:
                self._ids.add(tool_id)
            else:
                self._anonymous += 1
        elif event_type == "tool_result":
            results = event.get("tool_results")
            ids = [
                tool_id
                for tool_id in (
                    [part.get("tool_use_id") for part in results if isinstance(part, dict)]
                    if isinstance(results, list)
                    else [event.get("tool_use_id")]
                )
                if isinstance(tool_id, str) and tool_id
            ]
            if ids:
                self._ids.difference_update(ids)
            elif self._anonymous:
                self._anonymous -= 1
        elif event_type in ("result", "turn_ended", "final"):
            self.clear()


def _supersede_partial_snapshot(
    code_events: list[Dict[str, Any]], latest_partials: dict[tuple[Any, Any, str], int], index: int
) -> None:
    """Drop the text of the item's previous partial snapshot once a newer one arrives.

    A partial stream event carries the whole block so far, so keeping every
    snapshot grows memory with the square of the output. The stored copy keeps
    its delta and its place (indices identify sandbox events across replays).
    """
    event = code_events[index]
    item_id = event.get("item_id")
    text = event.get("text")
    # An empty final (e.g. redacted thinking) supersedes nothing: the streamed
    # text is the only copy.
    if not isinstance(item_id, str) or not isinstance(text, str) or not text:
        return
    key = (event.get("type"), event.get("turn_id"), item_id)
    previous = latest_partials.pop(key, None)
    if previous is not None:
        code_events[previous] = {k: v for k, v in code_events[previous].items() if k != "text"}
    if event.get("partial") is True:
        latest_partials[key] = index


async def _run_agentic_loop(
    *,
    agent_id: str,
    task_id: str,
    sandbox_id: str,
    goal: str,
    ws_url: str,
    terminal: TerminalFilter,
    model: str,
    claude_token: Optional[str],
    action_summary: list[str],
    tool_call_id: str,
    gpt_token: Optional[str] = None,
    gpt_account_id: Optional[str] = None,
    gpt_token_source: Optional[str] = None,
    gpt_token_pool_id: Optional[str] = None,
    gpt_token_user_id: Optional[str] = None,
    code_provider: str = "claude",
    code_events: list[Dict[str, Any]] | None = None,
    terminal_output_persisted: Optional[list[bool]] = None,
    loop_started_at: Optional[float] = None,
    generation_yield: bool = True,
    code_session: Optional[Dict[str, Any]] = None,
    loop_outcome: Optional[Dict[str, Any]] = None,
) -> tuple[int, bool, list[Dict[str, str]], int, int, list[Dict[str, Any]]]:
    """Run the WebSocket-based agentic loop monitoring Claude Code output.

    Returns (exit_code, attempt_completion, conversation_history,
             total_input_tokens, total_output_tokens, code_events).
    exit_code is -1 if the process has not exited.

    ``terminal_output_persisted`` is an optional single-element list shared with
    the activity so fatal-path persistence is visible to outer cleanup.

    Raises ``CodeSessionDrained`` when the worker drains (or, with
    ``generation_yield``, its generation is superseded) before Claude Code
    exits. There is no runtime limit (C11): liveness is the activity heartbeat.
    A CLI that sends nothing for ``CODE_SESSION_IDLE_PAUSE_S`` with no tool in
    flight and no SDK result ends the loop with ``loop_outcome["idle_paused"]``
    set, and the caller pauses (never fails) the session. A tool in flight keeps
    the loop attached indefinitely. ``code_session["logged_events"]`` is kept
    current so each sandbox event is logged once across every attachment.
    """
    last_output_time = safe_time()
    exit_code: Optional[int] = None
    pending_output = False
    conversation_history: List[Dict[str, str]] = []
    attempt_completion = False
    ws_connected = True
    last_filtered_output = ""
    unchanged_output_count = 0
    last_stdin_send_time: float = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    code_events = code_events if code_events is not None else []
    completed_codex_turn_keys: set[str] = set()
    completed_codex_turns = 0
    # Tools awaiting their result: while any is in flight the session is
    # working and never idle-paused, however long the tool runs.
    tools_in_flight = ToolsInFlight()
    latest_partials: dict[tuple[Any, Any, str], int] = {}
    # Wall-clock of first SDK type=result; the exit-grace backstop ends the
    # monitor if the CLI never exits after declaring completion.
    sdk_result_at: float | None = None
    logged_events = int((code_session or {}).get("logged_events") or 0)
    # Attached to a live CLI (resumed_from=live_cli): the ws-server's events
    # before this spawn's goal turn belong to earlier spawns.
    history_events = int((code_session or {}).get("history_events") or 0)
    # Direct chat: while the user holds the session (ws-server `direct_chat`
    # held → released), the user's turns drive the CLI. The monitor must not
    # end the session on post-result grace / SDK idle, and never writes to
    # stdin, so no parent instruction competes with the user.
    direct_chat_active = False
    # Structured transcript (utils/code_transcript.py): records derive from the sandbox's
    # append-only event history, so `transcript_events` (events already
    # persisted) survives reattaches the same way `logged_events` does.
    transcript = CodeTranscriptBuilder(code_provider, goal=goal)
    transcript_flushed = int((code_session or {}).get("transcript_events") or 0)
    transcript_pending: dict[str, TranscriptRecord] = {}
    last_transcript_flush = 0.0
    # Every delivered user turn (direct chat or an agent-graph message) adds
    # one more SDK `result` before the CLI is done: a message that arrives
    # mid-run makes the first result an intermediate one, so the post-result
    # exit grace arms only once results catch up with turns.
    user_turns_seen = 0
    sdk_results_seen = 0
    native_codex_terminal: dict[str, Any] | None = None
    # The native Codex thread goal (ws-server `goal` events). While it is active
    # Codex starts continuation turns itself, so a completed turn is not the end.
    codex_goal_status: str | None = None
    # Agent turns the ws-server accepted (`queued`) and lost when the session
    # ended (`user_turn_undelivered`): their senders are told once (the backend
    # is idempotent per turn; this set only skips repeats within one attach).
    reported_undelivered: set[str] = set()

    diagnostic_turn_id: str | None = None
    last_monitor_diagnostic: tuple | None = None

    def log_monitor_state(reason: str = "state_changed") -> None:
        # Fixed metadata only: never serialize events, objectives, arguments or auth.
        nonlocal last_monitor_diagnostic
        terminal = native_codex_terminal or {}
        turn_id = terminal.get("turn_id")
        if not isinstance(turn_id, str) or len(turn_id) > 128 or not all(
            char.isascii() and (char.isalnum() or char in "-_.:") for char in turn_id
        ):
            turn_id = None
        goal_status = codex_goal_status if codex_goal_status in {
            None, "active", "complete", "budgetLimited", "cleared"
        } else "other"
        outcome = terminal.get("turn_outcome", terminal.get("status"))
        outcome = outcome if outcome in {"completed", "failed", "interrupted"} else None
        current_turn_id = diagnostic_turn_id
        if not isinstance(current_turn_id, str) or len(current_turn_id) > 128 or not all(
            char.isascii() and (char.isalnum() or char in "-_.:") for char in current_turn_id
        ):
            current_turn_id = None
        state = (goal_status, direct_chat_active, current_turn_id, turn_id, outcome)
        if reason == "state_changed" and state == last_monitor_diagnostic:
            return
        last_monitor_diagnostic = state
        try:
            logger.info(
                "[ClaudeCode] Monitor state",
                agent_id=agent_id, task_id=task_id, provider=code_provider,
                reason=reason, goal_status=goal_status, direct_hold=direct_chat_active,
                current_turn_id=current_turn_id, terminal_turn_id=turn_id, terminal_outcome=outcome,
                completion_admissible=(
                    code_provider == "codex" and native_codex_terminal is not None
                    and not direct_chat_active and codex_goal_status != "active"
                ),
            )
        except Exception:
            pass  # Diagnostic sink failures cannot affect the monitor.

    def _mark_terminal_output_persisted() -> None:
        if terminal_output_persisted is not None:
            terminal_output_persisted[0] = True

    async def on_output(data: str, is_buffer: bool) -> None:
        nonlocal last_output_time, pending_output
        if is_buffer:
            terminal.reset()
        terminal.feed(data)
        last_output_time = safe_time()
        if last_stdin_send_time and (safe_time() - last_stdin_send_time) < ECHO_SUPPRESS_S:
            return
        pending_output = True

    async def on_exit(code: int) -> None:
        nonlocal exit_code
        exit_code = code
        log_monitor_state("process_exit")
        logger.info("[ClaudeCode] Process exited", agent_id=agent_id, task_id=task_id, exit_code=code)

    async def on_connection_change(connected: bool) -> None:
        nonlocal ws_connected, last_output_time
        ws_connected = connected
        if connected:
            last_output_time = safe_time()
            logger.info("[ClaudeCode] WebSocket reconnected, silence timer reset", agent_id=agent_id, task_id=task_id)
        else:
            logger.info(
                "[ClaudeCode] WebSocket disconnected, pausing silence detection", agent_id=agent_id, task_id=task_id
            )

    async def on_stdin_error(error_msg: str) -> None:
        nonlocal pending_output
        logger.error(
            "[ClaudeCode] Server reported stdin error — re-enabling silence detection for retry",
            agent_id=agent_id,
            task_id=task_id,
            error=error_msg,
        )
        pending_output = True

    async def on_events_replay(events: list[dict[str, Any]]) -> None:
        # The ws-server resends its whole history on every (re)connect, so it
        # replaces everything seen so far instead of appending to it.
        nonlocal completed_codex_turns
        nonlocal direct_chat_active, transcript, user_turns_seen, sdk_results_seen, native_codex_terminal
        nonlocal codex_goal_status, diagnostic_turn_id
        diagnostic_turn_id = None
        code_events.clear()
        latest_partials.clear()
        native_codex_terminal = None
        codex_goal_status = None
        direct_chat_active = False
        user_turns_seen = 0
        sdk_results_seen = 0
        transcript = CodeTranscriptBuilder(code_provider, goal=goal)
        transcript_pending.clear()
        completed_codex_turn_keys.clear()
        completed_codex_turns = 0
        tools_in_flight.clear()
        for event in events:
            await on_event(event)
        logger.info(
            "[ClaudeCode] Replayed sandbox event history", agent_id=agent_id, task_id=task_id, events=len(events)
        )

    async def on_event(event: dict[str, Any]) -> None:
        nonlocal completed_codex_turns, last_output_time
        nonlocal sdk_result_at, logged_events
        nonlocal direct_chat_active, user_turns_seen, sdk_results_seen, native_codex_terminal
        nonlocal codex_goal_status
        nonlocal diagnostic_turn_id
        if event.get("provider") == "codex" and event.get("type") in ("turn_started", "turn_ended", "user_turn"):
            diagnostic_turn_id = event.get("turn_id")
        last_output_time = safe_time()
        code_events.append(event)
        event_index = len(code_events) - 1
        # Events before this spawn's goal turn (attached to a live CLI) are the
        # earlier turns' history: their usage limit or login failure is not this run's.
        history = event_index < history_events
        if not history and event.get("type") == "user_turn_undelivered":
            lost = [
                str(turn_id)
                for turn_id in event.get("turn_ids") or []
                if isinstance(turn_id, str) and turn_id and turn_id not in reported_undelivered
            ]
            reported_undelivered.update(lost)
            await report_undelivered_code_turns(agent_id, task_id, lost, str(event.get("reason") or "rejected"))
        usage_limit = None if history else usage_limit_from_event(event)
        if usage_limit is not None and loop_outcome is not None and "usage_limit" not in loop_outcome:
            loop_outcome["usage_limit"] = usage_limit
        login_unusable = None if history else login_unusable_from_event(event)
        if login_unusable is not None and loop_outcome is not None and "login_unusable" not in loop_outcome:
            loop_outcome["login_unusable"] = login_unusable
        for record in transcript.feed(event, event_index):
            if event_index >= transcript_flushed:
                transcript_pending[record["id"]] = record
        _supersede_partial_snapshot(code_events, latest_partials, event_index)
        if event.get("provider") == "codex":
            if event.get("type") == "goal":
                codex_goal_status = event.get("status") if isinstance(event.get("status"), str) else None
            if event.get("type") == "turn_started":
                native_codex_terminal = None
            elif event.get("type") == "user_turn" and (
                native_codex_terminal is not None
                and event.get("turn_id") != native_codex_terminal.get("turn_id")
            ):
                native_codex_terminal = None
            elif (
                event.get("type") == "turn_ended"
                and event.get("raw_type") == "turn/completed"
                and event.get("turn_outcome", event.get("status")) in {"completed", "failed", "interrupted"}
            ):
                native_codex_terminal = event
        if event.get("type") == "direct_chat":
            direct_chat_active = event.get("status") == "held"
            if not direct_chat_active and sdk_result_at is not None:
                # Released after a finished turn: the CLI is told to exit now,
                # so restart the post-result grace from the hand back.
                sdk_result_at = safe_time()
        elif event.get("type") == "user_turn":
            # A new turn: the next SDK result belongs to it.
            user_turns_seen += 1
            sdk_result_at = None
        elif event.get("type") == "turn_started" and sdk_result_at is not None:
            # The CLI started a turn of its own after a caught-up result (a /goal
            # continuation, a background-task notification): it is working again,
            # so that result was not its last. The turn's own result re-arms the grace.
            user_turns_seen += 1
            sdk_result_at = None
        if event.get("type") in ("goal", "turn_started", "turn_ended", "user_turn", "direct_chat"):
            log_monitor_state()
        if is_codex_completed_turn_event(event):
            key = codex_completed_turn_event_key(event, len(code_events) - 1)
            if key not in completed_codex_turn_keys:
                completed_codex_turn_keys.add(key)
                completed_codex_turns += 1
        event_type = event.get("type")
        tools_in_flight.observe(event)
        if event_type == "result":
            # The result of the latest delivered turn arms the exit-grace
            # backstop (r31 a2/a3); an earlier turn's result does not.
            sdk_results_seen += 1
            # The ws-server knows when the CLI has caught up: Claude Code merges
            # a turn written mid-turn into the running one (one result for two
            # writes) and reports its queue. Older sandboxes: count results.
            pending_turns = event.get("pending_turns")
            caught_up = (
                pending_turns == 0 if isinstance(pending_turns, int) else sdk_results_seen > user_turns_seen
            )
            if sdk_result_at is None and caught_up:
                sdk_result_at = safe_time()
                logger.info(
                    "[ClaudeCode] SDK result observed; arming post-result exit grace",
                    agent_id=agent_id,
                    task_id=task_id,
                    grace_s=SDK_POST_RESULT_EXIT_GRACE_S,
                    status=event.get("status"),
                )
        # code_events mirrors the sandbox's append-only event history, so an
        # event's index identifies it across reconnect replays and reattaches.
        if len(code_events) <= logged_events:
            return
        logged_events = len(code_events)
        if code_session is not None:
            code_session["logged_events"] = logged_events
        # Below the replay guard: a reconnect resends the whole history, and a
        # replayed rate_limit_event must not be reported again as a fresh observation.
        subscription_usage = None if history else subscription_usage_from_event(event)
        if subscription_usage is not None:
            report_code_session_usage_soon(task_id, subscription_usage)
        _log_code_agent_event(event, agent_id=agent_id, task_id=task_id, sandbox_id=sandbox_id)
        logger.debug(
            "[ClaudeCode] Received code-agent event",
            agent_id=agent_id,
            task_id=task_id,
            event_type=event_type,
            provider=event.get("provider"),
        )

    async def flush_transcript(force: bool = False) -> None:
        nonlocal transcript_flushed, last_transcript_flush
        if not transcript_pending:
            return
        if not force and safe_time() - last_transcript_flush < TRANSCRIPT_FLUSH_INTERVAL_S:
            return
        batch = dict(transcript_pending)
        events_to = len(code_events)
        records = sorted(batch.values(), key=lambda record: (record["seq"], record["rev"]))
        last_transcript_flush = safe_time()
        saved = await save_code_transcript_batch(
            agent_id=agent_id,
            task_id=task_id,
            tool_call_id=tool_call_id,
            code_provider=code_provider,
            records=[to_ui_record(record) for record in records],
            events_from=transcript_flushed,
            events_to=events_to,
            text=render_transcript_text(records),
        )
        if not saved:
            return  # keep the batch; the next flush retries it
        for record_id, record in batch.items():
            # A newer revision that arrived during the save stays pending.
            if transcript_pending.get(record_id) is record:
                del transcript_pending[record_id]
        transcript_flushed = events_to
        if code_session is not None:
            code_session["transcript_events"] = events_to
            # Carry the new mark in a heartbeat so a retry after a drain or a
            # crash does not re-persist this range.
            safe_heartbeat("Code transcript persisted")

    ws_client = ClaudeCodeWsClient(
        ws_url=ws_url,
        on_output=on_output,
        on_exit=on_exit,
        on_connection_change=on_connection_change,
        on_stdin_error=on_stdin_error,
        on_event=on_event,
        heartbeat_fn=safe_heartbeat,
        on_events_replay=on_events_replay,
    )

    # A reattach after a long run must survive a brief blip: one failed
    # handshake used to end the whole session (tar_2b02efd49fc2466497723b3e).
    if not await ws_client.connect_with_retry():
        raise Exception(f"Failed to connect to Claude Code WebSocket: {ws_client.last_error or 'unknown error'}")

    listen_task = asyncio.create_task(ws_client.listen())
    action_summary.append("Connected to Claude Code WebSocket")

    loop_start = loop_started_at if loop_started_at is not None else safe_time()
    last_status_update = safe_time()
    # Codex reports completed turns; Claude runs have no progress counter, so they
    # write no status rows (their transcript batches carry liveness).
    last_status_progress: Optional[int] = 0 if code_provider == "codex" else None
    last_drain_check = 0.0
    last_keepalive = safe_time()

    try:
        while exit_code is None and not attempt_completion:
            elapsed = safe_time() - loop_start
            safe_heartbeat(f"Monitoring Claude Code ({int(elapsed)}s)")

            if safe_time() - last_drain_check >= DRAIN_CHECK_INTERVAL_S:
                last_drain_check = safe_time()
                drain_reason = await activity_drain_reason(allow_generation_handoff=generation_yield)
                if drain_reason:
                    logger.info(
                        "[ClaudeCode] Worker draining; detaching from the running sandbox",
                        agent_id=agent_id,
                        task_id=task_id,
                        sandbox_id=sandbox_id,
                        reason=drain_reason,
                        elapsed_s=int(elapsed),
                    )
                    log_monitor_state("worker_drain_detach")
                    raise CodeSessionDrained(drain_reason)

            # C15: a provider usage limit (or an unusable login the sandbox may
            # not switch away from) ends this run; the session is saved and the
            # parent retries on the next available credential.
            if loop_outcome is not None and loop_outcome.get("login_unusable") and not loop_outcome.get("usage_limit"):
                logger.warning(
                    "[ClaudeCode] Provider login unusable; saving the session",
                    agent_id=agent_id,
                    task_id=task_id,
                    login_unusable=loop_outcome["login_unusable"],
                )
                log_monitor_state("provider_login_unusable")
                action_summary.append("Provider login unusable; session saved for a retry")
                break
            if loop_outcome is not None and loop_outcome.get("usage_limit"):
                logger.warning(
                    "[ClaudeCode] Provider usage limit reached; saving the session",
                    agent_id=agent_id,
                    task_id=task_id,
                    usage_limit=loop_outcome["usage_limit"],
                )
                log_monitor_state("provider_usage_limit")
                action_summary.append("Provider usage limit reached; session saved for a retry")
                break

            # No runtime limit (C11). While attached, keep the sandbox's idle
            # window open; once keepalives stop it pauses on its own.
            if safe_time() - last_keepalive >= CODE_SESSION_KEEPALIVE_INTERVAL_S:
                last_keepalive = safe_time()
                keepalive = await keepalive_e2b_session(task_id, code_provider)
                resize = keepalive.get("resize_requested") if isinstance(keepalive, dict) else None
                if isinstance(resize, dict) and resize.get("request_id") and not direct_chat_active:
                    # C16: the code agent asked for another sandbox size. Detach (the
                    # final transcript flush still runs); the activity applies it.
                    logger.info(
                        "[ClaudeCode] Code agent asked for another sandbox size; detaching to apply it",
                        agent_id=agent_id,
                        task_id=task_id,
                        request_id=resize.get("request_id"),
                        to=resize.get("to"),
                    )
                    action_summary.append(f"Sandbox resize requested: {resize.get('to')}")
                    log_monitor_state("resize_detach")
                    raise CodeSessionResizeRequested(resize)

            # Codex app-server deliberately stays alive between native turns.
            # Its terminal lifecycle, rather than process exit or a silence
            # timeout, completes this attachment. A later start invalidates it;
            # a direct-chat hold keeps the attachment open until hand-back.
            if (
                code_provider == "codex"
                and native_codex_terminal is not None
                and not direct_chat_active
                and codex_goal_status != "active"
            ):
                outcome = native_codex_terminal.get("turn_outcome", native_codex_terminal.get("status"))
                exit_code = 0 if outcome == "completed" else 1
                log_monitor_state("native_terminal_admitted")
                action_summary.append(f"Codex native turn ended: {outcome}")
                break

            # After the first SDK result, healthy CLIs exit in <1s. If the
            # process is still alive past the grace window (stdout may keep
            # refreshing last_output_time), end the monitor: the run is
            # verified on its result and closing pauses (not kills) the sandbox.
            if (
                sdk_result_at is not None
                and exit_code is None
                and not direct_chat_active
                and (safe_time() - sdk_result_at) >= SDK_POST_RESULT_EXIT_GRACE_S
            ):
                log_monitor_state("sdk_post_result_exit_grace")
                grace_elapsed = safe_time() - sdk_result_at
                logger.warning(
                    "[ClaudeCode] Post-result exit grace exceeded",
                    agent_id=agent_id,
                    task_id=task_id,
                    grace_s=SDK_POST_RESULT_EXIT_GRACE_S,
                    since_result_s=grace_elapsed,
                )
                action_summary.append(
                    f"Post-result exit grace exceeded ({int(grace_elapsed)}s after SDK result)"
                )
                break

            if ws_client.reconnects_exhausted:
                await _handle_reconnect_exhausted(
                    sandbox_id, agent_id, task_id, action_summary, code_provider, last_error=ws_client.last_error
                )

            silence_duration = safe_time() - last_output_time

            # Idle → pause, never fail (C11). A tool in flight is work, so only a
            # CLI with nothing in flight and no SDK result is idle.
            if (
                silence_duration >= CODE_SESSION_IDLE_PAUSE_S
                and not tools_in_flight
                and sdk_result_at is None
                and not pending_output
                and ws_connected
                and exit_code is None
                and not direct_chat_active
            ):
                logger.info(
                    "[ClaudeCode] Code session idle; pausing it (resumable)",
                    agent_id=agent_id,
                    task_id=task_id,
                    silence_s=silence_duration,
                    idle_pause_s=CODE_SESSION_IDLE_PAUSE_S,
                )
                log_monitor_state("idle_pause")
                action_summary.append(
                    f"Session idle for {int(silence_duration)}s with no tool in flight; pausing it (resumable)"
                )
                if loop_outcome is not None:
                    loop_outcome["idle_paused"] = True
                    loop_outcome["idle_s"] = int(silence_duration)
                break

            if silence_duration >= SILENCE_TIMEOUT_S and pending_output and ws_connected:
                filtered_output = terminal.get_display()
                pending_output = False

                if filtered_output:
                    if filtered_output == last_filtered_output:
                        unchanged_output_count += 1
                        if unchanged_output_count >= MAX_UNCHANGED_OUTPUTS:
                            logger.info(
                                "[ClaudeCode] Terminal output unchanged for %d checks, skipping LLM call",
                                unchanged_output_count,
                                agent_id=agent_id,
                                task_id=task_id,
                            )
                            continue
                    else:
                        unchanged_output_count = 0
                        last_filtered_output = filtered_output

                    # Credential failures are structured CLI events (C15: usage limit,
                    # out of credits, rejected login) that pause the session; the
                    # terminal text is never classified (an agent may print any banner).
                    logger.info(
                        "[ClaudeCode] Code-agent output observed; waiting for SDK completion",
                        agent_id=agent_id,
                        task_id=task_id,
                        provider=code_provider,
                        output_len=len(filtered_output),
                        silence_s=silence_duration,
                    )
                    continue

            await flush_transcript()

            # A status row only when the reported progress changed: elapsed time
            # is derivable from started_at, and a row per interval grows the
            # tool call's lifecycle rows (and their per-write scan) with run time.
            progress = completed_codex_turns if code_provider == "codex" else None
            if progress != last_status_progress and safe_time() - last_status_update >= 30:
                await save_claude_code_message(
                    agent_id=agent_id,
                    title="Claude Code working",
                    task_id=task_id,
                    message=f"Completed turns: {progress}",
                    status=MessageStatus.FINISHED,
                    action_type="agent_observation",
                    tool_call_id=tool_call_id,
                    code_provider=code_provider,
                    completed_turns=progress,
                )
                last_status_progress = progress
                last_status_update = safe_time()

            await asyncio.sleep(0.1)

    finally:
        listen_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listen_task
        # Close the socket too: after a drain the sandbox outlives this attempt
        # and would otherwise keep buffering broadcasts for a dead reader.
        with contextlib.suppress(Exception):
            await ws_client.disconnect()
        try:
            await flush_transcript(force=True)
        except Exception as flush_err:  # never mask the loop's own outcome
            logger.warning(
                "[ClaudeCode] Final transcript flush failed",
                agent_id=agent_id,
                task_id=task_id,
                error=str(flush_err)[:200],
            )

    logger.info(
        "[ClaudeCode] Agentic loop finished",
        agent_id=agent_id,
        task_id=task_id,
        exit_code=exit_code,
        attempt_completion=attempt_completion,
    )
    action_summary.append("Agentic loop finished")

    return (
        exit_code if exit_code is not None else -1,
        attempt_completion,
        conversation_history,
        total_input_tokens,
        total_output_tokens,
        code_events,
    )


def _size_label(size: Any) -> str:
    if isinstance(size, dict) and size.get("cpu") and size.get("memory_gb"):
        return f"{size['cpu']} vCPU, {size['memory_gb']} GB RAM"
    return "a new size"


async def _apply_sandbox_resize(
    *,
    agent_id: str,
    tool_call_id: str,
    task_id: str,
    code_provider: str,
    request: Dict[str, Any],
    session_record: Dict[str, Any],
    code_agent_meta: Dict[str, Any],
    started_at: float,
    facts: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """C16: have the backend move the session into a sandbox of the requested size.

    Heartbeats while the backend checkpoints, creates and restores. Returns the
    session record of the task that continues the session (new sandbox, or the
    old size when the move failed), or None when the backend could not apply it;
    the caller then reattaches to the old sandbox as after a drain.
    """
    call = asyncio.create_task(resize_e2b_session(task_id, str(request["request_id"]), code_provider))
    while not call.done():
        safe_heartbeat("Resizing the code-agent sandbox")
        await asyncio.wait({call}, timeout=CODE_SESSION_RESIZE_HEARTBEAT_S)
    resized = call.result()
    continuing = code_session_record(resized)
    if not isinstance(resized, dict) or continuing is None:
        logger.warning("[ClaudeCode] Sandbox resize not applied; reattaching", agent_id=agent_id, task_id=task_id)
        return None
    size = resized.get("sandbox_size") if isinstance(resized.get("sandbox_size"), dict) else None
    applied = resized.get("resized") is True
    # The record the next attempt reattaches to (also what later heartbeats carry).
    session_record.update(
        {
            "task_id": continuing["task_id"],
            "sandbox_id": continuing["sandbox_id"],
            "ws_url": continuing["ws_url"],
            "logged_events": 0,
            "transcript_events": 0,
            # The resized sandbox's ws-server is new: none of its events are history.
            "history_events": int(resized.get("history_events") or 0),
            "sandbox_size": size or session_record.get("sandbox_size"),
            "code_session_id": resized.get("code_session_id") or session_record.get("code_session_id"),
        }
    )
    if size is not None:
        code_agent_meta["sandbox_size"] = size
    view = resized.get("request") if isinstance(resized.get("request"), dict) else {}
    code_agent_meta["sandbox_sizing"] = record_sandbox_resize(
        code_agent_meta.get("sandbox_sizing"),
        from_size=view.get("from") or request.get("from"),
        to_size=view.get("to") or request.get("to"),
        applied=applied,
        observed=size,
    )
    session_record["sandbox_sizing"] = code_agent_meta["sandbox_sizing"]
    title = (
        f"Environment resized to {_size_label(size)}"
        if applied
        else f"Environment resize failed; still on {_size_label(size)}"
    )
    await save_claude_code_message(
        agent_id=agent_id,
        title=title,
        task_id=continuing["task_id"],
        message=str(view.get("reason") or title),
        status=MessageStatus.FINISHED if applied else MessageStatus.ERROR,
        action_type="environment_resized",
        tool_call_id=tool_call_id,
        ws_url=continuing["ws_url"],
        code_provider=code_provider,
        code_agent={
            **build_code_agent_metadata(
                target_meta=code_agent_meta,
                task_id=continuing["task_id"],
                provider=code_provider,
                state="live",
                started_at=started_at,
                facts=facts,
            ),
            "environment_resize": {
                "request_id": request.get("request_id"),
                "status": "done" if applied else "failed",
                "from": view.get("from") or request.get("from"),
                "to": view.get("to") or request.get("to"),
                "reason": view.get("reason"),
                "error": view.get("error"),
                "from_task_id": task_id,
            },
        },
    )
    logger.info(
        "[ClaudeCode] Session continues in a resized sandbox" if applied else "[ClaudeCode] Resize failed; continuing",
        agent_id=agent_id,
        from_task_id=task_id,
        task_id=continuing["task_id"],
        sandbox_id=continuing["sandbox_id"],
        sandbox_size=size,
    )
    return dict(session_record)


async def _handle_reconnect_exhausted(
    sandbox_id: str,
    agent_id: str,
    task_id: str,
    action_summary: list[str],
    code_provider: str = "claude",
    last_error: Optional[str] = None,
) -> None:
    """Handle WebSocket reconnection exhaustion by diagnosing sandbox state."""
    logger.error(
        "[ClaudeCode] WebSocket connection lost after all reconnect attempts",
        agent_id=agent_id,
        task_id=task_id,
    )

    sandbox_health = await check_sandbox_health(sandbox_id, agent_id, code_provider)

    if sandbox_health:
        if sandbox_health.get("alive"):
            ws_running = sandbox_health.get("ws_server_running", False)
            claude_running = sandbox_health.get("claude_running", False)
            diagnostics = sandbox_health.get("diagnostics", {})

            logger.error(
                "[ClaudeCode] DIAGNOSIS: E2B sandbox is ALIVE but WS disconnected",
                sandbox_id=sandbox_id,
                ws_server_running=ws_running,
                claude_running=claude_running,
                diagnosis="ws-server.mjs or claude process crashed" if not ws_running else "WS connection issue",
            )

            if diagnostics:
                logger.error(
                    "[ClaudeCode] DIAGNOSTICS extracted from sandbox",
                    sandbox_id=sandbox_id,
                    oom_check=diagnostics.get("oom_check", "N/A")[:500],
                    memory=diagnostics.get("memory", "N/A")[:500],
                    processes=diagnostics.get("processes", "N/A")[:1000],
                    crash_files=diagnostics.get("crash_files", "N/A")[:200],
                    ws_log=diagnostics.get("ws_log", "N/A")[:1000],
                )

            action_summary.append(
                f"DIAGNOSIS: Sandbox alive (ws={ws_running}, claude={claude_running}) - internal crash"
            )
        else:
            logger.error(
                "[ClaudeCode] DIAGNOSIS: E2B sandbox is DEAD",
                sandbox_id=sandbox_id,
                error=sandbox_health.get("error"),
                diagnosis="E2B infrastructure killed the sandbox",
            )
            action_summary.append(f"DIAGNOSIS: Sandbox DEAD - {sandbox_health.get('error', 'unknown')}")
    else:
        action_summary.append("DIAGNOSIS: Could not determine sandbox health")

    detail = f": {last_error}" if last_error else ""
    raise Exception(f"WebSocket connection to Claude Code sandbox lost after all reconnect attempts{detail}")


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


_ABSOLUTE_PATH_RE = re.compile(r"/home/user/[A-Za-z0-9._/\-]+")
_CODEX_COMMAND_RE = re.compile(r"\[Codex command (?:started|completed)\]\s*(.+)")


def _build_codex_evidence_summary(final_output: str) -> str:
    """Summarize Codex's readable transcript for the schema verifier.

    Codex runs non-interactively and reports file paths/commands as terminal
    events. Terminal wrapping can split markdown labels, so this summary pulls
    absolute paths and command markers from the full transcript before the
    verifier sees only the recent terminal tail.
    """
    if not final_output:
        return ""

    paths = sorted(set(_ABSOLUTE_PATH_RE.findall(final_output)))
    commands = []
    for match in _CODEX_COMMAND_RE.finditer(final_output):
        command = match.group(1).strip()
        if command and command not in commands:
            commands.append(command)

    exit_codes = sorted(set(re.findall(r"\[Codex exit_code\]\s*([^\s]+)", final_output)))

    if not paths and not commands and not exit_codes:
        return ""

    lines = ["Codex Evidence Summary:"]
    if paths:
        lines.append("Absolute paths mentioned by Codex:")
        lines.extend(f"- {path}" for path in paths[:20])
    if commands:
        lines.append("Commands executed by Codex:")
        lines.extend(f"- {command}" for command in commands[:10])
    if exit_codes:
        lines.append("Codex command exit codes: " + ", ".join(exit_codes[:10]))

    return "\n".join(lines)


def _extract_codex_command_numbers(final_output: str) -> list[int]:
    command_match = re.search(
        r"\[Codex command output\]\s*(.*?)(?:\n\s*\[Codex exit_code\]|\Z)",
        final_output or "",
        re.DOTALL,
    )
    if not command_match:
        return []
    command_output = command_match.group(1)
    return [int(match.group(1)) for match in re.finditer(r"(?m)^\s*(\d+)\s*$", command_output)]


def _extract_codex_sequence_numbers(final_output: str) -> list[int]:
    """Extract a complete numeric sequence from Codex's final answer.

    The terminal view can wrap command output mid-number, so a value like 1132
    may appear as two standalone lines: "113" and "2". Prefer Codex's explicit
    final sequence block, where comma separation survives wrapping.
    """
    if not final_output:
        return []

    for match in re.finditer(
        r"(?is)(?:printed\s+collatz\s+integers|collatz\s+sequence|ordered\s+integers).*?`([\s\d,]+)`",
        final_output,
    ):
        block = match.group(1)
        if "," not in block:
            continue
        numbers = [int(n) for n in re.findall(r"\d+", block)]
        if len(numbers) >= 2:
            return numbers

    for match in re.finditer(
        r"(?is)(?:exact\s+full\s+sequence|full\s+sequence).*?(\[[\s\d,]+\])",
        final_output,
    ):
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if len(numbers) >= 2:
            return numbers

    for match in re.finditer(
        r"(?is)(?:exact\s+full\s+sequence|full\s+sequence).*?```(?:\w+)?\s*([\s\d,]+?)```",
        final_output,
    ):
        block = match.group(1)
        if "," not in block:
            continue
        numbers = [int(n) for n in re.findall(r"\d+", block)]
        if len(numbers) >= 2:
            return numbers

    return []


def _is_valid_collatz_sequence(numbers: list[int], total_steps: Optional[int]) -> bool:
    if not numbers or numbers[-1] != 1:
        return False
    if total_steps is not None and len(numbers) - 1 != total_steps:
        return False
    for current, following in zip(numbers, numbers[1:]):
        expected = current // 2 if current % 2 == 0 else current * 3 + 1
        if following != expected:
            return False
    return True


def _synthesize_codex_structured_response(
    final_output: str,
    output_schema: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a verifier response from Codex terminal evidence when schema validation flakes.

    This is intentionally narrow: it only returns a response if every declared
    output field can be populated from explicit terminal evidence.
    """
    if not final_output or not output_schema:
        return None

    paths = sorted(set(_ABSOLUTE_PATH_RE.findall(final_output)))
    script_paths = [path for path in paths if path.endswith(".py")]
    sequence_numbers = _extract_codex_sequence_numbers(final_output)
    numbers = sequence_numbers or _extract_codex_command_numbers(final_output)
    steps_match = re.search(
        r"(?:Total steps|Step count|total step count|steps|[\"']total_steps[\"'])\s*:\s*`?(\d+)",
        final_output,
        re.IGNORECASE,
    )
    total_steps = int(steps_match.group(1)) if steps_match else None
    if total_steps is not None and len(numbers) >= 2 and numbers[-1] == total_steps and numbers[-2] == 1:
        numbers = numbers[:-1]

    structured: Dict[str, Any] = {}
    for field in output_schema.get("fields", []):
        name = field.get("name")
        field_type = str(field.get("type") or "").lower()
        descriptor = " ".join(
            str(field.get(key, "")) for key in ("name", "label", "description") if field.get(key)
        ).lower()
        if not name:
            return None

        if "file_pointer" in field_type:
            path = script_paths[0] if script_paths else (paths[0] if paths else None)
            if not path:
                if field.get("required", True):
                    return None
                continue
            structured[name] = {"path": path}
        elif field_type.startswith("list") and "sequence" in descriptor:
            if not numbers:
                if field.get("required", True):
                    return None
                continue
            if "collatz" in descriptor and not _is_valid_collatz_sequence(numbers, total_steps):
                if field.get("required", True):
                    return None
                continue
            structured[name] = numbers
        elif field_type in {"integer", "int", "number"} and "step" in descriptor:
            if total_steps is None:
                if field.get("required", True):
                    return None
                continue
            structured[name] = total_steps
        elif "path" in descriptor and paths:
            structured[name] = script_paths[0] if script_paths else paths[0]
        else:
            if field.get("required", True):
                return None

    return structured or None


def _remote_path_for_output_file(file_pointer: Dict[str, Any]) -> str:
    filename = str(file_pointer.get("original_filename") or "")
    return f"/home/user/output/{filename}" if filename else "/home/user/output"


def _structured_response_from_output_files(
    output_schema: Optional[Dict[str, Any]],
    output_files: list[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Populate file-pointer fields from files already extracted from /home/user/output."""
    if not output_schema or not output_files:
        return None

    structured: Dict[str, Any] = {}
    file_index = 0
    for field in output_schema.get("fields", []):
        name = field.get("name")
        field_type = str(field.get("type") or "").lower()
        if not name:
            return None

        if field_type == "file_pointer":
            if file_index >= len(output_files):
                return None
            structured[name] = {"path": _remote_path_for_output_file(output_files[file_index])}
            file_index += 1
        elif field_type == "list[file_pointer]":
            structured[name] = [{"path": _remote_path_for_output_file(fp)} for fp in output_files]
        else:
            continue

    return structured or None


def _structured_response_satisfies_declared_schema(
    output_schema: Optional[Dict[str, Any]],
    structured_response: Optional[Dict[str, Any]],
) -> bool:
    """Return true only when every declared field satisfies its output contract."""
    fields = output_schema.get("fields", []) if isinstance(output_schema, dict) else []
    if not fields:
        return True
    if structured_response is None:
        # No structured evidence at all: equivalent to an empty response, so a
        # schema whose fields are all optional is still satisfied.
        structured_response = {}
    if not isinstance(structured_response, dict):
        return False
    for field in fields:
        if not isinstance(field, dict) or not field.get("name"):
            return False
        if not _declared_field_satisfied(field, structured_response):
            return False
    return True


def _declared_field_satisfied(field: Dict[str, Any], structured_response: Dict[str, Any]) -> bool:
    """Return true when one declared field satisfies its output contract."""
    name = field["name"]
    if name not in structured_response:
        return field.get("required", True) is False
    value = structured_response[name]
    if value is None and field.get("nullable") is True:
        return True
    # A present empty value only satisfies a required field when its bounds
    # explicitly allow it; an optional field may be present and empty.
    if (
        field.get("required", True) is not False
        and (value is None or value == "" or value == [] or value == {})
        and not explicitly_allows_empty_value(value, field)
    ):
        return False
    return value_satisfies_output_field_constraints(value, field)


def _disabled_verifier_outcome(
    output_schema: Optional[Dict[str, Any]],
    structured_response: Optional[Dict[str, Any]],
    *,
    attempt_completion: bool,
    exit_code: Optional[int],
) -> tuple[str, list[str]]:
    """Classify a session that no LLM verifier judged.

    Returns ``(outcome, field_names)``. ``failed``: no process/completion
    success signal. ``incomplete``: a declared output file was not extracted.
    ``unverified``: the session succeeded and its files are present, but these
    declared fields have no mechanical evidence; only a verifier could have
    judged them. ``complete``: every declared field is satisfied.
    """
    if not (attempt_completion or exit_code == 0):
        return "failed", []
    if _structured_response_satisfies_declared_schema(output_schema, structured_response):
        return "complete", []
    fields = output_schema.get("fields", []) if isinstance(output_schema, dict) else []
    structured = structured_response if isinstance(structured_response, dict) else {}
    unsatisfied = [
        field
        for field in fields
        if not isinstance(field, dict) or not field.get("name") or not _declared_field_satisfied(field, structured)
    ]
    missing_files = [
        str(field.get("name"))
        for field in unsatisfied
        if not isinstance(field, dict) or not field.get("name") or "file_pointer" in str(field.get("type") or "")
    ]
    if missing_files:
        return "incomplete", missing_files
    return "unverified", [str(field["name"]) for field in unsatisfied]


def _code_event_text_evidence(code_events: list[Dict[str, Any]]) -> list[str]:
    """Collect provider stdout/result strings without depending on terminal truncation."""
    texts: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                texts.append(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            for key in ("content", "text", "output", "result", "stdout", "message"):
                if key in value:
                    collect(value[key])

    for event in code_events:
        if isinstance(event, dict):
            collect(event)
    return texts


def _structured_response_from_code_evidence(
    output_schema: Optional[Dict[str, Any]],
    output_files: list[Dict[str, Any]],
    code_events: list[Dict[str, Any]],
    terminal_output: str,
) -> Optional[Dict[str, Any]]:
    """Materialize supported primitive and file fields from normalized SDK evidence."""
    if not output_schema:
        return None

    structured = _structured_response_from_output_files(output_schema, output_files) or {}
    evidence_texts = _code_event_text_evidence(code_events)
    if terminal_output.strip():
        evidence_texts.append(terminal_output.strip())

    total_steps: Optional[int] = None
    sequence: list[int] = []
    stdout = ""
    for evidence in sorted(evidence_texts, key=len, reverse=True):
        steps_match = re.search(
            r"(?:Total steps|Step count|total step count|steps|[\"']total_steps[\"'])\s*:\s*`?(\d+)",
            evidence,
            re.IGNORECASE,
        )
        candidate_steps = int(steps_match.group(1)) if steps_match else None
        candidate_sequence = [
            int(match.group(1))
            for match in re.finditer(r"(?m)^\s*(\d+)\s*$", evidence)
        ]
        if (
            candidate_steps is not None
            and _is_valid_collatz_sequence(candidate_sequence, candidate_steps)
        ):
            total_steps = candidate_steps
            sequence = candidate_sequence
            stdout = evidence
            break

    for field in output_schema.get("fields", []):
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not name or name in structured:
            continue
        field_type = str(field.get("type") or "").lower()
        descriptor = " ".join(
            str(field.get(key, "")) for key in ("name", "label", "description") if field.get(key)
        ).lower()
        if field_type.startswith("list") and "sequence" in descriptor and sequence:
            # A sequence outside the field's declared bounds would fail the
            # whole response; leave the field unset instead.
            if value_satisfies_output_field_constraints(sequence, field):
                structured[name] = sequence
        elif field_type in {"integer", "int", "number"} and "step" in descriptor:
            if total_steps is not None:
                structured[name] = total_steps
        elif field_type in {"str", "string"} and (
            "stdout" in descriptor or "output" in descriptor
        ):
            if stdout:
                structured[name] = stdout

    return structured or None


# The code agent returns its result to the parent in one JSON file (Andy,
# 2026-10-24); the declared output_schema fixes its structure. Keep in sync with
# CODE_AGENT_HANDOFF_PATH in back-end/src/providers/codeAgentPrompt.ts.
CODE_AGENT_HANDOFF_PATH = "/home/user/.rose/handoff.json"
# The ws-server writes the result JSON Schema here from the task text, so the
# /goal condition can name it without spending Claude's 4000-character goal cap.
CODE_AGENT_HANDOFF_SCHEMA_PATH = "/home/user/.rose/handoff.schema.json"
# Standard handoff fields every code agent writes, next to the declared fields
# (Andy, 2026-11-08): how the run ended plus its git facts. No question field:
# code agents do not ask questions, they assume and report.
HANDOFF_STANDARD_KEY = "rose_handoff"
HANDOFF_COMPLETE = "complete"
HANDOFF_STATUSES = (HANDOFF_COMPLETE, "blocked", "failed")
HANDOFF_STANDARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "How this run ended. Always required.",
    "properties": {
        "status": {
            "type": "string",
            "enum": list(HANDOFF_STATUSES),
            "description": (
                "complete: the task is done and verified. blocked: an external condition you cannot change "
                "stops you after reasonable attempts (name it and its evidence in summary). failed: the task "
                "cannot be done here, shown by what you tried. Never blocked or failed because the work is "
                "hard, slow, uncertain or would benefit from clarification."
            ),
        },
        "summary": {
            "type": "string",
            "description": "What you did, what you verified, and every assumption you made.",
        },
        "branch": {"type": ["string", "null"], "description": "Branch your commits are on, or null."},
        "commit": {"type": ["string", "null"], "description": "Full SHA of your latest commit, or null."},
        "pr_url": {"type": ["string", "null"], "description": "Pull request URL, or null."},
        "merged": {
            "type": ["boolean", "null"],
            "description": "True only if your change is verified merged into its merge target.",
        },
    },
    "required": ["status", "summary", "branch", "commit", "pr_url", "merged"],
    "additionalProperties": False,
}
# Structure used when the parent declared no output fields.
DEFAULT_HANDOFF_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {HANDOFF_STANDARD_KEY: HANDOFF_STANDARD_SCHEMA},
    "required": [HANDOFF_STANDARD_KEY],
    "additionalProperties": False,
}
# The goal's end condition (Andy, 2026-11-08): every code-agent task is a /goal
# whose condition ends with writing the handoff. The ws-server prefixes /goal.
HANDOFF_END_CONDITION = (
    f"End condition: you are done only when {CODE_AGENT_HANDOFF_PATH} holds one JSON object valid against the "
    f"JSON Schema in {CODE_AGENT_HANDOFF_SCHEMA_PATH} (show it in this conversation by printing the written "
    "file and validating it against that schema with a command), with "
    f"{HANDOFF_STANDARD_KEY}.status set to complete (the task is done and verified), or to blocked or failed "
    "only after reasonable attempts show the task cannot be finished here, never because it is hard, slow, "
    "uncertain or would benefit from clarification. Do not ask questions: nobody answers them. Make "
    f"reasonable assumptions, proceed, and list them in {HANDOFF_STANDARD_KEY}.summary."
)
# Close responses without a ``handoff`` key come from sessions outside E2B (a
# Mac folder or the local billing fallback); they keep the session-output result.
_NO_HANDOFF_CONTRACT = object()


def _with_field_descriptions(schema: Dict[str, Any], field: Dict[str, Any]) -> Dict[str, Any]:
    """Carry each declared field description into its JSON Schema, nested fields included."""
    from helpers.llm import _nested_fields_for_object_field

    description = field.get("description")
    if isinstance(description, str) and description.strip() and "description" not in schema:
        schema["description"] = description
    target = schema.get("items") if isinstance(schema.get("items"), dict) else schema
    nested_properties = target.get("properties") if isinstance(target, dict) else None
    if isinstance(nested_properties, dict) and field.get("type") in ("object", "list[object]"):
        for nested in _nested_fields_for_object_field(field):
            if isinstance(nested_properties.get(nested.get("name")), dict):
                _with_field_descriptions(nested_properties[nested["name"]], nested)
    return schema


def handoff_json_schema(output_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """JSON Schema of the handoff file; file fields are ``{"path": ...}`` sandbox paths."""
    from helpers.llm import _json_schema_for_response_field

    remote_schema = transform_schema_for_remote_session(output_schema) or {}
    properties: Dict[str, Any] = {}
    required: list[str] = []
    for field in remote_schema.get("fields", []):
        name = field.get("name") if isinstance(field, dict) else None
        if not isinstance(name, str) or not name:
            continue
        if name == HANDOFF_STANDARD_KEY:
            continue  # reserved for the standard fields
        field_schema, _open_shape = _json_schema_for_response_field(field)
        properties[name] = _with_field_descriptions(field_schema, field)
        if field.get("required", True):
            required.append(name)
    if not properties:
        return DEFAULT_HANDOFF_SCHEMA
    return {
        "type": "object",
        "properties": {HANDOFF_STANDARD_KEY: HANDOFF_STANDARD_SCHEMA, **properties},
        "required": [HANDOFF_STANDARD_KEY, *required],
        "additionalProperties": False,
    }


def handoff_task_text(goal: str, output_schema: Optional[Dict[str, Any]]) -> str:
    """The task sent to the code agent: the goal, its end condition, then its result schema.

    The ws-server writes the schema after the marker to CODE_AGENT_HANDOFF_SCHEMA_PATH
    and starts the run as ``/goal <goal + end condition>``.
    """
    schema = json.dumps(handoff_json_schema(output_schema), sort_keys=True)
    return f"{goal}\n\nResult handoff instructions:\n{HANDOFF_END_CONDITION}\n\nResult JSON Schema for {CODE_AGENT_HANDOFF_PATH}:\n{schema}"


def handoff_standard_fields(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The standard ``rose_handoff`` object of a parsed handoff, when present."""
    standard = result.get(HANDOFF_STANDARD_KEY) if isinstance(result, dict) else None
    return standard if isinstance(standard, dict) else None


def _read_handoff(
    handoff: Optional[Dict[str, Any]], output_schema: Optional[Dict[str, Any]]
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return ``(result, None)`` for a schema-valid handoff, else ``(None, reason)``.

    A ``complete`` handoff must satisfy the whole schema. A blocked or failed one
    reports the run truthfully: only its standard fields are required, and the
    declared fields it does fill must still match their types.
    """
    import jsonschema

    status = handoff.get("status") if isinstance(handoff, dict) else "missing"
    if status == "unreadable":
        return None, f"The handoff file at {CODE_AGENT_HANDOFF_PATH} could not be read: {handoff.get('error')}."
    if status != "present" or not isinstance(handoff.get("content"), str):
        return None, f"The code agent wrote no handoff file at {CODE_AGENT_HANDOFF_PATH}."
    try:
        result = json.loads(handoff["content"])
    except json.JSONDecodeError as exc:
        return None, f"The handoff file at {CODE_AGENT_HANDOFF_PATH} is not valid JSON: {exc}."
    schema = handoff_json_schema(output_schema)
    standard = handoff_standard_fields(result)
    if standard is not None and standard.get("status") != HANDOFF_COMPLETE:
        schema = {**schema, "required": [HANDOFF_STANDARD_KEY]}
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(result), key=lambda error: list(error.absolute_path))
    if errors:
        problems = "; ".join(
            f"{'/'.join(str(part) for part in error.absolute_path) or '(root)'}: {error.message}" for error in errors
        )
        return None, (
            f"The handoff file at {CODE_AGENT_HANDOFF_PATH} does not match the declared output schema: {problems}."
        )
    return result, None


# Event types whose partial events are snapshots a later event of the same item
# replaces. Lifecycle events such as a Codex ``command``/``file_change``
# ``item/started`` (``partial: true``) are not: the start is distinct evidence
# that its completion does not repeat, so judge evidence keeps them.
_SNAPSHOT_EVENT_TYPES = frozenset({"agent_message", "thinking", "tool_use"})


def _stream_item_key(event: Dict[str, Any], *, keep_lifecycle_starts: bool) -> Optional[tuple[Any, str]]:
    """The streamed item an event upserts: its ``item_id``, else a ``tool_use`` call's id.

    Claude streams a tool call's block start as a partial ``tool_use`` (empty
    input) and then the assistant envelope for the same ``tool_use_id``.
    """
    if keep_lifecycle_starts and event.get("type") not in _SNAPSHOT_EVENT_TYPES:
        return None
    item_id = event.get("item_id")
    if isinstance(item_id, str):
        return (event.get("type"), item_id)
    tool_use_id = event.get("tool_use_id")
    if event.get("type") == "tool_use" and isinstance(tool_use_id, str):
        return ("tool_use", tool_use_id)
    return None


def _latest_stream_snapshots(
    code_events: list[Dict[str, Any]], *, keep_lifecycle_starts: bool = False
) -> list[Dict[str, Any]]:
    """Drop partial stream events superseded by a later event for the same item.

    Partial ``agent_message``/``thinking`` events carry the cumulative text, so
    every earlier snapshot is a prefix of the latest one (the same upsert-by-item
    rule ``utils.code_transcript`` applies), and a partial ``tool_use`` is
    replaced by its call's full record. The judge evidence would otherwise grow
    quadratically with the streamed answer and repeat each call. Model-visible
    context also folds a Codex item's start into its completion; judge evidence
    (``keep_lifecycle_starts``) keeps those starts (``_SNAPSHOT_EVENT_TYPES``).
    """
    latest: Dict[tuple[Any, str], int] = {}
    for index, event in enumerate(code_events):
        key = _stream_item_key(event, keep_lifecycle_starts=keep_lifecycle_starts)
        if key is not None:
            latest[key] = index
    return [
        event
        for index, event in enumerate(code_events)
        if not (
            event.get("partial") is True
            and (key := _stream_item_key(event, keep_lifecycle_starts=keep_lifecycle_starts)) is not None
            and latest[key] != index
        )
    ]


# One judge-evidence row holds at most this many events (the bound the single
# row used to cut the session to); longer sessions span more rows.
_JUDGE_EVIDENCE_PAGE_EVENTS = 200


def _build_code_event_context(
    *,
    code_events: list[Dict[str, Any]],
    output_files: list[Dict[str, Any]],
    action_summary: list[str],
) -> str:
    """Model-visible SDK event context: the latest snapshots, last 200 events."""
    return _code_event_context_json(
        events=_latest_stream_snapshots(code_events)[-200:],
        output_files=output_files,
        action_summary=action_summary,
    )


def _code_event_context_json(
    *,
    events: list[Dict[str, Any]],
    output_files: list[Dict[str, Any]],
    action_summary: list[str],
) -> str:
    evidence = {
        "code_agent_events": events,
        "extracted_output_files": [
            {
                "original_filename": fp.get("original_filename"),
                "mime_type": fp.get("mime_type"),
                "size_bytes": fp.get("size_bytes"),
                "path": _remote_path_for_output_file(fp),
            }
            for fp in output_files
        ],
        "execution_summary": action_summary[-20:],
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2)


async def _persist_code_agent_judge_evidence(
    *,
    agent_id: str,
    task_id: str,
    code_provider: str,
    code_events: list[Dict[str, Any]] | None,
    output_files: list[Dict[str, Any]] | None,
    action_summary: list[str],
) -> None:
    """Persist non-orchestration code-agent evidence for harness judges.

    Orchestration chips use ``toolName=spawn_claude_code_agent`` and therefore
    contribute 0 leaf steps to ``_agent_class_step_counts``. When the LLM
    verifier is skipped, nothing else records coding-agent work, so declared
    judges see every code agent at step_count=0 and fail "coding-agent fetch"
    criteria. These TOOL messages use distinct non-orchestration tool names +
    ``agentType=code`` + ``task_id``/``subAgentId`` so they count as grounded
    code-agent steps and remain readable in the trajectory.
    """
    # The evidence is the latest snapshot of every streamed item, complete:
    # steps_taken counts exactly these events, and they are stored in bounded
    # pages (``sdk_event_range``) rather than cut to a tail, so a judge can
    # still detect a missing page as truncated evidence.
    events = _latest_stream_snapshots(
        [event for event in code_events or [] if isinstance(event, dict)], keep_lifecycle_starts=True
    )
    files = output_files or []
    summary = [str(item).strip() for item in (action_summary or []) if str(item).strip()]
    if not events and not files and not summary:
        return

    # Prefer explicit event/action cardinality so judges see non-zero steps even
    # when only a single evidence chip is present. Do not force steps_taken=1
    # from orchestration-only summaries with zero SDK events/files — those are
    # lifecycle breadcrumbs, not grounded coding-agent steps.
    steps_taken = max(len(events), 1 if files else 0)
    if steps_taken == 0:
        return

    for start in range(0, max(len(events), 1), _JUDGE_EVIDENCE_PAGE_EVENTS):
        page = events[start : start + _JUDGE_EVIDENCE_PAGE_EVENTS]
        event_context = _code_event_context_json(
            events=page,
            output_files=files,
            action_summary=summary,
        )
        await save_claude_code_message(
            agent_id=agent_id,
            title="🧾 Code-agent SDK events",
            task_id=task_id,
            message=f"Code-agent SDK event JSON:\n\n{event_context}",
            status=MessageStatus.FINISHED,
            action_type="system",
            code_provider=code_provider,
            tool_name="code_agent_sdk_events",
            agent_type="code",
            steps_taken=steps_taken,
            sdk_event_range=(start, start + len(page)),
        )
    await save_claude_code_message(
        agent_id=agent_id,
        title="🧾 Code-agent execution summary",
        task_id=task_id,
        message="Execution Summary:\n\n" + "\n".join(summary[-20:]),
        status=MessageStatus.FINISHED,
        action_type="system",
        code_provider=code_provider,
        tool_name="claude_code_history",
        agent_type="code",
        steps_taken=steps_taken,
    )


async def _handoff_result(
    *,
    agent_id: str,
    task_id: str,
    output_schema: Optional[Dict[str, Any]],
    handoff: Optional[Dict[str, Any]],
    attempt_completion: bool,
    exit_code: Optional[int],
    code_provider: str,
    code_events: list[Dict[str, Any]],
    output_files: list[Dict[str, Any]],
    action_summary: list[str],
) -> tuple[float, str, Optional[Dict[str, Any]], int, int]:
    """Verifier-off result: the code agent's handoff file, validated against its schema.

    The handoff is the result the parent sees. Terminal output and SDK events are
    evidence for judges only and never enter the tool result.

    The handoff JSON Schema is this result's one contract: the code agent was
    given it and ``_read_handoff`` enforced it, explicit bounds included. A valid
    handoff from a successful session is therefore complete. It is not re-judged
    by the extraction predicate, whose required-means-non-empty rule is not in
    that schema and rejected honest empty answers such as ``open_issues: []``.
    """
    result, handoff_error = _read_handoff(handoff, output_schema)
    standard = handoff_standard_fields(result)
    reported_status = standard.get("status") if standard is not None else None
    has_fields = bool(isinstance(output_schema, dict) and output_schema.get("fields"))
    # The parent's declared fields; the standard fields stay in the reasoning.
    structured = (
        {key: value for key, value in result.items() if key != HANDOFF_STANDARD_KEY}
        if has_fields and isinstance(result, dict)
        else None
    )
    success_signal = attempt_completion or exit_code == 0
    # A schema-valid handoff from a successful session is complete (H-21) unless the
    # code agent itself reported a non-complete standard status.
    complete = result is not None and success_signal and reported_status in (None, HANDOFF_COMPLETE)
    if complete:
        outcome = "complete"
    elif not success_signal:
        outcome = "failed"
    elif result is None:
        outcome = "invalid_handoff"
    else:
        outcome = "reported_incomplete"
    score = 1.0 if complete else 0.5
    header = (
        f"Code-agent verifier off (provider={code_provider}, exit_code={exit_code}, "
        f"artifacts={len(output_files)})."
    )
    if not success_signal:
        header += " The session ended without a success signal."
    if reported_status not in (None, HANDOFF_COMPLETE):
        header += f" The code agent reported {HANDOFF_STANDARD_KEY}.status={reported_status}."
    if result is not None:
        reasoning = (
            f"{header} Handoff from {CODE_AGENT_HANDOFF_PATH}, validated against the declared output schema:\n\n"
            f"{json.dumps(result, ensure_ascii=False, indent=2)}"
        )
    else:
        reasoning = f"{header} {handoff_error}"
    logger.info(
        "[ClaudeCode] Verifier skipped; returning handoff",
        agent_id=agent_id,
        task_id=task_id,
        code_provider=code_provider,
        exit_code=exit_code,
        handoff_status=handoff.get("status") if isinstance(handoff, dict) else None,
        handoff_valid=result is not None,
        handoff_reported_status=reported_status,
        outcome=outcome,
        score=score,
    )
    if complete:
        try:
            await _persist_code_agent_judge_evidence(
                agent_id=agent_id,
                task_id=task_id,
                code_provider=code_provider,
                code_events=code_events,
                output_files=output_files,
                action_summary=action_summary,
            )
        except Exception as evidence_err:
            logger.warning(
                "[ClaudeCode] judge evidence persist failed; continuing verifier-disabled completion",
                agent_id=agent_id,
                task_id=task_id,
                error=str(evidence_err)[:200],
            )
    return score, reasoning, structured, 0, 0


async def _run_verifier(
    *,
    agent_id: str,
    task_id: str,
    goal: str,
    model: str,
    terminal: TerminalFilter,
    action_summary: list[str],
    output_schema: Dict[str, Any],
    claude_token: Optional[str],
    gpt_token: Optional[str] = None,
    gpt_account_id: Optional[str] = None,
    gpt_token_source: Optional[str] = None,
    gpt_token_pool_id: Optional[str] = None,
    gpt_token_user_id: Optional[str] = None,
    fast_mode_cache_enabled: bool = False,
    fast_mode_goal: Optional[str] = None,
    fast_mode_step: int = 0,
    attempt_completion: bool = False,
    exit_code: Optional[int] = None,
    code_provider: str = "claude",
    code_events: Optional[list[Dict[str, Any]]] = None,
    output_files: Optional[list[Dict[str, Any]]] = None,
    handoff: Any = _NO_HANDOFF_CONTRACT,
) -> tuple[float, str, Optional[Dict[str, Any]], int, int]:
    """Run the verifier to confirm completion.

    Returns (score, reasoning, structured_response, input_tokens, output_tokens).

    The dual-agent LLM verifier is off by default (Claude Code and Codex); it
    runs only when ``ROSE_ENABLE_CODE_AGENT_VERIFIER`` opts in. Otherwise the
    session result is returned as-is so the parent tool call finishes when the
    code agent ends.
    """
    from helpers.llm import check_progress
    from loop.activity_helper import call_temporal_activity_from_activity
    from utils.prompts import build_verifier_prompt

    verifier_call_id = str(uuid_module.uuid4())
    input_tokens = 0
    output_tokens = 0
    extracted_output_files = output_files or []
    terminal_output = (terminal.get_display() or "").strip()
    evidence_structured = _structured_response_from_code_evidence(
        output_schema,
        extracted_output_files,
        code_events or [],
        terminal_output,
    )
    schema_complete = _structured_response_satisfies_declared_schema(
        output_schema, evidence_structured
    )

    if not code_agent_verifier_enabled() and handoff is not _NO_HANDOFF_CONTRACT:
        return await _handoff_result(
            agent_id=agent_id,
            task_id=task_id,
            output_schema=output_schema,
            handoff=handoff,
            attempt_completion=attempt_completion,
            exit_code=exit_code,
            code_provider=code_provider,
            code_events=code_events or [],
            output_files=extracted_output_files,
            action_summary=action_summary,
        )

    if not code_agent_verifier_enabled():
        # Skip the dual-agent LLM verifier, but do not claim success without
        # both extracted /home/user/output artifacts and a process/completion
        # success signal. A crashed process can leave stale partial files
        # behind. Surface the same SDK-event JSON the LLM verifier would have
        # read; a one-line skip note with no Bash/Write evidence made the parent
        # treat a completed Collatz session as empty and re-spawn until the
        # smoke cap (claude-code-agent 04977ce8, 2026-09-04).
        # The schema is the contract: file-only, primitive-only, and mixed
        # schemas all require every declared field before the disabled verifier
        # may report success or persist successful judge evidence.
        # A declared field with no mechanical evidence is unverified, not
        # missing: only output files can be checked without the verifier.
        has_artifacts = schema_complete
        has_success_signal = attempt_completion or exit_code == 0
        score = 1.0 if has_artifacts and has_success_signal else 0.5
        outcome, outcome_fields = _disabled_verifier_outcome(
            output_schema,
            evidence_structured,
            attempt_completion=attempt_completion,
            exit_code=exit_code,
        )
        outcome_note = {
            "complete": "",
            "failed": " The session ended without a success signal." if not has_artifacts else "",
            "incomplete": " Required output artifacts were not completely extracted - incomplete.",
            "unverified": (
                " The session completed, but these declared fields were not verified: "
                f"{', '.join(outcome_fields)}. Read them from the session output below."
            ),
        }[outcome]
        v_reasoning = (
            "Code-agent verifier off. "
            f"Returning session result as-is (provider={code_provider}, exit_code={exit_code}"
            f", artifacts={len(extracted_output_files)})." + outcome_note
        )
        if terminal_output:
            v_reasoning = f"{v_reasoning}\n\nTerminal output:\n\n{terminal_output}"
        event_context = _build_code_event_context(
            code_events=code_events or [],
            output_files=extracted_output_files,
            action_summary=action_summary,
        )
        if event_context:
            v_reasoning = f"{v_reasoning}\n\nCode-agent SDK event JSON:\n\n{event_context}"
        logger.info(
            "[ClaudeCode] Verifier skipped (off by default)",
            agent_id=agent_id,
            task_id=task_id,
            code_provider=code_provider,
            exit_code=exit_code,
            has_artifacts=has_artifacts,
            has_success_signal=has_success_signal,
            has_structured_response=bool(evidence_structured),
            score=score,
        )
        # Persist SDK/history evidence so harness step counts + declared judges
        # still see grounded successful work when the LLM verifier is off. A
        # failed process may leave files behind, but those are not completion
        # evidence and must not inflate the judge's successful step count.
        if has_success_signal and has_artifacts:
            try:
                await _persist_code_agent_judge_evidence(
                    agent_id=agent_id,
                    task_id=task_id,
                    code_provider=code_provider,
                    code_events=code_events,
                    output_files=extracted_output_files,
                    action_summary=action_summary,
                )
            except Exception as evidence_err:
                # Evidence write failures must not block verifier-disabled
                # completion when artifacts / exit_code already prove success.
                logger.warning(
                    "[ClaudeCode] judge evidence persist failed; continuing "
                    "verifier-disabled completion",
                    agent_id=agent_id,
                    task_id=task_id,
                    error=str(evidence_err)[:200],
                )
        return score, v_reasoning, evidence_structured, input_tokens, output_tokens

    try:
        await save_claude_code_message(
            agent_id=agent_id,
            title="⚖️ Verifying completion",
            task_id=task_id,
            message="",
            status=MessageStatus.IN_PROGRESS,
            action_type="system",
            tool_call_id=verifier_call_id,
            code_provider=code_provider,
        )

        final_output = terminal.get_display()
        normalized_events = code_events or []
        event_context = _build_code_event_context(
            code_events=normalized_events,
            output_files=extracted_output_files,
            action_summary=action_summary,
        )
        verifier_terminal_output = event_context[-12000:]

        output_files_info = ""
        if "output" in final_output.lower() or "/home/user/output" in final_output:
            output_files_info = (
                "\n\nNote: Check if any files were created in /home/user/output/ based on terminal output."
            )

        verifier_context = [
            {"role": "USER", "text": f"Goal: {goal}"},
            {
                "role": "TOOL",
                "text": f"Code-agent SDK event JSON:\n\n{verifier_terminal_output}",
                "metadata": {"toolName": "code_agent_sdk_events", "toolStatus": "FINISHED"},
            },
            {
                "role": "TOOL",
                "text": "Execution Summary:\n\n" + "\n".join(action_summary[-20:]) + output_files_info,
                "metadata": {"toolName": "claude_code_history", "toolStatus": "FINISHED"},
            },
        ]

        remote_session_schema = transform_schema_for_remote_session(output_schema)
        verifier_prompt = build_verifier_prompt(response_schema=remote_session_schema)

        logger.info("[ClaudeCode] Running verifier", agent_id=agent_id, task_id=task_id, model=model)
        # B2: `check_progress` forces rose-1-ultra + claude_token internally and
        # ignores the `model` arg. We still pass it so the activity's positional
        # signature matches (matches the browser-agent call site at
        # spawn_browser_agent_v3/handler.py).
        # C7: dead `phase` positional arg has been removed from check_progress;
        # callers no longer pass it.
        # C14: `verifier_context` is built locally from this Claude Code
        # session's terminal output and action summary — nothing from parent
        # agents reaches it.
        progress_result = await call_temporal_activity_from_activity(
            check_progress,
            args=[
                verifier_context,
                verifier_prompt,
                model,  # ignored — verifier always uses rose-1-ultra
                remote_session_schema,
                claude_token,
                "verifier_claude_code",
                agent_id,
                None,
                None,
                None,
                gpt_token,
                gpt_account_id,
                gpt_token_source,
                gpt_token_pool_id,
                gpt_token_user_id,
                None,
                None,
                None,
                fast_mode_cache_enabled,
                fast_mode_goal or goal,
                fast_mode_step,
            ],
            start_to_close_timeout=timedelta(seconds=VERIFIER_ACTIVITY_TIMEOUT_SECONDS),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        v_usage = progress_result.get("usage", {})
        input_tokens = v_usage.get("prompt_tokens", 0)
        output_tokens = v_usage.get("completion_tokens", 0)

        score = progress_result.get("score", 0.0)
        v_reasoning = progress_result.get("reasoning", "")
        structured_response = progress_result.get("structured_response")

        await save_claude_code_message(
            agent_id=agent_id,
            title=f"⚖️ Progress: {score * 100:.0f}%",
            task_id=task_id,
            message=v_reasoning,
            status=MessageStatus.FINISHED,
            action_type="system",
            tool_call_id=verifier_call_id,
            code_provider=code_provider,
        )

        logger.info(
            "[ClaudeCode] Verifier result",
            agent_id=agent_id,
            task_id=task_id,
            score=score,
            reasoning=v_reasoning[:200],
        )

        return score, v_reasoning, structured_response, input_tokens, output_tokens

    except Exception as verifier_error:
        logger.error("[ClaudeCode] Verifier failed", agent_id=agent_id, task_id=task_id, error=str(verifier_error))
        await save_claude_code_message(
            agent_id=agent_id,
            title="⚠️ Verification failed",
            task_id=task_id,
            message=f"Error: {str(verifier_error)[:200]}",
            status=MessageStatus.FINISHED,
            action_type="error",
            tool_call_id=verifier_call_id,
            code_provider=code_provider,
        )
        return 0.5, "Verification failed - assuming partial completion", None, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# File extraction helpers
# ---------------------------------------------------------------------------


def extract_additional_paths(
    structured_response: Optional[Dict[str, Any]],
    output_schema: Optional[Dict[str, Any]],
) -> list[str]:
    """Extract file paths from verifier's structured response to download specific files."""
    additional_paths: list[str] = []
    if not structured_response or not output_schema:
        return additional_paths

    for field in output_schema.get("fields", []):
        field_name = field.get("name")
        field_type = field.get("type")
        if not field_name or field_name not in structured_response:
            continue
        value = structured_response[field_name]
        if field_type == "file_pointer" and isinstance(value, dict):
            path = value.get("path")
            if path and path.startswith("/"):
                additional_paths.append(path)
        elif field_type == "list[file_pointer]" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    path = item.get("path")
                    if path and path.startswith("/"):
                        additional_paths.append(path)

    return additional_paths


def build_usage_items(
    total_input_tokens: int,
    total_output_tokens: int,
    model: str,
    model_pricing: Dict[str, Any],
    execution_time_ms: float,
    claude_code_usage: Optional[Dict[str, Any]] = None,
    code_agent_meta: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Build standardized usage items for cost tracking.

    ``code_agent_meta`` prices the sandbox: its ``sandbox_size`` vCPUs (C16),
    and nothing for a run on the user's own Mac (``target`` "local").
    """
    usage_items: list[Dict[str, Any]] = []

    if total_input_tokens > 0 or total_output_tokens > 0:
        verifier_cost = round(
            (total_input_tokens / 1000) * model_pricing["input_cents_per_1k"]
            + (total_output_tokens / 1000) * model_pricing["output_cents_per_1k"],
            4,
        )
        usage_items.append(
            {
                "type": "llm",
                "cost_cents": verifier_cost,
                "provider": f"Verifier ({model})",
                "model": model,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }
        )

    if claude_code_usage:
        input_tokens = int(claude_code_usage.get("inputTokens", 0) or 0)
        output_tokens = int(claude_code_usage.get("outputTokens", 0) or 0)
        cache_read = int(claude_code_usage.get("cacheReadInputTokens", 0) or 0)
        cache_create = int(claude_code_usage.get("cacheCreationInputTokens", 0) or 0)
        if input_tokens > 0 or output_tokens > 0 or cache_read > 0 or cache_create > 0:
            usage_items.append(
                {
                    "type": "llm",
                    "cost_cents": 0,
                    "provider": "Claude Code CLI (user plan)",
                    "model": "claude-code",
                    "number_of_tokens": input_tokens + output_tokens,
                    "tokens_in": input_tokens,
                    "tokens_out": output_tokens,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_create,
                }
            )

    meta = code_agent_meta or {}
    if meta.get("target") == "local":
        return usage_items
    size = meta.get("sandbox_size") if isinstance(meta.get("sandbox_size"), dict) else {}
    vcpus = size.get("cpu")
    if not isinstance(vcpus, (int, float)) or isinstance(vcpus, bool) or vcpus <= 0:
        vcpus = E2B_DEFAULT_SANDBOX_VCPUS
    e2b_cost_cents = round((execution_time_ms / 1000) * E2B_COST_CENTS_PER_VCPU_SECOND * vcpus, 4)
    usage_items.append(
        {
            "type": "service",
            "cost_cents": e2b_cost_cents,
            "provider": "E2B Desktop Sandbox",
        }
    )

    return usage_items


def _sandbox_usage_items(code_agent_meta: Dict[str, Any], started_at: float) -> list[Dict[str, Any]]:
    """Usage of a run that ends without a verifier call: its sandbox time only."""
    return build_usage_items(0, 0, "", {}, (safe_time() - started_at) * 1000, code_agent_meta=code_agent_meta)


# ---------------------------------------------------------------------------
# Persistent code-session facts (C1-C5, C13)
# ---------------------------------------------------------------------------


def _spawn_code_session_facts(session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Facts from the create response: session id, branch, merge target, resume source."""
    if not session_data.get("code_session_id"):
        return None
    return {
        "code_session_id": session_data.get("code_session_id"),
        "branch": session_data.get("branch"),
        "base_branch": session_data.get("base_branch"),
        "merge_target": session_data.get("merge_target"),
        "parent_code_session_id": session_data.get("parent_code_session_id"),
        "resumed_from": session_data.get("resumed_from"),
    }


def _merge_close_code_session_facts(
    facts: Optional[Dict[str, Any]], close_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Add the close response's checkpoint and merge facts to the spawn facts."""
    closed = close_data.get("code_session")
    if not isinstance(closed, dict):
        return facts
    merged = dict(facts or {})
    merge = closed.get("merge") if isinstance(closed.get("merge"), dict) else {}
    repo = closed.get("repo") if isinstance(closed.get("repo"), dict) else {}
    merged.update(
        {
            "code_session_id": closed.get("code_session_id") or merged.get("code_session_id"),
            "state": closed.get("state"),
            "checkpoint_seq": closed.get("checkpoint_seq"),
            "branch": closed.get("branch") or merged.get("branch"),
            "merge_target": closed.get("merge_target") or merged.get("merge_target"),
            "head": merge.get("head") or repo.get("head"),
            "local_commits": merge.get("local_commits", repo.get("localCommits")),
            "pushed": merge.get("pushed", repo.get("pushed")),
            "dirty_files": repo.get("dirtyFiles"),
            "files": repo.get("files") or [],
            "pr": merge.get("pr"),
            # True only when GitHub confirms the PR merged and the target contains head (C4).
            "merged": merge.get("merged") is True,
            "merge_target_contains_head": merge.get("merge_target_contains_head"),
            # How GitHub answered; `merged` is only a confirmed fact when it did.
            "merge_source": merge.get("source"),
            "rate_limited_until": merge.get("rate_limited_until"),
        }
    )
    return merged


def iso_utc(epoch_s: Optional[float]) -> Optional[str]:
    """``2026-10-20T16:50:00Z`` for an epoch (the ``code_agent`` timestamp format)."""
    if epoch_s is None:
        return None
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def build_code_agent_metadata(
    *,
    task_id: Optional[str],
    provider: str,
    state: str,
    started_at: Optional[float],
    ended_at: Optional[float] = None,
    facts: Optional[Dict[str, Any]] = None,
    failure_message: Optional[str] = None,
    target_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The ``code_agent`` message metadata block the UI thread renders.

    ``state`` is spinning_up | live | completed | failed | paused (paused =
    idle, user-paused, waiting on its parent or stopped by a usage limit;
    resumable, never failed). ``summary.merged_into`` is set only when the
    merge was confirmed (C4). ``failure_message`` is the human-readable reason
    for the user when state is failed or paused (e.g. the usage-limit reset).
    ``target_meta`` carries the run target ('cloud' | 'local'), device_name and
    folder (owned by the local-target lane); cloud when omitted. It also carries
    ``preview_url``: the session's live dev-server preview (Rose dev accounts, A1).
    """
    facts = facts or {}
    block: Dict[str, Any] = {
        "task_id": task_id,
        "provider": "codex" if provider == "codex" else "claude",
        "target": "cloud",
        **(target_meta or {}),
        "branch": facts.get("branch"),
        # Where this branch merges back: the parent's branch for sub-agents,
        # the project/repo default branch for new agents (C13).
        "base_branch": facts.get("merge_target") or facts.get("base_branch"),
        "state": state,
        "started_at": iso_utc(started_at),
    }
    if ended_at is not None:
        block["ended_at"] = iso_utc(ended_at)
    if state in ("failed", "paused") and failure_message:
        block["failure_message"] = failure_message
    if facts.get("head"):
        summary: Dict[str, Any] = {
            "commit": facts.get("head"),
            "branch": facts.get("branch"),
            "files": facts.get("files") or [],
        }
        if facts.get("merged") is True and facts.get("merge_target"):
            summary["merged_into"] = facts.get("merge_target")
        block["summary"] = summary
    return block


def subscription_usage_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The account usage a Claude Code ``rate_limit_event`` reports, for EVERY status.

    ``allowed`` / ``allowed_warning`` carry the window's utilization and reset
    just like ``rejected`` does; all of them feed the compute screen's usage bar
    (only ``rejected`` is also a usage limit, see ``usage_limit_from_event``).
    """
    if event.get("type") != "cli_message" or event.get("raw_type") != "rate_limit_event":
        return None
    message = event.get("cli_message") if isinstance(event.get("cli_message"), dict) else {}
    info = message.get("rate_limit_info")
    return usage_from_rate_limit_event(info) if isinstance(info, dict) else None


# Per task, one background drain sends the LATEST usage snapshot: a status change
# at once, a same-status update once the interval since the last delivery ends
# (deferred, never dropped), and a snapshot counts as delivered only when the
# backend accepted it; a failed send is retried after the interval, bounded.
_CODE_USAGE_REPORT_MIN_INTERVAL_S = 15.0
_CODE_USAGE_REPORT_MAX_FAILURES = 3
_CODE_USAGE_MAX_TRACKED_TASKS = 4096
_code_usage_reporters: Dict[str, Dict[str, Any]] = {}
_code_usage_tasks: set["asyncio.Task[Any]"] = set()


def _code_usage_reporter(task_id: str) -> Dict[str, Any]:
    state = _code_usage_reporters.get(task_id)
    if state is None:
        state = {"pending": None, "status": None, "next_at": 0.0, "retry_at": 0.0, "failures": 0, "draining": False}
        _code_usage_reporters[task_id] = state
        for stale in list(_code_usage_reporters)[: max(0, len(_code_usage_reporters) - _CODE_USAGE_MAX_TRACKED_TASKS)]:
            if not _code_usage_reporters[stale]["draining"]:
                _code_usage_reporters.pop(stale)
    return state


def _merge_code_usage(newer: Dict[str, Any], older: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One CLI rate_limit_event carries one window: keep the latest observation of each window."""
    if not older:
        return newer
    kept = {
        key: older[key] for key in ("fiveHour", "sevenDay") if newer.get(key) is None and older.get(key) is not None
    }
    return {**newer, **kept}


def report_code_session_usage_soon(task_id: str, usage: Dict[str, Any]) -> None:
    """Fire-and-forget: hand the task's latest usage to its background drain; never blocks or raises."""
    with contextlib.suppress(Exception):
        state = _code_usage_reporter(task_id)
        state["pending"] = _merge_code_usage(usage, state["pending"])
        if state["draining"]:
            return
        state["draining"] = True
        task = asyncio.get_running_loop().create_task(_drain_code_session_usage(task_id, state))
        _code_usage_tasks.add(task)
        task.add_done_callback(_code_usage_tasks.discard)


async def _drain_code_session_usage(task_id: str, state: Dict[str, Any]) -> None:
    try:
        while state["pending"] is not None:
            usage = state["pending"]
            due = max(state["retry_at"], state["next_at"] if usage.get("status") == state["status"] else 0.0)
            wait = due - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
                continue  # send whatever is latest once the interval ends
            state["pending"] = None
            delivered = False
            with contextlib.suppress(Exception):
                delivered = await asyncio.wait_for(report_code_session_subscription_usage(task_id, usage), timeout=5.0)
            if delivered:
                state.update(
                    status=usage.get("status"),
                    next_at=time.monotonic() + _CODE_USAGE_REPORT_MIN_INTERVAL_S,
                    retry_at=0.0,
                    failures=0,
                )
                continue
            state["failures"] += 1
            state["retry_at"] = time.monotonic() + _CODE_USAGE_REPORT_MIN_INTERVAL_S
            if state["pending"] is not None:
                state["pending"] = _merge_code_usage(state["pending"], usage)
            elif state["failures"] < _CODE_USAGE_REPORT_MAX_FAILURES:
                state["pending"] = usage
    finally:
        state["draining"] = False


def usage_limit_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A provider usage limit from structured CLI events only (C15; no prose matching).

    Claude Code: ``rate_limit_event`` whose ``rate_limit_info.status`` is
    ``rejected``. Codex: an error event carrying the ws-server's ``usage_limit``
    (Codex's own rate-limit snapshot reports the window exhausted).
    """
    if event.get("type") == "cli_message" and event.get("raw_type") == "rate_limit_event":
        message = event.get("cli_message") if isinstance(event.get("cli_message"), dict) else {}
        info = message.get("rate_limit_info") if isinstance(message.get("rate_limit_info"), dict) else {}
        if info.get("status") == "rejected":
            return {"provider": "claude", "resets_at": info.get("resetsAt"), "limit_type": info.get("rateLimitType")}
    if event.get("type") == "cli_message" and event.get("raw_type") == "assistant.api_error":
        # Out of credits: the SDK's structured ``billing_error`` on an assistant message.
        message = event.get("cli_message") if isinstance(event.get("cli_message"), dict) else {}
        if message.get("error") == "billing_error":
            return {"provider": "claude", "resets_at": None, "limit_type": "billing"}
    if event.get("type") == "error" and isinstance(event.get("usage_limit"), dict):
        usage = event["usage_limit"]
        return {"provider": "codex", "resets_at": usage.get("resets_at"), "limit_type": usage.get("reached_type")}
    return None


def login_unusable_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A login the provider rejected, from structured CLI events only (C15; no prose matching).

    Claude Code: the SDK's ``authentication_failed`` on an assistant message (a
    401/403, e.g. the sandbox's access token expired or was revoked). Codex: the
    ws-server's ``login_unusable`` error (refresh failed or revoked), reported
    only when the sandbox has no Azure switch armed (subscriptions first). The
    session pauses like a usage limit and the parent's retry re-leases a
    credential and resumes the conversation. ``credential`` is the ws-server's
    fingerprint of the token it held (never the token).
    """
    unusable: Optional[Dict[str, Any]] = None
    credential = event.get("credential")
    if event.get("type") == "cli_message" and event.get("raw_type") == "assistant.api_error":
        message = event.get("cli_message") if isinstance(event.get("cli_message"), dict) else {}
        if message.get("error") == "authentication_failed":
            unusable = {"provider": "claude", "reason": "authentication_failed"}
    elif event.get("type") == "error" and isinstance(event.get("login_unusable"), dict):
        credential = event["login_unusable"].get("credential")
        unusable = {"provider": "codex", "reason": event["login_unusable"].get("reason") or "oauth_refresh_invalid"}
    if unusable is not None and isinstance(credential, str) and credential:
        unusable["credential"] = credential[:64]
    return unusable


def _provider_label(code_provider: str) -> str:
    return "Codex" if code_provider == "codex" else "Claude Code"


def local_workspace_place(target_meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """Where a local (Mac) run's work lives, e.g. "~/Developer/rose on Andy's MacBook"."""
    if not target_meta or target_meta.get("target") != "local":
        return None
    folder = target_meta.get("folder") or "the shared folder"
    return f"{folder} on {target_meta.get('device_name') or 'the Mac'}"


def _account_line(account: Dict[str, Any]) -> str:
    """``- Claude Code me@example.com: limited until <reset> (this run)``."""
    if account.get("status") == "limited":
        state = f"limited until {account.get('limited_until')}"
    elif account.get("status") == "reconnect":
        state = "disconnected (the user must reconnect it)"
    else:
        state = "available"
    this_run = " (this run)" if account.get("this_run") else ""
    return f"- {_provider_label(str(account.get('provider')))} {account.get('label')}: {state}{this_run}"


def _usage_accounts_block(accounts: Optional[List[Dict[str, Any]]]) -> str:
    """The user's connected accounts as stored (the backend's usage-limit answer)."""
    if accounts is None:
        return ""
    if not accounts:
        return "\nConnected accounts: none."
    return "\nConnected accounts:\n" + "\n".join(_account_line(account) for account in accounts)


def _shared_path_line(path: Dict[str, Any]) -> str:
    """``- shared Codex subscriptions: 2 of 3 available`` / ``- funded Anthropic API key: available``."""
    status = path.get("status")
    until = path.get("limited_until")
    if path.get("kind") == "pool":
        total = path.get("total_count") or 0
        if status == "unconfigured":
            state = "none connected"
        elif status == "available":
            state = f"{path.get('available_count') or 0} of {total} available"
        else:
            state = f"all {total} limited" + (f" until {until}" if until else "")
    elif status == "unconfigured":
        state = "not configured"
    elif status == "available":
        state = "available"
    else:
        state = "limited" + (f" until {until}" if until else "")
    return f"- {path.get('label')}: {state}"


def _shared_paths_block(paths: Optional[List[Dict[str, Any]]]) -> str:
    """Shared subscription pools and funded keys (stored state only; no usage estimates)."""
    if not paths:
        return ""
    return "\nShared credential paths:\n" + "\n".join(_shared_path_line(path) for path in paths)


def _lease_name(next_lease: Dict[str, Any]) -> str:
    """An own account reads ``Codex me@x``; a shared path's label already names its provider."""
    provider = _provider_label(str(next_lease.get("provider")))
    if next_lease.get("kind") in ("pool", "funded"):
        return str(next_lease.get("label"))
    label = next_lease.get("label")
    return f"{provider} {label}" if label else f"a shared {provider} account"


def _respawn_clause(next_lease: Optional[Dict[str, Any]], *, local: bool, from_provider: Optional[str] = None) -> str:
    """What a retry leases and how it continues: native resume, provider handoff, or nothing yet."""
    if local:
        then = "continues from the files in that folder"
        if not isinstance(next_lease, dict) or not next_lease.get("provider"):
            return f"the respawn uses one of your own logins that is not rate-limited and {then}."
        provider = _provider_label(str(next_lease["provider"]))
        label = next_lease.get("label")
        if next_lease.get("available_now"):
            now = f"now {provider} {label}" if label else f"now a shared {provider} account"
            return f"the respawn uses one of your own logins that is not rate-limited ({now}) and {then}."
        return (
            f"no login is free until {next_lease.get('available_at')} "
            f"(the earliest reset: {provider} {label}); a respawn after that uses it and {then}."
        )
    if not isinstance(next_lease, dict) or not next_lease.get("provider"):
        return "the retry leases an account that is not rate-limited and resumes the paused session."
    provider = _provider_label(str(next_lease["provider"]))
    if not next_lease.get("available_now"):
        return (
            f"no credential is free until {next_lease.get('available_at')} "
            f"(the earliest reset: {_lease_name(next_lease)}); a retry before that stays paused and "
            "starts nothing, so wait with wait_for(until_timestamp=<that time>) first."
        )
    label = next_lease.get("label")
    funded = next_lease.get("kind") == "funded"
    if funded:
        now = f"now the {label}" if label else f"now the funded {provider} key"
    else:
        now = f"now {provider} {label}" if label else f"now a shared {provider} account"
    if from_provider is None or next_lease["provider"] == from_provider:
        if funded:
            return (
                f"no {provider} subscription is left, so the retry uses the {label or 'funded key'}, "
                f"resumes the paused sandbox and continues the same {provider} conversation."
            )
        return (
            f"the retry leases an account that is not rate-limited ({now}), resumes the paused "
            f"sandbox and continues the same {provider} conversation."
        )
    # Subscriptions come first (C15): a subscription of the other provider is
    # taken while this provider's funded key is still available.
    left = "subscription" if next_lease.get("kind") in ("own", "pool") else "credential"
    return (
        f"no {_provider_label(from_provider)} {left} is left, so the retry continues this session on "
        f"{provider} ({now}) from the latest checkpoint, with a handoff prompt that points it at the "
        "previous agent's trajectory."
    )


def format_usage_limit_message(
    *,
    code_provider: str,
    resets_at_iso: Optional[str],
    facts: Optional[Dict[str, Any]],
    local_place: Optional[str] = None,
    accounts: Optional[List[Dict[str, Any]]] = None,
    next_lease: Optional[Dict[str, Any]] = None,
    paths: Optional[List[Dict[str, Any]]] = None,
    limit_type: Optional[str] = None,
) -> str:
    """The parent's tool error for a usage limit (C15): what stopped, every path, the retry.

    The activity never fails over on its own: the parent retries with the
    session id and the next spawn re-credentials. ``accounts``, ``paths`` and
    ``next_lease`` are the backend's stored view (no provider calls).
    ``local_place`` (a Mac run): the work stays in that folder and the run used
    the user's own CLI login, so the respawn targets the Mac again.
    """
    facts = facts or {}
    resets = f" (resets {resets_at_iso})" if resets_at_iso else ""
    limited = next((a for a in accounts or [] if a.get("this_run")), None)
    on_account = f" {limited.get('label')}" if limited else ""
    block = _usage_accounts_block(accounts)
    unusable = limit_type == LOGIN_UNUSABLE_LIMIT_TYPE
    if local_place and unusable:
        return (
            f"Code agent stopped: your own {_provider_label(code_provider)} login{on_account} can no longer be "
            f"used (sign in again to reconnect it). Its files and commits stay in {local_place}.{block}\n"
            "Respawn with spawn_agent(type='code', target='local', goal=<the same goal>) to continue there: "
            + _respawn_clause(next_lease, local=True)
        )
    if local_place:
        return (
            f"Code agent stopped: the usage limit of your own {_provider_label(code_provider)} login"
            f"{on_account} was reached{resets}. Its files and commits stay in {local_place}.{block}\n"
            "Respawn with spawn_agent(type='code', target='local', goal=<the same goal>) to continue there: "
            + _respawn_clause(next_lease, local=True)
        )
    saved = "Its work is saved"
    if facts.get("branch"):
        saved += f" on branch {facts['branch']}"
    if facts.get("checkpoint_seq"):
        saved += f" (checkpoint {facts['checkpoint_seq']})"
    if unusable and code_provider == "codex":
        stopped = (
            f"the {_provider_label(code_provider)} login{on_account} was rejected "
            "(its access token expired or was revoked)"
        )
    elif unusable:
        # Claude Code's authentication_failed is any 401/403: an expired or revoked
        # login, or an invalid or disallowed API key.
        stopped = (
            f"the {_provider_label(code_provider)} credential{on_account} was rejected by the provider "
            "(authentication failed)"
        )
    elif limit_type == "billing":
        stopped = (
            f"the {_provider_label(code_provider)} credential{' ' + on_account.strip() if on_account else ''} "
            f"ran out of credits{resets}"
        )
    else:
        stopped = (
            f"the {_provider_label(code_provider)} usage limit was reached"
            f"{' on' + on_account if on_account else ''}{resets}"
        )
    if unusable and not (isinstance(next_lease, dict) and next_lease.get("provider")):
        clause = (
            "no other credential can be leased right now, so a retry stays paused and starts nothing "
            "until the credential is re-synced, reconnected or replaced, or another account is connected."
        )
    else:
        clause = _respawn_clause(next_lease, local=False, from_provider=code_provider)
    return (
        f"Code agent stopped: {stopped}. {saved}; the sandbox is paused, not killed."
        f"{block}{_shared_paths_block(paths)}\n"
        f"Retry with spawn_agent(type='code', code_session_id='{facts.get('code_session_id')}', "
        "goal=<the same goal>) to continue: " + clause
    )


def format_no_credential_message(
    *,
    code_session_id: str,
    accounts: Optional[List[Dict[str, Any]]] = None,
    paths: Optional[List[Dict[str, Any]]] = None,
    next_lease: Optional[Dict[str, Any]] = None,
) -> str:
    """The parent's tool error for a retry that can lease nothing (C15): nothing started, the earliest reset."""
    retry = f"spawn_agent(type='code', code_session_id='{code_session_id}', goal=<the same goal>)"
    if isinstance(next_lease, dict) and next_lease.get("available_at"):
        when = (
            f"The earliest reset is {next_lease['available_at']} ({_lease_name(next_lease)}). "
            f"Wait with wait_for(until_timestamp=<that time>), then retry with {retry}."
        )
    else:
        when = (
            "No reset time is known: an account must be reconnected or a funded key refilled "
            f"before {retry} can continue."
        )
    return (
        "Code agent not resumed: no Claude Code or Codex credential can be leased right now, so no "
        f"agent was started. Code session {code_session_id} stays paused with its work saved."
        f"{_usage_accounts_block(accounts)}{_shared_paths_block(paths)}\n{when}"
    )


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _reset_label(resets_at_iso: Optional[str], now: float) -> Optional[str]:
    """``4:50 PM UTC`` within 24 h of ``now``, otherwise ``Oct 3, 12:50 AM UTC``."""
    if not resets_at_iso:
        return None
    try:
        reset = datetime.fromisoformat(resets_at_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
    clock = f"{reset.hour % 12 or 12}:{reset.minute:02d} {'AM' if reset.hour < 12 else 'PM'} UTC"
    if 0 <= reset.timestamp() - now < 24 * 3600:
        return clock
    return f"{_MONTHS[reset.month - 1]} {reset.day}, {clock}"


def format_usage_limit_user_message(
    *,
    code_provider: str,
    resets_at_iso: Optional[str],
    local_place: Optional[str] = None,
    now: Optional[float] = None,
    limit_type: Optional[str] = None,
) -> str:
    """The user's one line for a usage-limit pause (``code_agent.failure_message``).

    The parent gets ``format_usage_limit_message`` (the exact retry) instead.
    """
    reset = _reset_label(resets_at_iso, safe_time() if now is None else now)
    provider = "Codex" if code_provider == "codex" else "Claude"
    resets = f" (resets {reset})" if reset else ""
    saved = f"Work stays in {local_place}." if local_place else "Work is saved."
    if limit_type == LOGIN_UNUSABLE_LIMIT_TYPE:
        if code_provider == "codex":
            return f"Paused: the {provider} login was rejected (expired or revoked). {saved}"
        return f"Paused: the {provider} credential was rejected (authentication failed). {saved}"
    return f"Paused: {provider} usage limit reached{resets}. {saved}"


# Resize entries kept in sandbox_sizing (the counts stay exact past it).
SANDBOX_RESIZE_HISTORY = 10


def _size_view(value: Any) -> Optional[Dict[str, int]]:
    """``{cpu, memory_gb}`` when both are positive ints, else None (unknown)."""
    if not isinstance(value, dict):
        return None
    cpu, memory = value.get("cpu"), value.get("memory_gb")
    valid = all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (cpu, memory))
    return {"cpu": cpu, "memory_gb": memory} if valid else None


def initial_sandbox_sizing(session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Requested vs observed size from a create response; None for a sessionless (local) start.

    ``requested`` is the spawn's explicit sandbox_size, ``initial`` the size the
    sandbox started on, ``final`` the latest size Rose recorded for it (resizes
    update it), ``source`` where the initial size came from. Unknown is None.
    """
    reported = session_data.get("sandbox_sizing") if isinstance(session_data.get("sandbox_sizing"), dict) else {}
    size = _size_view(session_data.get("sandbox_size"))
    if not reported and size is None:
        return None
    initial = _size_view(reported.get("initial")) or size
    return {
        "requested": _size_view(reported.get("requested")),
        "initial": initial,
        "final": size or initial,
        "source": reported.get("source") or "unknown",
        "resize_count": 0,
        "resize_failed_count": 0,
        "resizes": [],
    }


def record_sandbox_resize(
    sizing: Optional[Dict[str, Any]],
    *,
    from_size: Any,
    to_size: Any,
    applied: bool,
    observed: Any,
) -> Dict[str, Any]:
    """Add one applied or failed resize; ``final`` becomes the size the backend reports after it."""
    before = _size_view(from_size)
    updated = dict(
        sizing
        or {
            "requested": None,
            "initial": before,
            "final": before,
            "source": "unknown",
            "resize_count": 0,
            "resize_failed_count": 0,
            "resizes": [],
        }
    )
    key = "resize_count" if applied else "resize_failed_count"
    updated[key] = int(updated.get(key) or 0) + 1
    entry = {"from": before, "to": _size_view(to_size), "applied": applied}
    updated["resizes"] = [*list(updated.get("resizes") or []), entry][-SANDBOX_RESIZE_HISTORY:]
    after = _size_view(observed)
    if after is not None:
        updated["final"] = after
    elif applied:
        # Applied but no size reported: the old size is no longer true.
        updated["final"] = None
    return updated


def _size_text(size: Any) -> str:
    view = _size_view(size)
    return f"{view['cpu']} vCPU / {view['memory_gb']} GB" if view else "unknown"


def format_sandbox_sizing(sizing: Optional[Dict[str, Any]]) -> str:
    """One line for the tool result: final vs initial vs requested size, provenance, resizes."""
    if not isinstance(sizing, dict):
        return ""
    parts = [
        f"Sandbox size: {_size_text(sizing.get('final'))} at the end",
        f"started at {_size_text(sizing.get('initial'))} (source: {sizing.get('source') or 'unknown'})",
    ]
    requested = _size_view(sizing.get("requested"))
    if requested is not None:
        parts.append(f"requested {_size_text(requested)}")
        if sizing.get("source") in ("session", "resize", "legacy"):
            parts.append("not applied: a resumed session keeps the size it holds (`rose env size` changes it)")
    applied = int(sizing.get("resize_count") or 0)
    failed = int(sizing.get("resize_failed_count") or 0)
    if applied or failed:
        parts.append(f"resizes: {applied} applied, {failed} failed")
    return "; ".join(parts)


def _with_sandbox_sizing(facts: Optional[Dict[str, Any]], code_agent_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The facts with the run's sandbox sizing attached (the parent and resumes see it)."""
    sizing = code_agent_meta.get("sandbox_sizing")
    if facts is None or not isinstance(sizing, dict):
        return facts
    return {**facts, "sandbox_sizing": sizing}


def format_code_session_summary(facts: Optional[Dict[str, Any]]) -> str:
    """One factual block for the tool result: session, branch, commits, PR, merged."""
    if not facts or not facts.get("code_session_id"):
        return ""
    lines = [f"Code session: {facts['code_session_id']} ({facts.get('state') or 'active'})"]
    if facts.get("branch"):
        target = facts.get("merge_target") or "the default branch"
        lines.append(f"Branch: {facts['branch']} (merges into {target})")
    if facts.get("head"):
        pushed = "yes" if facts.get("pushed") else "no"
        lines.append(
            f"HEAD: {str(facts['head'])[:12]} | local commits not on the remote: "
            f"{facts.get('local_commits') or 0} | pushed: {pushed} | uncommitted files: "
            f"{facts.get('dirty_files') or 0}"
        )
    pr = facts.get("pr")
    if isinstance(pr, dict) and pr.get("number"):
        lines.append(f"PR: #{pr['number']} {pr.get('state') or ''} {pr.get('url') or ''}".rstrip())
    if facts.get("branch"):
        source = facts.get("merge_source")
        until = facts.get("rate_limited_until")
        if source == "github_rate_limited":
            retry = f"; retry after {until}" if until else ""
            lines.append(f"Merged: unknown (GitHub rate limit reached{retry})")
        elif source == "github_unavailable":
            lines.append("Merged: unknown (GitHub could not be reached)")
        else:
            lines.append(f"Merged: {'yes' if facts.get('merged') else 'no'} (confirmed with GitHub)")
    sizing = format_sandbox_sizing(facts.get("sandbox_sizing"))
    if sizing:
        lines.append(sizing)
    return "\n".join(lines)


def _no_credential_result(
    *,
    available: Dict[str, Any],
    code_session_id: str,
    code_provider: str,
    code_agent_meta: Dict[str, Any],
    started_at: float,
) -> Dict[str, Any]:
    """A retry with no leasable credential anywhere (C15): nothing created, the session stays paused."""
    outlook = available.get("outlook") if isinstance(available.get("outlook"), dict) else {}
    accounts = outlook.get("accounts") if isinstance(outlook.get("accounts"), list) else None
    paths = outlook.get("paths") if isinstance(outlook.get("paths"), list) else None
    next_lease = outlook.get("next_lease") if isinstance(outlook.get("next_lease"), dict) else None
    reset_iso = (next_lease or {}).get("available_at")
    reset = _reset_label(reset_iso, safe_time())
    user_message = (
        f"Paused: no Claude or Codex credential is available{f' (earliest reset {reset})' if reset else ''}. "
        "Work is saved."
    )
    facts = {"code_session_id": code_session_id, "state": "PAUSED"}
    logger.warning(
        "[ClaudeCode] Retry has no leasable credential; the session stays paused",
        code_session_id=code_session_id,
        earliest_reset=reset_iso,
    )
    return {
        "status": CODE_AGENT_USAGE_LIMIT_STATUS,
        "success": False,
        "failure_kind": "no_credentials",
        "retryable": True,
        "message": format_no_credential_message(
            code_session_id=code_session_id, accounts=accounts, paths=paths, next_lease=next_lease
        ),
        "usage_limit": {
            "provider": code_provider,
            "resets_at": reset_iso,
            "limit_type": None,
            "credential_marked": None,
            "accounts": accounts,
            "paths": paths,
            "next_lease": next_lease,
        },
        "retry": {"tool": "spawn_agent", "type": "code", "code_session_id": code_session_id},
        "task_id": None,
        "ws_url": None,
        "execution_time_ms": (safe_time() - started_at) * 1000,
        "usage_items": [],
        "structured_response": None,
        "output_files": [],
        "code_provider": code_provider,
        "code_session": facts,
        "code_agent": build_code_agent_metadata(
            target_meta=code_agent_meta,
            task_id=None,
            provider=code_provider,
            state="paused",
            started_at=started_at,
            ended_at=safe_time(),
            facts=facts,
            failure_message=user_message,
        ),
        "completed_turns": None,
    }


def _activity_errors_so_far() -> int:
    """How many errors earlier attempts of this activity raised (from their heartbeats).

    Worker deaths and heartbeat timeouts raise nothing, so they never count.
    """
    if not activity.in_activity():
        return 0
    for detail in activity.info().heartbeat_details or ():
        if isinstance(detail, dict):
            count = detail.get(CODE_AGENT_ERRORS_HEARTBEAT_KEY)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                return count
    return 0


def _bounded_activity_error(error: BaseException, errors: int) -> BaseException:
    """The error to raise when this activity has raised ``errors`` errors in total.

    Retries are otherwise unlimited (worker crashes and deploys never end a code
    session), so a deterministic failure must not loop forever: the
    ``CODE_AGENT_ERROR_RETRY_LIMIT``-th error the activity itself raises is
    non-retryable. Outside an activity nothing retries and the error is final.
    """
    if (
        activity.in_activity()
        and errors >= CODE_AGENT_ERROR_RETRY_LIMIT
        and not (isinstance(error, ApplicationError) and error.non_retryable)
    ):
        final = ApplicationError(
            extract_error_message(error),
            type=type(error).__name__,
            non_retryable=True,
        )
        final.__cause__ = error
        return final
    return error


def _error_ends_the_run(error: BaseException) -> bool:
    """Whether no later attempt of this activity will run after ``error``."""
    return not activity.in_activity() or (isinstance(error, ApplicationError) and error.non_retryable)


def code_session_failure_error(
    error: BaseException,
    original: BaseException,
    facts: Optional[Dict[str, Any]],
    *,
    local: bool = False,
) -> BaseException:
    """The final error of a run whose code session exists, carrying how to resume it.

    A run that died on a dropped connection used to reach the parent as a bare
    one-liner (tar_2b02efd49fc2466497723b3e): no code_session_id, so the saved
    sandbox, commits and checkpoint could not be resumed. This error keeps the
    exact exception and adds the session id, the failure checkpoint and the
    exact retry call. Without a code session (or on a Mac, which resumes by
    folder) ``error`` is returned unchanged.
    """
    code_session_id = (facts or {}).get("code_session_id")
    if local or not isinstance(code_session_id, str) or not code_session_id:
        return error
    exact = extract_error_message(original)
    error_type = type(original).__name__
    checkpoint_seq = (facts or {}).get("checkpoint_seq")
    saved = f"saved at checkpoint {checkpoint_seq}" if checkpoint_seq is not None else "kept"
    lines = [
        f"Code agent run ended abnormally: {error_type}: {exact}",
        f"Code session {code_session_id} was not lost: its sandbox files, commits and conversation are {saved}.",
    ]
    if (facts or {}).get("branch"):
        lines.append(f"Branch: {facts['branch']}")
    sizing = format_sandbox_sizing((facts or {}).get("sandbox_sizing"))
    if sizing:
        lines.append(sizing)
    lines.append(
        f"Resume it with spawn_agent(type='code', code_session_id='{code_session_id}', goal=<the same goal>)."
    )
    final = ApplicationError(
        "\n".join(lines),
        {
            "code_session_id": code_session_id,
            "code_session": dict(facts or {}),
            "error": exact,
            "error_type": error_type,
            "retry": {"tool": "spawn_agent", "type": "code", "code_session_id": code_session_id},
        },
        type=CODE_AGENT_SESSION_FAILED_CODE,
        non_retryable=True,
    )
    final.__cause__ = original
    return final


# ---------------------------------------------------------------------------
# Temporal activity
# ---------------------------------------------------------------------------


@activity.defn
async def resolve_code_agent_provider_activity(default_provider: str) -> str:
    """Record provider overrides in activity history for deterministic replay."""
    return resolve_code_agent_provider(default_provider)


LOGIN_UNUSABLE_LIMIT_TYPE = "login_unusable"


async def _unusable_login_outlook(
    user_id: str, code_provider: str, code_session_id: Optional[str], *, local: bool
) -> Dict[str, Any]:
    """The stored outlook after a rejected login was fed back (C15; no lease taken).

    The usage-limit report would rate-limit the login, so the plan answers instead:
    the connected accounts (a revoked login reads disconnected; an expired token
    waits for its device's re-sync), the shared paths and the next lease in the
    subscription-first order.
    """
    available = await fetch_code_credential_availability(
        user_id,
        own_only=local,
        code_session_id=None if local else code_session_id,
        prefer=code_provider,
    )
    outlook = (available or {}).get("outlook")
    return {"marked": "login_rejected", "resets_at": None, **(outlook if isinstance(outlook, dict) else {})}


def _next_code_credential(available: Optional[Dict[str, Any]], preferred: str) -> tuple[Optional[str], Optional[str]]:
    """The provider (and tier) a spawn takes: the backend's ``next_credential`` (C15 order).

    An older backend without it: the preferred provider when it has any
    credential, else the other one.
    """
    if available is None:
        return None, None
    step = available.get("next_credential")
    if isinstance(step, dict) and step.get("provider") in ("claude", "codex"):
        return str(step["provider"]), str(step.get("tier") or "") or None
    if "next_credential" in available:
        return None, None
    other = "codex" if preferred == "claude" else "claude"
    for provider in (preferred, other):
        if available.get(provider):
            return provider, None
    return None, None


async def _resolve_claude_token_for_activity(user_id: str, code_provider: str) -> Optional[str]:
    """Resolve Claude OAuth inside the activity so it never enters workflow history."""
    if code_provider != "claude":
        return None

    from helpers.llm import fetch_claude_max_token

    result = await fetch_claude_max_token(user_id)
    token = result.get("accessToken") if isinstance(result, dict) else None
    logger.info(
        "[ClaudeCode] Activity-local Claude credential resolved",
        user_id=user_id,
        source=result.get("source") if isinstance(result, dict) else None,
        credential_found=bool(token),
    )
    return token if isinstance(token, str) and token else None


@activity.defn
@traced("spawn_claude_code_activity")
async def spawn_claude_code_activity(
    user_id: str,
    agent_id: str,
    goal: str,
    tool_call_id: str,
    output_schema: Optional[Dict[str, Any]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    model: str = "rose-labs/rose-1-flash",
    github_repo_url: Optional[str] = None,
    _legacy_claude_token: Optional[str] = None,
    code_provider: str = "claude",
    gpt_token: Optional[str] = None,
    gpt_account_id: Optional[str] = None,
    gpt_token_source: Optional[str] = None,
    gpt_token_pool_id: Optional[str] = None,
    fast_mode_cache_enabled: bool = False,
    resume_session: Optional[Dict[str, Any]] = None,
    code_session_request: Optional[Dict[str, Any]] = None,
    target: str = "cloud",
    local_device_id: Optional[str] = None,
    local_device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Claude Code agent activity using E2B Desktop Sandbox.

    Flow:
    1. Call back-end API to create E2B session (with file attachments, optional GitHub repo)
    2. Post init message with stream_url
    3. Monitor Claude Code output via polling
    4. Run verifier to confirm completion
    5. On completion, close session and extract output files

    A worker drain detaches instead: the sandbox keeps running and the activity
    returns ``CODE_AGENT_DRAIN_YIELD_STATUS`` with the ``code_session`` to pass
    back as ``resume_session``. A retry after a crashed or timed-out attempt
    reattaches to the session carried on the previous attempt's heartbeats. The
    sandbox is closed whenever no attempt will reattach: a requested cancel, a
    terminated workflow, or the last attempt the retry policy allows.

    Args:
        output_schema: Response schema from parent agent's tool call (required for structured responses)
        attachments: File attachments from context to upload to sandbox
        model: Model to use for verifier (inherited from parent agent)
        github_repo_url: Optional GitHub repo URL to clone before starting Claude Code
        _legacy_claude_token: Ignored compatibility slot for activities scheduled by older workers
        code_provider: "claude" for Claude Code or "codex" for Codex CLI
        resume_session: Session returned by a drained attempt; its sandbox is reattached
            when still reachable. ``generation_yield=False`` keeps the run on this worker
            when its generation is superseded (no parked handoff is available).
        target: "cloud" (E2B sandbox) or "local" (the same ws-server on the agent's Mac,
            relayed by the backend; ``ws_url`` is then the backend relay URL). Everything
            after session create is identical for both targets.
        local_device_id / local_device_name: the Mac resolved by
            ``resolve_code_agent_target_activity`` for a local run.
        code_session_request: ``{"code_session_id": ...}`` to resume a specific
            persistent code session; omitted, the backend resumes the spawning
            agent's latest session for the repo (or creates one on a new branch).
            ``task_name`` (the spawn's) is stored on the task row as its title.
            ``github_repo_ref`` (the spawn's explicit connected-repo selection) rides
            here too, so the scheduled positional arguments stay unchanged.

    No runtime limit (C11): the activity runs while the CLI works and heartbeats.
    An idle CLI is paused (status ``paused``, resumable), never failed.
    """
    if not output_schema:
        # Invalid scheduled arguments cannot heal on a worker/crash retry. Fail
        # before installing heartbeat context or allocating a sandbox.
        raise ApplicationError(
            "output_schema is required for Claude Code agent - parent agent must provide structured output schema in tool call",
            type="CODE_AGENT_OUTPUT_SCHEMA_REQUIRED",
            non_retryable=True,
        )

    structlog.contextvars.bind_contextvars(agent_id=agent_id)
    source_code_provider = code_provider

    from helpers.model_pricing import get_model_pricing_with_fallback

    task_id = None
    sandbox_id = None
    ws_url = None
    start_time = safe_time()
    log_collector: Optional[SandboxLogCollector] = None
    session_record: Optional[Dict[str, Any]] = None
    session_heartbeats = contextlib.ExitStack()
    # Every heartbeat carries how many errors this activity raised so far, so
    # the retry budget counts only those, never worker deaths (see
    # _bounded_activity_error). Entered before anything can heartbeat.
    retry_state: Dict[str, Any] = {CODE_AGENT_ERRORS_HEARTBEAT_KEY: _activity_errors_so_far()}
    retry_heartbeats = contextlib.ExitStack()
    retry_heartbeats.enter_context(heartbeat_resume_details(retry_state))
    generation_yield = not (isinstance(resume_session, dict) and resume_session.get("generation_yield") is False)
    # A retry's heartbeat session is newer than the scheduled resume_session: the
    # earlier attempt may have replaced an unreachable sandbox with a fresh one.
    attached = _code_session_from_heartbeat() or code_session_record(resume_session)
    # The persistent session survives a paused/unreachable sandbox: a retry
    # resumes it by id even when its attached record must be discarded (C2).
    requested_code_session_id = (code_session_request or {}).get("code_session_id")
    github_repo_ref = explicit_github_repo_ref((code_session_request or {}).get("github_repo_ref"))
    code_session_id: Optional[str] = (attached or {}).get("code_session_id") or (
        requested_code_session_id if isinstance(requested_code_session_id, str) else None
    )
    lease_owner = f"tool:{agent_id}:{tool_call_id}"
    code_session_facts: Optional[Dict[str, Any]] = None
    loop_outcome: Dict[str, Any] = {}

    total_input_tokens = 0
    total_output_tokens = 0
    action_summary: list[str] = []
    code_events: list[Dict[str, Any]] = []
    terminal_output_persisted_box = [False]

    if isinstance(attached, dict) and attached.get("target") == "local":
        target = "local"
        local_device_id = attached.get("device_id") or local_device_id
        local_device_name = attached.get("device_name") or local_device_name
    code_agent_meta: Dict[str, Any] = {"target": "local" if target == "local" else "cloud"}
    if target == "local" and local_device_name:
        code_agent_meta["device_name"] = local_device_name
    if isinstance(attached, dict) and attached.get("folder"):
        code_agent_meta["folder"] = attached["folder"]
    if isinstance(attached, dict) and attached.get("preview_url"):
        code_agent_meta["preview_url"] = attached["preview_url"]
    task_name = (code_session_request or {}).get("task_name")
    if isinstance(task_name, str) and task_name:
        # metadata.code_agent.title: the code-agent view's header when opened without one.
        code_agent_meta["title"] = task_name
    if isinstance(attached, dict) and isinstance(attached.get("sandbox_size"), dict):
        code_agent_meta["sandbox_size"] = attached["sandbox_size"]
    if isinstance(attached, dict) and isinstance(attached.get("sandbox_sizing"), dict):
        code_agent_meta["sandbox_sizing"] = attached["sandbox_sizing"]

    logger.info(
        "[ClaudeCode] Using response schema from tool call",
        agent_id=agent_id,
        fields=len(output_schema.get("fields", [])),
    )

    terminal = TerminalFilter(cols=200, rows=500)

    try:
        if attached is not None:
            # A pause landing during the reattach probe must still find the sandbox to close.
            task_id = attached["task_id"]
        if attached is not None and not await ws_endpoint_reachable(
            attached["ws_url"], attempts=REATTACH_PROBE_ATTEMPTS, backoff_s=REATTACH_PROBE_BACKOFF_S
        ):
            logger.warning(
                "[ClaudeCode] Previous sandbox unreachable; closing it before a fresh start",
                agent_id=agent_id,
                task_id=attached["task_id"],
                sandbox_id=attached["sandbox_id"],
            )
            try:
                await close_e2b_session(
                    task_id=attached["task_id"],
                    code_provider=attached.get("code_provider", code_provider),
                )
            except Exception as close_error:
                logger.warning(
                    "[ClaudeCode] Closing the unreachable sandbox failed",
                    agent_id=agent_id,
                    task_id=attached["task_id"],
                    error=str(close_error),
                )
            attached = None

        if attached is not None:
            code_provider = attached.get("code_provider", code_provider)
            task_id = attached["task_id"]
            sandbox_id = attached["sandbox_id"]
            ws_url = attached["ws_url"]
            start_time = attached.get("started_at") or start_time
            actual_github_repo_url = attached.get("github_repo_url")
            loop_started_at = attached.get("loop_started_at")
            uploaded_files = []
            logger.info(
                "[ClaudeCode] Reattaching to running sandbox",
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                attempt=activity.info().attempt,
            )
            action_summary.append(f"Reattached to E2B sandbox: {sandbox_id}")

        if attached is None:
            # C15 re-credentialing, in the backend's order: every subscription
            # before any funded key. A resumed session (a retry after a usage
            # limit, or the root passing its code_session_id) prefers its own
            # provider: same harness, native resume of its conversation. With
            # none of its subscriptions left the session continues on the other
            # provider's subscription (provider handoff in the backend) before
            # its own funded key. A retry that can lease nothing stays paused
            # and starts nothing.
            # A Mac run only ever gets the user's own CLI login, never a pool token.
            resuming = code_session_id if target != "local" else None
            available = await fetch_code_credential_availability(
                user_id,
                own_only=target == "local",
                code_session_id=resuming,
                prefer=code_provider,
            )
            session_provider = (available or {}).get("session_provider")
            preferred = session_provider if session_provider in ("claude", "codex") else code_provider
            chosen, chosen_tier = _next_code_credential(available, preferred)
            if available is not None and chosen == preferred:
                if preferred != code_provider:
                    logger.info(
                        "[ClaudeCode] Resuming the code session on its own provider",
                        agent_id=agent_id,
                        requested_provider=code_provider,
                        session_provider=preferred,
                        code_session_id=code_session_id,
                    )
                    action_summary.append(f"Resuming code session on its own provider ({preferred})")
                code_provider = preferred
            elif available is not None and chosen is not None:
                logger.warning(
                    "[ClaudeCode] No subscription left for provider; continuing on the other one",
                    agent_id=agent_id,
                    from_provider=preferred,
                    to_provider=chosen,
                    to_tier=chosen_tier,
                    code_session_id=code_session_id,
                )
                left = "subscription" if chosen_tier == "subscription" else "credential"
                action_summary.append(f"No {preferred} {left} available; continuing on {chosen}")
                code_provider = chosen
            elif available is not None and resuming:
                return _no_credential_result(
                    available=available,
                    code_session_id=resuming,
                    code_provider=preferred,
                    code_agent_meta=code_agent_meta,
                    started_at=start_time,
                )

        claude_token = await _resolve_claude_token_for_activity(user_id, code_provider)

        if attached is None:
            # 1. Create E2B session via back-end API
            logger.info(
                "[ClaudeCode] Creating E2B session",
                agent_id=agent_id,
                goal=goal[:200],
                num_attachments=len(attachments) if attachments else 0,
            )

            session_data = await create_e2b_session(
                agent_id=agent_id,
                # A Mac folder has no sandbox handoff file; it keeps the session-output result.
                goal=goal if target == "local" else handoff_task_text(goal, output_schema),
                user_id=user_id,
                attachments=attachments,
                github_repo_url=github_repo_url,
                code_provider=code_provider,
                fast_mode_cache_enabled=fast_mode_cache_enabled,
                code_session_id=code_session_id,
                lease_owner=lease_owner,
                github_repo_ref=github_repo_ref,
                target=target,
                local_device_id=local_device_id,
                fresh_conversation=(code_session_request or {}).get("fresh_conversation") is True,
                task_name=(code_session_request or {}).get("task_name"),
                sandbox_size=(code_session_request or {}).get("sandbox_size"),
                cli_settings=(code_session_request or {}).get("cli_settings"),
                resume_config={
                    "source_code_provider": source_code_provider,
                    "spawn_args": {
                        "type": "code", "goal": goal, "output_schema": output_schema,
                        "github_repo_link": github_repo_url, "target": target, "task_name": task_name,
                        **({"github_repo_ref": github_repo_ref} if github_repo_ref else {}),
                        "attachments": [item.get("original_filename") for item in attachments or []],
                    },
                    "attachments": attachments or [],
                    "model": model,
                    "fast_mode_cache_enabled": fast_mode_cache_enabled,
                },
            )
            code_session_id = session_data.get("code_session_id") or code_session_id
            code_session_facts = _spawn_code_session_facts(session_data)
            task_id = session_data["task_id"]
            sandbox_id = session_data["sandbox_id"]
            ws_url = session_data["ws_url"]
            uploaded_files = session_data.get("uploaded_files", [])
            actual_github_repo_url = session_data.get("github_repo_url")
            loop_started_at = None
            if target == "local" and session_data.get("folder"):
                # Display path of the folder the Mac granted (e.g. "~/Developer/rose").
                code_agent_meta["folder"] = session_data["folder"]
            if session_data.get("preview_url"):
                # Live dev-server preview of this session (Rose dev accounts, A1).
                code_agent_meta["preview_url"] = session_data["preview_url"]
            if isinstance(session_data.get("sandbox_size"), dict):
                # vCPUs / memory of the E2B sandbox (C16).
                code_agent_meta["sandbox_size"] = session_data["sandbox_size"]
            sizing = initial_sandbox_sizing(session_data)
            if sizing is not None:
                # Requested vs observed size and its provenance (reported on every end path).
                code_agent_meta["sandbox_sizing"] = sizing
                code_session_facts = _with_sandbox_sizing(code_session_facts, code_agent_meta)
            if isinstance(session_data.get("startup"), dict):
                # How long the sandbox's ws-server took to serve and its repo clone
                # ({ws_ready_ms, clone_ms}); the code-agent benchmark reads it.
                code_agent_meta["startup"] = session_data["startup"]

            logger.info(
                "[ClaudeCode] Session created",
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                ws_url=ws_url[:100] if ws_url else None,
                uploaded_files=len(uploaded_files),
            )
            action_summary.append(
                f"Started local session on {local_device_name or 'the Mac'}: {sandbox_id}"
                if target == "local"
                else f"Created E2B sandbox: {sandbox_id}"
            )

        # A spawn attached to a live CLI starts its logs and transcript at its own goal turn.
        history_events = int((attached if attached is not None else session_data).get("history_events") or 0)
        session_record = {
            "task_id": task_id,
            "sandbox_id": sandbox_id,
            "ws_url": ws_url,
            "code_provider": code_provider,
            "github_repo_url": actual_github_repo_url,
            "started_at": start_time,
            "loop_started_at": loop_started_at,
            "logged_events": max(int((attached or {}).get("logged_events") or 0), history_events),
            "history_events": history_events,
            "code_session_id": code_session_id,
            "branch": (code_session_facts or {}).get("branch") or (attached or {}).get("branch"),
            "merge_target": (code_session_facts or {}).get("merge_target")
            or (attached or {}).get("merge_target"),
            "transcript_events": max(int((attached or {}).get("transcript_events") or 0), history_events),
            "preview_url": code_agent_meta.get("preview_url"),
            "sandbox_size": code_agent_meta.get("sandbox_size"),
            "sandbox_sizing": code_agent_meta.get("sandbox_sizing"),
        }
        if code_session_facts is None and code_session_id:
            code_session_facts = _with_sandbox_sizing(
                {
                    "code_session_id": code_session_id,
                    "branch": session_record["branch"],
                    "merge_target": session_record["merge_target"],
                },
                code_agent_meta,
            )
        if target == "local":
            # A drained/retried attempt must reattach to the same Mac, never the cloud.
            session_record.update(
                {
                    "target": "local",
                    "device_id": local_device_id,
                    "device_name": local_device_name,
                    "folder": code_agent_meta.get("folder"),
                }
            )
        # The session must ride along before anything else can fail. The create
        # heartbeats while it waits, so this beat can sit out one throttle
        # interval; a retry after a crash in that window creates again and the
        # backend hands back this same running task (createWithCodeSession).
        session_heartbeats.enter_context(
            heartbeat_resume_details({CODE_SESSION_HEARTBEAT_KEY: session_record}, retry_state)
        )
        safe_heartbeat("Claude Code sandbox attached")

        log_collector = SandboxLogCollector(sandbox_id, agent_id, task_id)
        await log_collector.start()

        if attached is None:
            if uploaded_files:
                action_summary.append(f"Uploaded {len(uploaded_files)} files to /home/user/attachments/")

            # 2. Post init message
            await save_claude_code_message(
                agent_id=agent_id,
                title="Claude Code Started",
                task_id=task_id,
                message=f"Goal: {goal}",
                status=MessageStatus.IN_PROGRESS,
                action_type="claude_code_started",
                tool_call_id=tool_call_id,
                is_init_message=True,
                ws_url=ws_url,
                output_schema=output_schema,
                github_repo_url=actual_github_repo_url,
                code_provider=code_provider,
                code_session=code_session_facts,
                code_agent=build_code_agent_metadata(
                    target_meta=code_agent_meta,
                    task_id=task_id,
                    provider=code_provider,
                    state="live",
                    started_at=start_time,
                    facts=code_session_facts,
                ),
            )

            await save_claude_code_message(
                agent_id=agent_id,
                title="Task Started",
                task_id=task_id,
                message=goal,
                status=MessageStatus.FINISHED,
                action_type="task_started",
                tool_call_id=tool_call_id,
                code_provider=code_provider,
                code_agent=build_code_agent_metadata(
                    target_meta=code_agent_meta,
                    task_id=task_id,
                    provider=code_provider,
                    state="live",
                    started_at=start_time,
                    facts=code_session_facts,
                ),
            )
            action_summary.append(f"Started task: {goal[:100]}")

        # 3. WebSocket-based agentic loop
        if loop_started_at is None:
            loop_started_at = safe_time()
            session_record["loop_started_at"] = loop_started_at
        safe_heartbeat("Connecting to Claude Code WebSocket")

        model_pricing = await get_model_pricing_with_fallback(model)

        try:
            (
                exit_code,
                attempt_completion,
                conversation_history,
                loop_input_tokens,
                loop_output_tokens,
                code_events,
            ) = await _run_agentic_loop(
                agent_id=agent_id,
                task_id=task_id,
                sandbox_id=sandbox_id,
                goal=goal,
                ws_url=ws_url,
                terminal=terminal,
                model=model,
                claude_token=claude_token,
                gpt_token=gpt_token,
                gpt_account_id=gpt_account_id,
                gpt_token_source=gpt_token_source,
                gpt_token_pool_id=gpt_token_pool_id,
                gpt_token_user_id=user_id if gpt_token else None,
                code_provider=code_provider,
                action_summary=action_summary,
                tool_call_id=tool_call_id,
                code_events=code_events,
                terminal_output_persisted=terminal_output_persisted_box,
                loop_started_at=loop_started_at,
                generation_yield=generation_yield,
                code_session=session_record,
                loop_outcome=loop_outcome,
            )
        except CodeSessionDrained as drained:
            # Cleared before awaiting so a cancel landing in stop() reaches the
            # cancel handler below without stopping the collector twice.
            collector, log_collector = log_collector, None
            if collector:
                await collector.stop()
            if isinstance(drained, CodeSessionResizeRequested):
                resized = await _apply_sandbox_resize(
                    agent_id=agent_id,
                    tool_call_id=tool_call_id,
                    task_id=task_id,
                    code_provider=code_provider,
                    request=drained.request,
                    session_record=session_record,
                    code_agent_meta=code_agent_meta,
                    started_at=start_time,
                    facts=code_session_facts,
                )
                if resized is not None:
                    # The next attempt reattaches to the task that continues the session.
                    return {
                        "status": CODE_AGENT_DRAIN_YIELD_STATUS,
                        "yield_reason": CODE_AGENT_RESIZE_YIELD_REASON,
                        "code_session": resized,
                        "task_id": resized["task_id"],
                        "code_provider": code_provider,
                    }
            return {
                "status": CODE_AGENT_DRAIN_YIELD_STATUS,
                "yield_reason": drained.reason,
                "code_session": session_record,
                "task_id": task_id,
                "code_provider": code_provider,
            }
        total_input_tokens += loop_input_tokens
        total_output_tokens += loop_output_tokens

        # 4. Persist terminal output (durable fallback when PTY display is empty)
        filtered_output = terminal.get_display()
        # Only this spawn's events: an attached live CLI's earlier turns are history.
        credential_feedback = build_credential_feedback(
            code_events[int(session_record.get("history_events") or 0) :], code_provider
        )
        durable_output = build_durable_terminal_transcript(
            filtered_output,
            code_events,
            action_summary,
        )
        if durable_output:
            await persist_terminal_output(
                task_id,
                durable_output,
                agent_id,
                code_provider,
                credential_feedback=credential_feedback,
            )
            terminal_output_persisted_box[0] = True
        else:
            logger.warning(
                "[ClaudeCode] Skipping terminal persist — display and code events were empty",
                agent_id=agent_id,
                task_id=task_id,
            )

        usage_limit = loop_outcome.get("usage_limit")
        login_unusable = loop_outcome.get("login_unusable") if not isinstance(usage_limit, dict) else None
        if isinstance(usage_limit, dict) or isinstance(login_unusable, dict):
            # C15: save the session (checkpoint + pause), rate-limit the token
            # (an unusable login was marked for reconnect by the persisted
            # credential feedback above), and hand the parent the exact retry.
            # Never a Temporal retry: the same credential would fail again.
            close_data = await close_e2b_session(
                task_id=task_id,
                user_id=user_id,
                agent_id=agent_id,
                additional_paths=[],
                code_provider=code_provider,
                reason="usage_limit",
            )
            code_session_facts = _merge_close_code_session_facts(code_session_facts, close_data)
            local_place = local_workspace_place(code_agent_meta)
            if isinstance(usage_limit, dict):
                marked = await report_code_usage_limit(task_id, code_provider, usage_limit.get("resets_at"))
                limit_type = usage_limit.get("limit_type")
            else:
                marked = await _unusable_login_outlook(
                    user_id, code_provider, (code_session_facts or {}).get("code_session_id"), local=bool(local_place)
                )
                limit_type = LOGIN_UNUSABLE_LIMIT_TYPE
            session_heartbeats.close()
            if log_collector:
                await log_collector.stop()
            resets_at_iso = (marked or {}).get("resets_at")
            # The parent reads the error with every connected account and what a
            # respawn leases (the backend's stored view); the user reads a
            # resumable pause. The parent decides; nothing fails over here.
            accounts = (marked or {}).get("accounts")
            accounts = accounts if isinstance(accounts, list) else None
            next_lease = (marked or {}).get("next_lease")
            next_lease = next_lease if isinstance(next_lease, dict) else None
            paths = (marked or {}).get("paths")
            paths = paths if isinstance(paths, list) else None
            message = format_usage_limit_message(
                code_provider=code_provider,
                resets_at_iso=resets_at_iso,
                facts=code_session_facts,
                local_place=local_place,
                accounts=accounts,
                next_lease=next_lease,
                paths=paths,
                limit_type=limit_type,
            )
            user_message = format_usage_limit_user_message(
                code_provider=code_provider,
                resets_at_iso=resets_at_iso,
                local_place=local_place,
                limit_type=limit_type,
            )
            code_agent = build_code_agent_metadata(
                target_meta=code_agent_meta,
                task_id=task_id,
                provider=code_provider,
                state="paused",
                started_at=start_time,
                ended_at=safe_time(),
                facts=code_session_facts,
                failure_message=user_message,
            )
            await save_claude_code_message(
                agent_id=agent_id,
                title=(
                    "Code agent credential was rejected"
                    if limit_type == LOGIN_UNUSABLE_LIMIT_TYPE
                    else "Code agent hit a usage limit"
                ),
                task_id=task_id,
                message=user_message,
                status=MessageStatus.FINISHED,
                action_type="usage_limit",
                tool_call_id=tool_call_id,
                code_provider=code_provider,
                code_session=code_session_facts,
                code_agent=code_agent,
            )
            return {
                "status": CODE_AGENT_USAGE_LIMIT_STATUS,
                "success": False,
                "failure_kind": "login_unusable" if limit_type == LOGIN_UNUSABLE_LIMIT_TYPE else "usage_limit",
                "retryable": True,
                "message": message,
                "usage_limit": {
                    "provider": code_provider,
                    "resets_at": resets_at_iso,
                    "limit_type": limit_type,
                    "credential_marked": (marked or {}).get("marked"),
                    "accounts": accounts,
                    "paths": paths,
                    "next_lease": next_lease,
                },
                # A Mac run has no code session: its retry targets the same Mac folder.
                "retry": (
                    {"tool": "spawn_agent", "type": "code", "target": "local"}
                    if local_place
                    else {
                        "tool": "spawn_agent",
                        "type": "code",
                        "code_session_id": (code_session_facts or {}).get("code_session_id"),
                    }
                ),
                "task_id": task_id,
                "ws_url": ws_url,
                "execution_time_ms": (safe_time() - start_time) * 1000,
                "usage_items": _sandbox_usage_items(code_agent_meta, start_time),
                "structured_response": None,
                "output_files": close_data.get("output_files", []),
                "code_provider": code_provider,
                "code_session": code_session_facts,
                "code_agent": code_agent,
                "completed_turns": codex_completed_turn_count(code_events) if code_provider == "codex" else None,
            }

        # A Codex run whose credentials stayed exhausted after its in-sandbox
        # fallback is a terminal tool error, not an ordinary incomplete result
        # for the verifier (which would make the parent re-spawn paid sandboxes).
        fatal_error = check_terminal_fatal_error(
            code_provider=code_provider,
            exit_code=exit_code,
            credential_feedback=credential_feedback,
        )
        if fatal_error:
            action_summary.append(f"FATAL: {fatal_error}")
            raise ApplicationError(
                fatal_error,
                type="CLAUDE_CODE_FATAL",
                non_retryable=True,
            )

        # 5. Close session and extract output files before verifier so the
        # verifier sees the SDK event JSON plus the actual extracted artifacts.
        # A persistent session is checkpointed and paused, never killed.
        idle_paused = loop_outcome.get("idle_paused") is True
        close_data = await close_e2b_session(
            task_id=task_id,
            user_id=user_id,
            agent_id=agent_id,
            additional_paths=[],
            code_provider=code_provider,
            reason="idle" if idle_paused else "close",
        )
        try:
            logger.info(
                "[ClaudeCode] Close acknowledged", agent_id=agent_id, task_id=task_id,
                provider=code_provider, close_reason="idle" if idle_paused else "close",
            )
        except Exception:
            pass  # Diagnostics cannot change close-result processing.
        output_files = close_data.get("output_files", [])
        claude_code_usage = close_data.get("usage")
        code_session_facts = _merge_close_code_session_facts(code_session_facts, close_data)
        session_heartbeats.close()

        if idle_paused:
            # Idle is not failure (C11): report the paused, resumable session.
            if log_collector:
                await log_collector.stop()
            local_place = local_workspace_place(code_agent_meta)
            if local_place:
                # A Mac session cannot be E2B-paused: its CLI is stopped and the
                # work stays in the user's folder; the next spawn continues there.
                message = (
                    f"Code session paused after {loop_outcome.get('idle_s', 0)}s with no activity "
                    f"(not failed). Its files and commits stay in {local_place}; spawn the code agent "
                    "again with target='local' to continue there."
                )
            else:
                message = (
                    f"Code session paused after {loop_outcome.get('idle_s', 0)}s with no activity "
                    "(not failed). Its files, commits and conversation are kept; spawn the code "
                    "agent again to resume it."
                )
            summary = format_code_session_summary(code_session_facts)
            await save_claude_code_message(
                agent_id=agent_id,
                title="Code session paused",
                task_id=task_id,
                message=message,
                status=MessageStatus.FINISHED,
                action_type="session_paused",
                tool_call_id=tool_call_id,
                code_provider=code_provider,
                code_session=code_session_facts,
                code_agent=build_code_agent_metadata(
                    target_meta=code_agent_meta,
                    task_id=task_id,
                    provider=code_provider,
                    state="paused",
                    started_at=start_time,
                    ended_at=safe_time(),
                    facts=code_session_facts,
                ),
            )
            return {
                "status": CODE_AGENT_PAUSED_STATUS,
                "success": False,
                "message": f"{message}\n{summary}" if summary else message,
                "task_id": task_id,
                "ws_url": ws_url,
                "execution_time_ms": (safe_time() - start_time) * 1000,
                "usage_items": _sandbox_usage_items(code_agent_meta, start_time),
                "structured_response": None,
                "output_files": output_files,
                "code_provider": code_provider,
                "code_session": code_session_facts,
                "code_agent": build_code_agent_metadata(
                    target_meta=code_agent_meta,
                    task_id=task_id,
                    provider=code_provider,
                    state="paused",
                    started_at=start_time,
                    ended_at=safe_time(),
                    facts=code_session_facts,
                ),
                "completed_turns": codex_completed_turn_count(code_events) if code_provider == "codex" else None,
            }
        safe_heartbeat("Verifying Claude Code result")

        logger.info(
            "[ClaudeCode] Session closed",
            agent_id=agent_id,
            task_id=task_id,
            output_files=len(output_files),
            claude_code_usage=claude_code_usage,
            code_events=len(code_events),
        )
        if output_files:
            action_summary.append(f"Extracted {len(output_files)} output files")

        # 6. Run verifier over normalized SDK events and extracted files.
        score, v_reasoning, structured_response, v_input, v_output = await _run_verifier(
            agent_id=agent_id,
            task_id=task_id,
            goal=goal,
            model=model,
            terminal=terminal,
            action_summary=action_summary,
            output_schema=output_schema,
            claude_token=claude_token,
            gpt_token=gpt_token,
            gpt_account_id=gpt_account_id,
            gpt_token_source=gpt_token_source,
            gpt_token_pool_id=gpt_token_pool_id,
            gpt_token_user_id=user_id if gpt_token else None,
            fast_mode_cache_enabled=fast_mode_cache_enabled,
            fast_mode_goal=goal,
            fast_mode_step=0,
            attempt_completion=attempt_completion,
            exit_code=exit_code,
            code_provider=code_provider,
            code_events=code_events,
            output_files=output_files,
            handoff=close_data["handoff"] if "handoff" in close_data else _NO_HANDOFF_CONTRACT,
        )
        total_input_tokens += v_input
        total_output_tokens += v_output
        # Judged before file-pointer resolution rewrites the response below.
        verified = code_agent_verifier_enabled() or (
            _disabled_verifier_outcome(
                output_schema,
                structured_response,
                attempt_completion=attempt_completion,
                exit_code=exit_code,
            )[0]
            != "unverified"
        )

        # 7. Add any verifier-selected extra paths to the structured response
        # contract, but do not use them for sandbox extraction. The canonical
        # artifact source is /home/user/output extracted before verification.
        additional_paths = extract_additional_paths(structured_response, output_schema)
        if additional_paths:
            logger.info(
                "[ClaudeCode] Verifier returned remote file paths",
                agent_id=agent_id,
                task_id=task_id,
                paths=additional_paths,
            )

        # 8. Resolve file pointers
        if structured_response and output_files:
            structured_response = resolve_file_pointers_from_extracted(
                structured_response=structured_response,
                output_schema=output_schema,
                extracted_files=output_files,
            )
            logger.info(
                "[ClaudeCode] Resolved file pointers in structured response", agent_id=agent_id, task_id=task_id
            )

        # Build result. Threshold is 1.0 to match the unified verifier semantics
        # (B2/C12): the rest of the system treats anything below 1.0 as a
        # rejection, and `build_context_summary` reports "✅ SUCCESS" only when
        # `success=True`, so we don't lie to the parent.
        success = score >= 1.0
        context_message = build_context_summary(
            success=success,
            close_reasoning=v_reasoning,
            action_summary=action_summary,
            verified=verified,
        )
        code_session_summary = format_code_session_summary(code_session_facts)
        if code_session_summary:
            context_message = f"{context_message}\n\n{code_session_summary}"

        # Log Claude Code LLM usage (not charged)
        if claude_code_usage:
            logger.info(
                "[ClaudeCode] Claude Code LLM usage (not charged, user's plan)",
                input_tokens=claude_code_usage.get("inputTokens", 0),
                output_tokens=claude_code_usage.get("outputTokens", 0),
                cache_read=claude_code_usage.get("cacheReadInputTokens", 0),
                cache_create=claude_code_usage.get("cacheCreationInputTokens", 0),
            )

        execution_time_ms = (safe_time() - start_time) * 1000
        usage_items = build_usage_items(
            total_input_tokens,
            total_output_tokens,
            model,
            model_pricing,
            execution_time_ms,
            claude_code_usage,
            code_agent_meta=code_agent_meta,
        )

        if log_collector:
            await log_collector.stop()

        result = {
            "status": "completed",
            "success": success,
            "message": context_message,
            "task_id": task_id,
            "ws_url": ws_url,
            "execution_time_ms": execution_time_ms,
            "usage_items": usage_items,
            "structured_response": structured_response,
            "output_files": output_files,
            "code_provider": code_provider,
            "completed_turns": codex_completed_turn_count(code_events) if code_provider == "codex" else None,
            "code_session": code_session_facts,
            "code_agent": build_code_agent_metadata(
                target_meta=code_agent_meta,
                task_id=task_id,
                provider=code_provider,
                state="completed",
                started_at=start_time,
                ended_at=safe_time(),
                facts=code_session_facts,
            ),
        }
        return result

    except asyncio.CancelledError:
        cancellation = activity.cancellation_details()
        logger.info(
            "[ClaudeCode] Activity cancelled",
            agent_id=agent_id,
            task_id=task_id,
            cancellation=repr(cancellation),
        )

        if log_collector:
            await log_collector.stop()

        if cancellation is not None and not cancellation.cancel_requested:
            # Not a cancel the workflow asked for, so the run is not reported as
            # cancelled. The sandbox stays open only for an attempt that will
            # reattach to it from this attempt's heartbeats.
            if task_id and not _next_attempt_reattaches(cancellation):
                logger.info(
                    "[ClaudeCode] No attempt will reattach; closing the sandbox",
                    agent_id=agent_id,
                    task_id=task_id,
                    cancellation=repr(cancellation),
                )
                try:
                    await close_e2b_session(task_id=task_id, code_provider=code_provider)
                except Exception as close_error:
                    logger.warning(
                        "[ClaudeCode] Closing the abandoned sandbox failed",
                        agent_id=agent_id,
                        task_id=task_id,
                        error=str(close_error),
                    )
            raise

        # A user pause is resumable: the chip ends paused, never failed.
        paused_code_agent = build_code_agent_metadata(
            target_meta=code_agent_meta,
            task_id=task_id,
            provider=code_provider,
            state="paused",
            started_at=start_time,
            ended_at=safe_time(),
            facts=code_session_facts,
        )
        if task_id:
            try:
                # Persist terminal output before closing (skip if already written)
                try:
                    if not terminal_output_persisted_box[0]:
                        cancelled_output = build_durable_terminal_transcript(
                            terminal.get_display(),
                            code_events,
                            action_summary,
                        )
                        if cancelled_output:
                            await persist_terminal_output(task_id, cancelled_output, agent_id, code_provider)
                            terminal_output_persisted_box[0] = True
                            logger.info(
                                "[ClaudeCode] Persisted terminal output on cancel",
                                agent_id=agent_id,
                                task_id=task_id,
                                output_len=len(cancelled_output),
                            )
                except Exception as output_error:
                    logger.warning(
                        "[ClaudeCode] Failed to persist output on cancel",
                        agent_id=agent_id,
                        error=str(output_error),
                    )

                await save_claude_code_message(
                    agent_id=agent_id,
                    title="Session Cancelled",
                    task_id=task_id,
                    message="Agent paused by user",
                    status=MessageStatus.FINISHED,
                    action_type="task_complete",
                    tool_call_id=tool_call_id,
                    code_provider=code_provider,
                    code_agent=paused_code_agent,
                )

                await close_e2b_session(task_id=task_id, code_provider=code_provider, reason="cancel")
            except Exception as cleanup_error:
                logger.warning(
                    "[ClaudeCode] Cleanup failed on cancel",
                    agent_id=agent_id,
                    task_id=task_id,
                    error=str(cleanup_error),
                )

        return {
            "status": "cancelled",
            "task_id": task_id,
            "message": "Claude Code agent cancelled by user",
            "code_provider": code_provider,
            "code_session": code_session_facts,
            "code_agent": paused_code_agent,
            "usage_items": _sandbox_usage_items(code_agent_meta, start_time) if task_id else [],
        }

    except Exception as e:
        retry_state[CODE_AGENT_ERRORS_HEARTBEAT_KEY] += 1
        error = _bounded_activity_error(e, retry_state[CODE_AGENT_ERRORS_HEARTBEAT_KEY])
        final = _error_ends_the_run(error)
        logger.error(
            "[ClaudeCode] Automation failed",
            agent_id=agent_id,
            task_id=task_id,
            error=extract_error_message(e),
            error_type=type(e).__name__,
            activity_errors=retry_state[CODE_AGENT_ERRORS_HEARTBEAT_KEY],
            final=final,
        )

        if log_collector:
            await log_collector.stop()

        if not final:
            # Temporal retries this attempt. The sandbox stays live and rides on
            # this last heartbeat (with the error count), so the retry reattaches
            # to the running CLI instead of pausing it and restarting the goal.
            safe_heartbeat("Code agent error; Temporal retries the attempt")
            raise e

        error_context = f"Failed during: {action_summary[-1] if action_summary else 'initialization'}"
        error_message = build_context_summary(
            success=False,
            close_reasoning=extract_error_message(e),
            action_summary=action_summary,
            error_context=error_context,
        )

        if task_id:
            try:
                try:
                    if not terminal_output_persisted_box[0]:
                        failed_output = build_durable_terminal_transcript(
                            terminal.get_display(),
                            code_events,
                            action_summary,
                        )
                        if failed_output:
                            await persist_terminal_output(
                                task_id,
                                failed_output,
                                agent_id,
                                code_provider,
                            )
                            terminal_output_persisted_box[0] = True
                            logger.info(
                                "[ClaudeCode] Persisted terminal output on failure",
                                agent_id=agent_id,
                                task_id=task_id,
                                output_len=len(failed_output),
                            )
                except Exception as output_error:
                    logger.warning(
                        "[ClaudeCode] Failed to persist output on failure",
                        agent_id=agent_id,
                        error=str(output_error),
                    )

                await save_claude_code_message(
                    agent_id=agent_id,
                    title="Task Failed",
                    task_id=task_id,
                    message=error_message,
                    status=MessageStatus.ERROR,
                    action_type="task_complete",
                    tool_call_id=tool_call_id,
                    code_provider=code_provider,
                    code_agent=build_code_agent_metadata(
                        target_meta=code_agent_meta,
                        task_id=task_id,
                        provider=code_provider,
                        state="failed",
                        started_at=start_time,
                        ended_at=safe_time(),
                        facts=code_session_facts,
                        failure_message=extract_error_message(e),
                    ),
                )

                # The failure close checkpoints the sandbox (repos, files, CLI log)
                # and pauses it; its facts name the checkpoint the resume restores.
                close_data = await close_e2b_session(task_id=task_id, code_provider=code_provider, reason="failure")
                code_session_facts = _merge_close_code_session_facts(code_session_facts, close_data)
            except Exception as cleanup_error:
                logger.warning(
                    "[ClaudeCode] Cleanup failed on error",
                    agent_id=agent_id,
                    task_id=task_id,
                    error=str(cleanup_error),
                )

        raise code_session_failure_error(
            error, e, code_session_facts, local=bool(local_workspace_place(code_agent_meta))
        )

    finally:
        session_heartbeats.close()
        retry_heartbeats.close()
