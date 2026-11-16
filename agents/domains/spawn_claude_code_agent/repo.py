"""Spawn-claude-code-agent domain data access — HTTP calls to back-end API."""

from __future__ import annotations

import asyncio
import random
import time
import uuid as uuid_module
from typing import Any, Optional

import httpx
import structlog
from temporalio.exceptions import ApplicationError

from config import load_config
from domains.spawn_claude_code_agent.config import (
    CODE_SESSION_CREATE_HEARTBEAT_S,
    CODE_SESSION_CREATE_TIMEOUT_S,
    CODE_SESSION_HELD_BACKOFF_S,
    CODE_SESSION_HELD_CODE,
    CODE_SESSION_HELD_WAIT_S,
    CODE_SESSION_RESIZE_ATTEMPTS,
    CODE_SESSION_RESIZE_RETRY_BACKOFF_S,
    CODE_SESSION_RESIZE_TIMEOUT_S,
    E2B_CAPACITY_EXHAUSTED_CODE,
    GITHUB_RATE_LIMITED_CODE,
    LOCAL_TARGET_UNAVAILABLE_CODE,
    SANDBOX_SIZE_UNSUPPORTED_CODE,
)
from helpers.activity_heartbeat import safe_heartbeat
from helpers.github_repo_ref import GITHUB_REPO_REF_DENIED_CODE, repo_ref_denied_message
from helpers.http_client import get_http_client
from helpers.workflow_generation import (
    WorkflowGenerationSealed,
    raise_if_generation_sealed,
    stamp_tool_workflow_generation,
    workflow_generation_headers,
)
from tools.base import MessageStatus
from utils.error_utils import extract_error_message

logger = structlog.get_logger()


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _session_base_path(code_provider: str = "claude") -> str:
    if code_provider == "codex":
        return "/api/codex-code-session"
    return "/api/claude-code-session"


async def save_claude_code_message(
    agent_id: str,
    title: str,
    message: str,
    status: MessageStatus,
    task_id: str,
    action_type: str = "default",
    tool_call_id: Optional[str] = None,
    is_init_message: bool = False,
    ws_url: Optional[str] = None,
    output_schema: Optional[dict] = None,
    structured_response: Optional[dict] = None,
    github_repo_url: Optional[str] = None,
    code_provider: str = "claude",
    completed_turns: Optional[int] = None,
    tool_name: Optional[str] = None,
    agent_type: Optional[str] = None,
    steps_taken: Optional[int] = None,
    sdk_event_range: Optional[tuple[int, int]] = None,
    code_session: Optional[dict[str, Any]] = None,
    code_agent: Optional[dict[str, Any]] = None,
) -> None:
    """Save Claude Code sub-action message directly to DB via back-end API.

    ``code_session`` (persistent session facts: id, branch, merge target, merge
    facts) rides on every message's metadata, so each trajectory entry of a
    code agent reports its branch. ``code_agent`` merges into
    ``metadata.code_agent`` (UI contract: the local-target lane owns
    ``target``/``device_name``/``folder``; the code-session lane owns
    ``branch``/``base_branch``/``state``/``summary``/``failure_message``).
    """
    config = load_config()
    unique_tool_call_id = tool_call_id or str(uuid_module.uuid4())

    metadata: dict[str, Any] = {
        "toolCallId": unique_tool_call_id,
        "toolName": tool_name or "spawn_claude_code_agent",
        "toolStatus": status.value,
        "title": title,
        "task_id": task_id,
        "action_type": action_type,
        "code_provider": code_provider,
    }

    if ws_url:
        metadata["ws_url"] = ws_url

    # Include output schema in init message for FE display
    if output_schema:
        metadata["output_schema"] = output_schema

    # Include structured response in task_complete message for FE display
    if structured_response:
        metadata["structured_response"] = structured_response

    if github_repo_url:
        metadata["github_repo_url"] = github_repo_url

    if completed_turns is not None:
        metadata["completed_turns"] = max(0, int(completed_turns))

    # Non-orchestration evidence chips (SDK events / history) need an explicit
    # agentType so harness `_agent_class_step_counts` attributes them to `code`.
    if agent_type:
        metadata["agentType"] = agent_type

    if steps_taken is not None:
        metadata["steps_taken"] = max(0, int(steps_taken))

    # Judge-evidence pages: [start, end) of the session's evidence events this
    # row holds, so a reader can tell a complete set of rows from a missing one.
    if sdk_event_range is not None:
        metadata["sdk_event_range"] = [int(sdk_event_range[0]), int(sdk_event_range[1])]

    if code_session:
        metadata["code_session"] = code_session
    # UI contract (rose-user-ui): {task_id, provider, target, device_name?,
    # folder?, branch, base_branch, state, started_at, ended_at?,
    # failure_message?, summary?{commit, branch, merged_into?, files}}.
    if code_agent:
        existing = metadata.get("code_agent")
        metadata["code_agent"] = {**(existing if isinstance(existing, dict) else {}), **code_agent}
    # Intermediate chip rows (progress, task_started, system) are child-scoped
    # UI and must stay out of the parent LLM context via subAgentId.
    # Parent-facing task_complete terminals reuse the spawn toolCallId — if they
    # are also marked subAgentId, drop_rows_never_sent_to_model strips them and
    # the ASSISTANT tool_call is left unpaired (role_alternation). That happens
    # on cancel/DAG-cancel when the handler never yields a twin ERROR event and
    # Session Cancelled from this helper is the only terminal for the toolCallId.
    if not is_init_message and action_type != "task_complete":
        metadata["subAgentId"] = task_id

    if status == MessageStatus.ERROR:
        metadata["error"] = True
    metadata = stamp_tool_workflow_generation(metadata) or metadata

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{config.backend.api_url}/api/agents/{agent_id}/messages",
                headers={
                    "x-api-key": config.backend.api_key or "",
                    "Content-Type": "application/json",
                    **workflow_generation_headers(),
                },
                json={
                    "message": message,
                    "role": "TOOL",
                    "metadata": metadata,
                },
            )
            raise_if_generation_sealed(response.status_code, "spawn_claude_code_agent")
            logger.info(
                "[ClaudeCode] Saved message",
                agent_id=agent_id,
                task_id=task_id,
                title=title,
                status=status.value,
                response_status=response.status_code,
            )
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as e:
        logger.error(
            "[ClaudeCode] Failed to save message",
            agent_id=agent_id,
            task_id=task_id,
            error=extract_error_message(e),
        )


async def save_code_transcript_batch(
    *,
    agent_id: str,
    task_id: str,
    tool_call_id: str,
    code_provider: str,
    records: list[dict[str, Any]],
    events_from: int,
    events_to: int,
    text: str,
) -> bool:
    """Persist one batch of structured transcript records on the code agent's rows.

    The row is child-scoped (``subAgentId``) like the chip's progress rows, so
    it never enters the parent planner's context. It carries no ``toolStatus``
    (it is not a lifecycle transition); ``isCodeTranscript`` +
    ``idempotencyKey`` give it a deterministic id, so an activity retry that
    re-flushes the same event range writes nothing new. Returns False on a
    transient failure so the caller keeps the batch and retries it.
    """
    config = load_config()
    metadata: dict[str, Any] = {
        "toolCallId": tool_call_id,
        "toolName": "spawn_claude_code_agent",
        "title": "Transcript",
        "task_id": task_id,
        "action_type": "code_transcript",
        "code_provider": code_provider,
        "subAgentId": task_id,
        "isCodeTranscript": True,
        "idempotencyKey": f"code-transcript:{task_id}:{events_from}-{events_to}",
        # UI contract: parsed transcript events for the Transcript tab, each
        # {type, ts, text?, tool?{name, target, added?, removed?, result?}}
        # plus upsert keys (id, rev, seq). Keep the latest rev per id.
        "code_agent_events": records,
        "transcript": {
            "version": 1,
            "task_id": task_id,
            "provider": code_provider,
            "events": [events_from, events_to],
        },
    }
    metadata = stamp_tool_workflow_generation(metadata) or metadata
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{config.backend.api_url}/api/agents/{agent_id}/messages",
                headers={
                    "x-api-key": config.backend.api_key or "",
                    "Content-Type": "application/json",
                    **workflow_generation_headers(),
                },
                json={"message": text, "role": "TOOL", "metadata": metadata},
            )
            raise_if_generation_sealed(response.status_code, "spawn_claude_code_agent")
            ok = response.status_code in (200, 201)
            logger.info(
                "[ClaudeCode] Saved transcript batch",
                agent_id=agent_id,
                task_id=task_id,
                records=len(records),
                events_from=events_from,
                events_to=events_to,
                response_status=response.status_code,
            )
            return ok
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as e:
        logger.warning(
            "[ClaudeCode] Failed to save transcript batch",
            agent_id=agent_id,
            task_id=task_id,
            error=extract_error_message(e),
        )
        return False


async def _post_with_heartbeats(request: Any, message: str) -> httpx.Response:
    """Await a long backend request, heartbeating the activity while it runs.

    A session create can take minutes (CODE_SESSION_CREATE_TIMEOUT_S); without
    beats the heartbeat timeout would start a second attempt beside this one.
    """
    pending = asyncio.ensure_future(request)
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=CODE_SESSION_CREATE_HEARTBEAT_S)
            if done:
                return pending.result()
            safe_heartbeat(message)
    finally:
        if not pending.done():
            pending.cancel()


def _is_held_response(response: httpx.Response) -> bool:
    return response.status_code == 409 and _json_or_empty(response).get("code") == CODE_SESSION_HELD_CODE


async def _create_waiting_out_direct_hold(post: Any, agent_id: str) -> httpx.Response:
    """POST the create; while the backend answers CODE_SESSION_HELD, wait for the hand back and repost.

    The live CLI's user is chatting with it directly and its ws-server cannot hold an attaching spawn, so
    the backend refuses without touching that CLI. No task row or session is created (the backend releases
    the lease), so a repost is the same create and the tool call's lease keeps it one spawn; each attempt
    only re-stages the sandbox's files idempotently. The wait heartbeats and backs off (capped, with
    +/-20% jitter so spawns held at the same moment don't repost in lockstep) for up to
    CODE_SESSION_HELD_WAIT_S, which covers the ws-server's 10-minute idle release; past it the spawn fails
    non-retryably with the true reason.
    """
    started = time.monotonic()
    delay, max_delay = CODE_SESSION_HELD_BACKOFF_S
    response = await post()
    while _is_held_response(response):
        waited = time.monotonic() - started
        if waited >= CODE_SESSION_HELD_WAIT_S:
            raise ApplicationError(
                "The code agent's session is still held: the user is still chatting with it directly; "
                "spawn it again after they hand it back.",
                type=CODE_SESSION_HELD_CODE,
                non_retryable=True,
            )
        logger.info(
            "[ClaudeCode] Code session held by the user's direct chat; waiting for the hand back",
            agent_id=agent_id,
            waited_s=round(waited, 1),
            retry_in_s=delay,
        )
        remaining = delay * random.uniform(0.8, 1.2)
        while remaining > 0:
            safe_heartbeat("Waiting for the user to hand the code agent back")
            step = min(remaining, 5.0)
            await asyncio.sleep(step)
            remaining -= step
        delay = min(delay * 2, max_delay)
        response = await post()
    return response


async def create_e2b_session(
    agent_id: str,
    goal: str,
    user_id: str,
    attachments: Optional[list[dict[str, Any]]] = None,
    github_repo_url: Optional[str] = None,
    code_provider: str = "claude",
    fast_mode_cache_enabled: bool = False,
    code_session_id: Optional[str] = None,
    lease_owner: Optional[str] = None,
    github_repo_ref: Optional[str] = None,
    target: str = "cloud",
    local_device_id: Optional[str] = None,
    fresh_conversation: bool = False,
    task_name: Optional[str] = None,
    cli_settings: Optional[dict[str, Any]] = None,
    resume_config: Optional[dict[str, Any]] = None,
    sandbox_size: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Create an E2B sandbox session via back-end API.

    ``sandbox_size`` (``{cpu?, memory_gb?}``) is the root's initial size for a new
    session; a size no built template within the user's cap satisfies answers 400
    ``SANDBOX_SIZE_UNSUPPORTED``, raised non-retryable with the available sizes.

    ``code_session_id`` resumes that persistent code session; omitted, the
    backend resumes the spawning agent's latest session for the repo.
    ``lease_owner`` is stable per tool call so Temporal retries keep the lease.
    ``github_repo_ref`` is the spawn's explicit connected-repo selection: it replaces
    the project restore (backend resolveCodeSessionWorkspace). A resume by
    ``code_session_id`` keeps that session's own repo, so the ref is not sent with it.
    A ref the user has no connection for answers 403 ``GITHUB_ACCESS_DENIED``, raised
    as a non-retryable ``GITHUB_REPO_REF_DENIED`` naming the ref (never a fallback).

    Returns the session data dict with task_id, sandbox_id, ws_url,
    code_session_id, branch, merge_target, resumed_from, etc.
    Raises on HTTP failure. ``target="local"`` asks the backend to start the
    same ws-server on the agent's Mac instead (no persistent code session: the
    work stays in the folder the user granted); an unavailable Mac answers 409
    ``LOCAL_TARGET_UNAVAILABLE``, raised as a non-retryable error carrying the
    exact user-facing text (no retry, no cloud fallback).
    """
    config = load_config()

    create_payload: dict[str, Any] = {
        "agent_id": agent_id,
        "goal": goal,
        "user_id": user_id,
        "fast_mode_cache_enabled": fast_mode_cache_enabled,
    }
    if attachments:
        create_payload["attachments"] = attachments
    if task_name:
        # The agents list shows the spawn's task_name as the code agent's title.
        create_payload["task_name"] = task_name[:200]
    if cli_settings:
        # A resuming turn's model / effort / speed (backend cliSettingsEnvOverrides).
        create_payload["cli_settings"] = cli_settings
    if resume_config is not None:
        create_payload["resume_config"] = resume_config
    if github_repo_url and target != "local":
        create_payload["github_repo_url"] = github_repo_url
    if target == "local":
        create_payload["target"] = "local"
        if local_device_id:
            create_payload["device_id"] = local_device_id
    else:
        if code_session_id:
            create_payload["code_session_id"] = code_session_id
        elif github_repo_ref:
            create_payload["github_repo_ref"] = github_repo_ref
        if lease_owner:
            create_payload["lease_owner"] = lease_owner[:200]
        if fresh_conversation:
            # Keep the session's sandbox and files; start a new CLI conversation.
            create_payload["fresh_conversation"] = True
        if sandbox_size:
            create_payload["sandbox_size"] = sandbox_size

    async with httpx.AsyncClient(timeout=httpx.Timeout(CODE_SESSION_CREATE_TIMEOUT_S, connect=10.0)) as client:

        async def post_create() -> httpx.Response:
            return await _post_with_heartbeats(
                client.post(
                    f"{config.backend.api_url}{_session_base_path(code_provider)}/create",
                    headers={
                        "x-api-key": config.backend.api_key or "",
                        "Content-Type": "application/json",
                    },
                    json=create_payload,
                ),
                "Starting the code-agent sandbox",
            )

        response = await _create_waiting_out_direct_hold(post_create, agent_id)

        if response.status_code == 409 and target == "local":
            body = _json_or_empty(response)
            if body.get("code") == LOCAL_TARGET_UNAVAILABLE_CODE and isinstance(body.get("error"), str):
                raise ApplicationError(
                    body["error"],
                    {"reason": body.get("reason"), "device_name": body.get("device_name")},
                    type=LOCAL_TARGET_UNAVAILABLE_CODE,
                    non_retryable=True,
                )

        if response.status_code == 403 and "github_repo_ref" in create_payload:
            body = _json_or_empty(response)
            if body.get("code") == "GITHUB_ACCESS_DENIED":
                # The explicit selection is not connected (or was disconnected): fail
                # naming it; neither a retry nor the project's repo can stand in for it.
                raise ApplicationError(
                    repo_ref_denied_message(create_payload["github_repo_ref"], str(body.get("error") or "")),
                    type=GITHUB_REPO_REF_DENIED_CODE,
                    non_retryable=True,
                )

        if response.status_code == 400:
            body = _json_or_empty(response)
            if body.get("code") == SANDBOX_SIZE_UNSUPPORTED_CODE and isinstance(body.get("error"), str):
                raise ApplicationError(
                    body["error"],
                    {"requested": body.get("requested"), "available": body.get("available")},
                    type=SANDBOX_SIZE_UNSUPPORTED_CODE,
                    non_retryable=True,
                )

        if response.status_code == 503:
            body = _json_or_empty(response)
            if body.get("code") == E2B_CAPACITY_EXHAUSTED_CODE and isinstance(body.get("error"), str):
                raise ApplicationError(body["error"], type=E2B_CAPACITY_EXHAUSTED_CODE, non_retryable=True)

        if response.status_code == 429:
            body = _json_or_empty(response)
            if body.get("code") == GITHUB_RATE_LIMITED_CODE and isinstance(body.get("error"), str):
                # Retrying inside GitHub's limit only spends more of it; the message
                # carries the reset time.
                raise ApplicationError(body["error"], type=GITHUB_RATE_LIMITED_CODE, non_retryable=True)

        if response.status_code not in (200, 201):
            response_text = response.text[:500]
            logger.error(
                "[ClaudeCode] Failed to create code session",
                agent_id=agent_id,
                code_provider=code_provider,
                status_code=response.status_code,
                response_text=response_text,
            )
            detail = f": {response_text}" if response_text else ""
            message = f"Failed to create code session (provider={code_provider}): HTTP {response.status_code}{detail}"
            upper = f"{message} {response_text}".upper()
            billing_markers = (
                "MISSING PAYMENT METHOD",
                "TEAM IS BLOCKED",
                "CREDIT BALANCE IS TOO LOW",
                "TOO LOW TO ACCESS THE ANTHROPIC API",
                "YOUR CREDIT BALANCE IS TOO LOW",
            )
            if any(m in upper for m in billing_markers):
                # Permanent billing blocks must not burn activity retries —
                # the tool manager soft-fails and the agent continues.
                raise ApplicationError(
                    message,
                    type="ClaudeCodeBillingBlocked",
                    non_retryable=True,
                )
            if response.status_code in (404, 409) and (
                "CODE_SESSION_NOT_FOUND" in upper or "CODE_SESSION_BUSY" in upper
            ):
                # An unknown/foreign code_session_id or a session another agent
                # holds does not heal on retry; the tool reports it instead.
                raise ApplicationError(message, type="CodeSessionUnavailable", non_retryable=True)
            raise Exception(message)

        return response.json()


async def report_undelivered_code_turns(agent_id: str, task_id: str, turn_ids: list[str], reason: str) -> None:
    """Tell the senders of agent turns this code session accepted but lost (``user_turn_undelivered``).

    The backend notifies each sending Rose agent once per turn (idempotent on the original message key),
    so a retry or a replayed event is safe. Best effort: a failure is logged, never raised into the monitor.
    """
    if not turn_ids:
        return
    config = load_config()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config.backend.api_url}/api/agents/{agent_id}/code-turns-undelivered",
                headers={"x-api-key": config.backend.api_key or ""},
                json={"session_id": task_id, "turn_ids": turn_ids[:50], "reason": (reason or "rejected")[:100]},
            )
            response.raise_for_status()
        logger.info(
            "[ClaudeCode] Reported undelivered agent turns",
            agent_id=agent_id,
            task_id=task_id,
            turns=len(turn_ids),
            notified=_json_or_empty(response).get("notified"),
        )
    except Exception as e:
        logger.warning(
            "[ClaudeCode] Could not report undelivered agent turns",
            agent_id=agent_id,
            task_id=task_id,
            error=extract_error_message(e),
        )


async def report_code_session_subscription_usage(task_id: str, usage: dict[str, Any]) -> bool:
    """Record a code session's subscription usage (from the CLI's rate_limit_event).

    Code sessions run the provider CLI in a sandbox, outside the lineage lease,
    so their usage goes to the task's credential row by task id. Best-effort and
    bounded (5 s); callers fire it in the background. Lives here, beside the
    other code-session reports, so importing the code-session domain never pulls
    in the LLM client stack (helpers.llm → openai).
    """
    if not task_id or not usage:
        return False
    config = load_config()
    try:
        client = get_http_client()
        response = await client.post(
            f"{config.backend.api_url}/api/integrations/compute-code/code-session-usage",
            headers={
                "X-API-Key": config.backend.api_key or "",
                "Content-Type": "application/json",
            },
            json={"taskId": task_id, "usage": usage},
            timeout=5.0,
        )
        return response.status_code == 200
    except Exception as e:
        logger.warning("[ClaudeCode] Code session usage report failed", task_id=task_id, error=extract_error_message(e))
        return False


async def persist_terminal_output(
    task_id: str,
    output: str,
    agent_id: str,
    code_provider: str = "claude",
    credential_feedback: Optional[list[dict[str, str]]] = None,
) -> None:
    """Persist filtered terminal output to DB."""
    config = load_config()
    body: dict[str, Any] = {"output": output}
    if credential_feedback is not None:
        body["credential_feedback"] = credential_feedback
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            output_response = await client.post(
                f"{config.backend.api_url}{_session_base_path(code_provider)}/{task_id}/output",
                headers={"x-api-key": config.backend.api_key or ""},
                json=body,
            )
            logger.info(
                "[ClaudeCode] Output persisted",
                agent_id=agent_id,
                task_id=task_id,
                output_len=len(output),
                response_status=output_response.status_code,
            )
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as e:
        logger.warning(
            "[ClaudeCode] Failed to persist output",
            agent_id=agent_id,
            task_id=task_id,
            error=extract_error_message(e),
        )


async def close_e2b_session(
    task_id: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    additional_paths: Optional[list[str]] = None,
    code_provider: str = "claude",
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Close the E2B sandbox session and extract output files.

    A persistent code session is checkpointed and paused (never killed);
    ``reason`` (close | failure | cancel | idle | drain) is recorded on the
    checkpoint. Returns dict with output_files, usage and ``code_session``
    facts (branch, merge facts, checkpoint). A lost response retries this same
    task's idempotent close; it never restarts the code agent.
    """
    config = load_config()
    payload: dict[str, Any] = {"task_id": task_id}
    if reason:
        payload["reason"] = reason
    if user_id:
        payload["user_id"] = user_id
    if agent_id:
        payload["agent_id"] = agent_id
    if additional_paths:
        payload["additional_paths"] = additional_paths

    async def reconcile_close() -> dict[str, Any]:
        last_error = "close was not confirmed"
        # Each request retains the existing transport bound. The backend may
        # still be closing after a lost response; its task receipt makes the
        # next request join that operation or replay its completed result.
        async with httpx.AsyncClient(timeout=180.0) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        f"{config.backend.api_url}{_session_base_path(code_provider)}/close",
                        headers={"x-api-key": config.backend.api_key or ""},
                        json=payload,
                    )
                    raise_if_generation_sealed(response.status_code, "spawn_claude_code_agent")
                    if response.status_code == 200:
                        result = response.json()
                        if not isinstance(result, dict) or not isinstance(result.get("output_files"), list):
                            raise ValueError("close response has no output_files result")
                        return result
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code not in (408, 429, 500, 502, 503, 504):
                        break
                except (httpx.TransportError, ValueError) as error:
                    last_error = extract_error_message(error) or type(error).__name__
                logger.warning(
                    "[ClaudeCode] Close unconfirmed; reconciling the same task",
                    task_id=task_id,
                    attempt=attempt + 1,
                    error=last_error,
                )
                if attempt < 2:
                    await asyncio.sleep(0.25 * (attempt + 1))
        # Generic activity retry reattaches to the CLI. This failure belongs
        # only to close, so it must not trigger that path or report empty output.
        raise ApplicationError(
            f"Code session close remains unconfirmed for task {task_id}: {last_error}",
            type="CodeSessionCloseUnconfirmed",
            non_retryable=True,
        )

    return await _post_with_heartbeats(reconcile_close(), "Closing the code-agent session")


async def keepalive_e2b_session(task_id: str, code_provider: str = "claude") -> Optional[dict[str, Any]]:
    """Extend an attached session's sandbox idle window and lease. Best-effort.

    Returns the backend's answer (``alive``, ``resize_requested``) or None when
    the call failed; a failed keepalive never ends the session.
    """
    config = load_config()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config.backend.api_url}{_session_base_path(code_provider)}/{task_id}/keepalive",
                headers={"x-api-key": config.backend.api_key or ""},
                json={},
            )
            if response.status_code != 200:
                return None
            body = response.json()
            return body if isinstance(body, dict) else None
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as e:
        logger.warning("[ClaudeCode] Keepalive failed", task_id=task_id, error=extract_error_message(e))
        return None


async def report_code_usage_limit(
    task_id: str, code_provider: str, resets_at: Optional[Any]
) -> Optional[dict[str, Any]]:
    """Rate-limit the credential this task ran on until the provider's reset (C15)."""
    config = load_config()
    body: dict[str, Any] = {}
    if isinstance(resets_at, (int, float, str)) and resets_at:
        body["resets_at"] = resets_at
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config.backend.api_url}{_session_base_path(code_provider)}/{task_id}/usage-limit",
                headers={"x-api-key": config.backend.api_key or ""},
                json=body,
            )
            if response.status_code != 200:
                logger.warning("[ClaudeCode] Usage-limit report failed", task_id=task_id, status=response.status_code)
                return None
            result = response.json()
            return result if isinstance(result, dict) else None
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as e:
        logger.warning("[ClaudeCode] Usage-limit report failed", task_id=task_id, error=extract_error_message(e))
        return None


async def resize_e2b_session(task_id: str, request_id: str, code_provider: str = "claude") -> Optional[dict[str, Any]]:
    """Apply a pending sandbox size change (C16); returns the task that continues the session.

    The backend checkpoints the session, moves it into a sandbox of the new size and
    resumes the CLI conversation there. Idempotent per ``request_id``: a retry after a
    lost response returns the same task. A concurrent apply (409 RESIZE_IN_PROGRESS)
    is waited out; any other failure returns None and the session stays where it is.
    """
    config = load_config()
    url = f"{config.backend.api_url}{_session_base_path(code_provider)}/{task_id}/resize"
    for attempt in range(1, CODE_SESSION_RESIZE_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=CODE_SESSION_RESIZE_TIMEOUT_S) as client:
                response = await client.post(
                    url,
                    headers={"x-api-key": config.backend.api_key or ""},
                    json={"request_id": request_id},
                )
            body = response.json() if response.content else {}
            if response.status_code == 200 and isinstance(body, dict):
                return body
            code = body.get("code") if isinstance(body, dict) else None
            retryable = response.status_code >= 500 or code == "RESIZE_IN_PROGRESS"
            logger.warning(
                "[ClaudeCode] Sandbox resize call failed",
                task_id=task_id,
                request_id=request_id,
                status=response.status_code,
                code=code,
                attempt=attempt,
            )
            if not retryable:
                return None
        except WorkflowGenerationSealed:
            raise  # a replaced generation must stop, never be swallowed
        except Exception as e:
            logger.warning(
                "[ClaudeCode] Sandbox resize call failed",
                task_id=task_id,
                request_id=request_id,
                error=extract_error_message(e),
                attempt=attempt,
            )
        await asyncio.sleep(CODE_SESSION_RESIZE_RETRY_BACKOFF_S * attempt)
    return None


async def fetch_code_credential_availability(
    user_id: str,
    own_only: bool = False,
    code_session_id: Optional[str] = None,
    prefer: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Whether a Claude / Codex credential can be leased right now (no lease taken).

    ``own_only``: count only the user's own logins (a local Mac run never gets a
    pool token or a funded key). ``next_credential``: the provider (and tier)
    the spawn takes in the backend's C15 order, every subscription before any
    funded key; ``None`` when nothing is leasable (absent from an older backend).
    ``code_session_id`` (a retry or a resume): also the session's own provider
    (``session_provider``, first in that order) and the stored ``outlook``
    (accounts, shared paths, earliest reset).
    """
    config = load_config()
    params = {"user_id": user_id}
    if own_only:
        params["own_only"] = "true"
    if code_session_id:
        params["code_session_id"] = code_session_id
    if prefer in ("claude", "codex"):
        params["prefer"] = prefer
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{config.backend.api_url}/api/claude-code-session/credential-availability",
                headers={"x-api-key": config.backend.api_key or ""},
                params=params,
            )
            if response.status_code != 200:
                return None
            body = response.json()
            if not isinstance(body, dict):
                return None
            answer: dict[str, Any] = {provider: body.get(provider) is True for provider in ("claude", "codex")}
            if "next_credential" in body:
                step = body.get("next_credential")
                answer["next_credential"] = (
                    step if isinstance(step, dict) and step.get("provider") in ("claude", "codex") else None
                )
            if body.get("session_provider") in ("claude", "codex"):
                answer["session_provider"] = body["session_provider"]
            if isinstance(body.get("outlook"), dict):
                answer["outlook"] = body["outlook"]
            return answer
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as e:
        logger.warning("[ClaudeCode] Credential availability lookup failed", error=extract_error_message(e))
        return None


async def fetch_local_agent_target(agent_id: str) -> dict[str, Any]:
    """Ask the backend whether ``agent_id`` may run a code agent on its Mac now.

    ``GET /api/local-devices/agent-target/:agentId`` answers from the agent
    tree's recorded origin and the Mac's live control channel; it never waits.
    """
    config = load_config()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{config.backend.api_url}/api/local-devices/agent-target/{agent_id}",
            headers={"x-api-key": config.backend.api_key or ""},
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Local target lookup failed for agent {agent_id}: HTTP {response.status_code} {response.text[:200]}"
        )
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def check_sandbox_health(
    sandbox_id: str,
    agent_id: str,
    code_provider: str = "claude",
) -> Optional[dict[str, Any]]:
    """Check E2B sandbox health. Returns health dict or None on failure."""
    config = load_config()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health_resp = await client.get(
                f"{config.backend.api_url}{_session_base_path(code_provider)}/{sandbox_id}/health",
                headers={"x-api-key": config.backend.api_key or ""},
            )
            if health_resp.status_code == 200:
                return health_resp.json()
    except WorkflowGenerationSealed:
        raise  # a replaced generation must stop, never be swallowed
    except Exception as health_err:
        logger.warning(
            "[ClaudeCode] Failed to check sandbox health",
            agent_id=agent_id,
            error=str(health_err),
        )
    return None
