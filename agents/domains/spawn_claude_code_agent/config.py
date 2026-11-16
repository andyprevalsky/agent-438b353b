"""Spawn-claude-code-agent domain configuration — constants and defaults."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Final

# The post-run code-agent completion verifier (Claude Code + Codex share this
# path) is off by default in prod, local and cloud benchmarks alike: the
# sandbox session's natural end is the result, labelled complete / incomplete /
# unverified / failed from mechanical evidence, with no verifier LLM pass.
# Explicit opt-in only. Values: 1 / true / yes / on
ROSE_ENABLE_CODE_AGENT_VERIFIER_ENV: Final[str] = "ROSE_ENABLE_CODE_AGENT_VERIFIER"


def code_agent_verifier_enabled() -> bool:
    """Return True only when the post-run code-agent LLM verifier is opted in."""
    raw = os.getenv(ROSE_ENABLE_CODE_AGENT_VERIFIER_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# Silence timeout — if no output for this long, Claude Code is "done talking"
SILENCE_TIMEOUT_S: Final[float] = 1.0

# ── No timeouts (rose-builds-rose C11) ──
# A code agent has no runtime limit. Liveness is heartbeat-only: the activity
# heartbeats Temporal and keeps the sandbox alive while it is attached, and a
# dead worker is detected by CODE_ACTIVITY_HEARTBEAT_TIMEOUT_S and retried.
#
# When the CLI sends no event for this window with no tool in flight and no
# SDK result, the session is PAUSED (checkpoint + E2B pause) and the tool
# returns status "paused" with its code_session_id: resumable on the next
# spawn, never a failure. E2B's pause keeps memory and processes, and the next
# spawn attaches to the CLI still running there (e2bCodeSessionSandbox.ts
# attachLiveCodeCli): its turn, background shells and a pending Monitor /
# ScheduleWakeup wait continue and the new goal is its next turn. Only when that
# CLI has exited, holds another credential or is unreachable does the spawn
# restart it with the files, commits and conversation (--resume). A tool in flight
# (long Bash, Write, network call) keeps the session attached indefinitely:
# silence there is work, not a hang (cuda k=3 r21/r22/r24 were productive
# sessions the old 180s/900s idle kills aborted).
CODE_SESSION_IDLE_PAUSE_S: Final[float] = 900.0

# After the SDK emits type=result the CLI has declared completion; healthy
# sessions exit within ~1s (cuda k=3 r31 a1). A zombie that keeps tooling
# after result (r31 a2) must not hold the session: once this grace passes the
# monitor ends and the run is verified on its result. Not a failure and not a
# kill: closing the session checkpoints and pauses the sandbox.
SDK_POST_RESULT_EXIT_GRACE_S: Final[float] = 180.0

# The attached activity extends the sandbox's E2B idle window this often
# (POST /:taskId/keepalive). Well inside the backend's 30-minute window, so an
# attached session never auto-pauses; once keepalives stop it pauses on its own.
# The same call reports a pending sandbox resize (C16) for the activity to apply.
# 10k attached sessions = ~83 keepalives/s, each O(1) indexed reads.
CODE_SESSION_KEEPALIVE_INTERVAL_S: Final[float] = 120.0
# C16: applying a size change = checkpoint + new sandbox + restore + ws-server
# start on the backend; one call may take minutes. A concurrent apply (409) is
# waited out with linear backoff.
CODE_SESSION_RESIZE_TIMEOUT_S: Final[float] = 900.0
CODE_SESSION_RESIZE_ATTEMPTS: Final[int] = 5
CODE_SESSION_RESIZE_RETRY_BACKOFF_S: Final[float] = 5.0
CODE_SESSION_RESIZE_HEARTBEAT_S: Final[float] = 15.0
CODE_AGENT_RESIZE_YIELD_REASON: Final[str] = "environment_resize"

# Temporal requires a start_to_close or schedule_to_close timeout on every
# activity. The code-agent activity passes this effectively-infinite value so
# its only liveness check is heartbeat_timeout (structural test:
# test_code_agent_activity_has_no_deadline_and_no_attempt_cap).
CODE_AGENT_ACTIVITY_NO_DEADLINE: Final[timedelta] = timedelta(days=36500)

# Temporal retries are unlimited so worker crashes and deploys never end a
# session. Errors the activity itself raises (not worker death) stop retrying
# after this many attempts so a deterministic failure cannot loop forever.
CODE_AGENT_ERROR_RETRY_LIMIT: Final[int] = 5
# Heartbeat detail key carrying how many errors earlier attempts raised. The
# attempt number also counts worker deaths, so it cannot be the error count.
CODE_AGENT_ERRORS_HEARTBEAT_KEY: Final[str] = "code_agent_errors"

# Tool result status for an idle session that was paused (resumable).
CODE_AGENT_PAUSED_STATUS: Final[str] = "paused"

# C15: the CLI stopped on a provider usage limit or ran out of credits
# (structured signals only). The session is saved and its sandbox paused; the
# parent gets the error with every credential path and the exact retry. A retry
# with the same code_session_id prefers the session's own provider (paused
# sandbox + the CLI's native resume), switches provider with a trajectory
# handoff only when that provider has nothing leasable, and starts nothing
# (same status, failure_kind "no_credentials") when no path is leasable.
CODE_AGENT_USAGE_LIMIT_STATUS: Final[str] = "usage_limit"

# Structured transcript rows (utils/code_transcript.py) are persisted in batches at most
# this often while new events arrive, plus once when monitoring ends. Live
# token streaming reaches viewers over the sandbox ws (`code_agent_delta`);
# the batches are the durable record, so ≤0.5 writes/s per active code agent.
TRANSCRIPT_FLUSH_INTERVAL_S: Final[float] = 2.0

# ── Deploy drain / reattach ──
# Same value as BROWSER_DRAIN_YIELD_STATUS: the activity detached from a sandbox
# that is still running and returned the session for the next attempt.
CODE_AGENT_DRAIN_YIELD_STATUS: Final[str] = "yielded_for_deploy_drain"
# Heartbeat detail key carrying the attached session, so a Temporal retry after
# a dead worker reattaches instead of creating a second sandbox.
CODE_SESSION_HEARTBEAT_KEY: Final[str] = "code_session"
DRAIN_CHECK_INTERVAL_S: Final[float] = 1.0
# The retry after a dead worker starts once this passes without a heartbeat, so
# the sandbox ws-servers serve their final buffer until the session is closed.
CODE_ACTIVITY_HEARTBEAT_TIMEOUT_S: Final[int] = 300
# Session create: the backend may wait for its shared E2B create slot (up to
# 15 min, back-end/src/providers/e2bCreateSlot.ts) and then start the sandbox; a
# fresh repo sandbox restores its project for minutes (150 s on 2026-11-06,
# when a 60 s read timed out and the retries started 3 tasks). The activity
# heartbeats while it waits, so liveness stays the heartbeat timeout; this
# bound only ends a backend that never answers. A retry of the same tool call
# gets the same task back (backend createWithCodeSession).
CODE_SESSION_CREATE_TIMEOUT_S: Final[float] = 30 * 60.0
CODE_SESSION_CREATE_HEARTBEAT_S: Final[float] = 30.0
# A resumed spawn whose live CLI the user holds in direct chat, behind a
# ws-server too old to hold an attaching spawn: the backend answers 409
# CODE_SESSION_HELD and never stops that CLI. The spawn waits (heartbeating,
# capped backoff) for the hand back; the ws-server releases an idle hold after
# 10 minutes, so this covers it with a margin before failing truthfully.
CODE_SESSION_HELD_CODE: Final[str] = "CODE_SESSION_HELD"
CODE_SESSION_HELD_WAIT_S: Final[float] = 12 * 60.0
CODE_SESSION_HELD_BACKOFF_S: Final[tuple[float, float]] = (5.0, 30.0)
# A reattaching attempt closes the previous sandbox and starts over only after
# this many failed handshakes, so one network blip does not discard a live run.
REATTACH_PROBE_ATTEMPTS: Final[int] = 3
REATTACH_PROBE_BACKOFF_S: Final[float] = 2.0

# Suppress PTY echo from triggering silence detection for this window
ECHO_SUPPRESS_S: Final[float] = 1.0

# After this many unchanged outputs, stop calling LLM (output is stale)
MAX_UNCHANGED_OUTPUTS: Final[int] = 3

# E2B sandbox cost per vCPU: $0.000014/second = 0.0014 cents/second. A sandbox
# reports its size (C16); one without a reported size is the default 2 vCPUs.
E2B_COST_CENTS_PER_VCPU_SECOND: Final[float] = 0.0014
E2B_DEFAULT_SANDBOX_VCPUS: Final[int] = 2

# ── Run target (C14: cloud sandbox vs the user's Mac) ──
CODE_TARGET_CLOUD: Final[str] = "cloud"
CODE_TARGET_LOCAL: Final[str] = "local"
# Error type for a local (Mac) target that cannot start; matches the backend's
# 409 ``code`` so the handler surfaces the exact text as the tool error.
LOCAL_TARGET_UNAVAILABLE_CODE: Final[str] = "LOCAL_TARGET_UNAVAILABLE"
# Backend code when E2B refuses a sandbox over the plan's concurrent-sandbox limit.
# Fails the tool immediately with the backend's message: no retry, no soft-fail.
E2B_CAPACITY_EXHAUSTED_CODE: Final[str] = "E2B_CAPACITY_EXHAUSTED"
# Backend 400 code when no built sandbox within the user's cap satisfies a spawn's
# explicit sandbox_size; the message lists the available sizes. Never retried.
SANDBOX_SIZE_UNSUPPORTED_CODE: Final[str] = "SANDBOX_SIZE_UNSUPPORTED"
# The backend answers session create 429 with this code when GitHub's rate limit stopped the repo access check.
GITHUB_RATE_LIMITED_CODE: Final[str] = "GITHUB_RATE_LIMITED"
# Error type of a run that ended abnormally after its code session existed. Its
# message and details carry the exact error plus the code_session_id (and the
# failure checkpoint) so the parent can resume instead of losing the work.
CODE_AGENT_SESSION_FAILED_CODE: Final[str] = "CODE_AGENT_SESSION_FAILED"
# Reason codes shared with back-end/src/providers/localDevices/messages.ts.
LOCAL_TARGET_NOT_MAC_ORIGIN: Final[str] = "not_mac_origin"
LOCAL_TARGET_DEVICE_DISCONNECTED: Final[str] = "device_disconnected"
# Exact user-facing tool errors (rose-builds-rose.md C14), shown in the chip's
# red failure box. The backend owns the same templates;
# code_agent_local_target.blackbox.py asserts the two never drift.
LOCAL_TARGET_NOT_MAC_ORIGIN_MESSAGE: Final[str] = (
    "Local runs need an agent started from the Rose Mac app. Start this agent from the Mac app, or run it in the cloud."
)
LOCAL_TARGET_DISCONNECTED_TEMPLATE: Final[str] = (
    "Can't run locally: the Rose Mac app on {device_name} is closed or disconnected. "
    "Open the Rose Mac app and retry, or run this agent in the cloud."
)
# Only when a Mac-origin agent's device row has no name.
LOCAL_TARGET_UNKNOWN_DEVICE_NAME: Final[str] = "your Mac"

# Explicit local runs can originate in the cloud; machine ownership is still required.
LOCAL_TARGET_NO_OWNED_DEVICE: Final[str] = "no_owned_device"
LOCAL_TARGET_SELECTION_REQUIRED: Final[str] = "device_selection_required"
LOCAL_TARGET_NO_OWNED_DEVICE_MESSAGE: Final[str] = (
    "No owned Mac is available for this local run. Connect your Mac in Rose and retry."
)
LOCAL_TARGET_SELECTION_REQUIRED_MESSAGE: Final[str] = (
    "More than one owned Mac is registered. Select the intended Mac explicitly before starting this local run."
)
