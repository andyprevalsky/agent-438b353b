# Code-session resize: guarantees, limits and telemetry (C16)

Code: `back-end/src/providers/codeSessions/{sandboxResize,resizeMove,resizeTransaction}.ts`.
Tests: `back-end/src/tests/blackbox/code-session-resize.blackbox.ts` and
`code-session-resize-faults.cases.ts` (run with `cd back-end && bash scripts/run-blackbox-tests.sh code-session-resize`).

```
rose env size ─► CodeSessionResize PENDING ─► keepalive ─► performSandboxResize (claim, idempotent)
  stop CLI ─► held disk preflight ─► checkpoint seq N ─► new-size sandbox
    ─► target disk preflight (restore + headroom) ─► restore exactly seq N ─► validate
    ─► cut over (row → target, original PAUSED) ─► ws-server serves ─► commit (kill original) ─► DONE
  any failure before cut-over: target killed, original paused and still named ─► FAILED, resumes original
  failure after cut-over:      row → original again, fallback resumes it, then target killed ─► FAILED
```

## Provider facts (E2B, verified 2026-11-16)

- CPU and memory are fixed per template build. The grid is cpu {2, 8} × memory {4, 8} GB,
  with one template per point. The E2B team caps memory at 8192 MiB.
- Disk is also fixed per template build. `minFreeDiskMb` applies only at build time and
  growth is best effort. E2B cannot grow the disk of a running sandbox. Every sized
  template lists about 25 GB (`GET /templates` `diskSizeMB` is 25015 to 25517).
  **A resize never adds disk.** More disk would need templates rebuilt with a larger
  `minFreeDiskMb`, which belongs to the template/sizing configuration, not to resize.

## What a resize preserves

- Workspace files, including uncommitted and untracked changes, local commits, every
  local ref, the checked-out branch and HEAD.
- The CLI conversation log, so `claude --resume` / `codex resume` continue it.
- The session's Rose skills and CLIs, which are re-provisioned on start.

Validation after restore checks HEAD and branch for each repo and the CLI log's sha256
against the checkpoint manifest. Restore uses exactly the checkpoint the resize just
took. It never falls back to an older checkpoint, because that would drop recent work.

## What a resize does not preserve (unavoidable)

- Running processes: dev servers, watchers, builds, shells, and the CLI's in-flight
  turn. The move creates a new VM; it does not live-migrate.
- Directories the checkpoint excludes: `node_modules`, `.venv`/`venv`, `dist`, `build`,
  `target`, `.next`, `.cache`, `.npm`, `__pycache__`. Also files over the per-file
  checkpoint cap (25 MB default), which the manifest lists as `skippedLargeFiles`.
- Anything outside HOME, such as system packages installed at runtime and `/tmp`.

This is not universal losslessness. It guarantees durable workspace and conversation
state, and it fails closed when that state cannot be moved intact.

## Failure codes

The code is the prefix of `CodeSessionResize.error` and of the agent's failure turn.

| Code | Meaning | Retry |
|---|---|---|
| `HELD_DISK_FULL` | Less than 512 MB free to write the checkpoint | no |
| `TARGET_DISK_CAPACITY` | New sandbox's free space < (held usage − target baseline) + artifacts + max(512 MB, 10%) | no |
| `CHECKPOINT_FAILED` / `CHECKPOINT_INTEGRITY` | Save failed, or the saved checkpoint does not verify | no |
| `RESTORE_FAILED` / `RESTORE_VALIDATION` | Restore errored, or the restored HOME does not match | 1 more fresh target (2 total) |
| `TARGET_CREATE_FAILED` | E2B create failed after its own retries | no |
| `TARGET_START_FAILED` | ws-server never served after cut-over: rolled back | no |

The target capacity estimate is an upper bound, because the held disk usage includes
excluded caches. When disk is near full it can refuse a move that would have fit. The
message tells the agent to free caches, which a resize would drop anyway.

## Telemetry

The `ResizeRequestView` (`rose env status`, resize responses) has `queue_ms`
(requested→claimed), `apply_ms` (claimed→settled) and `request_to_settled_ms`
(requested→ready or failed). They are derived from the row's own timestamps, with no
schema change. The `[CodeSandboxResize]` logs `Sandbox resized` / `Sandbox resize settled`
carry `telemetry`: outcome (`done`/`failed`/`rolled_back`), `failure_code`, the three
durations, `phases_ms` (stop, preflight, checkpoint, create, restore, validate, ready,
commit, rollback), `restore_attempts`, and held/target `df` bytes.

## Remaining risks

- If the backend dies after cut-over and before commit, the row already names the
  validated target. The activity's re-run of the stale claim (after 15 min) resumes the
  target and kills the original by id. If no re-run comes, the paused original remains
  until manual cleanup.
- The kill after a successful resize is best effort, and a failed kill is logged. It
  never runs while the row still names the original.
