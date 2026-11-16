/**
 * C16 resize as a transaction: the held sandbox stays the session's only valid
 * environment until the new-size sandbox is restored, validated and serving.
 *
 *   performSandboxResize ── runInResizeTransaction(tx)
 *     └─ create ─► acquireCodeSessionSandbox (resize path, moveToResizedSandbox)
 *          held: stop CLI → disk preflight (HELD_DISK_FULL) → checkpoint seq N
 *          target (≤ RESIZE_RESTORE_ATTEMPTS fresh sandboxes):
 *            disk preflight incl. restore headroom (TARGET_DISK_CAPACITY, no retry)
 *            → restore exactly seq N (never an older checkpoint)
 *            → validate HEAD / branch per repo + CLI log sha256 (RESTORE_VALIDATION)
 *            → cut over: the row names the target; the held sandbox is PAUSED, not killed
 *          any failure before cut-over: the target is killed (it holds only a copy
 *          of seq N), the held sandbox is paused and the row still names it
 *     └─ ws-server serves (create returned) ─► commit: kill the original
 *     └─ create failed after cut-over ─► rollback: the row names the original
 *          again; the fallback spawn resumes it; only then is the target killed
 *
 * E2B (SDK 2.55, verified 2026-11-16): CPU and memory are fixed per template
 * build; disk is fixed per template build too (`minFreeDiskMb` at build time,
 * ~25 GB on every sized template) and cannot grow on a running sandbox. A
 * resize therefore never adds disk: capacity is checked, never assumed.
 *
 * Process continuity: running processes (dev servers, watchers, builds, the
 * CLI's in-flight turn) and anything outside the checkpoint (node_modules,
 * build outputs, caches, see checkpoint.ts EXCLUDED_DIR_NAMES; files over the
 * per-file cap) do not survive a move; the conversation resumes from its log.
 *
 * 10k agents: per resize two `df` calls and one validation script per target;
 * the transaction is per-call AsyncLocalStorage, never shared state.
 */
import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash } from 'node:crypto';
import type { Sandbox } from 'e2b';
import { createLogger } from '../logger';
import type { CheckpointManifest, VerifiedCheckpoint } from './checkpoint';
import { CLI_LOG_ARTIFACT } from './cliFiles';
import type { CodeSessionFs } from './sandboxFs';

const log = createLogger('CodeSandboxResize');

/** Fresh target sandboxes tried for one resize (restore / validation failures only). */
export const RESIZE_RESTORE_ATTEMPTS = 2;
/** Free space the held sandbox needs to write the checkpoint archive. */
export const RESIZE_MIN_HELD_FREE_BYTES = 512 * 1024 * 1024;
/** Headroom left on the target after the restore: max(this, 10% of the restore). */
export const RESIZE_MIN_TARGET_HEADROOM_BYTES = 512 * 1024 * 1024;

export type ResizeFailureCode =
  | 'HELD_DISK_FULL'
  | 'TARGET_DISK_CAPACITY'
  | 'CHECKPOINT_FAILED'
  | 'CHECKPOINT_INTEGRITY'
  | 'RESTORE_FAILED'
  | 'RESTORE_VALIDATION'
  | 'TARGET_CREATE_FAILED'
  | 'TARGET_START_FAILED';

/** A resize step failed; `retryable` says whether a fresh target sandbox may help. */
export class ResizeStepError extends Error {
  constructor(
    readonly code: ResizeFailureCode,
    message: string,
    readonly retryable = false
  ) {
    super(message);
    this.name = 'ResizeStepError';
  }
}

export function resizeFailureCode(error: unknown): ResizeFailureCode {
  return error instanceof ResizeStepError ? error.code : 'TARGET_START_FAILED';
}

export type ResizePhase =
  | 'stop'
  | 'preflight'
  | 'checkpoint'
  | 'create'
  | 'restore'
  | 'validate'
  | 'ready'
  | 'commit'
  | 'rollback';

export interface DiskUsage {
  totalBytes: number;
  usedBytes: number;
  availBytes: number;
}

/** One resize in flight in this process (performSandboxResize owns it). */
export interface ResizeTransaction {
  requestId: string;
  /** Milliseconds per phase (summed when a phase repeats). */
  phases: Partial<Record<ResizePhase, number>>;
  restoreAttempts: number;
  disk: { held?: DiskUsage | null; target?: DiskUsage | null };
  /** The original sandbox, paused at cut-over and kept until commit or rollback. */
  retained: Sandbox | null;
  /** The new-size sandbox the row was cut over to. */
  targetSandboxId: string | null;
}

const store = new AsyncLocalStorage<ResizeTransaction>();

export function newResizeTransaction(requestId: string): ResizeTransaction {
  return {
    requestId,
    phases: {},
    restoreAttempts: 0,
    disk: {},
    retained: null,
    targetSandboxId: null,
  };
}

export function runInResizeTransaction<T>(tx: ResizeTransaction, fn: () => Promise<T>): Promise<T> {
  return store.run(tx, fn);
}

export function currentResizeTransaction(): ResizeTransaction | null {
  return store.getStore() ?? null;
}

/** Time `fn` into `phase` of the current (or given) transaction, success or failure. */
export async function timedPhase<T>(
  phase: ResizePhase,
  fn: () => Promise<T>,
  tx: ResizeTransaction | null = currentResizeTransaction()
): Promise<T> {
  const started = Date.now();
  try {
    return await fn();
  } finally {
    if (tx !== null) tx.phases[phase] = (tx.phases[phase] ?? 0) + (Date.now() - started);
  }
}

// ── capacity ──

/** `df` of the filesystem holding HOME, or null when it cannot be read (never guessed). */
export async function diskUsage(fs: CodeSessionFs): Promise<DiskUsage | null> {
  const result = await fs.exec(`df -Pk -- "${fs.home}" | tail -n 1`, { timeoutMs: 15_000 });
  const fields = result.exitCode === 0 ? result.stdout.trim().split(/\s+/) : [];
  const [total, used, avail] = fields.slice(1, 4).map((value) => Number(value) * 1024);
  if (![total, used, avail].every((value) => Number.isFinite(value) && value >= 0)) {
    log.warn('Disk usage unreadable; capacity check skipped', {
      exitCode: String(result.exitCode),
      stderr: result.stderr.slice(0, 200),
    });
    return null;
  }
  return { totalBytes: total, usedBytes: used, availBytes: avail };
}

export function gb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

const FREE_SPACE_HINT =
  'Free space in the sandbox first (node_modules, build outputs and caches are not carried ' +
  'over by a resize anyway), then retry. Every sandbox size has the same disk: E2B fixes ' +
  'disk per template, so a resize changes CPU and memory, never disk.';

/** The held sandbox must have room for the checkpoint archive before it is saved. */
export function checkHeldCapacity(held: DiskUsage | null): void {
  if (held === null || held.availBytes >= RESIZE_MIN_HELD_FREE_BYTES) return;
  throw new ResizeStepError(
    'HELD_DISK_FULL',
    `The sandbox disk is full (${gb(held.availBytes)} free of ${gb(held.totalBytes)}); ` +
      `saving the session needs at least ${gb(RESIZE_MIN_HELD_FREE_BYTES)} free. ${FREE_SPACE_HINT}`
  );
}

/**
 * Bytes the target needs: what the session added on the held disk beyond the
 * target's own baseline (an upper bound: excluded caches are counted), the
 * downloaded artifacts, and headroom.
 */
export function restoreBytesNeeded(
  held: DiskUsage,
  target: DiskUsage,
  artifactBytes: number
): number {
  const content = Math.max(0, held.usedBytes - target.usedBytes) + artifactBytes;
  return content + Math.max(RESIZE_MIN_TARGET_HEADROOM_BYTES, Math.ceil(content * 0.1));
}

export function checkTargetCapacity(
  held: DiskUsage | null,
  target: DiskUsage | null,
  artifactBytes: number
): void {
  if (held === null || target === null) return;
  const needed = restoreBytesNeeded(held, target, artifactBytes);
  if (target.availBytes >= needed) return;
  throw new ResizeStepError(
    'TARGET_DISK_CAPACITY',
    `The new sandbox has ${gb(target.availBytes)} free of ${gb(target.totalBytes)}; restoring ` +
      `this session needs about ${gb(needed)} (incl. headroom). ${FREE_SPACE_HINT}`
  );
}

export function checkpointArtifactBytes(checkpoint: VerifiedCheckpoint): number {
  let total = 0;
  for (const data of checkpoint.artifacts.values()) total += data.length;
  return total;
}

// ── restore validation ──

function safeRelativePath(path: string): boolean {
  return path === '' || (!path.startsWith('/') && !path.split('/').some((part) => part === '..'));
}

function quote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** Repos applyCheckpoint rebuilds (it skips the same ones). */
function restoredRepos(manifest: CheckpointManifest, artifacts: Map<string, Buffer>) {
  return (manifest.repos ?? []).filter(
    (repo) =>
      safeRelativePath(repo.path) &&
      (repo.head !== null || repo.branch !== null) &&
      ((repo.bundle !== null && artifacts.has(repo.bundle)) ||
        repo.remoteUrl !== null ||
        Object.keys(repo.refs ?? {}).length > 0)
  );
}

/**
 * The restored HOME matches the checkpoint: every rebuilt repo has the recorded
 * HEAD and branch, and the CLI log (what `--resume` reads) has the recorded hash.
 */
export async function validateRestoredCheckpoint(
  fs: CodeSessionFs,
  checkpoint: VerifiedCheckpoint
): Promise<void> {
  const { manifest, artifacts } = checkpoint;
  const problems: string[] = [];
  for (const repo of restoredRepos(manifest, artifacts)) {
    const dir = repo.path === '' ? fs.home : `${fs.home}/${repo.path}`;
    const result = await fs.exec(
      `cd ${quote(dir)} && git rev-parse HEAD && (git symbolic-ref -q --short HEAD || echo '-')`,
      { timeoutMs: 30_000 }
    );
    const [head, branch] = result.stdout.trim().split('\n');
    const label = repo.path || '.';
    if (result.exitCode !== 0) {
      problems.push(`${label}: not a git repo (${result.stderr.slice(0, 120)})`);
      continue;
    }
    if (repo.head !== null && head !== repo.head) {
      problems.push(`${label}: HEAD ${head} != ${repo.head}`);
    }
    if (repo.branch !== null && branch !== repo.branch) {
      problems.push(`${label}: branch ${branch} != ${repo.branch}`);
    }
  }
  const cliLog = manifest.artifacts[CLI_LOG_ARTIFACT];
  if (
    cliLog !== undefined &&
    artifacts.has(CLI_LOG_ARTIFACT) &&
    manifest.cliLogPath !== null &&
    safeRelativePath(manifest.cliLogPath)
  ) {
    const path = `${fs.home}/${manifest.cliLogPath}`;
    const restored = await fs.readFile(path).catch(() => null);
    const digest = restored === null ? null : createHash('sha256').update(restored).digest('hex');
    if (digest !== cliLog.sha256)
      problems.push(`CLI log ${manifest.cliLogPath} missing or altered`);
  }
  if (problems.length > 0) {
    throw new ResizeStepError(
      'RESTORE_VALIDATION',
      `The restored session does not match checkpoint ${manifest.seq}: ${problems.join('; ').slice(0, 400)}`,
      true
    );
  }
}

// ── telemetry ──

export interface ResizeTelemetry {
  request_id: string;
  outcome: 'done' | 'failed' | 'rolled_back';
  failure_code: ResizeFailureCode | null;
  /** requestedAt → startedAt (waiting for the attached activity). */
  queue_ms: number | null;
  /** startedAt → completedAt (the apply). */
  apply_ms: number | null;
  /** requestedAt → completedAt: request to ready (DONE) or to failure. */
  request_to_settled_ms: number | null;
  phases_ms: Partial<Record<ResizePhase, number>>;
  restore_attempts: number;
  held_disk: DiskUsage | null;
  target_disk: DiskUsage | null;
}

function span(from: Date | null, to: Date | null): number | null {
  return from !== null && to !== null ? to.getTime() - from.getTime() : null;
}

/** Durations from the request row's own timestamps (no extra columns). */
export function resizeDurations(row: {
  requestedAt: Date;
  startedAt: Date | null;
  completedAt: Date | null;
}): { queue_ms: number | null; apply_ms: number | null; request_to_settled_ms: number | null } {
  return {
    queue_ms: span(row.requestedAt, row.startedAt),
    apply_ms: span(row.startedAt, row.completedAt),
    request_to_settled_ms: span(row.requestedAt, row.completedAt),
  };
}

export function resizeTelemetry(
  row: { id: string; requestedAt: Date; startedAt: Date | null; completedAt: Date | null },
  tx: ResizeTransaction,
  outcome: ResizeTelemetry['outcome'],
  failureCode: ResizeFailureCode | null
): ResizeTelemetry {
  return {
    request_id: row.id,
    outcome,
    failure_code: failureCode,
    ...resizeDurations(row),
    phases_ms: { ...tx.phases },
    restore_attempts: tx.restoreAttempts,
    held_disk: tx.disk.held ?? null,
    target_disk: tx.disk.target ?? null,
  };
}
