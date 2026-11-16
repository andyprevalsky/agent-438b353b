/**
 * Persistent code-session lifecycle used by session create/close.
 *
 *   create ─► resolveCodeSession ─┬─ E2B paused sandbox ──connect──► resume (fs+memory)
 *                                 ├─ sandbox not found / none ──► new sandbox
 *                                 │     └─ restoreLatestCheckpoint ──► `claude --resume`
 *                                 ├─ other connect error ──► throw (spawn retries; never orphan)
 *                                 └─ local fallback: reuse HOME, else restore checkpoint
 *   close  ─► checkpointCodeSession ─► index code_event rows ─► pause (never kill)
 *                └─► archive job: mirror into the root's R2 bundle ─► prune (archiveMirror.ts)
 *   retire ─► explicit close or TTL: kill sandbox, delete checkpoints — each only
 *             once its R2 copy is confirmed (else the session stays for the next sweep)
 */
import { existsSync, mkdirSync, rmSync } from 'fs';
import { join, resolve, sep } from 'path';
import { NotFoundError, Sandbox } from 'e2b';
import { prisma } from '../database';
import { createLogger } from '../logger';
import { isCliTemplateSession } from '../cliTemplateReferences';
import { checkpointKey, deleteCheckpointObject, getCheckpointObject } from './artifactStore';
import {
  archiveTargetFor,
  ensureCheckpointArchived,
  mirrorCheckpointToArchive,
  scheduleCheckpointArchiveJob,
  waitForCodeSessionArchiveJobs,
  type CheckpointArchiveTarget,
} from './archiveMirror';
import {
  applyCheckpoint,
  CheckpointIntegrityError,
  CLI_LOG_ARTIFACT,
  createCheckpoint,
  loadVerifiedCheckpoint,
  MANIFEST_NAME,
  type CheckpointManifest,
  type CheckpointRepo,
  type StoredCheckpoint,
  type VerifiedCheckpoint,
} from './checkpoint';
import { indexCodeEvents } from './codeEvents';
import { type CodeSessionFs, e2bSandboxFs, localSandboxFs } from './sandboxFs';
import {
  allocateCheckpointSeq,
  commitCheckpoint,
  findExpiredCodeSessions,
  markRetired,
  type CodeSessionRow,
  type ExpiredSweepCursor,
} from './store';

const log = createLogger('CodeSessionLifecycle');

/** How far back a resume walks when newer checkpoints fail verification. */
const RESTORE_LOOKBACK = 5;

/** `live_cli`: the spawn attached to the CLI still running in the resumed sandbox. */
export type ResumedFrom = 'live_cli' | 'paused_sandbox' | 'local_home' | 'checkpoint' | 'new';

export interface RepoFacts {
  path: string;
  branch: string | null;
  head: string | null;
  localCommits: number;
  pushed: boolean;
  remoteUrl: string | null;
  dirtyFiles: number;
  /** origin/HEAD's branch when the checkpoint saw it. */
  defaultBranch?: string | null;
  /** Lines added/removed per file vs the merge target's merge base. */
  files?: Array<{ path: string; added: number; removed: number }>;
}

export interface CheckpointResult {
  seq: number;
  cliSessionId: string | null;
  repo: RepoFacts | null;
  indexedCodeEvents: number;
}

export function localCodeSessionRoot(): string {
  return resolve(
    process.env.CLAUDE_CODE_LOCAL_SESSIONS_DIR || '/tmp/claude-code-local/code-sessions'
  );
}

export function localCodeSessionHome(codeSessionId: string): string {
  if (!/^[A-Za-z0-9-]{1,128}$/.test(codeSessionId)) {
    throw new Error(`Invalid code session id ${JSON.stringify(codeSessionId)}`);
  }
  return join(localCodeSessionRoot(), codeSessionId, 'home');
}

function isOwnedLocalHome(path: string): boolean {
  const root = localCodeSessionRoot();
  return resolve(path).startsWith(root + sep);
}

/** The repo the tool result reports: the cloned workspace, else the first repo. */
export function primaryRepoFacts(manifest: Pick<CheckpointManifest, 'repos'>): RepoFacts | null {
  const repos: CheckpointRepo[] = manifest.repos ?? [];
  const repo = repos.find((candidate) => candidate.path === 'workspace') ?? repos[0];
  if (repo === undefined) return null;
  return {
    path: repo.path,
    branch: repo.branch,
    head: repo.head,
    localCommits: repo.localCommits,
    pushed: repo.pushed,
    remoteUrl: repo.remoteUrl,
    dirtyFiles: repo.dirtyFiles.length,
    defaultBranch: repo.defaultBranch ?? null,
    files: repo.files ?? [],
  };
}

/**
 * Write a checkpoint and index its CLI log. Best-effort by design: a failure is
 * logged and returns null so close can still pause the sandbox (pause itself
 * preserves the filesystem).
 */
export async function checkpointCodeSession(params: {
  session: CodeSessionRow;
  fs: CodeSessionFs;
  reason: string;
  agentId: string;
  /** Epoch ms the checkpoint must finish by; default: now + CODE_SESSION_CLOSE_CHECKPOINT_BUDGET_MS (75 s). */
  deadlineAt?: number;
}): Promise<CheckpointResult | null> {
  const { session, fs, reason, agentId } = params;
  // Every checkpoint is bounded well inside the worker's 180 s close call: an
  // unbounded tarball download held the lease and lost the handoff (2026-11-08).
  const budget = Number(process.env.CODE_SESSION_CLOSE_CHECKPOINT_BUDGET_MS);
  const deadlineAt = params.deadlineAt ?? Date.now() + (budget > 0 ? budget : 75_000);
  try {
    const seq = await allocateCheckpointSeq(session.id);
    const stored = await createCheckpoint({
      fs,
      userId: session.userId,
      codeSessionId: session.id,
      seq,
      reason,
      codeProvider: session.codeProvider,
      diffBase: session.mergeTarget,
      deadlineAt,
      previousSeq: session.latestCheckpointSeq,
    });
    const { manifest } = stored;
    const repo = primaryRepoFacts(manifest);
    await commitCheckpoint(session.id, seq, {
      cliSessionId: manifest.cliSessionId,
      repoFacts: repo,
    });
    // Off the close path: mirror into the trace archive, then prune. The root is
    // resolved here so the job itself never touches the database. Unresolvable
    // now: no mirror or prune this time (the next prune or the retire archives it).
    const target = await archiveTargetFor(session).catch((error: unknown) => {
      log.warn('Resolving the archive root failed; checkpoint archived later', {
        codeSessionId: session.id,
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    });
    if (target !== null) {
      scheduleCheckpointArchiveJob(
        archiveJobWithBytes(session, target, seq, stored),
        archiveJobFromStorage(session, target, seq)
      );
    }
    let indexedCodeEvents = 0;
    if (manifest.artifacts[CLI_LOG_ARTIFACT] !== undefined) {
      const key = checkpointKey(session.userId, session.id, seq, CLI_LOG_ARTIFACT);
      const cliLog = await getCheckpointObject(key);
      if (cliLog !== null) {
        try {
          indexedCodeEvents = await indexCodeEvents({
            userId: session.userId,
            rootAgentId: session.rootAgentId,
            agentId,
            codeSessionId: session.id,
            codeProvider: session.codeProvider,
            cliLog,
            archivePath: key,
            fallbackTime: new Date(manifest.createdAt),
          });
        } catch (indexError) {
          log.warn('Indexing code events failed; checkpoint kept', {
            codeSessionId: session.id,
            seq: String(seq),
            error: indexError instanceof Error ? indexError.message : String(indexError),
          });
        }
      }
    }
    return { seq, cliSessionId: manifest.cliSessionId, repo, indexedCodeEvents };
  } catch (error) {
    log.error('Code session checkpoint failed', {
      codeSessionId: session.id,
      reason,
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }
}

type ArchivedSession = Pick<CodeSessionRow, 'id' | 'userId' | 'rootAgentId'>;

/**
 * Mirror checkpoint `seq` into the root's trace-archive bundle, then prune the
 * checkpoint that just left the restore window. Restore never looks further back
 * than RESTORE_LOOKBACK, so periodic saves keep a bounded Supabase window; the
 * pruned checkpoint is deleted only once its archive copy is confirmed.
 */
async function archiveThenPrune(
  session: ArchivedSession,
  target: CheckpointArchiveTarget,
  seq: number,
  stored: StoredCheckpoint | null
): Promise<void> {
  try {
    if (stored !== null) {
      await mirrorCheckpointToArchive({
        target,
        seq,
        manifestBytes: stored.manifestBytes,
        artifacts: stored.artifacts,
      });
    } else {
      await ensureCheckpointArchived(session, seq, target);
    }
  } catch (error) {
    log.warn('Archiving a code session checkpoint failed; kept in storage and retried', {
      codeSessionId: session.id,
      seq: String(seq),
      error: error instanceof Error ? error.message : String(error),
    });
  }
  if (seq > RESTORE_LOOKBACK) {
    await pruneCheckpoint(session, seq - RESTORE_LOOKBACK, target).catch((error: unknown) => {
      log.warn('Pruning an old code session checkpoint failed', {
        codeSessionId: session.id,
        seq: String(seq - RESTORE_LOOKBACK),
        error: error instanceof Error ? error.message : String(error),
      });
    });
  }
}

/** Separate scopes: the storage job must not keep the checkpoint bytes alive. */
function archiveJobWithBytes(
  session: ArchivedSession,
  target: CheckpointArchiveTarget,
  seq: number,
  stored: StoredCheckpoint
): () => Promise<void> {
  return () => archiveThenPrune(session, target, seq, stored);
}

function archiveJobFromStorage(
  session: ArchivedSession,
  target: CheckpointArchiveTarget,
  seq: number
): () => Promise<void> {
  return () => archiveThenPrune(session, target, seq, null);
}

/** github.com remotes fetch with the platform token, as the ws-server clone does. */
export function githubAuthRemoteUrl(remoteUrl: string): string {
  const token = process.env.GITHUB_TOKEN;
  if (token === undefined || token === '' || !remoteUrl.startsWith('https://github.com/')) {
    return remoteUrl;
  }
  return remoteUrl.replace('https://github.com/', `https://x-access-token:${token}@github.com/`);
}

/**
 * The newest checkpoint that verifies, downloaded and hash-checked. A corrupted
 * checkpoint is rejected (never returned) and the next older one is tried.
 * Read-only: touches neither the session row nor any sandbox.
 */
export async function loadNewestVerifiedCheckpoint(
  session: CodeSessionRow
): Promise<VerifiedCheckpoint | null> {
  const newest = session.latestCheckpointSeq;
  for (let seq = newest; seq >= 1 && seq > newest - RESTORE_LOOKBACK; seq -= 1) {
    try {
      return await loadVerifiedCheckpoint(session.userId, session.id, seq);
    } catch (error) {
      if (error instanceof CheckpointIntegrityError) {
        log.warn('Rejected code session checkpoint', {
          codeSessionId: session.id,
          seq: String(seq),
          error: error.message,
        });
        continue;
      }
      throw error;
    }
  }
  return null;
}

/** Restore the newest checkpoint that verifies (see loadNewestVerifiedCheckpoint). */
export async function restoreLatestCheckpoint(
  session: CodeSessionRow,
  fs: CodeSessionFs
): Promise<{ seq: number; cliSessionId: string | null } | null> {
  const verified = await loadNewestVerifiedCheckpoint(session);
  if (verified === null) return null;
  return applyVerifiedCheckpoint(session, fs, verified);
}

/** Restore one verified checkpoint (a resize restores exactly the one it just took). */
export async function applyVerifiedCheckpoint(
  session: CodeSessionRow,
  fs: CodeSessionFs,
  verified: VerifiedCheckpoint
): Promise<{ seq: number; cliSessionId: string | null }> {
  const { cliSessionId } = await applyCheckpoint(fs, verified, {
    authRemoteUrl: githubAuthRemoteUrl,
  });
  // A checkpoint written by the other provider's CLI (a usage-limit handoff,
  // C15) restores the files and history but not that CLI's conversation.
  const sameProvider = verified.manifest.codeProvider === session.codeProvider;
  return { seq: verified.manifest.seq, cliSessionId: sameProvider ? cliSessionId : null };
}

/**
 * Connect (and thereby resume) a paused E2B sandbox; null only when E2B says it
 * no longer exists. Any other failure (429, 5xx, request timeout) rethrows so the
 * spawn retries the same sandbox: replacing it from a possibly older checkpoint
 * would orphan the paused sandbox and every change made since that checkpoint.
 */
export async function connectPausedSandbox(
  sandboxId: string,
  timeoutMs: number,
  connect: (id: string, opts: { apiKey?: string; timeoutMs: number }) => Promise<Sandbox> = (
    id,
    opts
  ) => Sandbox.connect(id, opts)
): Promise<Sandbox | null> {
  try {
    return await connect(sandboxId, { apiKey: process.env.E2B_API_KEY, timeoutMs });
  } catch (error) {
    if (!(error instanceof NotFoundError)) throw error;
    log.warn('Paused sandbox no longer exists; falling back to checkpoint', {
      sandboxId,
      error: error.message,
    });
    return null;
  }
}

/** Prepare the local-fallback HOME: reuse it when present, else restore. */
export async function prepareLocalCodeSessionHome(session: CodeSessionRow): Promise<{
  home: string;
  resumedFrom: ResumedFrom;
  cliSessionId: string | null;
}> {
  const home =
    session.homeDir !== null && isOwnedLocalHome(session.homeDir)
      ? session.homeDir
      : localCodeSessionHome(session.id);
  if (session.sandboxKind === 'local' && existsSync(home)) {
    return { home, resumedFrom: 'local_home', cliSessionId: session.cliSessionId };
  }
  mkdirSync(home, { recursive: true });
  if (session.latestCheckpointSeq > 0) {
    const restored = await restoreLatestCheckpoint(session, localSandboxFs(home));
    if (restored !== null) {
      return { home, resumedFrom: 'checkpoint', cliSessionId: restored.cliSessionId };
    }
  }
  return { home, resumedFrom: 'new', cliSessionId: null };
}

/**
 * Delete one checkpoint, manifest last. The manifest is the only index of the
 * artifact keys, so a failed read or artifact delete throws and keeps it: the
 * caller retries (pruning logs and moves on; retire stops before markRetired, so
 * the next TTL sweep tries again). Only an unparseable manifest is dropped.
 */
async function deleteCheckpoint(userId: string, codeSessionId: string, seq: number): Promise<void> {
  const manifestKey = checkpointKey(userId, codeSessionId, seq, MANIFEST_NAME);
  const raw = await getCheckpointObject(manifestKey);
  if (raw !== null) {
    let names: string[] = [];
    try {
      const manifest = JSON.parse(raw.toString('utf8')) as CheckpointManifest;
      names = Object.keys(manifest.artifacts ?? {});
    } catch {
      // an unparseable manifest names no artifacts; it still gets deleted below
    }
    for (const name of names) {
      await deleteCheckpointObject(checkpointKey(userId, codeSessionId, seq, name));
    }
  }
  await deleteCheckpointObject(manifestKey);
}

/**
 * Seqs whose checkpoint may still be in storage, newest first. Starts at the
 * highest allocated seq: a checkpoint can be uploaded and its commit lost, so
 * nextCheckpointSeq can sit far above latestCheckpointSeq (4 vs 23 in prod).
 * Goes down to 1, since a prune that failed or waited for its archive copy
 * keeps an older checkpoint. `lookback` caps the scan below the top but never
 * skips the restore window under the latest committed seq.
 */
export function checkpointSeqsToScan(
  session: Pick<CodeSessionRow, 'latestCheckpointSeq' | 'nextCheckpointSeq'>,
  lookback?: number
): number[] {
  const top = Math.max(session.latestCheckpointSeq, session.nextCheckpointSeq);
  const floor =
    lookback === undefined || !Number.isFinite(lookback)
      ? 1
      : Math.max(1, Math.min(top - lookback, session.latestCheckpointSeq - RESTORE_LOOKBACK));
  const seqs: number[] = [];
  for (let seq = top; seq >= floor; seq -= 1) seqs.push(seq);
  return seqs;
}

/** Delete one checkpoint once its trace-archive copy is confirmed (else throw and keep it). */
async function pruneCheckpoint(
  session: ArchivedSession,
  seq: number,
  target?: CheckpointArchiveTarget
): Promise<void> {
  if ((await ensureCheckpointArchived(session, seq, target)) === 'absent') return;
  await deleteCheckpoint(session.userId, session.id, seq);
}

async function deleteCheckpoints(session: CodeSessionRow): Promise<void> {
  const target = await archiveTargetFor(session);
  for (let seq = 1; seq <= session.nextCheckpointSeq; seq += 1) {
    await pruneCheckpoint(session, seq, target);
  }
}

/** Time a retire-time checkpoint may take: no close call waits on it. */
const RETIRE_CHECKPOINT_BUDGET_MS = 10 * 60_000;

/** The session's live filesystem for a last checkpoint, or null when it no longer exists. */
async function retiringSessionFs(session: CodeSessionRow): Promise<CodeSessionFs | null> {
  if (session.sandboxKind === 'e2b' && session.sandboxId !== null) {
    const sandbox = await connectPausedSandbox(session.sandboxId, RETIRE_CHECKPOINT_BUDGET_MS);
    return sandbox === null ? null : e2bSandboxFs(sandbox);
  }
  if (
    session.homeDir !== null &&
    isOwnedLocalHome(session.homeDir) &&
    existsSync(session.homeDir)
  ) {
    return localSandboxFs(session.homeDir);
  }
  return null;
}

/**
 * CLI files the latest checkpoint deferred (its close ran out of budget) exist
 * only in the sandbox: take one more checkpoint that captures them and confirm
 * its archive copy before the sandbox is destroyed. Any failure throws, so the
 * session stays unretired (sandbox and checkpoints kept) for the next sweep.
 * Returns the session row to retire (re-read when a checkpoint was added).
 */
async function captureDeferredCliFiles(session: CodeSessionRow): Promise<CodeSessionRow> {
  if (session.latestCheckpointSeq < 1) return session;
  const raw = await getCheckpointObject(
    checkpointKey(session.userId, session.id, session.latestCheckpointSeq, MANIFEST_NAME)
  );
  let deferred: string[] = [];
  try {
    deferred =
      raw === null
        ? []
        : ((JSON.parse(raw.toString('utf8')) as CheckpointManifest).cliFilesDeferred ?? []);
  } catch {
    deferred = [];
  }
  if (deferred.length === 0) return session;
  const fs = await retiringSessionFs(session);
  if (fs === null) {
    log.error('Deferred CLI files are lost: the code session sandbox no longer exists', {
      codeSessionId: session.id,
      seq: String(session.latestCheckpointSeq),
      deferred: String(deferred.length),
    });
    return session;
  }
  const result = await checkpointCodeSession({
    session,
    fs,
    reason: 'retire',
    agentId: session.ownerAgentId,
    deadlineAt: Date.now() + RETIRE_CHECKPOINT_BUDGET_MS,
  });
  const manifest =
    result === null
      ? null
      : await getCheckpointObject(
          checkpointKey(session.userId, session.id, result.seq, MANIFEST_NAME)
        );
  const stillDeferred =
    manifest === null
      ? deferred
      : ((JSON.parse(manifest.toString('utf8')) as CheckpointManifest).cliFilesDeferred ?? []);
  if (result === null || stillDeferred.length > 0) {
    log.warn('Retire checkpoint did not capture the deferred CLI files; session kept', {
      codeSessionId: session.id,
      deferred: String(stillDeferred.length),
    });
    throw new Error(
      `Code session ${session.id} still has deferred CLI files; retire retried later`
    );
  }
  // The checkpoint's own archive job may still be writing these objects: on a
  // miss, let scheduled jobs settle and confirm once more before giving up.
  const confirm = () => ensureCheckpointArchived(session, result.seq);
  const archived = await confirm().catch(async () => {
    await waitForCodeSessionArchiveJobs();
    return confirm();
  });
  if (archived === 'absent') {
    throw new Error(`Retire checkpoint ${result.seq} of code session ${session.id} is missing`);
  }
  return prisma.codeSession.findUniqueOrThrow({ where: { id: session.id } });
}

/**
 * Capture deferred CLI files, kill the sandbox, delete stored checkpoints (each
 * once archived) and mark the session retired.
 */
export async function retireCodeSession(retiring: CodeSessionRow): Promise<void> {
  if (await isCliTemplateSession(retiring.userId, retiring.id)) {
    throw new Error('Disconnect the CLI integration before retiring its template');
  }
  const session = await captureDeferredCliFiles(retiring);
  if (session.sandboxKind === 'e2b' && session.sandboxId !== null) {
    try {
      await Sandbox.kill(session.sandboxId, { apiKey: process.env.E2B_API_KEY });
    } catch (error) {
      log.warn('Killing retired code-session sandbox failed', {
        codeSessionId: session.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  if (session.homeDir !== null && isOwnedLocalHome(session.homeDir)) {
    rmSync(resolve(session.homeDir, '..'), { recursive: true, force: true });
  }
  await deleteCheckpoints(session);
  await markRetired(session.id);
  log.info('Code session retired', { codeSessionId: session.id });
}

/**
 * Retire expired sessions one by one. A session whose checkpoints cannot be
 * deleted yet stays unretired for the next sweep; it must not stop this sweep
 * from retiring the others.
 */
async function retireExpired(expired: CodeSessionRow[]): Promise<number> {
  let retired = 0;
  for (const session of expired) {
    try {
      await retireCodeSession(session);
      retired += 1;
    } catch (error: unknown) {
      log.warn('Retiring an expired code session failed; retried next sweep', {
        codeSessionId: session.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return retired;
}

/**
 * Opportunistic, bounded TTL sweep for one user. Never rejects; spawns start it
 * without awaiting so it never blocks them. A user who never spawns again is
 * never swept this way: sweepAllExpiredCodeSessions covers them.
 */
export function sweepExpiredCodeSessions(userId: string): Promise<void> {
  return findExpiredCodeSessions(userId)
    .then(async (expired) => {
      await retireExpired(expired);
    })
    .catch((error: unknown) => {
      log.warn('Code session TTL sweep failed', {
        userId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
}

/**
 * A TTL sweeper that pages through the expired backlog, oldest first, at most
 * `limit` sessions per call. Each call resumes after the last session the
 * previous call took and wraps to the oldest once it reaches the end, so
 * sessions that keep failing retirement (an archive copy not yet confirmed)
 * are retried once per pass and never block the sessions behind them.
 */
export function createExpiredCodeSessionSweeper(
  userId: string | null = null
): (limit?: number) => Promise<number> {
  let cursor: ExpiredSweepCursor | null = null;
  return async (limit = 20) => {
    let batch = await findExpiredCodeSessions(userId, limit, new Date(), cursor);
    if (batch.length === 0 && cursor !== null) {
      batch = await findExpiredCodeSessions(userId, limit, new Date(), null);
    }
    const last = batch[batch.length - 1];
    cursor =
      batch.length < limit || last === undefined
        ? null
        : { lastActiveAt: last.lastActiveAt, id: last.id };
    return retireExpired(batch);
  };
}

const sweepAllExpired = createExpiredCodeSessionSweeper(null);

/**
 * Scheduled TTL sweep across all users (createExpiredCodeSessionSweeper): a
 * backlog drains over ticks. Returns how many were retired.
 */
export async function sweepAllExpiredCodeSessions(limit = 20): Promise<number> {
  return sweepAllExpired(limit);
}

/**
 * Periodic save of an attached session (multi-day runs), driven by the
 * attached activity's keepalive:
 *
 *   keepalive ─► due? (interval since this backend's last save, jittered on first sight)
 *                 └─► checkpoint ─► running ≥ cap − margin? ─► pause ─► resume (same sandbox)
 *
 * A sandbox that stays attached would otherwise only be checkpointed on close,
 * so any sandbox loss rolled back to the previous close. E2B also caps one
 * continuous run (24h Pro, 1h Hobby); pause + resume resets that clock and keeps
 * memory and processes, so the cycle happens before the cap with a fresh save.
 *
 * 10k sessions: one map entry per attached session on the backend instance
 * that serves its keepalives; at most MAINTENANCE_CONCURRENCY saves run at once
 * per instance (a saturated tick is retried on the next keepalive), and first
 * sight is jittered over one interval so a deploy restart does not save every
 * session at once. Keepalives served by N instances save up to N× as often.
 */
export interface CodeSessionMaintenanceDeps {
  now: () => number;
  random: () => number;
  schedule: (job: () => Promise<unknown>) => void;
  sandboxInfo: (sandboxId: string) => Promise<{ startedAt: Date }>;
  connect: (sandboxId: string) => Promise<Sandbox>;
}

const MAINTENANCE_CONCURRENCY = 4;
const RUNTIME_CYCLE_MARGIN_MS = 60 * 60 * 1000;
const lastSavedAt = new Map<string, number>();
const maintaining = new Set<string>();

function positiveEnv(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function codeSessionCheckpointIntervalMs(): number {
  return positiveEnv('CODE_SESSION_CHECKPOINT_INTERVAL_MINUTES', 30) * 60 * 1000;
}

/** E2B's maximum continuous runtime for this account (24h on Pro). */
export function e2bMaxContinuousRuntimeMs(): number {
  return positiveEnv('E2B_SANDBOX_MAX_RUNTIME_HOURS', 24) * 60 * 60 * 1000;
}

export const defaultCodeSessionMaintenanceDeps: CodeSessionMaintenanceDeps = {
  now: () => Date.now(),
  random: () => Math.random(),
  schedule: (job) => {
    void job();
  },
  sandboxInfo: (sandboxId) => Sandbox.getInfo(sandboxId, { apiKey: process.env.E2B_API_KEY }),
  connect: (sandboxId) =>
    Sandbox.connect(sandboxId, {
      apiKey: process.env.E2B_API_KEY,
      timeoutMs: 30 * 60 * 1000,
    }),
};

/** Schedule a save (and, near E2B's cap, a pause/resume cycle) when one is due. */
export function scheduleCodeSessionMaintenance(
  task: { id: string; agentId: string; sandboxId: string; codeSessionId: string | null },
  deps: CodeSessionMaintenanceDeps = defaultCodeSessionMaintenanceDeps
): void {
  const codeSessionId = task.codeSessionId;
  if (codeSessionId === null) return;
  const now = deps.now();
  const interval = codeSessionCheckpointIntervalMs();
  const last = lastSavedAt.get(codeSessionId);
  if (last === undefined) {
    lastSavedAt.set(codeSessionId, now - deps.random() * interval);
    return;
  }
  if (now - last < interval || maintaining.has(codeSessionId)) return;
  if (maintaining.size >= MAINTENANCE_CONCURRENCY) return;
  maintaining.add(codeSessionId);
  deps.schedule(() =>
    maintainCodeSession(task, codeSessionId, deps)
      .catch((error: unknown) => {
        log.warn('Periodic code session save failed', {
          codeSessionId,
          error: error instanceof Error ? error.message : String(error),
        });
      })
      .finally(() => {
        lastSavedAt.set(codeSessionId, deps.now());
        maintaining.delete(codeSessionId);
      })
  );
}

/** No close has started on the task (close writes closeReason before it checkpoints and pauses). */
async function taskStillAttached(taskId: string): Promise<boolean> {
  const task = await prisma.claudeCodeSession.findUnique({
    where: { id: taskId },
    select: { status: true, closeReason: true },
  });
  return task?.status === 'ACTIVE' && task.closeReason === null;
}

/** The task closed: drop its maintenance clock (bounded map). */
export function forgetCodeSessionMaintenance(codeSessionId: string | null): void {
  if (codeSessionId !== null) lastSavedAt.delete(codeSessionId);
}

async function maintainCodeSession(
  task: { id: string; agentId: string; sandboxId: string },
  codeSessionId: string,
  deps: CodeSessionMaintenanceDeps
): Promise<void> {
  const session = await prisma.codeSession.findUnique({ where: { id: codeSessionId } });
  if (
    session === null ||
    session.state !== 'ACTIVE' ||
    session.sandboxId !== task.sandboxId ||
    !(await taskStillAttached(task.id))
  ) {
    lastSavedAt.delete(codeSessionId);
    return;
  }
  const info = await deps.sandboxInfo(task.sandboxId);
  const cycle =
    deps.now() - info.startedAt.getTime() >= e2bMaxContinuousRuntimeMs() - RUNTIME_CYCLE_MARGIN_MS;
  const sandbox = await deps.connect(task.sandboxId);
  const checkpoint = await checkpointCodeSession({
    session,
    fs: e2bSandboxFs(sandbox),
    reason: cycle ? 'runtime_cycle' : 'periodic',
    agentId: task.agentId,
  });
  if (!cycle) return;
  // A close sets closeReason first and pauses the sandbox itself: never pause
  // under it, and never resume what it paused.
  if (!(await taskStillAttached(task.id))) return;
  await sandbox.pause();
  if (!(await taskStillAttached(task.id))) return;
  await deps.connect(task.sandboxId);
  log.info('Cycled code session sandbox before the E2B runtime cap', {
    codeSessionId,
    sandboxId: task.sandboxId,
    checkpointSeq: String(checkpoint?.seq ?? 'none'),
  });
}
