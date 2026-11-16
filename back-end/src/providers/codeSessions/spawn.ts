/**
 * Glue between the code-session HTTP surface (claude-code / codex-code
 * domains) and the persistent CodeSession lifecycle.
 *
 * One spawn tool call starts one task. The agents activity is at-least-once
 * and retries a create that outlives its read (a fresh repo sandbox restores
 * its project for minutes) while the first request keeps running here, all
 * under the call's lease owner `tool:<agent>:<call>`:
 *
 *   create(tool:…) ─▶ per-lease-owner lock ─┬─ stored task still ACTIVE ─▶ that task (replay)
 *                                           └─ otherwise ─▶ resolve + lease ─▶ create() ─▶ store
 *
 * A replay never reaches create(), so it never restarts the CLI the first
 * request started (RCA 2026-11-06-code-spawn-create-retry-duplicated-sessions).
 * Resize and usage-limit switches close the task before they create again
 * under the same owner, so they create. Owners other than `tool:` (agent-level
 * spawns) name distinct spawns and are never replayed.
 */
import type { Prisma } from '@prisma/client';
import { Sandbox } from 'e2b';
import { rootPreviewPath } from '../codePreview/rootPath';
import { prisma } from '../database';
import { E2BSandboxCapacityError } from '../e2bSandboxCreate';
import { GithubRateLimitedError } from '../githubRateLimit';
import { createLogger } from '../logger';
import { getEphemeral, putEphemeral, withSerializedLock } from '../sharedState';
import { isLocalClaudeCodeSandboxId } from '../claudeCodeLocalSession';
import { isLocalDeviceSandboxId } from '../localDevices/hub';
import { resolveArchiveRootAgentId } from './archiveMirror';
import { effectiveOwnerAgentId, planCodeSessionBranch } from './branch';
import { drainStartupTurns } from './startupTurns';
import {
  checkpointCodeSession,
  defaultCodeSessionMaintenanceDeps,
  forgetCodeSessionMaintenance,
  retireCodeSession,
  scheduleCodeSessionMaintenance,
  sweepExpiredCodeSessions,
  type CodeSessionMaintenanceDeps,
} from './lifecycle';
import { e2bSandboxFs, localSandboxFs } from './sandboxFs';
import {
  CodeSessionBusyError,
  CodeSessionNotFoundError,
  extendLease,
  findOwnedCodeSession,
  markPaused,
  releaseLease,
  resolveCodeSession,
  type CodeSessionRow,
} from './store';
import {
  CODE_SANDBOX_IDLE_TIMEOUT_MS,
  CodeSessionHeldError,
  CodeSessionPauseUnconfirmedError,
  type SpawnSizingReport,
} from '../e2bCodeSessionSandbox';
import {
  revertCodeSessionProvider,
  switchCodeSessionProvider,
  type SpawnCodeSession,
} from './usageLimit';
import { pendingResizeForTask, sizeView, type SizeView } from './sandboxResize';
import { SandboxSizeUnsupportedError, type SandboxSize } from './sandboxSize';
import {
  codeResourceDeps,
  type CodeResourceDeps,
  type CodeResourceSample,
} from './sandboxResources';

const log = createLogger('CodeSessionSpawn');

export { CodeSessionBusyError, CodeSessionNotFoundError } from './store';

/** HTTP status + body for a code-session error, or null when it is not one. */
export function codeSessionHttpError(
  error: unknown
): { status: number; body: { error: string; code: string } & Record<string, unknown> } | null {
  if (error instanceof CodeSessionNotFoundError) {
    return {
      status: 404,
      body: { error: 'Code session not found', code: 'CODE_SESSION_NOT_FOUND' },
    };
  }
  if (error instanceof CodeSessionBusyError) {
    return { status: 409, body: { error: error.message, code: 'CODE_SESSION_BUSY' } };
  }
  if (error instanceof CodeSessionHeldError) {
    // Retryable: the spawn attaches once the user hands the live CLI back.
    return { status: 409, body: { error: error.message, code: 'CODE_SESSION_HELD' } };
  }
  if (error instanceof E2BSandboxCapacityError) {
    return { status: error.statusCode, body: { error: error.message, code: error.code } };
  }
  if (error instanceof SandboxSizeUnsupportedError) {
    // The root asked for a size no built template within the user's cap satisfies.
    return {
      status: 400,
      body: {
        error: error.message,
        code: error.code,
        requested: sizeView(error.requested),
        available: error.available.map(sizeView),
        max: sizeView(error.max),
      },
    };
  }
  if (error instanceof GithubRateLimitedError) {
    return {
      status: 429,
      body: {
        error: error.message,
        code: error.code,
        reset_at: error.resetAt,
        retry_after_seconds: error.retryAfterSeconds(),
      },
    };
  }
  return null;
}

export interface ResolveSpawnParams {
  agentId: string;
  userId: string | undefined;
  repoUrl: string | null;
  codeProvider: 'claude' | 'codex';
  codeSessionId?: string;
  leaseOwner?: string;
  /** Project default branch; null/undefined = the repo's own default branch. */
  baseBranch?: string | null;
}

/** Resolve (and lease) the code session a spawn attaches to. */
export async function resolveSpawnCodeSession(
  params: ResolveSpawnParams
): Promise<CodeSessionRow | null> {
  if (params.userId === undefined || params.userId === '') return null;
  const userId = params.userId;
  const agent = await prisma.agent.findUnique({
    where: { id: params.agentId },
    select: { busId: true },
  });
  // Every descendant carries its tree's busId; one without it (a sub-agent
  // created without bus_id) is keyed by its parent chain's root, never itself.
  const rootAgentId =
    agent?.busId !== undefined && agent.busId !== null && agent.busId !== ''
      ? agent.busId
      : await resolveArchiveRootAgentId(params.agentId);
  void sweepExpiredCodeSessions(userId);
  // "New agent in this branch" works in the shared agent's session; every
  // other agent owns its own worktree and branch (plan §1b).
  const ownerAgentId = await effectiveOwnerAgentId(userId, params.agentId);
  const { session, created } = await resolveCodeSession({
    userId,
    rootAgentId,
    ownerAgentId,
    repoUrl: params.repoUrl,
    codeProvider: params.codeProvider,
    codeSessionId: params.codeSessionId,
    leaseOwner: params.leaseOwner ?? `agent:${params.agentId}`,
    shared: ownerAgentId !== params.agentId,
    planBranch: (parallelOf) =>
      planCodeSessionBranch({
        userId,
        ownerAgentId,
        repoUrl: params.repoUrl,
        defaultBranch: params.baseBranch ?? null,
        parallelOf,
      }),
  });
  log.info(created ? 'Created code session' : 'Resuming code session', {
    codeSessionId: session.id,
    rootAgentId,
    ownerAgentId,
    branch: session.branch ?? undefined,
    mergeTarget: session.mergeTarget ?? undefined,
    state: session.state,
    checkpointSeq: String(session.latestCheckpointSeq),
  });
  return session;
}

type SpawnCreateResult = {
  taskId: string;
  resumedFrom?: string;
  cliSessionId?: string | null;
  sandboxSize?: SandboxSize | null;
  sandboxSizing?: SpawnSizingReport | null;
  historyEvents?: number;
  startup?: { wsReadyMs: number; cloneMs: number | null };
};

const SPAWN_CREATE_NAMESPACE = 'code-spawn-create';
/**
 * How long a retried create waits for its spawn's first request: that one may
 * wait for the shared E2B create slot (up to 15 minutes, e2bCreateSlot.ts) and
 * then start the sandbox.
 */
const SPAWN_CREATE_WAIT_MS = 30 * 60_000;
/** Replay window: activity retries of one spawn arrive within it. One row per spawn. */
const SPAWN_CREATE_RECORD_TTL_MS = 6 * 60 * 60_000;

/** The replay key: only a spawn tool call's lease owner identifies one spawn. */
function spawnCreateKey(leaseOwner: string | undefined): string | null {
  return leaseOwner !== undefined && leaseOwner.startsWith('tool:') ? leaseOwner : null;
}

/** The spawn's stored create when its task is still running on the requested provider. */
async function replayedSpawnCreate<T extends SpawnCreateResult>(
  key: string,
  codeProvider: string
): Promise<{ session: T; fields: CodeSessionSpawnFields } | null> {
  const stored = await getEphemeral<{ session: T; fields: CodeSessionSpawnFields }>(
    SPAWN_CREATE_NAMESPACE,
    key
  );
  if (stored === undefined || stored.fields?.code_provider !== codeProvider) return null;
  // A record that cannot name its task (an older shape, or a create that returned none) has nothing to replay;
  // a unique lookup on an undefined id is a Prisma validation error, not an absent task.
  const taskId: unknown = stored.session?.taskId;
  if (typeof taskId !== 'string' || taskId === '') return null;
  const task = await prisma.claudeCodeSession.findUnique({
    where: { id: taskId },
    select: { status: true },
  });
  if (task?.status !== 'ACTIVE') return null;
  log.info('Replaying a retried spawn create: its task is still running', {
    taskId: stored.session.taskId,
    codeSessionId: stored.fields.code_session_id ?? undefined,
    leaseOwner: key,
  });
  return stored;
}

/**
 * Resolve the spawn's code session and run the sandbox create once per spawn:
 * a retried create of the same tool call gets the task its first request
 * started (see the header).
 */
export async function createWithCodeSession<T extends SpawnCreateResult>(
  params: ResolveSpawnParams,
  create: (codeSession: SpawnCodeSession | null, handoff: string | null) => Promise<T>
): Promise<{ session: T; fields: CodeSessionSpawnFields }> {
  const key = spawnCreateKey(params.leaseOwner);
  if (key === null) return createSpawnTask(params, create);
  return withSerializedLock(
    `${SPAWN_CREATE_NAMESPACE}:${key}`,
    async () => {
      const replayed = await replayedSpawnCreate<T>(key, params.codeProvider);
      if (replayed !== null) {
        await deliverStartupTurns(key, replayed.session.taskId);
        return replayed;
      }
      const created = await createSpawnTask(params, create);
      await putEphemeral(
        SPAWN_CREATE_NAMESPACE,
        key,
        created as unknown as Prisma.InputJsonValue,
        SPAWN_CREATE_RECORD_TTL_MS
      ).catch((error: unknown) => {
        // The task exists and this response carries it; only a later retry loses its replay.
        log.error('Storing the spawn create for retries failed', {
          taskId: created.session.taskId,
          leaseOwner: key,
          error: error instanceof Error ? error.message : String(error),
        });
      });
      await deliverStartupTurns(key, created.session.taskId);
      return created;
    },
    { waitMs: SPAWN_CREATE_WAIT_MS }
  );
}

/**
 * Messages sent to this spawn before its session existed go to it now (see
 * startupTurns.ts). The task is running, so a failure here is logged and the
 * create still succeeds; a replayed create runs the drain again.
 */
async function deliverStartupTurns(key: string, taskId: string): Promise<void> {
  await drainStartupTurns(key, taskId).catch((error: unknown) => {
    log.error('Delivering the messages queued during startup failed', {
      taskId,
      leaseOwner: key,
      error: error instanceof Error ? error.message : String(error),
    });
  });
}

/**
 * Resolve the spawn's code session, run the sandbox create, and release the
 * lease when the create fails so a retry (or the next spawn) can resume.
 */
async function createSpawnTask<T extends SpawnCreateResult>(
  params: ResolveSpawnParams,
  create: (codeSession: SpawnCodeSession | null, handoff: string | null) => Promise<T>
): Promise<{ session: T; fields: CodeSessionSpawnFields }> {
  let codeSession: SpawnCodeSession | null = await resolveSpawnCodeSession(params);
  // C15: a retry after a usage limit may run on the other provider. It keeps
  // the session (branch, commits, files via the checkpoint) and gets a handoff
  // prompt instead of the other CLI's conversation; the previous CLI's session
  // log rides on the session handle, and the sandbox acquire writes it into
  // the new sandbox at the path the prompt names.
  let handoff: string | null = null;
  let switchedFrom: string | null = null;
  if (codeSession !== null && codeSession.codeProvider !== params.codeProvider) {
    switchedFrom = codeSession.codeProvider;
    // A concurrent duplicate loses with CodeSessionBusyError and keeps the
    // winner's lease (same owner), so exactly one spawn rotates the session.
    const switched = await switchCodeSessionProvider(codeSession, params.codeProvider, (id) =>
      Sandbox.kill(id, { apiKey: process.env.E2B_API_KEY }).then(() => undefined)
    );
    codeSession = {
      ...switched.session,
      handoffTrajectory: switched.trajectory,
      retainFailedStartLease: true,
    };
    handoff = switched.handoff;
  }
  let session: T;
  try {
    session = await create(codeSession, handoff);
  } catch (error) {
    // A start that failed after the switch hands the session back, so the
    // retry switches again and gets the handoff and trajectory once more.
    if (
      codeSession !== null &&
      switchedFrom !== null &&
      !(error instanceof CodeSessionPauseUnconfirmedError)
    ) {
      await revertCodeSessionProvider(
        codeSession.id,
        params.codeProvider,
        switchedFrom,
        codeSession.leaseOwner,
        (id) => Sandbox.kill(id, { apiKey: process.env.E2B_API_KEY }).then(() => undefined)
      ).catch((cleanupError: unknown) => {
        log.error('Provider switch rollback failed; retaining the original start error', {
          codeSessionId: codeSession?.id,
          startError: error instanceof Error ? error.message : String(error),
          cleanupError: cleanupError instanceof Error ? cleanupError.message : String(cleanupError),
        });
      });
    }
    // Refused because the user holds the live CLI in direct chat: the session is
    // live, so only this spawn's lease is dropped (marking it PAUSED would read as
    // stopped and let another spawn take the lease and stop that ws-server).
    if (codeSession !== null && error instanceof CodeSessionHeldError) {
      await releaseLease(codeSession.id, codeSession.leaseOwner).catch((cleanupError: unknown) => {
        log.error('Held-session lease release failed; retaining the original start error', {
          codeSessionId: codeSession?.id,
          cleanupError: cleanupError instanceof Error ? cleanupError.message : String(cleanupError),
        });
      });
    } else if (
      codeSession !== null &&
      switchedFrom === null &&
      !(error instanceof CodeSessionPauseUnconfirmedError)
    ) {
      await markPaused(codeSession.id).catch((cleanupError: unknown) => {
        log.error('Failed-start pause failed; retaining the original start error', {
          codeSessionId: codeSession?.id,
          startError: error instanceof Error ? error.message : String(error),
          cleanupError: cleanupError instanceof Error ? cleanupError.message : String(cleanupError),
        });
      });
    }
    throw error;
  }
  return {
    session,
    fields: {
      code_session_id: codeSession?.id ?? null,
      root_agent_id: codeSession?.rootAgentId ?? null,
      root_preview_path: codeSession ? rootPreviewPath(codeSession.rootAgentId) : null,
      resumed_from: session.resumedFrom ?? 'new',
      cli_session_id: session.cliSessionId ?? null,
      branch: codeSession?.branch ?? null,
      base_branch: codeSession?.baseBranch ?? null,
      merge_target: codeSession?.mergeTarget ?? null,
      parent_code_session_id: codeSession?.parentCodeSessionId ?? null,
      code_provider: codeSession?.codeProvider ?? params.codeProvider,
      provider_handoff: handoff !== null,
      sandbox_size:
        session.sandboxSize !== undefined && session.sandboxSize !== null
          ? sizeView(session.sandboxSize)
          : null,
      sandbox_sizing: sizingFields(session),
      history_events: session.historyEvents ?? 0,
      startup:
        session.startup !== undefined
          ? { ws_ready_ms: session.startup.wsReadyMs, clone_ms: session.startup.cloneMs }
          : null,
    },
  };
}

function sizingFields(session: SpawnCreateResult): CodeSessionSpawnFields['sandbox_sizing'] {
  const report = session.sandboxSizing ?? null;
  if (report === null) return null;
  return {
    requested: report.requested !== null ? sizeView(report.requested) : null,
    initial: session.sandboxSize ? sizeView(session.sandboxSize) : null,
    source: report.source,
  };
}

/** Session facts every create response (and so the tool's first trajectory entry) carries. */
export interface CodeSessionSpawnFields {
  /** Allocation-free stable entry prepared on the normal code-session spawn. */
  root_agent_id: string | null;
  root_preview_path: string | null;
  code_session_id: string | null;
  resumed_from: string;
  cli_session_id: string | null;
  /** Branch the agent works on (null for repo-less sessions). */
  branch: string | null;
  /** Ref the branch was cut from; null = the repo's default branch. */
  base_branch: string | null;
  /** Where the branch merges; null = the repo's default branch. */
  merge_target: string | null;
  parent_code_session_id: string | null;
  /** Provider this spawn runs on (a usage-limit retry may switch it, C15). */
  code_provider: string;
  /** True when this spawn took the session over from the other provider. */
  provider_handoff: boolean;
  /** vCPUs / memory of the E2B sandbox (C16); null for a local session. */
  sandbox_size: SizeView | null;
  /**
   * Requested vs observed size: `requested` is the spawn's explicit sandbox_size
   * (null: none), `initial` the size this spawn's sandbox started on (null:
   * unknown or local), `source` where that size came from. A resumed session
   * keeps its size, so a request it did not apply reads source 'session'.
   */
  sandbox_sizing: {
    requested: SizeView | null;
    initial: SizeView | null;
    source: SpawnSizingReport['source'];
  } | null;
  /**
   * resumed_from=live_cli: the ws-server's events before this spawn's goal turn
   * (the earlier turns of the live CLI); the monitor treats them as history.
   */
  history_events: number;
  /** Startup phases of the ws-server this spawn started (null: it attached to a live one). */
  startup: { ws_ready_ms: number; clone_ms: number | null } | null;
}

export interface CodeSessionView {
  code_session_id: string;
  root_agent_id: string;
  owner_agent_id: string;
  repo_url: string | null;
  code_provider: string;
  state: string;
  checkpoint_seq: number;
  cli_session_id: string | null;
  branch: string | null;
  base_branch: string | null;
  merge_target: string | null;
  parent_code_session_id: string | null;
  repo: unknown;
  last_active_at: Date;
}

function view(row: CodeSessionRow): CodeSessionView {
  return {
    code_session_id: row.id,
    root_agent_id: row.rootAgentId,
    owner_agent_id: row.ownerAgentId,
    repo_url: row.repoUrl,
    code_provider: row.codeProvider,
    state: row.state,
    checkpoint_seq: row.latestCheckpointSeq,
    cli_session_id: row.cliSessionId,
    branch: row.branch,
    base_branch: row.baseBranch,
    merge_target: row.mergeTarget,
    parent_code_session_id: row.parentCodeSessionId,
    repo: row.lastRepoFacts,
    last_active_at: row.lastActiveAt,
  };
}

export async function getOwnedCodeSessionView(
  userId: string,
  codeSessionId: string
): Promise<CodeSessionView> {
  const row = await findOwnedCodeSession(userId, codeSessionId);
  if (row === null) throw new CodeSessionNotFoundError(codeSessionId);
  return view(row);
}

/** Explicit close: kill the sandbox, delete checkpoints, retire the session. */
export async function retireOwnedCodeSession(
  userId: string,
  codeSessionId: string
): Promise<CodeSessionView> {
  const row = await findOwnedCodeSession(userId, codeSessionId);
  if (row === null) throw new CodeSessionNotFoundError(codeSessionId);
  if (row.state !== 'RETIRED') await retireCodeSession(row);
  return view((await findOwnedCodeSession(userId, codeSessionId)) ?? row);
}

async function sampleAndPublish(
  task: { id: string; agentId: string; sandboxId: string },
  resources: CodeResourceDeps
): Promise<CodeResourceSample | null> {
  const sample = await resources.sample(task.sandboxId);
  if (sample !== null) {
    await resources.publish(task.agentId, task.id, sample).catch((error: unknown) => {
      log.warn('Resource sample broadcast failed', { taskId: task.id, error: String(error) });
    });
  }
  return sample;
}

/**
 * Keepalive from the attached code-agent activity (liveness is heartbeat-only;
 * there is no runtime limit): extend the session lease and the E2B sandbox's
 * idle window, and save the session periodically (cycling the sandbox before
 * E2B's continuous-runtime cap; lifecycle.ts). Once keepalives stop, the
 * sandbox pauses after the window. The lease is extended first: a failed E2B
 * call must not let it lapse, or another spawn could take the session from
 * under the running CLI.
 */
export async function keepAliveCodeTask(
  taskId: string,
  setTimeoutFn: (sandboxId: string, timeoutMs: number) => Promise<void> = (sandboxId, timeoutMs) =>
    Sandbox.setTimeout(sandboxId, timeoutMs, { apiKey: process.env.E2B_API_KEY }),
  resources: CodeResourceDeps = codeResourceDeps,
  maintenance: CodeSessionMaintenanceDeps = defaultCodeSessionMaintenanceDeps
): Promise<{
  alive: boolean;
  code_session_id: string | null;
  /** A pending resize the attached activity applies (C16); present only when there is one. */
  resize_requested?: { request_id: string; from: SizeView; to: SizeView };
  /** Latest CPU / memory / disk sample of the sandbox (item 29); present only when sampled. */
  resources?: CodeResourceSample;
}> {
  const task = await prisma.claudeCodeSession.findUnique({ where: { id: taskId } });
  if (task === null || task.status === 'CLOSED') {
    forgetCodeSessionMaintenance(task?.codeSessionId ?? null);
    return { alive: false, code_session_id: task?.codeSessionId ?? null };
  }
  // Only an E2B sandbox has an idle window; a Mac session (C14) lives while its
  // CLI runs and the Rose Mac app is connected.
  // This keepalive is also the one resource sampler per running session (item 29):
  // a Mac session has no sandbox to measure. The sample runs alongside the lease
  // and resize reads so its bounded E2B read never delays them.
  const e2b =
    !isLocalClaudeCodeSandboxId(task.sandboxId) && !isLocalDeviceSandboxId(task.sandboxId);
  if (task.codeSessionId !== null) await extendLease(task.codeSessionId);
  if (e2b) {
    await setTimeoutFn(task.sandboxId, CODE_SANDBOX_IDLE_TIMEOUT_MS);
    scheduleCodeSessionMaintenance(task, maintenance);
  }
  const [sample, resize] = await Promise.all([
    e2b ? sampleAndPublish(task, resources) : Promise.resolve(null),
    pendingResizeForTask(task.codeSessionId),
  ]);
  return {
    alive: true,
    code_session_id: task.codeSessionId,
    ...(resize !== null ? { resize_requested: resize } : {}),
    ...(sample !== null ? { resources: sample } : {}),
  };
}

/**
 * Checkpoint a still-running task's session without pausing it (worker drain:
 * the sandbox keeps running for the reattaching attempt).
 */
export async function checkpointRunningTask(
  taskId: string,
  reason: string
): Promise<{ checkpoint_seq: number | null }> {
  const task = await prisma.claudeCodeSession.findUnique({ where: { id: taskId } });
  if (task === null || task.codeSessionId === null) return { checkpoint_seq: null };
  const session = await prisma.codeSession.findUnique({ where: { id: task.codeSessionId } });
  if (session === null || session.state === 'RETIRED') return { checkpoint_seq: null };
  let result = null;
  if (isLocalClaudeCodeSandboxId(task.sandboxId)) {
    if (session.homeDir !== null) {
      result = await checkpointCodeSession({
        session,
        fs: localSandboxFs(session.homeDir),
        reason,
        agentId: task.agentId,
      });
    }
  } else {
    const sandbox = await Sandbox.connect(task.sandboxId, { apiKey: process.env.E2B_API_KEY });
    result = await checkpointCodeSession({
      session,
      fs: e2bSandboxFs(sandbox),
      reason,
      agentId: task.agentId,
    });
  }
  return { checkpoint_seq: result?.seq ?? null };
}
