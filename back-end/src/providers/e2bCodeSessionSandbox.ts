import { prisma } from './database';
/**
 * Persistent code-session sandbox acquisition for E2B Claude Code / Codex
 * sessions (see providers/codeSessions/lifecycle.ts for the full flow).
 */
import { createHash } from 'crypto';
import type { Sandbox } from 'e2b';
import type { FilePointer } from './storage';
import { createLogger } from './logger';
import {
  createLocalClaudeCodeSessionIfBillingBlocked,
  isLocalClaudeCodeFallbackEligible,
  resolveLocalSessionWorkDir,
} from './claudeCodeLocalSession';
import { agentToolsForLocalHome } from './agentTools/workspace';
import type { UserUiWorkspace } from './userUi/workspace';
import type {
  ProjectRestoreReport,
  ProjectTemplateSnapshot,
  ProjectWorkspace,
} from './projects/types';
import { moveToResizedSandbox } from './codeSessions/resizeMove';
import {
  connectPausedSandbox,
  prepareLocalCodeSessionHome,
  restoreLatestCheckpoint,
  type ResumedFrom,
} from './codeSessions/lifecycle';
import {
  heldSandboxSize,
  resolveSpawnSandbox,
  sameSize,
  defaultSandboxSize,
  type SandboxProvider,
  type SandboxSize,
} from './codeSessions/sandboxSize';
import { e2bSandboxFs, localSandboxFs } from './codeSessions/sandboxFs';
import { writeHandoffTrajectory, type SpawnCodeSession } from './codeSessions/usageLimit';
import { createSandboxWithRetry } from './e2bSandboxCreate';
import { withClaudeCodeCreateSlot } from './e2bCreateSlot';
import { NATIVE_CLI_PATH, resolveNativeCliTemplate } from './nativeCliTemplateBuilder';
import {
  codeSessionTurnToken,
  deliverUserTurnToCodeSession,
  type CodeTurnSettings,
} from './codeSessionTurns';
import {
  linkClaudeCodeSession,
  markPaused,
  recordSandbox,
  type CodeSessionRow,
} from './codeSessions/store';

const log = createLogger('E2B');
const WS_PORT = 3000;
/**
 * How long a sandbox nobody is attached to stays up. The attached activity
 * keeps extending it (POST /:taskId/keepalive), so this is an idle window,
 * not a runtime limit: on expiry a persistent session's sandbox pauses
 * (lifecycle onTimeout=pause) and resumes on the next spawn.
 */
export const CODE_SANDBOX_IDLE_TIMEOUT_MS = 30 * 60 * 1000;
const SANDBOX_TIMEOUT_MS = CODE_SANDBOX_IDLE_TIMEOUT_MS;

export interface CodeAgentSessionResult {
  wsUrl: string;
  taskId: string;
  sandboxId: string;
  uploadedFiles: string[];
  /** Where the persistent code session's state came from. */
  resumedFrom?: ResumedFrom;
  /** CLI conversation the session resumed (`claude --resume` / `codex resume`). */
  cliSessionId?: string | null;
  /** Restore summary when the session materialized a Project (PJ3). */
  projectRestore?: ProjectRestoreReport | null;
  /** Public URL of the session's live dev-server preview (Rose dev accounts, A1). */
  previewUrl?: string | null;
  /** vCPUs / memory of the E2B sandbox (C16); absent for local sessions. */
  sandboxSize?: SandboxSize | null;
  /** Attached to a live CLI: its events before this spawn's goal are history. */
  historyEvents?: number;
  /** How long the ws-server took to serve, and the repo clone inside that (null: none). */
  startup?: { wsReadyMs: number; cloneMs: number | null };
}

export type { ClaudeCodeSessionOptions } from './codeSessionLaunchOptions';

/** Sizing of a spawn (C16): the size it wants and the agent to attribute checkpoints to. */
export interface SpawnSizing {
  provider?: SandboxProvider;
  size?: SandboxSize;
  agentId?: string;
}

/**
 * Persistent code sessions resume their paused sandbox (filesystem + memory).
 * When it is gone, a new sandbox is created and the latest verified
 * checkpoint restored into it before the ws-server starts.
 *
 * C16: when the spawn wants another size than the held sandbox has, the held
 * sandbox is checkpointed, a new one of the wanted size restores exactly that
 * checkpoint and is validated, and only then does the row move to it
 * (codeSessions/resizeMove.ts). A failed save, restore or validation keeps the
 * held sandbox (paused) and throws, so nothing is lost.
 */
export async function acquireCodeSessionSandbox(
  codeSession: CodeSessionRow | null,
  create: () => Promise<Sandbox>,
  sizing: SpawnSizing = {}
): Promise<{ sandbox: Sandbox; resumedFrom: ResumedFrom; cliSessionId: string | null }> {
  const session = codeSession;
  if (session?.sandboxKind === 'e2b' && session.sandboxId !== null) {
    const sandboxId = session.sandboxId;
    const resume = () => connectPausedSandbox(sandboxId, SANDBOX_TIMEOUT_MS);
    const resumed = await (sizing.provider === 'claude'
      ? withClaudeCodeCreateSlot(resume)
      : resume());
    if (resumed !== null) {
      // The resumed sandbox may still run the paused CLI (E2B keeps memory and
      // processes): the spawn either attaches to it (attachLiveCodeCli) or
      // stops it before starting another one. Only a resize stops it here.
      if (sizing.size === undefined || sameSize(sizing.size, heldSandboxSize(session))) {
        return {
          sandbox: resumed,
          resumedFrom: 'paused_sandbox',
          cliSessionId: session.cliSessionId,
        };
      }
      await stopStaleWsServer(resumed);
      const moved = await moveToResizedSandbox(session, resumed, create, sizing);
      return { ...moved, resumedFrom: 'checkpoint' };
    }
  }
  const sandbox = await create();
  if (session !== null && session.latestCheckpointSeq > 0) {
    const restored = await restoreLatestCheckpoint(session, e2bSandboxFs(sandbox));
    if (restored !== null) {
      await recordNewSandbox(session, sandbox, sizing);
      return { sandbox, resumedFrom: 'checkpoint', cliSessionId: restored.cliSessionId };
    }
  }
  if (session !== null) await recordNewSandbox(session, sandbox, sizing);
  return { sandbox, resumedFrom: 'new', cliSessionId: null };
}

/** The row names the sandbox the session holds, with its size, as soon as it exists (C16). */
async function recordNewSandbox(
  session: CodeSessionRow,
  sandbox: Sandbox,
  sizing: SpawnSizing
): Promise<void> {
  if (sizing.size === undefined) return;
  await recordSandbox(session.id, {
    sandboxKind: 'e2b',
    sandboxId: sandbox.sandboxId,
    size: sizing.size,
  });
}

/** Caller identity may supply a new spawn owner but never replace session authority. */
export function codeSandboxUserId(
  codeSession: CodeSessionRow | null,
  userId?: string
): string | undefined {
  if (codeSession !== null && userId !== undefined && codeSession.userId !== userId) {
    throw new Error('Code session does not belong to the spawning user');
  }
  return codeSession?.userId ?? userId;
}

/**
 * Acquire the sandbox for a spawn: resume the session's paused sandbox, else
 * create one (restoring the latest checkpoint). With a `provider`, the
 * session's size (C16: an active resize, the held size, or a new session's
 * project default) picks the template; `templateId` is the unsized fallback.
 * The timeout is the idle window every keepalive extends, not a runtime cap.
 */
export async function acquireCodeSandbox(
  codeSession: SpawnCodeSession | null,
  templateId: string,
  envs: Record<string, string>,
  label: string,
  sizing: {
    provider?: SandboxProvider;
    projectSize?: unknown;
    agentId?: string;
    userId?: string;
    /**
     * The session's project: its default size, and its published template
     * snapshot, which a BRAND-NEW session sandbox starts from.
     */
    project?: ProjectWorkspace['project'] | null;
    snapshot?: ProjectTemplateSnapshot | null;
  } = {}
): Promise<{
  sandbox: Sandbox;
  resumedFrom: ResumedFrom;
  cliSessionId: string | null;
  size?: SandboxSize;
  /** The sandbox was created from the project's template snapshot. */
  fromSnapshot: boolean;
}> {
  const userId = codeSandboxUserId(codeSession, sizing.userId);
  const sized =
    sizing.provider !== undefined
      ? await resolveSpawnSandbox(
          codeSession,
          sizing.provider,
          sizing.project?.sandboxSize ?? sizing.projectSize ?? null
        )
      : null;
  const spawnSizing: SpawnSizing = {
    ...(sized !== null ? { size: sized.size } : {}),
    agentId: sizing.agentId,
    provider: sizing.provider,
  };
  const createOptions = {
    apiKey: process.env.E2B_API_KEY,
    timeoutMs: SANDBOX_TIMEOUT_MS,
    envs: { ...envs, PATH: NATIVE_CLI_PATH },
    ...codeSessionCreateOptions(codeSession),
    ...(codeSession !== null ? { metadata: codeSessionSandboxMetadata(codeSession) } : {}),
  };
  // Resolve on the exact base image, including project snapshots, before reserving
  // the runtime sandbox. Template builds have their own bounded creation slot.
  const createWithNativeCli = async (baseTemplateId: string, size: SandboxSize) => {
    const selectedTemplateId =
      userId && sizing.provider
        ? await resolveNativeCliTemplate(userId, sizing.provider, size, baseTemplateId)
        : baseTemplateId;
    const create = () => createSandboxWithRetry(selectedTemplateId, createOptions, label);
    return sizing.provider === 'claude' ? withClaudeCodeCreateSlot(create) : create();
  };
  let fromSnapshot = false;
  const acquired = await acquireCodeSessionSandbox(
    codeSession,
    async () => {
      const snapshot = sizing.snapshot ?? sizing.project?.templateSnapshot ?? null;
      // Project maintenance builds Claude images (maintenanceE2b.ts). A Codex
      // ws-server overlay cannot supply the missing Codex binary; its workspace
      // must instead be restored onto the Codex provider's own image.
      // A lost sandbox being replaced or resized keeps the base path.
      if (
        sizing.provider === 'claude' &&
        snapshot !== null &&
        (codeSession === null || codeSession.sandboxId === null)
      ) {
        try {
          const sandbox = await createWithNativeCli(
            snapshot.id,
            snapshot.size ?? sized?.size ?? defaultSandboxSize()
          );
          fromSnapshot = true;
          // The held size is the snapshot's: a later resume must not read it as a resize.
          if (snapshot.size !== null && spawnSizing.size !== undefined) {
            spawnSizing.size = snapshot.size;
          }
          return sandbox;
        } catch (error) {
          log.warn('Project template snapshot could not start; using the base template', {
            snapshotId: snapshot.id,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
      if (sized !== null && sized.templateId === null) {
        // The unsized fallback is a fixed-size image: never hand it to another size.
        throw new Error(
          `No ${label} sandbox template is built for ${sized.size.cpu} vCPU / ${sized.size.memoryGb} GB; configure E2B_SANDBOX_TEMPLATES for that size`
        );
      }
      return createWithNativeCli(
        sized?.templateId ?? templateId,
        sized?.size ?? defaultSandboxSize()
      );
    },
    spawnSizing
  );
  const trajectory = codeSession?.handoffTrajectory ?? null;
  if (trajectory !== null) {
    // C15 provider switch: after the checkpoint restore, so the previous CLI's
    // session log is in the HOME the new CLI starts in, where the handoff points.
    // The sandbox is new (a switch drops the old one); on a failed write it is
    // killed and the spawn fails, so no CLI starts on a prompt naming a missing file.
    try {
      await writeHandoffTrajectory(e2bSandboxFs(acquired.sandbox), trajectory);
    } catch (error) {
      await acquired.sandbox.kill().catch(() => undefined);
      throw error;
    }
  }
  return {
    ...acquired,
    ...(spawnSizing.size !== undefined ? { size: spawnSizing.size } : {}),
    fromSnapshot,
  };
}

/**
 * Env a running Claude Code CLI cannot adopt and must restart for: its
 * credential (claudeCodeSandboxAuth.ts + gateway base) and its git commit
 * identity (codeAgentGitIdentity.ts). A CLI started with another identity,
 * or none (before the pushing account was injected), would keep committing
 * as an author GitHub cannot link, which Vercel refuses to deploy.
 */
export const CLAUDE_CREDENTIAL_ENV_KEYS = [
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_BASE_URL',
  'CLAUDE_CODE_CREDENTIALS',
  'CLAUDE_CODE_OAUTH_TOKEN',
  'GIT_AUTHOR_EMAIL',
  'GIT_AUTHOR_NAME',
  'GIT_COMMITTER_EMAIL',
  'GIT_COMMITTER_NAME',
] as const;

/**
 * Other start-time env of a Claude Code spawn. A running CLI cannot adopt
 * them; the ws-server reports the ones whose value differs (logged).
 */
export const CLAUDE_CONFIG_ENV_KEYS = [
  'CODE_AGENT_SYSTEM_PROMPT',
  'CODE_AGENT_BRANCH',
  'CODE_AGENT_BASE_BRANCH',
  'CODE_AGENT_MERGE_TARGET',
  'GITHUB_REPO_URL',
] as const;

function envDigest(value: unknown): string {
  return createHash('sha256')
    .update(JSON.stringify(value ?? null))
    .digest('hex');
}

/** Digest of `keys` in `envs`; the ws-server computes the same over its own env. */
export function credentialEnvDigest(
  keys: readonly string[],
  envs: Record<string, string | undefined>
): { keys: string[]; digest: string } {
  const names = [...new Set(keys)].sort();
  return { keys: names, digest: envDigest(names.map((key) => [key, envs[key] ?? null])) };
}

/**
 * The turn id of a spawn's goal. A tool call's lease owner
 * (`tool:<agent>:<tool_call_id>`) is the same on every retry of that call, so a
 * retried create gets `duplicate`; any other owner (`agent:<id>` is shared by
 * every spawn of that agent) falls back to this spawn's own task id.
 */
function spawnTurnId(codeSession: CodeSessionRow | null, taskId: string): string {
  const owner = codeSession?.leaseOwner ?? null;
  const key = owner !== null && owner.startsWith('tool:') ? owner : taskId;
  return `spawn-${createHash('sha256').update(key).digest('hex').slice(0, 40)}`;
}

/**
 * A resumed spawn keeps the paused CLI when it can (C1: memory too when the
 * sandbox pause survived). The live ws-server still serves the session's
 * latest task; the spawn's goal is delivered to it as an agent turn carrying
 * this spawn's turn token, model/effort/speed (`settings`) and credential
 * digest:
 *
 *   delivered / duplicate with `attached: true`
 *                         ─► attached: the in-flight turn, tools and background
 *                            processes continue; the goal is the CLI's next turn
 *   queued (reason direct_hold) with `attached: true`
 *                         ─► attached: the user is chatting with the CLI
 *                            directly; the goal is its first turn after the
 *                            hand back
 *   relay_paused (a ws-server built before held attaches; the user holds the
 *   CLI)                  ─► the live ws-server is NEVER stopped (that would
 *                            end the user's chat and drop held turns): throws
 *                            CodeSessionHeldError; the spawn's retry attaches
 *                            after the hand back
 *   session_ended (stdin closed, CLI exited), credential_changed (the spawn
 *   leased another credential than the CLI holds), unsupported_settings,
 *   unreachable, any other refusal, an ack without `attached` (a ws-server
 *   started before this support), or no previous task of this provider on
 *   the sandbox
 *                         ─► the stale ws-server is stopped; the caller starts
 *                            a new one (`--resume` of the conversation)
 *
 * The decision is the ws-server's structured ack, never output text. Like
 * Codex's resume of a running thread, start-time config the running CLI
 * cannot adopt is kept and logged (`ignored_config`).
 */
export async function attachLiveCodeCli(
  sandbox: Sandbox,
  codeSession: CodeSessionRow | null,
  params: {
    provider: 'claude';
    wsUrl: string;
    taskId: string;
    text: string;
    envs: Record<string, string | undefined>;
    credentialKeys: readonly string[];
    configKeys?: readonly string[];
  },
  turnToken: (taskId: string) => string = codeSessionTurnToken
): Promise<{ attached: boolean; reason: string; historyEvents?: number }> {
  let reason: string;
  try {
    const live = await deliverGoalToLiveCli(sandbox, codeSession, params, turnToken);
    if (live.attached) return live;
    reason = live.reason;
  } catch (error) {
    if (error instanceof CodeSessionHeldError) throw error;
    // No turn token (BACKEND_API_KEY unset), a failed task lookup or delivery:
    // the restart path works without any of them, as before live attach.
    reason = `attach_error: ${error instanceof Error ? error.message : String(error)}`;
  }
  log.info('Restarting the code-session CLI on resume', {
    codeSessionId: codeSession?.id,
    taskId: params.taskId,
    reason: reason.slice(0, 200),
  });
  await stopStaleWsServer(sandbox);
  return { attached: false, reason };
}

/**
 * The user holds the live CLI in direct chat and its ws-server cannot hold an
 * attaching spawn (an older build): the session must not be stopped, so the
 * create fails retryably and the spawn attaches after the hand back.
 */
export class CodeSessionHeldError extends Error {
  constructor() {
    super(
      'The user is chatting with this code session directly; the spawn attaches after they hand it back'
    );
    this.name = 'CodeSessionHeldError';
  }
}

/** One attach attempt; throws on a missing token or a failed lookup (the caller restarts). */
async function deliverGoalToLiveCli(
  sandbox: Sandbox,
  codeSession: CodeSessionRow | null,
  params: Parameters<typeof attachLiveCodeCli>[2],
  turnToken: (taskId: string) => string
): Promise<{ attached: boolean; reason: string; historyEvents?: number }> {
  if (codeSession === null) return { attached: false, reason: 'no_previous_task' };
  const previous = await prisma.claudeCodeSession.findFirst({
    where: { codeSessionId: codeSession.id, sandboxId: sandbox.sandboxId },
    orderBy: { createdAt: 'desc' },
    select: { id: true, codeProvider: true },
  });
  if (previous === null || (previous.codeProvider ?? 'claude') !== params.provider) {
    return { attached: false, reason: 'no_previous_task' };
  }
  const { envs } = params;
  const speed = envs.CLAUDE_CODE_SPEED;
  const settings: CodeTurnSettings = {
    ...(envs.CLAUDE_CODE_MODEL ? { model: envs.CLAUDE_CODE_MODEL } : {}),
    ...(envs.CLAUDE_CODE_EFFORT ? { effort: envs.CLAUDE_CODE_EFFORT } : {}),
    ...(speed === 'fast' || speed === 'standard' ? { speed } : {}),
  };
  const delivery = await deliverUserTurnToCodeSession({
    wsUrl: params.wsUrl,
    taskId: previous.id,
    turnId: spawnTurnId(codeSession, params.taskId),
    text: params.text,
    sender: 'agent',
    settings,
    attach: {
      token: turnToken(params.taskId),
      credential: credentialEnvDigest(params.credentialKeys, envs),
      config: Object.fromEntries(
        (params.configKeys ?? []).map((key) => [key, envDigest(envs[key])])
      ),
    },
  });
  if (delivery.status === 'rejected' && delivery.reason === 'relay_paused') {
    throw new CodeSessionHeldError();
  }
  // Held for the user's direct chat: attached, the goal waits for the hand back.
  const held = delivery.status === 'queued' && delivery.reason === 'direct_hold';
  const reached = delivery.status === 'delivered' || delivery.status === 'duplicate' || held;
  if (!reached || delivery.attached !== true) {
    return {
      attached: false,
      reason: reached ? 'attach_unsupported' : (delivery.reason ?? delivery.status),
    };
  }
  log.info('Spawn attached to the live code-session CLI', {
    codeSessionId: codeSession.id,
    previousTaskId: previous.id,
    taskId: params.taskId,
    status: delivery.status,
    ignoredConfig: (delivery.ignored_config ?? []).join(','),
  });
  return {
    attached: true,
    reason: delivery.status,
    ...(delivery.history_events !== undefined ? { historyEvents: delivery.history_events } : {}),
  };
}

/**
 * A resumed sandbox still runs the previous (lingering) ws-server on WS_PORT.
 * Stopping it also ends its CLI and that CLI's processes (E2B's pause itself
 * keeps them; live probe 2026-11-04), so only a spawn that cannot attach to
 * that CLI (attachLiveCodeCli) or moves the session (resize) stops it. The
 * `[n]ode` pattern keeps `pkill -f` from matching, and killing, the shell
 * running this command, which skipped the wait for the port.
 */
export async function stopStaleWsServer(sandbox: Sandbox): Promise<void> {
  try {
    await sandbox.commands.run(
      `pkill -f '[n]ode /home/user/ws-server.mjs' || true; for i in $(seq 1 50); do netstat -tln | grep -q ':${WS_PORT} ' || exit 0; sleep 0.1; done; exit 0`,
      { timeoutMs: 10_000 }
    );
  } catch (error) {
    log.warn('Stopping the resumed sandbox ws-server failed', {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

/** Task-row fields tying a spawn to its persistent session and CLI provider. */
export function codeSessionTaskFields(
  codeSession: CodeSessionRow | null,
  codeProvider: 'claude' | 'codex'
): { codeSessionId: string | null; codeProvider: string } {
  return { codeSessionId: codeSession?.id ?? null, codeProvider };
}

/** Remember which E2B sandbox now holds the session (resume connects to it). */
export async function recordE2bCodeSessionSandbox(
  codeSession: CodeSessionRow | null,
  sandboxId: string
): Promise<void> {
  if (codeSession !== null) await recordSandbox(codeSession.id, { sandboxKind: 'e2b', sandboxId });
}

/** Cleanup did not confirm a pause; retain the session lease for reconciliation. */
export class CodeSessionPauseUnconfirmedError extends Error {
  constructor(cause: unknown) {
    super(
      `Code session pause unconfirmed: ${cause instanceof Error ? cause.message : String(cause)}`,
      { cause }
    );
    this.name = 'CodeSessionPauseUnconfirmedError';
  }
}

/** A persistent session's sandbox is paused (never killed) on a failed start. */
export async function releaseSandboxAfterFailedStart(
  sandbox: Sandbox,
  codeSession: SpawnCodeSession | null
): Promise<void> {
  if (codeSession === null) {
    await sandbox.kill();
    return;
  }
  try {
    await recordSandbox(codeSession.id, { sandboxKind: 'e2b', sandboxId: sandbox.sandboxId });
    await sandbox.pause();
  } catch (error) {
    log.warn('Pausing the sandbox after a failed start failed', {
      error: error instanceof Error ? error.message : String(error),
    });
    throw new CodeSessionPauseUnconfirmedError(error);
  }
  if (codeSession.retainFailedStartLease === true) {
    // Keep this attempt's unique switch custody until the outer rollback finishes.
    await prisma.codeSession.updateMany({
      where: {
        id: codeSession.id,
        leaseOwner: codeSession.leaseOwner,
        codeProvider: codeSession.codeProvider,
        sandboxId: sandbox.sandboxId,
        state: { not: 'RETIRED' },
      },
      data: { state: 'PAUSED', lastActiveAt: new Date() },
    });
  } else {
    await markPaused(codeSession.id);
  }
}

/**
 * A project restore that failed. A persistent session's sandbox may hold work
 * newer than its last checkpoint (a resumed sandbox keeps its filesystem), so
 * it is paused like any failed start and the retry resumes it; only a one-shot
 * sandbox is killed, with the restore error kept if the kill itself fails.
 */
export async function releaseSandboxAfterFailedRestore(
  sandbox: Sandbox,
  codeSession: SpawnCodeSession | null
): Promise<void> {
  if (codeSession === null) {
    await sandbox.kill().catch(() => undefined);
    return;
  }
  await releaseSandboxAfterFailedStart(sandbox, codeSession);
}

/**
 * Branch the sandbox ws-server checks out before the CLI starts (plan §1b).
 * CODE_AGENT_BASE_BRANCH empty = cut from the repository's default branch.
 */
export function codeSessionBranchEnvs(codeSession: CodeSessionRow | null): Record<string, string> {
  if (codeSession?.branch === null || codeSession?.branch === undefined) return {};
  return {
    CODE_AGENT_BRANCH: codeSession.branch,
    CODE_AGENT_BASE_BRANCH: codeSession.baseBranch ?? '',
    CODE_AGENT_MERGE_TARGET: codeSession.mergeTarget ?? '',
  };
}

/** The session a sandbox belongs to, so the orphan reaper can match it to its row. */
export function codeSessionSandboxMetadata(codeSession: CodeSessionRow): Record<string, string> {
  return {
    codeSessionId: codeSession.id,
    ownerAgentId: codeSession.ownerAgentId,
    rootAgentId: codeSession.rootAgentId,
    userId: codeSession.userId,
  };
}

/** Persistent code sessions pause (keeping fs + memory) on timeout. */
export function codeSessionCreateOptions(codeSession: CodeSessionRow | null): {
  lifecycle?: { onTimeout: 'pause' };
} {
  return codeSession !== null ? { lifecycle: { onTimeout: 'pause' } } : {};
}

/**
 * Local ws-server fallback (E2B billing block / create timeout) for a
 * persistent session: reuse its HOME, else restore the latest checkpoint.
 */
export async function createLocalCodeSessionFallback(
  err: unknown,
  codeSession: SpawnCodeSession | null,
  params: {
    taskId: string;
    agentId: string;
    goal: string;
    envs: Record<string, string>;
    attachments?: FilePointer[];
    userUiWorkspace?: UserUiWorkspace | null;
    projectWorkspace?: ProjectWorkspace | null;
    claudeTokenSource: string | null;
    claudeTokenIntegrationId: string | null;
    taskName?: string | null;
  }
): Promise<CodeAgentSessionResult | null> {
  if (!isLocalClaudeCodeFallbackEligible(err)) return null;
  const localHome = codeSession !== null ? await prepareLocalCodeSessionHome(codeSession) : null;
  const trajectory = codeSession?.handoffTrajectory ?? null;
  if (localHome !== null && trajectory !== null) {
    // C15 provider switch on this host: the log lands under the session HOME
    // and the handoff prompt names that path instead of the sandbox one.
    const written = await writeHandoffTrajectory(localSandboxFs(localHome.home), trajectory);
    params = { ...params, goal: params.goal.split(trajectory.path).join(written) };
  }
  const resumeId = localHome?.cliSessionId ?? null;
  // Same-host session: loopback tools URL, MCP path under the session HOME
  // (the persistent code-session HOME when there is one, else the task dir).
  const toolsHome = localHome?.home ?? resolveLocalSessionWorkDir(params.taskId);
  const local = await createLocalClaudeCodeSessionIfBillingBlocked(err, {
    ...params,
    userUiWorkspace: agentToolsForLocalHome(params.userUiWorkspace, toolsHome),
    envs:
      resumeId !== null ? { ...params.envs, CLAUDE_CODE_RESUME_SESSION_ID: resumeId } : params.envs,
    ...(localHome !== null ? { homeDir: localHome.home } : {}),
  });
  if (local === null) return null;
  if (codeSession !== null && localHome !== null) {
    await recordSandbox(codeSession.id, {
      sandboxKind: 'local',
      sandboxId: local.sandboxId,
      homeDir: localHome.home,
    });
    await linkClaudeCodeSession(local.taskId, codeSession.id);
  }
  return { ...local, resumedFrom: localHome?.resumedFrom ?? 'new', cliSessionId: resumeId };
}
