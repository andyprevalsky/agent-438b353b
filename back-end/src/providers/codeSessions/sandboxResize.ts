/**
 * C16 resize flows (see sandboxSize.ts for the diagram and the size model):
 *
 *   requestSandboxResize ── guards (reason, cap, in flight, per-dimension cooldown / reverse)
 *     └─ CodeSessionResize PENDING ──► pendingResizeForTask (keepalive)
 *   performSandboxResize ── claim (idempotent) → stop CLI → close task → create on the
 *     new size (acquire checkpoints, restores, kills the held sandbox) → DONE | FAILED
 *
 * 10k agents: per call one task row, one session row and RESIZE_HISTORY rows of
 * that session's own history (indexed by codeSessionId).
 */
import { Sandbox } from 'e2b';
import { prisma } from '../database';
import { createLogger } from '../logger';
import { isLocalClaudeCodeSandboxId } from '../claudeCodeLocalSession';
import { isLocalDeviceSandboxId } from '../localDevices/hub';
import { unregisterWorkspaceExecutor } from '../projects/workspaceExecutor';
import {
  availableSizes,
  closestAvailableSize,
  DIMENSION_LABEL,
  fromSize,
  heldSandboxSize,
  isStaleRunning,
  maxSizeFor,
  RESIZE_HISTORY,
  RESIZE_REASON_MAX,
  RESIZE_STALE_MS,
  resizeCooldownMs,
  resizeReverseWindowMs,
  sameSize,
  SIZE_DIMENSIONS,
  sizeKey,
  sizeLabel,
  toSize,
  type ResizeRow,
  type SandboxProvider,
  type SandboxSize,
  type SizeDimension,
} from './sandboxSize';
import {
  newResizeTransaction,
  resizeDurations,
  resizeFailureCode,
  resizeTelemetry,
  runInResizeTransaction,
  timedPhase,
  type ResizeTransaction,
} from './resizeTransaction';
import { recordSandbox, type CodeSessionRow } from './store';

const log = createLogger('CodeSandboxResize');

// ── requests ──

export class SandboxResizeError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details: Record<string, unknown> = {}
  ) {
    super(message);
    this.name = 'SandboxResizeError';
  }
}

/** HTTP status + body for a resize error, or null when it is not one. */
export function sandboxResizeHttpError(
  error: unknown
): { status: number; body: Record<string, unknown> } | null {
  if (!(error instanceof SandboxResizeError)) return null;
  return {
    status: error.status,
    body: { error: error.message, code: error.code, ...error.details },
  };
}

export interface SizeView {
  cpu: number;
  memory_gb: number;
}

export interface ResizeRequestView {
  request_id: string;
  status: string;
  from: SizeView;
  to: SizeView;
  reason: string;
  requested_at: string;
  completed_at: string | null;
  new_task_id: string | null;
  error: string | null;
  /** requestedAt → startedAt; null until claimed. */
  queue_ms: number | null;
  /** startedAt → completedAt; null until settled. */
  apply_ms: number | null;
  /** requestedAt → completedAt: request to ready (done) or to failure. */
  request_to_settled_ms: number | null;
}

export function sizeView(size: SandboxSize): SizeView {
  return { cpu: size.cpu, memory_gb: size.memoryGb };
}

export function resizeView(row: ResizeRow): ResizeRequestView {
  return {
    request_id: row.id,
    status: row.status.toLowerCase(),
    from: sizeView(fromSize(row)),
    to: sizeView(toSize(row)),
    reason: row.reason,
    requested_at: row.requestedAt.toISOString(),
    completed_at: row.completedAt?.toISOString() ?? null,
    new_task_id: row.newTaskId,
    error: row.error,
    ...resizeDurations(row),
  };
}

function providerOf(codeProvider: string | null | undefined): SandboxProvider {
  return codeProvider === 'codex' ? 'codex' : 'claude';
}

/** Task + session the request comes from, or the reason resizing does not apply. */
async function resizableSession(
  userId: string,
  agentId: string,
  taskId: string | null
): Promise<{ task: { id: string }; session: CodeSessionRow }> {
  if (taskId === null) {
    throw new SandboxResizeError(
      403,
      'NOT_A_CODE_SESSION',
      'Only a code-agent session can change its environment size'
    );
  }
  const task = await prisma.claudeCodeSession.findUnique({
    where: { id: taskId },
    select: { id: true, agentId: true, sandboxId: true, status: true, codeSessionId: true },
  });
  if (task === null || task.agentId !== agentId) {
    throw new SandboxResizeError(404, 'SESSION_NOT_FOUND', 'Code session not found');
  }
  if (isLocalDeviceSandboxId(task.sandboxId) || isLocalClaudeCodeSandboxId(task.sandboxId)) {
    throw new SandboxResizeError(
      409,
      'NOT_APPLICABLE',
      'Not applicable: this session runs on a local machine, not in a sized cloud sandbox. ' +
        'Its CPU and memory are that machine’s own.'
    );
  }
  if (task.status === 'CLOSED') {
    throw new SandboxResizeError(409, 'SESSION_CLOSED', 'This code session has ended');
  }
  const session =
    task.codeSessionId !== null
      ? await prisma.codeSession.findUnique({ where: { id: task.codeSessionId } })
      : null;
  if (session === null || session.userId !== userId || session.state === 'RETIRED') {
    throw new SandboxResizeError(
      409,
      'NOT_APPLICABLE',
      'Not applicable: this sandbox is not a persistent code session, so it cannot move to another size'
    );
  }
  return { task, session };
}

async function resizeHistory(codeSessionId: string): Promise<ResizeRow[]> {
  return prisma.codeSessionResize.findMany({
    where: { codeSessionId },
    orderBy: { requestedAt: 'desc' },
    take: RESIZE_HISTORY,
  });
}

function changes(row: ResizeRow, dim: SizeDimension): boolean {
  return fromSize(row)[dim] !== toSize(row)[dim];
}

/**
 * Guardrails for a new request, per dimension it changes: the same pending
 * request is returned (idempotent retry); another in flight, a change of the
 * same dimension inside the cooldown, or the reverse of that dimension's last
 * change inside its window is refused (409).
 */
function checkHistory(
  history: ResizeRow[],
  held: SandboxSize,
  target: SandboxSize,
  now: Date
): ResizeRow | null {
  const inFlight = history.find(
    (row) => (row.status === 'PENDING' || row.status === 'RUNNING') && !isStaleRunning(row, now)
  );
  if (inFlight !== undefined) {
    if (sameSize(toSize(inFlight), target)) return inFlight;
    throw new SandboxResizeError(
      409,
      'RESIZE_IN_PROGRESS',
      `A resize to ${sizeLabel(toSize(inFlight))} is already in progress`,
      { request: resizeView(inFlight) }
    );
  }
  const at = (row: ResizeRow) => (row.completedAt ?? row.requestedAt).getTime();
  for (const dim of SIZE_DIMENSIONS.filter((d) => target[d] !== held[d])) {
    const label = DIMENSION_LABEL[dim];
    const last = history.find((row) => row.status === 'DONE' && changes(row, dim));
    if (
      last !== undefined &&
      toSize(last)[dim] === held[dim] &&
      fromSize(last)[dim] === target[dim] &&
      now.getTime() - at(last) < resizeReverseWindowMs()
    ) {
      const retryAfterS = Math.ceil((at(last) + resizeReverseWindowMs() - now.getTime()) / 1000);
      const ago = Math.round((now.getTime() - at(last)) / 60_000);
      throw new SandboxResizeError(
        409,
        'REVERSE_RESIZE',
        `${label} moved from ${target[dim]} to ${held[dim]} ${ago} min ago; moving it straight ` +
          `back is refused for ${Math.ceil(retryAfterS / 60)} more min. ` +
          'If a process is running away, kill or fix it instead.',
        { dimension: dim, retry_after_s: retryAfterS }
      );
    }
    const recent = history.find(
      (row) => (row.status === 'DONE' || row.status === 'FAILED') && changes(row, dim)
    );
    if (recent !== undefined && now.getTime() - at(recent) < resizeCooldownMs()) {
      const retryAfterS = Math.ceil((at(recent) + resizeCooldownMs() - now.getTime()) / 1000);
      throw new SandboxResizeError(
        409,
        'RATE_LIMITED',
        `${label} can change once per ${Math.round(resizeCooldownMs() / 60_000)} min; ` +
          `try again in ${Math.ceil(retryAfterS / 60)} min`,
        { dimension: dim, retry_after_s: retryAfterS }
      );
    }
  }
  return null;
}

export interface ResizeRequestResult {
  httpStatus: 200 | 202;
  body: ResizeRequestView & { message: string };
}

export async function requestSandboxResize(params: {
  userId: string;
  agentId: string;
  taskId: string | null;
  cpu?: number;
  memoryGb?: number;
  reason: string;
  now?: Date;
}): Promise<ResizeRequestResult> {
  const now = params.now ?? new Date();
  const reason = params.reason.trim();
  if (reason === '') {
    throw new SandboxResizeError(400, 'REASON_REQUIRED', 'A reason is required');
  }
  if (params.cpu === undefined && params.memoryGb === undefined) {
    throw new SandboxResizeError(400, 'SIZE_REQUIRED', 'Give --cpu, --memory-gb or both');
  }
  const { task, session } = await resizableSession(params.userId, params.agentId, params.taskId);
  const provider = providerOf(session.codeProvider);
  const held = heldSandboxSize(session);
  const want = { cpu: params.cpu ?? held.cpu, memoryGb: params.memoryGb ?? held.memoryGb };
  const target = closestAvailableSize(provider, want);
  if (target === null) {
    throw new SandboxResizeError(
      409,
      'SIZE_UNAVAILABLE',
      `No ${provider} sandbox with at least ${sizeLabel(want)} is available`,
      { available: availableSizes(provider).map(sizeView) }
    );
  }
  if (sameSize(target, held)) {
    return {
      httpStatus: 200,
      body: {
        request_id: '',
        status: 'unchanged',
        from: sizeView(held),
        to: sizeView(held),
        reason,
        requested_at: now.toISOString(),
        completed_at: null,
        new_task_id: null,
        error: null,
        queue_ms: null,
        apply_ms: null,
        request_to_settled_ms: null,
        message: `This session already runs on ${sizeLabel(held)}, the closest size to that request.`,
      },
    };
  }
  const cap = await maxSizeFor(params.userId);
  const over = SIZE_DIMENSIONS.filter((dim) => target[dim] > held[dim] && target[dim] > cap[dim]);
  if (over.length > 0) {
    throw new SandboxResizeError(
      403,
      'SIZE_ABOVE_CAP',
      `${sizeLabel(target)} is above this account's limit (${sizeLabel(cap)})`,
      { max: sizeView(cap), dimensions: over }
    );
  }
  const existing = checkHistory(await resizeHistory(session.id), held, target, now);
  const row =
    existing ??
    (await prisma.codeSessionResize.create({
      data: {
        codeSessionId: session.id,
        userId: params.userId,
        agentId: params.agentId,
        taskId: task.id,
        fromCpu: held.cpu,
        fromMemoryGb: held.memoryGb,
        toCpu: target.cpu,
        toMemoryGb: target.memoryGb,
        reason: reason.slice(0, RESIZE_REASON_MAX),
      },
    }));
  log.info('Sandbox resize requested', {
    codeSessionId: session.id,
    requestId: row.id,
    from: sizeKey(held),
    to: sizeKey(target),
  });
  return {
    httpStatus: 202,
    body: {
      ...resizeView(row),
      message:
        `Resize to ${sizeLabel(target)} accepted (request ${row.id}). Within about two minutes ` +
        'this session is saved and continues in the new sandbox with the same branch, files and ' +
        'conversation; you will get a message when it is ready. Do not start new heavy work until then.',
    },
  };
}

/** Keepalive payload: the session's pending request, for the attached activity to apply. */
export async function pendingResizeForTask(
  codeSessionId: string | null
): Promise<{ request_id: string; from: SizeView; to: SizeView } | null> {
  if (codeSessionId === null) return null;
  const row = await prisma.codeSessionResize.findFirst({
    where: { codeSessionId, status: 'PENDING' },
    orderBy: { requestedAt: 'desc' },
  });
  return row === null
    ? null
    : { request_id: row.id, from: sizeView(fromSize(row)), to: sizeView(toSize(row)) };
}

/** What `rose env status` shows about the session's size (the CLI adds live usage). */
export async function environmentStatus(params: {
  userId: string;
  agentId: string;
  taskId: string | null;
}): Promise<Record<string, unknown>> {
  let resolved;
  try {
    resolved = await resizableSession(params.userId, params.agentId, params.taskId);
  } catch (err) {
    if (err instanceof SandboxResizeError && err.code === 'NOT_APPLICABLE') {
      return { applicable: false, message: err.message };
    }
    throw err;
  }
  const { session } = resolved;
  const provider = providerOf(session.codeProvider);
  const history = await resizeHistory(session.id);
  return {
    applicable: true,
    size: sizeView(heldSandboxSize(session)),
    max: sizeView(await maxSizeFor(params.userId)),
    available: availableSizes(provider).map(sizeView),
    cooldown_min: Math.round(resizeCooldownMs() / 60_000),
    resizes: history.slice(0, 5).map(resizeView),
  };
}

// ── applying a request ──

/** The task that continues the session (a create response carries more fields). */
export interface ResizeSessionResult {
  task_id: string;
  sandbox_id: string;
  ws_url: string;
  code_session_id?: string | null;
}

export interface ResizeCreateInput {
  agentId: string;
  userId: string;
  goal: string;
  codeSessionId: string;
  leaseOwner: string;
  projectId: string | null;
}

/** The turn the resumed CLI conversation receives after a resize. */
export function resizedTurn(row: ResizeRow): string {
  return [
    `Your environment was resized from ${sizeLabel(fromSize(row))} to ${sizeLabel(toSize(row))}, ` +
      `as you requested ("${row.reason}").`,
    'This is a new sandbox restored from your session: same branch, commits, uncommitted files and conversation.',
    'Processes that were running (dev servers, watchers, builds) were stopped; start again the ones you need.',
    'Dependency and build directories (node_modules, virtualenvs, dist/build outputs, caches) are not carried over; reinstall or rebuild what you need.',
    'Continue the task where you left off. Downsize with `rose env size` when the heavy phase is over.',
  ].join('\n');
}

export function resizeFailedTurn(row: ResizeRow, error: string, current: SandboxSize): string {
  return [
    `The resize from ${sizeLabel(fromSize(row))} to ${sizeLabel(toSize(row))} failed (${error.slice(0, 300)}).`,
    `You are on ${sizeLabel(current)}, with your branch, files and conversation as they were.`,
    'Processes that were running were stopped; start again the ones you need, then continue the task.',
  ].join('\n');
}

async function existingResult(row: ResizeRow): Promise<ResizeSessionResult | null> {
  if (row.newTaskId === null) return null;
  const task = await prisma.claudeCodeSession.findUnique({
    where: { id: row.newTaskId },
    select: { id: true, sandboxId: true, streamUrl: true, codeSessionId: true },
  });
  if (task === null || task.streamUrl === null) return null;
  return {
    task_id: task.id,
    sandbox_id: task.sandboxId,
    ws_url: task.streamUrl,
    code_session_id: task.codeSessionId,
  };
}

export interface PerformResizeResult<T extends ResizeSessionResult = ResizeSessionResult> {
  resized: boolean;
  request: ResizeRequestView;
  /** The create response; an idempotent replay returns only the task identity. */
  session: T | ResizeSessionResult;
}

/**
 * Apply a request for the task's activity. Idempotent per request id: a
 * retry after completion returns the same continuing task; a concurrent one
 * gets 409 RESIZE_IN_PROGRESS and retries.
 */
export async function performSandboxResize<T extends ResizeSessionResult>(params: {
  taskId: string;
  requestId: string;
  create: (input: ResizeCreateInput) => Promise<T>;
  stopCli: (sandboxId: string) => Promise<void>;
  closeTask: (taskId: string) => Promise<void>;
  now?: Date;
}): Promise<PerformResizeResult<T>> {
  const now = params.now ?? new Date();
  const task = await prisma.claudeCodeSession.findUnique({ where: { id: params.taskId } });
  const row = await prisma.codeSessionResize.findUnique({ where: { id: params.requestId } });
  if (task === null || row === null || row.codeSessionId !== task.codeSessionId) {
    throw new SandboxResizeError(404, 'RESIZE_NOT_FOUND', 'Resize request not found');
  }
  const settled = async (current: ResizeRow): Promise<PerformResizeResult<T> | null> => {
    if (current.status !== 'DONE' && current.status !== 'FAILED') return null;
    const session = await existingResult(current);
    if (session === null) {
      throw new SandboxResizeError(409, 'RESIZE_FAILED', current.error ?? 'Resize failed', {
        request: resizeView(current),
      });
    }
    return { resized: current.status === 'DONE', request: resizeView(current), session };
  };
  const done = await settled(row);
  if (done !== null) return done;

  const claimed = await prisma.codeSessionResize.updateMany({
    where: {
      id: row.id,
      OR: [
        { status: 'PENDING' },
        { status: 'RUNNING', startedAt: { lt: new Date(now.getTime() - RESIZE_STALE_MS) } },
      ],
    },
    data: { status: 'RUNNING', startedAt: now },
  });
  if (claimed.count === 0) {
    const current = await prisma.codeSessionResize.findUniqueOrThrow({ where: { id: row.id } });
    const finished = await settled(current);
    if (finished !== null) return finished;
    throw new SandboxResizeError(409, 'RESIZE_IN_PROGRESS', 'This resize is already running', {
      request: resizeView(current),
    });
  }

  const session = await prisma.codeSession.findUniqueOrThrow({ where: { id: row.codeSessionId } });
  const input = {
    agentId: task.agentId,
    userId: row.userId,
    codeSessionId: session.id,
    // The spawn keeps its lease: keepalives and its close still match it.
    leaseOwner: session.leaseOwner ?? `resize:${row.id}`,
    projectId: task.projectId,
  };
  const tx = newResizeTransaction(row.id);
  if (task.status !== 'CLOSED') {
    // Stop the old CLI first so nothing changes after the checkpoint.
    await timedPhase(
      'stop',
      () =>
        params.stopCli(task.sandboxId).catch((error: unknown) => {
          log.warn('Stopping the CLI before a resize failed', {
            taskId: task.id,
            error: error instanceof Error ? error.message : String(error),
          });
        }),
      tx
    );
    await params.closeTask(task.id);
  }

  try {
    const created = await runInResizeTransaction(tx, () =>
      timedPhase('ready', () => params.create({ ...input, goal: resizedTurn(row) }))
    );
    // The new sandbox serves: only now does the original go.
    await timedPhase('commit', () => commitResizeCutover(session.id, task.sandboxId, tx), tx);
    const updated = await prisma.codeSessionResize.update({
      where: { id: row.id },
      data: { status: 'DONE', newTaskId: created.task_id, completedAt: new Date() },
    });
    log.info('Sandbox resized', {
      codeSessionId: session.id,
      requestId: row.id,
      from: sizeKey(fromSize(row)),
      to: sizeKey(toSize(row)),
      newTaskId: created.task_id,
      telemetry: resizeTelemetry(updated, tx, 'done', null),
    });
    return { resized: true, request: resizeView(updated), session: created };
  } catch (error) {
    const code = resizeFailureCode(error);
    const message = `${code}: ${error instanceof Error ? error.message : String(error)}`;
    log.error('Sandbox resize failed; continuing on the previous size', {
      codeSessionId: session.id,
      requestId: row.id,
      error: message.slice(0, 500),
    });
    const abandoned = await timedPhase(
      'rollback',
      () => rollbackResizeCutover(session.id, task.sandboxId, row, tx),
      tx
    );
    const failed = await prisma.codeSessionResize.update({
      where: { id: row.id },
      data: { status: 'FAILED', error: message.slice(0, 500), completedAt: new Date() },
    });
    log.info('Sandbox resize settled', {
      codeSessionId: session.id,
      requestId: row.id,
      telemetry: resizeTelemetry(failed, tx, abandoned !== null ? 'rolled_back' : 'failed', code),
    });
    const after = await prisma.codeSession.findUniqueOrThrow({ where: { id: session.id } });
    const created = await params.create({
      ...input,
      goal: resizeFailedTurn(row, message, heldSandboxSize(after)),
    });
    if (abandoned !== null && abandoned !== created.sandbox_id) {
      // The original serves again: the half-started target is no longer needed.
      await killSandboxById(abandoned, session.id);
    }
    const updated = await prisma.codeSessionResize.update({
      where: { id: row.id },
      data: { newTaskId: created.task_id },
    });
    return { resized: false, request: resizeView(updated), session: created };
  }
}

async function killSandboxById(sandboxId: string, codeSessionId: string): Promise<void> {
  try {
    await Sandbox.kill(sandboxId, { apiKey: process.env.E2B_API_KEY });
  } catch (error) {
    log.warn('Killing a sandbox after a resize failed', {
      codeSessionId,
      sandboxId,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

/**
 * Commit: the session row names the new sandbox and it serves, so the original
 * (the requesting task's sandbox) is killed. Durable across a backend restart:
 * a re-run of a stale claim finds the original from the task row. Nothing is
 * killed while the row still names the original.
 */
async function commitResizeCutover(
  codeSessionId: string,
  originalSandboxId: string,
  tx: ResizeTransaction
): Promise<void> {
  const current = await prisma.codeSession.findUniqueOrThrow({ where: { id: codeSessionId } });
  if (current.sandboxId === null || current.sandboxId === originalSandboxId) return;
  if (tx.retained !== null && tx.retained.sandboxId === originalSandboxId) {
    await tx.retained.kill().catch((error: unknown) => {
      log.warn('Killing the original sandbox after a resize failed', {
        codeSessionId,
        sandboxId: originalSandboxId,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  } else {
    await killSandboxById(originalSandboxId, codeSessionId);
  }
  log.info('Resize committed; original sandbox released', {
    codeSessionId,
    previous: originalSandboxId,
    sandboxId: current.sandboxId,
  });
}

/**
 * Rollback after a failed create: when the row was cut over to a target that
 * never served while the original is retained (paused), the row names the
 * original and its size again. Returns the abandoned target, which the caller
 * kills only once the fallback spawn runs on the original.
 */
async function rollbackResizeCutover(
  codeSessionId: string,
  originalSandboxId: string,
  row: ResizeRow,
  tx: ResizeTransaction
): Promise<string | null> {
  if (tx.retained === null || tx.retained.sandboxId !== originalSandboxId) return null;
  const current = await prisma.codeSession.findUniqueOrThrow({ where: { id: codeSessionId } });
  if (current.sandboxId === originalSandboxId || current.sandboxId === null) return null;
  await recordSandbox(codeSessionId, {
    sandboxKind: 'e2b',
    sandboxId: originalSandboxId,
    size: fromSize(row),
  });
  log.warn('Resize rolled back to the original sandbox', {
    codeSessionId,
    sandboxId: originalSandboxId,
    abandoned: current.sandboxId,
  });
  return current.sandboxId;
}

/** Stop the ws-server and its CLI in an E2B sandbox, leaving the sandbox (and its files) up. */
export async function stopSandboxCli(sandboxId: string): Promise<void> {
  const sandbox = await Sandbox.connect(sandboxId, { apiKey: process.env.E2B_API_KEY });
  await sandbox.commands.run(
    // `[n]ode`: a plain pattern also matches this command's own shell, which the loop then killed.
    'for p in $(pgrep -f \'[n]ode /home/user/ws-server.mjs\'); do pkill -TERM -P "$p"; kill -TERM "$p"; done; sleep 1; true',
    { timeoutMs: 15_000 }
  );
}

/** The resized session continues as a new task; the old task row is closed (no output extraction). */
export async function closeTaskForResize(taskId: string): Promise<void> {
  unregisterWorkspaceExecutor(taskId);
  await prisma.claudeCodeSession.updateMany({
    where: { id: taskId, status: { not: 'CLOSED' } },
    data: { status: 'CLOSED', closedAt: new Date() },
  });
}
