/**
 * C16: code-agent sandbox sizes. E2B fixes CPU and memory per template build,
 * so every (vCPU, memory) point of a configurable grid is its own template and
 * a resize moves the persistent code session into a new sandbox (W5b
 * checkpoint → restore → `--resume`). CPU and memory adapt independently.
 *
 *   rose env size [--cpu N] [--memory-gb M] --reason …   (in-sandbox CLI / MCP)
 *     └─ POST /api/agent-tools/:agentId/environment/size
 *          requestSandboxResize: a missing flag keeps that dimension · closest
 *          built template satisfying both (cpu ≥ N, memory ≥ M) · per-dimension
 *          cap (403) · per-dimension cooldown and no immediate reverse of the
 *          same dimension (409) · reason required (400)
 *          └─ CodeSessionResize PENDING ──► 202
 *   attached code-agent activity ── keepalive ──► resize_requested
 *     └─ POST /api/{claude,codex}-code-session/:taskId/resize {request_id}
 *          performSandboxResize: claim (PENDING → RUNNING, idempotent)
 *            ├─ stop the old CLI, close its task (the lease stays with the spawn)
 *            ├─ create(goal = resize turn): acquire sees the wanted size ≠ held size
 *            │    → checkpoint the held sandbox → new sandbox from the size's
 *            │      template → restore (branch, commits, files, CLI log)
 *            │      → kill the held sandbox → `claude --resume` / `codex resume`
 *            ├─ DONE (newTaskId) ──► the activity reattaches to the new task
 *            └─ failure → FAILED, continue on the old size with a turn that says so
 *   a session with no attached activity applies a PENDING request on its next spawn.
 *
 * Size of a new session (newSessionSize): the spawn's explicit `sandbox_size`
 * (root-selected, validated by resolveRequestedSandboxSize), else its Project's
 * `sandboxSize` {cpu, memoryGb}, else the new-session default: the operator's
 * CODE_SANDBOX_DEFAULT_* when set, else PREFERRED_DEFAULT_SANDBOX_SIZE (8 vCPU,
 * 8 GB) when that template is built and within the user's cap, else the base
 * size. A session that already holds a sandbox keeps its size; only a resize
 * request moves it. The local Mac target has no size (not applicable).
 *
 * This file is the size model (grid, templates, caps, the size a spawn wants);
 * sandboxResize.ts holds the request guards and the apply flow.
 *
 * 10k agents: each call reads one task row, one session row and at most
 * RESIZE_HISTORY rows of that session's own resize history (indexed by
 * codeSessionId); keepalive adds one indexed read. No shared row or counter.
 */
import { z } from 'zod';
import { prisma } from '../database';
import { createLogger } from '../logger';
import { isRoseDevUser } from '../../utils/roseDevAccess';
import type { CodeSessionRow } from './store';
import { CODEX_CODE_TEMPLATE, DEFAULT_CODEX_CODE_TEMPLATE_ID } from '../codexTemplate';
import { CLAUDE_CODE_TEMPLATE, DEFAULT_CLAUDE_CODE_TEMPLATE_ID } from '../claudeCodeTemplate';

const log = createLogger('CodeSandboxSize');

export type SandboxProvider = 'claude' | 'codex';
export type SizeDimension = 'cpu' | 'memoryGb';
export const SIZE_DIMENSIONS: readonly SizeDimension[] = ['cpu', 'memoryGb'];

/** vCPUs and memory (GB) of one sandbox template. */
export interface SandboxSize {
  cpu: number;
  memoryGb: number;
}

/**
 * The template grid the sized build config produces (e2b-templates/sized-templates.json,
 * pinned equal by a structural test): 4 templates per provider. Every point is a
 * separately built E2B template, so the grid stays small.
 *
 * Memory stops at 8 GB: the E2B team refuses templates over 8192 MiB ("Memory
 * can't be higher than 8192 MiB", 2026-10-28). A 16 GB size needs E2B support
 * to raise that cap first, then a rebuild of the grid.
 */
export const DEFAULT_SANDBOX_GRID: { cpu: number[]; memoryGb: number[] } = {
  cpu: [2, 8],
  memoryGb: [4, 8],
};
/** A configured grid larger than this (per provider) is refused. */
export const MAX_GRID_POINTS = 9;

const DEFAULT_COOLDOWN_MIN = 10;
const DEFAULT_REVERSE_WINDOW_MIN = 30;
/** A RUNNING claim older than this was abandoned (backend restart mid-resize). */
export const RESIZE_STALE_MS = 15 * 60 * 1000;
export const RESIZE_HISTORY = 10;
export const RESIZE_REASON_MAX = 1000;
export const DIMENSION_LABEL: Record<SizeDimension, string> = { cpu: 'CPU', memoryGb: 'Memory' };

function positiveInts(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const out = value.filter((entry): entry is number => Number.isInteger(entry) && entry > 0);
  return out.length === value.length ? [...new Set(out)].sort((a, b) => a - b) : null;
}

/**
 * CODE_SANDBOX_GRID (`{"cpu":[2,8],"memoryGb":[4,8]}`), else the default grid.
 * A grid of more than MAX_GRID_POINTS templates is refused (default grid).
 */
export function sandboxGrid(): { cpu: number[]; memoryGb: number[] } {
  const raw = process.env.CODE_SANDBOX_GRID?.trim();
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as { cpu?: unknown; memoryGb?: unknown };
      const cpu = positiveInts(parsed.cpu);
      const memoryGb = positiveInts(parsed.memoryGb);
      if (cpu !== null && memoryGb !== null && cpu.length * memoryGb.length <= MAX_GRID_POINTS) {
        return { cpu, memoryGb };
      }
    } catch {
      // fall through to the warning
    }
    log.warn(
      `CODE_SANDBOX_GRID is invalid or over ${MAX_GRID_POINTS} sizes; using the default grid`
    );
  }
  return DEFAULT_SANDBOX_GRID;
}

export function gridSizes(): SandboxSize[] {
  const grid = sandboxGrid();
  return grid.cpu.flatMap((cpu) => grid.memoryGb.map((memoryGb) => ({ cpu, memoryGb })));
}

export function sizeKey(size: SandboxSize): string {
  return `${size.cpu}x${size.memoryGb}`;
}

export function sameSize(a: SandboxSize, b: SandboxSize): boolean {
  return a.cpu === b.cpu && a.memoryGb === b.memoryGb;
}

export function sizeLabel(size: SandboxSize): string {
  return `${size.cpu} vCPU, ${size.memoryGb} GB RAM`;
}

function onGrid(size: SandboxSize): boolean {
  return gridSizes().some((point) => sameSize(point, size));
}

function intEnv(name: string): number | null {
  const value = Number(process.env[name]);
  return Number.isInteger(value) && value > 0 ? value : null;
}

/**
 * The base size: CODE_SANDBOX_DEFAULT_CPU / _MEMORY_GB (on the grid), else the
 * grid's smallest point. It is the size of the unsized template and of
 * sessions from before C16; a new session uses newSessionDefaultSize.
 */
export function defaultSandboxSize(): SandboxSize {
  const grid = sandboxGrid();
  const smallest = { cpu: grid.cpu[0], memoryGb: grid.memoryGb[0] };
  const configured = {
    cpu: intEnv('CODE_SANDBOX_DEFAULT_CPU') ?? smallest.cpu,
    memoryGb: intEnv('CODE_SANDBOX_DEFAULT_MEMORY_GB') ?? smallest.memoryGb,
  };
  return onGrid(configured) ? configured : smallest;
}

/**
 * Default size of a new session when neither the spawn nor its project chose
 * one. 8 vCPU / 8 GB templates are built and ready for both providers
 * (E2B template list, 2026-11-16); 8 GB is E2B's per-template memory cap.
 */
export const PREFERRED_DEFAULT_SANDBOX_SIZE: SandboxSize = { cpu: 8, memoryGb: 8 };

function fitsWithin(size: SandboxSize, max: SandboxSize): boolean {
  return size.cpu <= max.cpu && size.memoryGb <= max.memoryGb;
}

/**
 * New-session default (no spawn or project choice): an operator-set
 * CODE_SANDBOX_DEFAULT_* wins; else PREFERRED_DEFAULT_SANDBOX_SIZE when its
 * template is built for `provider` and it fits `max` (the user's cap); else
 * the base size. Never a size without a built template.
 */
export function newSessionDefaultSize(provider: SandboxProvider, max?: SandboxSize): SandboxSize {
  const base = defaultSandboxSize();
  if (
    intEnv('CODE_SANDBOX_DEFAULT_CPU') !== null ||
    intEnv('CODE_SANDBOX_DEFAULT_MEMORY_GB') !== null
  ) {
    return base;
  }
  const preferred = PREFERRED_DEFAULT_SANDBOX_SIZE;
  const built = sandboxTemplateFor(provider, preferred) !== null;
  return built && (max === undefined || fitsWithin(preferred, max)) ? { ...preferred } : base;
}

/** Size of the sandbox a session holds (sessions from before C16 hold the base size). */
export function heldSandboxSize(
  session: Pick<CodeSessionRow, 'sandboxCpu' | 'sandboxMemoryGb'>
): SandboxSize {
  const fallback = defaultSandboxSize();
  return {
    cpu: session.sandboxCpu ?? fallback.cpu,
    memoryGb: session.sandboxMemoryGb ?? fallback.memoryGb,
  };
}

function isDimension(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0;
}

/** A stored `{cpu?, memoryGb?}` (project default); a missing dimension takes `fallback`. */
export function parseSandboxSize(
  value: unknown,
  fallback: SandboxSize = defaultSandboxSize()
): SandboxSize | null {
  if (typeof value !== 'object' || value === null) return null;
  const { cpu, memoryGb } = value as { cpu?: unknown; memoryGb?: unknown };
  if (!isDimension(cpu) && !isDimension(memoryGb)) return null;
  return {
    cpu: isDimension(cpu) ? cpu : fallback.cpu,
    memoryGb: isDimension(memoryGb) ? memoryGb : fallback.memoryGb,
  };
}

function minutesEnv(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return process.env[name] !== undefined && Number.isFinite(value) && value >= 0 ? value : fallback;
}

export function resizeCooldownMs(): number {
  return minutesEnv('CODE_SANDBOX_RESIZE_COOLDOWN_MIN', DEFAULT_COOLDOWN_MIN) * 60_000;
}

export function resizeReverseWindowMs(): number {
  return minutesEnv('CODE_SANDBOX_RESIZE_REVERSE_WINDOW_MIN', DEFAULT_REVERSE_WINDOW_MIN) * 60_000;
}

/** Template name the sized build config gives a grid point. */
export function sizedTemplateName(provider: SandboxProvider, size: SandboxSize): string {
  const prefix =
    provider === 'codex'
      ? CODEX_CODE_TEMPLATE.sizedAliasPrefix
      : CLAUDE_CODE_TEMPLATE.sizedAliasPrefix;
  return `${prefix}-c${size.cpu}-m${size.memoryGb}`;
}

type TemplateMap = Partial<Record<SandboxProvider, Record<string, string>>>;

/**
 * E2B_SANDBOX_TEMPLATES: `grid` (every grid point is built under its
 * sizedTemplateName) or JSON `{"claude":{"4x8":"claude-code-c4-m8"},…}`.
 */
function configuredTemplates(): TemplateMap | 'grid' {
  const raw = process.env.E2B_SANDBOX_TEMPLATES?.trim();
  if (!raw) return {};
  if (raw === 'grid') return 'grid';
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed === 'object' && parsed !== null) return parsed as TemplateMap;
  } catch {
    // fall through to the warning
  }
  log.warn('E2B_SANDBOX_TEMPLATES is neither "grid" nor JSON; sized templates disabled');
  return {};
}

function currentTemplate(provider: SandboxProvider, size: SandboxSize): string | null {
  if (provider === 'claude') {
    return sameSize(size, CLAUDE_CODE_TEMPLATE.size) ? DEFAULT_CLAUDE_CODE_TEMPLATE_ID : null;
  }
  return process.env.E2B_CODEX_CODE_TEMPLATE_ID?.trim() || DEFAULT_CODEX_CODE_TEMPLATE_ID;
}

/**
 * Template for (provider, size). Without a configured sized template only the
 * default size is available (the current template); another size is never
 * silently mapped to a template of a different size.
 */
export function sandboxTemplateFor(provider: SandboxProvider, size: SandboxSize): string | null {
  if (!onGrid(size)) return null;
  const configured = configuredTemplates();
  if (configured === 'grid') return sizedTemplateName(provider, size);
  const named = configured[provider]?.[sizeKey(size)];
  if (typeof named === 'string' && named.trim() !== '') return named.trim();
  return sameSize(size, defaultSandboxSize()) ? currentTemplate(provider, size) : null;
}

export function availableSizes(provider: SandboxProvider): SandboxSize[] {
  return gridSizes().filter((size) => sandboxTemplateFor(provider, size) !== null);
}

/**
 * The built template closest to `want` that satisfies it (cpu ≥ want.cpu and
 * memory ≥ want.memoryGb): least relative excess, then fewer vCPUs.
 */
export function closestAvailableSize(
  provider: SandboxProvider,
  want: SandboxSize
): SandboxSize | null {
  const excess = (size: SandboxSize) =>
    (size.cpu - want.cpu) / want.cpu + (size.memoryGb - want.memoryGb) / want.memoryGb;
  const fits = availableSizes(provider).filter(
    (size) => size.cpu >= want.cpu && size.memoryGb >= want.memoryGb
  );
  fits.sort((a, b) => excess(a) - excess(b) || a.cpu - b.cpu || a.memoryGb - b.memoryGb);
  return fits[0] ?? null;
}

/**
 * Largest vCPUs / memory this user's code agents may resize to: the user's
 * own cap (User.codeSandboxMax*), else CODE_SANDBOX_MAX_CPU / _MEMORY_GB, else
 * the grid's largest size. Rose dev accounts ignore the deployment cap.
 */
export async function maxSizeFor(userId: string): Promise<SandboxSize> {
  const grid = sandboxGrid();
  const top = {
    cpu: grid.cpu[grid.cpu.length - 1],
    memoryGb: grid.memoryGb[grid.memoryGb.length - 1],
  };
  const user = await prisma.user.findUnique({
    where: { userId },
    select: { codeSandboxMaxCpu: true, codeSandboxMaxMemoryGb: true },
  });
  const fallback = isRoseDevUser(userId)
    ? top
    : {
        cpu: intEnv('CODE_SANDBOX_MAX_CPU') ?? top.cpu,
        memoryGb: intEnv('CODE_SANDBOX_MAX_MEMORY_GB') ?? top.memoryGb,
      };
  return {
    cpu: user?.codeSandboxMaxCpu ?? fallback.cpu,
    memoryGb: user?.codeSandboxMaxMemoryGb ?? fallback.memoryGb,
  };
}

export type ResizeRow = Awaited<ReturnType<typeof prisma.codeSessionResize.findFirst>> & object;

export function fromSize(row: ResizeRow): SandboxSize {
  return { cpu: row.fromCpu, memoryGb: row.fromMemoryGb };
}

export function toSize(row: ResizeRow): SandboxSize {
  return { cpu: row.toCpu, memoryGb: row.toMemoryGb };
}

export function isStaleRunning(row: ResizeRow, now: Date): boolean {
  return (
    row.status === 'RUNNING' &&
    row.startedAt !== null &&
    now.getTime() - row.startedAt.getTime() > RESIZE_STALE_MS
  );
}

/** Newest PENDING or live RUNNING request of a session (abandoned claims are failed). */
export async function activeResizeFor(
  codeSessionId: string,
  now = new Date()
): Promise<ResizeRow | null> {
  const row = await prisma.codeSessionResize.findFirst({
    where: { codeSessionId, status: { in: ['PENDING', 'RUNNING'] } },
    orderBy: { requestedAt: 'desc' },
  });
  if (row === null) return null;
  if (isStaleRunning(row, now)) {
    await prisma.codeSessionResize.updateMany({
      where: { id: row.id, status: 'RUNNING' },
      data: { status: 'FAILED', error: 'abandoned while running', completedAt: now },
    });
    return null;
  }
  return row;
}

/**
 * Where a spawn's size came from: `request` (the spawn's explicit sandbox_size),
 * `project` (its Project's sandboxSize), `default` (the new-session default),
 * `resize` (an active resize request), `session` (the size the session holds),
 * `legacy` (a session from before C16, on the base size).
 */
export type SandboxSizeSource = 'request' | 'project' | 'default' | 'resize' | 'session' | 'legacy';

/** A spawn's sizing inputs: its explicit request and its project's stored size. */
export interface SpawnSizeChoice {
  requested?: SandboxSize | null;
  projectSize?: unknown;
  /** The user's cap; the new-session default never exceeds it. */
  max?: SandboxSize;
}

/** Size of a brand-new session: the request, else the project's size, else the default. */
export function newSessionSize(
  provider: SandboxProvider,
  choice: SpawnSizeChoice
): { size: SandboxSize; source: SandboxSizeSource } {
  const fallback = newSessionDefaultSize(provider, choice.max);
  if (choice.requested) return { size: choice.requested, source: 'request' };
  const project = parseSandboxSize(choice.projectSize, fallback);
  return project !== null
    ? { size: project, source: 'project' }
    : { size: fallback, source: 'default' };
}

/**
 * Size a spawn of this session should run on: an active resize request's
 * target, else the size the session already holds; a brand-new session takes
 * the spawn's request, its project's size or the new-session default.
 */
export async function wantedSessionSize(
  session: CodeSessionRow | null,
  provider: SandboxProvider,
  choice: SpawnSizeChoice
): Promise<{ size: SandboxSize; source: SandboxSizeSource }> {
  if (session === null) return newSessionSize(provider, choice);
  const active = await activeResizeFor(session.id);
  if (active !== null) return { size: toSize(active), source: 'resize' };
  if (session.sandboxCpu !== null || session.sandboxMemoryGb !== null) {
    return { size: heldSandboxSize(session), source: 'session' };
  }
  const fresh = session.sandboxId === null && session.latestCheckpointSeq === 0;
  return fresh
    ? newSessionSize(provider, choice)
    : { size: defaultSandboxSize(), source: 'legacy' };
}

/**
 * Size + template for a spawn: the closest built template satisfying the
 * wanted size, else the base size. A size preference never blocks a session
 * from starting (an explicit request was validated before: resolveRequestedSandboxSize).
 */
export async function resolveSpawnSandbox(
  session: CodeSessionRow | null,
  provider: SandboxProvider,
  choice: SpawnSizeChoice
): Promise<{
  size: SandboxSize;
  templateId: string | null;
  wanted: SandboxSize;
  source: SandboxSizeSource;
}> {
  const { size: wanted, source } = await wantedSessionSize(session, provider, choice);
  const size = closestAvailableSize(provider, wanted) ?? defaultSandboxSize();
  if (!sameSize(size, wanted)) {
    log.warn('No template built for the wanted sandbox size; using the closest built one', {
      provider,
      wanted: sizeKey(wanted),
      size: sizeKey(size),
      codeSessionId: session?.id,
    });
  }
  return { size, templateId: sandboxTemplateFor(provider, size), wanted, source };
}

/**
 * A spawn's explicit `sandbox_size` ({cpu?, memory_gb?}, at least one): the
 * shape check only; resolveRequestedSandboxSize checks it against the built
 * templates and the user's cap. Disk is not selectable: E2B fixes it per team.
 */
export const sandboxSizeRequestSchema = z
  .object({
    cpu: z.number().int().min(1).max(64).optional(),
    memory_gb: z.number().int().min(1).max(256).optional(),
  })
  .strict()
  .refine((size) => size.cpu !== undefined || size.memory_gb !== undefined, {
    message: 'give cpu, memory_gb or both',
  });

export type SandboxSizeRequest = z.infer<typeof sandboxSizeRequestSchema>;

/** A local (Mac) session has no sandbox, so a size request is refused, never ignored. */
export function refuseLocalSandboxSize(
  body: { target?: 'cloud' | 'local'; sandbox_size?: SandboxSizeRequest },
  ctx: z.RefinementCtx
): void {
  if (body.target === 'local' && body.sandbox_size !== undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['sandbox_size'],
      message: "sandbox_size applies to target='cloud' only; a local session runs on the Mac",
    });
  }
}

/** Validated size of a create request, or null without one. */
export async function requestedSandboxSize(
  provider: SandboxProvider,
  request: SandboxSizeRequest | undefined,
  userId: string | undefined
): Promise<SandboxSize | null> {
  if (request === undefined) return null;
  return resolveRequestedSandboxSize(
    provider,
    { cpu: request.cpu, memoryGb: request.memory_gb },
    userId
  );
}

/** A spawn's explicit sandbox_size that no built template within the user's cap satisfies. */
export class SandboxSizeUnsupportedError extends Error {
  readonly code = 'SANDBOX_SIZE_UNSUPPORTED';
  constructor(
    message: string,
    readonly requested: SandboxSize,
    readonly available: SandboxSize[],
    readonly max: SandboxSize
  ) {
    super(message);
    this.name = 'SandboxSizeUnsupportedError';
  }
}

/**
 * Validate a spawn's explicit `{cpu?, memoryGb?}`: a missing dimension takes the
 * new-session default; the request must fit the user's cap and some built
 * template must satisfy it (cpu ≥ N, memory ≥ M). Returns the request with both
 * dimensions; the spawn runs on closestAvailableSize of it.
 */
export async function resolveRequestedSandboxSize(
  provider: SandboxProvider,
  requested: { cpu?: number; memoryGb?: number },
  userId: string | undefined
): Promise<SandboxSize> {
  const max = userId !== undefined ? await maxSizeFor(userId) : undefined;
  const fallback = newSessionDefaultSize(provider, max);
  const want = {
    cpu: requested.cpu ?? fallback.cpu,
    memoryGb: requested.memoryGb ?? fallback.memoryGb,
  };
  const available = availableSizes(provider).filter(
    (size) => max === undefined || fitsWithin(size, max)
  );
  const choices = available.map(sizeLabel).join('; ') || 'none';
  if (max !== undefined && !fitsWithin(want, max)) {
    throw new SandboxSizeUnsupportedError(
      `Requested sandbox ${sizeLabel(want)} is over this account's limit of ${sizeLabel(max)}. Available: ${choices}.`,
      want,
      available,
      max
    );
  }
  if (closestAvailableSize(provider, want) === null) {
    throw new SandboxSizeUnsupportedError(
      `No ${provider} sandbox of at least ${sizeLabel(want)} is built. Available: ${choices}.`,
      want,
      available,
      max ?? want
    );
  }
  return want;
}
