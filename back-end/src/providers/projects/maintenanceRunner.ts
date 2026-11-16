/**
 * Execute ONE claimed template sync run (the Rose equivalent of a boxes.dev
 * Template box's maintenance script):
 *
 *   claimed run ─► sandbox (latest snapshot, else base template + fresh clone)
 *      ─► restoreWorkspace (env files, checkout)     reuses projects/workspace.ts
 *      ─► sync script under its timeout              (mode "snapshot": fast-forward only)
 *      ─► scrub secrets (env files, token remote, session dir)
 *      ─► sandbox.createSnapshot() ─► publishSnapshot (atomic, only while the lease is ours)
 *      ─► kill sandbox, release lease, drop the superseded snapshot
 *
 * Any failure before publication leaves the previous snapshot untouched: a
 * failed run publishes nothing, and an unpublished snapshot is deleted.
 * The runner owns exactly one sandbox, kills it in `finally`, and gives it a
 * self-expiring E2B timeout, so a crashed backend leaks nothing past the lease.
 *
 * 10k-agent behaviour: one template sandbox per PROJECT per run — never per
 * session or agent; sessions only read the published snapshot id.
 */
import { createLogger } from '../logger';
import {
  extendLease,
  finishRun,
  publishSnapshot,
  recordSandbox,
  type MaintenanceMode,
  type MaintenanceProject,
  type MaintenanceRunRecord,
  type MaintenanceStatus,
  MAINTENANCE_LOG_LIMIT,
} from './maintenanceStore';
import type { ProjectWorkspace } from './types';
import {
  projectSessionPaths,
  redactor,
  restoreWorkspace,
  tokenOf,
  type WorkspaceExecutor,
} from './workspace';
import {
  e2bWorkspaceExecutor,
  registerWorkspaceExecutor,
  shq,
  unregisterWorkspaceExecutor,
  type E2bSandboxLike,
} from './workspaceExecutor';

const log = createLogger('ProjectMaintenance');

/** Create + restore + scrub + snapshot headroom on top of the script's own timeout. */
export const MAINTENANCE_SLACK_MS = 10 * 60_000;
const STEP_TIMEOUT_MS = 2 * 60_000;
const MIN_TEMPLATE_MEMORY_MB = 3500;
export const MAINTENANCE_SCRIPT_REL = '.rose/project/maintenance.sh';

export interface MaintenanceSandbox extends E2bSandboxLike {
  sandboxId: string;
  kill(): Promise<unknown>;
  createSnapshot(): Promise<{ snapshotId: string }>;
}

export interface BaseTemplate {
  templateId: string;
  size: { cpu: number; memoryGb: number };
  /** Set when the template is smaller than a dependency install needs. */
  warning?: string;
}

export interface MaintenanceDeps {
  /** Sandbox HOME (`/home/user` on E2B). */
  homeDir: string;
  now(): Date;
  baseTemplate(project: MaintenanceProject): BaseTemplate;
  /** Create from a template id or a snapshot id. */
  createSandbox(
    templateOrSnapshotId: string,
    opts: { timeoutMs: number; metadata: Record<string, string> }
  ): Promise<MaintenanceSandbox>;
  /** Reset this owned sandbox lifetime, in milliseconds from now. */
  setSandboxTimeout(sandboxId: string, timeoutMs: number): Promise<void>;
  deleteSnapshot(snapshotId: string): Promise<boolean>;
}

export interface MaintenanceRunInput {
  project: MaintenanceProject;
  /** Decrypted workspace with the token-bearing clone URL (prepareSessionProject). */
  workspace: ProjectWorkspace;
  run: MaintenanceRunRecord;
  mode: MaintenanceMode;
}

export interface MaintenanceOutcome {
  status: Exclude<MaintenanceStatus, 'RUNNING'>;
  snapshotId: string | null;
  error: string | null;
}

class RunFailure extends Error {
  constructor(
    message: string,
    readonly status: 'FAILED' | 'TIMED_OUT' = 'FAILED',
    readonly exitCode: number | null = null
  ) {
    super(message);
  }
}

/** Bounded, redacted run log. */
class RunLog {
  private text = '';
  constructor(private readonly redact: (text: string) => string) {}
  line(message: string): void {
    this.add(`[sync] ${message}\n`);
  }
  add(raw: string): void {
    this.text = (this.text + this.redact(raw)).slice(-MAINTENANCE_LOG_LIMIT * 2);
  }
  tail(): string {
    return this.text.slice(-MAINTENANCE_LOG_LIMIT);
  }
}

export function maintenanceLeaseMs(project: Pick<MaintenanceProject, 'maintenanceTimeoutS'>) {
  return project.maintenanceTimeoutS * 1000 + MAINTENANCE_SLACK_MS;
}

/** Run to a terminal state; never throws (the row and lease are always settled). */
export async function executeMaintenanceRun(
  input: MaintenanceRunInput,
  deps: MaintenanceDeps
): Promise<MaintenanceOutcome> {
  const { project, workspace, run, mode } = input;
  const redact = redactor([...workspace.redactions, ...tokenOf(workspace.cloneUrl)]);
  const out = new RunLog(redact);
  const sessionId = `maintenance-${run.id}`;
  const paths = projectSessionPaths(deps.homeDir);
  let sandbox: MaintenanceSandbox | null = null;
  let createdSnapshotId: string | null = null;
  let published = false;
  let headSha: string | null = null;
  let exitCode: number | null = null;
  let outcome: MaintenanceOutcome;
  try {
    const base = deps.baseTemplate(project);
    if (base.warning) out.line(`warning: ${base.warning}`);
    const started = await startSandbox(input, deps, base, out);
    sandbox = started.sandbox;
    await recordSandbox(run.id, sandbox.sandboxId);
    out.line(
      `sandbox ${sandbox.sandboxId} from ${started.fromSnapshot ? 'the latest snapshot' : 'the base template'}`
    );

    const ex = e2bWorkspaceExecutor(sandbox, deps.homeDir);
    registerWorkspaceExecutor(sessionId, ex);
    await noteMemory(ex, deps.homeDir, out);
    const restore = await restoreWorkspace(
      sessionId,
      { ...workspace, startedFromSnapshot: false },
      { restoreLocalWork: false }
    );
    if (restore.checkout.error) throw new RunFailure(restore.checkout.error);
    out.line(`workspace ${restore.checkout.source} @ ${restore.checkout.headSha ?? 'unknown'}`);

    // Creation and checkout must not consume the sync script's lease budget.
    if (!(await extendLease(project.id, run.id, maintenanceLeaseMs(project), deps.now()))) {
      throw new RunFailure('the project lease was lost before the sync script started');
    }
    await deps.setSandboxTimeout(sandbox.sandboxId, maintenanceLeaseMs(project));

    // The script's `git fetch origin` needs the token; the scrub below removes it.
    if (workspace.cloneUrl) {
      await step(ex, paths.repoDir, `git remote set-url origin ${shq(workspace.cloneUrl)}`, out);
    }
    const branch = workspace.project.defaultBranch;
    if (mode === 'sync') {
      exitCode = await runScript(input, ex, paths, branch, out);
    } else {
      await step(
        ex,
        paths.repoDir,
        `git fetch --quiet origin ${shq(branch)} && git checkout --quiet ${shq(branch)} && git merge --ff-only --quiet ${shq(`origin/${branch}`)}`,
        out
      );
    }
    headSha = await readHead(ex, paths.repoDir);

    await scrub(ex, paths, restore.filesWritten, project.repoUrl, redact, out);
    const snapshot = await sandbox.createSnapshot();
    createdSnapshotId = snapshot.snapshotId;
    out.line(`snapshot ${snapshot.snapshotId} created`);

    const size = started.fromSnapshot ? started.snapshotSize : base.size;
    const result = await publishSnapshot({
      projectId: project.id,
      runId: run.id,
      snapshotId: snapshot.snapshotId,
      sha: headSha,
      size,
      now: deps.now(),
    });
    if (!result.published) {
      throw new RunFailure('the project lease was lost before the snapshot could be published');
    }
    published = true;
    out.line(`snapshot ${snapshot.snapshotId} published`);
    if (result.dropSnapshotId !== null) await dropSnapshot(deps, result.dropSnapshotId, out);
    outcome = { status: 'SUCCEEDED', snapshotId: snapshot.snapshotId, error: null };
  } catch (error) {
    const failure =
      error instanceof RunFailure
        ? error
        : new RunFailure(error instanceof Error ? error.message : String(error));
    out.line(`FAILED: ${failure.message}`);
    exitCode = failure.exitCode ?? exitCode;
    outcome = { status: failure.status, snapshotId: null, error: redact(failure.message) };
  } finally {
    unregisterWorkspaceExecutor(sessionId);
    if (!published && createdSnapshotId !== null) {
      await dropSnapshot(deps, createdSnapshotId, out);
    }
    if (sandbox !== null) {
      await sandbox.kill().catch((error: unknown) =>
        log.warn('Killing the template sandbox failed; it expires on its own timeout', {
          projectId: project.id,
          sandboxId: sandbox?.sandboxId,
          error: error instanceof Error ? error.message : String(error),
        })
      );
    }
  }
  try {
    await finishRun({
      runId: run.id,
      projectId: project.id,
      status: outcome.status,
      now: deps.now(),
      startedAt: run.startedAt,
      exitCode,
      headSha,
      snapshotId: outcome.snapshotId,
      error: outcome.error,
      logTail: out.tail(),
    });
  } catch (error) {
    log.error('Could not settle the maintenance run; its lease expires', {
      projectId: project.id,
      runId: run.id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
  log.info('Project template sync finished', {
    projectId: project.id,
    runId: run.id,
    status: outcome.status,
    snapshotId: outcome.snapshotId,
  });
  return outcome;
}

async function startSandbox(
  input: MaintenanceRunInput,
  deps: MaintenanceDeps,
  base: BaseTemplate,
  out: RunLog
): Promise<{
  sandbox: MaintenanceSandbox;
  fromSnapshot: boolean;
  snapshotSize: { cpu: number; memoryGb: number } | null;
}> {
  const { project, run } = input;
  const opts = {
    timeoutMs: maintenanceLeaseMs(project),
    metadata: { roseProjectId: project.id, roseMaintenanceRunId: run.id },
  };
  const snapshotSize = snapshotSizeOf(project);
  // A snapshot of another size than the project now wants (its sandboxSize
  // changed, or the default was raised) is rebuilt from the sized base template
  // instead of carrying the old size forward.
  const outdated =
    snapshotSize !== null &&
    (snapshotSize.cpu !== base.size.cpu || snapshotSize.memoryGb !== base.size.memoryGb);
  if (run.fromSnapshotId !== null && outdated) {
    out.line(
      `the latest snapshot is ${snapshotSize.cpu} vCPU / ${snapshotSize.memoryGb} GB; rebuilding at ${base.size.cpu} vCPU / ${base.size.memoryGb} GB from the base template`
    );
  }
  if (run.fromSnapshotId !== null && !outdated) {
    try {
      const sandbox = await deps.createSandbox(run.fromSnapshotId, opts);
      return { sandbox, fromSnapshot: true, snapshotSize };
    } catch (error) {
      out.line(
        `the latest snapshot could not start (${error instanceof Error ? error.message : String(error)}); starting from the base template`
      );
    }
  }
  return {
    sandbox: await deps.createSandbox(base.templateId, opts),
    fromSnapshot: false,
    snapshotSize: null,
  };
}

function snapshotSizeOf(project: MaintenanceProject): { cpu: number; memoryGb: number } | null {
  const { cpu, memoryGb } = (project.templateSnapshotSize ?? {}) as {
    cpu?: unknown;
    memoryGb?: unknown;
  };
  return Number.isInteger(cpu) && Number.isInteger(memoryGb)
    ? { cpu: cpu as number, memoryGb: memoryGb as number }
    : null;
}

/** A too-small sandbox is the likeliest cause of an install being OOM-killed: say so in the log. */
async function noteMemory(ex: WorkspaceExecutor, cwd: string, out: RunLog): Promise<void> {
  const mem = await ex.run("awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo", {
    cwd,
    timeoutMs: STEP_TIMEOUT_MS,
  });
  const mb = Number(mem.stdout.trim());
  if (mem.exitCode !== 0 || !Number.isFinite(mb)) return;
  out.line(`sandbox memory ${mb} MB`);
  if (mb < MIN_TEMPLATE_MEMORY_MB) {
    out.line(
      'warning: this sandbox has less than 4 GB; a dependency install may be killed for lack of memory (build the 4 GB sandbox template)'
    );
  }
}

async function step(
  ex: WorkspaceExecutor,
  cwd: string,
  command: string,
  out: RunLog
): Promise<void> {
  const result = await ex.run(command, { cwd, timeoutMs: STEP_TIMEOUT_MS });
  if (result.exitCode !== 0) {
    out.add(result.stdout + result.stderr);
    throw new RunFailure(
      `step failed (exit ${result.exitCode}): ${command.split(' ')[0]} ${command.split(' ')[1] ?? ''}`.trim()
    );
  }
}

async function runScript(
  input: MaintenanceRunInput,
  ex: WorkspaceExecutor,
  paths: ReturnType<typeof projectSessionPaths>,
  branch: string,
  out: RunLog
): Promise<number | null> {
  const script = input.project.maintenanceScript?.trim() ? input.project.maintenanceScript : null;
  if (script === null) throw new RunFailure('the project has no sync script');
  const scriptPath = await ex.writeFileInside(
    paths.homeDir,
    MAINTENANCE_SCRIPT_REL,
    Buffer.from(script, 'utf8'),
    0o700
  );
  const result = await ex.run(`bash ${shq(scriptPath)}`, {
    cwd: paths.repoDir,
    timeoutMs: input.project.maintenanceTimeoutS * 1000,
    env: {
      ROSE_PROJECT_DIR: paths.repoDir,
      ROSE_DEFAULT_BRANCH: branch,
      ROSE_MAINTENANCE: '1',
      GIT_TERMINAL_PROMPT: '0',
    },
  });
  out.add(result.stdout);
  out.add(result.stderr);
  if (result.timedOut) {
    throw new RunFailure(
      `the sync script exceeded its ${input.project.maintenanceTimeoutS}s timeout`,
      'TIMED_OUT'
    );
  }
  if (result.exitCode !== 0) {
    throw new RunFailure(
      `the sync script exited with ${result.exitCode}`,
      'FAILED',
      result.exitCode
    );
  }
  return result.exitCode;
}

async function readHead(ex: WorkspaceExecutor, repoDir: string): Promise<string | null> {
  const head = await ex.run('git rev-parse HEAD', { cwd: repoDir, timeoutMs: STEP_TIMEOUT_MS });
  const sha = head.stdout.trim();
  return head.exitCode === 0 && /^[0-9a-f]{40}$/.test(sha) ? sha : null;
}

/**
 * Nothing secret may enter the snapshot every session of the project starts
 * from: env files and links, the session dir (startup.json holds the clone
 * token), bundles and patches, and the token in the checkout's origin URL. A
 * session restore rewrites them all. If a check fails the run fails.
 */
async function scrub(
  ex: WorkspaceExecutor,
  paths: ReturnType<typeof projectSessionPaths>,
  envPaths: string[],
  repoUrl: string | null,
  redact: (text: string) => string,
  out: RunLog
): Promise<void> {
  const remove = envPaths.length > 0 ? `rm -f -- ${envPaths.map(shq).join(' ')} && ` : '';
  const restoreRemote =
    repoUrl !== null
      ? `git remote set-url origin ${shq(repoUrl)} && test "$(git config --get remote.origin.url)" = ${shq(repoUrl)}`
      : 'true';
  const removeLocalWork = [
    "refs=$(git for-each-ref --format='%(refname)' refs/heads/rose-sync/) || exit 1",
    'for ref in $refs; do git update-ref -d "$ref" || exit 1; done',
    "stashes=$(git stash list --format='%gd %gs') || exit 1",
    // Drop oldest first so lower stash indices stay stable; keep unrelated stashes.
    `while IFS= read -r entry; do case "$entry" in *"rose-sync uncommitted ("*) git stash drop "${'$'}{entry%% *}" >/dev/null || exit 1 ;; esac; done < <(printf '%s\n' "$stashes" | tac)`,
  ].join('; ');
  const result = await ex.run(
    `${removeLocalWork} && ${remove}rm -rf ${shq(paths.projectDir)} && ${restoreRemote}`,
    {
      cwd: paths.repoDir,
      timeoutMs: STEP_TIMEOUT_MS,
    }
  );
  if (result.exitCode !== 0) {
    out.add(redact(result.stdout + result.stderr));
    throw new RunFailure('the sandbox could not be scrubbed of credentials; no snapshot taken');
  }
  out.line('credentials scrubbed');
}

async function dropSnapshot(deps: MaintenanceDeps, snapshotId: string, out: RunLog): Promise<void> {
  try {
    const removed = await deps.deleteSnapshot(snapshotId);
    out.line(
      removed ? `snapshot ${snapshotId} deleted` : `snapshot ${snapshotId} was already gone`
    );
  } catch (error) {
    out.line(
      `could not delete snapshot ${snapshotId}: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}
