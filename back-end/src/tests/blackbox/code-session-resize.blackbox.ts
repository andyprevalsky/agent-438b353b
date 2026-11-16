#!/usr/bin/env tsx
/**
 * Black-box contract for adaptive code-agent sandbox sizes (rose-builds-rose C16).
 *
 *   rose env size ──POST /api/agent-tools/:agentId/environment/size──► CodeSessionResize PENDING (202)
 *   keepalive ──► resize_requested ──► performSandboxResize
 *     stop CLI → checkpoint held fake sandbox → Sandbox.create(<size's template>)
 *     → restore (branch, commits, uncommitted + untracked files, CLI log) → kill held
 *     → `--resume` turn naming the new size; retry returns the same task
 *   guards: reason (400), cap per dimension (403), in flight / cooldown / reverse per
 *   dimension (409), unchanged (200), local target not applicable (409)
 *   rose env status: size + top 5 processes by memory and CPU
 *   project sandboxSize: default size of a new session; the skill file lands in HOME
 *
 * Storage is hermetic (CODE_SESSION_STORAGE_BACKEND=local); E2B is a fake over
 * local HOME directories; nothing reaches E2B, GitHub or Supabase Storage.
 */
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { after, before, test } from 'node:test';
import { Sandbox, SandboxNotFoundError } from 'e2b';
import { config } from 'dotenv';
import { prisma } from '../../providers/database';
import { NO_CODE_RESOURCES } from '../../providers/codeSessions/sandboxResources';
import { mintAgentToolsToken } from '../../providers/agentTools/token';
import { ENV_SIZE_SKILL_PATH, withAgentTools } from '../../providers/agentTools/workspace';
import {
  closeTaskForResize,
  performSandboxResize,
  requestSandboxResize,
  SandboxResizeError,
  stopSandboxCli,
} from '../../providers/codeSessions/sandboxResize';
import {
  DEFAULT_SANDBOX_GRID,
  resolveSpawnSandbox,
  sandboxGrid,
  sizedTemplateName,
} from '../../providers/codeSessions/sandboxSize';
import { keepAliveCodeTask } from '../../providers/codeSessions/spawn';
import {
  acquireCodeSandbox,
  recordE2bCodeSessionSandbox,
} from '../../providers/e2bCodeSessionSandbox';
import { writeUserUiWorkspaceToSandbox } from '../../providers/userUi/workspace';
import { assertResizeMigrationIsolation } from './code-session-resize-migration.cases';

config();

const baseUrl = process.env.TEST_BASE_URL || 'http://localhost:3001';
const apiKey = process.env.BACKEND_API_KEY || '';
const fixtureDir = mkdtempSync(join(tmpdir(), 'code-session-resize-'));
const userId = `resize-user-${randomUUID()}`;
const agentId = randomUUID();
const cliSessionId = randomUUID();
const roseCli = join(__dirname, '../../assets/rose-tools/rose.cjs');
const TEMPLATES = {
  claude: { '2x4': 'tpl-c2-m4', '2x8': 'tpl-c2-m8', '8x4': 'tpl-c8-m4', '8x8': 'tpl-c8-m8' },
};
const ENV_KEYS = [
  'CODE_SESSION_STORAGE_BACKEND',
  'CODE_SESSION_STORAGE_LOCAL_DIR',
  'E2B_SANDBOX_TEMPLATES',
  'CODE_SANDBOX_GRID',
  'CODE_SANDBOX_MAX_CPU',
  'CODE_SANDBOX_MAX_MEMORY_GB',
] as const;
const savedEnv = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
const originalConnect = Sandbox.connect.bind(Sandbox);
const originalCreate = Sandbox.create.bind(Sandbox);

function git(cwd: string, ...args: string[]): string {
  return execFileSync('git', args, { cwd, encoding: 'utf8' }).trim();
}

/** `<home>/workspace` on rose/feature-1 with a local commit, an uncommitted edit, an untracked file and a CLI log. */
function seedWorkspace(home: string): void {
  const origin = join(fixtureDir, `origin-${randomUUID()}.git`);
  execFileSync('git', ['init', '-q', '--bare', '-b', 'main', origin]);
  const workspace = join(home, 'workspace');
  mkdirSync(home, { recursive: true });
  execFileSync('git', ['clone', '-q', origin, workspace], { stdio: 'ignore' });
  git(workspace, 'config', 'user.name', 'Test');
  git(workspace, 'config', 'user.email', 'test@example.com');
  git(workspace, 'checkout', '-q', '-b', 'main');
  writeFileSync(join(workspace, 'README.md'), 'line one\n');
  git(workspace, 'add', '-A');
  git(workspace, 'commit', '-q', '-m', 'initial');
  git(workspace, 'push', '-q', 'origin', 'main');
  git(workspace, 'checkout', '-q', '-b', 'rose/feature-1');
  writeFileSync(join(workspace, 'feature.txt'), 'committed locally\n');
  git(workspace, 'add', 'feature.txt');
  git(workspace, 'commit', '-q', '-m', 'local commit');
  writeFileSync(join(workspace, 'README.md'), 'line one\nuncommitted edit\n');
  writeFileSync(join(workspace, 'untracked.txt'), 'untracked file\n');
  const logDir = join(home, '.claude', 'projects', '-home-user-workspace');
  mkdirSync(logDir, { recursive: true });
  writeFileSync(
    join(logDir, `${cliSessionId}.jsonl`),
    `${JSON.stringify({ type: 'user', timestamp: new Date().toISOString(), message: { content: 'build it' } })}\n`
  );
}

/** E2B sandbox stand-in over a local HOME: commands and files map /home/user → home. */
function fakeSandbox(home: string, sandboxId: string) {
  const map = (value: string) => value.split('/home/user').join(home);
  const calls = { paused: 0, killed: 0, commands: [] as string[] };
  const write = async (path: string, data: ArrayBuffer | Blob | string) => {
    mkdirSync(dirname(map(path)), { recursive: true });
    const bytes =
      typeof data === 'string'
        ? Buffer.from(data)
        : data instanceof ArrayBuffer
          ? Buffer.from(new Uint8Array(data))
          : Buffer.from(await data.arrayBuffer());
    writeFileSync(map(path), bytes);
  };
  const sandbox = {
    sandboxId,
    commands: {
      run: async (cmd: string, opts?: { cwd?: string }) => {
        calls.commands.push(cmd);
        const result = spawnSync('bash', ['-c', map(cmd)], {
          cwd: map(opts?.cwd ?? '/home/user'),
          encoding: 'utf8',
          env: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home, GIT_TERMINAL_PROMPT: '0' },
        });
        return { exitCode: result.status ?? 1, stdout: result.stdout, stderr: result.stderr };
      },
    },
    files: {
      read: async (path: string, opts?: { format?: string }) => {
        const data = readFileSync(map(path));
        return opts?.format === 'text' ? data.toString('utf8') : new Uint8Array(data);
      },
      write: async (
        pathOrEntries: string | Array<{ path: string; data: ArrayBuffer | Blob | string }>,
        data?: ArrayBuffer | Blob | string
      ) => {
        if (Array.isArray(pathOrEntries)) {
          for (const entry of pathOrEntries) await write(entry.path, entry.data);
          return;
        }
        await write(pathOrEntries, data ?? '');
      },
    },
    pause: async () => {
      calls.paused += 1;
      return true;
    },
    kill: async () => {
      calls.killed += 1;
    },
  };
  return { sandbox: sandbox as unknown as Sandbox, calls };
}

function patchE2b(
  connect: (id: string) => Promise<Sandbox>,
  create: (templateId: string) => Promise<Sandbox>
): void {
  (Sandbox as unknown as { connect: typeof connect }).connect = connect;
  (Sandbox as unknown as { create: typeof create }).create = create;
}

function restoreE2b(): void {
  (Sandbox as unknown as { connect: unknown }).connect = originalConnect;
  (Sandbox as unknown as { create: unknown }).create = originalCreate;
}

async function http(method: string, path: string, init: { body?: unknown; token?: string } = {}) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (init.token !== undefined) headers.Authorization = `Bearer ${init.token}`;
  else Object.assign(headers, { 'X-API-Key': apiKey, 'X-User-Id': userId });
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    headers,
    ...(init.body !== undefined ? { body: JSON.stringify(init.body) } : {}),
  });
  const text = await response.text();
  let data: unknown = text;
  try {
    data = JSON.parse(text);
  } catch {
    // keep text
  }
  return { status: response.status, data: data as Record<string, unknown> };
}

/** A persistent E2B session at 2 vCPU / 4 GB with an ACTIVE task. */
async function sessionWithTask(options: { sandboxId?: string; held?: [number, number] } = {}) {
  const sandboxId = options.sandboxId ?? `sb-${randomUUID()}`;
  const session = await prisma.codeSession.create({
    data: {
      userId,
      rootAgentId: agentId,
      ownerAgentId: agentId,
      repoKey: `resize-${randomUUID()}`,
      branch: 'rose/feature-1',
      mergeTarget: 'main',
      sandboxKind: 'e2b',
      sandboxId,
      sandboxCpu: options.held?.[0] ?? 2,
      sandboxMemoryGb: options.held?.[1] ?? 4,
      leaseOwner: `tool:${agentId}:call-1`,
      leaseExpiresAt: new Date(Date.now() + 60 * 60_000),
    },
  });
  const task = await prisma.claudeCodeSession.create({
    data: {
      agentId,
      sandboxId,
      streamUrl: `wss://${sandboxId}`,
      goal: 'Build the feature',
      status: 'ACTIVE',
      codeSessionId: session.id,
      codeProvider: 'claude',
    },
  });
  return { session, task, sandboxId };
}

async function expectResizeError(
  promise: Promise<unknown>,
  status: number,
  code: string
): Promise<SandboxResizeError> {
  try {
    await promise;
  } catch (err) {
    assert.ok(err instanceof SandboxResizeError, String(err));
    assert.equal(err.status, status, err.message);
    assert.equal(err.code, code, err.message);
    return err;
  }
  assert.fail(`expected ${status} ${code}`);
}

/** Finish a request the way performSandboxResize does, without a sandbox. */
async function completeResize(requestId: string, codeSessionId: string): Promise<void> {
  const row = await prisma.codeSessionResize.update({
    where: { id: requestId },
    data: { status: 'DONE', completedAt: new Date() },
  });
  await prisma.codeSession.update({
    where: { id: codeSessionId },
    data: { sandboxCpu: row.toCpu, sandboxMemoryGb: row.toMemoryGb },
  });
}

before(async () => {
  process.env.CODE_SESSION_STORAGE_BACKEND = 'local';
  process.env.CODE_SESSION_STORAGE_LOCAL_DIR = join(fixtureDir, 'storage');
  process.env.E2B_SANDBOX_TEMPLATES = JSON.stringify(TEMPLATES);
  delete process.env.CODE_SANDBOX_GRID;
  delete process.env.CODE_SANDBOX_MAX_CPU;
  delete process.env.CODE_SANDBOX_MAX_MEMORY_GB;
  await prisma.user.create({ data: { userId, name: 'Resize User' } });
  await prisma.agent.create({ data: { id: agentId, userId, goal: 'Build the feature' } });
});

after(async () => {
  restoreE2b();
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  const sessions = await prisma.codeSession.findMany({ where: { userId }, select: { id: true } });
  await prisma.codeSessionResize.deleteMany({ where: { userId } });
  await prisma.agentTrajectorySearch.deleteMany({ where: { userId } });
  await prisma.claudeCodeSession.deleteMany({ where: { agentId } });
  await prisma.codeSession.deleteMany({ where: { id: { in: sessions.map((row) => row.id) } } });
  await prisma.project.deleteMany({ where: { userId } });
  await prisma.agent.deleteMany({ where: { userId } });
  await prisma.user.deleteMany({ where: { userId } });
  rmSync(fixtureDir, { recursive: true, force: true });
  await prisma.$disconnect();
});

test(
  'resize migration keeps bookkeeping private from client roles',
  assertResizeMigrationIsolation
);

test('resize round trip: new sandbox from the size template keeps branch, files and conversation; retry is idempotent', async () => {
  const homeA = join(fixtureDir, 'home-a');
  seedWorkspace(homeA);
  const { session, task, sandboxId } = await sessionWithTask();
  const held = fakeSandbox(homeA, sandboxId);
  const homeB = join(fixtureDir, 'home-b');
  mkdirSync(homeB, { recursive: true });
  const next = fakeSandbox(homeB, `sb-${randomUUID()}`);
  const templates: string[] = [];
  const clock: Record<string, number> = {};
  patchE2b(
    async (id) => {
      if (id === sandboxId) return held.sandbox;
      if (id === next.sandbox.sandboxId) return next.sandbox;
      throw new SandboxNotFoundError(`Paused sandbox ${id} not found`);
    },
    async (templateId) => {
      clock.create = performance.now();
      templates.push(templateId);
      return next.sandbox;
    }
  );
  try {
    const requested = await requestSandboxResize({
      userId,
      agentId,
      taskId: task.id,
      memoryGb: 8,
      reason: 'jest workers OOM-killed 3 times at 4 GB',
    });
    assert.equal(requested.httpStatus, 202);
    assert.deepEqual(requested.body.to, { cpu: 2, memory_gb: 8 }, 'cpu unchanged, memory raised');
    const requestId = requested.body.request_id;
    const alive = await keepAliveCodeTask(task.id, async () => undefined, NO_CODE_RESOURCES);
    assert.equal(
      alive.resize_requested?.request_id,
      requestId,
      'keepalive hands it to the activity'
    );

    const goals: string[] = [];
    const create = async (input: { goal: string; codeSessionId: string; agentId: string }) => {
      goals.push(input.goal);
      const row = await prisma.codeSession.findUniqueOrThrow({
        where: { id: input.codeSessionId },
      });
      // The same acquire e2b.ts runs: the session's size picks the template.
      const acquired = await acquireCodeSandbox(row, 'unsized-fallback', {}, 'test', {
        provider: 'claude',
        agentId: input.agentId,
      });
      clock.acquired = performance.now();
      await recordE2bCodeSessionSandbox(row, acquired.sandbox.sandboxId);
      const created = await prisma.claudeCodeSession.create({
        data: {
          agentId,
          sandboxId: acquired.sandbox.sandboxId,
          streamUrl: `wss://${acquired.sandbox.sandboxId}`,
          goal: input.goal,
          codeSessionId: row.id,
          codeProvider: 'claude',
        },
      });
      return {
        task_id: created.id,
        sandbox_id: acquired.sandbox.sandboxId,
        ws_url: `wss://${acquired.sandbox.sandboxId}`,
        cli_session_id: acquired.cliSessionId,
        resumed_from: acquired.resumedFrom,
      };
    };
    clock.start = performance.now();
    const result = await performSandboxResize({
      taskId: task.id,
      requestId,
      stopCli: stopSandboxCli,
      closeTask: closeTaskForResize,
      create,
    });
    clock.end = performance.now();
    console.log(
      `  resize timing (fake E2B): stop+checkpoint ${Math.round(clock.create - clock.start)} ms, ` +
        `create+restore+kill ${Math.round(clock.acquired - clock.create)} ms, ` +
        `total ${Math.round(clock.end - clock.start)} ms`
    );

    assert.equal(result.resized, true);
    assert.deepEqual(templates, ['tpl-c2-m8'], 'the new size’s template is used');
    assert.ok(
      held.calls.commands.some((cmd) => cmd.includes('pgrep -f')),
      'the old CLI is stopped first'
    );
    assert.equal(held.calls.killed, 1, 'the held sandbox is killed after the restore');
    const session1 = result.session as { cli_session_id?: string; resumed_from?: string };
    assert.equal(session1.resumed_from, 'checkpoint');
    assert.equal(session1.cli_session_id, cliSessionId, 'the CLI conversation resumes');
    assert.match(goals[0] ?? '', /resized from 2 vCPU, 4 GB RAM to 2 vCPU, 8 GB RAM/);
    assert.match(goals[0] ?? '', /jest workers OOM-killed/);

    const workspace = join(homeB, 'workspace');
    assert.equal(git(workspace, 'rev-parse', '--abbrev-ref', 'HEAD'), 'rose/feature-1');
    assert.equal(git(workspace, 'log', '-1', '--format=%s'), 'local commit');
    assert.equal(
      readFileSync(join(workspace, 'README.md'), 'utf8'),
      'line one\nuncommitted edit\n'
    );
    assert.equal(readFileSync(join(workspace, 'untracked.txt'), 'utf8'), 'untracked file\n');
    assert.ok(
      existsSync(
        join(homeB, '.claude', 'projects', '-home-user-workspace', `${cliSessionId}.jsonl`)
      )
    );

    const row = await prisma.codeSession.findUniqueOrThrow({ where: { id: session.id } });
    assert.equal(row.sandboxId, next.sandbox.sandboxId);
    assert.deepEqual([row.sandboxCpu, row.sandboxMemoryGb], [2, 8]);
    assert.ok(row.latestCheckpointSeq >= 1, 'the move went through a checkpoint');
    const oldTask = await prisma.claudeCodeSession.findUniqueOrThrow({ where: { id: task.id } });
    assert.equal(oldTask.status, 'CLOSED');
    const stored = await prisma.codeSessionResize.findUniqueOrThrow({ where: { id: requestId } });
    assert.equal(stored.status, 'DONE');
    assert.equal(stored.newTaskId, result.session.task_id);

    // A Temporal retry of the same call returns the same task; no second sandbox.
    const retried = await performSandboxResize({
      taskId: task.id,
      requestId,
      stopCli: stopSandboxCli,
      closeTask: closeTaskForResize,
      create,
    });
    assert.equal(retried.session.task_id, result.session.task_id);
    assert.equal(templates.length, 1, 'no duplicate sandbox on retry');
    assert.equal(
      (await keepAliveCodeTask(retried.session.task_id, async () => undefined, NO_CODE_RESOURCES))
        .resize_requested,
      undefined
    );
  } finally {
    restoreE2b();
  }
});

test('guards: reason 400, per-dimension cap 403, in flight / cooldown / reverse 409, unchanged 200', async () => {
  process.env.CODE_SANDBOX_GRID = JSON.stringify({ cpu: [2, 4, 8], memoryGb: [4, 6, 8] });
  process.env.E2B_SANDBOX_TEMPLATES = 'grid';
  try {
    const { session, task } = await sessionWithTask();
    const ask = (size: { cpu?: number; memoryGb?: number }, reason = 'builds are OOM-killed') =>
      requestSandboxResize({ userId, agentId, taskId: task.id, ...size, reason });

    await expectResizeError(ask({ memoryGb: 8 }, '  '), 400, 'REASON_REQUIRED');
    await expectResizeError(ask({}), 400, 'SIZE_REQUIRED');
    await expectResizeError(ask({ cpu: 32 }), 409, 'SIZE_UNAVAILABLE');
    const unchanged = await ask({ cpu: 2 });
    assert.equal(unchanged.httpStatus, 200);
    assert.equal(unchanged.body.status, 'unchanged');

    await prisma.user.update({ where: { userId }, data: { codeSandboxMaxMemoryGb: 6 } });
    const capped = await expectResizeError(ask({ memoryGb: 8 }), 403, 'SIZE_ABOVE_CAP');
    assert.deepEqual(capped.details.dimensions, ['memoryGb']);
    const closest = await ask({ memoryGb: 5 });
    assert.deepEqual(closest.body.to, { cpu: 2, memory_gb: 6 }, 'closest size that satisfies');
    await prisma.user.update({ where: { userId }, data: { codeSandboxMaxMemoryGb: null } });

    const again = await ask({ memoryGb: 6 });
    assert.equal(again.body.request_id, closest.body.request_id, 'same pending request (retry)');
    await expectResizeError(ask({ cpu: 8 }), 409, 'RESIZE_IN_PROGRESS');

    await completeResize(closest.body.request_id, session.id); // now 2 vCPU / 6 GB
    await expectResizeError(ask({ memoryGb: 8 }), 409, 'RATE_LIMITED');
    const reverse = await expectResizeError(ask({ memoryGb: 4 }), 409, 'REVERSE_RESIZE');
    assert.equal(reverse.details.dimension, 'memoryGb');
    const cpu = await ask({ cpu: 8 }, 'tsc + vitest CPU-bound for 6 min');
    assert.equal(cpu.httpStatus, 202, 'the other dimension is independent');
    await completeResize(cpu.body.request_id, session.id); // now 8 vCPU / 6 GB
    const cpuBack = await expectResizeError(ask({ cpu: 2 }), 409, 'REVERSE_RESIZE');
    assert.equal(cpuBack.details.dimension, 'cpu');
    await expectResizeError(ask({ cpu: 4 }), 409, 'RATE_LIMITED');

    process.env.CODE_SANDBOX_GRID = JSON.stringify({ cpu: [1, 2, 4, 8], memoryGb: [1, 2, 4, 8] });
    assert.deepEqual(sandboxGrid(), DEFAULT_SANDBOX_GRID, 'a grid over 9 sizes is refused');
  } finally {
    delete process.env.CODE_SANDBOX_GRID;
    process.env.E2B_SANDBOX_TEMPLATES = JSON.stringify(TEMPLATES);
  }
});

test('the default grid offers 4 and 8 GB and refuses 16 GB (E2B team cap of 8192 MiB)', async () => {
  process.env.E2B_SANDBOX_TEMPLATES = 'grid';
  try {
    assert.deepEqual(DEFAULT_SANDBOX_GRID, { cpu: [2, 8], memoryGb: [4, 8] });
    assert.deepEqual(sandboxGrid(), DEFAULT_SANDBOX_GRID);
    const { task } = await sessionWithTask();
    const refused = await expectResizeError(
      requestSandboxResize({
        userId,
        agentId,
        taskId: task.id,
        memoryGb: 16,
        reason: 'jest workers OOM-killed 3 times at 8 GB',
      }),
      409,
      'SIZE_UNAVAILABLE'
    );
    assert.deepEqual(refused.details.available, [
      { cpu: 2, memory_gb: 4 },
      { cpu: 2, memory_gb: 8 },
      { cpu: 8, memory_gb: 4 },
      { cpu: 8, memory_gb: 8 },
    ]);
    const fresh = await prisma.codeSession.create({
      data: { userId, rootAgentId: agentId, ownerAgentId: agentId, repoKey: `g-${randomUUID()}` },
    });
    // A new session takes the new-session default (8 vCPU / 8 GB when built).
    assert.deepEqual(await resolveSpawnSandbox(fresh, 'codex', {}), {
      size: { cpu: 8, memoryGb: 8 },
      templateId: sizedTemplateName('codex', { cpu: 8, memoryGb: 8 }),
      wanted: { cpu: 8, memoryGb: 8 },
      source: 'default',
    });
  } finally {
    process.env.E2B_SANDBOX_TEMPLATES = JSON.stringify(TEMPLATES);
  }
});

test('HTTP: no reason 400, local target not applicable, status lists the top consumers', async () => {
  const { task } = await sessionWithTask();
  const token = mintAgentToolsToken(userId, agentId, { provider: 'claude', taskId: task.id }).token;
  const base = `/api/agent-tools/${agentId}`;

  const noReason = await http('POST', `${base}/environment/size`, {
    token,
    body: { memory_gb: 8 },
  });
  assert.equal(noReason.status, 400, JSON.stringify(noReason.data));
  assert.equal(noReason.data.code, 'REASON_REQUIRED');
  const internal = await http('POST', `${base}/environment/size`, {
    body: { memory_gb: 8, reason: 'x' },
  });
  assert.equal(internal.status, 403, 'only a code-agent session token can resize');

  const localTask = await prisma.claudeCodeSession.create({
    data: {
      agentId,
      sandboxId: `mac-${randomUUID()}-${randomUUID()}`,
      goal: 'local',
      status: 'ACTIVE',
      codeProvider: 'claude',
    },
  });
  const localToken = mintAgentToolsToken(userId, agentId, {
    provider: 'claude',
    taskId: localTask.id,
  }).token;
  const local = await http('POST', `${base}/environment/size`, {
    token: localToken,
    body: { cpu: 8, reason: 'slow build' },
  });
  assert.equal(local.status, 409, JSON.stringify(local.data));
  assert.equal(local.data.code, 'NOT_APPLICABLE');
  const localStatus = await http('GET', `${base}/environment`, { token: localToken });
  assert.equal(localStatus.data.applicable, false);

  const status = await http('GET', `${base}/environment`, { token });
  assert.equal(status.status, 200, JSON.stringify(status.data));
  assert.deepEqual(status.data.size, { cpu: 2, memory_gb: 4 });

  const env = {
    ...process.env,
    ROSE_TOOLS_URL: `${baseUrl}${base}`,
    ROSE_TOOLS_TOKEN: token,
    ROSE_ENV_CPU_SAMPLE_MS: '200',
  };
  const json = JSON.parse(
    execFileSync(process.execPath, [roseCli, 'env', 'status', '--json'], { env, encoding: 'utf8' })
  ) as {
    rose: { applicable: boolean; size: unknown };
    memory: { total_gb: number };
    top_memory: Array<{ pid: number; rss_mb: number; command: string }>;
    top_cpu: Array<{ pid: number; cpu_pct: number }>;
  };
  assert.equal(json.rose.applicable, true);
  assert.deepEqual(json.rose.size, { cpu: 2, memory_gb: 4 });
  assert.ok(json.memory.total_gb > 0);
  assert.equal(json.top_memory.length, 5);
  assert.equal(json.top_cpu.length, 5);
  const rss = json.top_memory.map((row) => row.rss_mb);
  assert.deepEqual(
    rss,
    [...rss].sort((a, b) => b - a),
    'sorted by memory'
  );
  assert.ok(json.top_memory.every((row) => row.pid > 0 && row.command.length > 0));
  const text = execFileSync(process.execPath, [roseCli, 'env', 'status'], {
    env,
    encoding: 'utf8',
  });
  assert.match(text, /Environment: 2 vCPU, 4 GB RAM/);
  assert.match(text, /Top 5 processes by memory \(RSS\):/);
  assert.match(text, /Top 5 processes by CPU/);
  const cli = spawnSync(process.execPath, [roseCli, 'env', 'size', '--memory-gb', '8'], {
    env,
    encoding: 'utf8',
  });
  assert.equal(cli.status, 1, 'the CLI refuses a resize without a reason');
  assert.match(cli.stderr, /reason is required/);

  const unknown = await http('POST', `/api/claude-code-session/${task.id}/resize`, {
    body: { request_id: randomUUID() },
  });
  assert.equal(unknown.status, 404, JSON.stringify(unknown.data));
  assert.equal(unknown.data.code, 'RESIZE_NOT_FOUND');
});

test('project sandboxSize is the default size of a new session', async () => {
  const created = await http('POST', '/api/projects', {
    body: { name: `resize-${randomUUID().slice(0, 8)}`, sandboxSize: { memoryGb: 8 } },
  });
  assert.equal(created.status, 201, JSON.stringify(created.data));
  const project = (created.data.project ?? created.data) as { id: string; sandboxSize: unknown };
  assert.deepEqual(project.sandboxSize, { memoryGb: 8 });
  const invalid = await http('PUT', `/api/projects/${project.id}`, {
    body: { sandboxSize: {} },
  });
  assert.equal(invalid.status, 400, JSON.stringify(invalid.data));
  const stored = await prisma.project.findUniqueOrThrow({ where: { id: project.id } });

  const fresh = await prisma.codeSession.create({
    data: { userId, rootAgentId: agentId, ownerAgentId: agentId, repoKey: `p-${randomUUID()}` },
  });
  // A project's partial size fills the missing dimension from the new-session default.
  const sized = await resolveSpawnSandbox(fresh, 'claude', { projectSize: stored.sandboxSize });
  assert.deepEqual(sized, {
    size: { cpu: 8, memoryGb: 8 },
    templateId: 'tpl-c8-m8',
    wanted: { cpu: 8, memoryGb: 8 },
    source: 'project',
  });
  const { session } = await sessionWithTask();
  const existing = await resolveSpawnSandbox(session, 'claude', {
    projectSize: stored.sandboxSize,
  });
  assert.deepEqual(existing.size, { cpu: 2, memoryGb: 4 }, 'an existing session keeps its size');

  const cleared = await http('PUT', `/api/projects/${project.id}`, { body: { sandboxSize: null } });
  assert.equal(cleared.status, 200, JSON.stringify(cleared.data));
  const clearedProject = (cleared.data.project ?? cleared.data) as { sandboxSize: unknown };
  assert.equal(clearedProject.sandboxSize, null);
});

test('the change-environment-size skill is installed in the session HOME', async () => {
  const workspace = withAgentTools(null, {
    userId,
    agentId,
    taskId: randomUUID(),
    provider: 'claude',
  });
  assert.ok(workspace);
  const home = join(fixtureDir, 'home-skill');
  mkdirSync(home, { recursive: true });
  await writeUserUiWorkspaceToSandbox(
    fakeSandbox(home, 'sb-skill').sandbox,
    '/home/user',
    workspace
  );
  const skill = readFileSync(join(home, ENV_SIZE_SKILL_PATH), 'utf8');
  assert.match(skill, /^---\nname: change-environment-size\ndescription: Use .+\n---\n/);
  assert.match(skill, /rose env status/);
  assert.match(skill, /When not to use/);
  assert.ok(ENV_SIZE_SKILL_PATH.startsWith('.rose/skills/change-environment-size/'));
});
