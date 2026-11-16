/**
 * Fixtures for the code-session-resize blackbox: a fake E2B over local HOME
 * directories (with per-command fault injection), seeded workspaces, sessions
 * and the create callback the resize activity runs. Nothing reaches E2B.
 */
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { Sandbox } from 'e2b';
import { config } from 'dotenv';
import { prisma } from '../../providers/database';
import { SandboxResizeError } from '../../providers/codeSessions/sandboxResize';

config();

export const baseUrl = process.env.TEST_BASE_URL || 'http://localhost:3001';
export const apiKey = process.env.BACKEND_API_KEY || '';
export const fixtureDir = mkdtempSync(join(tmpdir(), 'code-session-resize-'));
export const userId = `resize-user-${randomUUID()}`;
export const agentId = randomUUID();
export const cliSessionId = randomUUID();
export const roseCli = join(__dirname, '../../assets/rose-tools/rose.cjs');
export const TEMPLATES = {
  claude: { '2x4': 'tpl-c2-m4', '2x8': 'tpl-c2-m8', '8x4': 'tpl-c8-m4', '8x8': 'tpl-c8-m8' },
};
export const ENV_KEYS = [
  'CODE_SESSION_STORAGE_BACKEND',
  'CODE_SESSION_STORAGE_LOCAL_DIR',
  'E2B_SANDBOX_TEMPLATES',
  'CODE_SANDBOX_GRID',
  'CODE_SANDBOX_MAX_CPU',
  'CODE_SANDBOX_MAX_MEMORY_GB',
] as const;
export const savedEnv = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
const originalConnect = Sandbox.connect.bind(Sandbox);
const originalCreate = Sandbox.create.bind(Sandbox);
const originalKill = Sandbox.kill.bind(Sandbox);

export function git(cwd: string, ...args: string[]): string {
  return execFileSync('git', args, { cwd, encoding: 'utf8' }).trim();
}

/** `<home>/workspace` on rose/feature-1 with a local commit, an uncommitted edit, an untracked file and a CLI log. */
export function seedWorkspace(home: string): void {
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
/** A command result that replaces the real one (fault injection), or null to run it. */
export type CommandFault = (
  cmd: string
) => { exitCode: number; stdout: string; stderr: string } | null;

export function fakeSandbox(home: string, sandboxId: string, fault: CommandFault = () => null) {
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
        // A fixed, roomy disk unless a case injects another: never the host's own free space.
        const injected = fault(cmd) ?? (cmd.includes('df -Pk') ? dfOutput(25, 20) : null);
        if (injected !== null) return injected;
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

/** Sandbox ids killed through the static `Sandbox.kill(id)` while patched. */
export const killedById: string[] = [];

export function patchE2b(
  connect: (id: string) => Promise<Sandbox>,
  create: (templateId: string) => Promise<Sandbox>
): void {
  (Sandbox as unknown as { connect: typeof connect }).connect = connect;
  (Sandbox as unknown as { create: typeof create }).create = create;
  (Sandbox as unknown as { kill: (id: string) => Promise<boolean> }).kill = async (id) => {
    killedById.push(id);
    return true;
  };
}

export function restoreE2b(): void {
  (Sandbox as unknown as { connect: unknown }).connect = originalConnect;
  (Sandbox as unknown as { create: unknown }).create = originalCreate;
  (Sandbox as unknown as { kill: unknown }).kill = originalKill;
}

/** `df -Pk` output for a filesystem with `totalGb` of which `availGb` is free. */
export function dfOutput(totalGb: number, availGb: number) {
  const kb = (value: number) => Math.round(value * 1024 * 1024);
  return {
    exitCode: 0,
    stdout: `/dev/vda ${kb(totalGb)} ${kb(totalGb - availGb)} ${kb(availGb)} 50% /\n`,
    stderr: '',
  };
}

export async function http(
  method: string,
  path: string,
  init: { body?: unknown; token?: string } = {}
) {
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
export async function sessionWithTask(
  options: { sandboxId?: string; held?: [number, number] } = {}
) {
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

export async function expectResizeError(
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
export async function completeResize(requestId: string, codeSessionId: string): Promise<void> {
  const row = await prisma.codeSessionResize.update({
    where: { id: requestId },
    data: { status: 'DONE', completedAt: new Date() },
  });
  await prisma.codeSession.update({
    where: { id: codeSessionId },
    data: { sandboxCpu: row.toCpu, sandboxMemoryGb: row.toMemoryGb },
  });
}
