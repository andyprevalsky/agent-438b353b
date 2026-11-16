/**
 * Fault injection for the transactional resize (resizeTransaction.ts):
 *
 *   held disk full ─► HELD_DISK_FULL before any checkpoint, held resumed
 *   target disk too small ─► TARGET_DISK_CAPACITY, no retry, target killed, held kept
 *   restore fails once ─► second fresh target restores; first target killed
 *   restore never validates ─► RESTORE_VALIDATION after RESIZE_RESTORE_ATTEMPTS, held kept
 *   ws-server fails after cut-over ─► rollback: row names the original again,
 *     fallback resumes it, only then the target is killed
 *   backend restart after cut-over ─► stale claim re-run commits (kills the original by id)
 *   duplicate requests ─► one request id, one apply, one target sandbox
 *
 * Every case asserts the invariant: the original sandbox is never killed
 * unless the session row names a restored target that served.
 */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { SandboxNotFoundError } from 'e2b';
import { prisma } from '../../providers/database';
import {
  closeTaskForResize,
  performSandboxResize,
  requestSandboxResize,
  SandboxResizeError,
  stopSandboxCli,
} from '../../providers/codeSessions/sandboxResize';
import {
  acquireCodeSandbox,
  recordE2bCodeSessionSandbox,
  releaseSandboxAfterFailedStart,
} from '../../providers/e2bCodeSessionSandbox';
import {
  agentId,
  dfOutput,
  fakeSandbox,
  fixtureDir,
  git,
  killedById,
  patchE2b,
  restoreE2b,
  seedWorkspace,
  sessionWithTask,
  userId,
  type CommandFault,
} from './code-session-resize.support';

type Fake = ReturnType<typeof fakeSandbox>;

const isDf: (cmd: string) => boolean = (cmd) => cmd.includes('df -Pk');
const failTar: CommandFault = (cmd) =>
  cmd.includes('tar -xzf') ? { exitCode: 2, stdout: '', stderr: 'tar: injected failure' } : null;
const wrongHead: CommandFault = (cmd) =>
  cmd.includes('git rev-parse HEAD && (git symbolic-ref')
    ? { exitCode: 0, stdout: `${'d'.repeat(40)}\nrose/feature-1\n`, stderr: '' }
    : null;

/** A seeded session at 2x4 on `held`, with `targets` handed out by Sandbox.create in order. */
async function scenario(options: { heldFault?: CommandFault; targetFaults?: CommandFault[] }) {
  const id = randomUUID();
  const homeA = join(fixtureDir, `held-${id}`);
  seedWorkspace(homeA);
  const { session, task, sandboxId } = await sessionWithTask();
  const held = fakeSandbox(homeA, sandboxId, options.heldFault);
  const faults = options.targetFaults ?? [() => null, () => null];
  const homes = faults.map((_, i) => join(fixtureDir, `target-${id}-${i}`));
  const targets: Fake[] = faults.map((fault, i) => {
    mkdirSync(homes[i] as string, { recursive: true });
    return fakeSandbox(homes[i] as string, `sb-target-${i}-${randomUUID()}`, fault);
  });
  const created: string[] = [];
  patchE2b(
    async (sid) => {
      if (sid === sandboxId) return held.sandbox;
      const target = targets.find((t) => t.sandbox.sandboxId === sid);
      if (target !== undefined) return target.sandbox;
      throw new SandboxNotFoundError(`sandbox ${sid} not found`);
    },
    async () => {
      const next = targets[created.length];
      if (next === undefined) throw new Error('no more target sandboxes');
      created.push(next.sandbox.sandboxId);
      return next.sandbox;
    }
  );
  killedById.length = 0;
  return { session, task, held, targets, homes, created };
}

/** The create the resize activity runs; `failAfterAcquire` fakes a ws-server that never serves. */
function resizeCreate(options: { failAfterAcquire?: number } = {}) {
  const calls: Array<{ goal: string; resumedFrom: string; sandboxId: string }> = [];
  const create = async (input: { goal: string; codeSessionId: string; agentId: string }) => {
    const row = await prisma.codeSession.findUniqueOrThrow({ where: { id: input.codeSessionId } });
    const acquired = await acquireCodeSandbox(row, 'unsized-fallback', {}, 'test', {
      provider: 'claude',
      agentId: input.agentId,
    });
    const sandboxId = acquired.sandbox.sandboxId;
    calls.push({ goal: input.goal, resumedFrom: acquired.resumedFrom, sandboxId });
    if (options.failAfterAcquire === calls.length) {
      await releaseSandboxAfterFailedStart(acquired.sandbox, row);
      throw new Error('WebSocket server did not become ready');
    }
    await recordE2bCodeSessionSandbox(row, sandboxId);
    const task = await prisma.claudeCodeSession.create({
      data: {
        agentId,
        sandboxId,
        streamUrl: `wss://${sandboxId}`,
        goal: input.goal,
        codeSessionId: row.id,
        codeProvider: 'claude',
      },
    });
    return { task_id: task.id, sandbox_id: sandboxId, ws_url: `wss://${sandboxId}` };
  };
  return { create, calls };
}

async function resize(taskId: string, create: ReturnType<typeof resizeCreate>['create']) {
  const requested = await requestSandboxResize({
    userId,
    agentId,
    taskId,
    memoryGb: 8,
    reason: 'vitest workers OOM-killed at 4 GB',
  });
  assert.equal(requested.httpStatus, 202);
  const result = await performSandboxResize({
    taskId,
    requestId: requested.body.request_id,
    stopCli: stopSandboxCli,
    closeTask: closeTaskForResize,
    create,
  });
  return { requestId: requested.body.request_id, result };
}

async function sessionRow(id: string) {
  return prisma.codeSession.findUniqueOrThrow({ where: { id } });
}

/** The failure left the session on the original sandbox and size, resumed in place. */
async function assertKeptOriginal(
  sessionId: string,
  held: Fake,
  calls: ReturnType<typeof resizeCreate>['calls']
) {
  const row = await sessionRow(sessionId);
  assert.equal(row.sandboxId, held.sandbox.sandboxId, 'the row names the original sandbox');
  assert.deepEqual([row.sandboxCpu, row.sandboxMemoryGb], [2, 4], 'and its size');
  assert.equal(held.calls.killed, 0, 'the original sandbox is never killed');
  assert.ok(!killedById.includes(held.sandbox.sandboxId), 'not even by id');
  const fallback = calls[calls.length - 1];
  assert.equal(fallback?.sandboxId, held.sandbox.sandboxId);
  assert.equal(fallback?.resumedFrom, 'paused_sandbox', 'the original resumes in place');
}

export function registerResizeFaultTests(): void {
  test('fault: held disk full fails before any checkpoint with an actionable error', async () => {
    const s = await scenario({
      heldFault: (cmd) => (isDf(cmd) ? dfOutput(25, 0.2) : null),
    });
    const { create, calls } = resizeCreate();
    try {
      const { result } = await resize(s.task.id, create);
      assert.equal(result.resized, false);
      assert.equal(result.request.status, 'failed');
      assert.match(result.request.error ?? '', /^HELD_DISK_FULL: .*0\.2 GB free.*Free space/);
      assert.equal(typeof result.request.request_to_settled_ms, 'number');
      assert.deepEqual(s.created, [], 'no target sandbox is created');
      assert.equal((await sessionRow(s.session.id)).latestCheckpointSeq, 0, 'no checkpoint');
      assert.match(calls[calls.length - 1]?.goal ?? '', /HELD_DISK_FULL/);
      await assertKeptOriginal(s.session.id, s.held, calls);
    } finally {
      restoreE2b();
    }
  });

  test('fault: target disk too small for the restore is refused, not retried', async () => {
    const s = await scenario({
      heldFault: (cmd) => (isDf(cmd) ? dfOutput(25, 5) : null),
      targetFaults: [(cmd) => (isDf(cmd) ? dfOutput(10, 7) : null), () => null],
    });
    const { create, calls } = resizeCreate();
    try {
      const { result } = await resize(s.task.id, create);
      assert.equal(result.resized, false);
      assert.match(result.request.error ?? '', /^TARGET_DISK_CAPACITY: .*7\.0 GB free/);
      assert.equal(s.created.length, 1, 'a capacity refusal is not retried');
      assert.equal(s.targets[0]?.calls.killed, 1, 'the rejected target is killed');
      assert.ok((await sessionRow(s.session.id)).latestCheckpointSeq >= 1);
      await assertKeptOriginal(s.session.id, s.held, calls);
    } finally {
      restoreE2b();
    }
  });

  test('fault: a failed restore retries once on a fresh target and completes', async () => {
    const s = await scenario({ targetFaults: [failTar, () => null] });
    const { create } = resizeCreate();
    try {
      const { result } = await resize(s.task.id, create);
      assert.equal(result.resized, true, result.request.error ?? '');
      assert.equal(s.created.length, 2);
      assert.equal(s.targets[0]?.calls.killed, 1, 'the failed target is killed');
      assert.equal(s.targets[1]?.calls.killed, 0);
      assert.equal(s.held.calls.killed, 1, 'the original goes only after the target served');
      const row = await sessionRow(s.session.id);
      assert.equal(row.sandboxId, s.targets[1]?.sandbox.sandboxId);
      assert.deepEqual([row.sandboxCpu, row.sandboxMemoryGb], [2, 8]);
      const workspace = join(s.homes[1] as string, 'workspace');
      assert.equal(git(workspace, 'rev-parse', '--abbrev-ref', 'HEAD'), 'rose/feature-1');
      assert.equal(
        readFileSync(join(workspace, 'README.md'), 'utf8'),
        'line one\nuncommitted edit\n',
        'uncommitted changes survive'
      );
      assert.equal(typeof result.request.apply_ms, 'number');
    } finally {
      restoreE2b();
    }
  });

  test('fault: a restore that never validates fails after bounded attempts, original kept', async () => {
    const s = await scenario({ targetFaults: [wrongHead, wrongHead, () => null] });
    const { create, calls } = resizeCreate();
    try {
      const { result } = await resize(s.task.id, create);
      assert.equal(result.resized, false);
      assert.match(result.request.error ?? '', /^RESTORE_VALIDATION: .*HEAD d{40} != [0-9a-f]{40}/);
      assert.equal(s.created.length, 2, 'bounded: RESIZE_RESTORE_ATTEMPTS fresh targets');
      assert.equal(s.targets[0]?.calls.killed, 1);
      assert.equal(s.targets[1]?.calls.killed, 1);
      await assertKeptOriginal(s.session.id, s.held, calls);
    } finally {
      restoreE2b();
    }
  });

  test('fault: interruption after cut-over rolls back to the original sandbox', async () => {
    const s = await scenario({});
    const { create, calls } = resizeCreate({ failAfterAcquire: 1 });
    try {
      const { result } = await resize(s.task.id, create);
      assert.equal(result.resized, false);
      assert.match(result.request.error ?? '', /^TARGET_START_FAILED: WebSocket server/);
      assert.equal(calls[0]?.sandboxId, s.targets[0]?.sandbox.sandboxId, 'the target was cut over');
      assert.ok(s.held.calls.paused >= 1, 'the original was paused at cut-over, not killed');
      await assertKeptOriginal(s.session.id, s.held, calls);
      assert.deepEqual(
        killedById,
        [s.targets[0]?.sandbox.sandboxId],
        'the half-started target is killed only after the original serves again'
      );
    } finally {
      restoreE2b();
    }
  });

  test('fault: a backend restart after cut-over is committed by the stale-claim re-run', async () => {
    const s = await scenario({});
    const target = s.targets[0] as Fake;
    const targetHome = join(fixtureDir, `restarted-${randomUUID()}`);
    seedWorkspace(targetHome);
    const restored = fakeSandbox(targetHome, target.sandbox.sandboxId);
    patchE2b(
      async (sid) => {
        if (sid === restored.sandbox.sandboxId) return restored.sandbox;
        if (sid === s.held.sandbox.sandboxId) return s.held.sandbox;
        throw new SandboxNotFoundError(sid);
      },
      async () => assert.fail('a committed cut-over needs no new sandbox')
    );
    const { create } = resizeCreate();
    try {
      const requested = await requestSandboxResize({
        userId,
        agentId,
        taskId: s.task.id,
        memoryGb: 8,
        reason: 'build OOM',
      });
      // State a crash leaves after cut-over: claim RUNNING (stale), row on the target.
      await prisma.codeSessionResize.update({
        where: { id: requested.body.request_id },
        data: { status: 'RUNNING', startedAt: new Date(Date.now() - 20 * 60_000) },
      });
      await prisma.codeSession.update({
        where: { id: s.session.id },
        data: { sandboxId: target.sandbox.sandboxId, sandboxCpu: 2, sandboxMemoryGb: 8 },
      });
      const result = await performSandboxResize({
        taskId: s.task.id,
        requestId: requested.body.request_id,
        stopCli: stopSandboxCli,
        closeTask: closeTaskForResize,
        create,
      });
      assert.equal(result.resized, true);
      assert.equal(result.session.sandbox_id, target.sandbox.sandboxId);
      assert.deepEqual(killedById, [s.held.sandbox.sandboxId], 'the original is released by id');
      assert.equal(
        git(join(targetHome, 'workspace'), 'rev-parse', '--abbrev-ref', 'HEAD'),
        'rose/feature-1'
      );
    } finally {
      restoreE2b();
    }
  });

  test('fault: duplicate requests and concurrent applies run one resize', async () => {
    const s = await scenario({});
    const { create } = resizeCreate();
    try {
      const params = { userId, agentId, taskId: s.task.id, memoryGb: 8, reason: 'OOM' };
      const first = await requestSandboxResize(params);
      const second = await requestSandboxResize(params);
      assert.equal(second.body.request_id, first.body.request_id, 'same pending request');
      const apply = () =>
        performSandboxResize({
          taskId: s.task.id,
          requestId: first.body.request_id,
          stopCli: stopSandboxCli,
          closeTask: closeTaskForResize,
          create,
        });
      const settled = await Promise.allSettled([apply(), apply()]);
      const done = settled.filter((r) => r.status === 'fulfilled');
      assert.ok(done.length >= 1);
      for (const r of settled) {
        if (r.status === 'rejected') {
          assert.ok(r.reason instanceof SandboxResizeError);
          assert.equal((r.reason as SandboxResizeError).code, 'RESIZE_IN_PROGRESS');
        }
      }
      assert.equal(s.created.length, 1, 'one target sandbox');
      const again = await apply();
      const winner = (done[0] as PromiseFulfilledResult<Awaited<ReturnType<typeof apply>>>).value;
      assert.equal(again.session.task_id, winner.session.task_id, 'a retry returns the same task');
      assert.equal(s.created.length, 1);
      assert.equal(s.held.calls.killed, 1);
    } finally {
      restoreE2b();
    }
  });
}
