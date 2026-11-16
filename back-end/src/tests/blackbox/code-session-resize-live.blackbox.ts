#!/usr/bin/env tsx
/**
 * LIVE E2B check of the C16 resize transaction (opt-in: RUN_LIVE_E2B_RESIZE=1;
 * otherwise every case skips and nothing reaches E2B). Bounded spend: at most
 * three real sandboxes from the built sized templates, each with a 15-minute
 * E2B timeout, all killed in `after`. No CLI, credential or LLM is involved.
 *
 *   held 2x4 sandbox: ~/workspace on rose/feature-1 (local commit, uncommitted
 *   edit, untracked file) + a Claude CLI log
 *     ├─ moveToResizedSandbox → real 8x8 sandbox: nproc / MemTotal, branch,
 *     │    commits, files and CLI log sha256 restored; held paused; phases timed
 *     ├─ create fails → TARGET_CREATE_FAILED: the row still names the held
 *     │    sandbox and its work is intact (rollback)
 *     └─ held disk filled → HELD_DISK_FULL before any target is created
 *
 * Checkpoints go to local storage (CODE_SESSION_STORAGE_BACKEND=local).
 */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { Sandbox } from 'e2b';
import { config } from 'dotenv';
import { prisma } from '../../providers/database';
import { moveToResizedSandbox } from '../../providers/codeSessions/resizeMove';
import {
  newResizeTransaction,
  resizeFailureCode,
  runInResizeTransaction,
} from '../../providers/codeSessions/resizeTransaction';
import { sizedTemplateName, type SandboxSize } from '../../providers/codeSessions/sandboxSize';

config();

const live = process.env.RUN_LIVE_E2B_RESIZE === '1' && Boolean(process.env.E2B_API_KEY);
const skip = live ? false : 'set RUN_LIVE_E2B_RESIZE=1 (and E2B_API_KEY) to run against real E2B';
const userId = `resize-live-${randomUUID()}`;
const agentId = randomUUID();
const cliSessionId = randomUUID();
const SMALL: SandboxSize = { cpu: 2, memoryGb: 4 };
const LARGE: SandboxSize = { cpu: 8, memoryGb: 8 };
const created: Sandbox[] = [];
const saved = {
  backend: process.env.CODE_SESSION_STORAGE_BACKEND,
  dir: process.env.CODE_SESSION_STORAGE_LOCAL_DIR,
};
const CLI_LOG = `/home/user/.claude/projects/-home-user-workspace/${cliSessionId}.jsonl`;

async function create(size: SandboxSize): Promise<Sandbox> {
  const sandbox = await Sandbox.create(sizedTemplateName('claude', size), {
    apiKey: process.env.E2B_API_KEY,
    timeoutMs: 15 * 60_000,
    metadata: { roseTest: 'code-session-resize-live' },
  });
  created.push(sandbox);
  return sandbox;
}

async function sh(sandbox: Sandbox, cmd: string): Promise<string> {
  const result = await sandbox.commands.run(cmd, { timeoutMs: 120_000 });
  return result.stdout.trim();
}

async function seed(sandbox: Sandbox): Promise<string> {
  await sh(
    sandbox,
    [
      'set -e',
      'git init -q --bare -b main ~/origin.git',
      'git clone -q ~/origin.git ~/workspace 2>/dev/null',
      'cd ~/workspace && git config user.name Test && git config user.email test@example.com',
      'git checkout -q -b main && echo "line one" > README.md && git add -A && git commit -q -m initial',
      'git push -q origin main && git checkout -q -b rose/feature-1',
      'echo "committed locally" > feature.txt && git add feature.txt && git commit -q -m "local commit"',
      'printf "line one\\nuncommitted edit\\n" > README.md && echo "untracked file" > untracked.txt',
      `mkdir -p "$(dirname ${CLI_LOG})"`,
      `echo '{"type":"user","message":{"content":"build it"}}' > ${CLI_LOG}`,
    ].join('\n')
  );
  return sh(sandbox, `sha256sum ${CLI_LOG} | cut -d' ' -f1`);
}

async function workState(sandbox: Sandbox) {
  return {
    branch: await sh(sandbox, 'git -C ~/workspace rev-parse --abbrev-ref HEAD'),
    log: await sh(sandbox, 'git -C ~/workspace log --format=%s'),
    readme: await sh(sandbox, 'cat ~/workspace/README.md'),
    untracked: await sh(sandbox, 'cat ~/workspace/untracked.txt'),
    cliSha: await sh(sandbox, `sha256sum ${CLI_LOG} | cut -d' ' -f1`),
  };
}

async function sessionFor(held: Sandbox) {
  return prisma.codeSession.create({
    data: {
      userId,
      rootAgentId: agentId,
      ownerAgentId: agentId,
      repoKey: `resize-live-${randomUUID()}`,
      branch: 'rose/feature-1',
      mergeTarget: 'main',
      sandboxKind: 'e2b',
      sandboxId: held.sandboxId,
      sandboxCpu: SMALL.cpu,
      sandboxMemoryGb: SMALL.memoryGb,
      cliSessionId,
    },
  });
}

before(async () => {
  if (!live) return;
  process.env.CODE_SESSION_STORAGE_BACKEND = 'local';
  process.env.CODE_SESSION_STORAGE_LOCAL_DIR = mkdtempSync(join(tmpdir(), 'resize-live-'));
  await prisma.user.create({ data: { userId, name: 'Resize Live' } });
});

after(async () => {
  await Promise.all(created.map((sandbox) => sandbox.kill().catch(() => undefined)));
  if (live) {
    await prisma.codeSession.deleteMany({ where: { userId } });
    await prisma.user.deleteMany({ where: { userId } });
  }
  process.env.CODE_SESSION_STORAGE_BACKEND = saved.backend;
  process.env.CODE_SESSION_STORAGE_LOCAL_DIR = saved.dir;
  await prisma.$disconnect();
});

test(
  'live: 2x4 → 8x8 keeps branch, commits, edits and the CLI log; times every phase',
  { skip },
  async () => {
    const held = await create(SMALL);
    const cliSha = await seed(held);
    const before = await workState(held);
    const session = await sessionFor(held);
    const tx = newResizeTransaction(`live-${randomUUID()}`);
    const started = Date.now();
    const moved = await runInResizeTransaction(tx, () =>
      moveToResizedSandbox(session, held, () => create(LARGE), { size: LARGE, agentId })
    );
    const elapsedMs = Date.now() - started;
    const target = moved.sandbox;
    assert.equal(await sh(target, 'nproc'), '8');
    const memKb = Number((await sh(target, "awk '/MemTotal/ {print $2}' /proc/meminfo")) || 0);
    assert.ok(memKb > 7 * 1024 * 1024 && memKb <= 8 * 1024 * 1024, `MemTotal ${memKb} kB`);
    const after = await workState(target);
    assert.deepEqual(after, before, 'branch, commits, uncommitted + untracked files and CLI log');
    assert.equal(after.cliSha, cliSha);
    assert.equal(moved.cliSessionId, cliSessionId);
    const row = await prisma.codeSession.findUniqueOrThrow({ where: { id: session.id } });
    assert.equal(row.sandboxId, target.sandboxId, 'the row names the target after cut-over');
    assert.deepEqual([row.sandboxCpu, row.sandboxMemoryGb], [8, 8]);
    assert.equal(tx.retained?.sandboxId, held.sandboxId, 'the original is kept for rollback');
    const info = await Sandbox.getInfo(held.sandboxId, { apiKey: process.env.E2B_API_KEY });
    assert.equal(info.state, 'paused', 'the original is paused, not killed');
    console.log(
      `[resize-live] 2x4→8x8 moveToResizedSandbox ${elapsedMs} ms; phases ${JSON.stringify(tx.phases)}; ` +
        `restoreAttempts ${tx.restoreAttempts}; held df ${JSON.stringify(tx.disk.held)}; target df ${JSON.stringify(tx.disk.target)}; ` +
        `checksum ${createHash('sha256').update(JSON.stringify(after)).digest('hex').slice(0, 12)}`
    );
  }
);

test(
  'live: a failed target create rolls back to the held sandbox; a full held disk refuses first',
  { skip },
  async () => {
    const held = await create(SMALL);
    await seed(held);
    const before = await workState(held);
    const session = await sessionFor(held);

    const createFails = await runInResizeTransaction(newResizeTransaction('live-fail'), () =>
      moveToResizedSandbox(
        session,
        held,
        async () => {
          throw new Error('injected: E2B create refused');
        },
        { size: LARGE, agentId }
      )
    ).catch((error: unknown) => error);
    assert.equal(resizeFailureCode(createFails), 'TARGET_CREATE_FAILED', String(createFails));
    let row = await prisma.codeSession.findUniqueOrThrow({ where: { id: session.id } });
    assert.equal(row.sandboxId, held.sandboxId, 'the row still names the original');
    const resumed = await Sandbox.connect(held.sandboxId, { apiKey: process.env.E2B_API_KEY });
    assert.deepEqual(await workState(resumed), before, 'the original keeps every change');

    // Leave < 512 MB free on HOME's filesystem: the move must stop before checkpointing.
    await sh(
      resumed,
      "avail=$(df -Pk ~ | tail -1 | awk '{print $4}'); fallocate -l $(( (avail - 200*1024) * 1024 )) ~/fill.bin && df -Pk ~ | tail -1"
    );
    let targets = 0;
    const diskFull = await runInResizeTransaction(newResizeTransaction('live-disk'), () =>
      moveToResizedSandbox(
        session,
        resumed,
        async () => {
          targets += 1;
          return create(LARGE);
        },
        { size: LARGE, agentId }
      )
    ).catch((error: unknown) => error);
    assert.equal(resizeFailureCode(diskFull), 'HELD_DISK_FULL', String(diskFull));
    assert.equal(targets, 0, 'no target sandbox is created on a full held disk');
    row = await prisma.codeSession.findUniqueOrThrow({ where: { id: session.id } });
    assert.equal(row.sandboxId, held.sandboxId);
    const again = await Sandbox.connect(held.sandboxId, { apiKey: process.env.E2B_API_KEY });
    await sh(again, 'rm -f ~/fill.bin');
    assert.deepEqual(await workState(again), before, 'work intact after the refused move');
  }
);
