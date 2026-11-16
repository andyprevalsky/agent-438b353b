#!/usr/bin/env tsx
/**
 * Black-box tests for the project template sync: a project's sync script runs
 * on ONE template sandbox, a successful run is snapshotted, and new code
 * sessions start from the latest snapshot (clone + startup when there is none).
 *
 * Real bash/git run inside a fake E2B (project-maintenance-fake-e2b.ts); the
 * production runner, store, service and session-start code run unchanged.
 *
 * Invariants covered:
 *   - run succeeds → snapshot published, credentials scrubbed from it, log redacted
 *   - the next run starts from that snapshot (incremental), the previous one is dropped later
 *   - run fails / times out → nothing published, sandbox killed, lease released,
 *     and the following run starts clean from the base template
 *   - concurrent triggers dedupe: one lease, one sandbox; slot keys dedupe scheduled/GitHub runs
 *   - an expired lease is reconciled and reclaimable; publication refuses a lost lease
 *   - cron tick: nightly once per day, GitHub head change once per sha, process cap
 *   - Snapshot now fast-forwards and snapshots without running the script
 *   - a new session sandbox is created from the snapshot (fallback: base template),
 *     and the restored checkout is fast-forwarded to the branch head
 *   - settings round-trip, validation and owner scoping over HTTP
 *
 * Usage: bash scripts/run-blackbox-tests.sh project-maintenance
 */
// First import: refuses a non-local DATABASE_URL before any module can query.
import { assert } from './projects-harness';
import { execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { after, afterEach, test } from 'node:test';
import { Sandbox } from 'e2b';
import {
  runMaintenanceTick,
  startMaintenance,
  type MaintenanceServiceDeps,
  type StartResult,
} from '../../domains/projects/maintenanceService';
import { prisma } from '../../providers/database';
import { acquireCodeSandbox } from '../../providers/e2bCodeSessionSandbox';
import { newSessionDefaultSize } from '../../providers/codeSessions/sandboxSize';
import * as store from '../../providers/projects/maintenanceStore';
import { githubSlotKey, nightlySlotKey } from '../../providers/projects/maintenanceSchedule';
import type { ProjectWorkspace } from '../../providers/projects/types';
import {
  registerWorkspaceExecutor,
  unregisterWorkspaceExecutor,
} from '../../providers/projects/workspaceExecutor';
import { e2bWorkspaceExecutor, restoreWorkspace } from '../../providers/projects/workspace';
import { FakeE2b, tempRoot } from './project-maintenance-fake-e2b';
import { registerMaintenanceProviderCases } from './project-maintenance-provider.cases';
import { registerMaintenanceLeaseCases } from './project-maintenance-lease.cases';
import { registerMaintenanceReviewCases } from './project-maintenance-review.cases';

const SECRET = `sync-fixture-secret-${randomBytes(8).toString('hex')}`;
const root = tempRoot('project-maintenance-bb-');
const userIds: string[] = [];

// Completed fixtures must not remain due during another case's cron tick.
afterEach(async () => {
  await prisma.project.updateMany({
    where: { userId: { in: userIds } },
    data: { maintenanceNightly: false, maintenanceOnGithubChange: false },
  });
});

after(async () => {
  await prisma.user.deleteMany({ where: { userId: { in: userIds } } });
  await prisma.$disconnect();
});

/** The sync script: installs only when package.json changed, prints the env secret. */
export const SYNC_SCRIPT = `set -euo pipefail
cd "$ROSE_PROJECT_DIR"
git fetch --quiet origin "$ROSE_DEFAULT_BRANCH" && git checkout --quiet "$ROSE_DEFAULT_BRANCH" && git merge --ff-only --quiet "origin/$ROSE_DEFAULT_BRANCH"
STAMP="$HOME/.cache/rose-maintenance"; mkdir -p "$STAMP"
hash=$(sha256sum package.json | cut -d' ' -f1)
if [ "$hash" != "$(cat "$STAMP/js" 2>/dev/null || true)" ]; then
  echo install >> "$HOME/installs.log"; mkdir -p node_modules; echo "$hash" > "$STAMP/js"
fi
. ./.env; echo "synced with token $FIXTURE_SECRET"
`;

export function git(cwd: string, ...args: string[]): string {
  return execFileSync('git', ['-c', 'user.name=t', '-c', 'user.email=t@example.com', ...args], {
    cwd,
    encoding: 'utf8',
  }).trim();
}

interface Fixture {
  userId: string;
  projectId: string;
  remote: string;
  work: string;
  fake: FakeE2b;
  deps: MaintenanceServiceDeps;
  workspace: () => ProjectWorkspace;
  pushCommit: (file: string, content: string) => string;
}

export async function fixture(
  name: string,
  overrides: { script?: string | null; timeoutS?: number; repo?: boolean } = {}
): Promise<Fixture> {
  const dir = join(root, name);
  const remote = join(dir, 'remote.git');
  const work = join(dir, 'work');
  mkdirSync(dir, { recursive: true });
  git(dir, 'init', '-q', '--bare', '-b', 'main', remote);
  git(dir, 'clone', '-q', remote, work);
  writeFileSync(join(work, 'package.json'), '{"name":"fixture","version":"1"}\n');
  git(work, 'add', '-A');
  git(work, 'commit', '-qm', 'init');
  git(work, 'push', '-q', 'origin', 'HEAD:main');
  const userId = `maint-${randomBytes(6).toString('hex')}`;
  userIds.push(userId);
  await prisma.user.create({ data: { userId, name: 'project maintenance blackbox' } });
  const project = await prisma.project.create({
    data: {
      userId,
      name: 'PJ-Maint',
      syncSalt: 'salt',
      repoUrl: overrides.repo === false ? null : remote,
      maintenanceScript: overrides.script === undefined ? SYNC_SCRIPT : overrides.script,
      maintenanceTimeoutS: overrides.timeoutS ?? 60,
    },
  });
  const fake = new FakeE2b(join(dir, 'fake'));
  const workspace = (): ProjectWorkspace => ({
    project: {
      id: project.id,
      name: 'PJ-Maint',
      repoUrl: remote,
      defaultBranch: 'main',
      startupScript: null,
      startupTimeoutS: 60,
    },
    revision: 0,
    manifest: null,
    files: [
      {
        path: '.env',
        kind: 'env',
        content: Buffer.from(`FIXTURE_SECRET=${SECRET}\n`),
        linkTarget: null,
      },
    ],
    gitBundle: null,
    uncommittedPatch: null,
    redactions: [SECRET],
    // The real session resolves a token-bearing URL here; a distinct spelling
    // of the remote proves the scrub restores the plain one.
    cloneUrl: remote,
  });
  const deps: MaintenanceServiceDeps = {
    runner: () => fake.deps(),
    prepareWorkspace: async () => workspace(),
    headSha: async () => null,
    now: () => new Date(),
    maxConcurrent: () => 4,
  };
  const pushCommit = (file: string, content: string) => {
    writeFileSync(join(work, file), content);
    git(work, 'add', '-A');
    git(work, 'commit', '-qm', `change ${file}`);
    git(work, 'push', '-q', 'origin', 'HEAD:main');
    return git(work, 'rev-parse', 'HEAD');
  };
  return { userId, projectId: project.id, remote, work, fake, deps, workspace, pushCommit };
}

export async function runOnce(
  f: Fixture,
  mode: 'sync' | 'snapshot' = 'sync'
): Promise<StartResult> {
  const result = await startMaintenance(f.projectId, { trigger: 'manual', mode }, f.deps);
  if (result.kind === 'started') await result.done;
  return result;
}

export const projectRow = (id: string) => prisma.project.findUniqueOrThrow({ where: { id } });
export const lastRun = (projectId: string) =>
  prisma.projectMaintenanceRun.findFirstOrThrow({
    where: { projectId },
    orderBy: { startedAt: 'desc' },
  });
const installs = (f: Fixture) => {
  const path = join(f.fake.home, 'installs.log');
  return existsSync(path) ? readFileSync(path, 'utf8').trim().split('\n').length : 0;
};

test('a successful run publishes a snapshot with credentials scrubbed and the log redacted', async () => {
  const f = await fixture('success');
  const result = await runOnce(f);
  assert(result.kind === 'started', `run started: ${result.kind}`);
  const run = await lastRun(f.projectId);
  assert(run.status === 'SUCCEEDED', `status ${run.status}: ${run.error}`);
  assert(run.snapshotId === 'snap-1' && run.fromSnapshotId === null, 'first run: base template');
  assert(f.fake.created[0] === 'base-template', `created ${f.fake.created}`);
  assert(/^[0-9a-f]{40}$/.test(run.headSha ?? ''), 'run records the head sha');
  const row = await projectRow(f.projectId);
  assert(row.templateSnapshotId === 'snap-1', 'snapshot published');
  assert(row.templateSnapshotSha === run.headSha, 'snapshot records its sha');
  assert(row.maintenanceRunId === null && row.maintenanceLeaseUntil === null, 'lease released');
  assert(f.fake.killed.length === 1, 'template sandbox killed');
  const log = run.logTail ?? '';
  assert(log.includes('synced with token [REDACTED]'), `log redacted: ${log.slice(-300)}`);
  assert(!log.includes(SECRET), 'secret absent from the stored log');
  // The snapshot every session starts from holds no credential.
  const snapshotDir = join(f.fake.home, '..', 'snapshots', 'snap-1');
  const grep = (needle: string) => spawnGrep(snapshotDir, needle);
  assert(!grep(SECRET), 'env secret absent from the snapshot');
  assert(!existsSync(join(snapshotDir, 'workspace', '.env')), 'env file removed from the snapshot');
  assert(
    !existsSync(join(snapshotDir, '.rose', 'project')),
    'session dir removed from the snapshot'
  );
  const origin = git(join(snapshotDir, 'workspace'), 'config', '--get', 'remote.origin.url');
  assert(origin === f.remote, `origin restored to the plain repo URL, got ${origin}`);
  assert(installs(f) === 1, 'first run installed');
});

function spawnGrep(dir: string, needle: string): boolean {
  try {
    execFileSync('grep', ['-rIl', '--', needle, dir], { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

test('the next run starts from the snapshot, is incremental, and drops the superseded snapshot', async () => {
  const f = await fixture('incremental');
  await runOnce(f);
  const second = await runOnce(f);
  assert(second.kind === 'started', 'second run started');
  const run2 = await lastRun(f.projectId);
  assert(run2.fromSnapshotId === 'snap-1', `run 2 starts from snap-1, got ${run2.fromSnapshotId}`);
  assert(f.fake.created[1] === 'snap-1', `second sandbox from the snapshot: ${f.fake.created}`);
  assert(installs(f) === 1, 'unchanged package.json: no second install');
  let row = await projectRow(f.projectId);
  assert(
    row.templateSnapshotId === 'snap-2' && row.templatePrevSnapshotId === 'snap-1',
    'pointer moved'
  );
  assert(f.fake.deleted.length === 0, 'the previous snapshot is kept until the next publish');
  f.pushCommit('package.json', '{"name":"fixture","version":"2"}\n');
  await runOnce(f);
  assert(installs(f) === 2, 'changed package.json: reinstalled');
  row = await projectRow(f.projectId);
  assert(
    row.templateSnapshotId === 'snap-3' && row.templatePrevSnapshotId === 'snap-2',
    'pointer moved again'
  );
  assert(
    JSON.stringify(f.fake.deleted) === JSON.stringify(['snap-1']),
    `dropped ${f.fake.deleted}`
  );
  const head = git(f.work, 'rev-parse', 'HEAD');
  assert(row.templateSnapshotSha === head, 'the snapshot is at the new head');
});

test('a failed run publishes nothing and the next run starts clean', async () => {
  const f = await fixture('failure');
  await runOnce(f);
  assert((await projectRow(f.projectId)).templateSnapshotId === 'snap-1', 'baseline snapshot');
  await prisma.project.update({
    where: { id: f.projectId },
    data: { maintenanceScript: 'echo about to fail\nexit 7\n' },
  });
  const before = f.fake.snapshots.length;
  await runOnce(f);
  const run = await lastRun(f.projectId);
  assert(
    run.status === 'FAILED' && run.exitCode === 7,
    `failed run: ${run.status} ${run.exitCode}`
  );
  assert(run.snapshotId === null, 'failed run records no snapshot');
  assert(f.fake.snapshots.length === before, 'no snapshot was even taken');
  const row = await projectRow(f.projectId);
  assert(row.templateSnapshotId === 'snap-1', 'the previous snapshot stays published');
  assert(row.maintenanceRunId === null, 'lease released after a failure');
  assert(f.fake.killed.length === 2, 'both sandboxes killed');
  assert((run.logTail ?? '').includes('about to fail'), 'failure output is in the log');
  await prisma.project.update({
    where: { id: f.projectId },
    data: { maintenanceScript: SYNC_SCRIPT },
  });
  await runOnce(f);
  const retry = await lastRun(f.projectId);
  assert(
    retry.status === 'SUCCEEDED' && retry.fromSnapshotId === null,
    'retry after a failure starts clean'
  );
  assert(f.fake.created.at(-1) === 'base-template', `retry sandbox: ${f.fake.created}`);
});

test('a run that exceeds its timeout is TIMED_OUT and publishes nothing', async () => {
  const f = await fixture('timeout', { script: 'sleep 30\n', timeoutS: 1 });
  await runOnce(f);
  const run = await lastRun(f.projectId);
  assert(run.status === 'TIMED_OUT', `status ${run.status}`);
  assert((await projectRow(f.projectId)).templateSnapshotId === null, 'nothing published');
  assert(
    f.fake.snapshots.length === 0 && f.fake.killed.length === 1,
    'no snapshot, sandbox killed'
  );
});

test('publication is refused when the run no longer holds the lease', async () => {
  const f = await fixture('lost-lease');
  const claim = await store.claimRun({
    projectId: f.projectId,
    trigger: 'manual',
    mode: 'sync',
    fromSnapshotId: null,
    leaseMs: 60_000,
    now: new Date(),
  });
  assert(claim.kind === 'claimed', 'claimed');
  // Another owner took the lease over (this one expired and was reclaimed).
  await prisma.project.update({
    where: { id: f.projectId },
    data: { maintenanceRunId: 'someone-else' },
  });
  const refused = await store.publishSnapshot({
    projectId: f.projectId,
    runId: claim.run.id,
    snapshotId: 'snap-x',
    sha: null,
    size: null,
    now: new Date(),
  });
  assert(!refused.published, 'publication refused without the lease');
  assert((await projectRow(f.projectId)).templateSnapshotId === null, 'pointer untouched');
});

test('concurrent triggers dedupe: one lease, one sandbox; slot keys dedupe scheduled runs', async () => {
  const f = await fixture('dedupe');
  f.fake.createDelayMs = 300;
  const results = await Promise.all([
    startMaintenance(f.projectId, { trigger: 'manual', mode: 'sync' }, f.deps),
    startMaintenance(f.projectId, { trigger: 'manual', mode: 'sync' }, f.deps),
    startMaintenance(f.projectId, { trigger: 'manual', mode: 'sync' }, f.deps),
  ]);
  const started = results.filter((r) => r.kind === 'started');
  assert(started.length === 1, `exactly one started: ${results.map((r) => r.kind)}`);
  assert(results.filter((r) => r.kind === 'busy').length === 2, 'the others are busy');
  for (const r of started) if (r.kind === 'started') await r.done;
  const sandboxesCreated = () => f.fake.created.length;
  assert(sandboxesCreated() === 1, `one sandbox for three triggers: ${f.fake.created}`);
  const rows = await prisma.projectMaintenanceRun.count({ where: { projectId: f.projectId } });
  assert(rows === 1, `rejected triggers leave no run rows, got ${rows}`);

  const slot = nightlySlotKey(new Date());
  const first = await startMaintenance(
    f.projectId,
    { trigger: 'nightly', mode: 'sync', slotKey: slot },
    f.deps
  );
  assert(first.kind === 'started', 'first nightly trigger starts');
  if (first.kind === 'started') await first.done;
  const again = await startMaintenance(
    f.projectId,
    { trigger: 'nightly', mode: 'sync', slotKey: slot },
    f.deps
  );
  assert(again.kind === 'duplicate', `same slot again: ${again.kind}`);
  const sha = 'a'.repeat(40);
  const g1 = await startMaintenance(
    f.projectId,
    { trigger: 'github', mode: 'sync', slotKey: githubSlotKey(sha), watchedSha: sha },
    f.deps
  );
  if (g1.kind === 'started') await g1.done;
  const g2 = await startMaintenance(
    f.projectId,
    { trigger: 'github', mode: 'sync', slotKey: githubSlotKey(sha), watchedSha: sha },
    f.deps
  );
  assert(g1.kind === 'started' && g2.kind === 'duplicate', `github slot: ${g1.kind}/${g2.kind}`);
  assert(sandboxesCreated() === 3, `one sandbox per admitted run: ${f.fake.created}`);
});

test('an abandoned run is reconciled after its lease expires and the project is reclaimable', async () => {
  const f = await fixture('abandoned');
  const claim = await store.claimRun({
    projectId: f.projectId,
    trigger: 'manual',
    mode: 'sync',
    fromSnapshotId: null,
    leaseMs: 60_000,
    now: new Date(Date.now() - 2 * 60_000),
  });
  assert(claim.kind === 'claimed', 'claimed in the past');
  const busy = await store.claimRun({
    projectId: f.projectId,
    trigger: 'manual',
    mode: 'sync',
    fromSnapshotId: null,
    leaseMs: 60_000,
    now: new Date(Date.now() - 90_000),
  });
  assert(busy.kind === 'busy', 'a live lease blocks a second claim');
  assert((await store.reconcileStaleRuns(new Date())) === 1, 'one stale run reconciled');
  const run = await prisma.projectMaintenanceRun.findUniqueOrThrow({ where: { id: claim.run.id } });
  assert(
    run.status === 'FAILED' && (run.error ?? '').startsWith('abandoned'),
    `run: ${run.status} ${run.error}`
  );
  assert((await projectRow(f.projectId)).maintenanceRunId === null, 'lease cleared');
  const next = await runOnce(f);
  assert(next.kind === 'started', 'the project can run again');
});

test('cron tick: nightly once per day, GitHub change once per sha, process cap respected', async () => {
  const a = await fixture('tick-a');
  const b = await fixture('tick-b');
  await prisma.project.updateMany({
    where: { id: { in: [a.projectId, b.projectId] } },
    data: { maintenanceNightly: false },
  });
  await prisma.project.update({ where: { id: a.projectId }, data: { maintenanceNightly: true } });
  let head = 'b'.repeat(40);
  const deps: MaintenanceServiceDeps = { ...a.deps, headSha: async () => head };
  const first = await runMaintenanceTick(deps);
  await Promise.all(first.done);
  assert(first.nightly === 1, `nightly runs: ${first.nightly}`);
  const nightlyRuns = await prisma.projectMaintenanceRun.findMany({
    where: { projectId: a.projectId },
  });
  assert(
    nightlyRuns.length === 1 && nightlyRuns[0].slotKey === nightlySlotKey(new Date()),
    'slot recorded'
  );
  const second = await runMaintenanceTick(deps);
  await Promise.all(second.done);
  assert(second.nightly === 0, 'the same day does not run again');

  // GitHub watch: the first observation seeds the sha, only a change builds.
  await prisma.project.update({
    where: { id: b.projectId },
    data: { maintenanceOnGithubChange: true },
  });
  const poll = async () => {
    await prisma.project.update({
      where: { id: b.projectId },
      data: { maintenanceWatchCheckedAt: null },
    });
    const tick = await runMaintenanceTick({ ...b.deps, headSha: async () => head });
    await Promise.all(tick.done);
    return tick;
  };
  assert((await poll()).github === 0, 'first observation only seeds');
  assert((await projectRow(b.projectId)).maintenanceWatchedSha === head, 'sha remembered');
  assert((await poll()).github === 0, 'unchanged head: no run');
  head = 'c'.repeat(40);
  assert((await poll()).github === 1, 'changed head: one run');
  assert((await projectRow(b.projectId)).maintenanceWatchedSha === head, 'new sha remembered');
  assert((await poll()).github === 0, 'the same new head does not run twice');
  const ghRuns = await prisma.projectMaintenanceRun.count({
    where: { projectId: b.projectId, trigger: 'github' },
  });
  assert(ghRuns === 1, `one github run, got ${ghRuns}`);

  // Cap: with room for none, nothing starts and nothing is claimed.
  const capped = await fixture('tick-cap');
  await prisma.project.update({
    where: { id: capped.projectId },
    data: { maintenanceNightly: true },
  });
  const none = await runMaintenanceTick({ ...capped.deps, maxConcurrent: () => 0 });
  assert(none.nightly === 0, 'cap of 0 starts nothing');
  assert(
    (await prisma.projectMaintenanceRun.count({ where: { projectId: capped.projectId } })) === 0,
    'no run row was claimed under the cap'
  );
});

test('Snapshot now fast-forwards and snapshots without running the sync script', async () => {
  const f = await fixture('snapshot-now');
  await runOnce(f);
  const head = f.pushCommit('NEWS.md', 'new\n');
  const marker = join(f.fake.home, 'script-ran.txt');
  await prisma.project.update({
    where: { id: f.projectId },
    data: { maintenanceScript: `touch ${marker}\n` },
  });
  await runOnce(f, 'snapshot');
  const run = await lastRun(f.projectId);
  assert(run.status === 'SUCCEEDED' && run.mode === 'snapshot', `run ${run.status} ${run.error}`);
  assert(
    (await projectRow(f.projectId)).templateSnapshotSha === head,
    'snapshot is at the new head'
  );
  assert(!existsSync(marker), 'the sync script did not run');
  assert(installs(f) === 1, 'no install');
});

test('a snapshot of an outdated size is rebuilt from the sized base template', async () => {
  const f = await fixture('outdated-size');
  await runOnce(f);
  assert(
    JSON.stringify((await projectRow(f.projectId)).templateSnapshotSize) ===
      JSON.stringify({ cpu: 2, memoryGb: 4 }),
    'first snapshot at the base size'
  );
  // The default (or the project's size) was raised since the snapshot was built.
  const runner = f.deps.runner;
  f.deps.runner = () => ({
    ...runner(),
    baseTemplate: () => ({ templateId: 'base-c8-m8', size: { cpu: 8, memoryGb: 8 } }),
  });
  await runOnce(f);
  const run = await lastRun(f.projectId);
  assert(run.status === 'SUCCEEDED', `run ${run.status} ${run.error}`);
  assert(f.fake.created[1] === 'base-c8-m8', `rebuilt from the sized base: ${f.fake.created}`);
  assert(
    JSON.stringify((await projectRow(f.projectId)).templateSnapshotSize) ===
      JSON.stringify({ cpu: 8, memoryGb: 8 }),
    'the new snapshot records the new size'
  );
  await runOnce(f);
  assert(f.fake.created[2] === 'snap-2', `a matching snapshot is reused: ${f.fake.created}`);
});

test('a run without a repository or a sync script is refused before any sandbox exists', async () => {
  const noRepo = await fixture('no-repo', { repo: false });
  assert((await runOnce(noRepo)).kind === 'ineligible', 'no repository');
  const noScript = await fixture('no-script', { script: '' });
  assert((await runOnce(noScript)).kind === 'ineligible', 'cleared script');
  assert(noRepo.fake.created.length === 0 && noScript.fake.created.length === 0, 'no sandbox');
  assert((await runOnce(noScript, 'snapshot')).kind === 'started', 'Snapshot now needs no script');
});

test('a new session sandbox is created from the snapshot, with the base template as fallback', async () => {
  const created: string[] = [];
  const originalCreate = Sandbox.create;
  const failing = new Set<string>();
  Sandbox.create = (async (template: string) => {
    created.push(template);
    if (failing.has(template)) throw new Error(`template ${template} not found`);
    return { sandboxId: `fake-${created.length}` } as unknown as Sandbox;
  }) as unknown as typeof Sandbox.create;
  try {
    // A snapshot at the size a new session resolves to (one of another size is skipped:
    // code-session-sizing.blackbox.ts).
    const snapshot = { id: 'snap-7', size: newSessionDefaultSize('claude') };
    const viaSnapshot = await acquireCodeSandbox(null, 'base-tpl', {}, 'Test', {
      provider: 'claude',
      snapshot,
    });
    assert(viaSnapshot.fromSnapshot && created[0] === 'snap-7', `snapshot used: ${created}`);
    created.length = 0;
    failing.add('snap-7');
    const fallback = await acquireCodeSandbox(null, 'base-tpl', {}, 'Test', {
      provider: 'claude',
      snapshot,
    });
    assert(!fallback.fromSnapshot, 'deleted snapshot: not from the snapshot');
    assert(
      created[0] === 'snap-7' && created.length === 2 && created[1] !== 'snap-7',
      `fallback: ${created}`
    );
    created.length = 0;
    const none = await acquireCodeSandbox(null, 'base-tpl', {}, 'Test', { provider: 'claude' });
    assert(
      !none.fromSnapshot && created.length === 1 && created[0] !== 'snap-7',
      `no snapshot: ${created}`
    );
  } finally {
    Sandbox.create = originalCreate;
  }
});

test('a session restored from the snapshot gets the branch head, fresh env files and no leftovers', async () => {
  const f = await fixture('session-start');
  await runOnce(f);
  const head = f.pushCommit('LATER.md', 'after the snapshot\n');
  // "Start a session": a sandbox created from the published snapshot.
  const sandbox = await f.fake.createSandbox('snap-1');
  const sessionId = `session-${randomBytes(4).toString('hex')}`;
  registerWorkspaceExecutor(sessionId, e2bWorkspaceExecutor(sandbox, f.fake.home));
  try {
    const report = await restoreWorkspace(sessionId, {
      ...f.workspace(),
      startedFromSnapshot: true,
    });
    assert(report.checkout.source === 'snapshot', `source ${report.checkout.source}`);
    assert(
      report.checkout.headSha === head,
      `fast-forwarded to ${head}, got ${report.checkout.headSha}`
    );
    assert(report.warnings.length === 0, `no warnings: ${report.warnings}`);
    const repo = join(f.fake.home, 'workspace');
    assert(
      readFileSync(join(repo, '.env'), 'utf8').includes(SECRET),
      'env file rewritten for the session'
    );
    assert(existsSync(join(repo, 'node_modules')), 'dependencies came with the snapshot');
    assert(
      git(repo, 'config', '--get', 'remote.origin.url') === f.remote,
      'origin stays token-free'
    );
    // Without a snapshot the same restore is the clone path.
    const fresh = await fixture('session-fresh');
    const plain = await fresh.fake.createSandbox('base-template');
    const freshId = `session-${randomBytes(4).toString('hex')}`;
    registerWorkspaceExecutor(freshId, e2bWorkspaceExecutor(plain, fresh.fake.home));
    try {
      const cloned = await restoreWorkspace(freshId, fresh.workspace());
      assert(
        cloned.checkout.source === 'remote' && cloned.checkout.cloned,
        'no snapshot: fresh clone'
      );
      assert(
        !existsSync(join(fresh.fake.home, 'workspace', 'node_modules')),
        'no deps without a snapshot'
      );
    } finally {
      unregisterWorkspaceExecutor(freshId);
    }
  } finally {
    unregisterWorkspaceExecutor(sessionId);
  }
});

registerMaintenanceReviewCases({ fixture, runOnce, lastRun, projectRow, git, SYNC_SCRIPT });

registerMaintenanceLeaseCases();

registerMaintenanceProviderCases({ fixture, runOnce, lastRun, git, SECRET });
