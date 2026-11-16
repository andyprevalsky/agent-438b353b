#!/usr/bin/env tsx
/**
 * Black-box contract for the root-facing code-agent sizing policy.
 *
 *   spawn_agent(type='code', sandbox_size={cpu?, memory_gb?})
 *     └─ POST /api/{claude,codex}-code-session/create {sandbox_size}
 *          shape (400 INVALID_REQUEST; target='local' refused) · cap / no built
 *          template (400 SANDBOX_SIZE_UNSUPPORTED, lists the available sizes)
 *          └─ acquireCodeSandbox(requestedSize) → template of the closest built size
 *   new-session size: request > project sandboxSize > default (8 vCPU / 8 GB when
 *   built and within the user's cap, else the base size; CODE_SANDBOX_DEFAULT_* wins)
 *   an existing session keeps the size it holds (source 'session')
 *   a project snapshot of another size is skipped at spawn and rebuilt by maintenance
 *
 * E2B is faked (Sandbox.create records the template); nothing reaches E2B.
 */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { after, before, test } from 'node:test';
import { Sandbox } from 'e2b';
import { config } from 'dotenv';
import { prisma } from '../../providers/database';
import {
  newSessionDefaultSize,
  newSessionSize,
  PREFERRED_DEFAULT_SANDBOX_SIZE,
  resolveRequestedSandboxSize,
  resolveSpawnSandbox,
  SandboxSizeUnsupportedError,
  sizedTemplateName,
} from '../../providers/codeSessions/sandboxSize';
import { codeSessionHttpError } from '../../providers/codeSessions/spawn';
import { acquireCodeSandbox } from '../../providers/e2bCodeSessionSandbox';
import { wantedTemplateSize } from '../../providers/projects/maintenanceE2b';

config();

const baseUrl = process.env.TEST_BASE_URL || 'http://localhost:3001';
const apiKey = process.env.BACKEND_API_KEY || '';
const userId = `sizing-user-${randomUUID()}`;
const cappedUserId = `sizing-capped-${randomUUID()}`;
const agentId = randomUUID();
const cappedAgentId = randomUUID();
const ENV_KEYS = [
  'E2B_SANDBOX_TEMPLATES',
  'CODE_SANDBOX_GRID',
  'CODE_SANDBOX_DEFAULT_CPU',
  'CODE_SANDBOX_DEFAULT_MEMORY_GB',
  'CODE_SANDBOX_MAX_CPU',
  'CODE_SANDBOX_MAX_MEMORY_GB',
] as const;
const savedEnv = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));

function setEnv(values: Partial<Record<(typeof ENV_KEYS)[number], string>>): void {
  for (const key of ENV_KEYS) delete process.env[key];
  Object.assign(process.env, values);
}

async function http(path: string, body: unknown) {
  const response = await fetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let data: Record<string, unknown> = {};
  try {
    data = JSON.parse(text) as Record<string, unknown>;
  } catch {
    data = { text };
  }
  return { status: response.status, data };
}

before(async () => {
  await prisma.user.create({ data: { userId, name: 'Sizing User' } });
  await prisma.user.create({
    data: {
      userId: cappedUserId,
      name: 'Capped Sizing User',
      codeSandboxMaxCpu: 2,
      codeSandboxMaxMemoryGb: 4,
    },
  });
  await prisma.agent.create({ data: { id: agentId, userId, goal: 'Size the sandbox' } });
  await prisma.agent.create({
    data: { id: cappedAgentId, userId: cappedUserId, goal: 'Size the sandbox' },
  });
});

after(async () => {
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  await prisma.codeSession.deleteMany({ where: { userId: { in: [userId, cappedUserId] } } });
  await prisma.agent.deleteMany({ where: { id: { in: [agentId, cappedAgentId] } } });
  await prisma.user.deleteMany({ where: { userId: { in: [userId, cappedUserId] } } });
  await prisma.$disconnect();
});

test('default: 8 vCPU / 8 GB when built, else the base size; an operator default and a cap win', () => {
  setEnv({ E2B_SANDBOX_TEMPLATES: 'grid' });
  assert.deepEqual(PREFERRED_DEFAULT_SANDBOX_SIZE, { cpu: 8, memoryGb: 8 });
  assert.deepEqual(newSessionDefaultSize('claude'), { cpu: 8, memoryGb: 8 });
  assert.deepEqual(newSessionDefaultSize('codex'), { cpu: 8, memoryGb: 8 });
  assert.deepEqual(
    newSessionDefaultSize('claude', { cpu: 2, memoryGb: 4 }),
    { cpu: 2, memoryGb: 4 },
    'a user capped below the default keeps the base size'
  );
  setEnv({
    E2B_SANDBOX_TEMPLATES: 'grid',
    CODE_SANDBOX_DEFAULT_CPU: '2',
    CODE_SANDBOX_DEFAULT_MEMORY_GB: '8',
  });
  assert.deepEqual(newSessionDefaultSize('claude'), { cpu: 2, memoryGb: 8 }, 'operator default');
  setEnv({});
  assert.deepEqual(
    newSessionDefaultSize('claude'),
    { cpu: 2, memoryGb: 4 },
    'no sized templates: never a size without a built template'
  );
});

test('overrides: request > project sandboxSize > default; a partial project size fills from the default', () => {
  setEnv({ E2B_SANDBOX_TEMPLATES: 'grid' });
  assert.deepEqual(newSessionSize('claude', {}), {
    size: { cpu: 8, memoryGb: 8 },
    source: 'default',
  });
  assert.deepEqual(newSessionSize('claude', { projectSize: { cpu: 2, memoryGb: 4 } }), {
    size: { cpu: 2, memoryGb: 4 },
    source: 'project',
  });
  assert.deepEqual(newSessionSize('claude', { projectSize: { memoryGb: 4 } }), {
    size: { cpu: 8, memoryGb: 4 },
    source: 'project',
  });
  assert.deepEqual(
    newSessionSize('claude', {
      requested: { cpu: 2, memoryGb: 8 },
      projectSize: { cpu: 8, memoryGb: 8 },
    }),
    { size: { cpu: 2, memoryGb: 8 }, source: 'request' }
  );
});

test('an existing session keeps its held size even when the spawn asks for another', async () => {
  setEnv({ E2B_SANDBOX_TEMPLATES: 'grid' });
  const held = await prisma.codeSession.create({
    data: {
      userId,
      rootAgentId: agentId,
      ownerAgentId: agentId,
      repoKey: `sizing-${randomUUID()}`,
      sandboxKind: 'e2b',
      sandboxId: `sb-${randomUUID()}`,
      sandboxCpu: 2,
      sandboxMemoryGb: 4,
    },
  });
  const resolved = await resolveSpawnSandbox(held, 'claude', {
    requested: { cpu: 8, memoryGb: 8 },
  });
  assert.deepEqual(resolved.size, { cpu: 2, memoryGb: 4 });
  assert.equal(resolved.source, 'session');
  const fresh = await prisma.codeSession.create({
    data: {
      userId,
      rootAgentId: agentId,
      ownerAgentId: agentId,
      repoKey: `sizing-${randomUUID()}`,
    },
  });
  const requested = await resolveSpawnSandbox(fresh, 'codex', {
    requested: { cpu: 8, memoryGb: 6 },
  });
  assert.deepEqual(requested, {
    size: { cpu: 8, memoryGb: 8 },
    templateId: sizedTemplateName('codex', { cpu: 8, memoryGb: 8 }),
    wanted: { cpu: 8, memoryGb: 6 },
    source: 'request',
  });
});

test('validation: an unbuilt or over-cap request is refused with the available sizes', async () => {
  setEnv({ E2B_SANDBOX_TEMPLATES: 'grid' });
  assert.deepEqual(await resolveRequestedSandboxSize('claude', { memoryGb: 4 }, userId), {
    cpu: 8,
    memoryGb: 4,
  });
  const tooBig = await resolveRequestedSandboxSize('claude', { cpu: 16 }, userId).catch(
    (error: unknown) => error
  );
  assert.ok(tooBig instanceof SandboxSizeUnsupportedError, String(tooBig));
  assert.equal(tooBig.available.length, 4);
  const mapped = codeSessionHttpError(tooBig);
  assert.equal(mapped?.status, 400);
  assert.equal(mapped?.body.code, 'SANDBOX_SIZE_UNSUPPORTED');
  assert.deepEqual(mapped?.body.requested, { cpu: 16, memory_gb: 8 });
  for (const offGrid of [{ cpu: 4 }, { memoryGb: 6 }, { cpu: 2, memoryGb: 2 }]) {
    const refused = await resolveRequestedSandboxSize('claude', offGrid, userId).catch(
      (error: unknown) => error
    );
    assert.ok(refused instanceof SandboxSizeUnsupportedError, `${JSON.stringify(offGrid)} refused`);
    assert.match(refused.message, /supported vCPUs 2 or 8, memory 4 or 8 GB/);
  }
  for (const onGrid of [
    { cpu: 2, memoryGb: 4 },
    { cpu: 2, memoryGb: 8 },
    { cpu: 8, memoryGb: 4 },
    { cpu: 8, memoryGb: 8 },
  ]) {
    assert.deepEqual(await resolveRequestedSandboxSize('codex', onGrid, userId), onGrid);
  }
  const capped = await resolveRequestedSandboxSize('claude', { cpu: 8 }, cappedUserId).catch(
    (error: unknown) => error
  );
  assert.ok(capped instanceof SandboxSizeUnsupportedError, String(capped));
  assert.match(capped.message, /over this account's limit of 2 vCPU, 4 GB RAM/);
  assert.deepEqual(capped.available, [{ cpu: 2, memoryGb: 4 }]);
});

test('HTTP: create refuses a malformed, local, over-cap or unbuilt sandbox_size with 400', async () => {
  const local = await http('/api/claude-code-session/create', {
    agent_id: agentId,
    goal: 'x',
    target: 'local',
    sandbox_size: { cpu: 8 },
  });
  assert.equal(local.status, 400, JSON.stringify(local.data));
  assert.equal(local.data.code, 'INVALID_REQUEST');
  assert.match(String(local.data.error), /sandbox_size/);
  const empty = await http('/api/claude-code-session/create', {
    agent_id: agentId,
    goal: 'x',
    sandbox_size: {},
  });
  assert.equal(empty.status, 400, JSON.stringify(empty.data));
  const disk = await http('/api/claude-code-session/create', {
    agent_id: agentId,
    goal: 'x',
    sandbox_size: { cpu: 8, disk_gb: 50 },
  });
  assert.equal(disk.status, 400, 'disk is not a selectable dimension');
  const capped = await http('/api/claude-code-session/create', {
    agent_id: cappedAgentId,
    goal: 'x',
    sandbox_size: { cpu: 8, memory_gb: 8 },
  });
  assert.equal(capped.status, 400, JSON.stringify(capped.data));
  assert.equal(capped.data.code, 'SANDBOX_SIZE_UNSUPPORTED');
  const unbuilt = await http('/api/codex-code-session/create', {
    agent_id: agentId,
    user_id: randomUUID(),
    goal: 'x',
    sandbox_size: { cpu: 64 },
  });
  assert.equal(unbuilt.status, 400, JSON.stringify(unbuilt.data));
  assert.equal(unbuilt.data.code, 'SANDBOX_SIZE_UNSUPPORTED');
  assert.ok(Array.isArray(unbuilt.data.available), 'lists the available sizes');
});

test('forwarding: the requested size picks the template; a snapshot of another size is skipped', async () => {
  setEnv({ E2B_SANDBOX_TEMPLATES: 'grid' });
  const created: string[] = [];
  const originalCreate = Sandbox.create;
  Sandbox.create = (async (template: string) => {
    created.push(template);
    return { sandboxId: `fake-${created.length}` } as unknown as Sandbox;
  }) as unknown as typeof Sandbox.create;
  try {
    const requested = await acquireCodeSandbox(null, 'base-tpl', {}, 'Test', {
      provider: 'claude',
      requestedSize: { cpu: 2, memoryGb: 8 },
    });
    assert.deepEqual(created, [sizedTemplateName('claude', { cpu: 2, memoryGb: 8 })]);
    assert.deepEqual(requested.size, { cpu: 2, memoryGb: 8 });
    assert.equal(requested.sizeSource, 'request');

    created.length = 0;
    const oldSnapshot = { id: 'snap-2x4', size: { cpu: 2, memoryGb: 4 } };
    const raised = await acquireCodeSandbox(null, 'base-tpl', {}, 'Test', {
      provider: 'claude',
      snapshot: oldSnapshot,
    });
    assert.deepEqual(created, [sizedTemplateName('claude', { cpu: 8, memoryGb: 8 })]);
    assert.equal(raised.fromSnapshot, false, 'a 2x4 snapshot does not pin a default 8x8 session');
    assert.equal(raised.sizeSource, 'default');

    created.length = 0;
    const matching = await acquireCodeSandbox(null, 'base-tpl', {}, 'Test', {
      provider: 'claude',
      snapshot: { id: 'snap-8x8', size: { cpu: 8, memoryGb: 8 } },
    });
    assert.deepEqual(created, ['snap-8x8']);
    assert.equal(matching.fromSnapshot, true);
  } finally {
    Sandbox.create = originalCreate;
  }
});

test('maintenance: a project without a size builds its snapshot at the new default', () => {
  setEnv({ E2B_SANDBOX_TEMPLATES: 'grid' });
  assert.deepEqual(wantedTemplateSize({ sandboxSize: null }), { cpu: 8, memoryGb: 8 });
  assert.deepEqual(wantedTemplateSize({ sandboxSize: { cpu: 2, memoryGb: 4 } }), {
    cpu: 2,
    memoryGb: 4,
  });
  setEnv({});
  assert.deepEqual(wantedTemplateSize({ sandboxSize: null }), { cpu: 2, memoryGb: 4 });
});
