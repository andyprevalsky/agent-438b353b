import assert from 'node:assert/strict';
import { mock } from 'node:test';
import type { CodeSessionRow } from '../../providers/codeSessions/store';

async function main() {
  let resume = true;
  let rootReads = 0;
  let creates = 0;
  let recorded = 0;
  let missing = false;
  let resumes = 0;
  let sizingReads = 0;
  let expectedProvider = 'claude';
  let expectedBase = 'base-v1';
  let expectedSize = { cpu: 2, memoryGb: 4 };
  let rejectSnapshot = false;
  let slotDepth = 0;
  let slotCalls = 0;
  let handoffFails = false;
  let handoffs = 0;
  let kills = 0;
  const templateCalls: unknown[][] = [];
  await mock.module('../../providers/codeSessions/usageLimit', {
    namedExports: {
      writeHandoffTrajectory: async () => {
        handoffs++;
        if (handoffFails) throw new Error('handoff write failed');
      },
    },
  });
  await mock.module('../../providers/e2bCreateSlot', {
    namedExports: {
      withClaudeCodeCreateSlot: async (create: () => Promise<unknown>) => {
        slotCalls++;
        slotDepth++;
        try {
          return await create();
        } finally {
          slotDepth--;
        }
      },
    },
  });
  const held = { sandboxId: 'held', commands: { run: async () => ({}) } };
  const fresh = {
    sandboxId: 'fresh',
    kill: async () => {
      kills++;
    },
  };
  await mock.module('../../providers/codeSessions/lifecycle', {
    namedExports: {
      connectPausedSandbox: async () => {
        resumes++;
        return resume ? held : null;
      },
      checkpointCodeSession: async () => null,
      prepareLocalCodeSessionHome: async () => undefined,
      restoreLatestCheckpoint: async () => null,
    },
  });
  await mock.module('../../providers/codeSessions/store', {
    namedExports: {
      recordSandbox: async () => {
        recorded++;
      },
      linkClaudeCodeSession: async () => undefined,
      markPaused: async () => undefined,
    },
  });
  await mock.module('../../providers/codeSessions/sandboxSize', {
    namedExports: {
      resolveSpawnSandbox: async () => {
        sizingReads++;
        return { size: { cpu: 2, memoryGb: 4 }, templateId: 'base-v1' };
      },
      heldSandboxSize: () => ({ cpu: 2, memoryGb: 4 }),
      sameSize: () => true,
      defaultSandboxSize: () => ({ cpu: 2, memoryGb: 4 }),
      maxSizeFor: async () => ({ cpu: 8, memoryGb: 8 }),
    },
  });
  await mock.module('../../providers/nativeCliTemplateBuilder', {
    namedExports: {
      NATIVE_CLI_PATH: '/fixture/native/bin:/usr/bin',
      resolveNativeCliTemplate: async (...args: unknown[]) => {
        rootReads++;
        templateCalls.push(args);
        const base = args[3];
        assert.equal(args[0], 'owner');
        assert.equal(args[1], expectedProvider);
        assert.deepEqual(
          args[2],
          base === 'project-snapshot' ? expectedSize : { cpu: 2, memoryGb: 4 }
        );
        assert.ok(base === expectedBase || (rejectSnapshot && base === 'base-v1'));
        if (missing) throw new Error('native variant missing');
        return args[3] === 'project-snapshot' ? 'overlaid-project' : 'snapshot-v1';
      },
    },
  });
  await mock.module('../../providers/e2bSandboxCreate', {
    namedExports: {
      createSandboxWithRetry: async (
        template: string,
        options: { envs: Record<string, string> }
      ) => {
        creates++;
        assert.equal(
          slotDepth,
          expectedProvider === 'claude' ? 1 : 0,
          'every Claude runtime create holds the slot'
        );
        assert.ok(template === 'snapshot-v1' || template === 'overlaid-project');
        if (rejectSnapshot && template === 'overlaid-project')
          throw new Error('snapshot unavailable');
        assert.equal(options.envs.PATH, '/fixture/native/bin:/usr/bin');
        assert.equal(options.envs.AUTH_FOR_CURRENT_AGENT, 'runtime-only');
        return fresh;
      },
    },
  });
  const { acquireCodeSandbox } = await import('../../providers/e2bCodeSessionSandbox');
  const session = {
    id: 'session',
    userId: 'owner',
    sandboxKind: 'e2b',
    sandboxId: 'held',
    cliSessionId: 'prior-conversation',
    latestCheckpointSeq: 0,
  } as CodeSessionRow;
  const invoke = () =>
    acquireCodeSandbox(session, 'fallback', { AUTH_FOR_CURRENT_AGENT: 'runtime-only' }, 'fixture', {
      provider: 'claude',
    });
  const beforeForeign = [resumes, sizingReads, rootReads, creates, recorded];
  await assert.rejects(
    acquireCodeSandbox(session, 'fallback', {}, 'foreign', { provider: 'claude', userId: 'other' }),
    /does not belong/
  );
  assert.deepEqual(
    [resumes, sizingReads, rootReads, creates, recorded],
    beforeForeign,
    'foreign callers cannot inspect sizing, resume a held sandbox, read templates or allocate'
  );
  const resumed = await invoke();
  assert.equal(resumed.sandbox, held);
  assert.equal(resumed.cliSessionId, 'prior-conversation');
  assert.equal(rootReads, 0, 'resuming an existing sandbox does not consult or change its root');
  assert.equal(creates, 0);
  resume = false;
  const created = await invoke();
  assert.equal(created.sandbox, fresh);
  assert.equal(created.resumedFrom, 'new');
  assert.equal(rootReads, 1);
  assert.equal(creates, 1);
  assert.equal(recorded, 1);
  missing = true;
  await assert.rejects(invoke(), /native variant missing/);
  assert.equal(creates, 1, 'a missing published variant does not allocate an unconfigured base');
  missing = false;
  for (const provider of ['claude', 'codex'] as const) {
    expectedProvider = provider;
    const allocated = await acquireCodeSandbox(
      null,
      'fallback',
      { AUTH_FOR_CURRENT_AGENT: 'runtime-only' },
      'new caller',
      { provider, userId: 'owner' }
    );
    assert.equal(allocated.sandbox, fresh);
  }
  assert.equal(rootReads, 4, 'both stateless providers consult the authenticated user template');
  assert.equal(creates, 3);
  assert.equal(recorded, 1, 'stateless spawns do not invent a persistent session');
  const beforeForeignNew = [resumes, sizingReads, rootReads, creates, recorded];
  await assert.rejects(
    acquireCodeSandbox({ ...session, sandboxId: null }, 'fallback', {}, 'foreign', {
      provider: 'codex',
      userId: 'other',
    }),
    /does not belong/
  );
  assert.deepEqual([resumes, sizingReads, rootReads, creates, recorded], beforeForeignNew);
  expectedProvider = 'claude';
  expectedBase = 'project-snapshot';
  expectedSize = { cpu: 4, memoryGb: 8 };
  const snapshot = { id: 'project-snapshot', size: expectedSize };
  const newSession = { ...session, sandboxId: null };
  const options = {
    provider: 'claude' as const,
    userId: 'owner',
    project: {
      id: 'project',
      name: 'project',
      repoUrl: null,
      defaultBranch: 'main',
      startupScript: null,
      startupTimeoutS: 30,
      templateSnapshot: snapshot,
    },
  };
  let startCalls = templateCalls.length;
  const fromProject = await acquireCodeSandbox(
    newSession,
    'fallback',
    { AUTH_FOR_CURRENT_AGENT: 'runtime-only' },
    'project',
    options
  );
  assert.equal(fromProject.fromSnapshot, true);
  assert.deepEqual(fromProject.size, snapshot.size);
  assert.equal(templateCalls.length, startCalls + 1);
  assert.equal(templateCalls.at(-1)?.[3], snapshot.id);
  expectedSize = { cpu: 2, memoryGb: 4 };
  const unknownSize = await acquireCodeSandbox(
    null,
    'fallback',
    { AUTH_FOR_CURRENT_AGENT: 'runtime-only' },
    'unknown size',
    {
      ...options,
      snapshot: { id: snapshot.id, size: null },
    }
  );
  assert.equal(unknownSize.fromSnapshot, true);
  assert.deepEqual(unknownSize.size, expectedSize);
  // A failed snapshot create falls back to an overlaid base, never an unconfigured image.
  rejectSnapshot = true;
  startCalls = templateCalls.length;
  expectedSize = snapshot.size;
  const fallback = await acquireCodeSandbox(
    null,
    'fallback',
    { AUTH_FOR_CURRENT_AGENT: 'runtime-only' },
    'fallback',
    options
  );
  assert.equal(fallback.fromSnapshot, false);
  assert.deepEqual(fallback.size, { cpu: 2, memoryGb: 4 });
  assert.deepEqual(
    templateCalls.slice(startCalls).map((args) => args[3]),
    ['project-snapshot', 'base-v1']
  );
  rejectSnapshot = false;
  resume = true;
  startCalls = templateCalls.length;
  const resumedWithProject = await acquireCodeSandbox(session, 'fallback', {}, 'resume', options);
  assert.equal(resumedWithProject.sandbox, held);
  assert.equal(resumedWithProject.fromSnapshot, false);
  assert.equal(templateCalls.length, startCalls);
  resume = false;
  const slotsBefore = slotCalls;
  await acquireCodeSandbox(
    { ...newSession, handoffTrajectory: { id: 'fixture' } } as unknown as CodeSessionRow,
    'fallback',
    { AUTH_FOR_CURRENT_AGENT: 'runtime-only' },
    'handoff',
    options
  );
  assert.equal(handoffs, 1);
  assert.equal(slotCalls, slotsBefore + 1);
  handoffFails = true;
  await assert.rejects(
    acquireCodeSandbox(
      { ...newSession, handoffTrajectory: { id: 'fixture' } } as unknown as CodeSessionRow,
      'fallback',
      { AUTH_FOR_CURRENT_AGENT: 'runtime-only' },
      'handoff failure',
      options
    ),
    /handoff write failed/
  );
  assert.equal(kills, 1, 'failed handoff closes the exact newly acquired sandbox');
  process.stdout.write(
    'Native CLI provision: resumes unchanged, new allocation selects native root and live auth, missing variant fails closed\n'
  );
}
main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
