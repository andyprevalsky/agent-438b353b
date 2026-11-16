/**
 * The sandbox half of a C16 resize (see resizeTransaction.ts for the whole
 * transaction): save the held sandbox, restore exactly that checkpoint into a
 * fresh sandbox of the new size, validate it, then cut the session row over.
 * Until the cut-over the held sandbox is the only environment the row names;
 * after it, the held sandbox is paused (inside a resize transaction, kept for
 * commit or rollback) or killed (a plain spawn that applied a pending request).
 */
import type { Sandbox } from 'e2b';
import { createLogger } from '../logger';
import { CheckpointIntegrityError, loadVerifiedCheckpoint } from './checkpoint';
import { applyVerifiedCheckpoint, checkpointCodeSession } from './lifecycle';
import {
  checkHeldCapacity,
  checkpointArtifactBytes,
  checkTargetCapacity,
  currentResizeTransaction,
  diskUsage,
  RESIZE_RESTORE_ATTEMPTS,
  ResizeStepError,
  timedPhase,
  validateRestoredCheckpoint,
} from './resizeTransaction';
import { e2bSandboxFs } from './sandboxFs';
import type { SandboxSize } from './sandboxSize';
import { recordSandbox, type CodeSessionRow } from './store';

const log = createLogger('CodeSandboxResize');

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Move a session from its (resumed, CLI stopped) held sandbox to a new one
 * made by `create`. Throws a ResizeStepError with the held sandbox paused and
 * still named by the row whenever the new sandbox is not restored and valid.
 */
export async function moveToResizedSandbox(
  session: CodeSessionRow,
  held: Sandbox,
  create: () => Promise<Sandbox>,
  sizing: { size?: SandboxSize; agentId?: string }
): Promise<{ sandbox: Sandbox; cliSessionId: string | null }> {
  const tx = currentResizeTransaction();
  const keepHeld = async (error: unknown): Promise<never> => {
    await held.pause().catch(() => undefined);
    throw error;
  };
  const heldDisk = await timedPhase('preflight', () => diskUsage(e2bSandboxFs(held)));
  if (tx !== null) tx.disk.held = heldDisk;
  try {
    checkHeldCapacity(heldDisk);
  } catch (error) {
    return keepHeld(error);
  }
  const checkpoint = await timedPhase('checkpoint', () =>
    checkpointCodeSession({
      session,
      fs: e2bSandboxFs(held),
      reason: 'resize',
      agentId: sizing.agentId ?? session.ownerAgentId,
    })
  );
  if (checkpoint === null) {
    return keepHeld(
      new ResizeStepError(
        'CHECKPOINT_FAILED',
        'The session could not be saved before moving it to another sandbox size'
      )
    );
  }
  let verified;
  try {
    // Exactly the checkpoint just taken: an older one would drop the latest work.
    verified = await loadVerifiedCheckpoint(session.userId, session.id, checkpoint.seq);
  } catch (error) {
    return keepHeld(
      error instanceof CheckpointIntegrityError
        ? new ResizeStepError('CHECKPOINT_INTEGRITY', error.message)
        : error
    );
  }
  const saved = {
    ...session,
    latestCheckpointSeq: Math.max(session.latestCheckpointSeq, checkpoint.seq),
    cliSessionId: checkpoint.cliSessionId ?? session.cliSessionId,
  };

  let lastError: unknown = null;
  for (let attempt = 1; attempt <= RESIZE_RESTORE_ATTEMPTS; attempt += 1) {
    let target: Sandbox;
    try {
      target = await timedPhase('create', create);
    } catch (error) {
      return keepHeld(new ResizeStepError('TARGET_CREATE_FAILED', messageOf(error)));
    }
    if (tx !== null) tx.restoreAttempts = attempt;
    try {
      const fs = e2bSandboxFs(target);
      const targetDisk = await timedPhase('preflight', () => diskUsage(fs));
      if (tx !== null) tx.disk.target = targetDisk;
      checkTargetCapacity(heldDisk, targetDisk, checkpointArtifactBytes(verified));
      const restored = await timedPhase('restore', async () => {
        try {
          return await applyVerifiedCheckpoint(saved, fs, verified);
        } catch (error) {
          throw new ResizeStepError('RESTORE_FAILED', messageOf(error), true);
        }
      });
      await timedPhase('validate', () => validateRestoredCheckpoint(fs, verified));
      await cutOver(saved, held, target, sizing.size);
      return { sandbox: target, cliSessionId: restored.cliSessionId };
    } catch (error) {
      // The target holds only a copy of the stored checkpoint: safe to drop.
      await target.kill().catch(() => undefined);
      lastError = error;
      log.warn('Resize target sandbox rejected', {
        codeSessionId: session.id,
        sandboxId: target.sandboxId,
        attempt: String(attempt),
        error: messageOf(error).slice(0, 200),
      });
      if (!(error instanceof ResizeStepError) || !error.retryable) break;
    }
  }
  return keepHeld(lastError);
}

/**
 * The row names the restored, validated target now. In a resize transaction
 * the held sandbox is paused and kept until the target serves (commit) or
 * fails to (rollback); outside one (a spawn applying a pending request) it is
 * killed as soon as the row moved, as before.
 */
async function cutOver(
  session: CodeSessionRow,
  held: Sandbox,
  target: Sandbox,
  size: SandboxSize | undefined
): Promise<void> {
  await recordSandbox(session.id, {
    sandboxKind: 'e2b',
    sandboxId: target.sandboxId,
    ...(size !== undefined ? { size } : {}),
  });
  const tx = currentResizeTransaction();
  if (tx !== null) {
    tx.targetSandboxId = target.sandboxId;
    tx.retained = held;
    await held.pause().catch((error: unknown) => {
      log.warn('Pausing the original sandbox at resize cut-over failed; it stays up', {
        codeSessionId: session.id,
        sandboxId: held.sandboxId,
        error: messageOf(error),
      });
    });
    return;
  }
  try {
    await held.kill();
    log.info('Replaced the code-session sandbox with a resized one', {
      codeSessionId: session.id,
      previous: held.sandboxId,
      sandboxId: target.sandboxId,
    });
  } catch (error) {
    log.warn('Killing the replaced code-session sandbox failed', {
      codeSessionId: session.id,
      sandboxId: held.sandboxId,
      error: messageOf(error),
    });
  }
}
