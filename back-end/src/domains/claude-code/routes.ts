/**
 * Claude Code session routes -- thin controller layer.
 *
 * POST /api/claude-code-session/create   - Create a new E2B sandbox session
 * POST /api/claude-code-session/close    - Close a session (checkpoint + pause a persistent
 *                                          code session's sandbox; kill legacy sessions)
 * POST /api/claude-code-session/checkpoint - Checkpoint a running task's code session
 * GET  /api/claude-code-session/code-sessions/:id    - Owned code session (404 otherwise)
 * DELETE /api/claude-code-session/code-sessions/:id - Retire it (kill + delete checkpoints)
 * POST /api/claude-code-session/code-sessions/:id/inspect - Read-only inspection of its latest
 *                                          checkpoint in a throwaway sandbox (internal)
 * GET  /api/claude-code-session/:taskId  - Get session info
 * GET  /api/claude-code-session/agent/:agentId - Get all sessions for an agent
 * POST /api/claude-code-session/:taskId/output - Persist terminal output
 * GET  /api/claude-code-session/:sandboxId/health - Health check sandbox
 */

import { isFeedbackCredential } from '../../providers/codeLoginRejection';
import { type Response, Router } from 'express';
import { ZodError } from 'zod';
import { type AuthRequest, authenticate, authenticateInternal } from '../../providers/auth';
import { callerCanAccessAgent, callerCanAccessCodeTask } from '../../providers/agentAccess';
import { createLogger } from '../../providers/logger';
import { asyncHandler } from '../../utils/asyncHandler';
import { createSessionSchema, closeSessionSchema } from './types';
import type { FilePointer } from '../../providers/storage';
import * as service from './service';
import { registerCodeSessionRoutes, sendCodeSessionError } from './routes.codeSessions';
import { registerCodeInspectRoutes } from './routes.codeInspect';
import { GithubRepoRefNotFoundError } from '../../providers/githubRepos/resolve';
import { AgentOwnerMismatchError, MissingUserIdError, GitHubAccessDeniedError } from './service';
import {
  LocalTargetUnavailableError,
  localTargetUnavailableBody,
} from '../../providers/localDevices/messages';
import {
  LocalSessionOwnerMismatchError,
  createLocalCodeSessionResponse,
} from '../../providers/localDevices/session';
import { ProjectNotFoundForSessionError } from '../../providers/projects/session';

const log = createLogger('ClaudeCodeRoutes');
const router = Router();

registerCodeSessionRoutes(router);
registerCodeInspectRoutes(router);

/**
 * C14: a local session runs on the agent tree's Mac with the user's own login,
 * in the folder they granted (no clone, Project restore or persistent code
 * session: the work stays in that folder). Otherwise an E2B sandbox resumes or
 * starts the agent's persistent code session.
 */
async function createSessionForTarget(body: unknown) {
  const {
    agent_id,
    goal,
    attachments,
    github_repo_url,
    github_repo_ref,
    user_id,
    code_session_id,
    fresh_conversation,
    lease_owner,
    base_branch,
    project_id,
    target,
    device_id,
    task_name,
    cli_settings,
    resume_config,
    sandbox_size,
  } = createSessionSchema.parse(body);
  if (target === 'local') {
    const local = await createLocalCodeSessionResponse({
      agentId: agent_id,
      goal,
      provider: 'claude',
      expectedUserId: user_id,
      deviceId: device_id,
      taskName: task_name ?? null,
      cliSettings: cli_settings,
      resumeConfig: resume_config,
      leaseOwner: lease_owner,
    });
    return { ...local, project: null };
  }
  return service.createSession({
    agent_id,
    goal,
    attachments: attachments as FilePointer[] | undefined,
    github_repo_url,
    github_repo_ref,
    user_id,
    code_session_id,
    fresh_conversation,
    lease_owner,
    base_branch,
    project_id,
    task_name,
    cli_settings,
    resume_config,
    sandbox_size,
  });
}

/**
 * POST /api/claude-code-session/create
 * Create a new E2B sandbox with Claude Code and WebSocket streaming.
 * Called by agents service to create a new session.
 * ANTHROPIC_API_KEY is injected from backend environment.
 */
router.post(
  '/create',
  authenticateInternal,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    try {
      const result = await createSessionForTarget(req.body);

      res.json({
        success: true,
        ...result,
      });
    } catch (error) {
      if (error instanceof ZodError) {
        // A malformed request is the caller's error, not a server failure: a 500
        // here read as a transient sandbox flake to the agent (RCA 942d8562).
        res.status(400).json({
          error: `Invalid request: ${error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`,
          code: 'INVALID_REQUEST',
        });
        return;
      }
      if (error instanceof LocalTargetUnavailableError) {
        res.status(409).json(localTargetUnavailableBody(error));
        return;
      }
      if (error instanceof LocalSessionOwnerMismatchError) {
        res.status(403).json({ error: error.message, code: 'AGENT_OWNER_MISMATCH' });
        return;
      }
      if (error instanceof MissingUserIdError) {
        res.status(400).json({
          error: 'user_id is required when using github_repo_url',
          code: 'MISSING_USER_ID',
        });
        return;
      }
      if (error instanceof GitHubAccessDeniedError) {
        res.status(403).json({
          error: error.detail,
          code: 'GITHUB_ACCESS_DENIED',
        });
        return;
      }
      if (error instanceof GithubRepoRefNotFoundError) {
        res.status(403).json({ error: error.message, code: 'GITHUB_ACCESS_DENIED' });
        return;
      }
      if (error instanceof ProjectNotFoundForSessionError) {
        res.status(404).json({ error: error.message, code: 'PROJECT_NOT_FOUND' });
        return;
      }
      if (error instanceof AgentOwnerMismatchError) {
        res.status(403).json({
          error: error.message,
          code: 'AGENT_OWNER_MISMATCH',
        });
        return;
      }
      if (sendCodeSessionError(res, error)) return;
      log.error('Failed to create Claude Code session', {
        error: error instanceof Error ? error.message : String(error),
      });
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Failed to create Claude Code session',
      });
    }
  })
);

/**
 * POST /api/claude-code-session/close
 * Close a Claude Code session (kills the sandbox).
 */
router.post(
  '/close',
  authenticateInternal,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    try {
      const { task_id, user_id, agent_id, additional_paths, reason } = closeSessionSchema.parse(
        req.body
      );

      const result = await service.closeSession({
        task_id,
        user_id,
        agent_id,
        additional_paths,
        reason,
      });

      res.json({
        success: true,
        ...result,
      });
    } catch (error) {
      log.error('Failed to close Claude Code session', {
        error: error instanceof Error ? error.message : String(error),
      });
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Failed to close Claude Code session',
      });
    }
  })
);

/**
 * GET /api/claude-code-session/:taskId
 * Get session info for a task.
 */
router.get(
  '/:taskId',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    try {
      const { taskId } = req.params;

      const session = (await callerCanAccessCodeTask(req, taskId))
        ? await service.getSession(taskId)
        : null;

      if (session === null) {
        res.status(404).json({ error: 'Session not found' });
        return;
      }

      res.json(session);
    } catch (error) {
      log.error('Failed to get Claude Code session', {
        error: error instanceof Error ? error.message : String(error),
      });
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Failed to get session info',
      });
    }
  })
);

/**
 * GET /api/claude-code-session/agent/:agentId
 * Get all sessions for an agent.
 */
router.get(
  '/agent/:agentId',
  authenticate,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    try {
      const { agentId } = req.params;

      if (!(await callerCanAccessAgent(req, agentId))) {
        res.status(404).json({ error: 'Agent not found' });
        return;
      }
      const sessions = await service.getAgentSessions(agentId);

      res.json({ sessions });
    } catch (error) {
      log.error('Failed to get agent sessions', {
        error: error instanceof Error ? error.message : String(error),
      });
      res.status(500).json({ error: 'Failed to get agent sessions' });
    }
  })
);

/**
 * POST /api/claude-code-session/:taskId/output
 * Persist terminal output for a Claude Code session.
 * Called by the agent worker after the session completes.
 */
router.post(
  '/:taskId/output',
  authenticateInternal,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    try {
      const { taskId } = req.params;
      const { output, credential_feedback: rawCredentialFeedback } = req.body as {
        output: unknown;
        credential_feedback?: unknown;
      };

      if (output === undefined || output === null || typeof output !== 'string' || output === '') {
        res.status(400).json({ error: 'output field required' });
        return;
      }
      // Structured credential failures (agents build_credential_feedback); absent = none observed.
      const credentialFeedback = rawCredentialFeedback ?? [];
      const validFeedback =
        Array.isArray(credentialFeedback) &&
        credentialFeedback.length <= 3 &&
        credentialFeedback.every((item: unknown) => {
          if (typeof item !== 'object' || item === null) return false;
          const candidate = item as Record<string, unknown>;
          return (
            candidate.source === 'initial' &&
            candidate.reason === 'authentication_failed' &&
            isFeedbackCredential(candidate.credential)
          );
        });
      if (!validFeedback) {
        res.status(400).json({ error: 'credential_feedback event is invalid' });
        return;
      }

      await service.persistOutput(
        taskId,
        output,
        credentialFeedback as service.ClaudeCredentialFeedback[]
      );

      res.json({ success: true });
    } catch (error) {
      log.error('Failed to persist Claude Code output', {
        error: error instanceof Error ? error.message : String(error),
      });
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Failed to persist output',
      });
    }
  })
);

/**
 * GET /api/claude-code-session/:sandboxId/health
 * Check if an E2B sandbox is still alive and responsive.
 * Used by agents to determine root cause of WebSocket disconnects.
 *
 * If sandbox is alive, extracts diagnostic info to help debug crashes.
 */
router.get(
  '/:sandboxId/health',
  authenticateInternal,
  asyncHandler(async (req: AuthRequest, res: Response) => {
    const { sandboxId } = req.params;

    const result = await service.checkSandboxHealth(sandboxId);

    res.json(result);
  })
);

export default router;
