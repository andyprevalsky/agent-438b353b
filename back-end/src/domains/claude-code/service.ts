/**
 * Claude Code domain service -- business logic for session management.
 */

import * as repo from './repo';
import {
  createClaudeCodeSession,
  closeClaudeCodeSession,
  getClaudeCodeSession,
  type ClaudeCodeSessionOptions,
} from '../../providers/e2b';
import { checkGithubRepoAccess, ensureAgentTreeRepo } from '../../providers/github';
import {
  repoAccessUserFor,
  resolveCodeSessionWorkspace,
} from '../../providers/githubRepos/resolve';
import { GithubRateLimitedError } from '../../providers/githubRateLimit';
import { createLogger } from '../../providers/logger';
import {
  markPoolTokenRateLimited,
  markUserTokenRateLimited,
  storedClaudeLogin,
} from '../../providers/claudeCredentials';
import { classifyLoginRejection } from '../../providers/codeLoginRejection';
import { markAnthropicApiKeyOnCooldown } from '../../providers/computeCredentials';
import { createWithCodeSession } from '../../providers/codeSessions/spawn';
import { requestedSandboxSize, type SandboxSize } from '../../providers/codeSessions/sandboxSize';

export { checkSandboxHealth, type HealthCheckResult } from './health';

const log = createLogger('ClaudeCode');

// ---------------------------------------------------------------------------
// Error classes
// ---------------------------------------------------------------------------

export class MissingUserIdError extends Error {
  constructor() {
    super('user_id is required when using github_repo_url');
    this.name = 'MissingUserIdError';
  }
}

export class GitHubAccessDeniedError extends Error {
  constructor(public detail: string) {
    super(detail);
    this.name = 'GitHubAccessDeniedError';
  }
}

export class AgentOwnerMismatchError extends Error {
  constructor() {
    super('The supplied user_id does not own this agent');
    this.name = 'AgentOwnerMismatchError';
  }
}

export class AgentContextLookupError extends Error {
  constructor() {
    super('Failed to resolve the agent owner and GitHub repository');
    this.name = 'AgentContextLookupError';
  }
}

// ---------------------------------------------------------------------------
// Create session
// ---------------------------------------------------------------------------

export type { CreateSessionParams, CreateSessionResult } from './types';
import type { CloseSessionParams, CreateSessionParams, CreateSessionResult } from './types';

/** Resolve the owning user and optional GitHub repo from the agent tree. */
export async function resolveAgentContext(
  agentId: string,
  userId: string | undefined
): Promise<{
  repoUrl: string | undefined;
  userId: string | undefined;
  projectId?: string | undefined;
  repoRef?: string | undefined;
}> {
  try {
    const agent = await repo.findAgentGithubInfo(agentId);
    let storedOwnerId = agent?.userId ?? undefined;
    let repoUrl = agent?.githubRepoUrl ?? undefined;
    let projectId = agent?.projectId ?? undefined;
    let repoRef = agent?.githubRepoRef ?? undefined;

    if (
      repoUrl === undefined &&
      agent?.busId !== undefined &&
      agent.busId !== null &&
      agent.busId !== '' &&
      agent.busId !== agentId
    ) {
      const rootAgent = await repo.findRootAgentGithubInfo(agent.busId);
      repoUrl = rootAgent?.githubRepoUrl ?? undefined;
      storedOwnerId ??= rootAgent?.userId;
      projectId ??= rootAgent?.projectId ?? undefined;
      repoRef ??= rootAgent?.githubRepoRef ?? undefined;
    }

    const callerUserId = userId !== undefined && userId !== '' ? userId : undefined;
    if (
      storedOwnerId !== undefined &&
      callerUserId !== undefined &&
      callerUserId !== storedOwnerId
    ) {
      throw new AgentOwnerMismatchError();
    }
    const resolvedUserId = storedOwnerId ?? callerUserId;

    return { repoUrl, userId: resolvedUserId, projectId, repoRef };
  } catch (err) {
    log.error('Failed to look up agent owner and GitHub repo', {
      error: err instanceof Error ? err.message : String(err),
    });
    if (err instanceof AgentOwnerMismatchError) throw err;
    throw new AgentContextLookupError();
  }
}

/** The spawn's options for createClaudeCodeSession (project, code session, CLI conversation, title). */
function sessionOptions(
  params: CreateSessionParams,
  projectWorkspace: ClaudeCodeSessionOptions['projectWorkspace'],
  codeSession: ClaudeCodeSessionOptions['codeSession'],
  sandboxSize: SandboxSize | null
): ClaudeCodeSessionOptions {
  return {
    sandboxSize,
    projectWorkspace,
    codeSession,
    freshConversation: params.fresh_conversation === true,
    taskName: params.task_name ?? null,
    cliSettings: params.cli_settings ?? null,
    resumeConfig: params.resume_config,
    leaseOwner: params.lease_owner,
  };
}

/**
 * Project restore, unless a canonical repo ref (spawn's github_repo_ref or agent override)
 * replaces it; then the repo's connected folder restores (its startup script, env files).
 */
function sessionWorkspace(
  params: CreateSessionParams,
  resolved: Awaited<ReturnType<typeof resolveAgentContext>>
) {
  return resolveCodeSessionWorkspace(resolved.userId, {
    githubRepoRef: params.github_repo_ref,
    agentRepoRef: resolved.repoRef,
    projectId: params.project_id ?? resolved.projectId,
    requestedRepoUrl: params.github_repo_url,
  });
}

/** The agent's owner and repo, and the spawn's validated sandbox size (400 when unsatisfiable). */
async function resolveSpawnContext(params: CreateSessionParams) {
  const resolved = await resolveAgentContext(params.agent_id, params.user_id);
  const sandboxSize = await requestedSandboxSize('claude', params.sandbox_size, resolved.userId);
  return { resolved, sandboxSize };
}

export async function createSession(params: CreateSessionParams): Promise<CreateSessionResult> {
  const { agent_id, goal, attachments, user_id } = params;
  let effectiveGithubRepoUrl = params.github_repo_url;
  let effectiveUserId = user_id;

  // The agents service does not need to duplicate the user ID on every internal
  // code-session request. Resolve ownership from the authoritative Agent row so
  // repo-less agents can still consume their user's stored OAuth credential.
  const { resolved, sandboxSize } = await resolveSpawnContext(params);
  effectiveUserId = resolved.userId;
  const workspace = await sessionWorkspace(params, resolved);
  const { projectWorkspace, repoOverrideUrl } = workspace;
  if (repoOverrideUrl !== undefined) effectiveGithubRepoUrl = resolved.repoUrl = repoOverrideUrl;
  if (projectWorkspace !== null) {
    // The project restore owns the checkout; never also clone github_repo_url.
    effectiveGithubRepoUrl = undefined;
    resolved.repoUrl = undefined;
  }

  if (effectiveGithubRepoUrl === undefined || effectiveGithubRepoUrl === '') {
    // Auto-inject the tree's repo; its first code session creates it.
    if (projectWorkspace === null) resolved.repoUrl ??= await ensureAgentTreeRepo(agent_id);
    if (resolved.repoUrl !== undefined) {
      effectiveGithubRepoUrl = resolved.repoUrl;
      log.info(`Auto-injecting GitHub repo from agent tree: ${effectiveGithubRepoUrl}`);
    }
  }

  const attachmentCount = attachments !== undefined ? attachments.length : 0;
  log.info(`Creating session for agent ${agent_id}`, {
    attachments: attachmentCount > 0 ? String(attachmentCount) : undefined,
    github_repo: effectiveGithubRepoUrl ?? undefined,
  });

  // Check GitHub repo access if specified
  let normalizedRepoUrl: string | undefined;
  if (effectiveGithubRepoUrl !== undefined && effectiveGithubRepoUrl !== '') {
    if (effectiveUserId === undefined || effectiveUserId === '') {
      throw new MissingUserIdError();
    }

    // A teammate's shared repo clones under its owner's grant; the session stays the caller's.
    const accessUserId = repoAccessUserFor(workspace, effectiveGithubRepoUrl, effectiveUserId);
    const accessCheck = await checkGithubRepoAccess(accessUserId, effectiveGithubRepoUrl);
    if (accessCheck.rateLimit !== undefined)
      throw new GithubRateLimitedError(accessCheck.rateLimit);
    if (!accessCheck.allowed) {
      log.info(`GitHub access denied for user ${accessUserId}: ${accessCheck.error ?? ''}`);
      throw new GitHubAccessDeniedError(accessCheck.error!);
    }
    normalizedRepoUrl = accessCheck.normalizedUrl!;
    log.info(`GitHub access granted for ${normalizedRepoUrl}`);
  }

  const { agentFastMode } = await import('../../providers/demoCacheStore');
  const fastMode = await agentFastMode(agent_id);
  // Code-agent Anthropic/Codex proxy is DEMO cache replay — not the Fast Mode button.
  const fastModeCacheEnabled = fastMode.enabled;

  // A project session is scoped (and branched) by its project; its repo is the
  // project's repo, cut from the project's default branch.
  const sessionRepoUrl =
    projectWorkspace !== null
      ? (projectWorkspace.project.repoUrl ?? `rose-project:${projectWorkspace.project.id}`)
      : (normalizedRepoUrl ?? null);
  const { session, fields } = await createWithCodeSession(
    {
      agentId: agent_id,
      userId: effectiveUserId,
      repoUrl: sessionRepoUrl,
      codeProvider: 'claude',
      codeSessionId: params.code_session_id,
      leaseOwner: params.lease_owner,
      baseBranch: params.base_branch ?? projectWorkspace?.project.defaultBranch ?? null,
    },
    (codeSession, handoff) =>
      createClaudeCodeSession(
        agent_id,
        handoff !== null ? `${handoff}\n\n${goal}` : goal,
        attachments,
        normalizedRepoUrl,
        effectiveUserId,
        fastModeCacheEnabled,
        sessionOptions(params, projectWorkspace, codeSession, sandboxSize)
      )
  );

  return {
    task_id: session.taskId,
    sandbox_id: session.sandboxId,
    ws_url: session.wsUrl,
    uploaded_files: session.uploadedFiles,
    github_repo_url: normalizedRepoUrl ?? null,
    ...fields,
    project: session.projectRestore ?? null,
    preview_url: session.previewUrl ?? null,
  };
}

// ---------------------------------------------------------------------------
// Close session
// ---------------------------------------------------------------------------

export type { CloseSessionParams } from './types';

export async function closeSession(params: CloseSessionParams) {
  const { task_id, user_id, agent_id, additional_paths } = params;

  const pathCount =
    additional_paths !== undefined && additional_paths.length > 0 ? additional_paths.length : 0;
  log.info(`Closing Claude Code session ${task_id}`, {
    additional_paths: pathCount > 0 ? String(pathCount) : undefined,
  });

  const result = await closeClaudeCodeSession(task_id, user_id, agent_id, additional_paths, {
    reason: params.reason,
  });

  return {
    status: 'CLOSED' as const,
    output_files: result.outputFiles,
    usage: result.usage,
    code_session: result.codeSession ?? null,
    ...(result.handoff !== undefined ? { handoff: result.handoff } : {}),
  };
}

// ---------------------------------------------------------------------------
// Get session
// ---------------------------------------------------------------------------

export async function getSession(taskId: string) {
  const session = await getClaudeCodeSession(taskId);

  if (session === undefined || session === null) {
    return null;
  }

  return {
    task_id: session.id,
    sandbox_id: session.sandboxId,
    stream_url: session.streamUrl,
    goal: session.goal,
    status: session.status,
    output: session.output,
    created_at: session.createdAt,
    closed_at: session.closedAt,
  };
}

// ---------------------------------------------------------------------------
// Get agent sessions
// ---------------------------------------------------------------------------

export async function getAgentSessions(agentId: string) {
  return repo.findSessionsByAgent(agentId);
}

// ---------------------------------------------------------------------------
// Persist output
// ---------------------------------------------------------------------------

/**
 * A credential failure the run observed, from the CLI's structured events only
 * (agents ``build_credential_feedback``): ``authentication_failed`` is the
 * SDK's 401/403 kind; ``credential`` is the ws-server's fingerprint of the
 * token the sandbox held. Usage limits and out of credits arrive through the
 * C15 usage-limit report instead.
 */
export type ClaudeCredentialFeedback = {
  source: 'initial';
  reason: 'authentication_failed';
  credential?: string;
};

export async function persistOutput(
  taskId: string,
  output: string,
  credentialFeedback: ClaudeCredentialFeedback[] = []
) {
  log.info(`Persisting output for session ${taskId} (${output.length} bytes)`);
  await repo.persistSessionOutput(taskId, output);
  const rejection = credentialFeedback.find((item) => item.reason === 'authentication_failed');
  if (rejection !== undefined) await markRejectedLogin(taskId, rejection.credential);
}

/**
 * The provider rejected the credential this task ran on. The terminal text is
 * never read: an agent may print any banner while doing healthy work.
 *
 * The shared funded key backs off for every spawn. A leased login is disabled
 * (needs reconnect) only when the stored credential is still the token the
 * sandbox held and has not simply expired (codeLoginRejection.ts); an expired or
 * since re-synced token is left to its device, and serving already skips an
 * expired one.
 */
async function markRejectedLogin(taskId: string, heldFingerprint: string | undefined) {
  const session = await repo.findSessionById(taskId);
  if (session === null) {
    log.warn(`Claude Code login rejection for unknown session ${taskId}`);
    return;
  }
  const tokenSource = session.claudeTokenSource ?? '';
  const integrationId = session.claudeTokenIntegrationId ?? '';
  if (tokenSource === '' || tokenSource === 'anthropic') {
    log.warn(`Claude Code rejected the funded ANTHROPIC_API_KEY for ${taskId}`);
    await markAnthropicApiKeyOnCooldown('claude_code_auth_failure');
    return;
  }
  if (integrationId === '') {
    log.warn(`Claude Code login rejection lacks credential identity for session ${taskId}`);
    return;
  }
  const rejection = classifyLoginRejection(await storedClaudeLogin(integrationId), heldFingerprint);
  if (rejection !== 'revoked') {
    log.info(
      `Claude Code login ${integrationId} rejected in ${taskId}: ${rejection}; not disabled`
    );
    return;
  }
  log.warn(`Claude Code login ${integrationId} revoked (${tokenSource}) in ${taskId}`);
  if (tokenSource === 'pool') {
    await markPoolTokenRateLimited(integrationId, { failureReason: 'claude_code_auth_error' });
  } else if (tokenSource === 'user') {
    const agent = await repo.findAgentById(session.agentId);
    if ((agent?.userId ?? '') !== '') {
      await markUserTokenRateLimited(agent!.userId, {
        failureReason: 'claude_code_auth_error',
        integrationId,
      });
    }
  }
}
