import * as repo from './repo';
import {
  createCodexCodeSession,
  closeClaudeCodeSession,
  getClaudeCodeSession,
} from '../../providers/e2b';
import { checkGithubRepoAccess, ensureAgentTreeRepo } from '../../providers/github';
import {
  repoAccessUserFor,
  resolveCodeSessionWorkspace,
} from '../../providers/githubRepos/resolve';
import { GithubRateLimitedError } from '../../providers/githubRateLimit';
import { createLogger } from '../../providers/logger';
import { codexSpeedFor } from '../../providers/codexSpeed';

import { createWithCodeSession } from '../../providers/codeSessions/spawn';
import { requestedSandboxSize, type SandboxSize } from '../../providers/codeSessions/sandboxSize';
import type { CodexCodeSessionOptions } from '../../providers/codeSessionLaunchOptions';

export type { HealthCheckResult } from './health';
export { checkSandboxHealth } from './health';

export type { CodexCredentialFeedback } from './credentialFeedback';
import { detectAndMarkCodexExhaustion, type CodexCredentialFeedback } from './credentialFeedback';
export { isCodexCredentialFeedback } from './credentialFeedback';

const log = createLogger('CodexCode');

export class MissingUserIdError extends Error {
  constructor() {
    super('user_id is required to create a Codex Code session');
    this.name = 'MissingUserIdError';
  }
}

export class GitHubAccessDeniedError extends Error {
  constructor(public detail: string) {
    super(detail);
    this.name = 'GitHubAccessDeniedError';
  }
}

export type { CloseSessionParams, CreateSessionParams } from './types';
import type { CloseSessionParams, CreateSessionParams } from './types';

async function resolveGithubRepo(
  agentId: string,
  userId: string
): Promise<{ repoUrl: string | undefined; userId: string }> {
  try {
    const agent = await repo.findAgentGithubInfo(agentId);
    if (agent?.githubRepoUrl !== undefined && agent.githubRepoUrl !== null) {
      return { repoUrl: agent.githubRepoUrl, userId };
    }
    if (
      agent?.busId !== undefined &&
      agent.busId !== null &&
      agent.busId !== '' &&
      agent.busId !== agentId
    ) {
      const rootAgent = await repo.findRootAgentGithubInfo(agent.busId);
      if (rootAgent?.githubRepoUrl !== undefined && rootAgent.githubRepoUrl !== null) {
        return { repoUrl: rootAgent.githubRepoUrl, userId };
      }
    }
  } catch (err) {
    log.error('Failed to look up agent GitHub repo', {
      error: err instanceof Error ? err.message : String(err),
    });
  }
  // No repo yet: the tree gets one on its first code session.
  return { repoUrl: await ensureAgentTreeRepo(agentId), userId };
}

/**
 * The agent's project, else its tree root's (one indexed read per agent row).
 * A failed lookup is logged and reads as "no project", so session creation
 * falls back to the agent tree's repo exactly as it did before projects existed.
 */
async function resolveAgentProjectId(agentId: string): Promise<string | undefined> {
  try {
    const agent = await repo.findAgentGithubInfo(agentId);
    if (agent?.projectId !== undefined && agent.projectId !== null) return agent.projectId;
    if (agent?.busId !== undefined && agent.busId !== null && agent.busId !== agentId) {
      return (await repo.findRootAgentGithubInfo(agent.busId))?.projectId ?? undefined;
    }
  } catch (err) {
    log.error('Failed to look up agent project', {
      error: err instanceof Error ? err.message : String(err),
    });
  }
  return undefined;
}

/** The agent's persisted repo ref, else its tree root's (null on lookup failure). */
async function resolveAgentRepoRef(agentId: string): Promise<string | null> {
  try {
    const agent = await repo.findAgentGithubInfo(agentId);
    if (agent?.githubRepoRef !== undefined && agent.githubRepoRef !== null) {
      return agent.githubRepoRef;
    }
    if (agent?.busId !== undefined && agent.busId !== null && agent.busId !== agentId) {
      return (await repo.findRootAgentGithubInfo(agent.busId))?.githubRepoRef ?? null;
    }
  } catch (err) {
    log.error('Failed to look up agent repo ref', {
      error: err instanceof Error ? err.message : String(err),
    });
  }
  return null;
}

/**
 * Project restore, unless a canonical repo ref (spawn's github_repo_ref or agent override)
 * replaces it; then the repo's connected folder restores (its startup script, env files).
 */
async function sessionWorkspace(params: CreateSessionParams, userId: string) {
  return resolveCodeSessionWorkspace(userId, {
    githubRepoRef: params.github_repo_ref,
    agentRepoRef: await resolveAgentRepoRef(params.agent_id),
    projectId: await resolveAgentProjectId(params.agent_id),
    requestedRepoUrl: params.github_repo_url,
  });
}

/** The spawn's Codex launch options (harness, project, CLI conversation, title, size). */
function launchOptions(
  params: CreateSessionParams,
  projectWorkspace: CodexCodeSessionOptions['projectWorkspace'],
  sandboxSize: SandboxSize | null
): CodexCodeSessionOptions {
  return {
    harnessMode: params.harness_mode,
    roseApiBaseUrl: params.rose_api_base_url,
    memoryEnabled: params.memory_enabled,
    freshConversation: params.fresh_conversation === true,
    projectWorkspace,
    taskName: params.task_name ?? null,
    cliSettings: params.cli_settings ?? null,
    resumeConfig: params.resume_config,
    leaseOwner: params.lease_owner,
    sandboxSize,
  };
}

export async function createSession(params: CreateSessionParams) {
  const { agent_id, goal, attachments, user_id } = params;
  if (!user_id) throw new MissingUserIdError();

  let effectiveGithubRepoUrl = params.github_repo_url;
  let effectiveUserId = user_id;
  // Before any repo or sandbox work: an unsatisfiable size fails the spawn with 400.
  const sandboxSize = await requestedSandboxSize('codex', params.sandbox_size, user_id);

  // A project session restores the project's checkout; never also clone a repo.
  const workspace = await sessionWorkspace(params, effectiveUserId);
  const { projectWorkspace, repoOverrideUrl } = workspace;
  effectiveGithubRepoUrl = repoOverrideUrl ?? effectiveGithubRepoUrl;
  if (projectWorkspace !== null) effectiveGithubRepoUrl = undefined;

  if (
    projectWorkspace === null &&
    (effectiveGithubRepoUrl === undefined || effectiveGithubRepoUrl === '')
  ) {
    const resolved = await resolveGithubRepo(agent_id, effectiveUserId);
    if (resolved.repoUrl !== undefined) {
      effectiveGithubRepoUrl = resolved.repoUrl;
      effectiveUserId = resolved.userId;
      log.info(`Auto-injecting GitHub repo from agent tree: ${effectiveGithubRepoUrl}`);
    }
  }

  let normalizedRepoUrl: string | undefined;
  if (effectiveGithubRepoUrl !== undefined && effectiveGithubRepoUrl !== '') {
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
  }

  // A project session is scoped (and branched) by its project, as for Claude.
  const sessionRepoUrl =
    projectWorkspace !== null
      ? (projectWorkspace.project.repoUrl ?? `rose-project:${projectWorkspace.project.id}`)
      : (normalizedRepoUrl ?? null);
  // Codex never uses the DEMO / Fast Mode cache: the agent's speed only
  // selects Codex's native service tier.
  const codexSpeed = codexSpeedFor(await repo.findAgentSpeed(agent_id));
  const { session, fields } = await createWithCodeSession(
    {
      agentId: agent_id,
      userId: effectiveUserId,
      repoUrl: sessionRepoUrl,
      codeProvider: 'codex',
      codeSessionId: params.code_session_id,
      leaseOwner: params.lease_owner,
      baseBranch: params.base_branch ?? projectWorkspace?.project.defaultBranch ?? null,
    },
    (codeSession, handoff) =>
      createCodexCodeSession(
        agent_id,
        handoff !== null ? `${handoff}\n\n${goal}` : goal,
        attachments,
        normalizedRepoUrl,
        effectiveUserId,
        codexSpeed,
        launchOptions(params, projectWorkspace, sandboxSize),
        codeSession
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
    // Live dev-server preview URL (Rose dev accounts, A1); null otherwise.
    preview_url: session.previewUrl ?? null,
  };
}

export async function closeSession(params: CloseSessionParams) {
  const result = await closeClaudeCodeSession(
    params.task_id,
    params.user_id,
    params.agent_id,
    params.additional_paths,
    { reason: params.reason }
  );

  return {
    status: 'CLOSED' as const,
    output_files: result.outputFiles,
    usage: result.usage,
    code_session: result.codeSession ?? null,
    ...(result.handoff !== undefined ? { handoff: result.handoff } : {}),
  };
}

export async function getSession(taskId: string) {
  const session = await getClaudeCodeSession(taskId);
  if (!session) return null;
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

export async function getAgentSessions(agentId: string) {
  return repo.findSessionsByAgent(agentId);
}

export async function persistOutput(
  taskId: string,
  output: string,
  credentialFeedback?: CodexCredentialFeedback[]
) {
  log.info(`Persisting Codex output for session ${taskId} (${output.length} bytes)`);
  await repo.persistSessionOutput(taskId, output);
  await detectAndMarkCodexExhaustion(taskId, output, credentialFeedback);
}
