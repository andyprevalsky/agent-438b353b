import type { CodexServiceSpeed } from './codexSpeed';
import { prisma } from './database';
import { createLogger } from './logger';
import { buildFastModeAnthropicEnv, resolveFastModeWriteEnabled } from './codeAgentFastModeEnv';
import type { FilePointer } from './storage';
import { v4 as uuidv4 } from 'uuid';
import {
  fetchClaudeCodeCredentialsWithMeta,
  type ClaudeCodeCredentialResult,
} from './claudeCredentials';
import { isAnthropicApiKeyOnCooldown } from './computeCredentials';
import { buildCodeAgentSystemPrompt, buildCodexHarnessSystemPrompt } from './codeAgentPrompt';
import { uploadAttachmentsToSandbox } from './e2bAttachments';
import { overlayCodexRoseBridge } from './e2bCodexHarness';
import { workDirForGithubRepo, WS_SERVER_BACKGROUND } from './e2bWorkDir';
import { codeSandboxGithubEnvs } from './codeAgentGitIdentity';
import {
  fileHash,
  findLocalClaudeCodeWsServer,
  overlayClaudeCodeWsServer,
  overlayCodexCodeWsServer,
} from './e2bWsServerOverlay';
import {
  acquireCodeSandbox,
  attachLiveCodeCli,
  CLAUDE_CONFIG_ENV_KEYS,
  CLAUDE_CREDENTIAL_ENV_KEYS,
  codeSessionBranchEnvs,
  codeSandboxUserId,
  codeSessionTaskFields,
  type CodeAgentSessionResult,
  createLocalCodeSessionFallback,
  recordE2bCodeSessionSandbox,
  releaseSandboxAfterFailedRestore,
  releaseSandboxAfterFailedStart,
  stopStaleWsServer,
} from './e2bCodeSessionSandbox';
import { type CodeSessionRow } from './codeSessions/store';
import { resolveCodexSpawnAuth } from './codeSessions/usageLimit';
import { restoreSandboxProject } from './projects/workspace';
import {
  waitForSandboxWsServer,
  wsReadinessOutcome,
  codeSessionWsReadyOptions,
} from './e2bWsReady';
import { codeSessionTurnToken } from './codeSessionTurns';
import { applyUserUiSandboxEnvs, writeUserUiWorkspaceToSandboxSafe } from './userUi/workspace';
import { prepareCodeSessionWorkspace, promptFlagsForUser } from './codeSessionWorkspace';
import { applyPreviewUrlEnv, autostartRosePreview } from './codePreview/sandbox';
import { isRoseRepoUrl } from './codePreview/preview';
import {
  claudeTaskIntegrationId,
  isRefreshOnlyOauthEnvelope,
  resolveClaudeCodeAuthEnvs,
  resolveClaudeCodeSandboxApiKey,
  resolveClaudeCodeTokenSource,
  shouldEnableClaudeCodeFastMode,
  shouldIgnoreAnthropicApiKeyCooldownForSandbox,
  shouldInjectClaudeCodeFastModeProxy,
  shouldPreferSandboxAnthropicBaseUrlOverFastMode,
  shouldSkipSandboxAnthropicGatewayForOauth,
} from './claudeCodeSandboxAuth';
export { createSandboxWithRetry, isTransientSandboxCreateError } from './e2bSandboxCreate';
export type { CodeAgentSessionResult } from './e2bCodeSessionSandbox';
export {
  closeClaudeCodeSession,
  connectToSandbox,
  getClaudeCodeSession,
  getClaudeCodeSessionBySandboxId,
  parseClaudeCodeUsage,
  type ClaudeCodeUsage,
} from './e2bCloseSession';
export {
  CODE_AGENT_CREDENTIALS_UNAVAILABLE,
  CodeAgentCredentialError,
  isRefreshOnlyOauthEnvelope,
  OPENROUTER_ANTHROPIC_COMPAT_BASE_URL,
  resolveClaudeCodeAuthEnvs,
  resolveClaudeCodeSandboxApiKey,
  resolveClaudeCodeTokenSource,
  resolveCodexCodeAuthEnvs,
  resolveHostAnthropicBaseUrl,
  shouldEnableClaudeCodeFastMode,
  shouldIgnoreAnthropicApiKeyCooldownForSandbox,
  shouldInjectClaudeCodeFastModeProxy,
  shouldPreferSandboxAnthropicBaseUrlOverFastMode,
  shouldSkipSandboxAnthropicGatewayForOauth,
} from './claudeCodeSandboxAuth';
import { DEFAULT_CLAUDE_CODE_TEMPLATE_ID as templateId } from './claudeCodeTemplate';
import { DEFAULT_CODEX_CODE_TEMPLATE_ID } from './codexTemplate';
import { codeAgentResumeCliEnvs } from './codeAgentResumeCli';
import { persistCodeResumeConfig } from './codeAgentResumeConfigStore';
import type { ClaudeCodeSessionOptions, CodexCodeSessionOptions } from './codeSessionLaunchOptions';
export type { ClaudeCodeSessionOptions } from './codeSessionLaunchOptions';

const log = createLogger('E2B');

// Validate required environment variables
if (!process.env.E2B_API_KEY) {
  log.warn('E2B_API_KEY environment variable not set - Claude Code sessions disabled');
}

const WS_PORT = 3000;
/**
 * Create a new E2B sandbox with Claude Code and WebSocket streaming.
 *
 * @param agentId - The agent creating this session
 * @param goal - The goal for Claude Code to accomplish
 * @param attachments - Optional file attachments to upload to sandbox
 * @param githubRepoUrl - Optional GitHub repo URL to clone before starting Claude
 * @returns Session info including WebSocket URL for terminal streaming
 */
export async function createClaudeCodeSession(
  agentId: string,
  goal: string,
  attachments?: FilePointer[],
  githubRepoUrl?: string,
  userId?: string,
  fastModeCacheEnabled?: boolean,
  options: ClaudeCodeSessionOptions = {}
): Promise<CodeAgentSessionResult> {
  // E2B reserve calls are serialized inside acquireCodeSandbox (create/resume
  // only). Taking the same non-reentrant slot here would deadlock that inner call.
  return createClaudeCodeSessionUnlocked(
    agentId,
    goal,
    attachments,
    githubRepoUrl,
    userId,
    fastModeCacheEnabled,
    options
  );
}

async function createClaudeCodeSessionUnlocked(
  agentId: string,
  goal: string,
  attachments: FilePointer[] | undefined,
  githubRepoUrl: string | undefined,
  userId: string | undefined,
  fastModeCacheEnabled: boolean | undefined,
  options: ClaudeCodeSessionOptions
): Promise<CodeAgentSessionResult> {
  const projectWorkspace = options.projectWorkspace ?? null;
  const codeSession = options.codeSession ?? null;
  codeSandboxUserId(codeSession, userId);
  log.info(`Creating Claude Code session for agent ${agentId}`);

  if (!process.env.E2B_API_KEY) {
    throw new Error('E2B environment variables not configured');
  }

  const taskId = uuidv4();
  const fastModeWriteEnabled = resolveFastModeWriteEnabled(fastModeCacheEnabled, userId);
  const codeAgentFastModeActive = shouldEnableClaudeCodeFastMode({
    openrouterApiKey: process.env.OPENROUTER_API_KEY,
    fastModeCacheEnabled,
    fastModeWriteEnabled,
  });
  if (
    !codeAgentFastModeActive &&
    (fastModeCacheEnabled === true || fastModeWriteEnabled) &&
    process.env.OPENROUTER_API_KEY?.trim()
  ) {
    log.info('OpenRouter configured — skipping Fast Mode tee-proxy/flags for Claude Code session');
  }
  // The spawning agent's tier and Fast Mode button pick model, effort and speed.
  const { envs: cliSettings } = await codeAgentResumeCliEnvs('claude', agentId, options);
  const workDir = workDirForGithubRepo(githubRepoUrl);
  const runtimeFingerprint = [
    `template:${templateId}`,
    `wrapper:${fileHash(findLocalClaudeCodeWsServer())}`,
    `env:${process.env.FAST_MODE_CODE_AGENT_ENV_FINGERPRINT ?? 'default'}`,
  ].join('|');
  // The Rose skills table of contents (agent graph, talking to the user, ...)
  // lists the skills prepareCodeSessionWorkspace installs for this account.
  const systemPrompt = buildCodeAgentSystemPrompt({
    githubRepoUrl,
    workDir,
    branch: codeSession?.branch,
    mergeTarget: codeSession?.mergeTarget,
    ...promptFlagsForUser(userId),
    roseRepo: isRoseRepoUrl(projectWorkspace?.project.repoUrl ?? githubRepoUrl),
  });

  // Build environment variables for sandbox
  const envs: Record<string, string> = {
    CLAUDE_CODE_GOAL: goal,
    ...cliSettings,
    CODE_AGENT_SYSTEM_PROMPT: systemPrompt,
    ...codeSessionBranchEnvs(codeSession),
    // Write capability for direct-chat user turns (codeSessionTurns.ts). The
    // ws URL the browser reads stays read-only without it.
    ...(process.env.BACKEND_API_KEY
      ? { CODE_SESSION_TURN_TOKEN: codeSessionTurnToken(taskId) }
      : {}),
  };

  // Try user's own credentials first, then rotate through the pool. The agent's
  // outer loop already rotates through pool tokens when Anthropic returns 429;
  // we mirror that here so the inner CLI doesn't keep replaying a known-
  // exhausted token.
  let credentialMeta: ClaudeCodeCredentialResult | null = null;
  if (userId) {
    credentialMeta = await fetchClaudeCodeCredentialsWithMeta(userId);
  }

  const sharedAnthropicApiKey = process.env.ANTHROPIC_API_KEY;
  // Peek shared cooldown once: drives both OpenRouter auto-fallback and whether
  // a refresh-only OAuth envelope may be skipped.
  const sharedKeyOnCooldown = sharedAnthropicApiKey ? await isAnthropicApiKeyOnCooldown() : false;
  // When Fast Mode is active, prefer OpenRouter over the tee-proxy even if the
  // shared Anthropic key is not on cooldown — fleet guests rarely set
  // ANTHROPIC_BASE_URL, and Fast Mode + Anthropic upstream 403s Claude Code.
  const sandboxTransport = resolveClaudeCodeSandboxApiKey(
    process.env.CLAUDE_CODE_ANTHROPIC_API_KEY,
    sharedAnthropicApiKey,
    {
      anthropicBaseUrl: process.env.ANTHROPIC_BASE_URL,
      openrouterApiKey: process.env.OPENROUTER_API_KEY,
      sharedKeyOnCooldown,
      preferOpenRouterGateway: codeAgentFastModeActive,
    }
  );
  const sandboxApiKey = sandboxTransport.apiKey;

  // Attachment fingerprints keep Fast Mode traces distinct across H-02 eras
  // (corrupted 369851-byte UTF-8 uploads vs intact 207364-byte JPEGs).
  const attachmentFingerprint =
    attachments && attachments.length > 0
      ? attachments
          .map((fp) => `${fp.original_filename}:${fp.size_bytes}:${fp.storage_path}`)
          .sort()
          .join('|')
      : '';

  // Fast Mode tee-proxy vs funded gateway: resolve transport first. OpenRouter
  // (or any non-api.anthropic.com base) must not be overwritten by Fast Mode —
  // the proxy forwards to api.anthropic.com and 403s gateway keys.
  const preferSandboxBase = shouldPreferSandboxAnthropicBaseUrlOverFastMode(
    sandboxTransport.baseUrl
  );
  // The DEMO-cache tee-proxy only when its opt-in gate is active and we are not
  // already on a funded non-Anthropic gateway; Andy write-through alone must not
  // inject it (k=5 t5 recurrence, 2026-08-21). It never sets provider speed:
  // CLAUDE_CODE_SPEED comes from the agent's speed (cliSettings).
  const injectFastModeProxy = shouldInjectClaudeCodeFastModeProxy({
    codeAgentFastModeActive,
    preferSandboxBase,
  });
  const fastModeEnv: Record<string, string> = injectFastModeProxy
    ? buildFastModeAnthropicEnv(
        goal,
        fastModeCacheEnabled === true,
        process.env.FAST_MODE_ANTHROPIC_PROXY_URL,
        fastModeWriteEnabled,
        {
          sessionId: taskId,
          trace: {
            goal,
            systemPrompt,
            model: cliSettings.CLAUDE_CODE_MODEL,
            codeProvider: 'claude',
            templateId,
            runtimeFingerprint: attachmentFingerprint
              ? `${runtimeFingerprint}|att:${attachmentFingerprint}`
              : runtimeFingerprint,
            workDir,
            githubRepoUrl,
          },
        }
      )
    : {};
  const sandboxBaseUrlCandidate = sandboxTransport.baseUrl;
  const ignoreSharedCooldown = shouldIgnoreAnthropicApiKeyCooldownForSandbox({
    sandboxApiKey,
    sharedAnthropicApiKey,
    sandboxBaseUrl: sandboxBaseUrlCandidate,
  });
  // A refreshable OAuth envelope after direct-token auth_error must not block the
  // funded API-key path when the envelope's refresh_token is also dead
  // (invalid_grant): Claude Code would 401 in-sandbox instead of using the key.
  // Resolve this BEFORE choosing ANTHROPIC_BASE_URL so a nulled envelope can
  // still take the OpenRouter gateway, while a live oauth_token never does.
  if (credentialMeta && isRefreshOnlyOauthEnvelope(credentialMeta) && sandboxApiKey) {
    // Dedicated gateway keys / ANTHROPIC_BASE_URL overrides are never gated by
    // the shared Anthropic-org cooldown (that cooldown tracks api.anthropic.com).
    const envelopeBlockedBySharedCooldown = !ignoreSharedCooldown && sharedKeyOnCooldown;
    if (!envelopeBlockedBySharedCooldown) {
      log.info(
        `Preferring sandbox Anthropic API key over refresh-only OAuth envelope for agent ${agentId}`
      );
      credentialMeta = null;
    }
  }

  // OAuth sessions must hit api.anthropic.com. OpenRouter + oauth_token produced
  // "401 Missing Authentication header" on Pre-Deploy claude-code-agent (2026-08-21).
  const skipGatewayForOauth = shouldSkipSandboxAnthropicGatewayForOauth(credentialMeta);
  if (skipGatewayForOauth) {
    log.info(
      `Using Claude OAuth for agent ${agentId} — Anthropic default host (sandbox gateway base skipped)`
    );
  } else if (preferSandboxBase && sandboxTransport.baseUrl) {
    envs.ANTHROPIC_BASE_URL = sandboxTransport.baseUrl;
    log.info(
      `Routing sandbox Anthropic calls via funded gateway ${envs.ANTHROPIC_BASE_URL} (Fast Mode proxy skipped)`
    );
  } else if (fastModeEnv.ANTHROPIC_BASE_URL !== undefined) {
    envs.ANTHROPIC_BASE_URL = fastModeEnv.ANTHROPIC_BASE_URL;
    log.info('Fast Mode: routing sandbox Anthropic calls via the tee-proxy');
  } else if (sandboxTransport.baseUrl) {
    envs.ANTHROPIC_BASE_URL = sandboxTransport.baseUrl;
    log.info(`Routing sandbox Anthropic calls via ${envs.ANTHROPIC_BASE_URL}`);
  } else if (fastModeWriteEnabled) {
    log.warn(
      'Fast Mode cache active but FAST_MODE_ANTHROPIC_PROXY_URL unset — code agent not cached'
    );
  }
  // Only consult the shared Anthropic-org cooldown when the sandbox would use
  // that same shared key against api.anthropic.com.
  const apiKeyOnCooldown =
    !credentialMeta && sandboxApiKey && !ignoreSharedCooldown ? sharedKeyOnCooldown : false;
  Object.assign(envs, resolveClaudeCodeAuthEnvs(credentialMeta, sandboxApiKey, apiKeyOnCooldown));
  const claudeTokenSource = resolveClaudeCodeTokenSource(credentialMeta);

  if (credentialMeta) {
    log.info(
      `Using ${credentialMeta.source} Claude Code credentials for agent ${agentId} (integration ${credentialMeta.poolTokenId ?? 'user-owned'}, mode ${credentialMeta.credentialMode})`
    );
  } else {
    log.info(
      `Using ${ignoreSharedCooldown ? 'dedicated Claude Code' : 'ANTHROPIC_API_KEY'} funded fallback for agent ${agentId}`
    );
  }

  // Add GitHub config if repo specified
  // Clone/push credentials, and the pushing account as the commit identity
  // so GitHub links each commit and Vercel deploys it (codeAgentGitIdentity.ts).
  Object.assign(
    envs,
    await codeSandboxGithubEnvs(githubRepoUrl, projectWorkspace?.project.repoUrl, '', userId)
  );

  const userUiWorkspace = await prepareCodeSessionWorkspace(userId, agentId, taskId, 'claude');
  applyUserUiSandboxEnvs(envs, userUiWorkspace);

  // Create sandbox first (before DB record) to avoid placeholder conflicts.
  // When E2B team billing blocks creates, fall back to a same-host local
  // Claude Code ws-server so code-agent benches can still make progress.
  // Before acquisition can start a CLI (including the billing local fallback).
  await persistCodeResumeConfig({
    agentId,
    taskId,
    provider: 'claude',
    envs,
    resumeConfig: options.resumeConfig,
    leaseOwner: options.leaseOwner ?? codeSession?.leaseOwner,
    execution: {
      github_repo_link: githubRepoUrl ?? projectWorkspace?.project.repoUrl ?? null,
      target: 'cloud',
      code_session_id: codeSession?.id ?? null,
    },
  });
  let acquired;
  try {
    acquired = await acquireCodeSandbox(codeSession, templateId, envs, 'Claude Code', {
      provider: 'claude',
      userId,
      project: projectWorkspace?.project,
      requestedSize: options.sandboxSize ?? null,
      agentId,
    });
  } catch (err) {
    const local = await createLocalCodeSessionFallback(err, codeSession, {
      taskId,
      agentId,
      goal,
      envs,
      attachments,
      userUiWorkspace,
      projectWorkspace,
      claudeTokenSource,
      claudeTokenIntegrationId: claudeTaskIntegrationId(credentialMeta),
      taskName: options.taskName ?? null,
    });
    if (local) return local;
    throw err;
  }
  const { sandbox, resumedFrom, cliSessionId } = acquired;
  if (cliSessionId !== null && options.freshConversation !== true)
    envs.CLAUDE_CODE_RESUME_SESSION_ID = cliSessionId;

  log.info(`Using sandbox: ${sandbox.sandboxId} (${resumedFrom})`);
  await overlayClaudeCodeWsServer(sandbox);
  await writeUserUiWorkspaceToSandboxSafe(sandbox, userUiWorkspace, agentId);
  // Before the goal reaches a live CLI, which may read the files it names.
  const uploadedFiles = await uploadAttachmentsToSandbox(sandbox, attachments, 'Claude Code');

  // Get WebSocket URL for the frontend to connect to
  const wsHost = sandbox.getHost(WS_PORT);
  const wsUrl = `wss://${wsHost}`;
  const previewUrl = applyPreviewUrlEnv(envs, wsUrl, sandbox.sandboxId);
  const taskRow = {
    id: taskId,
    agentId,
    sandboxId: sandbox.sandboxId,
    streamUrl: wsUrl,
    goal,
    taskName: options.taskName ?? null,
    status: 'ACTIVE' as const,
    claudeTokenSource,
    claudeTokenIntegrationId: claudeTaskIntegrationId(credentialMeta),
    ...codeSessionTaskFields(codeSession, 'claude'),
    projectId: projectWorkspace?.project.id ?? null,
  };
  if (resumedFrom === 'paused_sandbox' && options.freshConversation !== true) {
    // The paused CLI may still be running in the resumed sandbox: keep it
    // (its turn, tools and processes continue) and hand it this goal.
    const live = await attachLiveCodeCli(sandbox, codeSession, {
      provider: 'claude',
      wsUrl,
      taskId,
      text: `Task:\n${goal.replace(/\0/g, '')}`,
      envs,
      credentialKeys: CLAUDE_CREDENTIAL_ENV_KEYS,
      configKeys: CLAUDE_CONFIG_ENV_KEYS,
    });
    if (live.attached) {
      await prisma.claudeCodeSession.create({ data: taskRow });
      await recordE2bCodeSessionSandbox(codeSession, sandbox.sandboxId);
      log.info(`Session attached to the live CLI: ${taskId}, WS: ${wsUrl}`);
      return {
        wsUrl,
        taskId,
        sandboxId: sandbox.sandboxId,
        uploadedFiles,
        resumedFrom: 'live_cli',
        cliSessionId,
        projectRestore: null,
        previewUrl,
        ...acquired.reported,
        historyEvents: live.historyEvents ?? 0,
      };
    }
  } else if (resumedFrom === 'paused_sandbox') {
    await stopStaleWsServer(sandbox);
  }
  const project = await restoreSandboxProject(
    taskId,
    sandbox,
    projectWorkspace,
    acquired.fromSnapshot,
    (held) => releaseSandboxAfterFailedRestore(held, codeSession)
  );
  const sessionEnvs = { ...envs, ...(project.envs ?? {}) };
  const wsLaunchedAt = Date.now();

  // Start the WebSocket server (runs Claude Code internally)
  // timeoutMs: 0 = no timeout (default is 60s which was killing the process!)
  // envs: a resumed sandbox keeps its creation env; pass this spawn's explicitly.
  // Filesystem snapshots default commands to root; the coding harness requires
  // the user account and its home even when the source template defaulted to user.
  sandbox.commands
    .run('node /home/user/ws-server.mjs', {
      ...WS_SERVER_BACKGROUND,
      user: 'user',
      cwd: '/home/user',
      envs: { ...sessionEnvs, HOME: '/home/user' },
    })
    .catch((err) =>
      log.error('Failed to start WS server', {
        error: err instanceof Error ? err.message : String(err),
      })
    );

  // Wait for WS server to be ready
  log.info('Waiting for WebSocket server to start');
  const ws = await waitForSandboxWsServer(sandbox, codeSessionWsReadyOptions('claude'));
  const wsFailure = wsReadinessOutcome(log, 'claude', ws);
  if (wsFailure !== null) {
    await releaseSandboxAfterFailedStart(sandbox, codeSession);
    throw wsFailure;
  }
  // Rose-in-Rose (A1): a dev Rose-repo session serves its front-end with HMR.
  await autostartRosePreview(sandbox, {
    envs: sessionEnvs,
    repoUrl: projectWorkspace?.project.repoUrl ?? githubRepoUrl,
    launchedAt: wsLaunchedAt,
  });

  // Create DB record with real values (no placeholders)
  await prisma.claudeCodeSession.create({ data: taskRow });
  await recordE2bCodeSessionSandbox(codeSession, sandbox.sandboxId);

  log.info(`Session ready: ${taskId}, WS: ${wsUrl}`);

  return {
    wsUrl,
    taskId,
    sandboxId: sandbox.sandboxId,
    uploadedFiles,
    resumedFrom,
    cliSessionId,
    projectRestore: project.report,
    previewUrl,
    ...acquired.reported,
    startup: { wsReadyMs: ws.elapsedMs, cloneMs: ws.cloneMs },
  };
}

export async function createCodexCodeSession(
  agentId: string,
  goal: string,
  attachments?: FilePointer[],
  githubRepoUrl?: string,
  userId?: string,
  codexSpeed: CodexServiceSpeed = 'priority',
  harnessOptions?: CodexCodeSessionOptions,
  codeSession: CodeSessionRow | null = null
): Promise<CodeAgentSessionResult> {
  codeSandboxUserId(codeSession, userId);
  const projectWorkspace = harnessOptions?.projectWorkspace ?? null;
  log.info(`Creating Codex Code session for agent ${agentId}`);

  if (!process.env.E2B_API_KEY) {
    throw new Error('E2B Codex environment variables not configured');
  }
  if (!userId) {
    throw new Error('user_id is required to create a Codex Code session');
  }

  const taskId = uuidv4();
  const templateId = process.env.E2B_CODEX_CODE_TEMPLATE_ID || DEFAULT_CODEX_CODE_TEMPLATE_ID;
  const cli = await codeAgentResumeCliEnvs('codex', agentId, { ...harnessOptions, codeSession });
  const harnessMode = harnessOptions?.harnessMode ?? 'code';
  const roseApiBaseUrl = harnessOptions?.roseApiBaseUrl?.replace(/\/$/, '');
  const systemPrompt =
    harnessMode === 'rose' && roseApiBaseUrl
      ? buildCodexHarnessSystemPrompt({
          roseApiBaseUrl,
          roseUserId: userId,
          anchorAgentId: agentId,
        })
      : buildCodeAgentSystemPrompt({
          githubRepoUrl,
          workDir: workDirForGithubRepo(githubRepoUrl),
          branch: codeSession?.branch,
          mergeTarget: codeSession?.mergeTarget,
          ...promptFlagsForUser(userId),
          roseRepo: isRoseRepoUrl(projectWorkspace?.project.repoUrl ?? githubRepoUrl),
        });

  // User OAuth → shared pool → our funded Azure Codex deployment unless it is
  // backed off (the OpenAI GPT_API_KEY is not funded). Mirrors the Claude Code path.
  const {
    credentialMeta,
    azureFunding,
    envs: authEnvs,
    servedBy,
  } = await resolveCodexSpawnAuth(userId, new Date(), cli.settings.tier);

  const envs: Record<string, string> = {
    CODEX_GOAL: goal,
    ...authEnvs,
    ...cli.envs,
    CODE_AGENT_SYSTEM_PROMPT: systemPrompt,
    CODE_AGENT_HARNESS_MODE: harnessMode,
    ...codeSessionBranchEnvs(codeSession),
    // Write capability for inbound turns (agent-graph messages, direct chat);
    // same per-task token as the Claude Code session.
    ...(process.env.BACKEND_API_KEY
      ? { CODE_SESSION_TURN_TOKEN: codeSessionTurnToken(taskId) }
      : {}),
  };
  if (harnessMode === 'rose' && roseApiBaseUrl) {
    const apiKey = process.env.BACKEND_API_KEY ?? '';
    if (!apiKey) {
      throw new Error('BACKEND_API_KEY is required for Codex rose harness sessions');
    }
    envs.ROSE_API_BASE_URL = roseApiBaseUrl;
    envs.ROSE_API_KEY = apiKey;
    envs.ROSE_USER_ID = userId!;
    envs.ROSE_ANCHOR_AGENT_ID = agentId;
    // Benchmark runs are memory-off unless their catalog task explicitly opts
    // into the planner-memory A/B arm.
    envs.ROSE_MEMORY_ENABLED = harnessOptions?.memoryEnabled === true ? 'true' : 'false';
  }
  if (credentialMeta) {
    log.info(`Using ${credentialMeta.source} GPT Code credentials for agent ${agentId}`);
  } else {
    log.info(
      `Using Azure ${azureFunding?.deployment} on ${azureFunding ? new URL(azureFunding.baseUrl).host : 'unconfigured'} funded fallback for Codex Code agent ${agentId}`
    );
  }

  // Codex talks to its provider directly (never a Rose proxy or the DEMO cache);
  // the Fast Mode button only selects its native service tier.
  // A person's resuming turn may pick its speed (cli_settings): it wins over the
  // agent's stored speed, as its model and effort overrides already do.
  envs.CODEX_SPEED = cli.speedOverride ?? codexSpeed;
  log.info('Codex code-agent speed', { agentId, taskId, codexSpeed: envs.CODEX_SPEED });

  // Clone/push credentials, and the pushing account as the commit identity
  // so GitHub links each commit and Vercel deploys it (codeAgentGitIdentity.ts).
  Object.assign(
    envs,
    await codeSandboxGithubEnvs(githubRepoUrl, projectWorkspace?.project.repoUrl, 'Codex', userId)
  );

  const userUiWorkspace = await prepareCodeSessionWorkspace(userId, agentId, taskId, 'codex');
  applyUserUiSandboxEnvs(envs, userUiWorkspace);

  await persistCodeResumeConfig({
    agentId,
    taskId,
    provider: 'codex',
    envs,
    resumeConfig: harnessOptions?.resumeConfig,
    leaseOwner: harnessOptions?.leaseOwner ?? codeSession?.leaseOwner,
    execution: {
      github_repo_link: githubRepoUrl ?? projectWorkspace?.project.repoUrl ?? null,
      target: 'cloud',
      code_session_id: codeSession?.id ?? null,
    },
  });
  const acquired = await acquireCodeSandbox(codeSession, templateId, envs, 'Codex Code', {
    provider: 'codex',
    userId,
    project: projectWorkspace?.project,
    requestedSize: harnessOptions?.sandboxSize ?? null,
    agentId,
  });
  const { sandbox, resumedFrom, cliSessionId } = acquired;
  if (cliSessionId !== null && harnessOptions?.freshConversation !== true)
    envs.CODEX_RESUME_SESSION_ID = cliSessionId;

  log.info(`Using Codex sandbox: ${sandbox.sandboxId} (${resumedFrom})`);
  await overlayCodexCodeWsServer(sandbox);
  if (harnessMode === 'rose') {
    await overlayCodexRoseBridge(sandbox);
  }
  await writeUserUiWorkspaceToSandboxSafe(sandbox, userUiWorkspace, agentId);

  const uploadedFiles = await uploadAttachmentsToSandbox(sandbox, attachments, 'Codex');

  const wsHost = sandbox.getHost(WS_PORT);
  const wsUrl = `wss://${wsHost}`;
  const previewUrl = applyPreviewUrlEnv(envs, wsUrl, sandbox.sandboxId);
  const taskRow = {
    id: taskId,
    agentId,
    sandboxId: sandbox.sandboxId,
    streamUrl: wsUrl,
    goal,
    taskName: harnessOptions?.taskName ?? null,
    status: 'ACTIVE' as const,
    // What serves the run: 'user' | 'pool' login, or 'azure' (the funded deployment alone).
    claudeTokenSource: servedBy,
    claudeTokenIntegrationId: credentialMeta?.integrationId ?? credentialMeta?.poolTokenId ?? null,
    ...codeSessionTaskFields(codeSession, 'codex'),
    projectId: projectWorkspace?.project.id ?? null,
  };
  // A resumed sandbox may still run the paused Codex CLI: stop it before
  // starting this spawn's (live attach is Claude Code only: Codex binds the
  // handoff outputSchema to its first turn).
  if (resumedFrom === 'paused_sandbox') await stopStaleWsServer(sandbox);
  const project = await restoreSandboxProject(
    taskId,
    sandbox,
    projectWorkspace,
    acquired.fromSnapshot,
    (held) => releaseSandboxAfterFailedRestore(held, codeSession)
  );
  const sessionEnvs = { ...envs, ...(project.envs ?? {}) };
  const wsLaunchedAt = Date.now();

  // Filesystem snapshots do not preserve the base template's default command user.
  sandbox.commands
    .run('node /home/user/ws-server.mjs', {
      ...WS_SERVER_BACKGROUND,
      user: 'user',
      cwd: '/home/user',
      envs: { ...sessionEnvs, HOME: '/home/user' },
    })
    .catch((err) =>
      log.error('Failed to start Codex WS server', {
        error: err instanceof Error ? err.message : String(err),
      })
    );

  log.info('Waiting for Codex WebSocket server to start');
  const ws = await waitForSandboxWsServer(sandbox, codeSessionWsReadyOptions('codex'));
  const wsFailure = wsReadinessOutcome(log, 'codex', ws);
  if (wsFailure !== null) {
    await releaseSandboxAfterFailedStart(sandbox, codeSession);
    throw wsFailure;
  }
  await autostartRosePreview(sandbox, {
    envs: sessionEnvs,
    repoUrl: projectWorkspace?.project.repoUrl ?? githubRepoUrl,
    launchedAt: wsLaunchedAt,
  });

  await prisma.claudeCodeSession.create({ data: taskRow });
  await recordE2bCodeSessionSandbox(codeSession, sandbox.sandboxId);

  log.info(`Codex session ready: ${taskId}, WS: ${wsUrl}`);

  return {
    wsUrl,
    taskId,
    sandboxId: sandbox.sandboxId,
    uploadedFiles,
    resumedFrom,
    cliSessionId,
    projectRestore: project.report,
    previewUrl,
    ...acquired.reported,
    startup: { wsReadyMs: ws.elapsedMs, cloneMs: ws.cloneMs },
  };
}
