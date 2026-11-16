import { z } from 'zod';
import type { FilePointer } from '../../providers/storage';
import { codeCliSettingsSchema } from '../../providers/codeCliSettings';
import {
  codeAgentResumeConfigSchema,
  validateCodeResumeLease,
  type CodeAgentResumeConfig,
} from '../../providers/codeAgentResumeConfig';
import {
  refuseLocalSandboxSize,
  sandboxSizeRequestSchema,
  type SandboxSizeRequest,
} from '../../providers/codeSessions/sandboxSize';

export const createSessionSchema = z
  .object({
    agent_id: z.string().uuid(),
    goal: z.string().min(1),
    attachments: z.array(z.any()).optional(),
    github_repo_url: z.string().optional(),
    // Connected repo by canonical ref (`gh:owner/repo`, URL or legacy project id). When it names
    // a different repo than the agent's project, the repo's own connected folder restores
    // instead (its startup script, env files). 403 when not connected.
    github_repo_ref: z.string().trim().min(1).max(300).optional(),
    user_id: z.string().uuid(),
    fast_mode_cache_enabled: z.boolean().optional(),
    memory_enabled: z.boolean().optional(),
    harness_mode: z.enum(['code', 'rose']).optional(),
    rose_api_base_url: z.string().url().optional(),
    code_session_id: z.string().uuid().optional(),
    // Resume this session's sandbox and files in a new CLI conversation (paused-task resume).
    fresh_conversation: z.boolean().optional(),
    lease_owner: z.string().min(1).max(200).optional(),
    base_branch: z.string().min(1).max(255).optional(),
    // C14: 'local' starts the session on the agent tree's Mac (Rose Mac app).
    target: z.enum(['cloud', 'local']).optional(),
    device_id: z.string().uuid().optional(),
    // The spawn's task_name; the agents list shows it as the code agent's title.
    task_name: z.string().min(1).max(200).nullable().optional(),
    // A resuming turn's model / effort / speed: the first resumed turn runs on them.
    cli_settings: codeCliSettingsSchema.optional(),
    resume_config: codeAgentResumeConfigSchema.optional(),
    // Root-selected initial size of a NEW session's sandbox; a resumed session keeps its size.
    sandbox_size: sandboxSizeRequestSchema.optional(),
  })
  .superRefine(validateCodeResumeLease)
  .superRefine(refuseLocalSandboxSize);

export const closeSessionSchema = z.object({
  task_id: z.string().uuid(),
  user_id: z.string().uuid().optional(),
  agent_id: z.string().uuid().optional(),
  additional_paths: z.array(z.string()).optional(),
  reason: z.string().min(1).max(64).optional(),
});

export interface CreateSessionParams {
  agent_id: string;
  goal: string;
  attachments?: FilePointer[];
  github_repo_url?: string;
  github_repo_ref?: string;
  user_id: string;
  fast_mode_cache_enabled?: boolean;
  memory_enabled?: boolean;
  harness_mode?: 'code' | 'rose';
  rose_api_base_url?: string;
  code_session_id?: string;
  fresh_conversation?: boolean;
  lease_owner?: string;
  base_branch?: string;
  /** The spawn's task_name (the code agent's title in the agents list). */
  task_name?: string | null;
  /** A resuming turn's model / effort / speed for the first resumed turn. */
  cli_settings?: z.infer<typeof codeCliSettingsSchema>;
  resume_config?: CodeAgentResumeConfig;
  /** Root-selected initial sandbox size (a new session only). */
  sandbox_size?: SandboxSizeRequest;
}

export interface CloseSessionParams {
  task_id: string;
  user_id?: string;
  agent_id?: string;
  additional_paths?: string[];
  reason?: string;
}

/** POST /:taskId/usage-limit — Codex stopped on a provider usage limit (C15). */
export const codeTaskUsageLimitSchema = z.object({
  // Epoch seconds/ms or ISO time the provider reports the limit resets.
  resets_at: z.union([z.number().positive(), z.string().min(1).max(64)]).optional(),
});

/** POST /:taskId/resize — the attached activity applies a pending size change (C16). */
export const codeTaskResizeSchema = z.object({ request_id: z.string().uuid() }).strict();
