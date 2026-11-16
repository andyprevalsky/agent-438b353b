import type { CodeCliSettingsOverride } from './codeAgentModels';
import type { CodeAgentResumeConfig } from './codeAgentResumeConfig';
import type { SandboxSize } from './codeSessions/sandboxSize';
import type { CodeSessionRow } from './codeSessions/store';
import type { ProjectWorkspace } from './projects/types';

interface CodeSessionLaunchOptions {
  /** Project restored into the session before the CLI starts. */
  projectWorkspace?: ProjectWorkspace | null;
  /** Keep files but start a new CLI conversation. */
  freshConversation?: boolean;
  taskName?: string | null;
  /** Explicit new-turn settings, merged by the caller over recorded settings. */
  cliSettings?: CodeCliSettingsOverride | null;
  resumeConfig?: CodeAgentResumeConfig;
  leaseOwner?: string;
  /** The spawn's explicit, validated sandbox size; applies to a new session only. */
  sandboxSize?: SandboxSize | null;
}

export interface ClaudeCodeSessionOptions extends CodeSessionLaunchOptions {
  codeSession?: CodeSessionRow | null;
}

export interface CodexCodeSessionOptions extends CodeSessionLaunchOptions {
  harnessMode?: 'code' | 'rose';
  roseApiBaseUrl?: string;
  memoryEnabled?: boolean;
}
