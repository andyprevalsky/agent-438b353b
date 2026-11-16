/**
 * The real E2B bindings of the template sync runner: which template the
 * template sandbox boots from, how it is created, and how snapshots are
 * deleted. Kept apart from maintenanceRunner.ts so tests drive the same
 * runner with a fake sandbox (no network, no spend).
 */
import { Sandbox } from 'e2b';
import { createSandboxWithRetry } from '../e2bSandboxCreate';
import { CLAUDE_CODE_TEMPLATE, DEFAULT_CLAUDE_CODE_TEMPLATE_ID } from '../claudeCodeTemplate';
import {
  closestAvailableSize,
  newSessionDefaultSize,
  parseSandboxSize,
  sandboxTemplateFor,
  type SandboxSize,
} from '../codeSessions/sandboxSize';
import type { BaseTemplate, MaintenanceDeps } from './maintenanceRunner';
import type { MaintenanceProject } from './maintenanceStore';
import { E2B_HOME } from './workspace';

/**
 * A dependency install of the Rose repo was OOM-killed on the 1 GB default
 * template (measured 2026-10-28), so the template sandbox wants at least this.
 */
export const MIN_TEMPLATE_SANDBOX: SandboxSize = { cpu: 2, memoryGb: 4 };

/**
 * The project's chosen size, else the new-session default (so the snapshot a
 * new session starts from has the size it would get anyway), never below
 * MIN_TEMPLATE_SANDBOX.
 */
export function wantedTemplateSize(project: Pick<MaintenanceProject, 'sandboxSize'>): SandboxSize {
  const fallback = newSessionDefaultSize('claude');
  const chosen = parseSandboxSize(project.sandboxSize, fallback) ?? fallback;
  return {
    cpu: Math.max(MIN_TEMPLATE_SANDBOX.cpu, chosen.cpu),
    memoryGb: Math.max(MIN_TEMPLATE_SANDBOX.memoryGb, chosen.memoryGb),
  };
}

/** Claude template (the snapshot is of a Claude code sandbox) of the closest built size. */
export function e2bBaseTemplate(project: MaintenanceProject): BaseTemplate {
  const wanted = wantedTemplateSize(project);
  const size = closestAvailableSize('claude', wanted);
  const templateId = size !== null ? sandboxTemplateFor('claude', size) : null;
  if (size !== null && templateId !== null) return { templateId, size };
  return {
    templateId: DEFAULT_CLAUDE_CODE_TEMPLATE_ID,
    size: { ...CLAUDE_CODE_TEMPLATE.size },
    warning: `no sandbox template of ${wanted.cpu} vCPU / ${wanted.memoryGb} GB is built; using the default template, which may be too small for a dependency install`,
  };
}

export function e2bMaintenanceDeps(): MaintenanceDeps {
  const apiKey = process.env.E2B_API_KEY;
  if (!apiKey) throw new Error('E2B_API_KEY not configured');
  return {
    homeDir: E2B_HOME,
    now: () => new Date(),
    baseTemplate: e2bBaseTemplate,
    createSandbox: (templateOrSnapshotId, opts) =>
      createSandboxWithRetry(
        templateOrSnapshotId,
        { apiKey, timeoutMs: opts.timeoutMs, metadata: opts.metadata },
        'Project sync'
      ),
    setSandboxTimeout: (sandboxId, timeoutMs) =>
      Sandbox.setTimeout(sandboxId, timeoutMs, { apiKey }),
    deleteSnapshot: (snapshotId) => Sandbox.deleteSnapshot(snapshotId, { apiKey }),
  };
}
