import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { parse as parseYaml } from 'yaml';
import { applyHermesSetup } from './hermes.js';

vi.mock('child_process', async () => {
  const actual = await vi.importActual<typeof import('child_process')>('child_process');
  return {
    ...actual,
    execSync: vi.fn(() => {
      throw new Error('mocked: hermes binary not available');
    }),
  };
});

describe('hermes setup handler', () => {
  let tmpDir: string;
  let originalHermesHome: string | undefined;
  let originalApiUrl: string | undefined;
  let originalApiKey: string | undefined;
  let originalEndpoint: string | undefined;
  let originalApiToken: string | undefined;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-handler-'));
    originalHermesHome = process.env.HERMES_HOME;
    originalApiUrl = process.env.AUTOMEM_API_URL;
    originalApiKey = process.env.AUTOMEM_API_KEY;
    originalEndpoint = process.env.AUTOMEM_ENDPOINT;
    originalApiToken = process.env.AUTOMEM_API_TOKEN;
    delete process.env.HERMES_HOME;
    delete process.env.AUTOMEM_API_URL;
    delete process.env.AUTOMEM_API_KEY;
    // Clear the legacy alias too — a developer shell exporting it would
    // otherwise win over recovered credentials and make these tests flaky.
    delete process.env.AUTOMEM_ENDPOINT;
    delete process.env.AUTOMEM_API_TOKEN;
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    const restore = (key: string, value: string | undefined) => {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    };
    restore('HERMES_HOME', originalHermesHome);
    restore('AUTOMEM_API_URL', originalApiUrl);
    restore('AUTOMEM_API_KEY', originalApiKey);
    restore('AUTOMEM_ENDPOINT', originalEndpoint);
    restore('AUTOMEM_API_TOKEN', originalApiToken);
    vi.clearAllMocks();
  });

  it('fresh install writes config.yaml and SOUL.md', async () => {
    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'http://127.0.0.1:8001',
      apiKey: 'sk-test',
      quiet: true,
    });

    const configPath = path.join(tmpDir, 'config.yaml');
    const soulPath = path.join(tmpDir, 'SOUL.md');

    expect(fs.existsSync(configPath)).toBe(true);
    expect(fs.existsSync(soulPath)).toBe(true);

    const parsed = parseYaml(fs.readFileSync(configPath, 'utf8')) as {
      mcp_servers: Record<
        string,
        {
          command: string;
          env: Record<string, string>;
          tools: { include: string[]; resources: boolean; prompts: boolean };
          sampling: { enabled: boolean };
        }
      >;
    };
    expect(parsed.mcp_servers.automem.command).toBe('npx');
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_URL).toBe('http://127.0.0.1:8001');
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_KEY).toBe('sk-test');
    expect(parsed.mcp_servers.automem.tools.include).toEqual([
      'recall_memory',
      'store_memory',
      'associate_memories',
      'update_memory',
      'check_database_health',
    ]);
    expect(parsed.mcp_servers.automem.tools.resources).toBe(false);
    expect(parsed.mcp_servers.automem.tools.prompts).toBe(false);
    expect(parsed.mcp_servers.automem.sampling.enabled).toBe(false);
    expect(parsed.mcp_servers.memory).toBeUndefined();

    const agents = fs.readFileSync(soulPath, 'utf8');
    expect(agents).toContain('<!-- BEGIN AUTOMEM HERMES RULES -->');
    expect(agents).toContain('<!-- END AUTOMEM HERMES RULES -->');
    expect(agents).toContain('MCP-only mode');
    expect(agents).toContain('mcp_automem_recall_memory');
    expect(agents).not.toMatch(/`automem_recall_memory`/);
    expect(agents).not.toContain('mcp__memory__recall_memory');
    expect(agents).toContain('test-project');
    // Hermes never loads <HERMES_HOME>/AGENTS.md, so the installer must not create it.
    expect(fs.existsSync(path.join(tmpDir, 'AGENTS.md'))).toBe(false);
  });

  it('re-running is idempotent (no second-run diff)', async () => {
    const opts = {
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'http://127.0.0.1:8001',
      apiKey: 'sk-test',
      quiet: true,
    };

    await applyHermesSetup(opts);
    const configPath = path.join(tmpDir, 'config.yaml');
    const soulPath = path.join(tmpDir, 'SOUL.md');
    const configAfterFirst = fs.readFileSync(configPath, 'utf8');
    const soulAfterFirst = fs.readFileSync(soulPath, 'utf8');

    await applyHermesSetup(opts);
    expect(fs.readFileSync(configPath, 'utf8')).toBe(configAfterFirst);
    expect(fs.readFileSync(soulPath, 'utf8')).toBe(soulAfterFirst);
  });

  it('--dry-run writes nothing', async () => {
    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'http://127.0.0.1:8001',
      apiKey: 'sk-test',
      dryRun: true,
      quiet: true,
    });

    expect(fs.existsSync(path.join(tmpDir, 'config.yaml'))).toBe(false);
    expect(fs.existsSync(path.join(tmpDir, 'SOUL.md'))).toBe(false);
  });

  it('HERMES_HOME env var resolves the install location', async () => {
    process.env.HERMES_HOME = tmpDir;
    await applyHermesSetup({
      projectName: 'test-project',
      endpoint: 'http://127.0.0.1:8001',
      quiet: true,
    });

    expect(fs.existsSync(path.join(tmpDir, 'config.yaml'))).toBe(true);
    expect(fs.existsSync(path.join(tmpDir, 'SOUL.md'))).toBe(true);
  });

  it('direct YAML path is exercised when Hermes CLI is unavailable', async () => {
    // The child_process mock at the top of this file makes every execSync throw,
    // which simulates a missing Hermes binary. A successful write to config.yaml
    // here means the YAML fallback was exercised.
    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'http://127.0.0.1:8001',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      mcp_servers: Record<string, unknown>;
    };
    expect(parsed.mcp_servers.automem).toBeDefined();
    expect(parsed.mcp_servers.memory).toBeUndefined();
  });

  it('honors $AUTOMEM_API_URL and $AUTOMEM_API_KEY when no flags are passed', async () => {
    process.env.AUTOMEM_API_URL = 'https://example.automem.test';
    process.env.AUTOMEM_API_KEY = 'sk-from-env';

    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      mcp_servers: Record<string, { env: Record<string, string> }>;
    };
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_URL).toBe('https://example.automem.test');
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_KEY).toBe('sk-from-env');
  });

  it('preserves installed credentials on a re-run with no flags or env', async () => {
    // First install with an explicit remote endpoint + key.
    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://remote.automem.test',
      apiKey: 'sk-remote',
      quiet: true,
    });

    // Re-run with zero flags. beforeEach already cleared AUTOMEM_API_URL/KEY,
    // so without preservation the endpoint would collapse to the local default
    // and the key would be dropped.
    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      mcp_servers: Record<string, { env: Record<string, string> }>;
    };
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_URL).toBe('https://remote.automem.test');
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_KEY).toBe('sk-remote');
  });

  it('explicit flags override previously installed credentials', async () => {
    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://old.automem.test',
      apiKey: 'sk-old',
      quiet: true,
    });

    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://new.automem.test',
      apiKey: 'sk-new',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      mcp_servers: Record<string, { env: Record<string, string> }>;
    };
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_URL).toBe('https://new.automem.test');
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_KEY).toBe('sk-new');
  });

  it('mcp re-run after a provider install recovers credentials from .env', async () => {
    // Provider mode writes credentials only to ~/.hermes/.env (no mcp_servers
    // entry), so the recovery path here exercises the .env reader.
    await applyHermesSetup({
      mode: 'provider',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://remote.automem.test',
      apiKey: 'sk-remote',
      quiet: true,
    });

    await applyHermesSetup({
      mode: 'mcp',
      targetDir: tmpDir,
      projectName: 'test-project',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      mcp_servers: Record<string, { env: Record<string, string> }>;
    };
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_URL).toBe('https://remote.automem.test');
    expect(parsed.mcp_servers.automem.env.AUTOMEM_API_KEY).toBe('sk-remote');
  });

  it('provider mode installs the Hermes memory provider without MCP tools', async () => {
    await applyHermesSetup({
      mode: 'provider',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      apiKey: 'sk-provider',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      memory: { provider: string };
      mcp_servers?: Record<string, unknown>;
    };
    expect(parsed.memory.provider).toBe('automem');
    expect(parsed.mcp_servers?.automem).toBeUndefined();

    const pluginRoot = path.join(tmpDir, 'plugins', 'automem');
    expect(fs.readFileSync(path.join(pluginRoot, '__init__.py'), 'utf8')).toContain(
      'class AutoMemMemoryProvider'
    );
    expect(fs.readFileSync(path.join(pluginRoot, 'plugin.yaml'), 'utf8')).toContain(
      'name: automem'
    );
    expect(fs.readFileSync(path.join(pluginRoot, 'plugin.yaml'), 'utf8')).toContain(
      'kind: exclusive'
    );
    expect(fs.readFileSync(path.join(pluginRoot, 'cli.py'), 'utf8')).toContain('def register_cli');
    expect(fs.readFileSync(path.join(pluginRoot, 'cli.py'), 'utf8')).toContain('doctor');
    expect(fs.readFileSync(path.join(pluginRoot, 'automem_policy.py'), 'utf8')).toContain(
      'PREFERENCE_RECALL_LIMIT = 5'
    );
    expect(fs.readFileSync(path.join(tmpDir, '.env'), 'utf8')).toContain(
      'AUTOMEM_API_URL=https://example.automem.test'
    );
    expect(fs.readFileSync(path.join(tmpDir, '.env'), 'utf8')).toContain(
      'AUTOMEM_API_KEY=sk-provider'
    );
    expect(fs.readFileSync(path.join(tmpDir, '.env'), 'utf8')).not.toContain(
      'AUTOMEM_HERMES_PROVIDER_TOOLS=false'
    );

    const agents = fs.readFileSync(path.join(tmpDir, 'SOUL.md'), 'utf8');
    expect(agents).toContain('Provider-only mode');
    expect(agents).toContain('automem_recall_memory');
    expect(agents).not.toContain('mcp_automem_recall_memory');
  });

  it('mcp mode removes AutoMem provider state from a prior provider install', async () => {
    await applyHermesSetup({
      mode: 'provider',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      apiKey: 'sk-provider',
      quiet: true,
    });

    await applyHermesSetup({
      mode: 'mcp',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      apiKey: 'sk-mcp',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      memory?: { provider?: string };
      mcp_servers: Record<string, unknown>;
    };
    expect(parsed.memory?.provider).not.toBe('automem');
    expect(parsed.mcp_servers.automem).toBeDefined();
    expect(parsed.mcp_servers.memory).toBeUndefined();
    expect(fs.readFileSync(path.join(tmpDir, '.env'), 'utf8')).not.toContain(
      'AUTOMEM_HERMES_PROVIDER_TOOLS=true'
    );
  });

  it('provider mode removes AutoMem MCP state from a prior MCP install', async () => {
    await applyHermesSetup({
      mode: 'mcp',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      apiKey: 'sk-mcp',
      quiet: true,
    });

    await applyHermesSetup({
      mode: 'provider',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      apiKey: 'sk-provider',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      memory: { provider: string };
      mcp_servers?: Record<string, unknown>;
    };
    expect(parsed.memory.provider).toBe('automem');
    expect(parsed.mcp_servers?.automem).toBeUndefined();
    expect(parsed.mcp_servers?.memory).toBeUndefined();
  });

  it('both mode installs the provider and MCP entry', async () => {
    await applyHermesSetup({
      mode: 'both',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'http://127.0.0.1:8001',
      quiet: true,
    });

    const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
      memory: { provider: string };
      mcp_servers: Record<string, unknown>;
    };
    expect(parsed.memory.provider).toBe('automem');
    expect(parsed.mcp_servers.automem).toBeDefined();
    expect(fs.readFileSync(path.join(tmpDir, '.env'), 'utf8')).toContain(
      'AUTOMEM_HERMES_PROVIDER_TOOLS=false'
    );
    expect(
      fs.readFileSync(path.join(tmpDir, 'plugins', 'automem', '__init__.py'), 'utf8')
    ).toContain('AUTOMEM_HERMES_PROVIDER_TOOLS');

    const agents = fs.readFileSync(path.join(tmpDir, 'SOUL.md'), 'utf8');
    expect(agents).toContain('Both mode');
    expect(agents).toContain('mcp_automem_recall_memory');
    expect(agents).toContain('provider explicit tools are disabled');
    expect(agents).not.toMatch(/`automem_recall_memory`/);
  });

  it('appends the rules to an existing SOUL.md and keeps the persona intact', async () => {
    const soulPath = path.join(tmpDir, 'SOUL.md');
    const persona = '# Persona\n\nYou are a terse assistant.\n';
    fs.writeFileSync(soulPath, persona);

    const opts = {
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    };
    await applyHermesSetup(opts);
    const afterFirst = fs.readFileSync(soulPath, 'utf8');
    expect(afterFirst.startsWith(persona)).toBe(true);
    expect(afterFirst.match(/BEGIN AUTOMEM HERMES RULES/g)).toHaveLength(1);

    // A mode switch replaces the block in place: one block, persona untouched.
    await applyHermesSetup({ ...opts, mode: 'provider' });
    const afterSwitch = fs.readFileSync(soulPath, 'utf8');
    expect(afterSwitch.startsWith(persona)).toBe(true);
    expect(afterSwitch.match(/BEGIN AUTOMEM HERMES RULES/g)).toHaveLength(1);
    expect(afterSwitch).toContain('Provider-only mode');
    expect(afterSwitch).not.toContain('MCP-only mode');

    await applyHermesSetup({ ...opts, mode: 'provider' });
    expect(fs.readFileSync(soulPath, 'utf8')).toBe(afterSwitch);
  });

  it('migrates legacy rules out of <HERMES_HOME>/AGENTS.md', async () => {
    const legacyPath = path.join(tmpDir, 'AGENTS.md');
    fs.writeFileSync(
      legacyPath,
      [
        '# Existing Hermes Notes',
        '',
        '<!-- BEGIN AUTOMEM HERMES RULES -->',
        'Old rules Hermes never loaded.',
        '<!-- END AUTOMEM HERMES RULES -->',
        '',
        '<!-- BEGIN AUTOMEM CODEX RULES -->',
        'Stale Codex guidance says use mcp__memory__recall_memory.',
        '<!-- END AUTOMEM CODEX RULES -->',
        '',
        'Keep this unrelated note.',
        '',
      ].join('\n')
    );

    await applyHermesSetup({
      mode: 'provider',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    });

    expect(fs.readFileSync(legacyPath, 'utf8')).toBe(
      '# Existing Hermes Notes\n\nKeep this unrelated note.\n'
    );
    const soul = fs.readFileSync(path.join(tmpDir, 'SOUL.md'), 'utf8');
    expect(soul).toContain('<!-- BEGIN AUTOMEM HERMES RULES -->');
    expect(soul).toContain('Provider-only mode');
    expect(soul).not.toContain('Old rules Hermes never loaded.');
  });

  it('removes a stale AutoMem Codex block from the legacy AGENTS.md on its own', async () => {
    const legacyPath = path.join(tmpDir, 'AGENTS.md');
    fs.writeFileSync(
      legacyPath,
      [
        '# Existing Hermes Notes',
        '',
        '<!-- BEGIN AUTOMEM CODEX RULES -->',
        'Stale Codex guidance says use mcp__memory__recall_memory.',
        '<!-- END AUTOMEM CODEX RULES -->',
        '',
      ].join('\n')
    );

    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    });

    expect(fs.readFileSync(legacyPath, 'utf8')).toBe('# Existing Hermes Notes\n');
  });

  it('deletes a legacy AGENTS.md that held only the AutoMem block', async () => {
    const legacyPath = path.join(tmpDir, 'AGENTS.md');
    fs.writeFileSync(
      legacyPath,
      '<!-- BEGIN AUTOMEM HERMES RULES -->\nold\n<!-- END AUTOMEM HERMES RULES -->\n'
    );

    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    });

    expect(fs.existsSync(legacyPath)).toBe(false);
    // The removed file is recoverable from its backup.
    expect(fs.readFileSync(`${legacyPath}.bak`, 'utf8')).toContain('old');
    expect(fs.readFileSync(path.join(tmpDir, 'SOUL.md'), 'utf8')).toContain(
      'BEGIN AUTOMEM HERMES RULES'
    );
  });

  it('leaves a legacy AGENTS.md with stray markers untouched', async () => {
    const legacyPath = path.join(tmpDir, 'AGENTS.md');
    const handWritten = '# Notes\n<!-- BEGIN AUTOMEM HERMES RULES -->\nhalf a block\n';
    fs.writeFileSync(legacyPath, handWritten);

    await applyHermesSetup({
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    });

    expect(fs.readFileSync(legacyPath, 'utf8')).toBe(handWritten);
    expect(fs.readFileSync(path.join(tmpDir, 'SOUL.md'), 'utf8')).toContain(
      'BEGIN AUTOMEM HERMES RULES'
    );
  });

  it('--rules pointing at the legacy AGENTS.md still writes there', async () => {
    const legacyPath = path.join(tmpDir, 'AGENTS.md');
    await applyHermesSetup({
      targetDir: tmpDir,
      rulesPath: legacyPath,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    });

    expect(fs.readFileSync(legacyPath, 'utf8')).toContain('BEGIN AUTOMEM HERMES RULES');
    expect(fs.existsSync(path.join(tmpDir, 'SOUL.md'))).toBe(false);
  });

  // Hermes carried the same defect the Grok PR filed three times: an inherited key was
  // written for whatever endpoint this run chose, with no check that the key was issued
  // for it. No finding named Hermes — it was found by sweeping the callers.
  describe('credential/endpoint pairing', () => {
    const readEntryEnv = (): Record<string, string> => {
      const parsed = parseYaml(fs.readFileSync(path.join(tmpDir, 'config.yaml'), 'utf8')) as {
        mcp_servers: { automem: { env: Record<string, string> } };
      };
      return parsed.mcp_servers.automem.env;
    };

    it('does not carry a shell key exported for a different endpoint', async () => {
      process.env.AUTOMEM_API_URL = 'https://shell.example.test';
      process.env.AUTOMEM_API_KEY = 'sk-shell';

      await applyHermesSetup({
        targetDir: tmpDir,
        endpoint: 'https://chosen.example.test',
        quiet: true,
        projectName: 'demo',
      });

      // Blank rather than absent — see buildAutoMemServerEntry.
      expect(readEntryEnv().AUTOMEM_API_KEY).toBe('');
    });

    it('reuses a shell key exported for the chosen endpoint', async () => {
      process.env.AUTOMEM_API_URL = 'https://chosen.example.test';
      process.env.AUTOMEM_API_KEY = 'sk-shell';

      await applyHermesSetup({ targetDir: tmpDir, quiet: true, projectName: 'demo' });

      expect(readEntryEnv().AUTOMEM_API_KEY).toBe('sk-shell');
    });

    it('does not carry the installed key to a different endpoint on re-run', async () => {
      await applyHermesSetup({
        targetDir: tmpDir,
        endpoint: 'https://first.example.test',
        apiKey: 'sk-first',
        quiet: true,
        projectName: 'demo',
      });

      await applyHermesSetup({
        targetDir: tmpDir,
        endpoint: 'https://second.example.test',
        quiet: true,
        projectName: 'demo',
      });

      const env = readEntryEnv();
      expect(env.AUTOMEM_API_URL).toBe('https://second.example.test');
      // Blank rather than absent — see buildAutoMemServerEntry.
      expect(env.AUTOMEM_API_KEY).toBe('');
      expect(env.AUTOMEM_API_TOKEN).toBe('');
    });

    it('preserves the installed key on a flagless re-run at the same endpoint', async () => {
      await applyHermesSetup({
        targetDir: tmpDir,
        endpoint: 'https://same.example.test',
        apiKey: 'sk-keep',
        quiet: true,
        projectName: 'demo',
      });

      await applyHermesSetup({ targetDir: tmpDir, quiet: true, projectName: 'demo' });

      const env = readEntryEnv();
      expect(env.AUTOMEM_API_URL).toBe('https://same.example.test');
      expect(env.AUTOMEM_API_KEY).toBe('sk-keep');
    });

    // mergeHermesEnvFile drops undefined updates so unrelated settings survive a
    // partial write — which meant an unresolved key left the previous one in place
    // while AUTOMEM_API_URL moved to the new host.
    it('removes the provider .env key when the endpoint changes', async () => {
      const envPath = path.join(tmpDir, '.env');
      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        endpoint: 'https://host-a.example.test',
        apiKey: 'sk-host-a',
        quiet: true,
        projectName: 'demo',
      });
      expect(fs.readFileSync(envPath, 'utf8')).toContain('sk-host-a');

      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        endpoint: 'https://host-b.example.test',
        quiet: true,
        projectName: 'demo',
      });

      const written = fs.readFileSync(envPath, 'utf8');
      expect(written).toContain('AUTOMEM_API_URL=https://host-b.example.test');
      expect(written).not.toContain('sk-host-a');
      expect(written).not.toMatch(/^AUTOMEM_API_KEY=/m);
      expect(written).not.toMatch(/^AUTOMEM_API_TOKEN=/m);
    });

    it('keeps the provider .env key on a flagless re-run at the same endpoint', async () => {
      const envPath = path.join(tmpDir, '.env');
      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        endpoint: 'https://same.example.test',
        apiKey: 'sk-keep',
        quiet: true,
        projectName: 'demo',
      });

      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        quiet: true,
        projectName: 'demo',
      });

      const written = fs.readFileSync(envPath, 'utf8');
      expect(written).toContain('AUTOMEM_API_KEY=sk-keep');
      expect(written).toContain('AUTOMEM_API_URL=https://same.example.test');
    });

    it('removes an export-prefixed provider key when the endpoint changes', async () => {
      const envPath = path.join(tmpDir, '.env');
      fs.mkdirSync(tmpDir, { recursive: true });
      fs.writeFileSync(
        envPath,
        'export AUTOMEM_API_URL=https://host-a.example.test\nexport AUTOMEM_API_KEY=sk-old\nKEEP=1\n'
      );

      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        endpoint: 'https://host-b.example.test',
        quiet: true,
        projectName: 'demo',
      });

      const written = fs.readFileSync(envPath, 'utf8');
      expect(written).not.toContain('sk-old');
      expect(written).toContain('KEEP=1');
    });

    it('removes a legacy provider token when the endpoint changes', async () => {
      const envPath = path.join(tmpDir, '.env');
      fs.mkdirSync(tmpDir, { recursive: true });
      fs.writeFileSync(
        envPath,
        'AUTOMEM_API_URL=https://host-a.example.test\nAUTOMEM_API_TOKEN=sk-legacy\nUNRELATED=keep\n'
      );

      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        endpoint: 'https://host-b.example.test',
        quiet: true,
        projectName: 'demo',
      });

      const written = fs.readFileSync(envPath, 'utf8');
      expect(written).not.toContain('sk-legacy');
      expect(written).toContain('UNRELATED=keep');
    });

    // Switching to mcp mode uninstalls the provider but used to leave its dotenv
    // credentials behind. Hermes loads that file before MCP discovery, so the key for
    // the previous endpoint was inherited by a server pointed at the new one.
    it('clears provider dotenv credentials when switching to mcp mode', async () => {
      const envPath = path.join(tmpDir, '.env');
      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'provider',
        endpoint: 'https://host-a.example.test',
        apiKey: 'sk-host-a',
        quiet: true,
        projectName: 'demo',
      });
      expect(fs.readFileSync(envPath, 'utf8')).toContain('sk-host-a');

      await applyHermesSetup({
        targetDir: tmpDir,
        mode: 'mcp',
        endpoint: 'https://host-b.example.test',
        quiet: true,
        projectName: 'demo',
      });

      const written = fs.readFileSync(envPath, 'utf8');
      expect(written).not.toContain('sk-host-a');
      expect(written).not.toContain('https://host-a.example.test');
    });

    it('recovers a credential stored under the deprecated AUTOMEM_API_TOKEN alias', async () => {
      const configPath = path.join(tmpDir, 'config.yaml');
      fs.mkdirSync(tmpDir, { recursive: true });
      fs.writeFileSync(
        configPath,
        [
          'mcp_servers:',
          '  automem:',
          '    command: npx',
          '    args:',
          '      - -y',
          '      - "@verygoodplugins/mcp-automem"',
          '    env:',
          '      AUTOMEM_API_URL: https://legacy.example.test',
          '      AUTOMEM_API_TOKEN: sk-legacy',
          '',
        ].join('\n')
      );

      await applyHermesSetup({ targetDir: tmpDir, quiet: true, projectName: 'demo' });

      const env = readEntryEnv();
      expect(env.AUTOMEM_API_URL).toBe('https://legacy.example.test');
      expect(env.AUTOMEM_API_KEY).toBe('sk-legacy');
    });
  });

  // Two starts and one end: both markers are present and correctly ordered, so an
  // indexOf-based check calls the file well-formed and replaces from the first start
  // through the end — silently deleting the second marker and the user's notes.
  it('refuses when the rules file has two start markers and one end', async () => {
    const soulPath = path.join(tmpDir, 'SOUL.md');
    const handWritten = [
      '# Existing Hermes Rules',
      '<!-- BEGIN AUTOMEM HERMES RULES -->',
      'stale half-block',
      '<!-- BEGIN AUTOMEM HERMES RULES -->',
      'notes the user wrote between the markers',
      '<!-- END AUTOMEM HERMES RULES -->',
      '',
    ].join('\n');
    fs.writeFileSync(soulPath, handWritten);

    await expect(
      applyHermesSetup({
        targetDir: tmpDir,
        projectName: 'test-project',
        endpoint: 'https://example.automem.test',
        quiet: true,
      })
    ).rejects.toThrow(/found 2 start markers and 1 end marker/);

    expect(fs.readFileSync(soulPath, 'utf8')).toBe(handWritten);
  });

  // A rejected run must not leave Hermes half-reconfigured. Validation runs before the
  // MCP entry, the memory provider, the .env, and the provider files are touched, so a
  // failed mode switch really does leave the install as it was.
  it('writes nothing at all when the rules file is rejected', async () => {
    await applyHermesSetup({
      mode: 'both',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://live.example.test',
      apiKey: 'sk-live',
      quiet: true,
    });

    const configPath = path.join(tmpDir, 'config.yaml');
    const envPath = path.join(tmpDir, '.env');
    const providerPath = path.join(tmpDir, 'plugins', 'automem', '__init__.py');
    const soulPath = path.join(tmpDir, 'SOUL.md');
    const configBefore = fs.readFileSync(configPath, 'utf8');
    const envBefore = fs.readFileSync(envPath, 'utf8');
    const providerBefore = fs.readFileSync(providerPath, 'utf8');

    // Now break the rules file and re-run as a different mode against another host.
    const handWritten = [
      '# Notes',
      '<!-- BEGIN AUTOMEM HERMES RULES -->',
      'stale half-block',
      '<!-- BEGIN AUTOMEM HERMES RULES -->',
      'notes the user wrote between the markers',
      '<!-- END AUTOMEM HERMES RULES -->',
      '',
    ].join('\n');
    fs.writeFileSync(soulPath, handWritten);

    await expect(
      applyHermesSetup({
        mode: 'mcp',
        targetDir: tmpDir,
        projectName: 'test-project',
        endpoint: 'https://other.example.test',
        apiKey: 'sk-other',
        quiet: true,
      })
    ).rejects.toThrow(/found 2 start markers and 1 end marker/);

    expect(fs.readFileSync(configPath, 'utf8')).toBe(configBefore);
    expect(fs.readFileSync(envPath, 'utf8')).toBe(envBefore);
    expect(fs.readFileSync(providerPath, 'utf8')).toBe(providerBefore);
    expect(fs.readFileSync(soulPath, 'utf8')).toBe(handWritten);
    expect(configBefore).toContain('https://live.example.test');
    expect(configBefore).not.toContain('https://other.example.test');
  });

  it('refuses a one-sided Hermes marker instead of appending', async () => {
    const soulPath = path.join(tmpDir, 'SOUL.md');
    const handWritten = ['# Notes', '<!-- BEGIN AUTOMEM HERMES RULES -->', 'half a block', ''].join(
      '\n'
    );
    fs.writeFileSync(soulPath, handWritten);

    await expect(
      applyHermesSetup({
        targetDir: tmpDir,
        projectName: 'test-project',
        endpoint: 'https://example.automem.test',
        quiet: true,
      })
    ).rejects.toThrow(/without a matching/);

    expect(fs.readFileSync(soulPath, 'utf8')).toBe(handWritten);
  });

  // Dropping the legacy Codex block is a migration, not this run's responsibility. A
  // stray Codex marker must not block the Hermes install — the Codex block is simply
  // left alone, markers and all, for the user to sort out.
  it('leaves a legacy Codex block with a stray marker alone', async () => {
    const legacyPath = path.join(tmpDir, 'AGENTS.md');
    const handWritten = [
      '# Existing Hermes Rules',
      '',
      '<!-- BEGIN AUTOMEM CODEX RULES -->',
      'stale codex guidance',
      '<!-- BEGIN AUTOMEM CODEX RULES -->',
      'notes the user wrote between the markers',
      '<!-- END AUTOMEM CODEX RULES -->',
      '',
      'Keep this unrelated Hermes rule.',
      '',
    ].join('\n');
    fs.writeFileSync(legacyPath, handWritten);

    await applyHermesSetup({
      mode: 'provider',
      targetDir: tmpDir,
      projectName: 'test-project',
      endpoint: 'https://example.automem.test',
      quiet: true,
    });

    expect(fs.readFileSync(path.join(tmpDir, 'SOUL.md'), 'utf8')).toContain(
      '<!-- BEGIN AUTOMEM HERMES RULES -->'
    );
    // Nothing between the stray Codex markers was taken along with the migration.
    expect(fs.readFileSync(legacyPath, 'utf8')).toBe(handWritten);
  });
});
