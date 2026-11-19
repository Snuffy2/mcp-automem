import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const SUITE_DIR = path.resolve(__dirname);

// The provider template is Python that runs inside Hermes; its on_memory_write
// mirror is covered by a stdlib unittest suite with a fake HTTP layer. Run it here
// so `npm test` (and CI) gates on it without adding a Python test runner.
describe('Hermes provider built-in memory mirror (Python)', () => {
  it('passes the on_memory_write unittest suite', () => {
    const result = spawnSync(
      'python3',
      ['-B', '-m', 'unittest', 'discover', '-s', SUITE_DIR, '-p', 'test_*.py'],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.error).toBeUndefined();
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
    expect(result.stderr).toMatch(/\nOK\b/);
  });
});
