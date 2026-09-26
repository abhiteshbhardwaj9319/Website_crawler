"""Replay actual reviewed runs and inspect CLI output without paid calls or ingestion."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from website_rag.config import Settings
from website_rag.graph import load_run
from website_rag.render import answer_panel

out = Path('artifacts/evolution/rehearsal')
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(x) for x in Path('eval/results/evolution-live/answers.jsonl').read_text(encoding='utf-8').splitlines()]
by_key = {(r['split'], r['id']): r for r in rows}
keys = [('test_observed', 'T01'), ('user', 'U03'), ('holdout_v1', 'H05'), ('isolation', 'I02'), ('isolation', 'I05')]
checks = []
for width in (80, 100, 140):
    for key in keys:
        run_id = by_key[key]['run_id']
        args = ['--no-color', 'demo', '--replay', run_id]
        proc = subprocess.run([sys.executable, '-m', 'website_rag.cli', *args],
                              env={**os.environ, 'COLUMNS': str(width), 'NO_COLOR': '1', 'PYTHONUTF8': '1'},
                              capture_output=True, text=True, encoding='utf-8')
        assert proc.returncode == 0, proc.stderr
        assert 'RECORDED REPLAY' in proc.stdout and '\x1b[' not in proc.stdout
        assert 'UNTRUSTED REJECTED' not in proc.stdout
        assert max(map(len, proc.stdout.splitlines())) <= width
        (out/f'{key[1]}-{width}.txt').write_text(proc.stdout, encoding='utf-8')
        checks.append({'command': args, 'width': width, 'exit_code': proc.returncode})
commands = [['--ascii', '--no-color', 'sites', 'list'], ['sites', 'coverage', '1', '--json'],
            ['sites', 'coverage', '2', '--json'], ['doctor', '--json'], ['costs', '--json'],
            ['demo', '--replay', by_key[('user','U03')]['run_id'], '--json'],
            ['sources', by_key[('user','U03')]['run_id']],
            ['trace', by_key[('user','U03')]['run_id']],
            ['ask', 'q', '--site', '999', '--json']]
for i, args in enumerate(commands):
    proc = subprocess.run([sys.executable, '-m', 'website_rag.cli', *args], capture_output=True,
                          text=True, encoding='utf-8', env={**os.environ, 'PYTHONUTF8': '1'})
    assert proc.returncode == (2 if args[0] == 'ask' else 0), proc.stderr
    if '--json' in args:
        json.loads(proc.stdout)
        assert not proc.stderr
    if '--ascii' in args:
        assert not any('\u2500' <= c <= '\u257f' for c in proc.stdout)
    (out/f'command-{i}.txt').write_text(proc.stdout, encoding='utf-8')
    checks.append({'command': args, 'exit_code': proc.returncode})
console = Console(file=io.StringIO(), record=True, width=100, no_color=True)
console.print('RECORDED REPLAY: actual OpenAI input() answer, 2026-09-26')
console.print(answer_panel(load_run(Settings(), by_key[('user','U03')]['run_id'])))
console.save_svg('docs/diagrams/cli-input-replay.svg', title='Website RAG | recorded answer')
svg = Path('docs/diagrams/cli-input-replay.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
receipt = {'checks': checks, 'result': 'pass', 'environment': 'Windows PowerShell, UTF-8 piped subprocess output',
           'limitations': 'Windows Terminal GUI was not opened; video recording remains a user task.'}
Path('eval/results/evolution-live/cli-rehearsal.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(f'{len(checks)} CLI checks passed')
