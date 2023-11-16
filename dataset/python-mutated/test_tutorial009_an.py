import os
import subprocess
import sys
from typer.testing import CliRunner
from docs_src.options_autocompletion import tutorial009_an as mod
runner = CliRunner()

def test_completion():
    if False:
        for i in range(10):
            print('nop')
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, ' '], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TUTORIAL009_AN.PY_COMPLETE': 'complete_zsh', '_TYPER_COMPLETE_ARGS': 'tutorial009_an.py --name Sebastian --name ', '_TYPER_COMPLETE_TESTING': 'True'})
    assert '"Camila":"The reader of books."' in result.stdout
    assert '"Carlos":"The writer of scripts."' in result.stdout
    assert '"Sebastian":"The type hints guy."' not in result.stdout
    assert '[]' in result.stderr or "['--name', 'Sebastian', '--name']" in result.stderr

def test_1():
    if False:
        while True:
            i = 10
    result = runner.invoke(mod.app, ['--name', 'Camila', '--name', 'Sebastian'])
    assert result.exit_code == 0
    assert 'Hello Camila' in result.output
    assert 'Hello Sebastian' in result.output

def test_script():
    if False:
        while True:
            i = 10
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    assert 'Usage' in result.stdout