import os
import subprocess
import sys
from typer.testing import CliRunner
from docs_src.options_autocompletion import tutorial002_an as mod
runner = CliRunner()

def test_completion():
    if False:
        for i in range(10):
            print('nop')
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, ' '], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TUTORIAL002_AN.PY_COMPLETE': 'complete_zsh', '_TYPER_COMPLETE_ARGS': 'tutorial002_an.py --name ', '_TYPER_COMPLETE_TESTING': 'True'})
    assert 'Camila' in result.stdout
    assert 'Carlos' in result.stdout
    assert 'Sebastian' in result.stdout

def test_1():
    if False:
        while True:
            i = 10
    result = runner.invoke(mod.app, ['--name', 'Camila'])
    assert result.exit_code == 0
    assert 'Hello Camila' in result.output

def test_script():
    if False:
        print('Hello World!')
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    assert 'Usage' in result.stdout