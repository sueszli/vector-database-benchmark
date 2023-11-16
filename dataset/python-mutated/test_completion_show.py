import os
import subprocess
import sys
from docs_src.commands.index import tutorial001 as mod

def test_completion_show_no_shell():
    if False:
        print('Hello World!')
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--show-completion'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TYPER_COMPLETE_TESTING': 'True', '_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION': 'True'})
    assert "Option '--show-completion' requires an argument" in result.stderr or '--show-completion option requires an argument' in result.stderr

def test_completion_show_bash():
    if False:
        print('Hello World!')
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--show-completion', 'bash'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TYPER_COMPLETE_TESTING': 'True', '_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION': 'True'})
    assert 'complete -o default -F _tutorial001py_completion tutorial001.py' in result.stdout

def test_completion_source_zsh():
    if False:
        print('Hello World!')
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--show-completion', 'zsh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TYPER_COMPLETE_TESTING': 'True', '_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION': 'True'})
    assert 'compdef _tutorial001py_completion tutorial001.py' in result.stdout

def test_completion_source_fish():
    if False:
        return 10
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--show-completion', 'fish'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TYPER_COMPLETE_TESTING': 'True', '_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION': 'True'})
    assert 'complete --command tutorial001.py --no-files' in result.stdout

def test_completion_source_powershell():
    if False:
        return 10
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--show-completion', 'powershell'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TYPER_COMPLETE_TESTING': 'True', '_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION': 'True'})
    assert 'Register-ArgumentCompleter -Native -CommandName tutorial001.py -ScriptBlock $scriptblock' in result.stdout

def test_completion_source_pwsh():
    if False:
        while True:
            i = 10
    result = subprocess.run([sys.executable, '-m', 'coverage', 'run', mod.__file__, '--show-completion', 'pwsh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', env={**os.environ, '_TYPER_COMPLETE_TESTING': 'True', '_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION': 'True'})
    assert 'Register-ArgumentCompleter -Native -CommandName tutorial001.py -ScriptBlock $scriptblock' in result.stdout