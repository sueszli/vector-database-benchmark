import sys
import pytest
from helpers import mock_legacy_venv, run_pipx_cli

def test_reinstall(pipx_temp_env, capsys):
    if False:
        print('Hello World!')
    assert not run_pipx_cli(['install', 'pycowsay'])
    assert not run_pipx_cli(['reinstall', '--python', sys.executable, 'pycowsay'])

def test_reinstall_nonexistent(pipx_temp_env, capsys):
    if False:
        while True:
            i = 10
    assert run_pipx_cli(['reinstall', '--python', sys.executable, 'nonexistent'])
    assert 'Nothing to reinstall for nonexistent' in capsys.readouterr().out

@pytest.mark.parametrize('metadata_version', [None, '0.1'])
def test_reinstall_legacy_venv(pipx_temp_env, capsys, metadata_version):
    if False:
        while True:
            i = 10
    assert not run_pipx_cli(['install', 'pycowsay'])
    mock_legacy_venv('pycowsay', metadata_version=metadata_version)
    assert not run_pipx_cli(['reinstall', '--python', sys.executable, 'pycowsay'])

def test_reinstall_suffix(pipx_temp_env, capsys):
    if False:
        while True:
            i = 10
    suffix = '_x'
    assert not run_pipx_cli(['install', 'pycowsay', f'--suffix={suffix}'])
    assert not run_pipx_cli(['reinstall', '--python', sys.executable, f'pycowsay{suffix}'])

@pytest.mark.parametrize('metadata_version', ['0.1'])
def test_reinstall_suffix_legacy_venv(pipx_temp_env, capsys, metadata_version):
    if False:
        i = 10
        return i + 15
    suffix = '_x'
    assert not run_pipx_cli(['install', 'pycowsay', f'--suffix={suffix}'])
    mock_legacy_venv(f'pycowsay{suffix}', metadata_version=metadata_version)
    assert not run_pipx_cli(['reinstall', '--python', sys.executable, f'pycowsay{suffix}'])

def test_reinstall_specifier(pipx_temp_env, capsys):
    if False:
        return 10
    assert not run_pipx_cli(['install', 'pylint==2.3.1'])
    captured = capsys.readouterr()
    assert not run_pipx_cli(['reinstall', '--python', sys.executable, 'pylint'])
    captured = capsys.readouterr()
    assert 'installed package pylint 2.3.1' in captured.out