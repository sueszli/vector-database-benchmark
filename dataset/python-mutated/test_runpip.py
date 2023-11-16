from helpers import run_pipx_cli

def test_runpip(pipx_temp_env, monkeypatch, capsys):
    if False:
        i = 10
        return i + 15
    assert not run_pipx_cli(['install', 'pycowsay'])
    assert not run_pipx_cli(['runpip', 'pycowsay', 'list'])