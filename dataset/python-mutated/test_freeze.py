import contextlib
import io
import pathlib
import typing as t
import pytest
from libtmux.server import Server
from tmuxp import cli
from tmuxp.config_reader import ConfigReader

@pytest.mark.parametrize('cli_args,inputs', [(['freeze', 'myfrozensession'], ['y\n', './la.yaml\n', 'y\n']), (['freeze', 'myfrozensession'], ['y\n', './exists.yaml\n', './la.yaml\n', 'y\n']), (['freeze'], ['y\n', './la.yaml\n', 'y\n']), (['freeze'], ['y\n', './exists.yaml\n', './la.yaml\n', 'y\n'])])
def test_freeze(server: 'Server', cli_args: t.List[str], inputs: t.List[str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    monkeypatch.setenv('HOME', str(tmp_path))
    exists_yaml = tmp_path / 'exists.yaml'
    exists_yaml.touch()
    server.new_session(session_name='myfirstsession')
    server.new_session(session_name='myfrozensession')
    second_session = server.sessions[1]
    first_pane_on_second_session_id = second_session.windows[0].panes[0].pane_id
    assert first_pane_on_second_session_id
    monkeypatch.setenv('TMUX_PANE', first_pane_on_second_session_id)
    monkeypatch.chdir(tmp_path)
    assert server.socket_name is not None
    cli_args = [*cli_args, '-L', server.socket_name]
    monkeypatch.setattr('sys.stdin', io.StringIO(''.join(inputs)))
    with contextlib.suppress(SystemExit):
        cli.cli(cli_args)
    yaml_config_path = tmp_path / 'la.yaml'
    assert yaml_config_path.exists()
    yaml_config = yaml_config_path.open().read()
    frozen_config = ConfigReader._load(format='yaml', content=yaml_config)
    assert frozen_config['session_name'] == 'myfrozensession'

@pytest.mark.parametrize('cli_args,inputs', [(['freeze', 'mysession', '--force'], ['\n', '\n', 'y\n', './exists.yaml\n', 'y\n']), (['freeze', '--force'], ['\n', '\n', 'y\n', './exists.yaml\n', 'y\n'])])
def test_freeze_overwrite(server: 'Server', cli_args: t.List[str], inputs: t.List[str], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('HOME', str(tmp_path))
    exists_yaml = tmp_path / 'exists.yaml'
    exists_yaml.touch()
    server.new_session(session_name='mysession')
    monkeypatch.chdir(tmp_path)
    assert server.socket_name is not None
    cli_args = [*cli_args, '-L', server.socket_name]
    monkeypatch.setattr('sys.stdin', io.StringIO(''.join(inputs)))
    with contextlib.suppress(SystemExit):
        cli.cli(cli_args)
    yaml_config_path = tmp_path / 'exists.yaml'
    assert yaml_config_path.exists()