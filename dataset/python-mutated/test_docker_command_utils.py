from __future__ import annotations
import json
from unittest import mock
from unittest.mock import call
import pytest
from airflow_breeze.utils.docker_command_utils import autodetect_docker_context, check_docker_compose_version, check_docker_version

@mock.patch('airflow_breeze.utils.docker_command_utils.check_docker_permission_denied')
@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_version_unknown(mock_get_console, mock_run_command, mock_check_docker_permission_denied):
    if False:
        i = 10
        return i + 15
    mock_check_docker_permission_denied.return_value = False
    with pytest.raises(SystemExit) as e:
        check_docker_version()
    assert e.value.code == 1
    expected_run_command_calls = [call(['docker', 'version', '--format', '{{.Client.Version}}'], no_output_dump_on_exception=True, capture_output=True, text=True, check=False, dry_run_override=False)]
    mock_run_command.assert_has_calls(expected_run_command_calls)
    mock_get_console.return_value.print.assert_called_with('\n[warning]Your version of docker is unknown. If the scripts fail, please make sure to[/]\n[warning]install docker at least: 23.0.0 version.[/]\n')

@mock.patch('airflow_breeze.utils.docker_command_utils.check_docker_permission_denied')
@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_version_too_low(mock_get_console, mock_run_command, mock_check_docker_permission_denied):
    if False:
        print('Hello World!')
    mock_check_docker_permission_denied.return_value = False
    mock_run_command.return_value.returncode = 0
    mock_run_command.return_value.stdout = '0.9'
    with pytest.raises(SystemExit) as e:
        check_docker_version()
    assert e.value.code == 1
    mock_check_docker_permission_denied.assert_called()
    mock_run_command.assert_called_with(['docker', 'version', '--format', '{{.Client.Version}}'], no_output_dump_on_exception=True, capture_output=True, text=True, check=False, dry_run_override=False)
    mock_get_console.return_value.print.assert_called_with('\n[error]Your version of docker is too old: 0.9.\n[/]\n[warning]Please upgrade to at least 23.0.0.\n[/]\nYou can find installation instructions here: https://docs.docker.com/engine/install/\n')

@mock.patch('airflow_breeze.utils.docker_command_utils.check_docker_permission_denied')
@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_version_ok(mock_get_console, mock_run_command, mock_check_docker_permission_denied):
    if False:
        return 10
    mock_check_docker_permission_denied.return_value = False
    mock_run_command.return_value.returncode = 0
    mock_run_command.return_value.stdout = '23.0.0'
    check_docker_version()
    mock_check_docker_permission_denied.assert_called()
    mock_run_command.assert_called_with(['docker', 'version', '--format', '{{.Client.Version}}'], no_output_dump_on_exception=True, capture_output=True, text=True, check=False, dry_run_override=False)
    mock_get_console.return_value.print.assert_called_with('[success]Good version of Docker: 23.0.0.[/]')

@mock.patch('airflow_breeze.utils.docker_command_utils.check_docker_permission_denied')
@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_version_higher(mock_get_console, mock_run_command, mock_check_docker_permission_denied):
    if False:
        for i in range(10):
            print('nop')
    mock_check_docker_permission_denied.return_value = False
    mock_run_command.return_value.returncode = 0
    mock_run_command.return_value.stdout = '24.0.0'
    check_docker_version()
    mock_check_docker_permission_denied.assert_called()
    mock_run_command.assert_called_with(['docker', 'version', '--format', '{{.Client.Version}}'], no_output_dump_on_exception=True, capture_output=True, text=True, check=False, dry_run_override=False)
    mock_get_console.return_value.print.assert_called_with('[success]Good version of Docker: 24.0.0.[/]')

@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_compose_version_unknown(mock_get_console, mock_run_command):
    if False:
        return 10
    with pytest.raises(SystemExit) as e:
        check_docker_compose_version()
    assert e.value.code == 1
    expected_run_command_calls = [call(['docker', 'compose', 'version'], no_output_dump_on_exception=True, capture_output=True, text=True, dry_run_override=False)]
    mock_run_command.assert_has_calls(expected_run_command_calls)
    mock_get_console.return_value.print.assert_called_with('\n[error]Unknown docker-compose version.[/]\n[warning]At least 2.14.0 needed! Please upgrade!\n[/]\nSee https://docs.docker.com/compose/install/ for installation instructions.\n\nMake sure docker-compose you install is first on the PATH variable of yours.\n\n')

@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_compose_version_low(mock_get_console, mock_run_command):
    if False:
        for i in range(10):
            print('nop')
    mock_run_command.return_value.returncode = 0
    mock_run_command.return_value.stdout = '1.28.5'
    with pytest.raises(SystemExit) as e:
        check_docker_compose_version()
    assert e.value.code == 1
    mock_run_command.assert_called_with(['docker', 'compose', 'version'], no_output_dump_on_exception=True, capture_output=True, text=True, dry_run_override=False)
    mock_get_console.return_value.print.assert_called_with('\n[error]You have too old version of docker-compose: 1.28.5!\n[/]\n[warning]At least 2.14.0 needed! Please upgrade!\n[/]\nSee https://docs.docker.com/compose/install/ for installation instructions.\n\nMake sure docker-compose you install is first on the PATH variable of yours.\n\n')

@mock.patch('airflow_breeze.utils.docker_command_utils.run_command')
@mock.patch('airflow_breeze.utils.docker_command_utils.get_console')
def test_check_docker_compose_version_ok(mock_get_console, mock_run_command):
    if False:
        i = 10
        return i + 15
    mock_run_command.return_value.returncode = 0
    mock_run_command.return_value.stdout = '2.14.0'
    check_docker_compose_version()
    mock_run_command.assert_called_with(['docker', 'compose', 'version'], no_output_dump_on_exception=True, capture_output=True, text=True, dry_run_override=False)
    mock_get_console.return_value.print.assert_called_with('[success]Good version of docker-compose: 2.14.0[/]')

def _fake_ctx_output(*names: str) -> str:
    if False:
        return 10
    return '\n'.join((json.dumps({'Name': name, 'DockerEndpoint': f'unix://{name}'}) for name in names))

@pytest.mark.parametrize('context_output, selected_context, console_output', [(_fake_ctx_output('default'), 'default', '[info]Using default as context'), ('\n', 'default', '[warning]Could not detect docker builder'), (_fake_ctx_output('a', 'b'), 'a', '[warning]Could not use any of the preferred docker contexts'), (_fake_ctx_output('a', 'desktop-linux'), 'desktop-linux', '[info]Using desktop-linux as context'), (_fake_ctx_output('a', 'default'), 'default', '[info]Using default as context'), (_fake_ctx_output('a', 'default', 'desktop-linux'), 'desktop-linux', '[info]Using desktop-linux as context'), (_fake_ctx_output('a', 'default', 'desktop-linux'), 'desktop-linux', '[info]Using desktop-linux as context'), ('[{"Name": "desktop-linux", "DockerEndpoint": "unix://desktop-linux"}]', 'desktop-linux', '[info]Using desktop-linux as context')])
def test_autodetect_docker_context(context_output: str, selected_context: str, console_output: str):
    if False:
        i = 10
        return i + 15
    with mock.patch('airflow_breeze.utils.docker_command_utils.run_command') as mock_run_command:
        mock_run_command.return_value.returncode = 0
        mock_run_command.return_value.stdout = context_output
        with mock.patch('airflow_breeze.utils.docker_command_utils.get_console') as mock_get_console:
            mock_get_console.return_value.input.return_value = selected_context
            assert autodetect_docker_context() == selected_context
            mock_get_console.return_value.print.assert_called_once()
            assert console_output in mock_get_console.return_value.print.call_args[0][0]
SOCKET_INFO = json.dumps([{'Name': 'default', 'Metadata': {}, 'Endpoints': {'docker': {'Host': 'unix:///not-standard/docker.sock', 'SkipTLSVerify': False}}, 'TLSMaterial': {}, 'Storage': {'MetadataPath': '<IN MEMORY>', 'TLSPath': '<IN MEMORY>'}}])
SOCKET_INFO_DESKTOP_LINUX = json.dumps([{'Name': 'desktop-linux', 'Metadata': {}, 'Endpoints': {'docker': {'Host': 'unix:///VERY_NON_STANDARD/docker.sock', 'SkipTLSVerify': False}}, 'TLSMaterial': {}, 'Storage': {'MetadataPath': '<IN MEMORY>', 'TLSPath': '<IN MEMORY>'}}])