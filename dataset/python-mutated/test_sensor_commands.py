import re
import click
import mock
import pytest
from click.testing import CliRunner
from dagster._cli.sensor import check_repo_and_scheduler, sensor_cursor_command, sensor_list_command, sensor_preview_command, sensor_start_command, sensor_stop_command
from dagster._core.host_representation import ExternalRepository
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import environ
from .test_cli_commands import sensor_command_contexts

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensors_list(gen_sensor_args):
    if False:
        print('Hello World!')
    with gen_sensor_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(sensor_list_command, cli_args)
            assert result.exit_code == 0
            assert result.output == 'Repository bar\n**************\nSensor: foo_sensor [STOPPED]\n'

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensors_start_and_stop(gen_sensor_args):
    if False:
        i = 10
        return i + 15
    with gen_sensor_args as (cli_args, instance):
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            runner = CliRunner()
            result = runner.invoke(sensor_start_command, cli_args + ['foo_sensor'])
            assert result.exit_code == 0
            assert result.output == 'Started sensor foo_sensor\n'
            result = runner.invoke(sensor_stop_command, cli_args + ['foo_sensor'])
            assert result.exit_code == 0
            assert result.output == 'Stopped sensor foo_sensor\n'

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensors_start_empty(gen_sensor_args):
    if False:
        print('Hello World!')
    with gen_sensor_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(sensor_start_command, cli_args)
            assert result.exit_code == 2
            assert 'Missing sensor name argument' in result.output

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensors_start_all(gen_sensor_args):
    if False:
        i = 10
        return i + 15
    with gen_sensor_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(sensor_start_command, cli_args + ['--start-all'])
            assert result.exit_code == 0
            assert result.output == 'Started all sensors for repository bar\n'

def test_check_repo_and_sensorr_no_external_sensors():
    if False:
        print('Hello World!')
    repository = mock.MagicMock(spec=ExternalRepository)
    repository.get_external_sensors.return_value = []
    instance = mock.MagicMock(spec=DagsterInstance)
    with pytest.raises(click.UsageError, match='There are no sensors defined for repository'):
        check_repo_and_scheduler(repository, instance)

def test_check_repo_and_scheduler_dagster_home_not_set():
    if False:
        while True:
            i = 10
    with environ({'DAGSTER_HOME': ''}):
        repository = mock.MagicMock(spec=ExternalRepository)
        repository.get_external_sensors.return_value = [mock.MagicMock()]
        instance = mock.MagicMock(spec=DagsterInstance)
        with pytest.raises(click.UsageError, match=re.escape('The environment variable $DAGSTER_HOME is not set.')):
            check_repo_and_scheduler(repository, instance)

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensor_preview(gen_sensor_args):
    if False:
        while True:
            i = 10
    with gen_sensor_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(sensor_preview_command, cli_args + ['foo_sensor'])
            assert result.exit_code == 0
            assert result.output == 'Sensor returning run requests for 1 run(s):\n\nfoo: FOO\n\n'

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensor_preview_since(gen_sensor_args):
    if False:
        return 10
    with gen_sensor_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(sensor_preview_command, cli_args + ['foo_sensor', '--since', 1.1])
            assert result.exit_code == 0
            assert result.output == 'Sensor returning run requests for 1 run(s):\n\nfoo: FOO\nsince: 1.1\n\n'

@pytest.mark.parametrize('gen_sensor_args', sensor_command_contexts())
def test_sensor_cursor(gen_sensor_args):
    if False:
        return 10
    with gen_sensor_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(sensor_cursor_command, cli_args + ['foo_sensor', '--set', 'foo'])
            assert result.exit_code == 0
            assert result.output == 'Set cursor state for sensor foo_sensor to "foo"\n'
            result = runner.invoke(sensor_cursor_command, cli_args + ['foo_sensor', '--delete'])
            assert result.exit_code == 0
            assert result.output == 'Cleared cursor state for sensor foo_sensor\n'