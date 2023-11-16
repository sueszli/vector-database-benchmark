import re
import click
import mock
import pytest
from click.testing import CliRunner
from dagster._cli.schedule import check_repo_and_scheduler, schedule_list_command, schedule_logs_command, schedule_restart_command, schedule_start_command, schedule_stop_command, schedule_wipe_command
from dagster._core.host_representation import ExternalRepository
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import environ
from .test_cli_commands import schedule_command_contexts, scheduler_instance

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_list(gen_schedule_args):
    if False:
        while True:
            i = 10
    with gen_schedule_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(schedule_list_command, cli_args)
            if result.exception:
                raise result.exception
            assert result.exit_code == 0
            assert result.output == "Repository bar\n**************\nSchedule: foo_schedule [STOPPED]\nCron Schedule: * * * * *\n**********************************\nSchedule: union_schedule [STOPPED]\nCron Schedule: ['* * * * *', '* * * * *']\n"

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_start_and_stop(gen_schedule_args):
    if False:
        i = 10
        return i + 15
    with gen_schedule_args as (cli_args, instance):
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            runner = CliRunner()
            result = runner.invoke(schedule_start_command, cli_args + ['foo_schedule'])
            assert result.exit_code == 0
            assert result.output == 'Started schedule foo_schedule\n'
            result = runner.invoke(schedule_stop_command, cli_args + ['foo_schedule'])
            assert result.exit_code == 0
            assert result.output == 'Stopped schedule foo_schedule\n'

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_start_empty(gen_schedule_args):
    if False:
        i = 10
        return i + 15
    with gen_schedule_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(schedule_start_command, cli_args)
            assert result.exit_code == 0
            assert 'Noop: dagster schedule start was called without any arguments' in result.output

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_start_all(gen_schedule_args):
    if False:
        i = 10
        return i + 15
    with gen_schedule_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(schedule_start_command, cli_args + ['--start-all'])
            assert result.exit_code == 0
            assert result.output == 'Started all schedules for repository bar\n'

def test_schedules_wipe_correct_delete_message():
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    with scheduler_instance() as instance, mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
        _instance.return_value = instance
        result = runner.invoke(schedule_wipe_command, input='DELETE\n')
        if result.exception:
            raise result.exception
        assert result.exit_code == 0
        assert 'Turned off all schedules and deleted all schedule history' in result.output

def test_schedules_wipe_incorrect_delete_message():
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    with scheduler_instance() as instance, mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
        _instance.return_value = instance
        result = runner.invoke(schedule_wipe_command, input='WRONG\n')
        assert result.exit_code == 0
        assert 'Exiting without turning off schedules or deleting schedule history' in result.output

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_restart(gen_schedule_args):
    if False:
        for i in range(10):
            print('nop')
    with gen_schedule_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(schedule_start_command, cli_args + ['foo_schedule'])
            result = runner.invoke(schedule_restart_command, cli_args + ['foo_schedule'])
            assert result.exit_code == 0
            assert 'Restarted schedule foo_schedule' in result.output

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_restart_all(gen_schedule_args):
    if False:
        for i in range(10):
            print('nop')
    with gen_schedule_args as (cli_args, instance):
        runner = CliRunner()
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            result = runner.invoke(schedule_start_command, cli_args + ['foo_schedule'])
            result = runner.invoke(schedule_restart_command, cli_args + ['foo_schedule', '--restart-all-running'])
            assert result.exit_code == 0
            assert result.output == 'Restarted all running schedules for repository bar\n'

@pytest.mark.parametrize('gen_schedule_args', schedule_command_contexts())
def test_schedules_logs(gen_schedule_args):
    if False:
        print('Hello World!')
    with gen_schedule_args as (cli_args, instance):
        with mock.patch('dagster._core.instance.DagsterInstance.get') as _instance:
            _instance.return_value = instance
            runner = CliRunner()
            result = runner.invoke(schedule_logs_command, cli_args + ['foo_schedule'])
            assert result.exit_code == 0
            assert 'scheduler.log' in result.output

def test_check_repo_and_scheduler_no_external_schedules():
    if False:
        i = 10
        return i + 15
    repository = mock.MagicMock(spec=ExternalRepository)
    repository.get_external_schedules.return_value = []
    instance = mock.MagicMock(spec=DagsterInstance)
    with pytest.raises(click.UsageError, match='There are no schedules defined for repository'):
        check_repo_and_scheduler(repository, instance)

def test_check_repo_and_scheduler_dagster_home_not_set():
    if False:
        print('Hello World!')
    with environ({'DAGSTER_HOME': ''}):
        repository = mock.MagicMock(spec=ExternalRepository)
        repository.get_external_schedules.return_value = [mock.MagicMock()]
        instance = mock.MagicMock(spec=DagsterInstance)
        with pytest.raises(click.UsageError, match=re.escape('The environment variable $DAGSTER_HOME is not set.')):
            check_repo_and_scheduler(repository, instance)