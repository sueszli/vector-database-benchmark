import click
import pytest
from click.testing import CliRunner
from dagster._cli.job import execute_launch_command, job_launch_command
from dagster._core.errors import DagsterRunAlreadyExists
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.test_utils import new_cwd
from dagster._utils import file_relative_path
from .test_cli_commands import default_cli_test_instance, launch_command_contexts, memoizable_job, non_existant_python_file_workspace_args, python_bar_cli_args, valid_external_job_target_cli_args

def run_launch(kwargs, instance, expected_count=None):
    if False:
        while True:
            i = 10
    run = execute_launch_command(instance, kwargs)
    assert run
    if expected_count:
        assert instance.get_runs_count() == expected_count
    instance.run_launcher.join()

def run_launch_cli(execution_args, instance, expected_count=None):
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    result = runner.invoke(job_launch_command, execution_args)
    assert result.exit_code == 0, result.stdout
    if expected_count:
        assert instance.get_runs_count() == expected_count

def run_job_launch_cli(execution_args, instance, expected_count=None):
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    result = runner.invoke(job_launch_command, execution_args)
    assert result.exit_code == 0, result.stdout
    if expected_count:
        assert instance.get_runs_count() == expected_count

@pytest.mark.parametrize('gen_job_args', launch_command_contexts())
def test_launch_job(gen_job_args):
    if False:
        while True:
            i = 10
    with gen_job_args as (cli_args, instance):
        run_launch(cli_args, instance, expected_count=1)

def test_launch_non_existant_file():
    if False:
        print('Hello World!')
    with default_cli_test_instance() as instance:
        kwargs = non_existant_python_file_workspace_args()
        with pytest.raises(click.UsageError, match='Error loading location'):
            run_launch(kwargs, instance)

@pytest.mark.parametrize('job_cli_args', valid_external_job_target_cli_args())
def test_launch_job_cli(job_cli_args):
    if False:
        for i in range(10):
            print('nop')
    with default_cli_test_instance() as instance:
        run_job_launch_cli(job_cli_args, instance, expected_count=1)

@pytest.mark.parametrize('gen_job_args', [python_bar_cli_args('qux')])
def test_launch_with_run_id(gen_job_args):
    if False:
        print('Hello World!')
    runner = CliRunner()
    run_id = 'my_super_cool_run_id'
    with default_cli_test_instance() as instance:
        with gen_job_args as args:
            result = runner.invoke(job_launch_command, args + ['--run-id', run_id])
            assert result.exit_code == 0
            run = instance.get_run_by_id(run_id)
            assert run is not None
            bad_result = runner.invoke(job_launch_command, args + ['--run-id', run_id])
            assert bad_result.exit_code == 1
            assert isinstance(bad_result.exception, DagsterRunAlreadyExists)

@pytest.mark.parametrize('gen_job_args', [python_bar_cli_args('qux')])
def test_job_launch_with_run_id(gen_job_args):
    if False:
        while True:
            i = 10
    runner = CliRunner()
    run_id = 'my_super_cool_run_id'
    with default_cli_test_instance() as instance:
        with gen_job_args as args:
            result = runner.invoke(job_launch_command, args + ['--run-id', run_id])
            assert result.exit_code == 0
            run = instance.get_run_by_id(run_id)
            assert run is not None
            bad_result = runner.invoke(job_launch_command, args + ['--run-id', run_id])
            assert bad_result.exit_code == 1
            assert isinstance(bad_result.exception, DagsterRunAlreadyExists)

@pytest.mark.parametrize('gen_job_args', [python_bar_cli_args('qux')])
def test_launch_queued(gen_job_args):
    if False:
        return 10
    runner = CliRunner()
    run_id = 'my_super_cool_run_id'
    with default_cli_test_instance(overrides={'run_coordinator': {'class': 'QueuedRunCoordinator', 'module': 'dagster._core.run_coordinator'}}) as instance:
        with gen_job_args as args:
            result = runner.invoke(job_launch_command, args + ['--run-id', run_id])
            assert result.exit_code == 0
            run = instance.get_run_by_id(run_id)
            assert run is not None
            assert run.status == DagsterRunStatus.QUEUED

@pytest.mark.parametrize('gen_job_args', [python_bar_cli_args('qux')])
def test_job_launch_queued(gen_job_args):
    if False:
        print('Hello World!')
    runner = CliRunner()
    run_id = 'my_super_cool_run_id'
    with default_cli_test_instance(overrides={'run_coordinator': {'class': 'QueuedRunCoordinator', 'module': 'dagster._core.run_coordinator'}}) as instance:
        with gen_job_args as args:
            result = runner.invoke(job_launch_command, args + ['--run-id', run_id])
            assert result.exit_code == 0
            run = instance.get_run_by_id(run_id)
            assert run is not None
            assert run.status == DagsterRunStatus.QUEUED

def test_default_working_directory():
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    import os
    with default_cli_test_instance() as instance:
        with new_cwd(os.path.dirname(__file__)):
            result = runner.invoke(job_launch_command, ['-f', file_relative_path(__file__, 'file_with_local_import.py'), '-a', 'qux_job'])
            assert result.exit_code == 0
            runs = instance.get_runs()
            assert len(runs) == 1

def test_launch_using_memoization():
    if False:
        for i in range(10):
            print('nop')
    runner = CliRunner()
    with default_cli_test_instance() as instance:
        with python_bar_cli_args('memoizable') as args:
            result = runner.invoke(job_launch_command, args + ['--run-id', 'first'])
            assert result.exit_code == 0
            run = instance.get_run_by_id('first')
            assert len(run.step_keys_to_execute) == 1
            result = memoizable_job.execute_in_process(instance=instance)
            assert result.success
            result = runner.invoke(job_launch_command, args + ['--run-id', 'second'])
            assert result.exit_code == 0
            run = instance.get_run_by_id('second')
            assert len(run.step_keys_to_execute) == 0

def test_launch_command_help():
    if False:
        i = 10
        return i + 15
    runner = CliRunner()
    result = runner.invoke(job_launch_command, ['--help'])
    assert 'multiple times' in result.stdout