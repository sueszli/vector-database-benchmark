import os
import mock
from click.testing import CliRunner
from dagster import DagsterEventType, job, op, reconstructable
from dagster._cli import api
from dagster._cli.api import ExecuteRunArgs, ExecuteStepArgs, verify_step
from dagster._core.execution.plan.state import KnownExecutionState
from dagster._core.execution.retries import RetryState
from dagster._core.execution.stats import RunStepKeyStatsSnapshot
from dagster._core.host_representation import JobHandle
from dagster._core.test_utils import create_run_for_test, environ, instance_for_test
from dagster._serdes import serialize_value
from dagster_tests.api_tests.utils import get_bar_repo_handle, get_foo_job_handle

def runner_execute_run(runner, cli_args):
    if False:
        i = 10
        return i + 15
    result = runner.invoke(api.execute_run_command, cli_args)
    if result.exit_code != 0:
        raise Exception(f'dagster runner_execute_run commands with cli_args {cli_args} returned exit_code {result.exit_code} with stdout:\n"{result.stdout}"\n exception: "\n{result.exception}"\n and result as string: "{result}"')
    return result

def test_execute_run():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            input_json = serialize_value(ExecuteRunArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, instance_ref=instance.get_ref()))
            result = runner_execute_run(runner, [input_json])
            assert 'RUN_SUCCESS' in result.stdout, f'no match, result: {result.stdout}'
            result = runner.invoke(api.execute_run_command, [input_json])
            assert result.exit_code == 0

@op
def needs_env_var():
    if False:
        return 10
    if os.getenv('FOO') != 'BAR':
        raise Exception('Missing env var')

@job
def needs_env_var_job():
    if False:
        for i in range(10):
            print('nop')
    needs_env_var()

def test_execute_run_with_secrets_loader(capfd):
    if False:
        print('Hello World!')
    recon_job = reconstructable(needs_env_var_job)
    runner = CliRunner()
    with environ({'FOO': None}):
        with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}, 'secrets': {'custom': {'module': 'dagster._core.test_utils', 'class': 'TestSecretsLoader', 'config': {'env_vars': {'FOO': 'BAR'}}}}}) as instance:
            run = create_run_for_test(instance, job_name='needs_env_var_job', run_id='new_run', job_code_origin=recon_job.get_python_origin())
            input_json = serialize_value(ExecuteRunArgs(job_origin=recon_job.get_python_origin(), run_id=run.run_id, instance_ref=instance.get_ref()))
            result = runner_execute_run(runner, [input_json])
            assert 'RUN_SUCCESS' in result.stdout, f'no match, result: {result.stdout}'
            (_, err) = capfd.readouterr()
            assert 'STEP_SUCCESS' in err, f'no match, result: {err}'
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        run = create_run_for_test(instance, job_name='needs_env_var_job', run_id='new_run', job_code_origin=recon_job.get_python_origin())
        input_json = serialize_value(ExecuteRunArgs(job_origin=recon_job.get_python_origin(), run_id=run.run_id, instance_ref=instance.get_ref()))
        result = runner_execute_run(runner, [input_json])
        assert 'RUN_FAILURE' in result.stdout, f'no match, result: {result.stdout}'
        (_, err) = capfd.readouterr()
        assert 'STEP_FAILURE' in err and 'Exception: Missing env var' in err, f'no match, result: {err}'

def test_execute_run_fail_job():
    if False:
        while True:
            i = 10
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_bar_repo_handle(instance) as repo_handle:
            job_handle = JobHandle('fail', repo_handle)
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            input_json = serialize_value(ExecuteRunArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, instance_ref=instance.get_ref()))
            result = runner_execute_run(runner, [input_json])
            assert result.exit_code == 0
            assert 'RUN_FAILURE' in result.stdout, f'no match, result: {result}'
            run = create_run_for_test(instance, job_name='foo', run_id='new_run_raise_on_error', job_code_origin=job_handle.get_python_origin())
            input_json_raise_on_failure = serialize_value(ExecuteRunArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, instance_ref=instance.get_ref(), set_exit_code_on_failure=True))
            result = runner.invoke(api.execute_run_command, [input_json_raise_on_failure])
            assert result.exit_code != 0, str(result.stdout)
            assert 'RUN_FAILURE' in result.stdout, f'no match, result: {result}'
            with mock.patch('dagster._core.execution.api.job_execution_iterator') as _mock_job_execution_iterator:
                _mock_job_execution_iterator.side_effect = Exception('Framework error')
                run = create_run_for_test(instance, job_name='foo', run_id='new_run_framework_error')
                input_json_raise_on_failure = serialize_value(ExecuteRunArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, instance_ref=instance.get_ref(), set_exit_code_on_failure=True))
                result = runner.invoke(api.execute_run_command, [input_json_raise_on_failure])
                assert result.exit_code != 0, str(result.stdout)

def test_execute_run_cannot_load():
    if False:
        while True:
            i = 10
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            input_json = serialize_value(ExecuteRunArgs(job_origin=job_handle.get_python_origin(), run_id='FOOBAR', instance_ref=instance.get_ref()))
            result = runner.invoke(api.execute_run_command, [input_json])
            assert result.exit_code != 0
            assert "Run with id 'FOOBAR' not found for run execution" in str(result.exception), f'no match, result: {result.stdout}'

def runner_execute_step(runner, cli_args, env=None):
    if False:
        i = 10
        return i + 15
    result = runner.invoke(api.execute_step_command, cli_args, env=env)
    if result.exit_code != 0:
        raise Exception(f'dagster runner_execute_step commands with cli_args {cli_args} returned exit_code {result.exit_code} with stdout:\n"{result.stdout}"\n exception: "\n{result.exception}"\n and result as string: "{result}"')
    return result

def test_execute_step():
    if False:
        return 10
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            args = ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref())
            result = runner_execute_step(runner, args.get_command_args()[5:])
        assert 'STEP_SUCCESS' in result.stdout
        assert '{"__class__": "StepSuccessData"' not in result.stdout

def test_execute_step_print_serialized_events():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            args = ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref(), print_serialized_events=True)
            result = runner_execute_step(runner, args.get_command_args()[5:])
        assert 'STEP_SUCCESS' in result.stdout
        assert '{"__class__": "StepSuccessData"' in result.stdout

def test_execute_step_with_secrets_loader():
    if False:
        for i in range(10):
            print('nop')
    recon_job = reconstructable(needs_env_var_job)
    runner = CliRunner()
    with environ({'FOO': None}):
        with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}, 'python_logs': {'dagster_handler_config': {'handlers': {'testHandler': {'class': 'dagster_tests.cli_tests.fake_python_logger_module.FakeHandler', 'level': 'INFO'}}}}, 'secrets': {'custom': {'module': 'dagster._core.test_utils', 'class': 'TestSecretsLoader', 'config': {'env_vars': {'FOO': 'BAR', 'REQUIRED_LOGGER_ENV_VAR': 'LOGGER_ENV_VAR_VALUE'}}}}}) as instance:
            run = create_run_for_test(instance, job_name='needs_env_var_job', run_id='new_run', job_code_origin=recon_job.get_python_origin())
            args = ExecuteStepArgs(job_origin=recon_job.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref())
            result = runner_execute_step(runner, args.get_command_args()[3:])
            assert 'STEP_SUCCESS' in result.stdout

def test_execute_step_with_env():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            args = ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref())
            result = runner_execute_step(runner, args.get_command_args(skip_serialized_namedtuple=True)[5:], env={d['name']: d['value'] for d in args.get_command_env()})
        assert 'STEP_SUCCESS' in result.stdout

def test_execute_step_non_compressed():
    if False:
        while True:
            i = 10
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            args = ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref())
            result = runner_execute_step(runner, [serialize_value(args)])
        assert 'STEP_SUCCESS' in result.stdout

def test_execute_step_1():
    if False:
        return 10
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            result = runner_execute_step(runner, ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref()).get_command_args()[5:])
        assert 'STEP_SUCCESS' in result.stdout

def test_execute_step_verify_step():
    if False:
        print('Hello World!')
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            retries = RetryState()
            assert verify_step(instance, run, retries, step_keys_to_execute=['do_something'])
            retries = RetryState()
            retries.mark_attempt('do_something')
            assert not verify_step(instance, run, retries, step_keys_to_execute=['do_something'])
            with mock.patch('dagster.cli.api.get_step_stats_by_key') as _step_stats_by_key:
                _step_stats_by_key.return_value = {'do_something': RunStepKeyStatsSnapshot(run_id=run.run_id, step_key='do_something', attempts=2)}
                retries = RetryState()
                retries.mark_attempt('do_something')
                assert not verify_step(instance, run, retries, step_keys_to_execute=['do_something'])
            runner_execute_step(runner, ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=None, instance_ref=instance.get_ref()).get_command_args()[5:])
            retries = RetryState()
            assert not verify_step(instance, run, retries, step_keys_to_execute=['do_something'])

@mock.patch('dagster.cli.api.verify_step')
def test_execute_step_verify_step_framework_error(mock_verify_step):
    if False:
        i = 10
        return i + 15
    with instance_for_test(overrides={'compute_logs': {'module': 'dagster._core.storage.noop_compute_log_manager', 'class': 'NoOpComputeLogManager'}}) as instance:
        with get_foo_job_handle(instance) as job_handle:
            runner = CliRunner()
            mock_verify_step.side_effect = Exception('Unexpected framework error text')
            run = create_run_for_test(instance, job_name='foo', run_id='new_run', job_code_origin=job_handle.get_python_origin())
            result = runner.invoke(api.execute_step_command, ExecuteStepArgs(job_origin=job_handle.get_python_origin(), run_id=run.run_id, step_keys_to_execute=['fake_step'], instance_ref=instance.get_ref(), should_verify_step=True, known_state=KnownExecutionState({}, {'blah': {'result': ['0', '1', '2']}})).get_command_args()[5:])
            assert result.exit_code != 0
            logs = instance.all_logs(run.run_id, of_type=DagsterEventType.ENGINE_EVENT)
            log_entry = logs[0]
            assert log_entry.message == 'An exception was thrown during step execution that is likely a framework error, rather than an error in user code.'
            assert log_entry.step_key == 'fake_step'
            assert 'Unexpected framework error text' in str(log_entry.dagster_event.event_specific_data.error)