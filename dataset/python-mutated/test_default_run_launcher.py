import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Mapping
import pytest
from dagster import DagsterEventType, DefaultRunLauncher, _check as check, _seven, file_relative_path, repository
from dagster._core.definitions import op
from dagster._core.errors import DagsterLaunchFailedError
from dagster._core.instance import DagsterInstance
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.storage.tags import GRPC_INFO_TAG
from dagster._core.test_utils import environ, instance_for_test, poll_for_event, poll_for_finished_run, poll_for_step_start
from dagster._core.workspace.context import WorkspaceProcessContext, WorkspaceRequestContext
from dagster._core.workspace.load_target import PythonFileTarget
from dagster._grpc.client import DagsterGrpcClient
from dagster._grpc.types import CancelExecutionRequest

@op
def noop_op(_):
    if False:
        while True:
            i = 10
    pass
from dagster import job

@job
def noop_job():
    if False:
        return 10
    pass

@op
def crashy_op(_):
    if False:
        i = 10
        return i + 15
    os._exit(1)

@job
def crashy_job():
    if False:
        for i in range(10):
            print('nop')
    crashy_op()

@op
def exity_op(_):
    if False:
        i = 10
        return i + 15
    sys.exit(1)

@job
def exity_job():
    if False:
        while True:
            i = 10
    exity_op()

@op
def sleepy_op(_):
    if False:
        return 10
    while True:
        time.sleep(0.1)

@job
def sleepy_job():
    if False:
        i = 10
        return i + 15
    sleepy_op()

@op
def slow_sudop(_):
    if False:
        i = 10
        return i + 15
    time.sleep(4)

@job
def slow_job():
    if False:
        return 10
    slow_sudop()

@op
def return_one(_):
    if False:
        i = 10
        return i + 15
    return 1

@op
def multiply_by_2(_, num):
    if False:
        return 10
    return num * 2

@op
def multiply_by_3(_, num):
    if False:
        i = 10
        return i + 15
    return num * 3

@op
def add(_, num1, num2):
    if False:
        print('Hello World!')
    return num1 + num2

@job
def math_diamond():
    if False:
        print('Hello World!')
    one = return_one()
    add(multiply_by_2(one), multiply_by_3(one))

@repository
def nope():
    if False:
        for i in range(10):
            print('nop')
    return [noop_job, crashy_job, exity_job, sleepy_job, slow_job, math_diamond]

def run_configs():
    if False:
        i = 10
        return i + 15
    return [None, {'execution': {'config': {'in_process': {}}}}]

def _check_event_log_contains(event_log, expected_type_and_message):
    if False:
        return 10
    types_and_messages = [(e.dagster_event.event_type_value, e.message) for e in event_log if e.is_dagster_event]
    for (expected_event_type, expected_message_fragment) in expected_type_and_message:
        assert any((event_type == expected_event_type and expected_message_fragment in message for (event_type, message) in types_and_messages)), f'Missing {expected_event_type}:{expected_message_fragment}'

@pytest.mark.parametrize('run_config', run_configs())
def test_successful_run(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        i = 10
        return i + 15
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('noop_job')
    dagster_run = instance.create_run_for_job(job_def=noop_job, run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = dagster_run.run_id
    run = instance.get_run_by_id(run_id)
    assert run
    assert run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run_id=dagster_run.run_id, workspace=workspace)
    dagster_run = instance.get_run_by_id(run_id)
    assert dagster_run
    assert dagster_run.run_id == run_id
    dagster_run = poll_for_finished_run(instance, run_id)
    assert dagster_run.status == DagsterRunStatus.SUCCESS

def test_successful_run_from_pending(instance: DagsterInstance, pending_workspace: WorkspaceRequestContext):
    if False:
        return 10
    code_location = pending_workspace.get_code_location('test2')
    external_job = code_location.get_repository('pending').get_full_external_job('my_cool_asset_job')
    external_execution_plan = code_location.get_external_execution_plan(external_job=external_job, run_config={}, step_keys_to_execute=None, known_state=None)
    call_counts = instance.run_storage.get_cursor_values({'compute_cacheable_data_called_a', 'compute_cacheable_data_called_b', 'get_definitions_called_a', 'get_definitions_called_b'})
    assert call_counts.get('compute_cacheable_data_called_a') == '1'
    assert call_counts.get('compute_cacheable_data_called_b') == '1'
    assert call_counts.get('get_definitions_called_a') == '1'
    assert call_counts.get('get_definitions_called_b') == '1'
    created_run = instance.create_run(job_name='my_cool_asset_job', run_id='xyzabc', run_config=None, resolved_op_selection=None, step_keys_to_execute=None, status=None, tags=None, root_run_id=None, parent_run_id=None, job_snapshot=external_job.job_snapshot, execution_plan_snapshot=external_execution_plan.execution_plan_snapshot, parent_job_snapshot=external_job.parent_job_snapshot, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin(), asset_selection=None, op_selection=None, asset_check_selection=None)
    run_id = created_run.run_id
    assert check.not_none(instance.get_run_by_id(run_id)).status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run_id=run_id, workspace=pending_workspace)
    stored_run = check.not_none(instance.get_run_by_id(run_id))
    assert created_run.run_id == stored_run.run_id
    assert created_run.execution_plan_snapshot_id == stored_run.execution_plan_snapshot_id
    assert created_run.has_repository_load_data and stored_run.has_repository_load_data
    finished_run = poll_for_finished_run(instance, run_id)
    assert finished_run.status == DagsterRunStatus.SUCCESS
    call_counts = instance.run_storage.get_cursor_values({'compute_cacheable_data_called_a', 'compute_cacheable_data_called_b', 'get_definitions_called_a', 'get_definitions_called_b'})
    assert call_counts.get('compute_cacheable_data_called_a') == '1'
    assert call_counts.get('compute_cacheable_data_called_b') == '1'
    assert int(call_counts['get_definitions_called_a']) < 6
    assert int(call_counts['get_definitions_called_b']) < 6

def test_invalid_instance_run():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryDirectory() as temp_dir:
        correct_run_storage_dir = os.path.join(temp_dir, 'history', '')
        wrong_run_storage_dir = os.path.join(temp_dir, 'wrong', '')
        with environ({'RUN_STORAGE_ENV': correct_run_storage_dir}):
            with instance_for_test(temp_dir=temp_dir, overrides={'run_storage': {'module': 'dagster._core.storage.runs', 'class': 'SqliteRunStorage', 'config': {'base_dir': {'env': 'RUN_STORAGE_ENV'}}}}) as instance:
                with environ({'RUN_STORAGE_ENV': wrong_run_storage_dir}):
                    with WorkspaceProcessContext(instance, PythonFileTarget(python_file=file_relative_path(__file__, 'test_default_run_launcher.py'), attribute='nope', working_directory=None, location_name='test')) as workspace_process_context:
                        workspace = workspace_process_context.create_request_context()
                        external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('noop_job')
                        run = instance.create_run_for_job(job_def=noop_job, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
                        with pytest.raises(DagsterLaunchFailedError, match=re.escape(f'gRPC server could not load run {run.run_id} in order to execute it')):
                            instance.launch_run(run_id=run.run_id, workspace=workspace)
                        failed_run = instance.get_run_by_id(run.run_id)
                        assert failed_run.status == DagsterRunStatus.FAILURE

@pytest.mark.parametrize('run_config', run_configs())
@pytest.mark.skipif(_seven.IS_WINDOWS, reason='Crashy jobs leave resources open on windows, causing filesystem contention')
def test_crashy_run(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        for i in range(10):
            print('nop')
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('crashy_job')
    run = instance.create_run_for_job(job_def=crashy_job, run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    run = instance.get_run_by_id(run_id)
    assert run
    assert run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run.run_id, workspace)
    failed_run = instance.get_run_by_id(run_id)
    assert failed_run
    assert failed_run.run_id == run_id
    failed_run = poll_for_finished_run(instance, run_id, timeout=5)
    assert failed_run.status == DagsterRunStatus.FAILURE
    event_records = instance.all_logs(run_id)
    if run_config is None:
        message = 'Multiprocess executor: child process for step crashy_op unexpectedly exited'
    else:
        message = f'Run execution process for {run_id} unexpectedly exited'
    assert _message_exists(event_records, message)

@pytest.mark.parametrize('run_config', run_configs())
@pytest.mark.skipif(_seven.IS_WINDOWS, reason='Crashy jobs leave resources open on windows, causing filesystem contention')
def test_exity_run(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        print('Hello World!')
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('exity_job')
    run = instance.create_run_for_job(job_def=exity_job, run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    run = instance.get_run_by_id(run_id)
    assert run
    assert run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run.run_id, workspace)
    failed_run = instance.get_run_by_id(run_id)
    assert failed_run
    assert failed_run.run_id == run_id
    failed_run = poll_for_finished_run(instance, run_id, timeout=5)
    assert failed_run.status == DagsterRunStatus.FAILURE
    event_records = instance.all_logs(run_id)
    assert _message_exists(event_records, 'Execution of step "exity_op" failed.')
    assert _message_exists(event_records, 'Execution of run for "exity_job" failed. Steps failed: [\'exity_op\']')

@pytest.mark.parametrize('run_config', run_configs())
def test_terminated_run(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        return 10
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('sleepy_job')
    run = instance.create_run_for_job(job_def=sleepy_job, run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    run = instance.get_run_by_id(run_id)
    assert run and run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run.run_id, workspace)
    poll_for_step_start(instance, run_id)
    launcher = instance.run_launcher
    assert launcher.terminate(run_id)
    terminated_run = poll_for_finished_run(instance, run_id, timeout=30)
    terminated_run = instance.get_run_by_id(run_id)
    assert terminated_run and terminated_run.status == DagsterRunStatus.CANCELED
    poll_for_event(instance, run_id, event_type='ENGINE_EVENT', message='Process for run exited')
    run_logs = instance.all_logs(run_id)
    if run_config is None:
        _check_event_log_contains(run_logs, [('PIPELINE_CANCELING', 'Sending run termination request.'), ('ENGINE_EVENT', 'Multiprocess executor: received termination signal - forwarding to active child process'), ('ENGINE_EVENT', 'Multiprocess executor: interrupted all active child processes'), ('STEP_FAILURE', 'Execution of step "sleepy_op" failed.'), ('PIPELINE_CANCELED', 'Execution of run for "sleepy_job" canceled.'), ('ENGINE_EVENT', 'Process for run exited')])
    else:
        _check_event_log_contains(run_logs, [('PIPELINE_CANCELING', 'Sending run termination request.'), ('STEP_FAILURE', 'Execution of step "sleepy_op" failed.'), ('PIPELINE_CANCELED', 'Execution of run for "sleepy_job" canceled.'), ('ENGINE_EVENT', 'Process for run exited')])

@pytest.mark.parametrize('run_config', run_configs())
def test_cleanup_after_force_terminate(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        print('Hello World!')
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('sleepy_job')
    run = instance.create_run_for_job(job_def=sleepy_job, run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    instance.launch_run(run.run_id, workspace)
    poll_for_step_start(instance, run_id)
    instance.report_run_canceling(run)
    instance.report_run_canceled(run)
    reloaded_run = instance.get_run_by_id(run_id)
    assert reloaded_run
    grpc_info = json.loads(reloaded_run.tags[GRPC_INFO_TAG])
    client = DagsterGrpcClient(port=grpc_info.get('port'), socket=grpc_info.get('socket'), host=grpc_info.get('host'))
    client.cancel_execution(CancelExecutionRequest(run_id=run_id))
    start_time = time.time()
    while True:
        if time.time() - start_time > 30:
            raise Exception('Timed out waiting for cleanup message')
        logs = instance.all_logs(run_id)
        if any(['Computational resources were cleaned up after the run was forcibly marked as canceled.' in str(event) for event in logs]):
            break
        time.sleep(1)
    run = instance.get_run_by_id(run_id)
    assert run and run.status == DagsterRunStatus.CANCELED

def _get_engine_events(event_records):
    if False:
        i = 10
        return i + 15
    return [er for er in event_records if er.dagster_event and er.dagster_event.event_type in {DagsterEventType.ENGINE_EVENT, DagsterEventType.STEP_WORKER_STARTING, DagsterEventType.STEP_WORKER_STARTED, DagsterEventType.RESOURCE_INIT_STARTED, DagsterEventType.RESOURCE_INIT_SUCCESS, DagsterEventType.RESOURCE_INIT_FAILURE}]

def _get_successful_step_keys(event_records):
    if False:
        while True:
            i = 10
    step_keys = set()
    for er in event_records:
        if er.dagster_event and er.dagster_event.is_step_success:
            step_keys.add(er.dagster_event.step_key)
    return step_keys

def _message_exists(event_records, message_text):
    if False:
        return 10
    for event_record in event_records:
        if message_text in event_record.message:
            return True
    return False

@pytest.mark.parametrize('run_config', run_configs())
def test_single_op_selection_execution(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        for i in range(10):
            print('nop')
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('math_diamond')
    run = instance.create_run_for_job(job_def=math_diamond, run_config=run_config, op_selection=['return_one'], external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    run = instance.get_run_by_id(run_id)
    assert run and run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run.run_id, workspace)
    finished_run = poll_for_finished_run(instance, run_id)
    event_records = instance.all_logs(run_id)
    assert finished_run
    assert finished_run.run_id == run_id
    assert finished_run.status == DagsterRunStatus.SUCCESS
    assert _get_successful_step_keys(event_records) == {'return_one'}

@pytest.mark.parametrize('run_config', run_configs())
def test_multi_op_selection_execution(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        return 10
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('math_diamond')
    run = instance.create_run_for_job(job_def=math_diamond, run_config=run_config, op_selection=['return_one', 'multiply_by_2'], external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    run = instance.get_run_by_id(run_id)
    assert run and run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run.run_id, workspace)
    finished_run = poll_for_finished_run(instance, run_id)
    event_records = instance.all_logs(run_id)
    assert finished_run
    assert finished_run.run_id == run_id
    assert finished_run.status == DagsterRunStatus.SUCCESS
    assert _get_successful_step_keys(event_records) == {'return_one', 'multiply_by_2'}

@pytest.mark.parametrize('run_config', run_configs())
def test_engine_events(instance: DagsterInstance, workspace: WorkspaceRequestContext, run_config: Mapping[str, Any]):
    if False:
        print('Hello World!')
    external_job = workspace.get_code_location('test').get_repository('nope').get_full_external_job('math_diamond')
    run = instance.create_run_for_job(job_def=math_diamond, run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
    run_id = run.run_id
    run = instance.get_run_by_id(run_id)
    assert run and run.status == DagsterRunStatus.NOT_STARTED
    instance.launch_run(run.run_id, workspace)
    finished_run = poll_for_finished_run(instance, run_id)
    assert finished_run
    assert finished_run.run_id == run_id
    assert finished_run.status == DagsterRunStatus.SUCCESS
    poll_for_event(instance, run_id, event_type='ENGINE_EVENT', message='Process for run exited')
    event_records = instance.all_logs(run_id)
    engine_events = _get_engine_events(event_records)
    if run_config is None:
        messages = ['Started process for run', 'Executing steps using multiprocess executor', 'Launching subprocess for "return_one"', 'Executing step "return_one" in subprocess.', 'Starting initialization of resources', 'Finished initialization of resources', '', '', '', '', '', '', '', '', 'Launching subprocess for "add"', 'Executing step "add" in subprocess', 'Starting initialization of resources', 'Finished initialization of resources', 'Multiprocess executor: parent process exiting', 'Process for run exited']
    else:
        messages = ['Started process for run', 'Executing steps in process', 'Starting initialization of resources', 'Finished initialization of resources', 'Finished steps in process', 'Process for run exited']
    events_iter = iter(engine_events)
    assert len(engine_events) == len(messages)
    for message in messages:
        next_log = next(events_iter)
        assert message in next_log.message

def test_not_initialized():
    if False:
        for i in range(10):
            print('nop')
    run_launcher = DefaultRunLauncher()
    run_id = 'dummy'
    assert run_launcher.join() is None
    assert run_launcher.terminate(run_id) is False