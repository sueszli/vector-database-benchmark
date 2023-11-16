from __future__ import annotations
import logging
import sys
from datetime import timedelta
from unittest import mock
import pendulum
import pytest
import time_machine
from airflow.callbacks.callback_requests import CallbackRequest
from airflow.cli.cli_config import DefaultHelpParser, GroupCommand
from airflow.cli.cli_parser import AirflowHelpFormatter
from airflow.executors.base_executor import BaseExecutor, RunningRetryAttemptType
from airflow.models.baseoperator import BaseOperator
from airflow.models.taskinstance import TaskInstance, TaskInstanceKey
from airflow.utils import timezone
from airflow.utils.state import State

def test_supports_sentry():
    if False:
        return 10
    assert not BaseExecutor.supports_sentry

def test_supports_pickling():
    if False:
        i = 10
        return i + 15
    assert BaseExecutor.supports_pickling

def test_is_local_default_value():
    if False:
        i = 10
        return i + 15
    assert not BaseExecutor.is_local

def test_is_single_threaded_default_value():
    if False:
        return 10
    assert not BaseExecutor.is_single_threaded

def test_is_production_default_value():
    if False:
        for i in range(10):
            print('nop')
    assert BaseExecutor.is_production

def test_infinite_slotspool():
    if False:
        print('Hello World!')
    executor = BaseExecutor(0)
    assert executor.slots_available == sys.maxsize

def test_get_task_log():
    if False:
        while True:
            i = 10
    executor = BaseExecutor()
    ti = TaskInstance(task=BaseOperator(task_id='dummy'))
    assert executor.get_task_log(ti=ti, try_number=1) == ([], [])

def test_serve_logs_default_value():
    if False:
        print('Hello World!')
    assert not BaseExecutor.serve_logs

def test_no_cli_commands_vended():
    if False:
        print('Hello World!')
    assert not BaseExecutor.get_cli_commands()

def test_get_event_buffer():
    if False:
        for i in range(10):
            print('nop')
    executor = BaseExecutor()
    date = timezone.utcnow()
    try_number = 1
    key1 = TaskInstanceKey('my_dag1', 'my_task1', date, try_number)
    key2 = TaskInstanceKey('my_dag2', 'my_task1', date, try_number)
    key3 = TaskInstanceKey('my_dag2', 'my_task2', date, try_number)
    state = State.SUCCESS
    executor.event_buffer[key1] = (state, None)
    executor.event_buffer[key2] = (state, None)
    executor.event_buffer[key3] = (state, None)
    assert len(executor.get_event_buffer(('my_dag1',))) == 1
    assert len(executor.get_event_buffer()) == 2
    assert len(executor.event_buffer) == 0

def test_fail_and_success():
    if False:
        print('Hello World!')
    executor = BaseExecutor()
    date = timezone.utcnow()
    try_number = 1
    success_state = State.SUCCESS
    fail_state = State.FAILED
    key1 = TaskInstanceKey('my_dag1', 'my_task1', date, try_number)
    key2 = TaskInstanceKey('my_dag2', 'my_task1', date, try_number)
    key3 = TaskInstanceKey('my_dag2', 'my_task2', date, try_number)
    executor.fail(key1, fail_state)
    executor.fail(key2, fail_state)
    executor.success(key3, success_state)
    assert len(executor.running) == 0
    assert len(executor.get_event_buffer()) == 3

@mock.patch('airflow.executors.base_executor.BaseExecutor.sync')
@mock.patch('airflow.executors.base_executor.BaseExecutor.trigger_tasks')
@mock.patch('airflow.executors.base_executor.Stats.gauge')
def test_gauge_executor_metrics(mock_stats_gauge, mock_trigger_tasks, mock_sync):
    if False:
        while True:
            i = 10
    executor = BaseExecutor()
    executor.heartbeat()
    calls = [mock.call('executor.open_slots', value=mock.ANY, tags={'status': 'open', 'name': 'BaseExecutor'}), mock.call('executor.queued_tasks', value=mock.ANY, tags={'status': 'queued', 'name': 'BaseExecutor'}), mock.call('executor.running_tasks', value=mock.ANY, tags={'status': 'running', 'name': 'BaseExecutor'})]
    mock_stats_gauge.assert_has_calls(calls)

@mock.patch('airflow.executors.base_executor.BaseExecutor.sync')
@mock.patch('airflow.executors.base_executor.BaseExecutor.trigger_tasks')
@mock.patch('airflow.executors.base_executor.Stats.gauge')
def test_gauge_executor_with_infinite_pool_metrics(mock_stats_gauge, mock_trigger_tasks, mock_sync):
    if False:
        for i in range(10):
            print('nop')
    executor = BaseExecutor(0)
    executor.heartbeat()
    calls = [mock.call('executor.open_slots', value=mock.ANY, tags={'status': 'open', 'name': 'BaseExecutor'}), mock.call('executor.queued_tasks', value=mock.ANY, tags={'status': 'queued', 'name': 'BaseExecutor'}), mock.call('executor.running_tasks', value=mock.ANY, tags={'status': 'running', 'name': 'BaseExecutor'})]
    mock_stats_gauge.assert_has_calls(calls)

def setup_dagrun(dag_maker):
    if False:
        return 10
    date = timezone.utcnow()
    start_date = date - timedelta(days=2)
    with dag_maker('test_try_adopt_task_instances'):
        BaseOperator(task_id='task_1', start_date=start_date)
        BaseOperator(task_id='task_2', start_date=start_date)
        BaseOperator(task_id='task_3', start_date=start_date)
    return dag_maker.create_dagrun(execution_date=date)

@pytest.mark.db_test
def test_try_adopt_task_instances(dag_maker):
    if False:
        return 10
    dagrun = setup_dagrun(dag_maker)
    tis = dagrun.task_instances
    assert {ti.task_id for ti in tis} == {'task_1', 'task_2', 'task_3'}
    assert BaseExecutor().try_adopt_task_instances(tis) == tis

def enqueue_tasks(executor, dagrun):
    if False:
        print('Hello World!')
    for task_instance in dagrun.task_instances:
        executor.queue_command(task_instance, ['airflow'])

def setup_trigger_tasks(dag_maker):
    if False:
        return 10
    dagrun = setup_dagrun(dag_maker)
    executor = BaseExecutor()
    executor.execute_async = mock.Mock()
    enqueue_tasks(executor, dagrun)
    return (executor, dagrun)

@pytest.mark.db_test
@pytest.mark.parametrize('open_slots', [1, 2, 3])
def test_trigger_queued_tasks(dag_maker, open_slots):
    if False:
        return 10
    (executor, _) = setup_trigger_tasks(dag_maker)
    executor.trigger_tasks(open_slots)
    assert executor.execute_async.call_count == open_slots

@pytest.mark.db_test
@pytest.mark.parametrize('can_try_num, change_state_num, second_exec', [(2, 3, False), (3, 3, True), (4, 3, True)])
@mock.patch('airflow.executors.base_executor.RunningRetryAttemptType.can_try_again')
def test_trigger_running_tasks(can_try_mock, dag_maker, can_try_num, change_state_num, second_exec):
    if False:
        print('Hello World!')
    can_try_mock.side_effect = [True for _ in range(can_try_num)] + [False]
    (executor, dagrun) = setup_trigger_tasks(dag_maker)
    open_slots = 100
    executor.trigger_tasks(open_slots)
    expected_calls = len(dagrun.task_instances)
    assert executor.execute_async.call_count == expected_calls
    ti = dagrun.task_instances[0]
    assert ti.key in executor.running
    assert ti.key not in executor.queued_tasks
    executor.queue_command(ti, ['airflow'])
    assert ti.key in executor.queued_tasks and ti.key in executor.running
    assert len(executor.attempts) == 0
    executor.trigger_tasks(open_slots)
    assert len(executor.attempts) == 1
    assert ti.key in executor.attempts
    for attempt in range(2, change_state_num + 2):
        executor.trigger_tasks(open_slots)
        if attempt <= min(can_try_num, change_state_num):
            assert ti.key in executor.queued_tasks and ti.key in executor.running
        if attempt == change_state_num:
            executor.change_state(ti.key, State.SUCCESS)
            assert ti.key not in executor.running
    if can_try_num >= change_state_num:
        assert ti.key in executor.running
    else:
        assert ti.key not in executor.running
    assert ti.key not in executor.queued_tasks
    assert not executor.attempts
    if second_exec is True:
        expected_calls += 1
    assert executor.execute_async.call_count == expected_calls

@pytest.mark.db_test
def test_validate_airflow_tasks_run_command(dag_maker):
    if False:
        i = 10
        return i + 15
    dagrun = setup_dagrun(dag_maker)
    tis = dagrun.task_instances
    print(f'command: {tis[0].command_as_list()}')
    (dag_id, task_id) = BaseExecutor.validate_airflow_tasks_run_command(tis[0].command_as_list())
    print(f'dag_id: {dag_id}, task_id: {task_id}')
    assert dag_id == dagrun.dag_id and task_id == tis[0].task_id

@pytest.mark.db_test
@mock.patch('airflow.models.taskinstance.TaskInstance.generate_command', return_value=['airflow', 'tasks', 'run', '--test_dag', '--test_task'])
def test_validate_airflow_tasks_run_command_with_complete_forloop(generate_command_mock, dag_maker):
    if False:
        return 10
    dagrun = setup_dagrun(dag_maker)
    tis = dagrun.task_instances
    (dag_id, task_id) = BaseExecutor.validate_airflow_tasks_run_command(tis[0].command_as_list())
    assert dag_id is None and task_id is None

@pytest.mark.db_test
@mock.patch('airflow.models.taskinstance.TaskInstance.generate_command', return_value=['airflow', 'task', 'run'])
def test_invalid_airflow_tasks_run_command(generate_command_mock, dag_maker):
    if False:
        return 10
    dagrun = setup_dagrun(dag_maker)
    tis = dagrun.task_instances
    with pytest.raises(ValueError):
        BaseExecutor.validate_airflow_tasks_run_command(tis[0].command_as_list())

@pytest.mark.db_test
@mock.patch('airflow.models.taskinstance.TaskInstance.generate_command', return_value=['airflow', 'tasks', 'run'])
def test_empty_airflow_tasks_run_command(generate_command_mock, dag_maker):
    if False:
        i = 10
        return i + 15
    dagrun = setup_dagrun(dag_maker)
    tis = dagrun.task_instances
    (dag_id, task_id) = BaseExecutor.validate_airflow_tasks_run_command(tis[0].command_as_list())
    assert dag_id is None, task_id is None

@pytest.mark.db_test
def test_deprecate_validate_api(dag_maker):
    if False:
        i = 10
        return i + 15
    dagrun = setup_dagrun(dag_maker)
    tis = dagrun.task_instances
    with pytest.warns(DeprecationWarning):
        BaseExecutor.validate_command(tis[0].command_as_list())

def test_debug_dump(caplog):
    if False:
        i = 10
        return i + 15
    executor = BaseExecutor()
    with caplog.at_level(logging.INFO):
        executor.debug_dump()
    assert 'executor.queued' in caplog.text
    assert 'executor.running' in caplog.text
    assert 'executor.event_buffer' in caplog.text

def test_base_executor_cannot_send_callback():
    if False:
        return 10
    cbr = CallbackRequest('some_file_path_for_callback')
    executor = BaseExecutor()
    with pytest.raises(ValueError):
        executor.send_callback(cbr)

def test_parser_and_formatter_class():
    if False:
        return 10
    executor = BaseExecutor()
    parser = executor._get_parser()
    assert isinstance(parser, DefaultHelpParser)
    assert parser.formatter_class is AirflowHelpFormatter

@mock.patch('airflow.cli.cli_parser._add_command')
@mock.patch('airflow.executors.base_executor.BaseExecutor.get_cli_commands', return_value=[GroupCommand(name='some_name', help='some_help', subcommands=['A', 'B', 'C'], description='some_description', epilog='some_epilog')])
def test_parser_add_command(mock_add_command, mock_get_cli_command):
    if False:
        print('Hello World!')
    executor = BaseExecutor()
    executor._get_parser()
    mock_add_command.assert_called_once()

@pytest.mark.parametrize('loop_duration, total_tries', [(0.5, 12), (1.0, 7), (1.7, 4), (10, 2)])
def test_running_retry_attempt_type(loop_duration, total_tries):
    if False:
        print('Hello World!')
    '\n    Verify can_try_again returns True until at least 5 seconds have passed.\n\n    For faster loops, we total tries will be higher.  If loops take longer than 5 seconds, still should\n    end up trying 2 times.\n    '
    min_seconds_for_test = 5
    with time_machine.travel(pendulum.now('UTC'), tick=False) as t:
        RunningRetryAttemptType.MIN_SECONDS = min_seconds_for_test
        a = RunningRetryAttemptType()
        while True:
            if not a.can_try_again():
                break
            t.shift(loop_duration)
        assert a.elapsed > min_seconds_for_test
    assert a.total_tries == total_tries
    assert a.tries_after_min == 1