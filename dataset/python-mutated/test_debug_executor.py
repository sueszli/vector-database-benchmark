from __future__ import annotations
from unittest import mock
from unittest.mock import MagicMock
from airflow.executors.debug_executor import DebugExecutor
from airflow.utils.state import State

class TestDebugExecutor:

    @mock.patch('airflow.executors.debug_executor.DebugExecutor._run_task')
    def test_sync(self, run_task_mock):
        if False:
            for i in range(10):
                print('nop')
        run_task_mock.return_value = True
        executor = DebugExecutor()
        ti1 = MagicMock(key='t1')
        ti2 = MagicMock(key='t2')
        executor.tasks_to_run = [ti1, ti2]
        executor.sync()
        assert not executor.tasks_to_run
        run_task_mock.assert_has_calls([mock.call(ti1), mock.call(ti2)])

    @mock.patch('airflow.models.taskinstance.TaskInstance')
    def test_run_task(self, task_instance_mock):
        if False:
            return 10
        ti_key = 'key'
        job_id = ' job_id'
        task_instance_mock.key = ti_key
        task_instance_mock.job_id = job_id
        executor = DebugExecutor()
        executor.running = {ti_key}
        succeeded = executor._run_task(task_instance_mock)
        assert succeeded
        task_instance_mock.run.assert_called_once_with(job_id=job_id)

    def test_queue_task_instance(self):
        if False:
            print('Hello World!')
        key = 'ti_key'
        ti = MagicMock(key=key)
        executor = DebugExecutor()
        executor.queue_task_instance(task_instance=ti, mark_success=True, pool='pool')
        assert key in executor.queued_tasks
        assert key in executor.tasks_params
        assert executor.tasks_params[key] == {'mark_success': True, 'pool': 'pool'}

    def test_trigger_tasks(self):
        if False:
            print('Hello World!')
        execute_mock = MagicMock()
        executor = DebugExecutor()
        executor.execute_async = execute_mock
        executor.queued_tasks = {'t1': (None, 1, None, MagicMock(key='t1')), 't2': (None, 2, None, MagicMock(key='t2'))}
        executor.trigger_tasks(open_slots=4)
        assert not executor.queued_tasks
        assert len(executor.running) == 2
        assert len(executor.tasks_to_run) == 2
        assert not execute_mock.called

    def test_end(self):
        if False:
            for i in range(10):
                print('nop')
        ti = MagicMock(key='ti_key')
        executor = DebugExecutor()
        executor.tasks_to_run = [ti]
        executor.running = {ti.key}
        executor.end()
        ti.set_state.assert_called_once_with(State.UPSTREAM_FAILED)
        assert not executor.running

    @mock.patch('airflow.executors.debug_executor.DebugExecutor.change_state')
    def test_fail_fast(self, change_state_mock):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict('os.environ', {'AIRFLOW__DEBUG__FAIL_FAST': 'True'}):
            executor = DebugExecutor()
        ti1 = MagicMock(key='t1')
        ti2 = MagicMock(key='t2')
        ti1.run.side_effect = Exception
        executor.tasks_to_run = [ti1, ti2]
        executor.sync()
        assert executor.fail_fast
        assert not executor.tasks_to_run
        change_state_mock.assert_has_calls([mock.call(ti1.key, State.FAILED), mock.call(ti2.key, State.UPSTREAM_FAILED)])

    def test_reschedule_mode(self):
        if False:
            return 10
        assert DebugExecutor.change_sensor_mode_to_reschedule

    def test_is_single_threaded(self):
        if False:
            i = 10
            return i + 15
        assert DebugExecutor.is_single_threaded

    def test_is_production_default_value(self):
        if False:
            i = 10
            return i + 15
        assert not DebugExecutor.is_production

    @mock.patch('time.sleep', autospec=True)
    def test_trigger_sleep_when_no_task(self, mock_sleep):
        if False:
            for i in range(10):
                print('nop')
        execute_mock = MagicMock()
        executor = DebugExecutor()
        executor.execute_async = execute_mock
        executor.queued_tasks = {}
        executor.trigger_tasks(open_slots=5)
        mock_sleep.assert_called()

    @mock.patch('airflow.executors.debug_executor.DebugExecutor.change_state')
    def test_sync_after_terminate(self, change_state_mock):
        if False:
            return 10
        executor = DebugExecutor()
        ti1 = MagicMock(key='t1')
        executor.tasks_to_run = [ti1]
        executor.terminate()
        executor.sync()
        change_state_mock.assert_has_calls([mock.call(ti1.key, State.FAILED)])