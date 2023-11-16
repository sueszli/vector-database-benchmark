import eventlet
import mock
from st2tests import config as test_config
test_config.parse_args()
from st2actions.cmd.scheduler import _run_scheduler
from st2actions.scheduler.handler import ActionExecutionSchedulingQueueHandler
from st2actions.scheduler.entrypoint import SchedulerEntrypoint
from st2tests.base import CleanDbTestCase
__all__ = ['SchedulerServiceEntryPointTestCase']

def mock_handler_run(self):
    if False:
        print('Hello World!')
    eventlet.sleep(0.2)
    raise Exception('handler run exception')

def mock_handler_cleanup(self):
    if False:
        for i in range(10):
            print('nop')
    eventlet.sleep(0.2)
    raise Exception('handler clean exception')

def mock_entrypoint_start(self):
    if False:
        print('Hello World!')
    eventlet.sleep(0.2)
    raise Exception('entrypoint start exception')

class SchedulerServiceEntryPointTestCase(CleanDbTestCase):

    @mock.patch.object(ActionExecutionSchedulingQueueHandler, 'run', mock_handler_run)
    @mock.patch('st2actions.cmd.scheduler.LOG')
    def test_service_exits_correctly_on_fatal_exception_in_handler_run(self, mock_log):
        if False:
            while True:
                i = 10
        run_thread = eventlet.spawn(_run_scheduler)
        result = run_thread.wait()
        self.assertEqual(result, 1)
        mock_log_exception_call = mock_log.exception.call_args_list[0][0][0]
        self.assertIn('Scheduler unexpectedly stopped', mock_log_exception_call)

    @mock.patch.object(ActionExecutionSchedulingQueueHandler, 'cleanup', mock_handler_cleanup)
    @mock.patch('st2actions.cmd.scheduler.LOG')
    def test_service_exits_correctly_on_fatal_exception_in_handler_cleanup(self, mock_log):
        if False:
            while True:
                i = 10
        run_thread = eventlet.spawn(_run_scheduler)
        result = run_thread.wait()
        self.assertEqual(result, 1)
        mock_log_exception_call = mock_log.exception.call_args_list[0][0][0]
        self.assertIn('Scheduler unexpectedly stopped', mock_log_exception_call)

    @mock.patch.object(SchedulerEntrypoint, 'start', mock_entrypoint_start)
    @mock.patch('st2actions.cmd.scheduler.LOG')
    def test_service_exits_correctly_on_fatal_exception_in_entrypoint_start(self, mock_log):
        if False:
            return 10
        run_thread = eventlet.spawn(_run_scheduler)
        result = run_thread.wait()
        self.assertEqual(result, 1)
        mock_log_exception_call = mock_log.exception.call_args_list[0][0][0]
        self.assertIn('Scheduler unexpectedly stopped', mock_log_exception_call)