import sys
import luigi
import luigi.date_interval
import luigi.notifications
from luigi.interface import _WorkerSchedulerFactory
from luigi.worker import Worker
from luigi.interface import core
from luigi.execution_summary import LuigiStatusCode
from mock import Mock, patch, MagicMock
from helpers import LuigiTestCase, with_config
luigi.notifications.DEBUG = True

class InterfaceTest(LuigiTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.worker = Worker()
        self.worker_scheduler_factory = _WorkerSchedulerFactory()
        self.worker_scheduler_factory.create_worker = Mock(return_value=self.worker)
        self.worker_scheduler_factory.create_local_scheduler = Mock()
        super(InterfaceTest, self).setUp()

        class NoOpTask(luigi.Task):
            param = luigi.Parameter()
        self.task_a = NoOpTask('a')
        self.task_b = NoOpTask('b')

    def _create_summary_dict_with(self, updates={}):
        if False:
            while True:
                i = 10
        summary_dict = {'completed': set(), 'already_done': set(), 'ever_failed': set(), 'failed': set(), 'scheduling_error': set(), 'still_pending_ext': set(), 'still_pending_not_ext': set(), 'run_by_other_worker': set(), 'upstream_failure': set(), 'upstream_missing_dependency': set(), 'upstream_run_by_other_worker': set(), 'upstream_scheduling_error': set(), 'not_run': set()}
        summary_dict.update(updates)
        return summary_dict

    def _summary_dict_module_path():
        if False:
            i = 10
            return i + 15
        return 'luigi.execution_summary._summary_dict'

    def test_interface_run_positive_path(self):
        if False:
            while True:
                i = 10
        self.worker.add = Mock(side_effect=[True, True])
        self.worker.run = Mock(return_value=True)
        self.assertTrue(self._run_interface())

    def test_interface_run_positive_path_with_detailed_summary_enabled(self):
        if False:
            while True:
                i = 10
        self.worker.add = Mock(side_effect=[True, True])
        self.worker.run = Mock(return_value=True)
        self.assertTrue(self._run_interface(detailed_summary=True).scheduling_succeeded)

    def test_interface_run_with_add_failure(self):
        if False:
            i = 10
            return i + 15
        self.worker.add = Mock(side_effect=[True, False])
        self.worker.run = Mock(return_value=True)
        self.assertFalse(self._run_interface())

    def test_interface_run_with_add_failure_with_detailed_summary_enabled(self):
        if False:
            print('Hello World!')
        self.worker.add = Mock(side_effect=[True, False])
        self.worker.run = Mock(return_value=True)
        self.assertFalse(self._run_interface(detailed_summary=True).scheduling_succeeded)

    def test_interface_run_with_run_failure(self):
        if False:
            i = 10
            return i + 15
        self.worker.add = Mock(side_effect=[True, True])
        self.worker.run = Mock(return_value=False)
        self.assertFalse(self._run_interface())

    def test_interface_run_with_run_failure_with_detailed_summary_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        self.worker.add = Mock(side_effect=[True, True])
        self.worker.run = Mock(return_value=False)
        self.assertFalse(self._run_interface(detailed_summary=True).scheduling_succeeded)

    @patch(_summary_dict_module_path())
    def test_that_status_is_success(self, fake_summary_dict):
        if False:
            for i in range(10):
                print('nop')
        fake_summary_dict.return_value = self._create_summary_dict_with()
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.SUCCESS)

    @patch(_summary_dict_module_path())
    def test_that_status_is_success_with_retry(self, fake_summary_dict):
        if False:
            while True:
                i = 10
        fake_summary_dict.return_value = self._create_summary_dict_with({'ever_failed': [self.task_a]})
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.SUCCESS_WITH_RETRY)

    @patch(_summary_dict_module_path())
    def test_that_status_is_failed_when_there_is_one_failed_task(self, fake_summary_dict):
        if False:
            while True:
                i = 10
        fake_summary_dict.return_value = self._create_summary_dict_with({'ever_failed': [self.task_a], 'failed': [self.task_a]})
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.FAILED)

    @patch(_summary_dict_module_path())
    def test_that_status_is_failed_with_scheduling_failure(self, fake_summary_dict):
        if False:
            print('Hello World!')
        fake_summary_dict.return_value = self._create_summary_dict_with({'ever_failed': [self.task_a], 'failed': [self.task_a], 'scheduling_error': [self.task_b]})
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.FAILED_AND_SCHEDULING_FAILED)

    @patch(_summary_dict_module_path())
    def test_that_status_is_scheduling_failed_with_one_scheduling_error(self, fake_summary_dict):
        if False:
            print('Hello World!')
        fake_summary_dict.return_value = self._create_summary_dict_with({'scheduling_error': [self.task_b]})
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.SCHEDULING_FAILED)

    @patch(_summary_dict_module_path())
    def test_that_status_is_not_run_with_one_task_not_run(self, fake_summary_dict):
        if False:
            while True:
                i = 10
        fake_summary_dict.return_value = self._create_summary_dict_with({'not_run': [self.task_a]})
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.NOT_RUN)

    @patch(_summary_dict_module_path())
    def test_that_status_is_missing_ext_with_one_task_with_missing_external_dependency(self, fake_summary_dict):
        if False:
            for i in range(10):
                print('nop')
        fake_summary_dict.return_value = self._create_summary_dict_with({'still_pending_ext': [self.task_a]})
        luigi_run_result = self._run_interface(detailed_summary=True)
        self.assertEqual(luigi_run_result.status, LuigiStatusCode.MISSING_EXT)

    def test_stops_worker_on_add_exception(self):
        if False:
            i = 10
            return i + 15
        worker = MagicMock()
        self.worker_scheduler_factory.create_worker = Mock(return_value=worker)
        worker.add = Mock(side_effect=AttributeError)
        self.assertRaises(AttributeError, self._run_interface)
        self.assertTrue(worker.__exit__.called)

    def test_stops_worker_on_run_exception(self):
        if False:
            i = 10
            return i + 15
        worker = MagicMock()
        self.worker_scheduler_factory.create_worker = Mock(return_value=worker)
        worker.add = Mock(side_effect=[True, True])
        worker.run = Mock(side_effect=AttributeError)
        self.assertRaises(AttributeError, self._run_interface)
        self.assertTrue(worker.__exit__.called)

    def test_just_run_main_task_cls(self):
        if False:
            return 10

        class MyTestTask(luigi.Task):
            pass

        class MyOtherTestTask(luigi.Task):
            my_param = luigi.Parameter()
        with patch.object(sys, 'argv', ['my_module.py', '--no-lock', '--local-scheduler']):
            luigi.run(main_task_cls=MyTestTask)
        with patch.object(sys, 'argv', ['my_module.py', '--no-lock', '--my-param', 'my_value', '--local-scheduler']):
            luigi.run(main_task_cls=MyOtherTestTask)

    def _run_interface(self, **env_params):
        if False:
            for i in range(10):
                print('nop')
        return luigi.interface.build([self.task_a, self.task_b], worker_scheduler_factory=self.worker_scheduler_factory, **env_params)

class CoreConfigTest(LuigiTestCase):

    @with_config({})
    def test_parallel_scheduling_processes_default(self):
        if False:
            print('Hello World!')
        self.assertEquals(0, core().parallel_scheduling_processes)

    @with_config({'core': {'parallel-scheduling-processes': '1234'}})
    def test_parallel_scheduling_processes(self):
        if False:
            print('Hello World!')
        from luigi.interface import core
        self.assertEquals(1234, core().parallel_scheduling_processes)