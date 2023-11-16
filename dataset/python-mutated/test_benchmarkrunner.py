import time
import unittest.mock as mock
from apps.core.benchmark import benchmarkrunner
from golem.task.taskbase import Task
from golem.testutils import TempDirFixture

class DummyTask(Task):

    def initialize(self, dir_manager):
        if False:
            print('Hello World!')
        pass

    def query_extra_data(self, perf_index, node_id, node_name):
        if False:
            print('Hello World!')
        pass

    def needs_computation(self):
        if False:
            return 10
        pass

    def finished_computation(self):
        if False:
            i = 10
            return i + 15
        pass

    def computation_finished(self, subtask_id, task_result, verification_finished):
        if False:
            return 10
        pass

    def computation_failed(self, subtask_id: str, ban_node: bool=True):
        if False:
            print('Hello World!')
        pass

    def verify_subtask(self, subtask_id):
        if False:
            i = 10
            return i + 15
        pass

    def verify_task(self):
        if False:
            return 10
        pass

    def get_total_tasks(self):
        if False:
            i = 10
            return i + 15
        pass

    def get_active_tasks(self):
        if False:
            print('Hello World!')
        pass

    def get_tasks_left(self):
        if False:
            print('Hello World!')
        pass

    def restart(self):
        if False:
            i = 10
            return i + 15
        pass

    def restart_subtask(self, subtask_id):
        if False:
            while True:
                i = 10
        pass

    def abort(self):
        if False:
            while True:
                i = 10
        pass

    def get_progress(self):
        if False:
            print('Hello World!')
        pass

    def update_task_state(self, task_state):
        if False:
            return 10
        pass

    def get_trust_mod(self, subtask_id):
        if False:
            i = 10
            return i + 15
        pass

    def add_resources(self, resources):
        if False:
            while True:
                i = 10
        pass

    def copy_subtask_results(self, subtask_id, old_subtask_info, results):
        if False:
            print('Hello World!')
        pass

    def query_extra_data_for_test_task(self):
        if False:
            print('Hello World!')
        pass

    def should_accept_client(self, node_id, offer_hash):
        if False:
            while True:
                i = 10
        pass

    def to_dictionary(self):
        if False:
            print('Hello World!')
        pass

    def accept_client(self, node_id, offer_hash, num_subtasks=1):
        if False:
            print('Hello World!')
        pass

class BenchmarkRunnerFixture(TempDirFixture):

    def _success(self):
        if False:
            i = 10
            return i + 15
        'Instance success_callback.'
        pass

    def _error(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Instance error_callback.'
        pass

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.benchmark = mock.MagicMock()
        self.instance = benchmarkrunner.BenchmarkRunner(task=DummyTask(None, None), root_path=self.tempdir, success_callback=self._success, error_callback=self._error, benchmark=self.benchmark)

class TestBenchmarkRunner(BenchmarkRunnerFixture):

    def test_task_thread_getter(self):
        if False:
            return 10
        'When docker_images is empty.'
        ctd = {}
        ctd['docker_images'] = []
        with self.assertRaises(Exception):
            self.instance._get_task_thread(ctd)

    def test_tt_cases(self):
        if False:
            return 10
        'run() with different tt values.'
        with mock.patch.multiple(self.instance, run=mock.DEFAULT, tt=None) as values:
            self.instance.run()
            values['run'].assert_called_once_with()
        with mock.patch.multiple(self.instance, tt=mock.DEFAULT) as values:
            self.instance.run()

    def test_task_computed_immidiately(self):
        if False:
            return 10
        "Special case when start_time and stop_time are identical.\n        It's higly unprobable on *NIX but happens a lot on Windows\n        wich has lower precision of time.time()."
        task_thread = mock.MagicMock()
        task_thread.error = False
        result_dict = {'data': object()}
        task_thread.result = (result_dict, None)
        try:
            self.instance.__class__.start_time = property(lambda self: self.end_time)
            self.instance.success_callback = mock.MagicMock()
            self.benchmark.verify_result.return_value = True
            self.benchmark.normalization_constant = 1
            self.instance.task_computed(task_thread)
            self.instance.success_callback.assert_called_once_with(mock.ANY)
        finally:
            del self.instance.__class__.start_time
        try:
            self.instance.__class__.start_time = property(lambda self: self.end_time + 10)
            self.instance.success_callback = mock.MagicMock()
            self.benchmark.verify_result.return_value = True
            self.benchmark.normalization_constant = 1
            self.instance.task_computed(task_thread)
            self.instance.success_callback.assert_called_once_with(mock.ANY)
        finally:
            del self.instance.__class__.start_time

    def test_task_computed_false_result_and_false_error_msg(self):
        if False:
            print('Hello World!')
        task_thread = mock.MagicMock()
        task_thread.result = None
        task_thread.error = False
        task_thread.error_msg = error_msg = None
        self.instance.error_callback = error_mock = mock.MagicMock()
        self.instance.task_computed(task_thread)
        error_mock.assert_called_once_with(error_msg)

    def test_task_computed_false_resulst_and_non_false_error_msg(self):
        if False:
            for i in range(10):
                print('nop')
        task_thread = mock.MagicMock()
        task_thread.result = None
        task_thread.error = True
        task_thread.error_msg = error_msg = 'dummy error msg:{}'.format(time.time())
        self.instance.error_callback = error_mock = mock.MagicMock()
        self.instance.task_computed(task_thread)
        error_mock.assert_called_once_with(error_msg)

    def test_task_computed_empty_result_dict(self):
        if False:
            print('Hello World!')
        task_thread = mock.MagicMock()
        task_thread.error = False
        result_dict = {}
        task_thread.result = (result_dict, None)
        self.instance.task_computed(task_thread)
        self.assertEqual(self.benchmark.verify_result.call_count, 0)

    def test_task_computed_result_dict_without_res(self):
        if False:
            print('Hello World!')
        task_thread = mock.MagicMock()
        task_thread.error = False
        result_dict = {'a': None}
        task_thread.result = (result_dict, None)
        self.instance.task_computed(task_thread)
        self.assertEqual(self.benchmark.verify_result.call_count, 0)

    def test_task_computed_result_dict_with_data_but_failed_verification(self):
        if False:
            while True:
                i = 10
        task_thread = mock.MagicMock()
        task_thread.error = False
        result_dict = {'data': object()}
        task_thread.result = (result_dict, None)
        self.instance.start_time = time.time()
        self.instance.success_callback = mock.MagicMock()
        self.benchmark.verify_result.return_value = False
        self.instance.task_computed(task_thread)
        self.benchmark.verify_result.assert_called_once_with(result_dict['data'])
        self.assertEqual(self.instance.success_callback.call_count, 0)

    def test_task_computed_result_dict_with_data_and_successful_verification(self):
        if False:
            while True:
                i = 10
        task_thread = mock.MagicMock()
        task_thread.error = False
        result_dict = {'data': object()}
        task_thread.result = (result_dict, None)
        self.instance.start_time = time.time()
        self.instance.success_callback = mock.MagicMock()
        self.benchmark.verify_result.reset_mock()
        self.benchmark.verify_result.return_value = True
        self.benchmark.normalization_constant = 1
        self.instance.task_computed(task_thread)
        self.benchmark.verify_result.assert_called_once_with(result_dict['data'])
        self.instance.success_callback.assert_called_once_with(mock.ANY)

class TestBenchmarkRunnerIsSuccess(BenchmarkRunnerFixture):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.task_thread = mock.MagicMock()
        self.task_thread.error = False
        self.instance.start_time = time.time()
        self.instance.end_time = self.instance.start_time + 4
        self.benchmark.verify_result.return_value = True

    def test_result_is_not_a_tuple(self):
        if False:
            print('Hello World!')
        self.task_thread.result = 5
        assert not self.instance.is_success(self.task_thread)

    def test_result_first_arg_is_none(self):
        if False:
            print('Hello World!')
        self.task_thread.result = (None, 30)
        assert not self.instance.is_success(self.task_thread)

    def test_result_first_arg_doesnt_have_data_in_dictionary(self):
        if False:
            while True:
                i = 10
        self.task_thread.result = ({'abc': 20}, 30)
        assert not self.instance.is_success(self.task_thread)

    def test_is_success(self):
        if False:
            print('Hello World!')
        self.task_thread.result = ({'data': 'some data'}, 30)
        assert self.instance.is_success(self.task_thread)

    def test_end_time_not_measured(self):
        if False:
            for i in range(10):
                print('nop')
        self.instance.end_time = None
        assert not self.instance.is_success(self.task_thread)

    def test_start_time_not_measured(self):
        if False:
            print('Hello World!')
        self.instance.end_time = self.instance.start_time
        self.instance.start_time = None
        assert not self.instance.is_success(self.task_thread)

    def test_not_verified_properly(self):
        if False:
            for i in range(10):
                print('nop')
        self.instance.start_time = self.instance.end_time - 5
        self.benchmark.verify_result.return_value = False
        assert not self.instance.is_success(self.task_thread)

class WrongTask(DummyTask):

    def query_extra_data(self, perf_index, node_id, node_name):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('Wrong task')

class BenchmarkRunnerWrongTaskTest(TempDirFixture):

    def test_run_with_error(self):
        if False:
            print('Hello World!')
        benchmark = mock.MagicMock()
        instance = benchmarkrunner.BenchmarkRunner(task=WrongTask(None, None), root_path=self.tempdir, success_callback=mock.Mock(), error_callback=mock.Mock(), benchmark=benchmark)
        instance.run()
        instance.success_callback.assert_not_called()
        instance.error_callback.assert_called_once()