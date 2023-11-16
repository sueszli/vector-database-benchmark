from unittest.mock import Mock, patch
from golem.task.taskbase import Task
from golem.task.tasktester import TaskTester, logger
from golem.tools.testdirfixture import TestDirFixture
from golem.tools.assertlogs import LogTestCase

class TaskThread:

    def __init__(self, result):
        if False:
            for i in range(10):
                print('nop')
        self.result = result
        self.error = False

class MemTaskThread(TaskThread):

    def __init__(self, result, memory, error):
        if False:
            print('Hello World!')
        TaskThread.__init__(self, (result, memory))
        self.error_msg = error
        self.error = False

    def get_error(self):
        if False:
            while True:
                i = 10
        return self.error

    def get_progress(self):
        if False:
            i = 10
            return i + 15
        return '30%'

@patch.multiple(Task, __abstractmethods__=frozenset())
class TestTaskTester(TestDirFixture, LogTestCase):
    node = 'node1'
    name = 'task1'

    def test_init(self):
        if False:
            while True:
                i = 10
        task = Task(Mock(), Mock())
        task.query_extra_data_for_test_task = Mock()
        self.assertIsNotNone(TaskTester(task, self.path, None, None))

    def test_task_computed(self):
        if False:
            while True:
                i = 10
        task = Task(Mock(), Mock())
        result = [{'data': True}, 123]
        task.header.node_name = self.node
        task.header.task_id = self.name
        task.root_path = self.path
        task.after_test = lambda res, tmp_dir: {}
        task.query_extra_data_for_test_task = Mock()
        tt = TaskTester(task, self.path, Mock(), Mock())
        tt.tmp_dir = self.path
        task_thread = TaskThread(result)
        tt.task_computed(task_thread)
        task_thread = MemTaskThread(None, 30210, 'Some error')
        with self.assertLogs(logger, level='WARNING'):
            tt.task_computed(task_thread)
        tt.error_callback.assert_called_with('Some error')
        task_thread = MemTaskThread('result', 2010, 'Another error')
        self.assertIsNone(tt.get_progress())
        tt.tt = task_thread
        self.assertEqual(tt.get_progress(), '30%')
        task_thread.error = True
        self.assertEqual(tt.get_progress(), 0)
        tt.task_computed(task_thread)
        tt.error_callback.assert_called_with('Another error')
        self.message = ''

        def success_callback(res, est_mem, time_spent, after_test_data):
            if False:
                print('Hello World!')
            self.message = 'Success ' + after_test_data['warnings']
        task.header.node_name = self.node
        task.header.task_id = self.name
        task.root_path = self.path
        task.after_test = lambda res, tmp_dir: {'warnings': 'bla ble'}
        task.query_extra_data_for_test_task = Mock()
        tt = TaskTester(task, self.path, success_callback, None)
        tt.tmp_dir = self.path
        task_thread = TaskThread(result)
        tt.task_computed(task_thread)
        self.assertTrue('bla' in self.message)
        self.assertTrue('ble' in self.message)

    def test_is_success(self):
        if False:
            print('Hello World!')
        task = Task(Mock(), Mock())
        task.query_extra_data_for_test_task = Mock()
        tt = TaskTester(task, self.path, Mock(), Mock())
        task_thread = Mock()
        task_thread.error = None
        task_thread.result = ({'data': True}, 123)
        assert tt.is_success(task_thread)
        task_thread.result = {'data': True}
        assert not tt.is_success(task_thread)
        assert task_thread.error == 'Wrong result format'