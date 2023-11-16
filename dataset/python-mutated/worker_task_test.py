import multiprocessing
from subprocess import check_call
import sys
from helpers import LuigiTestCase, StringContaining
import mock
from psutil import Process
from time import sleep
import luigi
import luigi.date_interval
import luigi.notifications
from luigi.worker import TaskException, TaskProcess
from luigi.scheduler import DONE, FAILED
luigi.notifications.DEBUG = True

class WorkerTaskTest(LuigiTestCase):

    def test_constructor(self):
        if False:
            while True:
                i = 10

        class MyTask(luigi.Task):

            def __init__(self):
                if False:
                    print('Hello World!')
                pass

        def f():
            if False:
                return 10
            luigi.build([MyTask()], local_scheduler=True)
        self.assertRaises(TaskException, f)

    def test_run_none(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                while True:
                    i = 10
            luigi.build([None], local_scheduler=True)
        self.assertRaises(TaskException, f)

class TaskProcessTest(LuigiTestCase):

    def test_update_result_queue_on_success(self):
        if False:
            while True:
                i = 10

        class SuccessTask(luigi.Task):

            def on_success(self):
                if False:
                    while True:
                        i = 10
                return 'test success expl'
        task = SuccessTask()
        result_queue = multiprocessing.Queue()
        task_process = TaskProcess(task, 1, result_queue, mock.Mock())
        with mock.patch.object(result_queue, 'put') as mock_put:
            task_process.run()
            mock_put.assert_called_once_with((task.task_id, DONE, 'test success expl', [], None))

    def test_update_result_queue_on_failure(self):
        if False:
            return 10

        class FailTask(luigi.Task):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                raise BaseException('Uh oh.')

            def on_failure(self, exception):
                if False:
                    i = 10
                    return i + 15
                return 'test failure expl'
        task = FailTask()
        result_queue = multiprocessing.Queue()
        task_process = TaskProcess(task, 1, result_queue, mock.Mock())
        with mock.patch.object(result_queue, 'put') as mock_put:
            task_process.run()
            mock_put.assert_called_once_with((task.task_id, FAILED, 'test failure expl', [], []))

    def test_fail_on_false_complete(self):
        if False:
            print('Hello World!')

        class NeverCompleteTask(luigi.Task):

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                return False
        task = NeverCompleteTask()
        result_queue = multiprocessing.Queue()
        task_process = TaskProcess(task, 1, result_queue, mock.Mock(), check_complete_on_run=True)
        with mock.patch.object(result_queue, 'put') as mock_put:
            task_process.run()
            mock_put.assert_called_once_with((task.task_id, FAILED, StringContaining('finished running, but complete() is still returning false'), [], None))

    def test_cleanup_children_on_terminate(self):
        if False:
            i = 10
            return i + 15
        '\n        Subprocesses spawned by tasks should be terminated on terminate\n        '

        class HangingSubprocessTask(luigi.Task):

            def run(self):
                if False:
                    while True:
                        i = 10
                python = sys.executable
                check_call([python, '-c', 'while True: pass'])
        task = HangingSubprocessTask()
        queue = mock.Mock()
        worker_id = 1
        task_process = TaskProcess(task, worker_id, queue, mock.Mock())
        task_process.start()
        parent = Process(task_process.pid)
        while not parent.children():
            sleep(0.01)
        [child] = parent.children()
        task_process.terminate()
        child.wait(timeout=1.0)
        self.assertFalse(parent.is_running())
        self.assertFalse(child.is_running())

    def test_disable_worker_timeout(self):
        if False:
            print('Hello World!')
        '\n        When a task sets worker_timeout explicitly to 0, it should disable the timeout, even if it\n        is configured globally.\n        '

        class Task(luigi.Task):
            worker_timeout = 0
        task_process = TaskProcess(task=Task(), worker_id=1, result_queue=mock.Mock(), status_reporter=mock.Mock(), worker_timeout=10)
        self.assertEqual(task_process.worker_timeout, 0)