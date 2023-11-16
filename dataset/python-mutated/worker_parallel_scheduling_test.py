import contextlib
import gc
import os
import pickle
import time
from helpers import unittest
import mock
import psutil
import luigi
from luigi.worker import Worker
from luigi.task_status import UNKNOWN

def running_children():
    if False:
        return 10
    children = set()
    process = psutil.Process(os.getpid())
    for child in process.children():
        if child.is_running():
            children.add(child.pid)
    return children

@contextlib.contextmanager
def pause_gc():
    if False:
        print('Hello World!')
    if not gc.isenabled():
        yield
    try:
        gc.disable()
        yield
    finally:
        gc.enable()

class SlowCompleteWrapper(luigi.WrapperTask):

    def requires(self):
        if False:
            i = 10
            return i + 15
        return [SlowCompleteTask(i) for i in range(4)]

class SlowCompleteTask(luigi.Task):
    n = luigi.IntParameter()

    def complete(self):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.1)
        return True

class OverlappingSelfDependenciesTask(luigi.Task):
    n = luigi.IntParameter()
    k = luigi.IntParameter()

    def complete(self):
        if False:
            while True:
                i = 10
        return self.n < self.k or self.k == 0

    def requires(self):
        if False:
            return 10
        return [OverlappingSelfDependenciesTask(self.n - 1, k) for k in range(self.k + 1)]

class ExceptionCompleteTask(luigi.Task):

    def complete(self):
        if False:
            while True:
                i = 10
        assert False

class ExceptionRequiresTask(luigi.Task):

    def requires(self):
        if False:
            i = 10
            return i + 15
        assert False

class UnpicklableExceptionTask(luigi.Task):

    def complete(self):
        if False:
            print('Hello World!')

        class UnpicklableException(Exception):
            pass
        raise UnpicklableException()

class ParallelSchedulingTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.sch = mock.Mock()
        self.w = Worker(scheduler=self.sch, worker_id='x')

    def added_tasks(self, status):
        if False:
            for i in range(10):
                print('nop')
        return [kw['task_id'] for (args, kw) in self.sch.add_task.call_args_list if kw['status'] == status]

    def test_number_of_processes(self):
        if False:
            print('Hello World!')
        import multiprocessing
        real_pool = multiprocessing.Pool(1)
        with mock.patch('multiprocessing.Pool') as mocked_pool:
            mocked_pool.return_value = real_pool
            self.w.add(OverlappingSelfDependenciesTask(n=1, k=1), multiprocess=True, processes=1234)
            mocked_pool.assert_called_once_with(processes=1234)

    def test_zero_processes(self):
        if False:
            i = 10
            return i + 15
        import multiprocessing
        real_pool = multiprocessing.Pool(1)
        with mock.patch('multiprocessing.Pool') as mocked_pool:
            mocked_pool.return_value = real_pool
            self.w.add(OverlappingSelfDependenciesTask(n=1, k=1), multiprocess=True, processes=0)
            mocked_pool.assert_called_once_with(processes=None)

    def test_children_terminated(self):
        if False:
            print('Hello World!')
        before_children = running_children()
        with pause_gc():
            self.w.add(OverlappingSelfDependenciesTask(5, 2), multiprocess=True)
            self.assertLessEqual(running_children(), before_children)

    def test_multiprocess_scheduling_with_overlapping_dependencies(self):
        if False:
            i = 10
            return i + 15
        self.w.add(OverlappingSelfDependenciesTask(5, 2), True)
        self.assertEqual(15, self.sch.add_task.call_count)
        self.assertEqual(set((OverlappingSelfDependenciesTask(n=1, k=1).task_id, OverlappingSelfDependenciesTask(n=2, k=1).task_id, OverlappingSelfDependenciesTask(n=2, k=2).task_id, OverlappingSelfDependenciesTask(n=3, k=1).task_id, OverlappingSelfDependenciesTask(n=3, k=2).task_id, OverlappingSelfDependenciesTask(n=4, k=1).task_id, OverlappingSelfDependenciesTask(n=4, k=2).task_id, OverlappingSelfDependenciesTask(n=5, k=2).task_id)), set(self.added_tasks('PENDING')))
        self.assertEqual(set((OverlappingSelfDependenciesTask(n=0, k=0).task_id, OverlappingSelfDependenciesTask(n=0, k=1).task_id, OverlappingSelfDependenciesTask(n=1, k=0).task_id, OverlappingSelfDependenciesTask(n=1, k=2).task_id, OverlappingSelfDependenciesTask(n=2, k=0).task_id, OverlappingSelfDependenciesTask(n=3, k=0).task_id, OverlappingSelfDependenciesTask(n=4, k=0).task_id)), set(self.added_tasks('DONE')))

    @mock.patch('luigi.notifications.send_error_email')
    def test_raise_exception_in_complete(self, send):
        if False:
            return 10
        self.w.add(ExceptionCompleteTask(), multiprocess=True)
        send.check_called_once()
        self.assertEqual(UNKNOWN, self.sch.add_task.call_args[1]['status'])
        self.assertFalse(self.sch.add_task.call_args[1]['runnable'])
        self.assertTrue('assert False' in send.call_args[0][1])

    @mock.patch('luigi.notifications.send_error_email')
    def test_raise_unpicklable_exception_in_complete(self, send):
        if False:
            while True:
                i = 10
        self.assertRaises(Exception, UnpicklableExceptionTask().complete)
        try:
            UnpicklableExceptionTask().complete()
        except Exception as e:
            ex = e
        self.assertRaises((pickle.PicklingError, AttributeError), pickle.dumps, ex)
        self.w.add(UnpicklableExceptionTask(), multiprocess=True)
        send.check_called_once()
        self.assertEqual(UNKNOWN, self.sch.add_task.call_args[1]['status'])
        self.assertFalse(self.sch.add_task.call_args[1]['runnable'])
        self.assertTrue('raise UnpicklableException()' in send.call_args[0][1])

    @mock.patch('luigi.notifications.send_error_email')
    def test_raise_exception_in_requires(self, send):
        if False:
            i = 10
            return i + 15
        self.w.add(ExceptionRequiresTask(), multiprocess=True)
        send.check_called_once()
        self.assertEqual(UNKNOWN, self.sch.add_task.call_args[1]['status'])
        self.assertFalse(self.sch.add_task.call_args[1]['runnable'])