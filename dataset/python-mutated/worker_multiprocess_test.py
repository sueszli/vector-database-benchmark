import logging
from helpers import unittest
import luigi.notifications
import luigi.worker
from luigi import Parameter, RemoteScheduler, Task
from luigi.worker import Worker
from mock import Mock
luigi.notifications.DEBUG = True

class DummyTask(Task):
    param = Parameter()

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(DummyTask, self).__init__(*args, **kwargs)
        self.has_run = False

    def complete(self):
        if False:
            print('Hello World!')
        old_value = self.has_run
        self.has_run = True
        return old_value

    def run(self):
        if False:
            while True:
                i = 10
        logging.debug('%s - setting has_run', self)
        self.has_run = True

class MultiprocessWorkerTest(unittest.TestCase):

    def run(self, result=None):
        if False:
            while True:
                i = 10
        self.scheduler = RemoteScheduler()
        self.scheduler.add_worker = Mock()
        self.scheduler.add_task = Mock()
        with Worker(scheduler=self.scheduler, worker_id='X', worker_processes=2) as worker:
            self.worker = worker
            super(MultiprocessWorkerTest, self).run(result)

    def gw_res(self, pending, task_id):
        if False:
            i = 10
            return i + 15
        return dict(n_pending_tasks=pending, task_id=task_id, running_tasks=0, n_unique_pending=0)

    def test_positive_path(self):
        if False:
            return 10
        a = DummyTask('a')
        b = DummyTask('b')

        class MultipleRequirementTask(DummyTask):

            def requires(self):
                if False:
                    while True:
                        i = 10
                return [a, b]
        c = MultipleRequirementTask('C')
        self.assertTrue(self.worker.add(c))
        self.scheduler.get_work = Mock(side_effect=[self.gw_res(3, a.task_id), self.gw_res(2, b.task_id), self.gw_res(1, c.task_id), self.gw_res(0, None), self.gw_res(0, None)])
        self.assertTrue(self.worker.run())
        self.assertTrue(c.has_run)

    def test_path_with_task_failures(self):
        if False:
            i = 10
            return i + 15

        class FailingTask(DummyTask):

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise Exception('I am failing')
        a = FailingTask('a')
        b = FailingTask('b')

        class MultipleRequirementTask(DummyTask):

            def requires(self):
                if False:
                    return 10
                return [a, b]
        c = MultipleRequirementTask('C')
        self.assertTrue(self.worker.add(c))
        self.scheduler.get_work = Mock(side_effect=[self.gw_res(3, a.task_id), self.gw_res(2, b.task_id), self.gw_res(1, c.task_id), self.gw_res(0, None), self.gw_res(0, None)])
        self.assertFalse(self.worker.run())

class SingleWorkerMultiprocessTest(unittest.TestCase):

    def test_default_multiprocessing_behavior(self):
        if False:
            while True:
                i = 10
        with Worker(worker_processes=1) as worker:
            task = DummyTask('a')
            task_process = worker._create_task_process(task)
            self.assertFalse(task_process.use_multiprocessing)

    def test_force_multiprocessing(self):
        if False:
            return 10
        with Worker(worker_processes=1, force_multiprocessing=True) as worker:
            task = DummyTask('a')
            task_process = worker._create_task_process(task)
            self.assertTrue(task_process.use_multiprocessing)