import logging
import time
from helpers import unittest
import luigi
import luigi.contrib.hadoop
import luigi.rpc
import luigi.scheduler
import luigi.worker

class DummyTask(luigi.Task):
    task_namespace = 'customized_run'
    n = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(DummyTask, self).__init__(*args, **kwargs)
        self.has_run = False

    def complete(self):
        if False:
            i = 10
            return i + 15
        return self.has_run

    def run(self):
        if False:
            while True:
                i = 10
        logging.debug('%s - setting has_run', self)
        self.has_run = True

class CustomizedLocalScheduler(luigi.scheduler.Scheduler):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(CustomizedLocalScheduler, self).__init__(*args, **kwargs)
        self.has_run = False

    def get_work(self, worker, host=None, **kwargs):
        if False:
            return 10
        r = super(CustomizedLocalScheduler, self).get_work(worker=worker, host=host)
        self.has_run = True
        return r

    def complete(self):
        if False:
            i = 10
            return i + 15
        return self.has_run

class CustomizedRemoteScheduler(luigi.rpc.RemoteScheduler):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(CustomizedRemoteScheduler, self).__init__(*args, **kwargs)
        self.has_run = False

    def get_work(self, worker, host=None):
        if False:
            i = 10
            return i + 15
        r = super(CustomizedRemoteScheduler, self).get_work(worker=worker, host=host)
        self.has_run = True
        return r

    def complete(self):
        if False:
            print('Hello World!')
        return self.has_run

class CustomizedWorker(luigi.worker.Worker):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(CustomizedWorker, self).__init__(*args, **kwargs)
        self.has_run = False

    def _run_task(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        super(CustomizedWorker, self)._run_task(task_id)
        self.has_run = True

    def complete(self):
        if False:
            i = 10
            return i + 15
        return self.has_run

class CustomizedWorkerSchedulerFactory:

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.scheduler = CustomizedLocalScheduler()
        self.worker = CustomizedWorker(self.scheduler)

    def create_local_scheduler(self):
        if False:
            print('Hello World!')
        return self.scheduler

    def create_remote_scheduler(self, url):
        if False:
            while True:
                i = 10
        return CustomizedRemoteScheduler(url)

    def create_worker(self, scheduler, worker_processes=None, assistant=False):
        if False:
            print('Hello World!')
        return self.worker

class CustomizedWorkerTest(unittest.TestCase):
    """ Test that luigi's build method (and ultimately the run method) can accept a customized worker and scheduler """

    def setUp(self):
        if False:
            return 10
        self.worker_scheduler_factory = CustomizedWorkerSchedulerFactory()
        self.time = time.time

    def tearDown(self):
        if False:
            return 10
        if time.time != self.time:
            time.time = self.time

    def setTime(self, t):
        if False:
            while True:
                i = 10
        time.time = lambda : t

    def test_customized_worker(self):
        if False:
            for i in range(10):
                print('nop')
        a = DummyTask(3)
        self.assertFalse(a.complete())
        self.assertFalse(self.worker_scheduler_factory.worker.complete())
        luigi.build([a], worker_scheduler_factory=self.worker_scheduler_factory)
        self.assertTrue(a.complete())
        self.assertTrue(self.worker_scheduler_factory.worker.complete())

    def test_cmdline_custom_worker(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.worker_scheduler_factory.worker.complete())
        luigi.run(['customized_run.DummyTask', '--n', '4'], worker_scheduler_factory=self.worker_scheduler_factory)
        self.assertTrue(self.worker_scheduler_factory.worker.complete())