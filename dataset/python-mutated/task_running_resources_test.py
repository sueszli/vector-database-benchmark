import os
import time
import signal
import multiprocessing
from contextlib import contextmanager
from helpers import unittest, RunOnceTask, with_config, skipOnGithubActions
import luigi
import luigi.server

class ResourceTestTask(RunOnceTask):
    param = luigi.Parameter()
    reduce_foo = luigi.BoolParameter()

    def process_resources(self):
        if False:
            return 10
        return {'foo': 2}

    def run(self):
        if False:
            return 10
        if self.reduce_foo:
            self.decrease_running_resources({'foo': 1})
        time.sleep(2)
        super(ResourceTestTask, self).run()

class ResourceWrapperTask(RunOnceTask):
    reduce_foo = ResourceTestTask.reduce_foo

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        return [ResourceTestTask(param='a', reduce_foo=self.reduce_foo), ResourceTestTask(param='b')]

class LocalRunningResourcesTest(unittest.TestCase):

    def test_resource_reduction(self):
        if False:
            for i in range(10):
                print('nop')
        sch = luigi.scheduler.Scheduler(resources={'foo': 2})
        with luigi.worker.Worker(scheduler=sch) as w:
            task = ResourceTestTask(param='a', reduce_foo=True)
            w.add(task)
            w.run()
            self.assertEqual(sch.get_running_task_resources(task.task_id)['resources']['foo'], 1)

class ConcurrentRunningResourcesTest(unittest.TestCase):

    @with_config({'scheduler': {'stable_done_cooldown_secs': '0'}})
    def setUp(self):
        if False:
            print('Hello World!')
        super(ConcurrentRunningResourcesTest, self).setUp()
        self._process = multiprocessing.Process(target=luigi.server.run)
        self._process.start()
        time.sleep(0.5)
        self.sch = luigi.rpc.RemoteScheduler()
        self.sch.update_resource('foo', 3)

    def tearDown(self):
        if False:
            while True:
                i = 10
        super(ConcurrentRunningResourcesTest, self).tearDown()
        self._process.terminate()
        self._process.join(timeout=1)
        if self._process.is_alive():
            os.kill(self._process.pid, signal.SIGKILL)

    @contextmanager
    def worker(self, scheduler=None, processes=2):
        if False:
            print('Hello World!')
        with luigi.worker.Worker(scheduler=scheduler or self.sch, worker_processes=processes) as w:
            w._config.wait_interval = 0.2
            w._config.check_unfulfilled_deps = False
            yield w

    @contextmanager
    def assert_duration(self, min_duration=0, max_duration=-1):
        if False:
            for i in range(10):
                print('nop')
        t0 = time.time()
        try:
            yield
        finally:
            duration = time.time() - t0
            self.assertGreater(duration, min_duration)
            if max_duration > 0:
                self.assertLess(duration, max_duration)

    def test_tasks_serial(self):
        if False:
            print('Hello World!')
        with self.worker() as w:
            w.add(ResourceWrapperTask(reduce_foo=False))
            with self.assert_duration(min_duration=4):
                w.run()

    @skipOnGithubActions('Temporary skipping on GH actions')
    def test_tasks_parallel(self):
        if False:
            i = 10
            return i + 15
        with self.worker() as w:
            w.add(ResourceWrapperTask(reduce_foo=True))
            with self.assert_duration(max_duration=4):
                w.run()