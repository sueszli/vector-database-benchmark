import luigi
from luigi.local_target import LocalTarget
from luigi.scheduler import Scheduler
import luigi.server
import luigi.worker
import luigi.task
from mock import patch
from helpers import with_config, unittest
import os
import tempfile
import shutil

class TestExternalFileTask(luigi.ExternalTask):
    """ Mocking tasks is a pain, so touch a file instead """
    path = luigi.Parameter()
    times_to_call = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(TestExternalFileTask, self).__init__(*args, **kwargs)
        self.times_called = 0

    def complete(self):
        if False:
            print('Hello World!')
        '\n        Create the file we need after a number of preconfigured attempts\n        '
        self.times_called += 1
        if self.times_called >= self.times_to_call:
            open(self.path, 'a').close()
        return os.path.exists(self.path)

    def output(self):
        if False:
            print('Hello World!')
        return LocalTarget(path=self.path)

class TestTask(luigi.Task):
    """
    Requires a single file dependency
    """
    tempdir = luigi.Parameter()
    complete_after = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(TestTask, self).__init__(*args, **kwargs)
        self.output_path = os.path.join(self.tempdir, 'test.output')
        self.dep_path = os.path.join(self.tempdir, 'test.dep')
        self.dependency = TestExternalFileTask(path=self.dep_path, times_to_call=self.complete_after)

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.dependency

    def output(self):
        if False:
            while True:
                i = 10
        return LocalTarget(path=self.output_path)

    def run(self):
        if False:
            return 10
        open(self.output_path, 'a').close()

class WorkerExternalTaskTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tempdir = tempfile.mkdtemp(prefix='luigi-test-')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.tempdir)

    def _assert_complete(self, tasks):
        if False:
            return 10
        for t in tasks:
            self.assert_(t.complete())

    def _build(self, tasks):
        if False:
            return 10
        with self._make_worker() as w:
            for t in tasks:
                w.add(t)
            w.run()

    def _make_worker(self):
        if False:
            for i in range(10):
                print('nop')
        self.scheduler = Scheduler(prune_on_get_work=True)
        return luigi.worker.Worker(scheduler=self.scheduler, worker_processes=1)

    def test_external_dependency_already_complete(self):
        if False:
            print('Hello World!')
        '\n        Test that the test task completes when its dependency exists at the\n        start of the execution.\n        '
        test_task = TestTask(tempdir=self.tempdir, complete_after=1)
        luigi.build([test_task], local_scheduler=True)
        assert os.path.exists(test_task.dep_path)
        assert os.path.exists(test_task.output_path)
        assert test_task.dependency.times_called == 2

    @with_config({'worker': {'retry_external_tasks': 'true'}, 'scheduler': {'retry_delay': '0.0'}})
    def test_external_dependency_gets_rechecked(self):
        if False:
            return 10
        '\n        Test that retry_external_tasks re-checks external tasks\n        '
        assert luigi.worker.worker().retry_external_tasks is True
        test_task = TestTask(tempdir=self.tempdir, complete_after=10)
        self._build([test_task])
        assert os.path.exists(test_task.dep_path)
        assert os.path.exists(test_task.output_path)
        self.assertGreaterEqual(test_task.dependency.times_called, 10)

    @with_config({'worker': {'retry_external_tasks': 'true', 'keep_alive': 'true', 'wait_interval': '0.00001'}, 'scheduler': {'retry_delay': '0.01'}})
    def test_external_dependency_worker_is_patient(self):
        if False:
            while True:
                i = 10
        '\n        Test that worker doesn\'t "give up" with keep_alive option\n\n        Instead, it should sleep for random.uniform() seconds, then ask\n        scheduler for work.\n        '
        assert luigi.worker.worker().retry_external_tasks is True
        with patch('random.uniform', return_value=0.001):
            test_task = TestTask(tempdir=self.tempdir, complete_after=5)
            self._build([test_task])
        assert os.path.exists(test_task.dep_path)
        assert os.path.exists(test_task.output_path)
        self.assertGreaterEqual(test_task.dependency.times_called, 5)

    def test_external_dependency_bare(self):
        if False:
            print('Hello World!')
        '\n        Test ExternalTask without altering global settings.\n        '
        assert luigi.worker.worker().retry_external_tasks is False
        test_task = TestTask(tempdir=self.tempdir, complete_after=5)
        scheduler = luigi.scheduler.Scheduler(retry_delay=0.01, prune_on_get_work=True)
        with luigi.worker.Worker(retry_external_tasks=True, scheduler=scheduler, keep_alive=True, wait_interval=1e-05, wait_jitter=0) as w:
            w.add(test_task)
            w.run()
        assert os.path.exists(test_task.dep_path)
        assert os.path.exists(test_task.output_path)
        self.assertGreaterEqual(test_task.dependency.times_called, 5)

    @with_config({'worker': {'retry_external_tasks': 'true'}, 'scheduler': {'retry_delay': '0.0'}})
    def test_external_task_complete_but_missing_dep_at_runtime(self):
        if False:
            i = 10
            return i + 15
        '\n        Test external task complete but has missing upstream dependency at\n        runtime.\n\n        Should not get "unfulfilled dependencies" error.\n        '
        test_task = TestTask(tempdir=self.tempdir, complete_after=3)
        test_task.run = NotImplemented
        assert len(test_task.deps()) > 0
        with self._make_worker() as w:
            w.add(test_task)
            open(test_task.output_path, 'a').close()
            success = w.run()
        self.assertTrue(success)
        self.assertFalse(os.path.exists(test_task.dep_path))