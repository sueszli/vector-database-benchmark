import os
import tempfile
import luigi.server
import server_test
tempdir = tempfile.mkdtemp()

class DummyTask(luigi.Task):
    id = luigi.IntParameter()

    def run(self):
        if False:
            i = 10
            return i + 15
        f = self.output().open('w')
        f.close()

    def output(self):
        if False:
            i = 10
            return i + 15
        return luigi.LocalTarget(os.path.join(tempdir, str(self.id)))

class RemoteSchedulerTest(server_test.ServerTestBase):

    def _test_run(self, workers):
        if False:
            i = 10
            return i + 15
        tasks = [DummyTask(id) for id in range(20)]
        luigi.build(tasks, workers=workers, scheduler_port=self.get_http_port())
        for t in tasks:
            self.assertEqual(t.complete(), True)
            self.assertTrue(os.path.exists(t.output().path))

    def test_single_worker(self):
        if False:
            print('Hello World!')
        self._test_run(workers=1)

    def test_multiple_workers(self):
        if False:
            print('Hello World!')
        self._test_run(workers=10)