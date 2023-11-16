import os
from unittest.mock import patch
from apps.dummy.benchmark.benchmark import DummyTaskBenchmark
from apps.dummy.task.dummytaskstate import DummyTaskDefinition, DummyTaskOptions
from golem.testutils import TempDirFixture

class TestDummyBenchmark(TempDirFixture):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.db = DummyTaskBenchmark()

    def test_is_instance(self):
        if False:
            return 10
        self.assertIsInstance(self.db, DummyTaskBenchmark)
        self.assertIsInstance(self.db.task_definition, DummyTaskDefinition)
        self.assertIsInstance(self.db.task_definition.options, DummyTaskOptions)

    def test_task_settings(self):
        if False:
            while True:
                i = 10
        self.assertTrue(os.path.isdir(self.db.dummy_task_path))
        self.assertEquals(self.db.task_definition.out_file_basename, 'out')
        self.assertTrue(all((os.path.isfile(x) for x in self.db.task_definition.shared_data_files)))
        self.assertEquals(self.db.task_definition.options.difficulty, 4294901760)
        self.assertEquals(self.db.task_definition.result_size, 256)
        self.assertEquals(self.db.task_definition.options.subtask_data_size, 128)

    def test_verify_result(self):
        if False:
            return 10
        files = [self.new_path / 'benchmark.result', self.new_path / 'benchmark.log']
        for f in files:
            f.touch()
        with patch('apps.dummy.task.verifier.DummyTaskVerifier._verify_result', returns=True):
            ret = self.db.verify_result([str(f) for f in files])
        assert ret