"""Fault tolerance tests for tf.data service cross-trainer cache."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

class CrossTrainerCacheFtTest(data_service_test_base.TestBase, parameterized.TestCase):
    """Fault tolerance tests for tf.data service cross-trainer cache."""

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testWorkerRestart(self):
        if False:
            for i in range(10):
                print('nop')
        cluster = self._create_cluster(num_workers=1)
        dataset = dataset_ops.Dataset.range(10000000).repeat()
        distributed_dataset = self.make_distributed_dataset(dataset, cluster, job_name='job', cross_trainer_cache=data_service_ops.CrossTrainerCache(trainer_id='Trainer 1'))
        get_next = self.getNext(distributed_dataset)
        elements = self._get_next(get_next, 100)
        self.assertEqual(elements, list(range(100)))
        cluster.workers[0].restart()
        while self.evaluate(get_next()) != 0:
            pass
        elements = self._get_next(get_next, 100)
        self.assertEqual(elements, list(range(1, 101)))

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testDispatcherRestart(self):
        if False:
            i = 10
            return i + 15
        cluster = self._create_cluster(num_workers=1)
        dataset = dataset_ops.Dataset.range(10000000).repeat()
        distributed_dataset = self.make_distributed_dataset(dataset, cluster, job_name='job', cross_trainer_cache=data_service_ops.CrossTrainerCache(trainer_id='Trainer 1'))
        get_next = self.getNext(distributed_dataset)
        elements = self._get_next(get_next, 100)
        self.assertEqual(elements, list(range(100)))
        cluster.restart_dispatcher()
        elements = self._get_next(get_next, 100)
        self.assertEqual(elements, list(range(100, 200)))

    def _get_next(self, get_next, num_elements):
        if False:
            while True:
                i = 10
        return [self.evaluate(get_next()) for _ in range(num_elements)]

    def _create_cluster(self, num_workers, cross_trainer_cache_size_bytes=10 * 2 ** 30):
        if False:
            return 10
        cluster = data_service_test_base.TestCluster(num_workers=0)
        for _ in range(num_workers):
            worker = data_service_test_base.TestWorker(dispatcher_address=cluster.dispatcher_address(), shutdown_quiet_period_ms=0, cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes)
            worker.start()
            cluster.workers.append(worker)
        return cluster
if __name__ == '__main__':
    test.main()