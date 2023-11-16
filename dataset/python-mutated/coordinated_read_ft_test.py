"""Fault tolerance tests for tf.data service coordinated reads."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

class CoordinatedReadFTTest(data_service_test_base.TestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(workers_to_add=[1, 3, 10])))
    def testAddWorkers(self, workers_to_add):
        if False:
            i = 10
            return i + 15
        starting_workers = 3
        cluster = data_service_test_base.TestCluster(num_workers=starting_workers)
        num_consumers = 7
        ds = self.make_coordinated_read_dataset(cluster, num_consumers)
        get_next = self.getNext(ds, requires_initialization=True)
        results = []
        zeros_seen = 0
        for _ in range(25):
            results.append(self.evaluate(get_next()))
            if results[-1] == 0:
                zeros_seen += 1
        for _ in range(workers_to_add):
            cluster.add_worker()
        while zeros_seen < starting_workers + workers_to_add:
            results.append(self.evaluate(get_next()))
            if results[-1] == 0:
                zeros_seen += 1
        for _ in range(25):
            results.append(self.evaluate(get_next()))
        self.checkCoordinatedReadGroups(results, num_consumers)
        cluster.stop_workers()

    @combinations.generate(test_base.eager_only_combinations())
    def testRestartWorker(self):
        if False:
            i = 10
            return i + 15
        num_workers = 3
        cluster = data_service_test_base.TestCluster(num_workers, worker_shutdown_quiet_period_ms=2000)
        num_consumers = 5
        ds = self.make_coordinated_read_dataset(cluster, num_consumers)
        get_next = self.getNext(ds, requires_initialization=True)
        results = []
        self.read(get_next, results, 20)
        cluster.workers[1].stop()
        self.read(get_next, results, 20)
        cluster.workers[1].restart()
        while results[-1] != 0:
            results.append(self.evaluate(get_next()))
        self.read(get_next, results, 20)
        self.checkCoordinatedReadGroups(results, num_consumers)
        cluster.stop_workers()

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(sharding_policy=[data_service_ops.ShardingPolicy.OFF, data_service_ops.ShardingPolicy.DYNAMIC])))
    def testMultiStartStop(self, sharding_policy):
        if False:
            print('Hello World!')
        num_workers = 3
        cluster = data_service_test_base.TestCluster(num_workers, worker_shutdown_quiet_period_ms=2000)
        num_consumers = 5
        ds = self.make_coordinated_read_dataset(cluster, num_consumers, sharding_policy)
        get_next = self.getNext(ds, requires_initialization=True)
        results = []
        self.read(get_next, results, 20)
        for i in range(num_workers):
            cluster.workers[i].stop()
            self.read(get_next, results, 20)
            cluster.workers[i].restart()
            self.read(get_next, results, 20)
        cluster.add_worker()
        cluster.restart_dispatcher()
        for i in range(num_workers):
            cluster.workers[i].stop()
        self.read(get_next, results, 20)
        self.checkCoordinatedReadGroups(results, num_consumers)
        cluster.stop_workers()
if __name__ == '__main__':
    test.main()