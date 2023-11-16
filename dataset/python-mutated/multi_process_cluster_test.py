"""Tests tf.data service cluster with local and remote workers."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations

class MultiProcessClusterTest(data_service_test_base.TestBase, parameterized.TestCase):
    """Verifies the local and remote workers are running and producing data."""

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(num_local_workers=[0, 1, 3], num_remote_workers=[0, 1, 3])))
    def testCluster(self, num_local_workers, num_remote_workers):
        if False:
            i = 10
            return i + 15
        cluster = multi_process_cluster.MultiProcessCluster(num_local_workers=num_local_workers, num_remote_workers=num_remote_workers)
        num_elements = 10
        num_workers = num_local_workers + num_remote_workers
        if num_workers == 0:
            return
        dataset = self.make_distributed_range_dataset(num_elements, cluster)
        self.assertDatasetProduces(dataset, num_workers * list(range(num_elements)), assert_items_equal=True)
if __name__ == '__main__':
    multi_process_cluster.test_main()