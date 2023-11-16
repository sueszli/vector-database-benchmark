import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphShardingStage2(TestMultipleGpus):

    def test_dygraph_sharding_stage2(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('dygraph_group_sharded_stage2.py')

    def test_dygraph_sharding_stage2_offload(self):
        if False:
            return 10
        self.run_mnist_2gpu('dygraph_group_sharded_stage2_offload.py')

    def test_dygraph_sharding_stage2_with_comm_overlap(self):
        if False:
            return 10
        self.run_mnist_2gpu('dygraph_group_sharded_stage2_comm_overlap.py')
if __name__ == '__main__':
    unittest.main()