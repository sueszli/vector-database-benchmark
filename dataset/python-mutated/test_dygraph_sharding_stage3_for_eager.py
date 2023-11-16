import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphShardingStage3(TestMultipleGpus):

    def test_dygraph_sharding_stage3(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('dygraph_group_sharded_stage3.py')

    def test_dygraph_sharding_stage3_offload(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('dygraph_group_sharded_stage3_offload.py')
if __name__ == '__main__':
    unittest.main()