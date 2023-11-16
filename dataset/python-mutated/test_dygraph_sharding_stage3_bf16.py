import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphShardingStage3(TestMultipleGpus):

    def test_dygraph_sharding_stage3_bf16(self):
        if False:
            return 10
        self.run_mnist_2gpu('dygraph_group_sharded_stage3_bf16.py')
if __name__ == '__main__':
    unittest.main()