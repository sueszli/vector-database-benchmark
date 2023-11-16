import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphShardingStage1(TestMultipleGpus):

    def test_dygraph_sharding_stage1_fp16(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('dygraph_group_sharded_stage1_fp16.py')
if __name__ == '__main__':
    unittest.main()