import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphGroupSharded(TestMultipleGpus):

    def test_dygraph_group_sharded(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('dygraph_group_sharded_api_eager.py')

    def test_dygraph_group_sharded_stage3(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('dygraph_group_sharded_stage3_eager.py')
if __name__ == '__main__':
    unittest.main()