import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestCollectiveAllToAllSingle(TestMultipleGpus):

    def test_collective_alltoall_single(self):
        if False:
            return 10
        self.run_mnist_2gpu('collective_alltoall_single.py')
if __name__ == '__main__':
    unittest.main()