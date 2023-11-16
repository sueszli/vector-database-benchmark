import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestCollectiveReduceScatter(TestMultipleGpus):

    def test_collective_reduce_scatter(self):
        if False:
            return 10
        self.run_mnist_2gpu('collective_reduce_scatter.py')
if __name__ == '__main__':
    unittest.main()