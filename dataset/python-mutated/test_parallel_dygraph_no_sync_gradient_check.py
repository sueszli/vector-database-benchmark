import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDataParallelLayer(TestMultipleGpus):

    def test_parallel_dygraph_dataparallel_no_sync(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('parallel_dygraph_no_sync_gradient_check.py')
if __name__ == '__main__':
    unittest.main()