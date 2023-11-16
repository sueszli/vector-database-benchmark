import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestParallelizer(TestMultipleGpus):

    def test_parallelizer_logic(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('auto_parallel_parallelizer.py')
if __name__ == '__main__':
    unittest.main()