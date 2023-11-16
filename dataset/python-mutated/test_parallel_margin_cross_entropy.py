import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestParallelMarginSoftmaxWithCrossEntropy(TestMultipleGpus):

    def test_parallel_margin_cross_entropy(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('parallel_margin_cross_entropy.py')
if __name__ == '__main__':
    unittest.main()