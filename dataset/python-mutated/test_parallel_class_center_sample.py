import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestParallelClassCenterSample(TestMultipleGpus):

    def test_parallel_class_center_sample(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('parallel_class_center_sample.py')
if __name__ == '__main__':
    unittest.main()