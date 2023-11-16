import unittest
from test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestAutoParallelAutoConvert(TestMultipleGpus):

    def test_auto_parallel_autoconvert(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('auto_parallel_autoconvert.py')
if __name__ == '__main__':
    unittest.main()