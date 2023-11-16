import unittest
from test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestAutoParallelDataUnshard(TestMultipleGpus):

    def test_auto_parallel_data_unshard(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('auto_parallel_data_unshard.py')
if __name__ == '__main__':
    unittest.main()