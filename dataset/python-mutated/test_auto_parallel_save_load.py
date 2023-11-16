import unittest
from test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestAutoParallelSaveLoad(TestMultipleGpus):

    def test_auto_parallel_save_load(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('auto_parallel_save_load.py')
if __name__ == '__main__':
    unittest.main()