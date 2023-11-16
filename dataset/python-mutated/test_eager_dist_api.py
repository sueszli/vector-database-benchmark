import unittest
from test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestProcessGroup(TestMultipleGpus):

    def test_process_group_nccl(self):
        if False:
            i = 10
            return i + 15
        self.run_mnist_2gpu('process_group_nccl.py')

    def test_process_group_gloo(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('process_group_gloo.py')

    def test_init_process_group(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('init_process_group.py')
if __name__ == '__main__':
    unittest.main()