import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestDygraphDataParallel(TestMultipleGpus):

    def test_dygraph_dataparallel_bf16(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('dygraph_dataparallel_bf16.py')
if __name__ == '__main__':
    unittest.main()