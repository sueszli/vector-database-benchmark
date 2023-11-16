import os
import unittest
from test_parallel_dygraph_dataparallel import TestMultipleXpus

class TestProcessGroup(TestMultipleXpus):

    def test_process_group_bkcl(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2xpu('process_group_bkcl.py')
if __name__ == '__main__':
    os.environ['BKCL_PCIE_RING'] = '1'
    os.environ['BKCL_CCIX_RING'] = '0'
    unittest.main()