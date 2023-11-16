import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestCollectiveBatchIsendIrecv(TestMultipleGpus):

    def test_collective_batch_isend_irecv(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('collective_batch_isend_irecv.py')
if __name__ == '__main__':
    unittest.main()