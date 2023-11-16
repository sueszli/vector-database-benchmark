import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestHybridParallel(TestMultipleGpus):

    def test_hybrid_parallel_hcg(self):
        if False:
            print('Hello World!')
        self.run_mnist_2gpu('hybrid_parallel_sep_model.py')
if __name__ == '__main__':
    unittest.main()