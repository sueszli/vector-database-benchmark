import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestHybridParallelInferenceHelper(TestMultipleGpus):

    def test_hybrid_parallel_inference_helper(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('hybrid_parallel_inference_helper.py')
if __name__ == '__main__':
    unittest.main()