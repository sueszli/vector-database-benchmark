import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestModelParallelLayer(TestMultipleGpus):

    def test_hybrid_parallel_mp_layer(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('hybrid_parallel_mp_layers.py')
if __name__ == '__main__':
    unittest.main()