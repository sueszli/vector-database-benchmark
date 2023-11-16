import unittest
from test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestPipelineParallel(TestMultipleGpus):

    def test_pipeline_parallel(self):
        if False:
            return 10
        self.run_mnist_2gpu('hybrid_parallel_pp_alexnet.py')

class TestModelParallelWithRecompute(TestMultipleGpus):

    def test_model_parallel_with_recompute(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('dygraph_recompute_hybrid.py')
if __name__ == '__main__':
    unittest.main()