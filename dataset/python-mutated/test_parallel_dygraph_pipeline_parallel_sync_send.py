import os
import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestHybridPipeParallel(TestMultipleGpus):

    def test_hybrid_parallel_pp_layer(self):
        if False:
            return 10
        self.run_mnist_2gpu(os.path.abspath('../../legacy_test/hybrid_parallel_pp_layer.py'), need_envs={'PADDLE_P2P_SYNC_SEND': '1'})

    def test_hybrid_parallel_pp_tuple_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('hybrid_parallel_pp_embedding.py', need_envs={'PADDLE_P2P_SYNC_SEND': '1'})

    def test_hybrid_parallel_shared_weight(self):
        if False:
            while True:
                i = 10
        self.run_mnist_2gpu('hybrid_parallel_shared_weight.py', need_envs={'PADDLE_P2P_SYNC_SEND': '1'})

    def test_pipeline_parallel_amp(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_mnist_2gpu('hybrid_parallel_pp_amp.py', need_envs={'PADDLE_P2P_SYNC_SEND': '1'})
if __name__ == '__main__':
    unittest.main()