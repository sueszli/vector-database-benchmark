import os
import unittest
from legacy_test.test_parallel_dygraph_dataparallel import TestMultipleGpus

class TestHybridParallel(TestMultipleGpus):

    def test_hybrid_parallel_sharding_logic(self):
        if False:
            return 10
        os.environ['FLAGS_shard_use_reduce'] = '1'
        os.environ['FLAGS_shard_norm_align_dp'] = '0'
        os.environ['FLAGS_shard_split_param'] = '1'
        self.run_mnist_2gpu('hybrid_parallel_sharding_model.py')
        os.environ['FLAGS_shard_use_reduce'] = '1'
        os.environ['FLAGS_shard_norm_align_dp'] = '0'
        os.environ['FLAGS_shard_split_param'] = '0'
        self.run_mnist_2gpu('hybrid_parallel_sharding_model.py')
        os.environ['FLAGS_shard_use_reduce'] = '0'
        os.environ['FLAGS_shard_norm_align_dp'] = '1'
        os.environ['FLAGS_shard_split_param'] = '0'
        self.run_mnist_2gpu('hybrid_parallel_sharding_model.py')

    def test_hybrid_parallel_sharding_tensor_fusion(self):
        if False:
            i = 10
            return i + 15
        os.environ['FLAGS_shard_split_param'] = '0'
        self.run_mnist_2gpu('hybrid_parallel_sharding_model_with_fusion.py')

    def test_hybrid_parallel_sharding_tensor_fusion_amp(self):
        if False:
            return 10
        os.environ['FLAGS_shard_split_param'] = '0'
        self.run_mnist_2gpu('hybrid_parallel_sharding_model_with_fusion_amp.py')

    def test_hybrid_parallel_sharding_state_dict(self):
        if False:
            return 10
        os.environ['FLAGS_shard_split_param'] = '0'
        self.run_mnist_2gpu('hybrid_parallel_sharding_state_dict.py')

    def test_group_param_tensor_fusion(self):
        if False:
            return 10
        self.run_mnist_2gpu('hybrid_parallel_tensor_fusion_with_group.py')
if __name__ == '__main__':
    unittest.main()