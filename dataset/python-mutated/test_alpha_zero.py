import unittest
import rllib_alpha_zero.alpha_zero as az
from rllib_alpha_zero.alpha_zero.custom_torch_models import DenseModel
import ray
from ray.rllib.examples.env.cartpole_sparse_rewards import CartPoleSparseRewards
from ray.rllib.utils.test_utils import check_train_results, framework_iterator

class TestAlphaZero(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        ray.shutdown()

    def test_alpha_zero_compilation(self):
        if False:
            i = 10
            return i + 15
        'Test whether AlphaZero can be built with all frameworks.'
        config = az.AlphaZeroConfig().environment(env=CartPoleSparseRewards).training(model={'custom_model': DenseModel})
        num_iterations = 1
        for _ in framework_iterator(config, frameworks='torch'):
            algo = config.build()
            for i in range(num_iterations):
                results = algo.train()
                check_train_results(results)
                print(results)
            algo.stop()
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', __file__]))