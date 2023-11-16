import unittest
import numpy as np
import paddle
from paddle.distributed import fleet

class TestDistTraining(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {'dp_degree': 1, 'mp_degree': self.model_parallel_size, 'pp_degree': 1}
        fleet.init(is_collective=True, strategy=strategy)

    def test_cuda_rng_tracker(self):
        if False:
            while True:
                i = 10
        seed_1 = 2021
        seed_2 = 1024
        size = [20, 15]
        paddle.seed(seed_1)
        target_11 = paddle.randn(size, 'float32')
        target_12 = paddle.randn(size, 'float32')
        paddle.seed(seed_2)
        target_21 = paddle.randn(size, 'float32')
        target_22 = paddle.randn(size, 'float32')
        paddle.seed(seed_1)
        fleet.meta_parallel.get_rng_state_tracker().add('test', seed_2)
        result_11 = paddle.randn(size, 'float32')
        with fleet.meta_parallel.get_rng_state_tracker().rng_state('test'):
            result_21 = paddle.randn(size, 'float32')
        result_12 = paddle.randn(size, 'float32')
        with fleet.meta_parallel.get_rng_state_tracker().rng_state('test'):
            result_22 = paddle.randn(size, 'float32')
        np.testing.assert_allclose(result_11.numpy(), target_11.numpy())
        np.testing.assert_allclose(result_12.numpy(), target_12.numpy())
        np.testing.assert_allclose(result_21.numpy(), target_21.numpy())
        np.testing.assert_allclose(result_22.numpy(), target_22.numpy())
if __name__ == '__main__':
    unittest.main()