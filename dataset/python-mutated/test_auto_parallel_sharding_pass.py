import random
import unittest
import numpy as np
from auto_parallel_pass_test_base import AutoPallelPassTestBase
import paddle
from paddle.distributed import fleet

class TestShardingPass(AutoPallelPassTestBase):

    def init(self):
        if False:
            print('Hello World!')
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-05
        self.atol = 1e-08
        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def apply_passes(self):
        if False:
            i = 10
            return i + 15
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {'sharding_degree': 2, 'stage': 2}
        fleet.init(is_collective=True, strategy=dist_strategy)

    def apply_no_passes(self):
        if False:
            i = 10
            return i + 15
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.pipeline = False
        dist_strategy.recompute = False
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_bs_8(self):
        if False:
            print('Hello World!')
        self.check_main(gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000)

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        if False:
            return 10
        return self.get_gpt_model('dp', place, batch_size, sequence_len, vocab_size)
if __name__ == '__main__':
    unittest.main()