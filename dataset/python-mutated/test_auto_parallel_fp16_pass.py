import random
import unittest
import numpy as np
from auto_parallel_pass_test_base import AutoPallelPassTestBase
import paddle
from paddle.distributed import fleet

class TestPF16Pass(AutoPallelPassTestBase):

    def init(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-05
        self.atol = 1e-08
        paddle.seed(2021)
        random.seed(2021)
        np.random.seed(2021)

    def apply_passes(self):
        if False:
            i = 10
            return i + 15
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.amp = True
        dist_strategy.amp_configs = {'custom_white_list': ['softmax', 'layer_norm', 'gelu'], 'custom_black_list': ['c_softmax_with_cross_entropy', 'elementwise_div', 'reduce_sum'], 'init_loss_scaling': 32768, 'use_dynamic_loss_scaling': True, 'use_pure_fp16': True, 'use_fp16_guard': False}
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_bs_8(self):
        if False:
            print('Hello World!')
        self.check_main(gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000)

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        if False:
            print('Hello World!')
        return self.get_gpt_model('mp', place, batch_size, sequence_len, vocab_size)
if __name__ == '__main__':
    unittest.main()