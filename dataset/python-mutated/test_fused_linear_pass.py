import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
from test_sparse_addmm_op import get_cuda_version
import paddle
from paddle.distributed.fleet import auto

def apply_pass(use_fused_passes=False, fused_passes_list=[]):
    if False:
        for i in range(10):
            print('nop')
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    fused_passes = strategy.fused_passes
    fused_passes.enable = use_fused_passes
    fused_passes.fused_passes_list = fused_passes_list
    return strategy

def reset_prog():
    if False:
        return 10
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestFusedLinearPass(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.rtol = 1e-05
        self.atol = 1e-08
        self.batch_size = 1
        self.batch_num = 1
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_fused_passes=False, fused_passes_list=[]):
        if False:
            print('Hello World!')
        reset_prog()
        strategy = apply_pass(use_fused_passes, fused_passes_list)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('serial')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        if False:
            return 10
        np.testing.assert_allclose(ref_losses, check_losses, rtol=rtol or self.rtol, atol=atol or self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_passes(self):
        if False:
            while True:
                i = 10
        losses = []
        if get_cuda_version() >= 11060:
            for use_fused_passes in [True, False]:
                engine = self.get_engine(use_fused_passes, ['fuse_gemm_epilogue'])
                history = engine.fit(self.dataset, 3, batch_size=self.batch_size)
                losses.append(np.array(history.history['loss']))
            self.check_results(losses[0], losses[1])
if __name__ == '__main__':
    unittest.main()