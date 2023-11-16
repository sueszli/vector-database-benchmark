import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed.fleet import auto
paddle.enable_static()

def apply_pass(use_gradient_merge=False):
    if False:
        return 10
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_gradient_merge:
        gradient_merge = strategy.gradient_merge
        gradient_merge.enable = True
        gradient_merge.k_steps = 4
        gradient_merge.avg = True
    return strategy

def reset_prog():
    if False:
        print('Hello World!')
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestGradientMergePass(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rtol = 1e-05
        self.atol = 1e-08
        self.batch_size = 8
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            while True:
                i = 10
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_gradient_merge=False):
        if False:
            print('Hello World!')
        reset_prog()
        strategy = apply_pass(use_gradient_merge)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('dp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        if False:
            print('Hello World!')
        np.testing.assert_allclose(ref_losses, check_losses, rtol=self.rtol, atol=self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_gradient_merge_pass(self):
        if False:
            i = 10
            return i + 15
        dp_engine = self.get_engine()
        history = dp_engine.fit(self.dataset, 3, batch_size=self.batch_size, log_freq=1)
        dp_losses = np.array(history.history['loss'])
        gm_engine = self.get_engine(True)
        history = gm_engine.fit(self.dataset, 3, batch_size=self.batch_size, log_freq=1)
        gm_losses = np.array(history.history['loss'])
        self.check_results(dp_losses, gm_losses)
if __name__ == '__main__':
    unittest.main()