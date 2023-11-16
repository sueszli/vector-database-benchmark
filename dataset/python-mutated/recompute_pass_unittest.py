import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed.fleet import auto

def apply_pass(use_recompute=False, no_recompute_segments=[]):
    if False:
        print('Hello World!')
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_recompute:
        recompute = strategy.recompute
        recompute.enable = True
        recompute.no_recompute_segments = no_recompute_segments
    return strategy

def reset_prog():
    if False:
        i = 10
        return i + 15
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestRecomputePass(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.rtol = 1e-06
        self.atol = 1e-08
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            print('Hello World!')
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_recompute=False, no_recompute_segments=[]):
        if False:
            while True:
                i = 10
        reset_prog()
        strategy = apply_pass(use_recompute, no_recompute_segments)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('mp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        if False:
            print('Hello World!')
        np.testing.assert_allclose(ref_losses, check_losses, rtol=self.rtol, atol=self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_recompute_pass(self):
        if False:
            print('Hello World!')
        mp_engine = self.get_engine()
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        mp_losses = np.array(history.history['loss'])
        rc_engine = self.get_engine(True)
        history = rc_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc_losses = np.array(history.history['loss'])
        self.check_results(mp_losses, rc_losses)
        rc1_engine = self.get_engine(True, [0])
        history = rc1_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc1_losses = np.array(history.history['loss'])
        self.check_results(mp_losses, rc1_losses)

    def test_recompute_pass_error(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError):
            rc_engine = self.get_engine(True, [2])
            history = rc_engine.fit(self.dataset, 3, batch_size=self.batch_size)
if __name__ == '__main__':
    unittest.main()