import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto
paddle.enable_static()

def apply_pass(use_1f1b=False):
    if False:
        while True:
            i = 10
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_1f1b:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = '1F1B'
        pipeline.accumulate_steps = 2
    else:
        gradient_merge = strategy.gradient_merge
        gradient_merge.enable = True
        gradient_merge.k_steps = 2
        gradient_merge.avg = True
    return strategy

def reset_prog():
    if False:
        while True:
            i = 10
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class Test1F1BPass(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.rtol = 1e-05
        self.atol = 1e-08
        self.batch_size = 2
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
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.base.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_1f1b=False):
        if False:
            for i in range(10):
                print('nop')
        reset_prog()
        strategy = apply_pass(use_1f1b)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('pp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        if False:
            return 10
        np.testing.assert_allclose(ref_losses, check_losses, rtol=self.rtol, atol=self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_1f1b_pass(self):
        if False:
            for i in range(10):
                print('nop')
        engine_pp = self.get_engine()
        history_pp = engine_pp.fit(self.dataset, 3, batch_size=self.batch_size, log_freq=1)
        assert engine_pp._strategy.pipeline.enable is False
        engine_1f1b = self.get_engine(True)
        history_1f1b = engine_1f1b.fit(self.dataset, 3, batch_size=self.batch_size, log_freq=1)
        assert engine_1f1b._strategy.pipeline.enable is True
        if paddle.distributed.get_rank() == 1:
            losses_pp = np.array(history_pp.history['loss'])
            losses_1f1b = np.array(history_1f1b.history['loss'])
            self.check_results(losses_pp, losses_1f1b)
if __name__ == '__main__':
    unittest.main()