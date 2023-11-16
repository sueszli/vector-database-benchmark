import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed.fleet import auto

def apply_pass(use_amp=False, level=None):
    if False:
        for i in range(10):
            print('nop')
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_amp:
        amp = strategy.amp
        amp.enable = True
        amp.dtype = 'float16'
        amp.level = level
        amp.custom_white_list = ['softmax', 'layer_norm', 'gelu']
        amp.custom_black_list = ['c_softmax_with_cross_entropy', 'elementwise_div', 'reduce_sum']
        amp.init_loss_scaling = 32768
        amp.use_fp16_guard = False
        print('amp level: ', level)
    return strategy

def reset_prog():
    if False:
        i = 10
        return i + 15
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestAMPPass(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.rtol = 1e-05
        self.atol = 1e-08
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            return 10
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_amp=False, level=None):
        if False:
            while True:
                i = 10
        reset_prog()
        strategy = apply_pass(use_amp, level)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('mp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_allclose(ref_losses, check_losses, rtol=rtol or self.rtol, atol=atol or self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_amp_pass(self):
        if False:
            while True:
                i = 10
        mp_engine = self.get_engine()
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        mp_losses = np.array(history.history['loss'])
        amp_o1_engine = self.get_engine(True, 'o1')
        history = amp_o1_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        amp_o1_losses = np.array(history.history['loss'])
        amp_o1_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
        amp_o2_engine = self.get_engine(True, 'o2')
        history = amp_o2_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        amp_o2_losses = np.array(history.history['loss'])
        amp_o2_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
        amp_o3_engine = self.get_engine(True, 'o3')
        history = amp_o3_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        amp_o3_losses = np.array(history.history['loss'])
        amp_o3_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
if __name__ == '__main__':
    unittest.main()