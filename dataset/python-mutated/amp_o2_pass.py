import os
import random
import re
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed.fleet import auto
from paddle.framework import core
paddle.enable_static()

def get_cuda_version():
    if False:
        for i in range(10):
            print('nop')
    result = os.popen('nvcc --version').read()
    regex = 'release (\\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        (integer, decimal) = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1

def apply_pass(use_amp=False, amp_dtype='bfloat16'):
    if False:
        return 10
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_amp:
        amp = strategy.amp
        amp.enable = True
        amp.dtype = amp_dtype
        amp.level = 'o2'
        amp.custom_black_list = ['c_softmax_with_cross_entropy', 'elementwise_div', 'reduce_sum']
    return strategy

def reset_prog():
    if False:
        for i in range(10):
            print('nop')
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestShardingStage2WithNewEXE(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.batch_size = 2
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_amp=False, amp_dtype='bfloat16'):
        if False:
            print('Hello World!')
        reset_prog()
        strategy = apply_pass(use_amp, amp_dtype)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('mp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_bf16(self, program):
        if False:
            print('Hello World!')
        num_bf16 = 0
        num_fp16 = 0
        num_fp32 = 0
        for p in program.all_parameters():
            if p.dtype == core.VarDesc.VarType.FP32:
                num_fp32 += 1
            if p.dtype == core.VarDesc.VarType.FP16:
                num_fp16 += 1
            if p.dtype == core.VarDesc.VarType.BF16:
                num_bf16 += 1
        self.assertEqual(num_bf16, 26)
        self.assertEqual(num_fp16, 0)
        self.assertEqual(num_fp32, 10)

    def test_param_grad_fuse_overlap(self):
        if False:
            for i in range(10):
                print('nop')
        mp_engine = self.get_engine(use_amp=False)
        mp_history = mp_engine.fit(self.dataset, 3, epochs=1, steps_per_epoch=self.batch_num, log_freq=1, batch_size=self.batch_size)
        loss0 = mp_history.history['loss'][0]
        mp_bf16_engine = self.get_engine(use_amp=True)
        if not (paddle.amp.is_bfloat16_supported() and paddle.device.cuda.get_device_capability()[0] >= 8):
            return
        mp_bf16_history = mp_bf16_engine.fit(self.dataset, 3, epochs=1, steps_per_epoch=self.batch_num, log_freq=1, batch_size=self.batch_size)
        loss1 = mp_bf16_history.history['loss'][0]
        np.testing.assert_allclose(loss0, loss1, atol=0.001, rtol=0.01)
        self.check_bf16(mp_bf16_engine.main_program)
if __name__ == '__main__':
    unittest.main()