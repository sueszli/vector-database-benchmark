import random
import sys
import unittest
import numpy as np
from get_gpt_model import FakeDataset
import paddle
from paddle.distributed.fleet import auto
sys.path.append('..')
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import GPTForPretraining, GPTModel, GPTPretrainingCriterion

def generate_model(use_new_recompute, recompute_granularity):
    if False:
        while True:
            i = 10
    modeling.init_global()
    modeling._global_parallel_strategy = 'serial'
    modeling._global_process_mesh = auto.ProcessMesh(mesh=[0], dim_names=['x'])
    gpt = GPTModel(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=8, intermediate_size=256, hidden_act='gelu', hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, max_position_embeddings=1024, type_vocab_size=1, initializer_range=0.02, pad_token_id=0, eos_token_id=7, bos_token_id=0, eol_token_id=3, use_new_recompute=use_new_recompute, recompute_granularity=recompute_granularity)
    model = GPTForPretraining(gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02)
    criterion = GPTPretrainingCriterion()
    return (model, criterion)

def apply_pass(use_recompute=False, no_recompute_segments=[]):
    if False:
        i = 10
        return i + 15
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
        return 10
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestRecomputePassWithRecomputeAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.rtol = 1e-06
        self.atol = 1e-08
        self.batch_size = 1
        self.batch_num = 2
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

    def get_engine(self, use_recompute=False, use_new_recompute=False, recompute_granularity='full', no_recompute_segments=[]):
        if False:
            print('Hello World!')
        reset_prog()
        strategy = apply_pass(use_recompute, no_recompute_segments)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model(use_new_recompute, recompute_granularity)
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        if False:
            for i in range(10):
                print('nop')
        np.testing.assert_allclose(ref_losses, check_losses, rtol=self.rtol, atol=self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def recompute_vars(self, program):
        if False:
            return 10
        return list(filter(lambda a: 'subprog' in a.name, program.list_vars()))

    def test_recompute_pass(self):
        if False:
            i = 10
            return i + 15
        mp_engine = self.get_engine()
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        mp_losses = np.array(history.history['loss'])
        rc4_engine = self.get_engine(True, False)
        history = rc4_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc4_losses = np.array(history.history['loss'])
        self.check_results(mp_losses, rc4_losses)
        rc1_engine = self.get_engine(True, True, 'core_attn', [0])
        history = rc1_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc1_losses = np.array(history.history['loss'])
        self.check_results(mp_losses, rc1_losses)
        rc2_engine = self.get_engine(True, True, 'full_attn')
        history = rc2_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc2_losses = np.array(history.history['loss'])
        self.check_results(mp_losses, rc2_losses)
        rc3_engine = self.get_engine(True, True, 'full')
        history = rc3_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc3_losses = np.array(history.history['loss'])
        self.check_results(mp_losses, rc3_losses)
        rc0_vars = self.recompute_vars(mp_engine.main_program)
        rc1_vars = self.recompute_vars(rc1_engine.main_program)
        rc2_vars = self.recompute_vars(rc2_engine.main_program)
        rc3_vars = self.recompute_vars(rc3_engine.main_program)
        assert rc0_vars == []
        assert len(rc1_vars) < len(rc2_vars) and len(rc2_vars) < len(rc3_vars)

    def test_recompute_pass_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError):
            rc_engine = self.get_engine(True, True, 'full', [2])
            history = rc_engine.fit(self.dataset, 3, batch_size=self.batch_size)
if __name__ == '__main__':
    unittest.main()