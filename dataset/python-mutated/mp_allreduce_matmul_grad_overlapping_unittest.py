import random
import unittest
import numpy as np
from get_gpt_model import FakeDataset, generate_model
import paddle
from paddle.distributed.fleet import auto
paddle.enable_static()

def reset_prog():
    if False:
        print('Hello World!')
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class TestMPAllreduceMatmulGradOverlapping(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.rtol = 1e-05
        self.atol = 1e-08
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        if False:
            print('Hello World!')
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_mp_engine(self, allreduce_matmul_grad_overlapping):
        if False:
            for i in range(10):
                print('nop')
        reset_prog()
        strategy = auto.Strategy()
        strategy.auto_mode = 'semi'
        strategy.reinit = True
        strategy.mp_optimization.allreduce_matmul_grad_overlapping = allreduce_matmul_grad_overlapping
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=1e-05, grad_clip=clip)
        (model, loss) = generate_model('mp')
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def run_mp(self, allreduce_matmul_grad_overlapping):
        if False:
            for i in range(10):
                print('nop')
        mp_engine = self.get_mp_engine(allreduce_matmul_grad_overlapping)
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        return np.array(history.history['loss'])

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        if False:
            return 10
        np.testing.assert_allclose(ref_losses, check_losses, rtol=rtol or self.rtol, atol=atol or self.atol, err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(__class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_mp_allreduce_matmul_grad_overlapping(self):
        if False:
            while True:
                i = 10
        losses_with_allreduce_matmul_grad_overlapping = self.run_mp(True)
        losses_without_allreduce_matmul_grad_overlapping = self.run_mp(False)
        np.testing.assert_equal(losses_with_allreduce_matmul_grad_overlapping, losses_without_allreduce_matmul_grad_overlapping)
if __name__ == '__main__':
    unittest.main()