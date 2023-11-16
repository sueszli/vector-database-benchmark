import unittest
from hybrid_parallel_mp_model import TestDistMPTraining
import paddle

class TestMPClipGrad(TestDistMPTraining):

    def build_optimizer(self, model):
        if False:
            while True:
                i = 10
        grad_clip = paddle.nn.ClipGradByGlobalNorm(2.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001, gamma=0.999, verbose=True)
        optimizer = paddle.optimizer.SGD(scheduler, grad_clip=grad_clip, parameters=model.parameters())
        return optimizer
if __name__ == '__main__':
    unittest.main()