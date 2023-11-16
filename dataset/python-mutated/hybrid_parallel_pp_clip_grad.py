import sys
import unittest
sys.path.append('../../legacy_test')
from hybrid_parallel_pp_alexnet import TestDistPPTraining
import paddle

class TestPPClipGrad(TestDistPPTraining):

    def build_optimizer(self, model):
        if False:
            print('Hello World!')
        grad_clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2], values=[0.001, 0.002], verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler, grad_clip=grad_clip, parameters=model.parameters())
        return (scheduler, optimizer)

class TestPPClipGradParamGroup(TestDistPPTraining):

    def build_optimizer(self, model):
        if False:
            return 10
        grad_clip = paddle.nn.ClipGradByGlobalNorm(0.5)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2], values=[0.001, 0.002], verbose=True)
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, grad_clip=grad_clip, parameters=[{'params': model.parameters()}])
        return (scheduler, optimizer)
if __name__ == '__main__':
    unittest.main()