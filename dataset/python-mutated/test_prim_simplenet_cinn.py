import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.nn import BatchNorm
np.random.seed(2023)

def apply_to_static(net, use_cinn):
    if False:
        return 10
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

class PrimeNet(paddle.nn.Layer):

    def __init__(self, shape):
        if False:
            print('Hello World!')
        super().__init__()
        self.bn = BatchNorm(shape[-1], data_layout='NHWC', act='relu')

    def forward(self, data, dout):
        if False:
            print('Hello World!')
        y = self.bn(data) * dout
        return y

class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            return 10
        self.data = None
        self.dout = None
        self.shape = None

    def train(self, use_prim):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        net = PrimeNet(self.shape)
        sgd = paddle.optimizer.SGD(learning_rate=1.0, parameters=net.parameters())
        core._set_prim_all_enabled(use_prim)
        net = paddle.amp.decorate(models=net, level='O2')
        if use_prim:
            net = apply_to_static(net, use_prim)
        res = []
        with paddle.amp.auto_cast(level='O2'):
            for _ in range(10):
                out = net(self.data, self.dout)
                loss = paddle.mean(out)
                loss.backward()
                sgd.step()
                sgd.clear_grad()
                res.append(loss.numpy())
            self.check_prim(net, use_prim)
        return res

    def check_prim(self, net, use_prim):
        if False:
            for i in range(10):
                print('nop')
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.get_concrete_program(self.data, self.dout)[1].train_program.block(0).ops]
        self.assertTrue('batch_norm' not in fwd_ops)

    def test_cinn_prim(self):
        if False:
            for i in range(10):
                print('nop')
        if paddle.device.get_device() == 'cpu':
            return
        self.shape = (16, 112, 112, 64)
        self.data = paddle.to_tensor(np.random.random(self.shape).astype('float16'))
        self.data.stop_gradient = False
        self.dout = paddle.to_tensor(np.random.random(self.shape).astype('float16'))
        dy2st_res = self.train(use_prim=False)
        prim_res = self.train(use_prim=True)
        for i in range(len(dy2st_res)):
            np.testing.assert_allclose(prim_res[i], dy2st_res[i], rtol=0.001, atol=0.001)
if __name__ == '__main__':
    unittest.main()