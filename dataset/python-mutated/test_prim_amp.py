import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.base import core, framework
from paddle.nn import BatchNorm
np.random.seed(2023)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv = nn.Conv2D(2, 4, (3, 3), bias_attr=False)
        self.bn = BatchNorm(4, act='relu')

    def forward(self, x):
        if False:
            return 10
        y = self.conv(x)
        out = self.bn(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res

class TestPrimAMPO1(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim v.s Dygraph in AMPO1.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        self.x = paddle.randn([4, 2, 6, 6], dtype='float32')
        self.x.stop_gradient = False

    def train(self, use_prim):
        if False:
            print('Hello World!')
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        if use_prim:
            net = paddle.jit.to_static(net, build_strategy=False)
        with paddle.amp.auto_cast(level='O1'):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            return loss

    def test_amp_01(self):
        if False:
            return 10
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(False)
            actual = self.train(True)
            np.testing.assert_allclose(expected, actual, rtol=0.001, atol=0.001)

    def test_amp_O1_infer(self):
        if False:
            return 10
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            net = PrimeNet()
            core._set_prim_all_enabled(False)
            net.eval()
            static_net = paddle.jit.to_static(net, build_strategy=False)
            res = static_net(self.x)
            core._set_prim_all_enabled(True)
            net.eval()
            static_net = paddle.jit.to_static(net, build_strategy=False)
            with paddle.amp.auto_cast(level='O1'):
                res_amp = static_net(self.x)
            np.testing.assert_allclose(res, res_amp, rtol=0.001, atol=0.001)
if __name__ == '__main__':
    unittest.main()