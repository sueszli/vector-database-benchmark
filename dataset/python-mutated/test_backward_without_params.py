import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
import paddle

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        out = x + 1
        return out

class TestBackwardWithoutParams(Dy2StTestBase):

    @test_legacy_and_pir
    def test_run(self):
        if False:
            print('Hello World!')
        net = paddle.jit.to_static(Net())
        x = paddle.ones([2, 2])
        x.stop_gradient = False
        out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        np.testing.assert_equal(x.grad.numpy(), np.full(x.shape, 0.25))

class ZeroSizeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        y = paddle.randn((0,))
        out = paddle.nn.functional.relu(x)
        y.stop_gradient = True
        return (y, out)

class TestZeroSizeNet(Dy2StTestBase):

    @test_legacy_and_pir
    def test_run(self):
        if False:
            print('Hello World!')
        net = paddle.jit.to_static(ZeroSizeNet())
        x = paddle.ones([2, 2])
        x.stop_gradient = False
        (_, out) = net(x)
        loss = paddle.mean(out)
        loss.backward()
        np.testing.assert_equal(x.grad.numpy(), np.full(x.shape, 0.25))
if __name__ == '__main__':
    unittest.main()