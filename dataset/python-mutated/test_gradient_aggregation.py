import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
import paddle
SEED = 2020
np.random.seed(SEED)

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 3)
        self.linear2 = paddle.nn.Linear(3, 1)

    def forward(self, x):
        if False:
            return 10
        out1 = self.linear1(x)
        out2 = self.linear2(out1)
        return [out1, out2]

class TestGradientAggregationInDy2Static(Dy2StTestBase):

    @test_legacy_and_pir
    def test_to_static(self):
        if False:
            while True:
                i = 10

        def simplenet_grad(inp, to_static=False):
            if False:
                return 10
            net = SimpleNet()
            if to_static:
                net = paddle.jit.to_static(net)
            loss = net(inp)
            loss[0].backward()
            return net.linear1.weight.grad
        inp = paddle.to_tensor(np.random.randn(10)).astype('float32')
        np.testing.assert_allclose(simplenet_grad(inp, True).numpy(), simplenet_grad(inp, False).numpy(), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()