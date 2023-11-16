import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
import paddle

class FakeNet:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.var = paddle.to_tensor([2.0])
f = FakeNet()
g = paddle.to_tensor([1.0])

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def forward(self, x):
        if False:
            return 10
        t = g * 2 + x
        t = f.var * t
        return t

class TestFallback(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = paddle.to_tensor(1.0).astype('int')

    @test_legacy_and_pir
    def test_name_load(self):
        if False:
            return 10
        net_dy = Net()
        net_st = Net()
        output_dy = net_dy(self.x)
        output_st = paddle.jit.to_static(net_st)(self.x)
        np.testing.assert_allclose(output_dy.numpy(), output_st.numpy())

class TestLoad2(Dy2StTestBase):

    @test_legacy_and_pir
    def test_name_load_nograd(self):
        if False:
            return 10

        @paddle.no_grad()
        def func(x):
            if False:
                i = 10
                return i + 15
            x = paddle.shape(x)
            return x
        x = paddle.to_tensor([[3, 3], [1, 1]])
        output_st = paddle.jit.to_static(func)(x)
        output_dy = func(x)
        np.testing.assert_allclose(output_dy.numpy(), output_st.numpy())
if __name__ == '__main__':
    unittest.main()