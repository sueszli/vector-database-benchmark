import unittest
import numpy as np
import paddle

def apply_to_static(net, use_cinn):
    if False:
        return 10
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

def softmax(x, axis):
    if False:
        print('Hello World!')
    'define composite rule of op softmax'
    is_amp = False
    from paddle.base.data_feeder import convert_dtype
    dtype = convert_dtype(x.dtype)
    if dtype in ['float16', 'uint16']:
        is_amp = True
        x = paddle.cast(x, 'float32')
    if not x.shape:
        res = paddle.exp(x - x)
        if is_amp:
            res = paddle.cast(res, 'float16')
        return res
    max_temp = paddle.max(x, axis, keepdim=True)
    max_temp.stop_gradient = True
    molecular = paddle.exp(x - max_temp)
    denominator = paddle.sum(molecular, axis=axis, keepdim=True)
    res = paddle.divide(molecular, denominator)
    if is_amp:
        res = paddle.cast(res, dtype)
    return res

def exp_sub(x):
    if False:
        for i in range(10):
            print('nop')
    y = paddle.exp(x)
    z = y - x
    return z

class CINNSubGraphNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.fn(x)
        return out

class CINNSoftmaxSubGraphNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fn = softmax

    def forward(self, x, axis=-1):
        if False:
            print('Hello World!')
        out = self.fn(x, axis=axis)
        return out

class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        self.shape = [64, 128]
        self.axis = -1
        self.prepare_data()

    def prepare_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = paddle.randn(self.shape, dtype='float32')
        self.x.stop_gradient = False

    def train(self, use_cinn):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        net = CINNSubGraphNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        return out

    def test_forward(self):
        if False:
            i = 10
            return i + 15
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-08)

class TestCinnSoftmax(TestCinnSubGraphBase):

    def train(self, use_cinn):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        net = CINNSoftmaxSubGraphNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x, self.axis)
        return out
if __name__ == '__main__':
    unittest.main()