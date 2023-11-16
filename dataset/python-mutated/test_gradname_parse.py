import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase
import paddle
from paddle.nn import BatchNorm, Linear

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear0 = Linear(100, 50)
        self.linear1 = Linear(50, 10)
        self.bn0 = BatchNorm(50)
        self.bn1 = BatchNorm(10)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x1 = self.linear0(x)
        x2 = self.bn0(x1)
        x3 = self.linear1(x2)
        x4 = self.bn1(x3)
        dx = paddle.grad(x4, x)
        return dx[0]

class TestGradNameParse(Dy2StTestBase):

    def test_grad_name_parse(self):
        if False:
            i = 10
            return i + 15
        net = SimpleNet()
        opt = paddle.optimizer.Adam(learning_rate=0.1, parameters=net.parameters(), weight_decay=paddle.regularizer.L1Decay(0.01))
        net = paddle.jit.to_static(net)
        inp = paddle.rand([100, 100], dtype='float32')
        inp.stop_gradient = False
        out = net(inp)
        loss = out.mean()
        loss.backward()
        for (name, param) in net.bn1.named_parameters():
            if name in ['bn_scale', 'bn_offset']:
                assert param.shape == param.grad.shape
        opt.minimize(loss)

def tanh_high_order_grad(x):
    if False:
        return 10
    y = paddle.tanh(x)
    return paddle.grad(y, x, create_graph=True)[0]

class TestTanhHighOrderGrad(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        self.func = tanh_high_order_grad
        x1 = paddle.ones((1,))
        x1.stop_gradient = False
        self.dy_input = (x1,)
        self.dy_grad_input = (x1,)
        x2 = paddle.ones((1,))
        x2.stop_gradient = False
        self.dy2st_input = (x2,)
        self.dy2st_grad_input = (x2,)

    def test_run(self):
        if False:
            return 10
        try:
            dy_out = self.func(*self.dy_input)
            dy_grad = paddle.grad(dy_out, self.dy_grad_input)
        except:
            dy_grad = [None for i in self.dy_grad_input]
        dy_grad = [t.numpy() if isinstance(t, paddle.Tensor) else t for t in dy_grad]
        dy2st_out = paddle.jit.to_static(self.func)(*self.dy2st_input)
        dy2st_grad = paddle.grad(dy2st_out, self.dy2st_grad_input)
        dy2st_grad = [t.numpy() if isinstance(t, paddle.Tensor) else t for t in dy_grad]
        np.testing.assert_equal(dy_grad, dy2st_grad)
        dy_input_grad = [t.grad.numpy() if isinstance(t.grad, paddle.Tensor) else None for t in self.dy_input]
        dy2st_input_grad = [t.grad.numpy() if isinstance(t.grad, paddle.Tensor) else None for t in self.dy2st_input]
        np.testing.assert_equal(dy_input_grad, dy2st_input_grad)

def matmul_high_order_grad(x, y):
    if False:
        for i in range(10):
            print('nop')
    z = paddle.matmul(x, y)
    g = paddle.grad(z, [x, y], create_graph=True)
    return g[0]

class TestMatMulHighOrderGrad1(TestTanhHighOrderGrad):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.func = matmul_high_order_grad
        x1 = paddle.ones([1])
        x1.stop_gradient = False
        y1 = paddle.ones([1])
        y1.stop_gradient = False
        self.dy_input = (x1, y1)
        self.dy_grad_input = (x1,)
        x2 = paddle.ones([1])
        x2.stop_gradient = False
        y2 = paddle.ones([1])
        y2.stop_gradient = False
        self.dy2st_input = (x2, y2)
        self.dy2st_grad_input = (x2,)

class TestMatMulHighOrderGrad2(TestTanhHighOrderGrad):

    def setUp(self):
        if False:
            print('Hello World!')
        self.func = matmul_high_order_grad
        x = np.random.randn(5, 5)
        y = np.random.randn(5, 5)
        x1 = paddle.to_tensor(x)
        x1.stop_gradient = False
        y1 = paddle.to_tensor(y)
        y1.stop_gradient = True
        self.dy_input = (x1, y1)
        self.dy_grad_input = (x1,)
        x2 = paddle.to_tensor(x)
        x2.stop_gradient = False
        y2 = paddle.to_tensor(y)
        y2.stop_gradient = True
        self.dy2st_input = (x2, y2)
        self.dy2st_grad_input = (x2,)
if __name__ == '__main__':
    unittest.main()