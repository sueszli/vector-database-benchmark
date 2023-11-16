import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
import paddle
from paddle.jit import to_static

class NetWithParameterList(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            return 10
        super().__init__()
        weight = self.create_parameter([in_size, out_size])
        bias = self.create_parameter([out_size], is_bias=True)
        self.params = paddle.nn.ParameterList([weight, bias])

    @to_static
    def forward(self, x):
        if False:
            return 10
        out = paddle.matmul(x, self.params[0])
        out = paddle.add(out, self.params[1])
        out = paddle.tanh(out)
        return out

class NetWithParameterListIter(NetWithParameterList):

    def __init__(self, in_size, out_size):
        if False:
            i = 10
            return i + 15
        super().__init__(in_size, out_size)

    @to_static
    def forward(self, x):
        if False:
            return 10
        params = list(self.params.__iter__())
        out = paddle.matmul(x, params[0])
        out = paddle.add(out, params[1])
        out = paddle.tanh(out)
        return out

class TestParameterList(Dy2StTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.seed = 2021
        self.iter_num = 5

    def train(self, is_iter, to_static):
        if False:
            while True:
                i = 10
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        paddle.jit.enable_to_static(to_static)
        if is_iter:
            net = NetWithParameterList(10, 3)
        else:
            net = NetWithParameterListIter(10, 3)
        sgd = paddle.optimizer.SGD(0.1, parameters=net.parameters())
        for batch_id in range(self.iter_num):
            x = paddle.rand([4, 10], dtype='float32')
            out = net(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
        return loss

    @test_legacy_and_pir
    def test_parameter_list(self):
        if False:
            for i in range(10):
                print('nop')
        static_loss = self.train(False, to_static=True)
        dygraph_loss = self.train(False, to_static=False)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)

class NetWithRawParamList(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        weight = self.add_parameter('w', self.create_parameter([in_size, out_size]))
        bias = self.add_parameter('b', self.create_parameter([out_size], is_bias=True))
        self.params = [weight]
        self.bias_dict = {'b': bias}

    @to_static
    def forward(self, x):
        if False:
            while True:
                i = 10
        out = paddle.matmul(x, self.params[0])
        out = paddle.add(out, self.bias_dict['b'])
        out = paddle.tanh(out)
        return out

class TestRawParameterList(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.seed = 2021
        self.iter_num = 5

    def init_net(self):
        if False:
            print('Hello World!')
        self.net = NetWithRawParamList(10, 3)

    def train(self, to_static):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        paddle.jit.enable_to_static(to_static)
        self.init_net()
        sgd = paddle.optimizer.SGD(0.1, parameters=self.net.parameters())
        for batch_id in range(self.iter_num):
            x = paddle.rand([4, 10], dtype='float32')
            out = self.net(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
        return loss

    @test_legacy_and_pir
    def test_parameter_list(self):
        if False:
            while True:
                i = 10
        static_loss = self.train(to_static=True)
        dygraph_loss = self.train(to_static=False)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)

class NetWithSubLayerParamList(paddle.nn.Layer):

    def __init__(self, sub_layer):
        if False:
            print('Hello World!')
        super().__init__()
        self.sub_layer = sub_layer
        self.params = [sub_layer.weight]
        self.bias_dict = {'b': sub_layer.bias}

    @to_static
    def forward(self, x):
        if False:
            while True:
                i = 10
        out = paddle.matmul(x, self.params[0])
        out = paddle.add(out, self.bias_dict['b'])
        out = paddle.tanh(out)
        return out

class TestSubLayerParameterList(TestRawParameterList):

    def init_net(self):
        if False:
            i = 10
            return i + 15
        fc = paddle.nn.Linear(10, 3)
        self.net = NetWithSubLayerParamList(fc)
if __name__ == '__main__':
    unittest.main()