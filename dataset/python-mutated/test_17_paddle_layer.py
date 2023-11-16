import unittest
from test_case_base import TestCaseBase
import paddle

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        out1 = self.linear1(x)
        return out1

class SimpleNet_bound(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 1)

    def add(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.add(x)
        out1 = self.linear1(x)
        return out1

def net_call(x: paddle.Tensor, net):
    if False:
        for i in range(10):
            print('nop')
    return net(x)

def net_call_passed_by_user(x: paddle.Tensor, net_forward):
    if False:
        while True:
            i = 10
    return net_forward(x)

class SimpleNetWithSequenital(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.seq = paddle.nn.Sequential(paddle.nn.Linear(10, 10), paddle.nn.Linear(10, 10), paddle.nn.Linear(10, 1))

    def forward(self, x):
        if False:
            print('Hello World!')
        out1 = self.seq(x)
        return out1

class TestLayer(TestCaseBase):

    def test_layer(self):
        if False:
            while True:
                i = 10
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        net = SimpleNet()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, y, net)
        self.assert_results(net_call_passed_by_user, x, net.forward)

    def test_layer_with_sequential(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        net = SimpleNetWithSequenital()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, y, net)
        self.assert_results(net_call_passed_by_user, x, net.forward)

    def test_bound(self):
        if False:
            return 10
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        net = SimpleNet_bound()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, y, net)
if __name__ == '__main__':
    unittest.main()