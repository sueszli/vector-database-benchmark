import unittest
import numpy as np
import paddle
from paddle.base import core

class Optimization_ex1(paddle.nn.Layer):

    def __init__(self, shape, param_attr=paddle.nn.initializer.Uniform(low=-5.0, high=5.0), dtype='float32'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(np.random.randn(4, 4) + np.random.randn(4, 4) * 1j)

    def forward(self):
        if False:
            return 10
        loss = paddle.add(self.theta, self.A)
        return loss.real()

class TestComplexSimpleNet(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.devices = ['cpu']
        if core.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.iter = 10
        self.learning_rate = 0.5
        self.theta_size = [4, 4]

    def train(self, device):
        if False:
            return 10
        paddle.set_device(device)
        myLayer = Optimization_ex1(self.theta_size)
        optimizer = paddle.optimizer.Adam(learning_rate=self.learning_rate, parameters=myLayer.parameters())
        for itr in range(self.iter):
            loss = myLayer()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    def test_train_success(self):
        if False:
            for i in range(10):
                print('nop')
        for dev in self.devices:
            self.train(dev)
if __name__ == '__main__':
    unittest.main()