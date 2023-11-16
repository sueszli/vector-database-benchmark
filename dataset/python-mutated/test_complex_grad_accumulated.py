import unittest
import numpy as np
import paddle
from paddle.base import core

class Optimization_ex1(paddle.nn.Layer):

    def __init__(self, shape, dtype, param_attr=paddle.nn.initializer.Uniform(low=-5.0, high=5.0)):
        if False:
            while True:
                i = 10
        super().__init__()
        self.theta0 = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.theta1 = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        self.A = paddle.to_tensor(np.random.random((4, 4)).astype(dtype) + np.random.random((4, 4)).astype(dtype) * 1j)
        self.B = paddle.to_tensor(np.random.random((4, 4)).astype(dtype) + np.random.random((4, 4)).astype(dtype) * 1j, stop_gradient=False)

    def forward(self, mode=1):
        if False:
            i = 10
            return i + 15
        jj = paddle.to_tensor(np.array([1j]).astype(np.complex64))
        if mode == 1:
            loss = paddle.sum(self.A + (self.theta0 + self.theta1 * jj)) * paddle.sum(self.A + (self.theta0 + self.theta1 * jj)).conj()
            return loss.real()
        elif mode == 2:
            self.theta = self.theta0 + self.theta1 * jj
            loss = paddle.sum(self.A + self.theta) * paddle.sum(self.A + self.theta).conj()
            return loss.real()
        elif mode == 3:
            loss = paddle.sum(self.A + self.B) * paddle.sum(self.A + self.B).conj()
            return loss.real()
        else:
            raise NotImplementedError

class TestComplexGradAccumulated(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.devices = ['cpu']
        if core.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.iter = 3
        self.learning_rate = 0.5
        self.dtypes = ['float32', 'float64']
        self.theta_size = [4, 4]

    def train(self, device, dtype, mode):
        if False:
            return 10
        paddle.set_device(device)
        myLayer = Optimization_ex1(self.theta_size, dtype)
        optimizer = paddle.optimizer.SGD(learning_rate=self.learning_rate, parameters=myLayer.parameters())
        for iter in range(self.iter):
            loss = myLayer(mode)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    def train_no_clear_grad(self, device, dtype, mode):
        if False:
            print('Hello World!')
        paddle.set_device(device)
        myLayer = Optimization_ex1(self.theta_size, dtype)
        optimizer = paddle.optimizer.SGD(learning_rate=self.learning_rate, parameters=myLayer.parameters())
        for iter in range(self.iter):
            loss = myLayer(mode)
            loss.backward()
            optimizer.step()

    def test_case_one_step(self):
        if False:
            return 10
        for dev in self.devices:
            for dtype in self.dtypes:
                self.train(dev, dtype, 1)
                self.train_no_clear_grad(dev, dtype, 1)

    def test_case_two_step(self):
        if False:
            i = 10
            return i + 15
        for dev in self.devices:
            for dtype in self.dtypes:
                self.train(dev, dtype, 2)
                self.train_no_clear_grad(dev, dtype, 2)

    def test_case_non_param(self):
        if False:
            for i in range(10):
                print('nop')
        for dev in self.devices:
            for dtype in self.dtypes:
                self.train(dev, dtype, 3)
                self.train_no_clear_grad(dev, dtype, 3)
if __name__ == '__main__':
    unittest.main()