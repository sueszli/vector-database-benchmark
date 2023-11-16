import unittest
import numpy as np
import paddle
from paddle import base, nn
from paddle.base import core
from paddle.nn import functional

class TestNNSigmoidAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_data()

    def init_data(self):
        if False:
            print('Hello World!')
        self.x_shape = [10, 15]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        if False:
            return 10
        return 1 / (1 + np.exp(-x))

    def ref_backward(self, y, dy):
        if False:
            i = 10
            return i + 15
        return dy * y * (1 - y)

    def check_static_api(self, place):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        main_program = paddle.static.Program()
        mysigmoid = nn.Sigmoid(name='api_sigmoid')
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=self.x_shape)
            x.stop_gradient = False
            y = mysigmoid(x)
            base.backward.append_backward(paddle.mean(y))
        exe = paddle.static.Executor(place)
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], self.y, rtol=1e-05)
        self.assertTrue(y.name.startswith('api_sigmoid'))

    def check_dynamic_api(self, place):
        if False:
            i = 10
            return i + 15
        paddle.disable_static(place)
        x = paddle.to_tensor(self.x)
        mysigmoid = nn.Sigmoid()
        y = mysigmoid(x)
        np.testing.assert_allclose(y.numpy(), self.y, rtol=1e-05)

    def test_check_api(self):
        if False:
            while True:
                i = 10
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            self.check_dynamic_api(place)
            self.check_static_api(place)

class TestNNFunctionalSigmoidAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.init_data()

    def init_data(self):
        if False:
            print('Hello World!')
        self.x_shape = [10, 15]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        if False:
            while True:
                i = 10
        return 1 / (1 + np.exp(-x))

    def check_static_api(self, place):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = functional.sigmoid(x, name='api_sigmoid')
        exe = paddle.static.Executor(base.CPUPlace())
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], self.y, rtol=1e-05)

    def check_dynamic_api(self):
        if False:
            return 10
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = functional.sigmoid(x)
        np.testing.assert_allclose(y.numpy(), self.y, rtol=1e-05)

    def test_check_api(self):
        if False:
            i = 10
            return i + 15
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            self.check_static_api(place)
            self.check_dynamic_api()