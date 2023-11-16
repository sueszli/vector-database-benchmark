import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestGcdAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x_np = [12]
        self.y_np = [20]
        self.x_shape = [1]
        self.y_shape = [1]

    def test_static_graph(self):
        if False:
            print('Hello World!')
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(name='input1', dtype='int32', shape=self.x_shape)
            y = paddle.static.data(name='input2', dtype='int32', shape=self.y_shape)
            out = paddle.gcd(x, y)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(base.default_main_program(), feed={'input1': self.x_np, 'input2': self.y_np}, fetch_list=[out])
            self.assertTrue((np.array(res[0]) == np.gcd(self.x_np, self.y_np)).all())

    def test_dygraph(self):
        if False:
            return 10
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        result = paddle.gcd(x, y)
        np.testing.assert_allclose(np.gcd(self.x_np, self.y_np), result.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestGcdAPI2(TestGcdAPI):

    def setUp(self):
        if False:
            return 10
        self.x_np = np.arange(6).astype(np.int32)
        self.y_np = np.array([20]).astype(np.int32)
        self.x_shape = [6]
        self.y_shape = [1]

class TestGcdAPI3(TestGcdAPI):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_np = 0
        self.y_np = 20
        self.x_shape = []
        self.y_shape = []

class TestGcdAPI4(TestGcdAPI):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_np = [0]
        self.y_np = [0]
        self.x_shape = [1]
        self.y_shape = [1]

class TestGcdAPI5(TestGcdAPI):

    def setUp(self):
        if False:
            return 10
        self.x_np = 12
        self.y_np = -20
        self.x_shape = []
        self.y_shape = []