import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestHypotAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [10, 10]
        self.y_shape = [10, 1]
        self.x_np = np.random.uniform(-10, 10, self.x_shape).astype(np.float32)
        self.y_np = np.random.uniform(-10, 10, self.y_shape).astype(np.float32)

    def test_static_graph(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(name='input1', dtype='float32', shape=self.x_shape)
            y = paddle.static.data(name='input2', dtype='float32', shape=self.y_shape)
            out = paddle.hypot(x, y)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(base.default_main_program(), feed={'input1': self.x_np, 'input2': self.y_np}, fetch_list=[out])
            np_out = np.hypot(self.x_np, self.y_np)
            np.testing.assert_allclose(res[0], np_out, atol=1e-05, rtol=1e-05)
            paddle.disable_static()

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        result = paddle.hypot(x, y)
        np.testing.assert_allclose(np.hypot(self.x_np, self.y_np), result.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_error(self):
        if False:
            return 10
        x = paddle.to_tensor(self.x_np)
        y = 3.8
        self.assertRaises(TypeError, paddle.hypot, x, y)
        self.assertRaises(TypeError, paddle.hypot, y, x)

class TestHypotAPIBroadCast(TestHypotAPI):

    def setUp(self):
        if False:
            return 10
        self.x_np = np.arange(6).astype(np.float32)
        self.y_np = np.array([20]).astype(np.float32)
        self.x_shape = [6]
        self.y_shape = [1]

class TestHypotAPI3(TestHypotAPI):

    def setUp(self):
        if False:
            return 10
        self.x_shape = []
        self.y_shape = []
        self.x_np = np.random.uniform(-10, 10, self.x_shape).astype(np.float32)
        self.y_np = np.random.uniform(-10, 10, self.y_shape).astype(np.float32)

class TestHypotAPI4(TestHypotAPI):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x_shape = [1]
        self.y_shape = [1]
        self.x_np = np.random.uniform(-10, 10, self.x_shape).astype(np.float32)
        self.y_np = np.random.uniform(-10, 10, self.y_shape).astype(np.float32)
if __name__ == '__main__':
    unittest.main()