import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestRad2degAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_dtype = 'float64'
        self.x_np = np.array([3.142, -3.142, 6.283, -6.283, 1.57, -1.57]).astype(np.float64)
        self.x_shape = [6]
        self.out_np = np.rad2deg(self.x_np)

    def test_static_graph(self):
        if False:
            return 10
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(name='input', dtype=self.x_dtype, shape=self.x_shape)
            out = paddle.rad2deg(x)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(base.default_main_program(), feed={'input': self.x_np}, fetch_list=[out])
            self.assertTrue((np.array(out[0]) == self.out_np).all())

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x1 = paddle.to_tensor([3.142, -3.142, 6.283, -6.283, 1.57, -1.57])
        result1 = paddle.rad2deg(x1)
        np.testing.assert_allclose(self.out_np, result1.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestRad2degAPI2(TestRad2degAPI):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_np = [np.pi / 2]
        self.x_shape = [1]
        self.out_np = 90
        self.x_dtype = 'float32'

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        x2 = paddle.to_tensor([np.pi / 2])
        result2 = paddle.rad2deg(x2)
        np.testing.assert_allclose(90, result2.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestRad2degAPI3(TestRad2degAPI):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_np = [1]
        self.x_shape = [1]
        self.out_np = 180 / np.pi
        self.x_dtype = 'int64'

    def test_dygraph(self):
        if False:
            return 10
        paddle.disable_static()
        x2 = paddle.to_tensor([1])
        result2 = paddle.rad2deg(x2)
        np.testing.assert_allclose(180 / np.pi, result2.numpy(), rtol=1e-05)
        paddle.enable_static()