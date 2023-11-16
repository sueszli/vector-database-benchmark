import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()

class TestDeg2radAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x_dtype = 'float64'
        self.x_np = np.array([180.0, -180.0, 360.0, -360.0, 90.0, -90.0]).astype(np.float64)
        self.x_shape = [6]
        self.out_np = np.deg2rad(self.x_np)

    def test_static_graph(self):
        if False:
            while True:
                i = 10
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(name='input', dtype=self.x_dtype, shape=self.x_shape)
            out = paddle.deg2rad(x)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(base.default_main_program(), feed={'input': self.x_np}, fetch_list=[out])
            self.assertTrue((np.array(out[0]) == self.out_np).all())

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x1 = paddle.to_tensor([180.0, -180.0, 360.0, -360.0, 90.0, -90.0])
        result1 = paddle.deg2rad(x1)
        np.testing.assert_allclose(self.out_np, result1.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestDeg2radAPI2(TestDeg2radAPI):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x_np = [180]
        self.x_shape = [1]
        self.out_np = np.pi
        self.x_dtype = 'int64'

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        x2 = paddle.to_tensor([180])
        result2 = paddle.deg2rad(x2)
        np.testing.assert_allclose(np.pi, result2.numpy(), rtol=1e-05)
        paddle.enable_static()