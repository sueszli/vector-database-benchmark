import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
from paddle import base, nn
from paddle.nn import functional as F

def sigmoid(x):
    if False:
        while True:
            i = 10
    return 1.0 / (1.0 + np.exp(-x))

def glu(x, dim=-1):
    if False:
        while True:
            i = 10
    (a, b) = np.split(x, 2, axis=dim)
    out = a * sigmoid(b)
    return out

class TestGLUV2(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.randn(5, 20)
        self.dim = -1
        self.out = glu(self.x, self.dim)

    def check_identity(self, place):
        if False:
            while True:
                i = 10
        with dg.guard(place):
            x_var = paddle.to_tensor(self.x)
            y_var = F.glu(x_var, self.dim)
            y_np = y_var.numpy()
        np.testing.assert_allclose(y_np, self.out)

    def test_case(self):
        if False:
            while True:
                i = 10
        self.check_identity(base.CPUPlace())
        if base.is_compiled_with_cuda():
            self.check_identity(base.CUDAPlace(0))

class TestGlu(unittest.TestCase):

    def glu_axis_size(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='float32')
        paddle.nn.functional.glu(x, axis=256)

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, self.glu_axis_size)

class TestnnGLU(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.randn(6, 20)
        self.dim = [-1, 0, 1]

    def check_identity(self, place):
        if False:
            while True:
                i = 10
        with dg.guard(place):
            x_var = paddle.to_tensor(self.x)
            for dim in self.dim:
                act = nn.GLU(dim)
                y_var = act(x_var)
                y_np = y_var.numpy()
                out = glu(self.x, dim)
                np.testing.assert_allclose(y_np, out)

    def test_case(self):
        if False:
            i = 10
            return i + 15
        self.check_identity(base.CPUPlace())
        if base.is_compiled_with_cuda():
            self.check_identity(base.CUDAPlace(0))
        act = nn.GLU(axis=0, name='test')
        self.assertTrue(act.extra_repr() == 'axis=0, name=test')

class TestnnGLUerror(unittest.TestCase):

    def glu_axis_size(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='float32')
        act = nn.GLU(256)
        act(x)

    def test_errors(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, self.glu_axis_size)
        act = nn.GLU(256)
        self.assertRaises(TypeError, act, 1)
        x_int32 = paddle.static.data(name='x_int32', shape=[10, 18], dtype='int32')
        self.assertRaises(TypeError, act, x_int32)
if __name__ == '__main__':
    unittest.main()