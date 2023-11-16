import unittest
import numpy as np
from numpy import linalg as LA
from op_test import OpTest
import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.framework import in_dynamic_mode

def test_squared_l2_norm(x):
    if False:
        return 10
    if in_dynamic_mode():
        return _C_ops.squared_l2_norm(x)
    else:
        return _legacy_C_ops.squared_l2_norm(x)

class TestSquaredL2NormF16Op(unittest.TestCase):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        X = np.random.uniform(-0.1, 0.1, (8, 5, 10)).astype('float32')
        return X

    def check_main(self, x_np, dtype):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        y = test_squared_l2_norm(x)
        x_g = paddle.grad(y, [x])
        paddle.enable_static()
        return (y, x_g)

    def test_main(self):
        if False:
            while True:
                i = 10
        x_np = self.init_test_case()
        (y_np_1, x_g_np_1) = self.check_main(x_np, 'float32')
        (y_np_2, x_g_np_2) = self.check_main(x_np, 'float16')

        def assert_equal(x, y):
            if False:
                i = 10
                return i + 15
            np.testing.assert_allclose(x, y, rtol=1e-05, atol=0.0)
        assert_equal(y_np_1, y_np_2)
        assert_equal(x_g_np_1, x_g_np_2)

class TestSquaredL2NormF16Op1(TestSquaredL2NormF16Op):

    def init_test_case(self):
        if False:
            return 10
        X = np.random.uniform(-2.0, 2.0, (30, 10)).astype('float32')
        return X

class TestSquaredL2NormF16Op2(TestSquaredL2NormF16Op):

    def init_test_case(self):
        if False:
            return 10
        X = np.random.uniform(-5.0, 5.0, (20, 10, 20)).astype('float32')
        return X

class TestL2LossOp(OpTest):
    """Test squared_l2_norm"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.python_api = test_squared_l2_norm
        self.op_type = 'squared_l2_norm'
        self.max_relative_error = 0.05
        X = np.random.uniform(-1, 1, (13, 19)).astype('float32')
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.array([np.square(LA.norm(X))])}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out', max_relative_error=self.max_relative_error)

class TestL2LossDeterministic(unittest.TestCase):

    def check_place(self, place):
        if False:
            for i in range(10):
                print('nop')
        with paddle.base.dygraph.guard(place):
            x_np = np.random.rand(5, 11, 13).astype('float32')
            x = paddle.to_tensor(x_np)
            y1 = _legacy_C_ops.squared_l2_norm(x)
            y2 = _legacy_C_ops.squared_l2_norm(x)
            np.testing.assert_array_equal(y1.numpy(), y2.numpy())

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_place(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.check_place(paddle.CUDAPlace(0))
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()