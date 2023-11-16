import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
from paddle import base

class TestComplexReshape(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._dtypes = ['float32', 'float64']
        self._places = [paddle.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_shape_norm_dims(self):
        if False:
            return 10
        for dtype in self._dtypes:
            x_np = np.random.randn(2, 3, 4).astype(dtype) + 1j * np.random.randn(2, 3, 4).astype(dtype)
            shape = (2, -1)
            for place in self._places:
                with dg.guard(place):
                    x_var = dg.to_variable(x_np)
                    y_var = paddle.reshape(x_var, shape)
                    y_np = y_var.numpy()
                    np.testing.assert_allclose(np.reshape(x_np, shape), y_np, rtol=1e-05)

    def test_shape_omit_dims(self):
        if False:
            while True:
                i = 10
        for dtype in self._dtypes:
            x_np = np.random.randn(2, 3, 4).astype(dtype) + 1j * np.random.randn(2, 3, 4).astype(dtype)
            shape = (0, -1)
            shape_ = (2, 12)
            for place in self._places:
                with dg.guard(place):
                    x_var = dg.to_variable(x_np)
                    y_var = paddle.reshape(x_var, shape)
                    y_np = y_var.numpy()
                    np.testing.assert_allclose(np.reshape(x_np, shape_), y_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()