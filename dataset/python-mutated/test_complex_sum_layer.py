import unittest
import numpy as np
from numpy.random import random as rand
import paddle
import paddle.base.dygraph as dg
from paddle import base, tensor

class TestComplexSumLayer(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._dtypes = ['float32', 'float64']
        self._places = [paddle.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_complex_basic_api(self):
        if False:
            return 10
        for dtype in self._dtypes:
            input = rand([2, 10, 10]).astype(dtype) + 1j * rand([2, 10, 10]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = dg.to_variable(input)
                    result = tensor.sum(var_x, axis=[1, 2]).numpy()
                    target = np.sum(input, axis=(1, 2))
                    np.testing.assert_allclose(result, target, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()