import unittest
import numpy as np
from numpy.random import random as rand
import paddle.base.dygraph as dg
from paddle import base, tensor

class TestComplexTraceLayer(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self._dtypes = ['float32', 'float64']
        self._places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self._places.append(base.CUDAPlace(0))

    def test_basic_api(self):
        if False:
            print('Hello World!')
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand([2, 20, 2, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = dg.to_variable(input)
                    result = tensor.trace(var_x, offset=1, axis1=0, axis2=2).numpy()
                    target = np.trace(input, offset=1, axis1=0, axis2=2)
                    np.testing.assert_allclose(result, target, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()