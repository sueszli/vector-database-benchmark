import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
from paddle import base

class TestComplexTransposeLayer(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._dtypes = ['float32', 'float64']
        self._places = [paddle.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_transpose_by_complex_api(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self._dtypes:
            data = np.random.random((2, 3, 4, 5)).astype(dtype) + 1j * np.random.random((2, 3, 4, 5)).astype(dtype)
            perm = [3, 2, 0, 1]
            np_trans = np.transpose(data, perm)
            for place in self._places:
                with dg.guard(place):
                    var = dg.to_variable(data)
                    trans = paddle.transpose(var, perm=perm)
                np.testing.assert_allclose(trans.numpy(), np_trans, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()