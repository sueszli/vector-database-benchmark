import unittest
import numpy as np
import paddle
import paddle.base.dygraph as dg
from paddle.base import core
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import convert_np_dtype_to_dtype_

class TestComplexVariable(unittest.TestCase):

    def compare(self):
        if False:
            return 10
        a = np.array([[1.0 + 1j, 2.0 + 1j], [3.0 + 1j, 4.0 + 1j]]).astype(self._dtype)
        b = np.array([[1.0 + 1j, 1.0 + 1j]]).astype(self._dtype)
        with dg.guard():
            x = dg.to_variable(a, 'x')
            y = dg.to_variable(b)
            out = paddle.add(x, y)
            self.assertIsNotNone(f'{out}')
        np.testing.assert_allclose(out.numpy(), a + b, rtol=1e-05)
        self.assertEqual(out.dtype, convert_np_dtype_to_dtype_(self._dtype))
        self.assertEqual(out.shape, x.shape)

    def test_attrs(self):
        if False:
            return 10
        self._dtype = 'complex64'
        self.compare()
        self._dtype = 'complex128'
        self.compare()

    def test_convert_np_dtype_to_dtype(self):
        if False:
            return 10
        self.assertEqual(convert_np_dtype_to_dtype_(np.complex64), core.VarDesc.VarType.COMPLEX64)
        self.assertEqual(convert_np_dtype_to_dtype_(np.complex64), core.VarDesc.VarType.COMPLEX64)

    def test_convert_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(convert_dtype(core.VarDesc.VarType.COMPLEX64), 'complex64')
        self.assertEqual(convert_dtype(core.VarDesc.VarType.COMPLEX128), 'complex128')
if __name__ == '__main__':
    unittest.main()