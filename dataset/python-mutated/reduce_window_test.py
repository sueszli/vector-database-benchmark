"""Tests for xla.reduce_window."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

class ReduceWindowTest(xla_test.XLATestCase):
    """Test cases for xla.reduce_window."""

    def _reduce_window(self, operand, init, reducer, **kwargs):
        if False:
            return 10
        with self.session():
            placeholder = array_ops.placeholder(operand.dtype)
            with self.test_scope():
                output = xla.reduce_window(placeholder, init, reducer, **kwargs)
            return output.eval(feed_dict={placeholder: operand})

    def testReduceWindow(self):
        if False:
            return 10
        for dtype in set(self.numeric_types).intersection(set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

            @function.Defun(dtype, dtype)
            def sum_reducer(x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y

            @function.Defun(dtype, dtype)
            def mul_reducer(x, y):
                if False:
                    return 10
                return x * y
            self.assertAllClose(np.array([3, 5, 7, 9, 11, 13], dtype=dtype), self._reduce_window(np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype), 0.0, sum_reducer, window_dimensions=[2]))
            self.assertAllClose(np.array([3, 7, 11], dtype=dtype), self._reduce_window(np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype), 0.0, sum_reducer, window_dimensions=[2], window_strides=[2]))
            self.assertAllClose(np.array([1, 4, 7], dtype=dtype), self._reduce_window(np.array([1, 2, 3, 4, 5, 6, 7], dtype=dtype), 0.0, sum_reducer, window_dimensions=[1], window_strides=[3]))
            self.assertAllClose(np.array([[24, 36, 24], [96, 0, 0]], dtype=dtype), self._reduce_window(np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 0, 1]], dtype=dtype), 1.0, mul_reducer, window_dimensions=[2, 2], window_strides=[1, 1]))
            self.assertAllClose(np.array([[0, 0, 0], [5, 10, 5], [2, 4, 1], [0, 0, 0]], dtype=dtype), self._reduce_window(np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 0, 1]], dtype=dtype), 0.0, sum_reducer, window_dimensions=[2, 2], window_strides=[2, 2], padding=[[2, 3], [1, 2]]))
if __name__ == '__main__':
    googletest.main()