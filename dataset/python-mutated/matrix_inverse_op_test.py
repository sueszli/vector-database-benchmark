"""Tests for tensorflow.ops.math_ops.matrix_inverse."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class InverseOpTest(xla_test.XLATestCase):

    def _verifyInverse(self, x, np_type):
        if False:
            for i in range(10):
                print('nop')
        for adjoint in (False, True):
            y = x.astype(np_type)
            with self.session() as sess:
                p = array_ops.placeholder(dtypes.as_dtype(y.dtype), y.shape, name='x')
                with self.test_scope():
                    inv = linalg_ops.matrix_inverse(p, adjoint=adjoint)
                    tf_ans = math_ops.matmul(inv, p, adjoint_b=adjoint)
                    np_ans = np.identity(y.shape[-1])
                    if x.ndim > 2:
                        tiling = list(y.shape)
                        tiling[-2:] = [1, 1]
                        np_ans = np.tile(np_ans, tiling)
                out = sess.run(tf_ans, feed_dict={p: y})
                self.assertAllClose(np_ans, out, rtol=0.001, atol=0.001)
                self.assertShapeEqual(y, tf_ans)

    def _verifyInverseReal(self, x):
        if False:
            i = 10
            return i + 15
        for np_type in self.float_types & {np.float64, np.float32}:
            self._verifyInverse(x, np_type)

    def _makeBatch(self, matrix1, matrix2):
        if False:
            i = 10
            return i + 15
        matrix_batch = np.concatenate([np.expand_dims(matrix1, 0), np.expand_dims(matrix2, 0)])
        matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
        return matrix_batch

    def testNonsymmetric(self):
        if False:
            return 10
        matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        matrix2 = np.array([[1.0, 3.0], [3.0, 5.0]])
        self._verifyInverseReal(matrix1)
        self._verifyInverseReal(matrix2)
        self._verifyInverseReal(self._makeBatch(matrix1, matrix2))

    def testSymmetricPositiveDefinite(self):
        if False:
            print('Hello World!')
        matrix1 = np.array([[2.0, 1.0], [1.0, 2.0]])
        matrix2 = np.array([[3.0, -1.0], [-1.0, 3.0]])
        self._verifyInverseReal(matrix1)
        self._verifyInverseReal(matrix2)
        self._verifyInverseReal(self._makeBatch(matrix1, matrix2))

    def testEmpty(self):
        if False:
            return 10
        self._verifyInverseReal(np.empty([0, 2, 2]))
        self._verifyInverseReal(np.empty([2, 0, 0]))
if __name__ == '__main__':
    googletest.main()