"""Tests for tensorflow.ops.tf.Cholesky."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test

class CholeskyOpTest(xla_test.XLATestCase):

    @property
    def float_types(self):
        if False:
            return 10
        return set(super(CholeskyOpTest, self).float_types).intersection((np.float64, np.float32, np.complex64, np.complex128))

    def _verifyCholeskyBase(self, sess, placeholder, x, chol, verification, atol):
        if False:
            i = 10
            return i + 15
        (chol_np, verification_np) = sess.run([chol, verification], {placeholder: x})
        self.assertAllClose(x, verification_np, atol=atol)
        self.assertShapeEqual(x, chol)
        if chol_np.shape[-1] > 0:
            chol_reshaped = np.reshape(chol_np, (-1, chol_np.shape[-2], chol_np.shape[-1]))
            for chol_matrix in chol_reshaped:
                self.assertAllClose(chol_matrix, np.tril(chol_matrix), atol=atol)
                self.assertTrue((np.diag(chol_matrix) > 0.0).all())

    def _verifyCholesky(self, x, atol=1e-06):
        if False:
            i = 10
            return i + 15
        with self.session() as sess:
            placeholder = array_ops.placeholder(dtypes.as_dtype(x.dtype), shape=x.shape)
            with self.test_scope():
                chol = linalg_ops.cholesky(placeholder)
            verification = test_util.matmul_without_tf32(chol, chol, adjoint_b=True)
            self._verifyCholeskyBase(sess, placeholder, x, chol, verification, atol)

    def testBasic(self):
        if False:
            while True:
                i = 10
        data = np.array([[4.0, -1.0, 2.0], [-1.0, 6.0, 0], [2.0, 0.0, 5.0]])
        for dtype in self.float_types:
            self._verifyCholesky(data.astype(dtype))

    def testBatch(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.float_types:
            simple_array = np.array([[[1.0, 0.0], [0.0, 5.0]]], dtype=dtype)
            self._verifyCholesky(simple_array)
            self._verifyCholesky(np.vstack((simple_array, simple_array)))
            odd_sized_array = np.array([[[4.0, -1.0, 2.0], [-1.0, 6.0, 0], [2.0, 0.0, 5.0]]], dtype=dtype)
            self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))
            matrices = np.random.rand(10, 5, 5).astype(dtype)
            for i in range(10):
                matrices[i] = np.dot(matrices[i].T, matrices[i])
            self._verifyCholesky(matrices, atol=0.0001)

    @test_util.run_v2_only
    def testNonSquareMatrixV2(self):
        if False:
            return 10
        for dtype in self.float_types:
            with self.assertRaises(errors.InvalidArgumentError):
                linalg_ops.cholesky(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=dtype))
            with self.assertRaises(errors.InvalidArgumentError):
                linalg_ops.cholesky(np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], dtype=dtype))

    @test_util.run_v1_only('Different error types')
    def testNonSquareMatrixV1(self):
        if False:
            while True:
                i = 10
        for dtype in self.float_types:
            with self.assertRaises(ValueError):
                linalg_ops.cholesky(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=dtype))
            with self.assertRaises(ValueError):
                linalg_ops.cholesky(np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], dtype=dtype))

    @test_util.run_v2_only
    def testWrongDimensionsV2(self):
        if False:
            return 10
        for dtype in self.float_types:
            tensor3 = constant_op.constant([1.0, 2.0], dtype=dtype)
            with self.assertRaises(errors.InvalidArgumentError):
                linalg_ops.cholesky(tensor3)
            with self.assertRaises(errors.InvalidArgumentError):
                linalg_ops.cholesky(tensor3)

    @test_util.run_v1_only('Different error types')
    def testWrongDimensionsV1(self):
        if False:
            while True:
                i = 10
        for dtype in self.float_types:
            tensor3 = constant_op.constant([1.0, 2.0], dtype=dtype)
            with self.assertRaises(ValueError):
                linalg_ops.cholesky(tensor3)
            with self.assertRaises(ValueError):
                linalg_ops.cholesky(tensor3)

    def testLarge2000x2000(self):
        if False:
            print('Hello World!')
        n = 2000
        shape = (n, n)
        data = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(np.ones(n).astype(np.float32))
        self._verifyCholesky(data, atol=0.0001)

    def testMatrixConditionNumbers(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.float_types:
            condition_number = 1000
            size = 20
            matrix = np.random.rand(size, size)
            matrix = np.dot(matrix.T, matrix)
            (_, w) = np.linalg.eigh(matrix)
            v = np.exp(-np.log(condition_number) * np.linspace(0, size, size) / size)
            matrix = np.dot(np.dot(w, np.diag(v)), w.T).astype(dtype)
            self._verifyCholesky(matrix, atol=0.0001)
if __name__ == '__main__':
    test.main()