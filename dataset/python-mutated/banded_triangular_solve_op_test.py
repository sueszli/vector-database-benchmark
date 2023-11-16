"""Tests for tensorflow.ops.math_ops.banded_triangular_solve."""
import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test

class BandedTriangularSolveOpTest(test.TestCase):

    def _verifySolveAllWays(self, x, y, dtypes, batch_dims=None):
        if False:
            return 10
        for lower in (False,):
            for adjoint in (False, True):
                for use_placeholder in (True, False):
                    self._verifySolve(x, y, lower=lower, adjoint=adjoint, batch_dims=batch_dims, use_placeholder=use_placeholder, dtypes=dtypes)

    def _verifySolveAllWaysReal(self, x, y, batch_dims=None):
        if False:
            for i in range(10):
                print('nop')
        self._verifySolveAllWays(x, y, (np.float32, np.float64), batch_dims)

    def _verifySolveAllWaysComplex(self, x, y, batch_dims=None):
        if False:
            i = 10
            return i + 15
        self._verifySolveAllWays(x, y, (np.complex64, np.complex128), batch_dims)

    def _verifySolve(self, x, y, lower=True, adjoint=False, batch_dims=None, use_placeholder=False, dtypes=(np.float32, np.float64)):
        if False:
            print('Hello World!')
        for np_type in dtypes:
            a = x.astype(np_type)
            b = y.astype(np_type)

            def make_diags(diags, lower=True):
                if False:
                    return 10
                n = len(diags[0])
                a = np.zeros(n * n, dtype=diags.dtype)
                if lower:
                    for (i, diag) in enumerate(diags):
                        a[n * i:n * n:n + 1] = diag[i:]
                else:
                    diags_flip = np.flip(diags, 0)
                    for (i, diag) in enumerate(diags_flip):
                        a[i:(n - i) * n:n + 1] = diag[:n - i]
                return a.reshape(n, n)
            if a.size > 0:
                a_np = make_diags(a, lower=lower)
            else:
                a_np = a
            if adjoint:
                a_np = np.conj(np.transpose(a_np))
            if batch_dims is not None:
                a = np.tile(a, batch_dims + [1, 1])
                a_np = np.tile(a_np, batch_dims + [1, 1])
                b = np.tile(b, batch_dims + [1, 1])
            with self.cached_session():
                a_tf = a
                b_tf = b
                if use_placeholder:
                    a_tf = array_ops.placeholder_with_default(a_tf, shape=None)
                    b_tf = array_ops.placeholder_with_default(b_tf, shape=None)
                tf_ans = linalg_ops.banded_triangular_solve(a_tf, b_tf, lower=lower, adjoint=adjoint)
                tf_val = self.evaluate(tf_ans)
                np_ans = np.linalg.solve(a_np, b)
                self.assertEqual(np_ans.shape, tf_val.shape)
                self.assertAllClose(np_ans, tf_val)

    @test_util.run_deprecated_v1
    def testSolve(self):
        if False:
            return 10
        matrix = np.array([[0.1]])
        rhs0 = np.array([[1.0]])
        self._verifySolveAllWaysReal(matrix, rhs0)
        matrix = np.array([[1.0, 4.0], [2.0, 3.0]])
        rhs0 = np.array([[1.0], [1.0]])
        self._verifySolveAllWaysReal(matrix, rhs0)
        rhs1 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        self._verifySolveAllWaysReal(matrix, rhs1)
        matrix = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, -1.0, -2.0, -3.0]])
        rhs0 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 2.0, 1.0], [0.0, -1.0, -1.0]])
        self._verifySolveAllWaysReal(matrix, rhs0)

    def testSolveBandSizeSmaller(self):
        if False:
            while True:
                i = 10
        rhs0 = np.random.randn(6, 4)
        matrix = 2.0 * np.random.uniform(size=[3, 6]) + 1.0
        self._verifySolveAllWaysReal(matrix, rhs0)
        matrix = 2.0 * np.random.uniform(size=[3, 6]) + 1.0
        self._verifySolveAllWaysReal(matrix, rhs0)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='ROCm does not support BLAS operations for complex types')
    @test_util.run_deprecated_v1
    def testSolveComplex(self):
        if False:
            while True:
                i = 10
        matrix = np.array([[0.1 + 1j * 0.1]])
        rhs0 = np.array([[1.0 + 1j]])
        self._verifySolveAllWaysComplex(matrix, rhs0)
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.complex64)
        matrix += 1j * matrix
        rhs0 = np.array([[1.0], [1.0]]).astype(np.complex64)
        rhs0 += 1j * rhs0
        self._verifySolveAllWaysComplex(matrix, rhs0)
        rhs1 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]).astype(np.complex64)
        rhs1 += 1j * rhs1
        self._verifySolveAllWaysComplex(matrix, rhs1)

    @test_util.run_deprecated_v1
    def testSolveBatch(self):
        if False:
            i = 10
            return i + 15
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        rhs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
        self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])
        matrix = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [-1.0, 1.0, 2.0, 3.0]])
        rhs = np.array([[-1.0, 2.0], [1.0, 1.0], [0.0, 1.0], [2.0, 3.0]])
        self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[2, 3])
        self._verifySolveAllWaysReal(matrix, rhs, batch_dims=[3, 2])

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='ROCm does not support BLAS operations for complex types')
    @test_util.run_deprecated_v1
    def testSolveBatchComplex(self):
        if False:
            while True:
                i = 10
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.complex64)
        matrix += 1j * matrix
        rhs = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]).astype(np.complex64)
        rhs += 1j * rhs
        self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[2, 3])
        self._verifySolveAllWaysComplex(matrix, rhs, batch_dims=[3, 2])

    @test_util.run_deprecated_v1
    def testWrongDimensions(self):
        if False:
            i = 10
            return i + 15
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        rhs = np.array([[1.0, 0.0]])
        with self.cached_session():
            with self.assertRaises(ValueError):
                self._verifySolve(matrix, rhs)
            with self.assertRaises(ValueError):
                self._verifySolve(matrix, rhs, batch_dims=[2, 3])
        matrix = np.ones((6, 4))
        rhs = np.ones((4, 2))
        with self.cached_session():
            with self.assertRaises(ValueError):
                self._verifySolve(matrix, rhs)
            with self.assertRaises(ValueError):
                self._verifySolve(matrix, rhs, batch_dims=[2, 3])

    @test_util.run_deprecated_v1
    @test_util.disable_xla('XLA cannot throw assertion errors during a kernel.')
    def testNotInvertible(self):
        if False:
            i = 10
            return i + 15
        singular_matrix = np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
        with self.cached_session():
            with self.assertRaisesOpError('Input matrix is not invertible.'):
                self._verifySolve(singular_matrix, singular_matrix)
            with self.assertRaisesOpError('Input matrix is not invertible.'):
                self._verifySolve(singular_matrix, singular_matrix, batch_dims=[2, 3])
if __name__ == '__main__':
    test.main()