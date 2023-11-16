"""Tests for tridiagonal solve ops."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl
from tensorflow.python.platform import test
_sample_diags = np.array([[2, 1, 4, 0], [1, 3, 2, 2], [0, 1, -1, 1]], dtype=np.float32)
_sample_rhs = np.array([1, 2, 3, 4], dtype=np.float32)
_sample_result = np.array([-9, 5, -4, 4], dtype=np.float32)

def _tfconst(array):
    if False:
        i = 10
        return i + 15
    return constant_op.constant(array, dtype=dtypes.float32)

def _tf_ones(shape):
    if False:
        return 10
    return array_ops.ones(shape, dtype=dtypes.float64)

class TridiagonalSolveOpsTest(xla_test.XLATestCase):
    """Test for tri-diagonal matrix related ops."""

    def testTridiagonalSolverSolves1Rhs(self):
        if False:
            while True:
                i = 10
        np.random.seed(19)
        batch_size = 8
        num_dims = 11
        diagonals_np = np.random.normal(size=(batch_size, 3, num_dims)).astype(np.float32)
        rhs_np = np.random.normal(size=(batch_size, num_dims, 1)).astype(np.float32)
        with self.session() as sess, self.test_scope():
            diags = array_ops.placeholder(shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
            rhs = array_ops.placeholder(shape=(batch_size, num_dims, 1), dtype=dtypes.float32)
            x_np = sess.run(linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=False), feed_dict={diags: diagonals_np, rhs: rhs_np})[:, :, 0]
        superdiag_np = diagonals_np[:, 0]
        diag_np = diagonals_np[:, 1]
        subdiag_np = diagonals_np[:, 2]
        y = np.zeros((batch_size, num_dims), dtype=np.float32)
        for i in range(num_dims):
            if i == 0:
                y[:, i] = diag_np[:, i] * x_np[:, i] + superdiag_np[:, i] * x_np[:, i + 1]
            elif i == num_dims - 1:
                y[:, i] = subdiag_np[:, i] * x_np[:, i - 1] + diag_np[:, i] * x_np[:, i]
            else:
                y[:, i] = subdiag_np[:, i] * x_np[:, i - 1] + diag_np[:, i] * x_np[:, i] + superdiag_np[:, i] * x_np[:, i + 1]
        self.assertAllClose(y, rhs_np[:, :, 0], rtol=0.0001, atol=0.0001)

    def testTridiagonalSolverSolvesKRhs(self):
        if False:
            while True:
                i = 10
        np.random.seed(19)
        batch_size = 8
        num_dims = 11
        num_rhs = 5
        diagonals_np = np.random.normal(size=(batch_size, 3, num_dims)).astype(np.float32)
        rhs_np = np.random.normal(size=(batch_size, num_dims, num_rhs)).astype(np.float32)
        with self.session() as sess, self.test_scope():
            diags = array_ops.placeholder(shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
            rhs = array_ops.placeholder(shape=(batch_size, num_dims, num_rhs), dtype=dtypes.float32)
            x_np = sess.run(linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=False), feed_dict={diags: diagonals_np, rhs: rhs_np})
        superdiag_np = diagonals_np[:, 0]
        diag_np = diagonals_np[:, 1]
        subdiag_np = diagonals_np[:, 2]
        for eq in range(num_rhs):
            y = np.zeros((batch_size, num_dims), dtype=np.float32)
            for i in range(num_dims):
                if i == 0:
                    y[:, i] = diag_np[:, i] * x_np[:, i, eq] + superdiag_np[:, i] * x_np[:, i + 1, eq]
                elif i == num_dims - 1:
                    y[:, i] = subdiag_np[:, i] * x_np[:, i - 1, eq] + diag_np[:, i] * x_np[:, i, eq]
                else:
                    y[:, i] = subdiag_np[:, i] * x_np[:, i - 1, eq] + diag_np[:, i] * x_np[:, i, eq] + superdiag_np[:, i] * x_np[:, i + 1, eq]
            self.assertAllClose(y, rhs_np[:, :, eq], rtol=0.0001, atol=0.0001)

    def _test(self, diags, rhs, expected, diags_format='compact', transpose_rhs=False):
        if False:
            i = 10
            return i + 15
        with self.session() as sess, self.test_scope():
            self.assertAllClose(sess.run(linalg_impl.tridiagonal_solve(_tfconst(diags), _tfconst(rhs), diags_format, transpose_rhs, conjugate_rhs=False, partial_pivoting=False)), np.asarray(expected, dtype=np.float32))

    def _testWithDiagonalLists(self, diags, rhs, expected, diags_format='compact', transpose_rhs=False):
        if False:
            print('Hello World!')
        with self.session() as sess, self.test_scope():
            self.assertAllClose(sess.run(linalg_impl.tridiagonal_solve([_tfconst(x) for x in diags], _tfconst(rhs), diags_format, transpose_rhs, conjugate_rhs=False, partial_pivoting=False)), sess.run(_tfconst(expected)))

    def testReal(self):
        if False:
            return 10
        self._test(diags=_sample_diags, rhs=_sample_rhs, expected=_sample_result)

    def test3x3(self):
        if False:
            while True:
                i = 10
        self._test(diags=[[2.0, -1.0, 0.0], [1.0, 3.0, 1.0], [0.0, -1.0, -2.0]], rhs=[1.0, 2.0, 3.0], expected=[-3.0, 2.0, 7.0])

    def test2x2(self):
        if False:
            return 10
        self._test(diags=[[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]], rhs=[1.0, 4.0], expected=[-5.0, 3.0])

    def test1x1(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(diags=[[0], [3], [0]], rhs=[6], expected=[2])

    def test0x0(self):
        if False:
            return 10
        self._test(diags=np.zeros(shape=(3, 0), dtype=np.float32), rhs=np.zeros(shape=(0, 1), dtype=np.float32), expected=np.zeros(shape=(0, 1), dtype=np.float32))

    def test2x2WithMultipleRhs(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(diags=[[2, 0], [1, 3], [0, 1]], rhs=[[1, 2, 3], [4, 8, 12]], expected=[[-5, -10, -15], [3, 6, 9]])

    def test1x1WithMultipleRhs(self):
        if False:
            return 10
        self._test(diags=[[0], [3], [0]], rhs=[[6, 9, 12]], expected=[[2, 3, 4]])

    @test_util.disable_mlir_bridge('Error messages differ')
    def testPartialPivotingRaises(self):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        batch_size = 8
        num_dims = 11
        num_rhs = 5
        diagonals_np = np.random.normal(size=(batch_size, 3, num_dims)).astype(np.float32)
        rhs_np = np.random.normal(size=(batch_size, num_dims, num_rhs)).astype(np.float32)
        with self.session() as sess, self.test_scope():
            with self.assertRaisesRegex(errors_impl.UnimplementedError, 'Current implementation does not yet support pivoting.'):
                diags = array_ops.placeholder(shape=(batch_size, 3, num_dims), dtype=dtypes.float32)
                rhs = array_ops.placeholder(shape=(batch_size, num_dims, num_rhs), dtype=dtypes.float32)
                sess.run(linalg_impl.tridiagonal_solve(diags, rhs, partial_pivoting=True), feed_dict={diags: diagonals_np, rhs: rhs_np})

    def testDiagonal(self):
        if False:
            i = 10
            return i + 15
        self._test(diags=[[0, 0, 0, 0], [1, 2, -1, -2], [0, 0, 0, 0]], rhs=[1, 2, 3, 4], expected=[1, 1, -3, -2])

    def testUpperTriangular(self):
        if False:
            while True:
                i = 10
        self._test(diags=[[2, 4, -1, 0], [1, 3, 1, 2], [0, 0, 0, 0]], rhs=[1, 6, 4, 4], expected=[13, -6, 6, 2])

    def testLowerTriangular(self):
        if False:
            while True:
                i = 10
        self._test(diags=[[0, 0, 0, 0], [2, -1, 3, 1], [0, 1, 4, 2]], rhs=[4, 5, 6, 1], expected=[2, -3, 6, -11])

    def testWithTwoRightHandSides(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(diags=_sample_diags, rhs=np.transpose([_sample_rhs, 2 * _sample_rhs]), expected=np.transpose([_sample_result, 2 * _sample_result]))

    def testBatching(self):
        if False:
            return 10
        self._test(diags=np.array([_sample_diags, -_sample_diags]), rhs=np.array([_sample_rhs, 2 * _sample_rhs]), expected=np.array([_sample_result, -2 * _sample_result]))

    def testWithTwoBatchingDimensions(self):
        if False:
            i = 10
            return i + 15
        self._test(diags=np.array([[_sample_diags, -_sample_diags, _sample_diags], [-_sample_diags, _sample_diags, -_sample_diags]]), rhs=np.array([[_sample_rhs, 2 * _sample_rhs, 3 * _sample_rhs], [4 * _sample_rhs, 5 * _sample_rhs, 6 * _sample_rhs]]), expected=np.array([[_sample_result, -2 * _sample_result, 3 * _sample_result], [-4 * _sample_result, 5 * _sample_result, -6 * _sample_result]]))

    def testBatchingAndTwoRightHandSides(self):
        if False:
            for i in range(10):
                print('nop')
        rhs = np.transpose([_sample_rhs, 2 * _sample_rhs])
        expected_result = np.transpose([_sample_result, 2 * _sample_result])
        self._test(diags=np.array([_sample_diags, -_sample_diags]), rhs=np.array([rhs, 2 * rhs]), expected=np.array([expected_result, -2 * expected_result]))

    def testSequenceFormat(self):
        if False:
            print('Hello World!')
        self._testWithDiagonalLists(diags=[[2, 1, 4], [1, 3, 2, 2], [1, -1, 1]], rhs=[1, 2, 3, 4], expected=[-9, 5, -4, 4], diags_format='sequence')

    def testSequenceFormatWithDummyElements(self):
        if False:
            print('Hello World!')
        dummy = 20
        self._testWithDiagonalLists(diags=[[2, 1, 4, dummy], [1, 3, 2, 2], [dummy, 1, -1, 1]], rhs=[1, 2, 3, 4], expected=[-9, 5, -4, 4], diags_format='sequence')

    def testSequenceFormatWithBatching(self):
        if False:
            i = 10
            return i + 15
        self._testWithDiagonalLists(diags=[[[2, 1, 4], [-2, -1, -4]], [[1, 3, 2, 2], [-1, -3, -2, -2]], [[1, -1, 1], [-1, 1, -1]]], rhs=[[1, 2, 3, 4], [1, 2, 3, 4]], expected=[[-9, 5, -4, 4], [9, -5, 4, -4]], diags_format='sequence')

    def testMatrixFormat(self):
        if False:
            print('Hello World!')
        self._test(diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]], rhs=[1, 2, 3, 4], expected=[-9, 5, -4, 4], diags_format='matrix')

    def testMatrixFormatWithMultipleRightHandSides(self):
        if False:
            for i in range(10):
                print('nop')
        self._test(diags=[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]], rhs=[[1, -1], [2, -2], [3, -3], [4, -4]], expected=[[-9, 9], [5, -5], [-4, 4], [4, -4]], diags_format='matrix')

    def testMatrixFormatWithBatching(self):
        if False:
            i = 10
            return i + 15
        self._test(diags=[[[1, 2, 0, 0], [1, 3, 1, 0], [0, -1, 2, 4], [0, 0, 1, 2]], [[-1, -2, 0, 0], [-1, -3, -1, 0], [0, 1, -2, -4], [0, 0, -1, -2]]], rhs=[[1, 2, 3, 4], [1, 2, 3, 4]], expected=[[-9, 5, -4, 4], [9, -5, 4, -4]], diags_format='matrix')

    def testRightHandSideAsColumn(self):
        if False:
            while True:
                i = 10
        self._test(diags=_sample_diags, rhs=np.transpose([_sample_rhs]), expected=np.transpose([_sample_result]), diags_format='compact')

    def testTransposeRhs(self):
        if False:
            while True:
                i = 10
        self._test(diags=_sample_diags, rhs=np.array([_sample_rhs, 2 * _sample_rhs]), expected=np.array([_sample_result, 2 * _sample_result]).T, transpose_rhs=True)

    def testTransposeRhsWithRhsAsVector(self):
        if False:
            print('Hello World!')
        self._test(diags=_sample_diags, rhs=_sample_rhs, expected=_sample_result, transpose_rhs=True)

    def testTransposeRhsWithRhsAsVectorAndBatching(self):
        if False:
            i = 10
            return i + 15
        self._test(diags=np.array([_sample_diags, -_sample_diags]), rhs=np.array([_sample_rhs, 2 * _sample_rhs]), expected=np.array([_sample_result, -2 * _sample_result]), transpose_rhs=True)

    def _gradientTest(self, diags, rhs, y, expected_grad_diags, expected_grad_rhs, diags_format='compact', transpose_rhs=False, feed_dict=None):
        if False:
            while True:
                i = 10
        expected_grad_diags = np.array(expected_grad_diags).astype(np.float32)
        expected_grad_rhs = np.array(expected_grad_rhs).astype(np.float32)
        with self.session() as sess, self.test_scope():
            diags = _tfconst(diags)
            rhs = _tfconst(rhs)
            y = _tfconst(y)
            x = linalg_impl.tridiagonal_solve(diags, rhs, diagonals_format=diags_format, transpose_rhs=transpose_rhs, conjugate_rhs=False, partial_pivoting=False)
            res = math_ops.reduce_sum(x * y)
            actual_grad_diags = sess.run(gradient_ops.gradients(res, diags)[0], feed_dict=feed_dict)
            actual_rhs_diags = sess.run(gradient_ops.gradients(res, rhs)[0], feed_dict=feed_dict)
        self.assertAllClose(expected_grad_diags, actual_grad_diags)
        self.assertAllClose(expected_grad_rhs, actual_rhs_diags)

    def testGradientSimple(self):
        if False:
            for i in range(10):
                print('nop')
        self._gradientTest(diags=_sample_diags, rhs=_sample_rhs, y=[1, 3, 2, 4], expected_grad_diags=[[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]], expected_grad_rhs=[1, 0, -1, 4])

    def testGradientWithMultipleRhs(self):
        if False:
            for i in range(10):
                print('nop')
        self._gradientTest(diags=_sample_diags, rhs=[[1, 2], [2, 4], [3, 6], [4, 8]], y=[[1, 5], [2, 6], [3, 7], [4, 8]], expected_grad_diags=[[-20, 28, -60, 0], [36, -35, 60, 80], [0, 63, -75, -80]], expected_grad_rhs=[[0, 2], [1, 3], [1, 7], [0, -10]])

    def _makeDataForGradientWithBatching(self):
        if False:
            while True:
                i = 10
        y = np.array([1, 3, 2, 4]).astype(np.float32)
        grad_diags = np.array([[-5, 0, 4, 0], [9, 0, -4, -16], [0, 0, 5, 16]]).astype(np.float32)
        grad_rhs = np.array([1, 0, -1, 4]).astype(np.float32)
        diags_batched = np.array([[_sample_diags, 2 * _sample_diags, 3 * _sample_diags], [4 * _sample_diags, 5 * _sample_diags, 6 * _sample_diags]]).astype(np.float32)
        rhs_batched = np.array([[_sample_rhs, -_sample_rhs, _sample_rhs], [-_sample_rhs, _sample_rhs, -_sample_rhs]]).astype(np.float32)
        y_batched = np.array([[y, y, y], [y, y, y]]).astype(np.float32)
        expected_grad_diags_batched = np.array([[grad_diags, -grad_diags / 4, grad_diags / 9], [-grad_diags / 16, grad_diags / 25, -grad_diags / 36]]).astype(np.float32)
        expected_grad_rhs_batched = np.array([[grad_rhs, grad_rhs / 2, grad_rhs / 3], [grad_rhs / 4, grad_rhs / 5, grad_rhs / 6]]).astype(np.float32)
        return (y_batched, diags_batched, rhs_batched, expected_grad_diags_batched, expected_grad_rhs_batched)

    def testGradientWithBatchDims(self):
        if False:
            print('Hello World!')
        (y, diags, rhs, expected_grad_diags, expected_grad_rhs) = self._makeDataForGradientWithBatching()
        self._gradientTest(diags=diags, rhs=rhs, y=y, expected_grad_diags=expected_grad_diags, expected_grad_rhs=expected_grad_rhs)

    def _assertRaises(self, diags, rhs, diags_format='compact'):
        if False:
            return 10
        with self.assertRaises(ValueError):
            linalg_impl.tridiagonal_solve(diags, rhs, diags_format)

    def testInvalidShapesCompactFormat(self):
        if False:
            for i in range(10):
                print('nop')

        def test_raises(diags_shape, rhs_shape):
            if False:
                print('Hello World!')
            self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), 'compact')
        test_raises((5, 4, 4), (5, 4))
        test_raises((5, 3, 4), (4, 5))
        test_raises((5, 3, 4), 5)
        test_raises(5, (5, 4))

    def testInvalidShapesSequenceFormat(self):
        if False:
            return 10

        def test_raises(diags_tuple_shapes, rhs_shape):
            if False:
                i = 10
                return i + 15
            diagonals = tuple((_tf_ones(shape) for shape in diags_tuple_shapes))
            self._assertRaises(diagonals, _tf_ones(rhs_shape), 'sequence')
        test_raises(((5, 4), (5, 4)), (5, 4))
        test_raises(((5, 4), (5, 4), (5, 6)), (5, 4))
        test_raises(((5, 3), (5, 4), (5, 6)), (5, 4))
        test_raises(((5, 6), (5, 4), (5, 3)), (5, 4))
        test_raises(((5, 4), (7, 4), (5, 4)), (5, 4))
        test_raises(((5, 4), (7, 4), (5, 4)), (3, 4))

    def testInvalidShapesMatrixFormat(self):
        if False:
            i = 10
            return i + 15

        def test_raises(diags_shape, rhs_shape):
            if False:
                return 10
            self._assertRaises(_tf_ones(diags_shape), _tf_ones(rhs_shape), 'matrix')
        test_raises((5, 4, 7), (5, 4))
        test_raises((5, 4, 4), (3, 4))
        test_raises((5, 4, 4), (5, 3))
if __name__ == '__main__':
    test.main()