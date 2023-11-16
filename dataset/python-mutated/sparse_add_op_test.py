"""Tests for SparseAdd."""
import timeit
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad
from tensorflow.python.platform import test

def _sparsify(x, thresh=0.5, index_dtype=np.int64):
    if False:
        print('Hello World!')
    x[x < thresh] = 0
    non_zero = np.where(x)
    x_indices = np.vstack(non_zero).astype(index_dtype).T
    x_values = x[non_zero]
    x_shape = x.shape
    return (sparse_tensor.SparseTensor(indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values))

class SparseAddTest(test.TestCase):

    def _randomTensor(self, size, np_dtype, sparse=True):
        if False:
            for i in range(10):
                print('nop')
        (n, m) = size
        x = np.random.randn(n, m).astype(np_dtype)
        return _sparsify(x) if sparse else x

    def _SparseTensorValue_3x3(self, negate=False):
        if False:
            print('Hello World!')
        ind = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
        val = np.array([1, 2, 3, 4])
        if negate:
            val = -np.array([1, 2, 3, 4])
        shape = np.array([3, 3])
        return sparse_tensor.SparseTensorValue(np.array(ind, np.int64), np.array(val, np.float32), np.array(shape, np.int64))

    def _SparseTensor_3x3(self, negate=False):
        if False:
            for i in range(10):
                print('nop')
        return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x3(negate))

    def _SparseTensor_3x3_v2(self):
        if False:
            i = 10
            return i + 15
        ind = np.array([[0, 1], [1, 0], [2, 0], [2, 1]])
        val = np.array([1, -1.9, 3, -4.2])
        shape = np.array([3, 3])
        return sparse_tensor.SparseTensor(constant_op.constant(ind, dtypes.int64), constant_op.constant(val, dtypes.float32), constant_op.constant(shape, dtypes.int64))

    def testAddSelf(self):
        if False:
            return 10
        with test_util.force_cpu():
            for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
                for sp_b in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
                    sp_sum = sparse_ops.sparse_add(sp_a, sp_b)
                    self.assertAllEqual((3, 3), sp_sum.get_shape())
                    sum_out = self.evaluate(sp_sum)
                    self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
                    self.assertAllEqual(sum_out.indices, [[0, 1], [1, 0], [2, 0], [2, 1]])
                    self.assertAllEqual(sum_out.values, [2, 4, 6, 8])
                    self.assertAllEqual(sum_out.dense_shape, [3, 3])

    def testAddSelfAndNegation(self):
        if False:
            while True:
                i = 10
        with test_util.force_cpu():
            sp_a = self._SparseTensor_3x3()
            sp_b = self._SparseTensor_3x3(negate=True)
            sp_sum = sparse_ops.sparse_add(sp_a, sp_b, 0.1)
            sum_out = self.evaluate(sp_sum)
            self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
            self.assertAllEqual(sum_out.indices, np.empty([0, 2]))
            self.assertAllEqual(sum_out.values, [])
            self.assertAllEqual(sum_out.dense_shape, [3, 3])

    def testSmallValuesShouldVanish(self):
        if False:
            return 10
        with test_util.force_cpu():
            sp_a = self._SparseTensor_3x3()
            sp_b = self._SparseTensor_3x3_v2()
            sp_sum = sparse_ops.sparse_add(sp_a, sp_b, thresh=0.21)
            sum_out = self.evaluate(sp_sum)
            self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
            self.assertAllEqual(sum_out.indices, [[0, 1], [2, 0]])
            self.assertAllEqual(sum_out.values, [2, 6])
            self.assertAllEqual(sum_out.dense_shape, [3, 3])
            sp_sum = sparse_ops.sparse_add(sp_a, sp_b, thresh=0.11)
            sum_out = self.evaluate(sp_sum)
            self.assertEqual(sp_sum.dense_shape.get_shape(), [2])
            self.assertAllEqual(sum_out.indices, [[0, 1], [2, 0], [2, 1]])
            self.assertAllClose(sum_out.values, [2, 6, -0.2])
            self.assertAllEqual(sum_out.dense_shape, [3, 3])

    @test_util.run_deprecated_v1
    def testGradients(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(1618)
        with self.session(use_gpu=False):
            for n in [10, 31]:
                for m in [4, 17]:
                    (sp_a, nnz_a) = self._randomTensor([n, m], np.float32)
                    (sp_b, nnz_b) = self._randomTensor([n, m], np.float32)
                    sp_sum = sparse_ops.sparse_add(sp_a, sp_b)
                    nnz_sum = len(self.evaluate(sp_sum.values))
                    err = gradient_checker.compute_gradient_error([sp_a.values, sp_b.values], [(nnz_a,), (nnz_b,)], sp_sum.values, (nnz_sum,))
                    self.assertLess(err, 0.001)

    def testAddSparseDense(self):
        if False:
            while True:
                i = 10
        np.random.seed(1618)
        (n, m) = np.random.randint(30, size=2)
        for dtype in [np.float32, np.float64, np.int64, np.complex64]:
            for index_dtype in [np.int32, np.int64]:
                rand_vals_np = np.random.randn(n, m).astype(dtype)
                dense_np = np.random.randn(n, m).astype(dtype)
                with test_util.force_cpu():
                    (sparse, unused_nnz) = _sparsify(rand_vals_np, index_dtype=index_dtype)
                    s = self.evaluate(sparse_ops.sparse_add(sparse, constant_op.constant(dense_np)))
                    self.assertAllEqual(dense_np + rand_vals_np, s)
                    self.assertTrue(s.dtype == dtype)
                    s = self.evaluate(sparse_ops.sparse_add(constant_op.constant(dense_np), sparse))
                    self.assertAllEqual(dense_np + rand_vals_np, s)
                    self.assertTrue(s.dtype == dtype)

    @test_util.run_deprecated_v1
    def testSparseTensorDenseAddGradients(self):
        if False:
            return 10
        np.random.seed(1618)
        (n, m) = np.random.randint(30, size=2)
        rand_vals_np = np.random.randn(n, m).astype(np.float32)
        dense_np = np.random.randn(n, m).astype(np.float32)
        with self.session(use_gpu=False):
            (sparse, nnz) = _sparsify(rand_vals_np)
            dense = constant_op.constant(dense_np, dtype=dtypes.float32)
            s = sparse_ops.sparse_add(sparse, dense)
            err = gradient_checker.compute_gradient_error([sparse.values, dense], [(nnz,), (n, m)], s, (n, m))
            self.assertLess(err, 0.001)

    def testInvalidSparseTensor(self):
        if False:
            for i in range(10):
                print('nop')
        with test_util.force_cpu():
            shape = [2, 2]
            val = [0]
            dense = constant_op.constant(np.zeros(shape, dtype=np.int32))
            for bad_idx in [[[-1, 0]], [[1, 3]]]:
                sparse = sparse_tensor.SparseTensorValue(bad_idx, val, shape)
                with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError), 'invalid index'):
                    s = sparse_ops.sparse_add(sparse, dense)
                    self.evaluate(s)

    def _testSparseDenseInvalidInputs(self, a_indices, a_values, a_shape, b, expected_error=''):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError), expected_error):
            a = sparse_tensor.SparseTensor(a_indices, a_values, a_shape)
            self.evaluate(sparse_ops.sparse_add(a, b))
        with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError), expected_error):
            self.evaluate(sparse_ops.gen_sparse_ops.sparse_tensor_dense_add(a_indices, a_values, a_shape, b))

    def testSparseDenseInvalidInputs(self):
        if False:
            for i in range(10):
                print('nop')
        self._testSparseDenseInvalidInputs(a_indices=constant_op.constant(0, shape=[17, 2], dtype=dtypes.int64), a_values=constant_op.constant(0, shape=[5], dtype=dtypes.float32), a_shape=constant_op.constant([3, 4], dtype=dtypes.int64), b=constant_op.constant(1, shape=[3, 4], dtype=dtypes.float32), expected_error='Dimensions 17 and 5 are not compatible')
        self._testSparseDenseInvalidInputs(a_indices=constant_op.constant(0, shape=[17, 4], dtype=dtypes.int64), a_values=constant_op.constant(0, shape=[17], dtype=dtypes.float32), a_shape=constant_op.constant([3, 4], dtype=dtypes.int64), b=constant_op.constant(1, shape=[3, 4], dtype=dtypes.float32), expected_error='Dimensions 4 and 2 are not compatible')
        self._testSparseDenseInvalidInputs(a_indices=constant_op.constant(7, shape=[17, 2], dtype=dtypes.int64), a_values=constant_op.constant(0, shape=[17], dtype=dtypes.float32), a_shape=constant_op.constant([3, 4], dtype=dtypes.int64), b=constant_op.constant(1, shape=[3, 4], dtype=dtypes.float32), expected_error='invalid index')

def _s2d_add_vs_sparse_add(sparsity, n, m, num_iters=50):
    if False:
        print('Hello World!')
    np.random.seed(1618)
    with session.Session(graph=ops.Graph()) as sess:
        sp_vals = np.random.rand(n, m).astype(np.float32)
        (sp_t, unused_nnz) = _sparsify(sp_vals, thresh=sparsity, index_dtype=np.int32)
        vals = np.random.rand(n, m).astype(np.float32)
        s2d = math_ops.add(sparse_ops.sparse_tensor_to_dense(sp_t), constant_op.constant(vals))
        sa = sparse_ops.sparse_add(sp_t, constant_op.constant(vals))
        timeit.timeit(lambda : sess.run(s2d), number=3)
        timeit.timeit(lambda : sess.run(sa), number=3)
        s2d_total = timeit.timeit(lambda : sess.run(s2d), number=num_iters)
        sa_total = timeit.timeit(lambda : sess.run(sa), number=num_iters)
    return (s2d_total * 1000.0 / num_iters, sa_total * 1000.0 / num_iters)

class SparseAddBenchmark(test.Benchmark):

    def benchmarkSparseAddDense(self):
        if False:
            return 10
        print('SparseAddDense: add with sparse_to_dense vs. sparse_add')
        print('%nnz \t n \t m \t millis(s2d) \t millis(sparse_add) \t speedup')
        for sparsity in [0.99, 0.5, 0.01]:
            for n in [1, 256, 50000]:
                for m in [100, 1000]:
                    (s2d_dt, sa_dt) = _s2d_add_vs_sparse_add(sparsity, n, m)
                    print('%.2f \t %d \t %d \t %.4f \t %.4f \t %.2f' % (sparsity, n, m, s2d_dt, sa_dt, s2d_dt / sa_dt))
if __name__ == '__main__':
    test.main()