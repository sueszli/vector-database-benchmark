"""Tests for sparse_ops.sparse_tensor_dense_matmul."""
import sys
import time
from absl import app
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test

def _maybe_complex(x):
    if False:
        for i in range(10):
            print('nop')
    if x.dtype.kind == 'c':
        return x + 1j * x
    return x

class SparseTensorDenseMatMulTest(test.TestCase):

    def _testMatmul(self, x, y, adjoint_a=False, adjoint_b=False, indices_dtype=np.int64):
        if False:
            for i in range(10):
                print('nop')
        x_mat = np.array(x)
        if adjoint_a:
            x_mat = x_mat.T.conj()
        y_mat = np.array(y)
        if adjoint_b:
            y_mat = y_mat.T.conj()
        np_ans = x_mat.dot(y_mat)
        x_indices = np.vstack(np.where(x)).astype(indices_dtype).T
        x_values = x[np.where(x)]
        x_shape = x.shape
        with self.cached_session():
            sp_x_value = sparse_tensor.SparseTensorValue(indices=x_indices, values=x_values, dense_shape=x_shape)
            tf_value_ans = sparse_ops.sparse_tensor_dense_matmul(sp_x_value, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
            tf_tensor_ans = sparse_ops.sparse_tensor_dense_matmul(sparse_tensor.SparseTensor.from_value(sp_x_value), y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
            self.assertEqual(tf_value_ans.get_shape()[1], np_ans.shape[1])
            self.assertEqual(tf_tensor_ans.get_shape()[1], np_ans.shape[1])
            for out in (self.evaluate(tf_value_ans), self.evaluate(tf_tensor_ans)):
                if x.dtype == np.float32:
                    self.assertAllClose(np_ans, out, rtol=0.0001, atol=0.0001)
                elif x.dtype == np.float64:
                    self.assertAllClose(np_ans, out, rtol=1e-06, atol=1e-06)
                elif x.dtype == np.float16:
                    self.assertAllClose(np_ans, out, rtol=0.001, atol=0.001)
                else:
                    self.assertAllClose(np_ans, out, rtol=0.001, atol=0.001)

    def _testBasic(self, value_dtype, indices_dtype=np.int64):
        if False:
            print('Hello World!')
        x = _maybe_complex(np.random.rand(10, 10).astype(value_dtype))
        x[np.abs(x) < 0.5] = 0
        y = _maybe_complex(np.random.randn(10, 20).astype(value_dtype))
        self._testMatmul(x, y, indices_dtype=indices_dtype)

    def testBasic(self):
        if False:
            return 10
        np.random.seed(127)
        self._testBasic(np.int32)
        self._testBasic(np.float16)
        self._testBasic(np.float32)
        self._testBasic(np.float64)
        self._testBasic(np.complex64)
        self._testBasic(np.complex128)
        self._testBasic(np.int32, indices_dtype=np.int32)
        self._testBasic(np.float32, indices_dtype=np.int32)

    def testShapeInference(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.rand(10, 10)
        x[np.abs(x) < 0.5] = 0
        y = np.random.randn(10, 20)
        x_indices = np.vstack(np.where(x)).astype(np.int64).T
        x_values = x[np.where(x)]
        x_shape = x.shape
        with ops.Graph().as_default():
            x_st = sparse_tensor.SparseTensor(x_indices, x_values, x_shape)
            result = sparse_ops.sparse_tensor_dense_matmul(x_st, y)
            self.assertEqual(result.get_shape(), (10, 20))
            x_shape_unknown = array_ops.placeholder(dtype=dtypes.int64, shape=None)
            x_st_shape_unknown = sparse_tensor.SparseTensor(x_indices, x_values, x_shape_unknown)
            result_left_shape_unknown = sparse_ops.sparse_tensor_dense_matmul(x_st_shape_unknown, y)
            self.assertEqual(result_left_shape_unknown.get_shape().as_list(), [None, 20])
            x_shape_inconsistent = [10, 15]
            x_st_shape_inconsistent = sparse_tensor.SparseTensor(x_indices, x_values, x_shape_inconsistent)
            with self.assertRaisesRegex(ValueError, 'Dimensions must be equal'):
                sparse_ops.sparse_tensor_dense_matmul(x_st_shape_inconsistent, y)

    @test_util.run_in_graph_and_eager_modes(use_gpu=False)
    def testInvalidIndicesForSparseTensorDenseMatmul(self):
        if False:
            while True:
                i = 10
        indices = np.array([[1, 10]]).astype(np.int64)
        values = np.array([10]).astype(np.float32)
        shape = [3, 2]
        sparse_t = sparse_tensor.SparseTensor(indices, values, shape)
        dense_t = np.array([[1] * 5, [2] * 5], dtype=np.float32)
        with self.assertRaisesOpError('k .10. from index.0,1. out of bounds .>=2.'):
            self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
        dense_t = np.array([[1] * 500, [2] * 500], dtype=np.float32)
        with self.assertRaisesOpError('k .10. from index.0,1. out of bounds .>=2.'):
            self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
        dense_t = np.array([[1] * 5, [2] * 5, [3] * 5], dtype=np.float32)
        with self.assertRaisesOpError('m .10. from index.0,1. out of bounds .>=2.'):
            self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t, adjoint_a=True))
        dense_t = np.array([[1] * 500, [2] * 500, [3] * 500], dtype=np.float32)
        with self.assertRaisesOpError('m .10. from index.0,1. out of bounds .>=2.'):
            self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t, adjoint_a=True))

    def testUnorderedIndicesForSparseTensorDenseMatmul(self):
        if False:
            print('Hello World!')
        indices = np.array([(2, 1), (0, 0)]).astype(np.int64)
        values = np.array([10, 11]).astype(np.float32)
        shape = [3, 2]
        sparse_t = sparse_tensor.SparseTensor(indices, values, shape)
        dense_t = np.array([[1] * 500, [2] * 500], dtype=np.float32)
        expected_t = np.array([[11] * 500, [0] * 500, [20] * 500], dtype=np.float32)
        self.assertAllClose(expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))

    @test_util.run_gpu_only
    def testInvalidIndicesForSparseTensorDenseMatmulOnGPU(self):
        if False:
            i = 10
            return i + 15
        indices = np.array([[1, 10]]).astype(np.int64)
        values = np.array([10]).astype(np.float32)
        shape = [3, 2]
        sparse_t = sparse_tensor.SparseTensor(indices, values, shape)
        dense_t = np.array([[1] * 5, [2] * 5], dtype=np.float32)
        expected_t = np.array([[0] * 5, [np.nan] * 5, [0] * 5], dtype=np.float32)
        self.assertAllClose(expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
        dense_t = np.array([[1] * 500, [2] * 500], dtype=np.float32)
        expected_t = np.array([[0] * 500, [np.nan] * 500, [0] * 500], dtype=np.float32)
        self.assertAllClose(expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t))
        dense_t = np.array([[1] * 5, [2] * 5, [3] * 5], dtype=np.float32)
        expected_t = np.array([[0] * 5, [0] * 5], dtype=np.float32)
        self.assertAllClose(expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t, adjoint_a=True))
        dense_t = np.array([[1] * 500, [2] * 500, [3] * 500], dtype=np.float32)
        expected_t = np.array([[0] * 500, [0] * 500], dtype=np.float32)
        self.assertAllClose(expected_t, sparse_ops.sparse_tensor_dense_matmul(sparse_t, dense_t, adjoint_a=True))

    def _testLarge(self, np_dtype):
        if False:
            print('Hello World!')
        r1 = np.random.randint(6000, 20000)
        r2 = np.random.randint(1, 10)
        r3 = np.random.randint(1, 10)
        for (m, k, n) in [(r1, r2, r3), (r2, r1, r3), (r2, r3, r1)]:
            x = _maybe_complex(np.random.rand(m, k).astype(np_dtype))
            x[np.abs(x) < 0.8] = 0
            y = _maybe_complex(np.random.randn(k, n).astype(np_dtype))
            self._testMatmul(x, y, adjoint_a=False, adjoint_b=False)
            self._testMatmul(x.transpose(), y, adjoint_a=True, adjoint_b=False)
            self._testMatmul(x, y.transpose(), adjoint_a=False, adjoint_b=True)
            self._testMatmul(x.transpose(), y.transpose(), adjoint_a=True, adjoint_b=True)

    def testLarge(self):
        if False:
            return 10
        np.random.seed(127)
        self._testLarge(np.float32)
        self._testLarge(np.float64)
        self._testLarge(np.complex64)
        self._testLarge(np.complex128)

    def testFloatRandom(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(127)
        for _ in range(8):
            for adjoint_a in [True, False]:
                for adjoint_b in [True, False]:
                    for thresh in [0.0, 0.2, 0.8, 1.0]:
                        (n, k, m) = np.random.randint(1, 100, size=3)
                        x = np.random.rand(n, k).astype(np.float32)
                        x[x < thresh] = 0
                        y = np.random.randn(k, m).astype(np.float32)
                        x = x.transpose() if adjoint_a else x
                        y = y.transpose() if adjoint_b else y
                        self._testMatmul(x, y, adjoint_a, adjoint_b)

def _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(x, y, adjoint_a, adjoint_b):
    if False:
        i = 10
        return i + 15

    def body(t, prev):
        if False:
            return 10
        with ops.control_dependencies([prev]):
            return (t + 1, math_ops.matmul(x, y, transpose_a=adjoint_a, transpose_b=adjoint_b, a_is_sparse=True, b_is_sparse=False))
    t0 = constant_op.constant(0)
    v0 = constant_op.constant(0.0)

    def _timeit(iterations, _):
        if False:
            i = 10
            return i + 15
        (_, final) = while_loop.while_loop(lambda t, _: t < iterations, body, (t0, v0), parallel_iterations=1, back_prop=False, shape_invariants=(tensor_shape.TensorShape(()), tensor_shape.TensorShape(None)))
        return [final]
    return _timeit

def _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(x_ind, x_val, x_shape, y, adjoint_a, adjoint_b):
    if False:
        print('Hello World!')
    sp_x = sparse_tensor.SparseTensor(indices=x_ind, values=x_val, dense_shape=x_shape)

    def body(t, prev):
        if False:
            return 10
        with ops.control_dependencies([prev]):
            return (t + 1, sparse_ops.sparse_tensor_dense_matmul(sp_x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b))
    t0 = constant_op.constant(0)
    v0 = constant_op.constant(0.0)

    def _timeit(iterations, _):
        if False:
            print('Hello World!')
        (_, final) = while_loop.while_loop(lambda t, _: t < iterations, body, (t0, v0), parallel_iterations=1, back_prop=False, shape_invariants=(tensor_shape.TensorShape(()), tensor_shape.TensorShape(None)))
        return [final]
    return _timeit

def sparse_tensor_dense_vs_dense_matmul_benchmark(thresh, m, k, n, adjoint_a, adjoint_b, use_gpu, skip_dense=False):
    if False:
        for i in range(10):
            print('nop')
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    np.random.seed([6, 117])
    x = np.random.rand(m, k).astype(np.float32)
    x[x < thresh] = 0
    y = np.random.randn(k, n).astype(np.float32)
    if adjoint_a:
        x = x.T
    if adjoint_b:
        y = y.T

    def _timer(sess, ops_fn, iterations):
        if False:
            for i in range(10):
                print('nop')
        sess.run(ops_fn(10, sess))
        start = time.time()
        sess.run(ops_fn(iterations, sess))
        end = time.time()
        return (end - start) / (1.0 * iterations)
    if skip_dense:
        delta_dense = float('nan')
    else:
        with session.Session(config=config, graph=ops.Graph()) as sess:
            if not use_gpu:
                with ops.device('/cpu:0'):
                    x_t = constant_op.constant(x)
                    y_t = constant_op.constant(y)
                    ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(x_t, y_t, adjoint_a, adjoint_b)
            else:
                with ops.device('/device:GPU:0'):
                    x_t = constant_op.constant(x)
                    y_t = constant_op.constant(y)
                    ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_dense(x_t, y_t, adjoint_a, adjoint_b)
            delta_dense = _timer(sess, ops_fn, 200)
    with session.Session('', config=config, graph=ops.Graph()) as sess:
        if not use_gpu:
            with ops.device('/cpu:0'):
                x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
                x_val = constant_op.constant(x[np.where(x)])
                x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
                y_t = constant_op.constant(y)
                ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
        else:
            with ops.device('/device:GPU:0'):
                x_ind = constant_op.constant(np.vstack(np.where(x)).astype(np.int64).T)
                x_val = constant_op.constant(x[np.where(x)])
                x_shape = constant_op.constant(np.array(x.shape).astype(np.int64))
                y_t = constant_op.constant(y)
                ops_fn = _sparse_tensor_dense_vs_dense_matmul_benchmark_sparse(x_ind, x_val, x_shape, y_t, adjoint_a, adjoint_b)
        delta_sparse = _timer(sess, ops_fn, 200)
    print('%g \t %d \t %s \t %d \t %d \t %g \t %g \t %g' % (1 - thresh, n, use_gpu, m, k, delta_dense, delta_sparse, delta_sparse / delta_dense))

def main(_):
    if False:
        for i in range(10):
            print('nop')
    print('DenseDense MatMul (w/ Sparse Flag) vs. SparseTensorDense MatMul')
    print('Matrix sizes:')
    print('  A sparse [m, k] with % nonzero values between 1% and 80%')
    print('  B dense [k, n]')
    print('')
    print('% nnz \t n \t gpu \t m \t k \t dt(dense) \t dt(sparse) \t dt(sparse)/dt(dense)')
    for thresh in (0.99, 0.8, 0.5, 0.2):
        for n in (50, 100):
            for use_gpu in (True, False):
                for m in (100, 1000):
                    for k in (100, 1000):
                        sparse_tensor_dense_vs_dense_matmul_benchmark(thresh, m, k, n, False, False, use_gpu=use_gpu)
if __name__ == '__main__':
    if '--benchmarks' in sys.argv:
        sys.argv.remove('--benchmarks')
        app.run()
    else:
        test.main()