"""Tests for tensorflow.ops.tf.Lu."""
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

@test_util.with_eager_op_as_function
class LuOpTest(test.TestCase):

    @property
    def float_types(self):
        if False:
            print('Hello World!')
        return set((np.float64, np.float32, np.complex64, np.complex128))

    def _verifyLuBase(self, x, lower, upper, perm, verification, output_idx_type):
        if False:
            for i in range(10):
                print('nop')
        (lower_np, upper_np, perm_np, verification_np) = self.evaluate([lower, upper, perm, verification])
        self.assertAllClose(x, verification_np)
        self.assertShapeEqual(x, lower)
        self.assertShapeEqual(x, upper)
        self.assertAllEqual(x.shape[:-1], perm.shape.as_list())
        self.assertEqual(x.dtype, lower_np.dtype)
        self.assertEqual(x.dtype, upper_np.dtype)
        self.assertEqual(output_idx_type.as_numpy_dtype, perm_np.dtype)
        if perm_np.shape[-1] > 0:
            perm_reshaped = np.reshape(perm_np, (-1, perm_np.shape[-1]))
            for perm_vector in perm_reshaped:
                self.assertAllClose(np.arange(len(perm_vector)), np.sort(perm_vector))

    def _verifyLu(self, x, output_idx_type=dtypes.int64):
        if False:
            i = 10
            return i + 15
        (lu, perm) = linalg_ops.lu(x, output_idx_type=output_idx_type)
        lu_shape = np.array(lu.shape.as_list())
        batch_shape = lu_shape[:-2]
        num_rows = lu_shape[-2]
        num_cols = lu_shape[-1]
        lower = array_ops.matrix_band_part(lu, -1, 0)
        if num_rows > num_cols:
            eye = linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=lower.dtype)
            lower = array_ops.concat([lower, eye[..., num_cols:]], axis=-1)
        elif num_rows < num_cols:
            lower = lower[..., :num_rows]
        ones_diag = array_ops.ones(np.append(batch_shape, num_rows), dtype=lower.dtype)
        lower = array_ops.matrix_set_diag(lower, ones_diag)
        upper = array_ops.matrix_band_part(lu, 0, -1)
        verification = test_util.matmul_without_tf32(lower, upper)
        if num_rows > 0:
            perm_reshaped = array_ops.reshape(perm, [-1, num_rows])
            verification_reshaped = array_ops.reshape(verification, [-1, num_rows, num_cols])
            inv_perm_reshaped = map_fn.map_fn(array_ops.invert_permutation, perm_reshaped)
            batch_size = perm_reshaped.shape.as_list()[0]
            batch_indices = math_ops.cast(array_ops.broadcast_to(math_ops.range(batch_size)[:, None], perm_reshaped.shape), dtype=output_idx_type)
            if inv_perm_reshaped.shape == [0]:
                inv_perm_reshaped = array_ops.zeros_like(batch_indices)
            permuted_verification_reshaped = array_ops.gather_nd(verification_reshaped, array_ops_stack.stack([batch_indices, inv_perm_reshaped], axis=-1))
            verification = array_ops.reshape(permuted_verification_reshaped, lu_shape)
        self._verifyLuBase(x, lower, upper, perm, verification, output_idx_type)

    def testBasic(self):
        if False:
            print('Hello World!')
        data = np.array([[4.0, -1.0, 2.0], [-1.0, 6.0, 0], [10.0, 0.0, 5.0]])
        for dtype in (np.float32, np.float64):
            for output_idx_type in (dtypes.int32, dtypes.int64):
                with self.subTest(dtype=dtype, output_idx_type=output_idx_type):
                    self._verifyLu(data.astype(dtype), output_idx_type=output_idx_type)
        for dtype in (np.complex64, np.complex128):
            for output_idx_type in (dtypes.int32, dtypes.int64):
                with self.subTest(dtype=dtype, output_idx_type=output_idx_type):
                    complex_data = np.tril(1j * data, -1).astype(dtype)
                    complex_data += np.triu(-1j * data, 1).astype(dtype)
                    complex_data += data
                    self._verifyLu(complex_data, output_idx_type=output_idx_type)

    def testPivoting(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.array([[1e-09, 1.0, 0.0], [1.0, 0.0, 0], [0.0, 1.0, 5]])
        self._verifyLu(data.astype(np.float32))
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                self._verifyLu(data.astype(dtype))
                (_, p) = linalg_ops.lu(data)
                p_val = self.evaluate([p])
                self.assertNotAllClose(np.arange(3), p_val)
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                complex_data = np.tril(1j * data, -1).astype(dtype)
                complex_data += np.triu(-1j * data, 1).astype(dtype)
                complex_data += data
                self._verifyLu(complex_data)
                (_, p) = linalg_ops.lu(data)
                p_val = self.evaluate([p])
                self.assertNotAllClose(np.arange(3), p_val)

    def testInvalidMatrix(self):
        if False:
            return 10
        for dtype in self.float_types:
            with self.subTest(dtype=dtype):
                with self.assertRaises(errors.InvalidArgumentError):
                    self.evaluate(linalg_ops.lu(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [2.0, 3.0, 4.0]], dtype=dtype)))
                with self.assertRaises(errors.InvalidArgumentError):
                    self.evaluate(linalg_ops.lu(np.array([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]], dtype=dtype)))

    def testBatch(self):
        if False:
            return 10
        simple_array = np.array([[[1.0, -1.0], [2.0, 5.0]]])
        self._verifyLu(simple_array)
        self._verifyLu(np.vstack((simple_array, simple_array)))
        odd_sized_array = np.array([[[4.0, -1.0, 2.0], [-1.0, 6.0, 0], [2.0, 0.0, 5.0]]])
        self._verifyLu(np.vstack((odd_sized_array, odd_sized_array)))
        batch_size = 200
        np.random.seed(42)
        matrices = np.random.rand(batch_size, 5, 5)
        self._verifyLu(matrices)
        np.random.seed(52)
        matrices = np.random.rand(batch_size, 5, 5) + 1j * np.random.rand(batch_size, 5, 5)
        self._verifyLu(matrices)

    def testLargeMatrix(self):
        if False:
            i = 10
            return i + 15
        n = 500
        np.random.seed(64)
        data = np.random.rand(n, n)
        self._verifyLu(data)
        np.random.seed(129)
        data = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        self._verifyLu(data)

    @test_util.disable_xla('b/206106619')
    @test_util.run_in_graph_and_eager_modes(use_gpu=True)
    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        self._verifyLu(np.empty([0, 2, 2]))
        self._verifyLu(np.empty([2, 0, 0]))

    @test_util.run_in_graph_and_eager_modes(use_gpu=True)
    def testConcurrentExecutesWithoutError(self):
        if False:
            while True:
                i = 10
        matrix_shape = [5, 5]
        seed = [42, 24]
        matrix1 = stateless_random_ops.stateless_random_normal(shape=matrix_shape, seed=seed)
        matrix2 = stateless_random_ops.stateless_random_normal(shape=matrix_shape, seed=seed)
        self.assertAllEqual(matrix1, matrix2)
        (lu1, p1) = linalg_ops.lu(matrix1)
        (lu2, p2) = linalg_ops.lu(matrix2)
        (lu1_val, p1_val, lu2_val, p2_val) = self.evaluate([lu1, p1, lu2, p2])
        self.assertAllEqual(lu1_val, lu2_val)
        self.assertAllEqual(p1_val, p2_val)

class LuBenchmark(test.Benchmark):
    shapes = [(4, 4), (10, 10), (16, 16), (101, 101), (256, 256), (1000, 1000), (1024, 1024), (2048, 2048), (4096, 4096), (513, 2, 2), (513, 8, 8), (513, 256, 256), (4, 513, 2, 2)]

    def _GenerateMatrix(self, shape):
        if False:
            print('Hello World!')
        batch_shape = shape[:-2]
        shape = shape[-2:]
        assert shape[0] == shape[1]
        n = shape[0]
        matrix = np.ones(shape).astype(np.float32) / (2.0 * n) + np.diag(np.ones(n).astype(np.float32))
        return np.tile(matrix, batch_shape + (1, 1))

    def benchmarkLuOp(self):
        if False:
            print('Hello World!')
        for shape in self.shapes:
            with ops.Graph().as_default(), session.Session(config=benchmark.benchmark_config()) as sess, ops.device('/cpu:0'):
                matrix = variables.Variable(self._GenerateMatrix(shape))
                (lu, p) = linalg_ops.lu(matrix)
                self.evaluate(variables.global_variables_initializer())
                self.run_op_benchmark(sess, control_flow_ops.group(lu, p), min_iters=25, name='lu_cpu_{shape}'.format(shape=shape))
            if test.is_gpu_available(True):
                with ops.Graph().as_default(), session.Session(config=benchmark.benchmark_config()) as sess, ops.device('/device:GPU:0'):
                    matrix = variables.Variable(self._GenerateMatrix(shape))
                    (lu, p) = linalg_ops.lu(matrix)
                    self.evaluate(variables.global_variables_initializer())
                    self.run_op_benchmark(sess, control_flow_ops.group(lu, p), min_iters=25, name='lu_gpu_{shape}'.format(shape=shape))
if __name__ == '__main__':
    test.main()