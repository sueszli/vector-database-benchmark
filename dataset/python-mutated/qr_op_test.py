"""Tests for tensorflow.ops.math_ops.matrix_inverse."""
import itertools
import unittest
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

@test_util.run_all_without_tensor_float_32('XLA QR op calls matmul. Also, matmul used for verification. Also with TensorFloat-32, mysterious "Unable to launch cuBLAS gemm" error occasionally occurs')
class QrOpTest(xla_test.XLATestCase, parameterized.TestCase):

    def AdjustedNorm(self, x):
        if False:
            return 10
        "Computes the norm of matrices in 'x', adjusted for dimension and type."
        norm = np.linalg.norm(x, axis=(-2, -1))
        return norm / (max(x.shape[-2:]) * np.finfo(x.dtype).eps)

    def CompareOrthogonal(self, x, y, rank):
        if False:
            return 10
        x = x[..., 0:rank]
        y = y[..., 0:rank]
        sum_of_ratios = np.sum(np.divide(y, x), -2, keepdims=True)
        phases = np.divide(sum_of_ratios, np.abs(sum_of_ratios))
        x *= phases
        self.assertTrue(np.all(self.AdjustedNorm(x - y) < 30.0))

    def CheckApproximation(self, a, q, r):
        if False:
            i = 10
            return i + 15
        precision = self.AdjustedNorm(a - np.matmul(q, r))
        self.assertTrue(np.all(precision < 11.0))

    def CheckUnitary(self, x):
        if False:
            return 10
        xx = math_ops.matmul(x, x, adjoint_a=True)
        identity = array_ops.matrix_band_part(array_ops.ones_like(xx), 0, 0)
        tol = 100 * np.finfo(x.dtype).eps
        self.assertAllClose(xx, identity, atol=tol)

    def _random_matrix(self, dtype, shape):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(1)

        def rng():
            if False:
                print('Hello World!')
            return np.random.uniform(low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype)
        x_np = rng()
        if np.issubdtype(dtype, np.complexfloating):
            x_np += rng() * dtype(1j)
        return x_np

    def _test(self, x_np, full_matrices, full_rank=True):
        if False:
            for i in range(10):
                print('nop')
        dtype = x_np.dtype
        shape = x_np.shape
        with self.session() as sess:
            x_tf = array_ops.placeholder(dtype)
            with self.device_scope():
                (q_tf, r_tf) = linalg_ops.qr(x_tf, full_matrices=full_matrices)
            (q_tf_val, r_tf_val) = sess.run([q_tf, r_tf], feed_dict={x_tf: x_np})
            q_dims = q_tf_val.shape
            np_q = np.ndarray(q_dims, dtype)
            np_q_reshape = np.reshape(np_q, (-1, q_dims[-2], q_dims[-1]))
            new_first_dim = np_q_reshape.shape[0]
            x_reshape = np.reshape(x_np, (-1, x_np.shape[-2], x_np.shape[-1]))
            for i in range(new_first_dim):
                if full_matrices:
                    (np_q_reshape[i, :, :], _) = np.linalg.qr(x_reshape[i, :, :], mode='complete')
                else:
                    (np_q_reshape[i, :, :], _) = np.linalg.qr(x_reshape[i, :, :], mode='reduced')
            np_q = np.reshape(np_q_reshape, q_dims)
            if full_rank:
                self.CompareOrthogonal(np_q, q_tf_val, min(shape[-2:]))
            self.CheckApproximation(x_np, q_tf_val, r_tf_val)
            self.CheckUnitary(q_tf_val)
    SIZES = [1, 2, 5, 10, 32, 100, 300, 603]
    DTYPES = [np.float32, np.complex64]
    PARAMS = itertools.product(SIZES, SIZES, DTYPES)

    @parameterized.parameters(*PARAMS)
    def testQR(self, rows, cols, dtype):
        if False:
            while True:
                i = 10
        for full_matrices in [True, False]:
            for batch_dims in [(), (3,)] + [(3, 2)] * (max(rows, cols) < 10):
                x_np = self._random_matrix(dtype, batch_dims + (rows, cols))
                self._test(x_np, full_matrices)

    def testLarge2000x2000(self):
        if False:
            return 10
        x_np = self._random_matrix(np.float32, (2000, 2000))
        self._test(x_np, full_matrices=True)

    @unittest.skip('Test times out on CI')
    def testLarge17500x128(self):
        if False:
            return 10
        x_np = self._random_matrix(np.float32, (17500, 128))
        self._test(x_np, full_matrices=True)

    @parameterized.parameters((23, 25), (513, 23))
    def testZeroColumn(self, rows, cols):
        if False:
            i = 10
            return i + 15
        x_np = self._random_matrix(np.complex64, (rows, cols))
        x_np[:, 7] = 0.0
        self._test(x_np, full_matrices=True)

    @parameterized.parameters((4, 4), (514, 20))
    def testRepeatedColumn(self, rows, cols):
        if False:
            return 10
        x_np = self._random_matrix(np.complex64, (rows, cols))
        x_np[:, 1] = x_np[:, 2]
        self._test(x_np, full_matrices=True, full_rank=False)
if __name__ == '__main__':
    test.main()