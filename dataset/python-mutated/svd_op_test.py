"""Tests for tensorflow.ops.svd."""
import itertools
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test

class SvdOpTest(xla_test.XLATestCase, parameterized.TestCase):

    def _compute_usvt(self, s, u, v):
        if False:
            return 10
        m = u.shape[-1]
        n = v.shape[-1]
        if m <= n:
            v = v[..., :m]
        else:
            u = u[..., :n]
        return np.matmul(u * s[..., None, :], np.swapaxes(v, -1, -2))

    def _testSvdCorrectness(self, dtype, shape):
        if False:
            return 10
        np.random.seed(1)
        x_np = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
        (m, n) = (shape[-2], shape[-1])
        (_, s_np, _) = np.linalg.svd(x_np)
        with self.session() as sess:
            x_tf = array_ops.placeholder(dtype)
            with self.test_scope():
                (s, u, v) = linalg_ops.svd(x_tf, full_matrices=True)
            (s_val, u_val, v_val) = sess.run([s, u, v], feed_dict={x_tf: x_np})
            u_diff = np.matmul(u_val, np.swapaxes(u_val, -1, -2)) - np.eye(m)
            v_diff = np.matmul(v_val, np.swapaxes(v_val, -1, -2)) - np.eye(n)
            self.assertLess(np.linalg.norm(u_diff), 0.01)
            self.assertLess(np.linalg.norm(v_diff), 0.01)
            self.assertLess(np.linalg.norm(s_val - s_np), 0.01)
            self.assertLess(np.linalg.norm(self._compute_usvt(s_val, u_val, v_val) - x_np), 0.02)
            with self.test_scope():
                (no_uv_s, no_uv_u, no_uv_v) = gen_linalg_ops.svd(x_tf, full_matrices=True, compute_uv=False)
            (no_uv_s_val, no_uv_u_val, no_uv_v_val) = sess.run([no_uv_s, no_uv_u, no_uv_v], feed_dict={x_tf: x_np})
            self.assertAllClose(no_uv_s_val, s_val, atol=0.0001, rtol=0.0001)
            self.assertEqual(no_uv_u_val.shape, tensor_shape.TensorShape([0]))
            self.assertEqual(no_uv_v_val.shape, tensor_shape.TensorShape([0]))
    SIZES = [1, 2, 5, 10, 32, 64]
    DTYPES = [np.float32]
    PARAMS = itertools.product(SIZES, DTYPES)

    @parameterized.parameters(*PARAMS)
    def testSvd(self, n, dtype):
        if False:
            for i in range(10):
                print('nop')
        for batch_dims in [(), (3,)] + [(3, 2)] * (n < 10):
            self._testSvdCorrectness(dtype, batch_dims + (n, n))
            self._testSvdCorrectness(dtype, batch_dims + (2 * n, n))
            self._testSvdCorrectness(dtype, batch_dims + (n, 2 * n))
if __name__ == '__main__':
    test.main()