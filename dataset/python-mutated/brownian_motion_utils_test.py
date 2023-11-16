"""Tests for `sample_paths` of `ItoProcess`."""
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util
from tf_quant_finance.models.legacy import brownian_motion_utils as bm_utils

class _TestClass(object):
    pass

class _TestClass2(object):

    def __call__(self, x):
        if False:
            i = 10
            return i + 15
        return x * x

def _test_fn(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    return x + 2 * y + 4 * z

@test_util.run_all_in_graph_and_eager_modes
class BrownianMotionUtilsTest(tf.test.TestCase):

    def test_is_callable(self):
        if False:
            i = 10
            return i + 15
        arg1 = lambda x: x * x
        self.assertTrue(bm_utils.is_callable(arg1))
        arg2 = _TestClass()
        self.assertFalse(bm_utils.is_callable(arg2))
        arg3 = _TestClass2()
        self.assertTrue(bm_utils.is_callable(arg3))
        self.assertTrue(bm_utils.is_callable(_test_fn))
        self.assertFalse(bm_utils.is_callable(2.0))

    def test_outer_multiply(self):
        if False:
            for i in range(10):
                print('nop')
        scalars = (tf.constant(2.0), tf.constant(3.0))
        vectors = (tf.constant([1.0, 2.0]), tf.constant([3.0, 5.0, 1.0]))
        matrices = (tf.constant([[1.0, 2], [2.0, 3]]), tf.constant([[1.0, 2, 3], [4, 5, 6]]))
        result1 = self.evaluate(bm_utils.outer_multiply(scalars[0], scalars[1]))
        self.assertEqual(result1, 6)
        result2 = self.evaluate(bm_utils.outer_multiply(scalars[0], vectors[1]))
        np.testing.assert_allclose(result2, [6.0, 10.0, 2.0])
        result3 = self.evaluate(bm_utils.outer_multiply(vectors[0], scalars[1]))
        np.testing.assert_allclose(result3, [3.0, 6.0])
        result4 = self.evaluate(bm_utils.outer_multiply(vectors[0], vectors[1]))
        np.testing.assert_allclose(result4, [[3.0, 5.0, 1.0], [6.0, 10.0, 2.0]])
        result5 = self.evaluate(bm_utils.outer_multiply(vectors[1], vectors[0]))
        np.testing.assert_allclose(result5, [[3.0, 6.0], [5.0, 10.0], [1.0, 2.0]])
        result6 = self.evaluate(bm_utils.outer_multiply(vectors[0], matrices[0]))
        np.testing.assert_allclose(result6, [[[1.0, 2], [2, 3]], [[2, 4], [4, 6]]])
        result7 = self.evaluate(bm_utils.outer_multiply(matrices[1], matrices[0]))
        np.testing.assert_allclose(result7, [[[[1.0, 2], [2.0, 3]], [[2.0, 4], [4.0, 6]], [[3.0, 6], [6.0, 9]]], [[[4.0, 8], [8.0, 12]], [[5.0, 10], [10.0, 15]], [[6.0, 12], [12.0, 18]]]])

    def test_construct_drift_default(self):
        if False:
            print('Hello World!')
        dtypes = [tf.float64, tf.float32]
        for dtype in dtypes:
            (drift_fn, total_drift_fn) = bm_utils.construct_drift_data(None, None, 2, dtype)
            times = tf.constant([0.3, 0.9, 1.5], dtype=dtype)
            drift_vals = self.evaluate(drift_fn(times))
            np.testing.assert_array_equal(drift_vals.shape, [3, 2])
            np.testing.assert_allclose(drift_vals, [[0.0, 0], [0, 0], [0, 0]])
            total_vals = self.evaluate(total_drift_fn(times - 0.2, times))
            np.testing.assert_array_equal(total_vals.shape, [3, 2])
            np.testing.assert_allclose(total_vals, [[0.0, 0], [0, 0], [0, 0]])

    def test_construct_drift_constant(self):
        if False:
            for i in range(10):
                print('nop')
        dtypes = [tf.float64, tf.float32]

        def make_total_drift_fn(v, dtype):
            if False:
                print('Hello World!')

            def fn(t1, t2):
                if False:
                    print('Hello World!')
                return bm_utils.outer_multiply(t2 - t1, v * tf.ones([2], dtype=dtype))
            return fn
        for dtype in dtypes:
            drift_const = tf.constant(2.0, dtype=dtype)
            (drift_fn, total_drift_fn) = bm_utils.construct_drift_data(drift_const, None, 2, dtype)
            times = tf.constant([0.3, 0.9, 1.5], dtype=dtype)
            drift_vals = self.evaluate(drift_fn(times))
            np.testing.assert_array_equal(drift_vals.shape, [3, 2])
            np.testing.assert_allclose(drift_vals, [[2.0, 2], [2, 2], [2, 2]])
            total_vals = self.evaluate(total_drift_fn(times - 0.2, times))
            np.testing.assert_array_equal(total_vals.shape, [3, 2])
            np.testing.assert_allclose(total_vals, [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]], atol=1e-07)
            (drift_fn_alt, total_drift_fn_alt) = bm_utils.construct_drift_data(drift_const, make_total_drift_fn(4.0, dtype), 2, dtype)
            drift_vals = self.evaluate(drift_fn_alt(times))
            np.testing.assert_array_equal(drift_vals.shape, [3, 2])
            np.testing.assert_allclose(drift_vals, [[2.0, 2], [2, 2], [2, 2]])
            total_vals_alt = self.evaluate(total_drift_fn_alt(times - 0.2, times))
            np.testing.assert_array_equal(total_vals.shape, [3, 2])
            np.testing.assert_allclose(total_vals_alt, [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]], atol=1e-05)

    def test_construct_drift_callable(self):
        if False:
            while True:
                i = 10
        dtype = tf.float64
        (a, b) = (0.1, -0.8)

        def test_drift_fn(t):
            if False:
                while True:
                    i = 10
            return tf.expand_dims(t * a + b, axis=-1)

        def test_total_drift_fn(t1, t2):
            if False:
                return 10
            res = (t2 ** 2 - t1 ** 2) * a / 2 + (t2 - t1) * b
            return tf.expand_dims(res, axis=-1)
        (drift_fn, total_drift_fn) = bm_utils.construct_drift_data(test_drift_fn, test_total_drift_fn, 1, dtype)
        times = tf.constant([0.0, 1.0, 2.0], dtype=dtype)
        drift_vals = self.evaluate(drift_fn(times))
        np.testing.assert_array_equal(drift_vals.shape, [3, 1])
        np.testing.assert_allclose(drift_vals, [[-0.8], [-0.7], [-0.6]])
        t1 = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
        t2 = tf.constant([1.5, 3.0, 5.0], dtype=dtype)
        total_vals = self.evaluate(total_drift_fn(t1, t2))
        np.testing.assert_array_equal(total_vals.shape, [3, 1])
        np.testing.assert_allclose(total_vals, [[-0.3375], [-0.55], [-0.8]], atol=1e-07)
        (_, total_drift) = bm_utils.construct_drift_data(test_drift_fn, None, 1, dtype)
        self.assertIsNone(total_drift)

    def test_construct_vol_defaults(self):
        if False:
            while True:
                i = 10
        dtype = np.float64
        (vol_fn, _) = bm_utils.construct_vol_data(None, None, 2, dtype)
        times = tf.constant([0.5, 1.0, 2.0, 3.0], dtype=dtype)
        vols = self.evaluate(vol_fn(times))
        np.testing.assert_array_equal(vols.shape, [4, 2, 2])

    def test_construct_vol_covar_and_no_vol(self):
        if False:
            return 10
        dtype = np.float64
        covar_matrix = np.array([[0.15, 0.3], [0.3, 0.6]]).astype(dtype)

        def covar_fn(t1, t2):
            if False:
                for i in range(10):
                    print('nop')
            return bm_utils.outer_multiply(t2 - t1, covar_matrix)
        (vol_fn, covar_fn) = bm_utils.construct_vol_data(None, covar_fn, 2, dtype)
        times = tf.constant([0.5, 1.0, 2.0, 3.0], dtype=dtype)
        vols = self.evaluate(vol_fn(times))
        np.testing.assert_array_equal(vols.shape, [4, 2, 2])
        for i in range(4):
            actual_covar = np.matmul(vols[i], vols[i].transpose())
            np.testing.assert_allclose(actual_covar, covar_matrix)
        times2 = times + 0.31415
        tc = self.evaluate(covar_fn(times, times2))
        np.testing.assert_array_equal(tc.shape, [4, 2, 2])
        for i in range(4):
            np.testing.assert_allclose(tc[i], covar_matrix * 0.31415)

    def test_construct_vol_covar_and_vol_callables(self):
        if False:
            for i in range(10):
                print('nop')
        dtype = np.float64
        vol_matrix = np.array([[1.0, 0.21, -0.33], [0.61, 1.5, 1.77], [-0.3, 1.19, -0.55]]).astype(dtype)
        covar_matrix = np.matmul(vol_matrix, vol_matrix.transpose())
        vol_fn = lambda time: bm_utils.outer_multiply(time, vol_matrix)

        def tc_fn(t1, t2):
            if False:
                print('Hello World!')
            return bm_utils.outer_multiply((t2 ** 2 - t1 ** 2) / 2, covar_matrix)
        times = np.array([[0.12, 0.44], [0.48, 1.698]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(vol_fn, tc_fn, 3, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 2, 3, 3])
        np.testing.assert_allclose(actual_vols, self.evaluate(vol_fn(times)))
        times2 = times + np.array([[0.12, 0.34], [0.56, 0.78]]).astype(dtype)
        actual_tc = self.evaluate(actual_tc_fn(times, times2))
        np.testing.assert_array_equal(actual_tc.shape, [2, 2, 3, 3])
        np.testing.assert_allclose(actual_tc, self.evaluate(actual_tc_fn(times, times2)))

    def test_construct_vol_covar_and_scalar_vol(self):
        if False:
            return 10
        dtype = np.float64
        vol = tf.constant(0.94, dtype=dtype)
        np.random.seed(1235)
        dim = 5
        vol_matrix = np.random.randn(dim, dim)
        covar_matrix = np.matmul(vol_matrix, vol_matrix.transpose())

        def tc_fn(t1, t2):
            if False:
                return 10
            return bm_utils.outer_multiply(t2 - t1, covar_matrix)
        times = np.array([[0.12, 0.44], [0.48, 1.698]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(vol, tc_fn, dim, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 2, dim, dim])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_vols[i, j], np.eye(dim).astype(dtype) * 0.94)
        actual_tc = self.evaluate(actual_tc_fn(times, times + 1.0))
        np.testing.assert_array_equal(actual_tc.shape, [2, 2, dim, dim])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_tc[i, j], covar_matrix)

    def test_construct_vol_covar_and_vector_vol(self):
        if False:
            i = 10
            return i + 15
        dtype = np.float64
        vol = np.array([0.94, 1.1, 0.42], dtype=dtype)
        np.random.seed(5321)
        dim = 3
        vol_matrix = np.random.randn(dim, dim)
        covar_matrix = np.matmul(vol_matrix, vol_matrix.transpose())

        def tc_fn(t1, t2):
            if False:
                while True:
                    i = 10
            return bm_utils.outer_multiply(t2 - t1, covar_matrix)
        times = np.array([[0.12], [0.48]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(tf.constant(vol), tc_fn, dim, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 1, dim, dim])
        for i in range(2):
            np.testing.assert_allclose(actual_vols[i, 0], np.diag(vol))
        actual_tc = self.evaluate(actual_tc_fn(times, times + 0.22))
        np.testing.assert_array_equal(actual_tc.shape, [2, 1, dim, dim])
        for i in range(2):
            np.testing.assert_allclose(actual_tc[i, 0], covar_matrix * 0.22)

    def test_construct_vol_covar_and_vol_matrix(self):
        if False:
            print('Hello World!')
        dtype = np.float64
        vol_matrix = np.array([[1.0, 0.21, -0.33], [0.61, 1.5, 1.77], [-0.3, 1.19, -0.55]]).astype(dtype)
        covar_matrix = np.matmul(vol_matrix, vol_matrix.transpose())

        def tc_fn(t1, t2):
            if False:
                for i in range(10):
                    print('nop')
            return bm_utils.outer_multiply(t2 - t1, covar_matrix)
        times = np.array([[0.12, 0.44], [0.48, 1.698]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(vol_matrix, tc_fn, 3, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 2, 3, 3])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_vols[i, j], vol_matrix)
        times2 = times + np.array([[0.12, 0.34], [0.56, 0.78]]).astype(dtype)
        actual_tc = self.evaluate(actual_tc_fn(times, times2))
        np.testing.assert_array_equal(actual_tc.shape, [2, 2, 3, 3])
        np.testing.assert_allclose(actual_tc, self.evaluate(actual_tc_fn(times, times2)))

    def test_construct_vol_no_covar_vol_callable(self):
        if False:
            return 10
        vol_fn = tf.sin
        (_, total_cov) = bm_utils.construct_vol_data(vol_fn, None, 1, tf.float32)
        self.assertIsNone(total_cov)

    def test_construct_vol_no_covar_and_scalar_vol(self):
        if False:
            return 10
        dtype = np.float64
        vol = tf.constant(0.94, dtype=dtype)
        np.random.seed(1235)
        dim = 5
        covar_matrix = np.eye(dim) * 0.94 * 0.94
        times = np.array([[0.12, 0.44], [0.48, 1.698]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(vol, None, dim, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 2, dim, dim])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_vols[i, j], np.eye(dim).astype(dtype) * 0.94)
        actual_tc = self.evaluate(actual_tc_fn(times, times + 1.0))
        np.testing.assert_array_equal(actual_tc.shape, [2, 2, dim, dim])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_tc[i, j], covar_matrix)

    def test_construct_vol_no_covar_and_vector_vol(self):
        if False:
            return 10
        dtype = np.float64
        vol = np.array([0.94, 1.1, 0.42], dtype=dtype)
        np.random.seed(5321)
        dim = 3
        vol_matrix = np.diag(vol)
        covar_matrix = np.matmul(vol_matrix, vol_matrix.transpose())
        times = np.array([[0.12], [0.48]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(tf.constant(vol), None, dim, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 1, dim, dim])
        for i in range(2):
            np.testing.assert_allclose(actual_vols[i, 0], np.diag(vol))
        actual_tc = self.evaluate(actual_tc_fn(times, times + 0.22))
        np.testing.assert_array_equal(actual_tc.shape, [2, 1, dim, dim])
        for i in range(2):
            np.testing.assert_allclose(actual_tc[i, 0], covar_matrix * 0.22)

    def test_construct_vol_no_covar_and_vol_matrix(self):
        if False:
            print('Hello World!')
        dtype = np.float64
        vol_matrix = np.array([[1.0, 0.21, -0.33], [0.61, 1.5, 1.77], [-0.3, 1.19, -0.55]]).astype(dtype)
        covar_matrix = np.matmul(vol_matrix, vol_matrix.transpose())
        times = np.array([[0.12, 0.44], [0.48, 1.698]]).astype(dtype)
        (actual_vol_fn, actual_tc_fn) = bm_utils.construct_vol_data(vol_matrix, None, 3, dtype)
        actual_vols = self.evaluate(actual_vol_fn(times))
        np.testing.assert_array_equal(actual_vols.shape, [2, 2, 3, 3])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_vols[i, j], vol_matrix)
        dt = np.array([[0.12, 0.34], [0.56, 0.78]]).astype(dtype)
        times2 = times + dt
        actual_tc = self.evaluate(actual_tc_fn(times, times2))
        np.testing.assert_array_equal(actual_tc.shape, [2, 2, 3, 3])
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(actual_tc[i, j], covar_matrix * dt[i, j])
if __name__ == '__main__':
    tf.test.main()