"""Tests for Sobol sequence generation."""
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
qmc = tff.math.qmc

@test_util.run_all_in_graph_and_eager_modes
class SobolTest(tf.test.TestCase):

    def test_normal_integral_mean_and_var_correctly_estimated(self):
        if False:
            print('Hello World!')
        n = int(1000)
        dtype = tf.float64
        mu_p = tf.constant([-1.0, 1.0], dtype=dtype)
        mu_q = tf.constant([0.0, 0.0], dtype=dtype)
        sigma_p = tf.constant([0.5, 0.5], dtype=dtype)
        sigma_q = tf.constant([1.0, 1.0], dtype=dtype)
        p = tfp.distributions.Normal(loc=mu_p, scale=sigma_p)
        q = tfp.distributions.Normal(loc=mu_q, scale=sigma_q)
        cdf_sample = qmc.sobol_sample(2, n + 1, sequence_indices=tf.range(1, n + 1), dtype=dtype)
        q_sample = q.quantile(cdf_sample)
        e_x = tf.reduce_mean(q_sample * p.prob(q_sample) / q.prob(q_sample), 0)
        e_x2 = tf.reduce_mean(q_sample ** 2 * p.prob(q_sample) / q.prob(q_sample) - e_x ** 2, 0)
        stddev = tf.sqrt(e_x2)
        with self.subTest('Shape'):
            self.assertEqual(p.batch_shape, e_x.shape)
        with self.subTest('Mean'):
            self.assertAllClose(self.evaluate(p.mean()), self.evaluate(e_x), rtol=0.01)
        with self.subTest('Variance'):
            self.assertAllClose(self.evaluate(p.stddev()), self.evaluate(stddev), rtol=0.02)

    def test_sobol_sample(self):
        if False:
            i = 10
            return i + 15
        expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.75, 0.75, 0.25], [0.75, 0.25, 0.25, 0.25, 0.75], [0.125, 0.625, 0.375, 0.125, 0.125], [0.625, 0.125, 0.875, 0.625, 0.625], [0.375, 0.375, 0.625, 0.875, 0.375], [0.875, 0.875, 0.125, 0.375, 0.875], [0.0625, 0.9375, 0.5625, 0.3125, 0.6875], [0.5625, 0.4375, 0.0625, 0.8125, 0.1875], [0.3125, 0.1875, 0.3125, 0.5625, 0.9375], [0.8125, 0.6875, 0.8125, 0.0625, 0.4375], [0.1875, 0.3125, 0.9375, 0.4375, 0.5625], [0.6875, 0.8125, 0.4375, 0.9375, 0.0625], [0.4375, 0.5625, 0.1875, 0.6875, 0.8125], [0.9375, 0.0625, 0.6875, 0.1875, 0.3125], [0.03125, 0.53125, 0.90625, 0.96875, 0.96875], [0.53125, 0.03125, 0.40625, 0.46875, 0.46875], [0.28125, 0.28125, 0.15625, 0.21875, 0.71875], [0.78125, 0.78125, 0.65625, 0.71875, 0.21875], [0.15625, 0.15625, 0.53125, 0.84375, 0.84375], [0.65625, 0.65625, 0.03125, 0.34375, 0.34375], [0.40625, 0.90625, 0.28125, 0.09375, 0.59375], [0.90625, 0.40625, 0.78125, 0.59375, 0.09375], [0.09375, 0.46875, 0.46875, 0.65625, 0.28125], [0.59375, 0.96875, 0.96875, 0.15625, 0.78125], [0.34375, 0.71875, 0.71875, 0.40625, 0.03125], [0.84375, 0.21875, 0.21875, 0.90625, 0.53125], [0.21875, 0.84375, 0.09375, 0.53125, 0.40625]], dtype=tf.float32)
        actual = qmc.sobol_sample(5, 29, validate_args=True)
        self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_sobol_sample_with_sequence_indices(self):
        if False:
            while True:
                i = 10
        indices = [1, 3, 10, 15, 19, 24, 28]
        expected = tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5], [0.75, 0.25, 0.25, 0.25, 0.75], [0.3125, 0.1875, 0.3125, 0.5625, 0.9375], [0.9375, 0.0625, 0.6875, 0.1875, 0.3125], [0.78125, 0.78125, 0.65625, 0.71875, 0.21875], [0.09375, 0.46875, 0.46875, 0.65625, 0.28125], [0.21875, 0.84375, 0.09375, 0.53125, 0.40625]], dtype=tf.float32)
        actual = qmc.sobol_sample(5, 29, sequence_indices=tf.constant(indices, dtype=tf.int64), validate_args=True)
        self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_sobol_sample_with_tent_transform(self):
        if False:
            return 10
        expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.75, 0.25, 0.25, 0.75], [0.75, 0.25, 0.25, 0.75, 0.75, 0.25], [0.75, 0.75, 0.75, 0.25, 0.75, 0.25], [0.25, 0.25, 0.25, 0.75, 0.25, 0.75]], dtype=tf.float32)
        actual = qmc.sobol_sample(6, 8, apply_tent_transform=True, validate_args=True)
        self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_sobol_sample_with_dtype(self):
        if False:
            return 10
        for dtype in [tf.float32, tf.float64]:
            expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.75, 0.75, 0.25, 0.25], [0.75, 0.25, 0.25, 0.25, 0.75, 0.75], [0.125, 0.625, 0.375, 0.125, 0.125, 0.375], [0.625, 0.125, 0.875, 0.625, 0.625, 0.875], [0.375, 0.375, 0.625, 0.875, 0.375, 0.125], [0.875, 0.875, 0.125, 0.375, 0.875, 0.625]], dtype=dtype)
            actual = qmc.sobol_sample(6, 8, validate_args=True, dtype=dtype)
            self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            self.assertEqual(actual.dtype, expected.dtype)

    def test_sobol_generating_matrices(self):
        if False:
            while True:
                i = 10
        dim = 5
        num_results = 31
        num_digits = 5
        expected = tf.constant([[16, 8, 4, 2, 1], [16, 24, 20, 30, 17], [16, 24, 12, 18, 29], [16, 24, 4, 10, 31], [16, 8, 4, 22, 31]], dtype=tf.int32)
        actual = qmc.sobol_generating_matrices(dim, num_results, num_digits, validate_args=True)
        self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
        self.assertEqual(actual.dtype, expected.dtype)

    def test_sobol_generating_matrices_with_dtype(self):
        if False:
            i = 10
            return i + 15
        dim = 5
        num_results = 31
        num_digits = 5
        for dtype in [tf.int32, tf.int64]:
            expected = tf.constant([[16, 8, 4, 2, 1], [16, 24, 20, 30, 17], [16, 24, 12, 18, 29], [16, 24, 4, 10, 31], [16, 8, 4, 22, 31]], dtype=dtype)
            actual = qmc.sobol_generating_matrices(dim, num_results, num_digits, validate_args=True, dtype=dtype)
            self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))
            self.assertEqual(actual.dtype, dtype)
if __name__ == '__main__':
    tf.test.main()