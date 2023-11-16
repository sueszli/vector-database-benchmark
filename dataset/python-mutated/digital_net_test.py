"""Tests for digital nets."""
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
qmc = tff.math.qmc

@test_util.run_all_in_graph_and_eager_modes
class DigitalNetTest(tf.test.TestCase):

    def test_random_digital_shift(self):
        if False:
            while True:
                i = 10
        dim = 6
        num_digits = 3
        seed = (2, 3)
        actual = qmc.random_digital_shift(dim, num_digits, seed, validate_args=True)
        power = tf.constant(num_digits)
        minval = qmc.utils.exp2(power - 1)
        maxval = qmc.utils.exp2(power)
        with self.subTest('Shape'):
            self.assertEqual(actual.shape, dim)
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, tf.int32)
        with self.subTest('Max Value'):
            self.assertAllLess(actual, maxval)
        with self.subTest('Min Value'):
            self.assertAllGreaterEqual(actual, minval)

    def test_random_digital_shift_with_dtype(self):
        if False:
            print('Hello World!')
        dim = 6
        num_digits = 3
        seed = (2, 3)
        for dtype in [tf.int32, tf.int64]:
            actual = qmc.random_digital_shift(dim, num_digits, seed, dtype=dtype, validate_args=True)
            power = tf.constant(num_digits, dtype=dtype)
            minval = qmc.utils.exp2(power - 1)
            maxval = qmc.utils.exp2(power)
            with self.subTest('Shape'):
                self.assertEqual(actual.shape, dim)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, dtype)
            with self.subTest('Max Value'):
                self.assertAllLess(actual, maxval)
            with self.subTest('Min Value'):
                self.assertAllGreaterEqual(actual, minval)

    def test_random_scrambling_matrices(self):
        if False:
            i = 10
            return i + 15
        dim = 6
        num_digits = 3
        seed = (2, 3)
        actual = qmc.random_scrambling_matrices(dim, num_digits, seed, validate_args=True)
        power = tf.constant(num_digits)
        minval = qmc.utils.exp2(power - 1)
        maxval = qmc.utils.exp2(power)
        with self.subTest('Shape'):
            self.assertEqual(actual.shape, (dim, num_digits))
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, tf.int32)
        with self.subTest('Max Value'):
            self.assertAllLess(actual, maxval)
        with self.subTest('Min Value'):
            self.assertAllGreaterEqual(actual, minval)

    def test_random_scrambling_matrices_with_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        dim = 6
        num_digits = 3
        seed = (2, 3)
        for dtype in [tf.int32, tf.int64]:
            actual = qmc.random_scrambling_matrices(dim, num_digits, seed, dtype=dtype, validate_args=True)
            power = tf.constant(num_digits, dtype=dtype)
            minval = qmc.utils.exp2(power - 1)
            maxval = qmc.utils.exp2(power)
            with self.subTest('Shape'):
                self.assertEqual(actual.shape, (dim, num_digits))
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, dtype)
            with self.subTest('Max Value'):
                self.assertAllLess(actual, maxval)
            with self.subTest('Min Value'):
                self.assertAllGreaterEqual(actual, minval)

    def test_digital_net_sample(self):
        if False:
            print('Hello World!')
        dim = 5
        num_results = 29
        num_digits = 5
        for dtype in [tf.int32, tf.int64]:
            expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.75, 0.75, 0.25], [0.75, 0.25, 0.25, 0.25, 0.75], [0.125, 0.625, 0.375, 0.125, 0.125], [0.625, 0.125, 0.875, 0.625, 0.625], [0.375, 0.375, 0.625, 0.875, 0.375], [0.875, 0.875, 0.125, 0.375, 0.875], [0.0625, 0.9375, 0.5625, 0.3125, 0.6875], [0.5625, 0.4375, 0.0625, 0.8125, 0.1875], [0.3125, 0.1875, 0.3125, 0.5625, 0.9375], [0.8125, 0.6875, 0.8125, 0.0625, 0.4375], [0.1875, 0.3125, 0.9375, 0.4375, 0.5625], [0.6875, 0.8125, 0.4375, 0.9375, 0.0625], [0.4375, 0.5625, 0.1875, 0.6875, 0.8125], [0.9375, 0.0625, 0.6875, 0.1875, 0.3125], [0.03125, 0.53125, 0.90625, 0.96875, 0.96875], [0.53125, 0.03125, 0.40625, 0.46875, 0.46875], [0.28125, 0.28125, 0.15625, 0.21875, 0.71875], [0.78125, 0.78125, 0.65625, 0.71875, 0.21875], [0.15625, 0.15625, 0.53125, 0.84375, 0.84375], [0.65625, 0.65625, 0.03125, 0.34375, 0.34375], [0.40625, 0.90625, 0.28125, 0.09375, 0.59375], [0.90625, 0.40625, 0.78125, 0.59375, 0.09375], [0.09375, 0.46875, 0.46875, 0.65625, 0.28125], [0.59375, 0.96875, 0.96875, 0.15625, 0.78125], [0.34375, 0.71875, 0.71875, 0.40625, 0.03125], [0.84375, 0.21875, 0.21875, 0.90625, 0.53125], [0.21875, 0.84375, 0.09375, 0.53125, 0.40625]], dtype=tf.float32)
            actual = qmc.digital_net_sample(qmc.sobol_generating_matrices(dim, num_results, num_digits, dtype=dtype), num_results, num_digits, validate_args=True)
            with self.subTest('Values'):
                self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, expected.dtype)

    def test_digital_net_sample_with_sequence_indices(self):
        if False:
            i = 10
            return i + 15
        dim = 5
        num_results = 29
        num_digits = 5
        indices = [1, 3, 10, 15, 19, 24, 28]
        expected = tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5], [0.75, 0.25, 0.25, 0.25, 0.75], [0.3125, 0.1875, 0.3125, 0.5625, 0.9375], [0.9375, 0.0625, 0.6875, 0.1875, 0.3125], [0.78125, 0.78125, 0.65625, 0.71875, 0.21875], [0.09375, 0.46875, 0.46875, 0.65625, 0.28125], [0.21875, 0.84375, 0.09375, 0.53125, 0.40625]], dtype=tf.float32)
        actual = qmc.digital_net_sample(qmc.sobol_generating_matrices(dim, num_results, num_digits), num_results, num_digits, sequence_indices=tf.constant(indices, dtype=tf.int64), validate_args=True)
        with self.subTest('Values'):
            self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, expected.dtype)

    def test_sample_sobol_with_tent_transform(self):
        if False:
            print('Hello World!')
        dim = 6
        num_results = 8
        num_digits = 3
        expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.75, 0.25, 0.25, 0.75], [0.75, 0.25, 0.25, 0.75, 0.75, 0.25], [0.75, 0.75, 0.75, 0.25, 0.75, 0.25], [0.25, 0.25, 0.25, 0.75, 0.25, 0.75]], dtype=tf.float32)
        actual = qmc.digital_net_sample(qmc.sobol_generating_matrices(dim, num_results, num_digits), num_results, num_digits, apply_tent_transform=True, validate_args=True)
        with self.subTest('Values'):
            self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, expected.dtype)

    def test_digital_net_sample_with_dtype(self):
        if False:
            while True:
                i = 10
        dim = 5
        num_results = 6
        num_digits = 3
        generating_matrices = qmc.sobol_generating_matrices(dim, num_results, num_digits)
        for dtype in [tf.float32, tf.float64]:
            expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.75, 0.75, 0.25], [0.75, 0.25, 0.25, 0.25, 0.75], [0.125, 0.625, 0.375, 0.125, 0.125], [0.625, 0.125, 0.875, 0.625, 0.625]], dtype=dtype)
            actual = qmc.digital_net_sample(generating_matrices, num_results, num_digits, validate_args=True, dtype=dtype)
            with self.subTest('Values'):
                self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, expected.dtype)

    def test_scramble_generating_matrices(self):
        if False:
            print('Hello World!')
        dim = 6
        num_results = 8
        num_digits = 3
        seed = (2, 3)
        for dtype in [tf.int32, tf.int64]:
            generating_matrices = qmc.sobol_generating_matrices(dim, num_results, num_digits, dtype=dtype)
            scrambling_matrices = qmc.random_scrambling_matrices(dim, num_digits, seed)
            actual = qmc.scramble_generating_matrices(generating_matrices, scrambling_matrices, num_digits, validate_args=True)
            with self.subTest('Shape'):
                self.assertEqual(actual.shape, generating_matrices.shape)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, dtype)

    def test_scramble_generating_matrices_with_minimum_scrambling_matrices(self):
        if False:
            print('Hello World!')
        dim = 6
        num_results = 8
        num_digits = 3
        for dtype in [tf.int32, tf.int64]:
            generating_matrices = qmc.sobol_generating_matrices(dim, num_results, num_digits, dtype=dtype)
            min_scrambling_matrices = tf.broadcast_to(qmc.utils.exp2(tf.cast(num_digits, dtype) - 1), shape=generating_matrices.shape)
            actual = qmc.scramble_generating_matrices(generating_matrices, min_scrambling_matrices, num_digits, dtype=dtype, validate_args=True)
            with self.subTest('Shape'):
                self.assertEqual(actual.shape, generating_matrices.shape)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, generating_matrices.dtype)
            with self.subTest('Values'):
                self.assertAllEqual(self.evaluate(actual), self.evaluate(generating_matrices))

    def test_scramble_generating_matrices_with_dtype(self):
        if False:
            print('Hello World!')
        dim = 6
        num_results = 8
        num_digits = 3
        seed = (2, 3)
        generating_matrices = qmc.sobol_generating_matrices(dim, num_results, num_digits)
        scrambling_matrices = qmc.random_scrambling_matrices(dim, num_digits, seed)
        for dtype in [tf.int32, tf.int64]:
            actual = qmc.scramble_generating_matrices(generating_matrices, scrambling_matrices, num_digits, dtype=dtype, validate_args=True)
            with self.subTest('Shape'):
                self.assertEqual(actual.shape, generating_matrices.shape)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, dtype)
if __name__ == '__main__':
    tf.test.main()