"""Tests for lattice rules."""
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
qmc = tff.math.qmc

@test_util.run_all_in_graph_and_eager_modes
class LatticeRuleTest(tf.test.TestCase):
    generating_vectors_values = [1, 387275, 314993, 50301, 174023, 354905, 303021, 486111, 286797, 463237, 211171, 216757, 29831, 155061, 315509, 193933, 129563, 276501, 395079, 139111]

    def generating_vectors(self, dtype=tf.int32):
        if False:
            while True:
                i = 10
        return tf.constant(self.generating_vectors_values, dtype=dtype)

    def test_random_scrambling_vectors(self):
        if False:
            print('Hello World!')
        dim = 20
        seed = (2, 3)
        actual = qmc.random_scrambling_vectors(dim, seed, validate_args=True)
        with self.subTest('Shape'):
            self.assertEqual(actual.shape, (dim,))
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, tf.float32)
        with self.subTest('Min Value'):
            self.assertAllLess(actual, tf.ones(shape=(), dtype=tf.float32))
        with self.subTest('Max Value'):
            self.assertAllGreaterEqual(actual, tf.zeros(shape=(), dtype=tf.float32))

    def test_random_scrambling_vectors_with_dtype(self):
        if False:
            i = 10
            return i + 15
        dim = 20
        seed = (2, 3)
        for dtype in [tf.float32, tf.float64]:
            actual = qmc.random_scrambling_vectors(dim, seed, dtype=dtype, validate_args=True)
            with self.subTest('Shape'):
                self.assertEqual(actual.shape, (dim,))
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, dtype)
            with self.subTest('Min Value'):
                self.assertAllLess(actual, tf.ones(shape=(), dtype=dtype))
            with self.subTest('Max Value'):
                self.assertAllGreaterEqual(actual, tf.zeros(shape=(), dtype=dtype))

    def test_lattice_rule_sample(self):
        if False:
            for i in range(10):
                print('nop')
        expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0625, 0.6875, 0.0625, 0.8125, 0.4375, 0.5625], [0.125, 0.375, 0.125, 0.625, 0.875, 0.125], [0.1875, 0.0625, 0.1875, 0.4375, 0.3125, 0.6875], [0.25, 0.75, 0.25, 0.25, 0.75, 0.25], [0.3125, 0.4375, 0.3125, 0.0625, 0.1875, 0.8125], [0.375, 0.125, 0.375, 0.875, 0.625, 0.375], [0.4375, 0.8125, 0.4375, 0.6875, 0.0625, 0.9375], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5625, 0.1875, 0.5625, 0.3125, 0.9375, 0.0625], [0.625, 0.875, 0.625, 0.125, 0.375, 0.625], [0.6875, 0.5625, 0.6875, 0.9375, 0.8125, 0.1875], [0.75, 0.25, 0.75, 0.75, 0.25, 0.75], [0.8125, 0.9375, 0.8125, 0.5625, 0.6875, 0.3125], [0.875, 0.625, 0.875, 0.375, 0.125, 0.875], [0.9375, 0.3125, 0.9375, 0.1875, 0.5625, 0.4375]], dtype=tf.float32)
        for dtype in [tf.int32, tf.int64]:
            actual = qmc.lattice_rule_sample(self.generating_vectors(dtype=dtype), 6, 16, validate_args=True)
            with self.subTest('Values'):
                self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, expected.dtype)

    def test_lattice_rule_sample_with_sequence_indices(self):
        if False:
            while True:
                i = 10
        indices = [2, 3, 6, 9, 11, 14]
        expected = tf.constant([[0.125, 0.375, 0.125, 0.625, 0.875, 0.125], [0.1875, 0.0625, 0.1875, 0.4375, 0.3125, 0.6875], [0.375, 0.125, 0.375, 0.875, 0.625, 0.375], [0.5625, 0.1875, 0.5625, 0.3125, 0.9375, 0.0625], [0.6875, 0.5625, 0.6875, 0.9375, 0.8125, 0.1875], [0.875, 0.625, 0.875, 0.375, 0.125, 0.875]], dtype=tf.float32)
        actual = qmc.lattice_rule_sample(self.generating_vectors(), 6, 16, sequence_indices=tf.constant(indices, dtype=tf.int32), validate_args=True)
        with self.subTest('Values'):
            self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, expected.dtype)

    def test_lattice_rule_sample_with_zero_additive_shift(self):
        if False:
            i = 10
            return i + 15
        generating_vectors = self.generating_vectors()
        expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.125, 0.375, 0.125, 0.625, 0.875], [0.25, 0.75, 0.25, 0.25, 0.75], [0.375, 0.125, 0.375, 0.875, 0.625], [0.5, 0.5, 0.5, 0.5, 0.5], [0.625, 0.875, 0.625, 0.125, 0.375], [0.75, 0.25, 0.75, 0.75, 0.25], [0.875, 0.625, 0.875, 0.375, 0.125]], dtype=tf.float32)
        for dtype in [tf.float32, tf.float64]:
            actual = qmc.lattice_rule_sample(generating_vectors, 5, 8, additive_shift=tf.zeros_like(generating_vectors, dtype=dtype), validate_args=True)
            with self.subTest('Values'):
                self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, expected.dtype)

    def test_lattice_rule_sample_with_non_zero_additive_shift(self):
        if False:
            i = 10
            return i + 15
        generating_vectors = self.generating_vectors()
        additive_shift = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        expected = tf.constant([[0.0, 0.05, 0.1, 0.15, 0.2], [0.125, 0.425, 0.225, 0.775, 0.075], [0.25, 0.8, 0.35, 0.4, 0.95], [0.375, 0.175, 0.475, 0.025, 0.825], [0.5, 0.55, 0.6, 0.65, 0.7], [0.625, 0.925, 0.725, 0.275, 0.575], [0.75, 0.3, 0.85, 0.9, 0.45], [0.875, 0.675, 0.975, 0.525, 0.325]], dtype=tf.float32)
        for dtype in [tf.float32, tf.float64]:
            actual = qmc.lattice_rule_sample(generating_vectors, 5, 8, additive_shift=tf.constant(additive_shift, dtype=dtype), validate_args=True)
            with self.subTest('Values'):
                self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, expected.dtype)

    def test_lattice_rule_sample_with_tent_transform(self):
        if False:
            for i in range(10):
                print('nop')
        expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.25, 0.75, 0.25], [0.5, 0.5, 0.5, 0.5, 0.5], [0.75, 0.25, 0.75, 0.25, 0.75], [1.0, 1.0, 1.0, 1.0, 1.0], [0.75, 0.25, 0.75, 0.25, 0.75], [0.5, 0.5, 0.5, 0.5, 0.5], [0.25, 0.75, 0.25, 0.75, 0.25]], dtype=tf.float32)
        actual = qmc.lattice_rule_sample(self.generating_vectors(), 5, 8, apply_tent_transform=True, validate_args=True)
        with self.subTest('Values'):
            self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
        with self.subTest('DType'):
            self.assertEqual(actual.dtype, expected.dtype)

    def test_lattice_rule_sample_with_dtype(self):
        if False:
            while True:
                i = 10
        generating_vectors = self.generating_vectors()
        for dtype in [tf.float32, tf.float64]:
            expected = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0], [0.125, 0.375, 0.125, 0.625, 0.875], [0.25, 0.75, 0.25, 0.25, 0.75], [0.375, 0.125, 0.375, 0.875, 0.625], [0.5, 0.5, 0.5, 0.5, 0.5], [0.625, 0.875, 0.625, 0.125, 0.375], [0.75, 0.25, 0.75, 0.75, 0.25], [0.875, 0.625, 0.875, 0.375, 0.125]], dtype=dtype)
            actual = qmc.lattice_rule_sample(generating_vectors, 5, 8, validate_args=True, dtype=dtype)
            with self.subTest('Values'):
                self.assertAllClose(self.evaluate(actual), self.evaluate(expected), rtol=1e-06)
            with self.subTest('DType'):
                self.assertEqual(actual.dtype, dtype)
if __name__ == '__main__':
    tf.test.main()