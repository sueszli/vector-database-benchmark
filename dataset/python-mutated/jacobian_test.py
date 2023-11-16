"""Tests for math.jacobian.py."""
import functools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

def quadratic(p, x):
    if False:
        while True:
            i = 10
    'Quadratic function.'
    a = tf.expand_dims(p[..., 0], axis=-1)
    b = tf.expand_dims(p[..., 1], axis=-1)
    c = tf.expand_dims(p[..., 2], axis=-1)
    return a * x ** 2 + b * x + c

def multivariate_quadratic(p, x, y):
    if False:
        for i in range(10):
            print('nop')
    'Multivariate quadratic function.'
    a = tf.expand_dims(p[..., 0], axis=-1)
    b = tf.expand_dims(p[..., 1], axis=-1)
    c = tf.expand_dims(p[..., 2], axis=-1)
    d = tf.expand_dims(p[..., 3], axis=-1)
    return a * x ** 2 + b * x * y + c * y ** 2 + d

@test_util.run_all_in_graph_and_eager_modes
class JacobianTest(parameterized.TestCase, tf.test.TestCase):
    """Jacobian test cases."""

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': tf.float32}, {'testcase_name': 'DoublePrecision', 'dtype': tf.float64})
    def test_jacobian(self, dtype):
        if False:
            print('Hello World!')
        'Test function jacobian.'
        x = tf.range(1, 3, dtype=dtype)
        func = functools.partial(quadratic, x=x)
        with self.subTest('Quadratic'):
            with self.subTest('SingleTensor'):
                ps = tf.constant([1.0, 2.0, -1.0], dtype=dtype)
                expected_jacobian = [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]]
                jacobian = self.evaluate(tff.math.jacobian(func, ps))
                self.assertEqual(jacobian.shape, (2, 3))
                np.testing.assert_allclose(jacobian, expected_jacobian)
            with self.subTest('BatchedTensor'):
                ps = tf.constant([[1.0, 2.0, -1.0], [2.0, 0.0, 0.0]], dtype=dtype)
                expected_jacobian = [[[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]], [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]]]
                jacobian = self.evaluate(tff.math.jacobian(func, ps))
                self.assertEqual(jacobian.shape, (2, 2, 3))
                np.testing.assert_allclose(jacobian, expected_jacobian)
        x = tf.range(1, 3, dtype=dtype)
        y = tf.range(-3, -1, dtype=dtype)
        func = functools.partial(multivariate_quadratic, x=x, y=y)
        with self.subTest('MultivariateQuadratic'):
            with self.subTest('SingleTensor'):
                ps = tf.constant([1.0, 2.0, -1.0, 0.0], dtype=dtype)
                expected_jacobian = [[1.0, -3.0, 9.0, 1.0], [4.0, -4.0, 4.0, 1.0]]
                jacobian = self.evaluate(tff.math.jacobian(func, ps))
                self.assertEqual(jacobian.shape, (2, 4))
                np.testing.assert_allclose(jacobian, expected_jacobian)
            with self.subTest('BatchedTensor'):
                ps = tf.constant([[1.0, 2.0, -1.0, 0.0], [1.0, 2.0, -1.0, 0.0]], dtype=dtype)
                expected_jacobian = [[[1.0, -3.0, 9.0, 1.0], [4.0, -4.0, 4.0, 1.0]], [[1.0, -3.0, 9.0, 1.0], [4.0, -4.0, 4.0, 1.0]]]
                jacobian = self.evaluate(tff.math.jacobian(func, ps))
                self.assertEqual(jacobian.shape, (2, 2, 4))
                np.testing.assert_allclose(jacobian, expected_jacobian)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': tf.float32}, {'testcase_name': 'DoublePrecision', 'dtype': tf.float64})
    def test_value_and_jacobian(self, dtype):
        if False:
            i = 10
            return i + 15
        'Test function value_and_jacobian.'
        x = tf.range(1, 3, dtype=dtype)
        func = functools.partial(quadratic, x=x)
        with self.subTest('Quadratic'):
            with self.subTest('SingleTensor'):
                func = functools.partial(quadratic, x=x)
                ps = tf.constant([1.0, 2.0, -1.0], dtype=dtype)
                (values, jacobian) = self.evaluate(tff.math.value_and_jacobian(func, ps))
                with self.subTest('Values'):
                    expected_values = [2.0, 7.0]
                    self.assertEqual(values.shape, (2,))
                    np.testing.assert_allclose(values, expected_values)
                with self.subTest('Jacobian'):
                    expected_jacobian = [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]]
                    self.assertEqual(jacobian.shape, (2, 3))
                    np.testing.assert_allclose(jacobian, expected_jacobian)
            with self.subTest('BatchedTensor'):
                func = functools.partial(quadratic, x=x)
                ps = tf.constant([[1.0, 2.0, -1.0], [2.0, 0.0, 0.0]], dtype=dtype)
                (values, jacobian) = self.evaluate(tff.math.value_and_jacobian(func, ps))
                with self.subTest('Values'):
                    expected_values = [[2.0, 7.0], [2.0, 8.0]]
                    self.assertEqual(values.shape, (2, 2))
                    np.testing.assert_allclose(values, expected_values)
                with self.subTest('Jacobian'):
                    expected_jacobian = [[[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]], [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0]]]
                    self.assertEqual(jacobian.shape, (2, 2, 3))
                    np.testing.assert_allclose(jacobian, expected_jacobian)
        x = tf.range(1, 3, dtype=dtype)
        y = tf.range(-3, -1, dtype=dtype)
        func = functools.partial(multivariate_quadratic, x=x, y=y)
        with self.subTest('MultivariateQuadratic'):
            with self.subTest('SingleTensor'):
                ps = tf.constant([1.0, 2.0, -1.0, 0.0], dtype=dtype)
                (values, jacobian) = self.evaluate(tff.math.value_and_jacobian(func, ps))
                with self.subTest('Values'):
                    expected_values = [-14.0, -8.0]
                    self.assertEqual(values.shape, (2,))
                    np.testing.assert_allclose(values, expected_values)
                with self.subTest('Jacobian'):
                    expected_jacobian = [[1.0, -3.0, 9.0, 1.0], [4.0, -4.0, 4.0, 1.0]]
                    self.assertEqual(jacobian.shape, (2, 4))
                    np.testing.assert_allclose(jacobian, expected_jacobian)
            with self.subTest('BatchedTensor'):
                ps = tf.constant([[1.0, 2.0, -1.0, 0.0], [0.0, 1.0, -2.0, 1.0]], dtype=dtype)
                (values, jacobian) = self.evaluate(tff.math.value_and_jacobian(func, ps))
                with self.subTest('Values'):
                    expected_values = [[-14.0, -8.0], [-20.0, -11.0]]
                    self.assertEqual(values.shape, (2, 2))
                    np.testing.assert_allclose(values, expected_values)
                with self.subTest('Jacobian'):
                    expected_jacobian = [[[1.0, -3.0, 9.0, 1.0], [4.0, -4.0, 4.0, 1.0]], [[1.0, -3.0, 9.0, 1.0], [4.0, -4.0, 4.0, 1.0]]]
                    self.assertEqual(jacobian.shape, (2, 2, 4))
                    np.testing.assert_allclose(jacobian, expected_jacobian)
if __name__ == '__main__':
    tf.test.main()