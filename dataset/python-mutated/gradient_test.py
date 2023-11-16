"""Tests for math.gradient.py."""
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class GradientTest(tf.test.TestCase):

    def test_forward_gradient(self):
        if False:
            print('Hello World!')
        t = tf.range(1, 3, dtype=tf.float32)
        func = lambda t: tf.stack([t, t ** 2, t ** 3], axis=0)
        with self.subTest('EagerExecution'):
            fwd_grad = self.evaluate(tff.math.fwd_gradient(func, t))
            self.assertEqual(fwd_grad.shape, (3, 2))
            np.testing.assert_allclose(fwd_grad, [[1.0, 1.0], [2.0, 4.0], [3.0, 12.0]])
        with self.subTest('GraphExecution'):

            @tf.function
            def grad_computation():
                if False:
                    i = 10
                    return i + 15
                y = func(t)
                return tff.math.fwd_gradient(y, t)
            fwd_grad = self.evaluate(grad_computation())
            self.assertEqual(fwd_grad.shape, (3, 2))
            np.testing.assert_allclose(fwd_grad, [[1.0, 1.0], [2.0, 4.0], [3.0, 12.0]])

    def test_forward_unconnected_gradient(self):
        if False:
            i = 10
            return i + 15
        t = tf.range(1, 3, dtype=tf.float32)
        zeros = tf.zeros([2], dtype=t.dtype)
        func = lambda t: tf.stack([zeros, zeros, zeros], axis=0)
        expected_result = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        with self.subTest('EagerExecution'):
            fwd_grad = self.evaluate(tff.math.fwd_gradient(func, t, unconnected_gradients=tf.UnconnectedGradients.ZERO))
            self.assertEqual(fwd_grad.shape, (3, 2))
            np.testing.assert_allclose(fwd_grad, expected_result)
        with self.subTest('GraphExecution'):

            @tf.function
            def grad_computation():
                if False:
                    return 10
                y = func(t)
                return tff.math.fwd_gradient(y, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            fwd_grad = self.evaluate(grad_computation())
            self.assertEqual(fwd_grad.shape, (3, 2))
            np.testing.assert_allclose(fwd_grad, expected_result)

    def test_backward_gradient(self):
        if False:
            while True:
                i = 10
        t = tf.range(1, 3, dtype=tf.float32)
        func = lambda t: tf.stack([t, t ** 2, t ** 3], axis=0)
        with self.subTest('EagerExecution'):
            backward_grad = self.evaluate(tff.math.gradients(func, t))
            self.assertEqual(backward_grad.shape, (2,))
            np.testing.assert_allclose(backward_grad, [6.0, 17.0])
        with self.subTest('GraphExecution'):

            @tf.function
            def grad_computation():
                if False:
                    i = 10
                    return i + 15
                y = func(t)
                return tff.math.gradients(y, t)
            backward_grad = self.evaluate(grad_computation())
            self.assertEqual(backward_grad.shape, (2,))
            np.testing.assert_allclose(backward_grad, [6.0, 17.0])

    def test_backward_unconnected_gradient(self):
        if False:
            return 10
        t = tf.range(1, 3, dtype=tf.float32)
        zeros = tf.zeros([2], dtype=t.dtype)
        expected_result = [0.0, 0.0]
        func = lambda t: tf.stack([zeros, zeros, zeros], axis=0)
        with self.subTest('EagerExecution'):
            backward_grad = self.evaluate(tff.math.gradients(func, t, unconnected_gradients=tf.UnconnectedGradients.ZERO))
            self.assertEqual(backward_grad.shape, (2,))
            np.testing.assert_allclose(backward_grad, expected_result)
        with self.subTest('GraphExecution'):

            @tf.function
            def grad_computation():
                if False:
                    for i in range(10):
                        print('nop')
                y = func(t)
                return tff.math.gradients(y, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            backward_grad = self.evaluate(grad_computation())
            self.assertEqual(backward_grad.shape, (2,))
            np.testing.assert_allclose(backward_grad, expected_result)

    def test_make_val_and_grad_fn(self):
        if False:
            i = 10
            return i + 15
        minimum = np.array([1.0, 1.0])
        scales = np.array([2.0, 3.0])

        @tff.math.make_val_and_grad_fn
        def quadratic(x):
            if False:
                i = 10
                return i + 15
            return tf.reduce_sum(input_tensor=scales * (x - minimum) ** 2)
        point = tf.constant([2.0, 2.0], dtype=tf.float64)
        (val, grad) = self.evaluate(quadratic(point))
        self.assertNear(val, 5.0, 1e-05)
        self.assertArrayNear(grad, [4.0, 6.0], 1e-05)
if __name__ == '__main__':
    tf.test.main()