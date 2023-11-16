"""Tests for parabolic PDE time marching schemes."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
fd_solvers = tff.math.pde.fd_solvers
dirichlet = tff.math.pde.boundary_conditions.dirichlet
neumann = tff.math.pde.boundary_conditions.neumann
grids = tff.math.pde.grids
crank_nicolson_scheme = tff.math.pde.steppers.crank_nicolson.crank_nicolson_scheme
explicit_scheme = tff.math.pde.steppers.explicit.explicit_scheme
extrapolation_scheme = tff.math.pde.steppers.extrapolation.extrapolation_scheme
implicit_scheme = tff.math.pde.steppers.implicit.implicit_scheme
weighted_implicit_explicit_scheme = tff.math.pde.steppers.weighted_implicit_explicit.weighted_implicit_explicit_scheme

@test_util.run_all_in_graph_and_eager_modes
class TimeMarchingSchemeTest(tf.test.TestCase, parameterized.TestCase):
    parameters = ({'testcase_name': 'Implicit', 'scheme': implicit_scheme, 'accuracy_order': 1}, {'testcase_name': 'Explicit', 'scheme': explicit_scheme, 'accuracy_order': 1}, {'testcase_name': 'Weighted', 'scheme': weighted_implicit_explicit_scheme(theta=0.3), 'accuracy_order': 1}, {'testcase_name': 'CrankNicolson', 'scheme': crank_nicolson_scheme, 'accuracy_order': 2}, {'testcase_name': 'Extrapolation', 'scheme': extrapolation_scheme, 'accuracy_order': 2})

    @parameterized.named_parameters(*parameters)
    def testHomogeneous(self, scheme, accuracy_order):
        if False:
            i = 10
            return i + 15
        time_step = 0.0001
        u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
        matrix = tf.constant([[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]], dtype=tf.float64)
        tridiag_form = self._convert_to_tridiagonal_format(matrix)
        actual = self.evaluate(scheme(u, 0, time_step, lambda t: (tridiag_form, None)))
        expected = self.evaluate(tf.squeeze(tf.matmul(tf.linalg.expm(matrix * time_step), tf.expand_dims(u, 1))))
        error_tolerance = 30 * time_step ** (accuracy_order + 1)
        self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

    @parameterized.named_parameters(*parameters)
    def testHomogeneousBackwards(self, scheme, accuracy_order):
        if False:
            while True:
                i = 10
        time_step = 0.0001
        u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
        matrix = tf.constant([[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]], dtype=tf.float64)
        tridiag_form = self._convert_to_tridiagonal_format(matrix)
        actual = self.evaluate(scheme(u, time_step, 0, lambda t: (tridiag_form, None)))
        expected = self.evaluate(tf.squeeze(tf.matmul(tf.linalg.expm(-matrix * time_step), tf.expand_dims(u, 1))))
        error_tolerance = 30 * time_step ** (accuracy_order + 1)
        self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

    @parameterized.named_parameters(*parameters)
    def testInhomogeneous(self, scheme, accuracy_order):
        if False:
            return 10
        time_step = 0.0001
        u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
        matrix = tf.constant([[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]], dtype=tf.float64)
        b = tf.constant([1, -1, -2, 2], dtype=tf.float64)
        tridiag_form = self._convert_to_tridiagonal_format(matrix)
        actual = self.evaluate(scheme(u, 0, time_step, lambda t: (tridiag_form, b)))
        exponent = tf.linalg.expm(matrix * time_step)
        eye = tf.eye(4, 4, dtype=tf.float64)
        u = tf.expand_dims(u, 1)
        b = tf.expand_dims(b, 1)
        expected = tf.matmul(exponent, u) + tf.matmul(exponent - eye, tf.matmul(tf.linalg.inv(matrix), b))
        expected = self.evaluate(tf.squeeze(expected))
        error_tolerance = 30 * time_step ** (accuracy_order + 1)
        self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

    @parameterized.named_parameters(*parameters)
    def testInhomogeneousBackwards(self, scheme, accuracy_order):
        if False:
            for i in range(10):
                print('nop')
        time_step = 0.0001
        u = tf.constant([1, 2, -1, -2], dtype=tf.float64)
        matrix = tf.constant([[1, -1, 0, 0], [3, 1, 2, 0], [0, -2, 1, 4], [0, 0, 3, 1]], dtype=tf.float64)
        b = tf.constant([1, -1, -2, 2], dtype=tf.float64)
        tridiag_form = self._convert_to_tridiagonal_format(matrix)
        actual = self.evaluate(scheme(u, time_step, 0, lambda t: (tridiag_form, b)))
        exponent = tf.linalg.expm(-matrix * time_step)
        eye = tf.eye(4, 4, dtype=tf.float64)
        u = tf.expand_dims(u, 1)
        b = tf.expand_dims(b, 1)
        expected = tf.matmul(exponent, u) + tf.matmul(exponent - eye, tf.matmul(tf.linalg.inv(matrix), b))
        expected = self.evaluate(tf.squeeze(expected))
        error_tolerance = 30 * time_step ** (accuracy_order + 1)
        self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

    def _convert_to_tridiagonal_format(self, matrix):
        if False:
            while True:
                i = 10
        matrix_np = self.evaluate(matrix)
        n = matrix_np.shape[0]
        superdiag = [matrix_np[i, i + 1] for i in range(n - 1)] + [0]
        diag = [matrix_np[i, i] for i in range(n)]
        subdiag = [0] + [matrix_np[i + 1, i] for i in range(n - 1)]
        return tuple((tf.constant(v, dtype=matrix.dtype) for v in (diag, superdiag, subdiag)))
if __name__ == '__main__':
    tf.test.main()