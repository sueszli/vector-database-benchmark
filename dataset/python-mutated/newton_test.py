"""Tests for math.root_finder_newton."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
newton_root = tff.math.root_search.newton_root

@test_util.run_all_in_graph_and_eager_modes
class RootFinderNewtonTest(parameterized.TestCase, tf.test.TestCase):
    """Tests for methods in root_finder_newton module."""

    def test_newton_root(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the newton root finder works on a square root example.'
        constants = np.array([4.0, 9.0, 16.0])
        initial_values = np.ones(len(constants))

        def objective_and_gradient(values):
            if False:
                print('Hello World!')
            objective = values ** 2 - constants
            gradient = 2.0 * values
            return (objective, gradient)
        (root_values, converged, failed) = self.evaluate(newton_root(objective_and_gradient, initial_values))
        roots_bench = np.array([2.0, 3.0, 4.0])
        converged_bench = np.array([True, True, True])
        failed_bench = np.array([False, False, False])
        np.testing.assert_array_equal(converged, converged_bench)
        np.testing.assert_array_equal(failed, failed_bench)
        np.testing.assert_almost_equal(root_values, roots_bench, decimal=7)

    def test_failure_and_non_convergence(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that we can determine when the root finder has failed.'
        constants = np.array([4.0, 9.0, 16.0])
        initial_values = np.zeros(len(constants))

        def objective_and_gradient(values):
            if False:
                return 10
            objective = values ** 2 - constants
            gradient = 2.0 * values
            return (objective, gradient)
        (_, converged, failed) = self.evaluate(newton_root(objective_and_gradient, initial_values))
        converged_bench = np.array([False, False, False])
        failed_bench = np.array([True, True, True])
        np.testing.assert_array_equal(converged, converged_bench)
        np.testing.assert_array_equal(failed, failed_bench)

    def test_too_low_max_iterations(self):
        if False:
            while True:
                i = 10
        'Tests that we can determine when max_iterations was too small.'
        constants = np.array([4.0, 9.0, 16.0])
        initial_values = np.ones(len(constants))

        def objective_and_gradient(values):
            if False:
                return 10
            objective = values ** 2 - constants
            gradient = 2.0 * values
            return (objective, gradient)
        (_, converged, failed) = self.evaluate(newton_root(objective_and_gradient, initial_values, max_iterations=1))
        converged_bench = np.array([False, False, False])
        failed_bench = np.array([False, False, False])
        np.testing.assert_array_equal(converged, converged_bench)
        np.testing.assert_array_equal(failed, failed_bench)
if __name__ == '__main__':
    tf.test.main()