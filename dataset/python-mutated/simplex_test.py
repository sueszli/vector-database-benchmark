"""Tests for open_spiel.python.algorithms.adidas_utils.helpers.simplex."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python.algorithms.adidas_utils.helpers import simplex

class SimplexTest(parameterized.TestCase):

    @parameterized.named_parameters(('inside', np.array([0.25, 0.75]), np.array([0.25, 0.75])), ('outside_1', np.ones(2), 0.5 * np.ones(2)), ('outside_2', np.array([2.0, 0.0]), np.array([1.0, 0.0])), ('outside_3', np.array([0.25, 0.25]), np.array([0.5, 0.5])))
    def test_euclidean_projection(self, vector, expected_projection):
        if False:
            for i in range(10):
                print('nop')
        projection = simplex.euclidean_projection_onto_simplex(vector, subset=False)
        self.assertListEqual(list(projection), list(expected_projection), msg='projection not accurate')

    @parameterized.named_parameters(('orth', np.array([0.75, 0.75]), np.array([0.0, 0.0])), ('oblique', np.array([1.0, 0.5]), np.array([0.25, -0.25])), ('tangent', np.array([0.25, 0.25, -0.5]), np.array([0.25, 0.25, -0.5])))
    def test_tangent_projection(self, vector, expected_projection):
        if False:
            return 10
        projection = simplex.project_grad(vector)
        self.assertListEqual(list(projection), list(expected_projection), msg='projection not accurate')

    @parameterized.named_parameters(('orth_1', np.array([0.5, 0.5]), np.array([0.75, 0.75]), 0.0), ('orth_2', np.array([1.0, 0.0]), np.array([0.75, 0.75]), 0.0), ('tangent_1', np.array([1.0, 0.0]), np.array([-0.5, 0.5]), 0.0), ('tangent_2', np.array([1.0, 0.0]), np.array([1.0, -1.0]), np.sqrt(2)))
    def test_grad_norm(self, dist, grad, expected_norm):
        if False:
            for i in range(10):
                print('nop')
        norm = simplex.grad_norm(dist, grad)
        self.assertAlmostEqual(norm, expected_norm, msg='norm not accurate')
if __name__ == '__main__':
    absltest.main()