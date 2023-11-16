"""Tests for open_spiel.python.algorithms.projected_replicator_dynamics."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import projected_replicator_dynamics

class ProjectedReplicatorDynamicsTest(absltest.TestCase):

    def test_two_players(self):
        if False:
            for i in range(10):
                print('nop')
        test_a = np.array([[2, 1, 0], [0, -1, -2]])
        test_b = np.array([[2, 1, 0], [0, -1, -2]])
        strategies = projected_replicator_dynamics.projected_replicator_dynamics([test_a, test_b], prd_initial_strategies=None, prd_iterations=50000, prd_dt=0.001, prd_gamma=1e-08, average_over_last_n_strategies=10)
        self.assertLen(strategies, 2, 'Wrong strategy length.')
        self.assertGreater(strategies[0][0], 0.999, 'Projected Replicator Dynamics failed in trivial case.')

    def test_three_players(self):
        if False:
            for i in range(10):
                print('nop')
        test_a = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
        test_b = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
        test_c = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
        strategies = projected_replicator_dynamics.projected_replicator_dynamics([test_a, test_b, test_c], prd_initial_strategies=None, prd_iterations=50000, prd_dt=0.001, prd_gamma=1e-06, average_over_last_n_strategies=10)
        self.assertLen(strategies, 3, 'Wrong strategy length.')
        self.assertGreater(strategies[0][0], 0.999, 'Projected Replicator Dynamics failed in trivial case.')
if __name__ == '__main__':
    absltest.main()