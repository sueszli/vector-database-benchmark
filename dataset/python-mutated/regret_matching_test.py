"""Tests for open_spiel.python.algorithms.regret_matching."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import regret_matching
from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel

class RegretMatchingTest(absltest.TestCase):

    def test_two_players(self):
        if False:
            return 10
        test_a = np.array([[2, 1, 0], [0, -1, -2]])
        test_b = np.array([[2, 1, 0], [0, -1, -2]])
        strategies = regret_matching.regret_matching([test_a, test_b], initial_strategies=None, iterations=50000, prd_gamma=1e-08, average_over_last_n_strategies=10)
        self.assertLen(strategies, 2, 'Wrong strategy length.')
        self.assertGreater(strategies[0][0], 0.999, 'Regret matching failed in trivial case.')

    def test_three_players(self):
        if False:
            while True:
                i = 10
        test_a = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
        test_b = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
        test_c = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
        strategies = regret_matching.regret_matching([test_a, test_b, test_c], initial_strategies=None, iterations=50000, gamma=1e-06, average_over_last_n_strategies=10)
        self.assertLen(strategies, 3, 'Wrong strategy length.')
        self.assertGreater(strategies[0][0], 0.999, 'Regret matching failed in trivial case.')

    def test_rps(self):
        if False:
            return 10
        game = pyspiel.load_game('matrix_rps')
        payoffs_array = game_payoffs_array(game)
        strategies = regret_matching.regret_matching([payoffs_array[0], payoffs_array[1]], initial_strategies=[np.array([0.1, 0.4, 0.5]), np.array([0.9, 0.1, 0.01])], iterations=50000, gamma=1e-06)
        self.assertLen(strategies, 2, 'Wrong strategy length.')
        self.assertAlmostEqual(strategies[0][0], 1 / 3.0, places=2)
        self.assertAlmostEqual(strategies[0][1], 1 / 3.0, places=2)
        self.assertAlmostEqual(strategies[0][2], 1 / 3.0, places=2)

    def test_biased_rps(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('matrix_brps')
        payoffs_array = game_payoffs_array(game)
        strategies = regret_matching.regret_matching([payoffs_array[0], payoffs_array[1]], iterations=50000, gamma=1e-08)
        self.assertLen(strategies, 2, 'Wrong strategy length.')
        self.assertAlmostEqual(strategies[0][0], 1 / 16.0, places=1)
        self.assertAlmostEqual(strategies[0][1], 10 / 16.0, places=1)
        self.assertAlmostEqual(strategies[0][2], 5 / 16.0, places=1)
if __name__ == '__main__':
    absltest.main()