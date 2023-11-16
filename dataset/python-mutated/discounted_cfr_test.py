"""Tests for open_spiel.python.algorithms.discounted_cfr."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel

class DiscountedCfrTest(absltest.TestCase):

    def test_discounted_cfr_on_kuhn(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('kuhn_poker')
        solver = discounted_cfr.DCFRSolver(game)
        for _ in range(300):
            solver.evaluate_and_update_policy()
        average_policy = solver.average_policy()
        average_policy_values = expected_game_score.policy_value(game.new_initial_state(), [average_policy] * 2)
        np.testing.assert_allclose(average_policy_values, [-1 / 18, 1 / 18], atol=0.001)

    def test_discounted_cfr_runs_against_leduc(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('leduc_poker')
        solver = discounted_cfr.DCFRSolver(game)
        for _ in range(10):
            solver.evaluate_and_update_policy()
        solver.average_policy()
if __name__ == '__main__':
    absltest.main()