"""Tests for open_spiel.python.algorithms.policy_value."""
from absl.testing import absltest
import numpy as np
from open_spiel.python import games
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel

class PolicyValueTest(absltest.TestCase):

    def test_expected_game_score_uniform_random_kuhn_poker(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('kuhn_poker')
        uniform_policy = policy.UniformRandomPolicy(game)
        uniform_policy_values = expected_game_score.policy_value(game.new_initial_state(), [uniform_policy] * 2)
        self.assertTrue(np.allclose(uniform_policy_values, [1 / 8, -1 / 8]))

    def test_expected_game_score_uniform_random_iterated_prisoner_dilemma(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('python_iterated_prisoners_dilemma(max_game_length=6)')
        pi = policy.UniformRandomPolicy(game)
        values = expected_game_score.policy_value(game.new_initial_state(), pi)
        np.testing.assert_allclose(values, [17.6385498, 17.6385498])
if __name__ == '__main__':
    absltest.main()