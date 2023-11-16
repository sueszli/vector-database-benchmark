"""Tests for open_spiel.python.algorithms.action_value_vs_best_response.py."""
from absl.testing import absltest
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import action_value_vs_best_response
import pyspiel

class ActionValuesVsBestResponseTest(absltest.TestCase):

    def test_kuhn_poker_uniform(self):
        if False:
            return 10
        game = pyspiel.load_game('kuhn_poker')
        calc = action_value_vs_best_response.Calculator(game)
        (expl, avvbr, cfrp, player_reach_probs) = calc(0, policy.UniformRandomPolicy(game), ['0', '1', '2', '0pb', '1pb', '2pb'])
        self.assertAlmostEqual(expl, 15 / 36)
        np.testing.assert_allclose(avvbr, [[-1.5, -2.0], [-0.5, -0.5], [0.5, 1.5], [-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]])
        np.testing.assert_allclose(cfrp, [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose([1, 1, 1, 1 / 2, 1 / 2, 1 / 2], player_reach_probs)

    def test_kuhn_poker_always_pass_p0(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('kuhn_poker')
        calc = action_value_vs_best_response.Calculator(game)
        (expl, avvbr, cfrp, player_reach_probs) = calc(0, policy.FirstActionPolicy(game), ['0', '1', '2', '0pb', '1pb', '2pb'])
        self.assertAlmostEqual(expl, 1.0)
        np.testing.assert_allclose(avvbr, [[-1, 1], [-1, 1], [-1, 1], [-1, -2], [-1, 2], [-1, 2]])
        np.testing.assert_allclose(cfrp, [1 / 3, 1 / 3, 1 / 3, 1 / 6, 1 / 6, 1 / 3])
        np.testing.assert_allclose([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], player_reach_probs)
if __name__ == '__main__':
    absltest.main()