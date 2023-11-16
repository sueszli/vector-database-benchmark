"""Tests for open_spiel.python.algorithms.action_value.py."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import action_value
import pyspiel

class ActionValuesTest(parameterized.TestCase):

    @parameterized.parameters([['kuhn_poker', 2], ['kuhn_poker', 3], ['leduc_poker', 2]])
    def test_runs_with_uniform_policies(self, game_name, num_players):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game(game_name, {'players': num_players})
        calc = action_value.TreeWalkCalculator(game)
        uniform_policy = policy.TabularPolicy(game)
        calc.compute_all_states_action_values([uniform_policy] * num_players)

    def test_kuhn_poker_always_pass_p0(self):
        if False:
            return 10
        game = pyspiel.load_game('kuhn_poker')
        calc = action_value.TreeWalkCalculator(game)
        uniform_policy = policy.TabularPolicy(game)
        always_pass_policy = policy.FirstActionPolicy(game).to_tabular()
        returned_values = calc([always_pass_policy, uniform_policy], always_pass_policy)
        root_node_values = calc.get_root_node_values([always_pass_policy, uniform_policy])
        self.assertTrue(np.allclose(root_node_values, returned_values.root_node_values))
        np.testing.assert_array_almost_equal(np.asarray([[-1.0, -0.5], [-1.0, -2.0], [-0.5, 0.5], [-1.0, 0.0], [0.0, 1.5], [-1.0, 2.0], [0.0, 1.0], [0, 0], [1.0, 1.0], [0, 0], [-1.0, 1.0], [0, 0]]), returned_values.action_values)
        np.testing.assert_array_almost_equal(np.asarray([1 / 3, 1 / 6, 1 / 3, 1 / 6, 1 / 3, 1 / 6, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0]), returned_values.counterfactual_reach_probs)
        np.testing.assert_array_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], returned_values.player_reach_probs)
        np.testing.assert_array_almost_equal(np.asarray([np.array([-1 / 3, -1 / 6]), np.array([-1 / 6, -1 / 3]), np.array([-1 / 6, 1 / 6]), np.array([-1 / 6, 0.0]), np.array([0.0, 0.5]), np.array([-1 / 6, 1 / 3]), np.array([0.0, 1 / 3]), np.array([0.0, 0.0]), np.array([1 / 3, 1 / 3]), np.array([0.0, 0.0]), np.array([-1 / 3, 1 / 3]), np.array([0.0, 0.0])]), returned_values.sum_cfr_reach_by_action_value)
if __name__ == '__main__':
    absltest.main()