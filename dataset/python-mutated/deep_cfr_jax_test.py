"""Tests for open_spiel.python.jax.deep_cfr."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import deep_cfr
import pyspiel

class DeepCFRTest(parameterized.TestCase):

    @parameterized.parameters('leduc_poker', 'kuhn_poker', 'liars_dice')
    def test_deep_cfr_runs(self, game_name):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game(game_name)
        deep_cfr_solver = deep_cfr.DeepCFRSolver(game, policy_network_layers=(8, 4), advantage_network_layers=(4, 2), num_iterations=2, num_traversals=2, learning_rate=0.001, batch_size_advantage=8, batch_size_strategy=8, memory_capacity=10000000.0)
        deep_cfr_solver.solve()

    def test_matching_pennies_3p(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game_as_turn_based('matching_pennies_3p')
        deep_cfr_solver = deep_cfr.DeepCFRSolver(game, policy_network_layers=(16, 8), advantage_network_layers=(32, 16), num_iterations=2, num_traversals=2, learning_rate=0.001, batch_size_advantage=8, batch_size_strategy=8, memory_capacity=10000000.0)
        deep_cfr_solver.solve()
        conv = exploitability.nash_conv(game, policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities))
        print('Deep CFR in Matching Pennies 3p. NashConv: {}'.format(conv))
if __name__ == '__main__':
    absltest.main()