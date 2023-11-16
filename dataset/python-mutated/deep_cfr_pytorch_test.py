"""Tests for open_spiel.python.pytorch.deep_cfr."""
from absl import app
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from open_spiel.python import policy
import pyspiel
from open_spiel.python.pytorch import deep_cfr
SEED = 24984617

class DeepCFRPyTorchTest(parameterized.TestCase):

    @parameterized.parameters('leduc_poker', 'kuhn_poker', 'liars_dice')
    def test_deep_cfr_runs(self, game_name):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game(game_name)
        deep_cfr_solver = deep_cfr.DeepCFRSolver(game, policy_network_layers=(8, 4), advantage_network_layers=(4, 2), num_iterations=2, num_traversals=2, learning_rate=0.001, batch_size_advantage=None, batch_size_strategy=None, memory_capacity=10000000.0)
        deep_cfr_solver.solve()

    def test_matching_pennies_3p(self):
        if False:
            return 10
        game = pyspiel.load_game_as_turn_based('matching_pennies_3p')
        deep_cfr_solver = deep_cfr.DeepCFRSolver(game, policy_network_layers=(16, 8), advantage_network_layers=(32, 16), num_iterations=2, num_traversals=2, learning_rate=0.001, batch_size_advantage=None, batch_size_strategy=None, memory_capacity=10000000.0)
        deep_cfr_solver.solve()
        conv = pyspiel.nash_conv(game, policy.python_policy_to_pyspiel_policy(policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)))
        logging.info('Deep CFR in Matching Pennies 3p. NashConv: %.2f', conv)

def main(_):
    if False:
        return 10
    torch.manual_seed(SEED)
    absltest.main()
if __name__ == '__main__':
    app.run(main)