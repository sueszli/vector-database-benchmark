"""Tests for dynamic_routing_to_mean_field_game."""
from absl.testing import absltest
from open_spiel.python import games
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.games import dynamic_routing_to_mean_field_game
from open_spiel.python.mfg import games as mfg_games
from open_spiel.python.mfg.algorithms import mirror_descent
import pyspiel

class DerivedNPlayerPolicyFromMeanFieldPolicyTest(absltest.TestCase):

    def test_state_conversion_method(self):
        if False:
            while True:
                i = 10
        'Test N player game state to mean field game state conversion.'

    def test_uniform_mfg_policy_conversion_to_n_player_uniform_policy(self):
        if False:
            i = 10
            return i + 15
        'Test conversion of uniform to uniform policy.'
        mfg_game = pyspiel.load_game('python_mfg_dynamic_routing', {'time_step_length': 0.05, 'max_num_time_step': 100})
        n_player_game = pyspiel.load_game('python_dynamic_routing', {'time_step_length': 0.05, 'max_num_time_step': 100})
        mfg_derived_policy = dynamic_routing_to_mean_field_game.DerivedNPlayerPolicyFromMeanFieldPolicy(n_player_game, policy.UniformRandomPolicy(mfg_game))
        derived_policy_value = expected_game_score.policy_value(n_player_game.new_initial_state(), mfg_derived_policy)
        uniform_policy_value = expected_game_score.policy_value(n_player_game.new_initial_state(), policy.UniformRandomPolicy(n_player_game))
        self.assertSequenceAlmostEqual(derived_policy_value, uniform_policy_value)

    def test_pigou_network_game_outcome_optimal_mfg_policy_in_n_player_game(self):
        if False:
            while True:
                i = 10
        'Test MFG Nash equilibrium policy for the Pigou network.'

    def test_learning_and_applying_mfg_policy_in_n_player_game(self):
        if False:
            for i in range(10):
                print('nop')
        'Test converting learnt MFG policy default game.'
        mfg_game = pyspiel.load_game('python_mfg_dynamic_routing')
        omd = mirror_descent.MirrorDescent(mfg_game, lr=1)
        for _ in range(10):
            omd.iteration()
        mfg_policy = omd.get_policy()
        n_player_game = pyspiel.load_game('python_dynamic_routing')
        mfg_derived_policy = dynamic_routing_to_mean_field_game.DerivedNPlayerPolicyFromMeanFieldPolicy(n_player_game, mfg_policy)
        expected_game_score.policy_value(n_player_game.new_initial_state(), mfg_derived_policy)
if __name__ == '__main__':
    absltest.main()