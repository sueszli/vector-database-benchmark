"""Tests for open_spiel.python.algorithms.psro_v2.best_response_oracle."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
import pyspiel

class BestResponseOracleTest(parameterized.TestCase, absltest.TestCase):

    @parameterized.parameters(('kuhn_poker', 2), ('kuhn_poker', 3), ('leduc_poker', 2))
    def test_cpp_python_best_response_oracle(self, game_name, num_players):
        if False:
            print('Hello World!')
        game = pyspiel.load_game(game_name, {'players': num_players})
        (all_states, _) = best_response.compute_states_and_info_states_if_none(game, all_states=None, state_to_information_state=None)
        current_best = [[policy.TabularPolicy(game).__copy__()] for _ in range(num_players)]
        probabilities_of_playing_policies = [[1.0] for _ in range(num_players)]
        py_oracle = best_response_oracle.BestResponseOracle(best_response_backend='py')
        cpp_oracle = best_response_oracle.BestResponseOracle(game=game, best_response_backend='cpp')
        training_params = [[{'total_policies': current_best, 'current_player': i, 'probabilities_of_playing_policies': probabilities_of_playing_policies}] for i in range(num_players)]
        py_best_rep = py_oracle(game, training_params)
        cpp_best_rep = cpp_oracle(game, training_params)
        for state in all_states.values():
            i_player = state.current_player()
            py_dict = py_best_rep[i_player][0].action_probabilities(state)
            cpp_dict = cpp_best_rep[i_player][0].action_probabilities(state)
            for action in py_dict.keys():
                self.assertEqual(py_dict.get(action, 0.0), cpp_dict.get(action, 0.0))
            for action in cpp_dict.keys():
                self.assertEqual(py_dict.get(action, 0.0), cpp_dict.get(action, 0.0))
if __name__ == '__main__':
    absltest.main()