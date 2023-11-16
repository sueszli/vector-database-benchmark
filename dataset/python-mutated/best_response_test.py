"""Tests for open_spiel.python.algorithms.best_response."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python import games
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import get_all_states
import pyspiel

class BestResponseTest(parameterized.TestCase, absltest.TestCase):

    def test_best_response_is_a_policy(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('kuhn_poker')
        test_policy = policy.UniformRandomPolicy(game)
        br = best_response.BestResponsePolicy(game, policy=test_policy, player_id=0)
        expected_policy = {'0': 1, '1': 1, '2': 0, '0pb': 0, '1pb': 1, '2pb': 1}
        self.assertEqual(expected_policy, {key: br.best_response_action(key) for key in expected_policy.keys()})

    @parameterized.parameters(['kuhn_poker', 'leduc_poker'])
    def test_cpp_and_python_implementations_are_identical(self, game_name):
        if False:
            return 10
        game = pyspiel.load_game(game_name)
        python_policy = policy.UniformRandomPolicy(game)
        pyspiel_policy = pyspiel.UniformRandomPolicy(game)
        all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False, to_string=lambda s: s.information_state_string())
        for current_player in range(game.num_players()):
            python_br = best_response.BestResponsePolicy(game, current_player, python_policy)
            cpp_br = pyspiel.TabularBestResponse(game, current_player, pyspiel_policy).get_best_response_policy()
            for state in all_states.values():
                if state.current_player() != current_player:
                    continue
                self.assertEqual(python_br.action_probabilities(state), {a: prob for (a, prob) in cpp_br.action_probabilities(state).items() if prob != 0})

    @parameterized.parameters(('kuhn_poker', 2))
    def test_cpp_and_python_best_response_are_identical(self, game_name, num_players):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game(game_name, {'players': num_players})
        test_policy = policy.TabularPolicy(game)
        for i_player in range(num_players):
            best_resp_py_backend = best_response.BestResponsePolicy(game, i_player, test_policy)
            best_resp_cpp_backend = best_response.CPPBestResponsePolicy(game, i_player, test_policy)
            for state in best_resp_cpp_backend.all_states.values():
                if i_player == state.current_player():
                    py_dict = best_resp_py_backend.action_probabilities(state)
                    cpp_dict = best_resp_cpp_backend.action_probabilities(state)
                    for (key, value) in py_dict.items():
                        self.assertEqual(value, cpp_dict.get(key, 0.0))
                    for (key, value) in cpp_dict.items():
                        self.assertEqual(value, py_dict.get(key, 0.0))

    @parameterized.parameters(('kuhn_poker', 2), ('kuhn_poker', 3))
    def test_cpp_and_python_value_are_identical(self, game_name, num_players):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game(game_name, {'players': num_players})
        test_policy = policy.TabularPolicy(game)
        root_state = game.new_initial_state()
        for i_player in range(num_players):
            best_resp_py_backend = best_response.BestResponsePolicy(game, i_player, test_policy)
            best_resp_cpp_backend = best_response.CPPBestResponsePolicy(game, i_player, test_policy)
            value_py_backend = best_resp_py_backend.value(root_state)
            value_cpp_backend = best_resp_cpp_backend.value(root_state)
            self.assertTrue(np.allclose(value_py_backend, value_cpp_backend))

    def test_best_response_tic_tac_toe_value_is_consistent(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('tic_tac_toe')
        pi = policy.TabularPolicy(game)
        rng = np.random.RandomState(1234)
        pi.action_probability_array[:] = rng.rand(*pi.legal_actions_mask.shape)
        pi.action_probability_array *= pi.legal_actions_mask
        pi.action_probability_array /= np.sum(pi.action_probability_array, axis=1, keepdims=True)
        br = best_response.BestResponsePolicy(game, 1, pi)
        self.assertAlmostEqual(expected_game_score.policy_value(game.new_initial_state(), [pi, br])[1], br.value(game.new_initial_state()))

    def test_best_response_oshi_zumo_simultaneous_game(self):
        if False:
            return 10
        'Test best response computation for simultaneous game.'
        game = pyspiel.load_game('oshi_zumo(horizon=5,coins=5)')
        test_policy = policy.UniformRandomPolicy(game)
        br = best_response.BestResponsePolicy(game, policy=test_policy, player_id=0)
        expected_policy = {'0, 0, 0, 3, 0, 2': 1, '0, 0, 1, 4, 3, 1': 0, '0, 0, 4, 1, 0, 2, 0, 2': 1, '0, 1, 1, 0, 1, 4': 1, '0, 1, 4, 1, 0, 0, 0, 1': 1, '0, 2, 2, 2, 3, 0, 0, 0': 0, '0, 5, 0, 0, 0, 0, 3, 0': 1}
        self.assertEqual(expected_policy, {key: br.best_response_action(key) for key in expected_policy})
        self.assertAlmostEqual(br.value(game.new_initial_state()), 0.856471051954)

    def test_best_response_prisoner_dilemma_simultaneous_game(self):
        if False:
            i = 10
            return i + 15
        'Test best response computation for simultaneous game.'
        game = pyspiel.load_game('python_iterated_prisoners_dilemma(max_game_length=5)')
        test_policy = policy.UniformRandomPolicy(game)
        br = best_response.BestResponsePolicy(game, policy=test_policy, player_id=0)
        self.assertEqual(br.best_response_action('us:CCCC op:CCCC'), 1)
        self.assertEqual(br.best_response_action('us:DDDD op:CCCC'), 1)
        self.assertEqual(br.best_response_action('us:CDCD op:DCDC'), 1)
        self.assertEqual(br.best_response_action('us:CCCC op:DDDD'), 1)
        self.assertAlmostEqual(br.value(game.new_initial_state()), 21.4320068359375)

class TabularBestResponseMDPTest(absltest.TestCase):

    def test_tabular_best_response_mdp(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('kuhn_poker')
        uniform_random_policy = pyspiel.UniformRandomPolicy(game)
        tbr_mdp = pyspiel.TabularBestResponseMDP(game, uniform_random_policy)
        tbr_info = tbr_mdp.nash_conv()
        self.assertGreater(tbr_info.nash_conv, 0)
if __name__ == '__main__':
    absltest.main()