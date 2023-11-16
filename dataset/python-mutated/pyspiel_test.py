"""Tests for open_spiel.python.pybind11.pyspiel."""
import os
from absl.testing import absltest
from open_spiel.python import games
from open_spiel.python import policy
from open_spiel.python.mfg import games as mfgs
import pyspiel
EXPECTED_GAMES = frozenset(['2048', 'add_noise', 'amazons', 'backgammon', 'bargaining', 'battleship', 'blackjack', 'blotto', 'breakthrough', 'bridge', 'bridge_uncontested_bidding', 'catch', 'chat_game', 'checkers', 'chess', 'cliff_walking', 'clobber', 'coin_game', 'colored_trails', 'connect_four', 'coop_box_pushing', 'coop_to_1p', 'coordinated_mp', 'crazy_eights', 'cursor_go', 'dark_chess', 'dark_hex', 'dark_hex_ir', 'deep_sea', 'dots_and_boxes', 'dou_dizhu', 'efg_game', 'euchre', 'first_sealed_auction', 'gin_rummy', 'go', 'goofspiel', 'havannah', 'hex', 'hearts', 'kriegspiel', 'kuhn_poker', 'laser_tag', 'lewis_signaling', 'leduc_poker', 'liars_dice', 'liars_dice_ir', 'maedn', 'mancala', 'markov_soccer', 'matching_pennies_3p', 'matrix_bos', 'matrix_brps', 'matrix_cd', 'matrix_coordination', 'matrix_mp', 'matrix_pd', 'matrix_rps', 'matrix_rpsw', 'matrix_sh', 'matrix_shapleys_game', 'mean_field_lin_quad', 'mfg_crowd_modelling', 'mfg_crowd_modelling_2d', 'mfg_dynamic_routing', 'mfg_garnet', 'misere', 'morpion_solitaire', 'negotiation', 'nfg_game', 'nim', 'nine_mens_morris', 'normal_form_extensive_game', 'oh_hell', 'oshi_zumo', 'othello', 'oware', 'pentago', 'pathfinding', 'phantom_go', 'phantom_ttt', 'phantom_ttt_ir', 'pig', 'python_block_dominoes', 'python_dynamic_routing', 'python_iterated_prisoners_dilemma', 'python_mfg_crowd_avoidance', 'python_mfg_crowd_modelling', 'python_mfg_dynamic_routing', 'python_mfg_periodic_aversion', 'python_mfg_predator_prey', 'python_kuhn_poker', 'python_tic_tac_toe', 'python_liars_poker', 'quoridor', 'repeated_game', 'rbc', 'restricted_nash_response', 'sheriff', 'skat', 'start_at', 'solitaire', 'stones_and_gems', 'tarok', 'tic_tac_toe', 'tiny_bridge_2p', 'tiny_bridge_4p', 'tiny_hanabi', 'trade_comm', 'turn_based_simultaneous_game', 'ultimate_tic_tac_toe', 'y', 'zerosum'])

class PyspielTest(absltest.TestCase):

    def test_registered_names(self):
        if False:
            i = 10
            return i + 15
        game_names = pyspiel.registered_names()
        expected = list(EXPECTED_GAMES)
        if os.environ.get('OPEN_SPIEL_BUILD_WITH_HANABI', 'OFF') == 'ON' and 'hanabi' not in expected:
            expected.append('hanabi')
        if os.environ.get('OPEN_SPIEL_BUILD_WITH_ACPC', 'OFF') == 'ON' and 'universal_poker' not in expected:
            expected.append('universal_poker')
        expected = sorted(expected)
        self.assertCountEqual(game_names, expected)

    def teste_default_loadable(self):
        if False:
            for i in range(10):
                print('nop')
        non_default_loadable = [game.short_name for game in pyspiel.registered_games() if not game.default_loadable]
        expected = ['add_noise', 'efg_game', 'nfg_game', 'misere', 'turn_based_simultaneous_game', 'normal_form_extensive_game', 'repeated_game', 'restricted_nash_response', 'start_at', 'zerosum']
        self.assertCountEqual(non_default_loadable, expected)

    def test_registered_game_attributes(self):
        if False:
            while True:
                i = 10
        game_list = {game.short_name: game for game in pyspiel.registered_games()}
        self.assertEqual(game_list['kuhn_poker'].dynamics, pyspiel.GameType.Dynamics.SEQUENTIAL)
        self.assertEqual(game_list['kuhn_poker'].chance_mode, pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
        self.assertEqual(game_list['kuhn_poker'].information, pyspiel.GameType.Information.IMPERFECT_INFORMATION)
        self.assertEqual(game_list['kuhn_poker'].utility, pyspiel.GameType.Utility.ZERO_SUM)
        self.assertEqual(game_list['kuhn_poker'].min_num_players, 2)

    def test_create_game(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('kuhn_poker')
        game_info = game.get_type()
        self.assertEqual(game_info.information, pyspiel.GameType.Information.IMPERFECT_INFORMATION)
        self.assertEqual(game.num_players(), 2)

    def test_play_kuhn_poker(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('kuhn_poker')
        state = game.new_initial_state()
        self.assertEqual(state.is_chance_node(), True)
        self.assertEqual(state.chance_outcomes(), [(0, 1 / 3), (1, 1 / 3), (2, 1 / 3)])
        state.apply_action(1)
        self.assertEqual(state.is_chance_node(), True)
        self.assertEqual(state.chance_outcomes(), [(0, 0.5), (2, 0.5)])
        state.apply_action(2)
        self.assertEqual(state.is_chance_node(), False)
        self.assertEqual(state.legal_actions(), [0, 1])
        sampler = pyspiel.UniformProbabilitySampler(0.0, 1.0)
        clone = state.resample_from_infostate(1, sampler)
        self.assertEqual(clone.information_state_string(1), state.information_state_string(1))

    def test_othello(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('othello')
        state = game.new_initial_state()
        self.assertFalse(state.is_chance_node())
        self.assertFalse(state.is_terminal())
        self.assertEqual(state.legal_actions(), [19, 26, 37, 44])

    def test_tic_tac_toe(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('tic_tac_toe')
        state = game.new_initial_state()
        self.assertFalse(state.is_chance_node())
        self.assertFalse(state.is_terminal())
        self.assertEqual(state.legal_actions(), [0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test_game_parameters_from_string_empty(self):
        if False:
            while True:
                i = 10
        self.assertEqual(pyspiel.game_parameters_from_string(''), {})

    def test_game_parameters_from_string_simple(self):
        if False:
            print('Hello World!')
        self.assertEqual(pyspiel.game_parameters_from_string('foo'), {'name': 'foo'})

    def test_game_parameters_from_string_with_options(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(pyspiel.game_parameters_from_string('foo(x=2,y=true)'), {'name': 'foo', 'x': 2, 'y': True})

    def test_game_parameters_from_string_with_subgame(self):
        if False:
            print('Hello World!')
        self.assertEqual(pyspiel.game_parameters_from_string('foo(x=2,y=true,subgame=bar(z=False))'), {'name': 'foo', 'x': 2, 'y': True, 'subgame': {'name': 'bar', 'z': False}})

    def test_game_parameters_to_string_empty(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(pyspiel.game_parameters_to_string({}), '')

    def test_game_parameters_to_string_simple(self):
        if False:
            return 10
        self.assertEqual(pyspiel.game_parameters_to_string({'name': 'foo'}), 'foo()')

    def test_game_parameters_to_string_with_options(self):
        if False:
            while True:
                i = 10
        self.assertEqual(pyspiel.game_parameters_to_string({'name': 'foo', 'x': 2, 'y': True}), 'foo(x=2,y=True)')

    def test_game_parameters_to_string_with_subgame(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(pyspiel.game_parameters_to_string({'name': 'foo', 'x': 2, 'y': True, 'subgame': {'name': 'bar', 'z': False}}), 'foo(subgame=bar(z=False),x=2,y=True)')

    def test_game_type(self):
        if False:
            return 10
        game_type = pyspiel.GameType('matrix_mp', 'Matching Pennies', pyspiel.GameType.Dynamics.SIMULTANEOUS, pyspiel.GameType.ChanceMode.DETERMINISTIC, pyspiel.GameType.Information.PERFECT_INFORMATION, pyspiel.GameType.Utility.ZERO_SUM, pyspiel.GameType.RewardModel.TERMINAL, 2, 2, True, True, False, False, dict())
        self.assertEqual(game_type.chance_mode, pyspiel.GameType.ChanceMode.DETERMINISTIC)

    def test_error_handling(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, "Unknown game 'invalid_game_name'"):
            unused_game = pyspiel.load_game('invalid_game_name')

    def test_can_create_cpp_tabular_policy(self):
        if False:
            i = 10
            return i + 15
        for game_name in ['kuhn_poker', 'leduc_poker', 'liars_dice']:
            game = pyspiel.load_game(game_name)
            policy.python_policy_to_pyspiel_policy(policy.TabularPolicy(game))

    def test_simultaneous_game_history(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('coop_box_pushing')
        state = game.new_initial_state()
        state.apply_action(0)
        state2 = game.new_initial_state()
        state2.apply_actions([0] * game.num_players())
        self.assertEqual(state.history(), state2.history())

    def test_record_batched_trajectories(self):
        if False:
            return 10
        for game_name in ['kuhn_poker', 'leduc_poker', 'liars_dice']:
            game = pyspiel.load_game(game_name)
            python_policy = policy.TabularPolicy(game)
            tabular_policy = policy.python_policy_to_pyspiel_policy(python_policy)
            policies = [tabular_policy] * 2
            seed = 0
            batch_size = 128
            include_full_observations = False
            pyspiel.record_batched_trajectories(game, policies, python_policy.state_lookup, batch_size, include_full_observations, seed, -1)
if __name__ == '__main__':
    absltest.main()