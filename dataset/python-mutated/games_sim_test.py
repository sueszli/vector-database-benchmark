"""Python spiel example."""
import pickle
from absl import app
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python import games
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.mfg import games as mfg_games
import pyspiel
from open_spiel.python.utils import file_utils
MAX_ACTIONS_PER_GAME = 1000
SPIEL_GAMES_LIST = pyspiel.registered_games()
SPIEL_LOADABLE_GAMES_LIST = [g for g in SPIEL_GAMES_LIST if g.default_loadable]
SPIEL_EXCLUDE_SIMS_TEST_GAMES_LIST = ['yacht']
assert len(SPIEL_LOADABLE_GAMES_LIST) >= 38, len(SPIEL_LOADABLE_GAMES_LIST)
SPIEL_SIMULTANEOUS_GAMES_LIST = [g for g in SPIEL_LOADABLE_GAMES_LIST if g.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS]
assert len(SPIEL_SIMULTANEOUS_GAMES_LIST) >= 14, len(SPIEL_SIMULTANEOUS_GAMES_LIST)
SPIEL_MULTIPLAYER_GAMES_LIST = [(g, p) for g in SPIEL_LOADABLE_GAMES_LIST for p in range(max(g.min_num_players, 2), 1 + min(g.max_num_players, 6)) if g.max_num_players > 2 and g.max_num_players > g.min_num_players and (g.short_name != 'tiny_hanabi') and (g.short_name != 'universal_poker') and (g.short_name != 'scotland_yard')]
assert len(SPIEL_MULTIPLAYER_GAMES_LIST) >= 35, len(SPIEL_MULTIPLAYER_GAMES_LIST)

class GamesSimTest(parameterized.TestCase):

    def apply_action(self, state, action):
        if False:
            return 10
        if state.is_simultaneous_node():
            assert isinstance(action, list)
            state.apply_actions(action)
        else:
            state.apply_action(action)

    def apply_action_test_clone(self, state, action):
        if False:
            i = 10
            return i + 15
        "Applies the action and tests the clone method if it's implemented."
        try:
            state_clone = state.clone()
        except Exception:
            self.apply_action(state, action)
            return
        self.assertEqual(str(state), str(state_clone))
        self.assertEqual(state.history(), state_clone.history())
        self.apply_action(state, action)
        self.apply_action(state_clone, action)
        self.assertEqual(str(state), str(state_clone))
        self.assertEqual(state.history(), state_clone.history())

    def serialize_deserialize(self, game, state, check_pyspiel_serialization, check_pickle_serialization):
        if False:
            for i in range(10):
                print('nop')
        if check_pyspiel_serialization:
            ser_str = pyspiel.serialize_game_and_state(game, state)
            (new_game, new_state) = pyspiel.deserialize_game_and_state(ser_str)
            self.assertEqual(str(game), str(new_game))
            self.assertEqual(str(state), str(new_state))
        if check_pickle_serialization:
            pickled_state = pickle.dumps(state)
            unpickled_state = pickle.loads(pickled_state)
            self.assertEqual(str(state), str(unpickled_state))

    def sim_game(self, game, check_pyspiel_serialization=True, check_pickle_serialization=True, make_distribution_fn=lambda states: [1 / len(states)] * len(states) if states else []):
        if False:
            print('Hello World!')
        min_utility = game.min_utility()
        max_utility = game.max_utility()
        self.assertLess(min_utility, max_utility)
        if check_pickle_serialization:
            pickled_game = pickle.dumps(game)
            unpickled_game = pickle.loads(pickled_game)
            self.assertEqual(str(game), str(unpickled_game))
            pickled_game_type = pickle.dumps(game.get_type())
            unpickled_game_type = pickle.loads(pickled_game_type)
            self.assertEqual(game.get_type(), unpickled_game_type)
        for state in game.new_initial_states():
            total_actions = 0
            next_serialize_check = 1
            while not state.is_terminal() and total_actions <= MAX_ACTIONS_PER_GAME:
                total_actions += 1
                if total_actions >= next_serialize_check:
                    self.serialize_deserialize(game, state, check_pyspiel_serialization, check_pickle_serialization)
                    next_serialize_check *= 2
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    self.assertNotEmpty(outcomes)
                    (action_list, prob_list) = zip(*outcomes)
                    action = np.random.choice(action_list, p=prob_list)
                    state.apply_action(action)
                elif state.is_simultaneous_node():
                    chosen_actions = []
                    for pid in range(game.num_players()):
                        legal_actions = state.legal_actions(pid)
                        action = 0 if not legal_actions else np.random.choice(legal_actions)
                        chosen_actions.append(action)
                    self.apply_action_test_clone(state, chosen_actions)
                elif state.is_mean_field_node():
                    self.assertEqual(game.get_type().dynamics, pyspiel.GameType.Dynamics.MEAN_FIELD)
                    state.update_distribution(make_distribution_fn(state.distribution_support()))
                else:
                    self.assertTrue(state.is_player_node())
                    action = np.random.choice(state.legal_actions(state.current_player()))
                    self.apply_action_test_clone(state, action)
            self.assertGreater(total_actions, 0, 'No actions taken in sim of ' + str(game))
            if state.is_terminal():
                self.assertEmpty(state.legal_actions())
                for player in range(game.num_players()):
                    self.assertEmpty(state.legal_actions(player))
                utilities = state.returns()
                for player in range(game.num_players()):
                    self.assertEqual(state.player_return(player), utilities[player])
                for utility in utilities:
                    self.assertGreaterEqual(utility, game.min_utility())
                    self.assertLessEqual(utility, game.max_utility())
                print('Sim of game {} terminated with {} total actions. Utilities: {}'.format(game, total_actions, utilities))
            else:
                print('Sim of game {} terminated after maximum number of actions {}'.format(game, MAX_ACTIONS_PER_GAME))

    @parameterized.named_parameters(((game_info.short_name, game_info) for game_info in SPIEL_LOADABLE_GAMES_LIST))
    def test_game_sim(self, game_info):
        if False:
            return 10
        if game_info.short_name in SPIEL_EXCLUDE_SIMS_TEST_GAMES_LIST:
            print(f'{game_info.short_name} is excluded from sim tests. Skipping.')
            return
        game = pyspiel.load_game(game_info.short_name)
        self.assertLessEqual(game_info.min_num_players, game.num_players())
        self.assertLessEqual(game.num_players(), game_info.max_num_players)
        self.sim_game(game)

    @parameterized.named_parameters(((game_info.short_name, game_info) for game_info in SPIEL_SIMULTANEOUS_GAMES_LIST))
    def test_simultaneous_game_as_turn_based(self, game_info):
        if False:
            for i in range(10):
                print('nop')
        converted_game = pyspiel.load_game_as_turn_based(game_info.short_name)
        self.sim_game(converted_game)

    @parameterized.named_parameters(((f'{p}p_{g.short_name}', g, p) for (g, p) in SPIEL_MULTIPLAYER_GAMES_LIST))
    def test_multiplayer_game(self, game_info, num_players):
        if False:
            while True:
                i = 10
        if game_info.short_name == 'python_mfg_predator_prey':
            reward_matrix = np.ones((num_players, num_players))
            zero_mat = np.zeros((5, 5))
            pop_1 = zero_mat.copy()
            pop_1[0, 0] = 1.0
            pop_1 = pop_1.tolist()
            pop_2 = zero_mat.copy()
            pop_2[0, -1] = 1.0
            pop_2 = pop_2.tolist()
            pop_3 = zero_mat.copy()
            pop_3[-1, 0] = 1.0
            pop_3 = pop_3.tolist()
            pop_4 = zero_mat.copy()
            pop_4[-1, -1] = 1.0
            pop_4 = pop_4.tolist()
            pops = [pop_1, pop_2, pop_3, pop_4]
            init_distrib = []
            for p in range(num_players):
                init_distrib += pops[p % 4]
            init_distrib = np.array(init_distrib)
            dict_args = {'players': num_players, 'reward_matrix': ' '.join((str(v) for v in reward_matrix.flatten())), 'init_distrib': ' '.join((str(v) for v in init_distrib.flatten()))}
        else:
            dict_args = {'players': num_players}
        game = pyspiel.load_game(game_info.short_name, dict_args)
        self.sim_game(game)

    def test_breakthrough(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('breakthrough(rows=6,columns=6)')
        self.sim_game(game)

    def test_pig(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('pig(players=2,winscore=15)')
        self.sim_game(game)

    def test_efg_game(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_efg_game(pyspiel.get_sample_efg_data())
        for _ in range(0, 100):
            self.sim_game(game, check_pyspiel_serialization=False, check_pickle_serialization=False)
        game = pyspiel.load_efg_game(pyspiel.get_kuhn_poker_efg_data())
        for _ in range(0, 100):
            self.sim_game(game, check_pyspiel_serialization=False, check_pickle_serialization=False)
        filename = file_utils.find_file('third_party/open_spiel/games/efg/sample.efg', 2)
        if filename is not None:
            game = pyspiel.load_game('efg_game(filename=' + filename + ')')
            for _ in range(0, 100):
                self.sim_game(game)
        filename = file_utils.find_file('third_party/open_spiel/games/efg/sample.efg', 2)
        if filename is not None:
            game = pyspiel.load_game('efg_game(filename=' + filename + ')')
            for _ in range(0, 100):
                self.sim_game(game)

    def test_backgammon_checker_moves(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('backgammon')
        state = game.new_initial_state()
        state.apply_action(0)
        action = state.legal_actions()[0]
        checker_moves = state.spiel_move_to_checker_moves(0, action)
        print('Checker moves:')
        for i in range(2):
            print('pos {}, num {}, hit? {}'.format(checker_moves[i].pos, checker_moves[i].num, checker_moves[i].hit))
        action2 = state.checker_moves_to_spiel_move(checker_moves)
        self.assertEqual(action, action2)
        action3 = state.translate_action(0, 0, True)
        self.assertEqual(action3, 0)

    def test_backgammon_checker_moves_with_hit_info(self):
        if False:
            return 10
        game = pyspiel.load_game('backgammon')
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                (action_list, prob_list) = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                legal_actions = state.legal_actions()
                player = state.current_player()
                for action in legal_actions:
                    action_str = state.action_to_string(player, action)
                    checker_moves = state.augment_with_hit_info(player, state.spiel_move_to_checker_moves(player, action))
                    if checker_moves[0].hit or checker_moves[1].hit:
                        self.assertGreaterEqual(action_str.find('*'), 0)
                    else:
                        self.assertLess(action_str.find('*'), 0)
                    if action_str.find('*') > 0:
                        self.assertTrue(checker_moves[0].hit or checker_moves[1].hit)
                    else:
                        self.assertTrue(not checker_moves[0].hit and (not checker_moves[1].hit))
                action = np.random.choice(legal_actions)
                state.apply_action(action)

    def test_leduc_get_and_set_private_cards(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('leduc_poker')
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(1)
        print(state)
        private_cards = state.get_private_cards()
        self.assertEqual(private_cards, [0, 1])
        state.set_private_cards([2, 3])
        print(state)
        private_cards = state.get_private_cards()
        self.assertEqual(private_cards, [2, 3])

    def test_dots_and_boxes_with_notation(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('dots_and_boxes')
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(1)
        dbn = state.dbn_string()
        self.assertEqual(dbn, '110000000000')

    @parameterized.parameters({'game_name': 'blotto'}, {'game_name': 'goofspiel'}, {'game_name': 'kuhn_poker'}, {'game_name': 'tiny_hanabi'}, {'game_name': 'phantom_ttt'}, {'game_name': 'matrix_rps'}, {'game_name': 'kuhn_poker'})
    def test_restricted_nash_response_test(self, game_name):
        if False:
            return 10
        rnr_game = pyspiel.load_game(f'restricted_nash_response(game={game_name}())')
        for _ in range(10):
            self.sim_game(rnr_game, check_pyspiel_serialization=False, check_pickle_serialization=False)

    @parameterized.parameters({'game_name': 'python_mfg_crowd_modelling'}, {'game_name': 'mfg_crowd_modelling'}, {'game_name': 'kuhn_poker'}, {'game_name': 'leduc_poker'})
    def test_has_at_least_an_action(self, game_name):
        if False:
            return 10
        "Check that all population's state have at least one action."
        game = pyspiel.load_game(game_name)
        to_string = lambda s: s.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
        states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False, include_mean_field_states=False, to_string=to_string)
        for state in states.values():
            self.assertNotEmpty(state.legal_actions())

def main(_):
    if False:
        print('Hello World!')
    absltest.main()
if __name__ == '__main__':
    app.run(main)