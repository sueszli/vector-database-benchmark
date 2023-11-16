"""Tests for third_party.open_spiel.python.observation."""
import collections
import random
import time
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.observation import INFO_STATE_OBS_TYPE
from open_spiel.python.observation import make_observation
import pyspiel

class ObservationTest(absltest.TestCase):

    def test_leduc_observation(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('leduc_poker')
        observation = make_observation(game)
        state = game.new_initial_state()
        state.apply_action(1)
        state.apply_action(2)
        state.apply_action(2)
        state.apply_action(1)
        state.apply_action(3)
        observation.set_from(state, player=0)
        np.testing.assert_array_equal(observation.tensor, [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 3])
        self.assertEqual(list(observation.dict), ['player', 'private_card', 'community_card', 'pot_contribution'])
        np.testing.assert_array_equal(observation.dict['player'], [1, 0])
        np.testing.assert_array_equal(observation.dict['private_card'], [0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(observation.dict['community_card'], [0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(observation.dict['pot_contribution'], [3, 3])
        self.assertEqual(observation.string_from(state, 0), '[Observer: 0][Private: 1][Round 2][Player: 0][Pot: 6][Money: 97 97][Public: 3][Ante: 3 3]')

    def test_leduc_info_state(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('leduc_poker')
        observation = make_observation(game, INFO_STATE_OBS_TYPE)
        state = game.new_initial_state()
        state.apply_action(1)
        state.apply_action(2)
        state.apply_action(2)
        state.apply_action(1)
        state.apply_action(3)
        observation.set_from(state, player=0)
        np.testing.assert_array_equal(observation.tensor, [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(observation.dict), ['player', 'private_card', 'community_card', 'betting'])
        np.testing.assert_array_equal(observation.dict['player'], [1, 0])
        np.testing.assert_array_equal(observation.dict['private_card'], [0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(observation.dict['community_card'], [0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(observation.dict['betting'], [[[0, 1], [1, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0]]])
        self.assertEqual(observation.string_from(state, 0), '[Observer: 0][Private: 1][Round 2][Player: 0][Pot: 6][Money: 97 97][Public: 3][Round1: 2 1][Round2: ]')

    def test_leduc_info_state_as_single_tensor(self):
        if False:
            return 10
        game = pyspiel.load_game('leduc_poker')
        observation = make_observation(game, INFO_STATE_OBS_TYPE, pyspiel.game_parameters_from_string('single_tensor'))
        state = game.new_initial_state()
        state.apply_action(1)
        state.apply_action(2)
        state.apply_action(2)
        state.apply_action(1)
        state.apply_action(3)
        observation.set_from(state, player=0)
        self.assertEqual(list(observation.dict), ['info_state'])
        np.testing.assert_array_equal(observation.dict['info_state'], [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_leduc_all_player_privates(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game('leduc_poker')
        observation = make_observation(game, pyspiel.IIGObservationType(perfect_recall=True, private_info=pyspiel.PrivateInfoType.ALL_PLAYERS))
        state = game.new_initial_state()
        state.apply_action(1)
        state.apply_action(2)
        state.apply_action(2)
        state.apply_action(1)
        state.apply_action(3)
        observation.set_from(state, player=0)
        np.testing.assert_array_equal(observation.dict['private_cards'], [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

    def test_benchmark_state_generation(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('chess')
        trajectories = []
        for _ in range(20):
            state = game.new_initial_state()
            while not state.is_terminal():
                state.apply_action(random.choice(state.legal_actions()))
            trajectories.append(state.history())
        total = 0
        observation = make_observation(game)
        start = time.time()
        for trajectory in trajectories:
            state = game.new_initial_state()
            for action in trajectory:
                state.apply_action(action)
                observation.set_from(state, 0)
                total += np.mean(observation.tensor)
        end = time.time()
        print(f'New API time per iteration {1000 * (end - start) / len(trajectories)}msec')
        total = 0
        start = time.time()
        for trajectory in trajectories:
            state = game.new_initial_state()
            for action in trajectory:
                state.apply_action(action)
                obs = state.observation_tensor(0)
                tensor = np.asarray(obs)
                total += np.mean(tensor)
        end = time.time()
        print(f'Old API time per iteration {1000 * (end - start) / len(trajectories)}msec')

    def test_compression_binary(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('leduc_poker')
        obs1 = make_observation(game, INFO_STATE_OBS_TYPE)
        obs2 = make_observation(game, INFO_STATE_OBS_TYPE)
        self.assertLen(obs1.tensor, 30)
        for state in get_all_states.get_all_states(game).values():
            for player in range(game.num_players()):
                obs1.set_from(state, player)
                compressed = obs1.compress()
                self.assertEqual(type(compressed), bytes)
                self.assertLen(compressed, 5)
                obs2.decompress(compressed)
                np.testing.assert_array_equal(obs1.tensor, obs2.tensor)

    def test_compression_none(self):
        if False:
            return 10
        game = pyspiel.load_game('leduc_poker')
        obs1 = make_observation(game)
        obs2 = make_observation(game)
        self.assertLen(obs1.tensor, 16)
        freq = collections.Counter()
        for state in get_all_states.get_all_states(game).values():
            for player in range(game.num_players()):
                obs1.set_from(state, player)
                compressed = obs1.compress()
                self.assertEqual(type(compressed), bytes)
                freq[len(compressed)] += 1
                obs2.decompress(compressed)
                np.testing.assert_array_equal(obs1.tensor, obs2.tensor)
        expected_freq = {3: 840, 65: 17760}
        self.assertEqual(freq, expected_freq)
if __name__ == '__main__':
    absltest.main()