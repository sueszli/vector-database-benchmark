"""Tests for open_spiel.python.algorithms.get_all_states."""
from absl.testing import absltest
from open_spiel.python.algorithms import get_all_states
import pyspiel

class GetAllStatesTest(absltest.TestCase):

    def test_tic_tac_toe_number_histories(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('tic_tac_toe')
        states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False, to_string=lambda s: s.history_str())
        self.assertLen(states, 549946)
        states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False, to_string=str)
        self.assertLen(states, 5478)

    def test_simultaneous_python_game_get_all_state(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('python_iterated_prisoners_dilemma(max_game_length=6)')
        states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False, to_string=lambda s: s.history_str())
        self.assertLen(states, 10921)
        states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False, to_string=str)
        self.assertLen(states, 5461)

    def test_simultaneous_game_get_all_state(self):
        if False:
            i = 10
            return i + 15
        game = game = pyspiel.load_game('goofspiel', {'num_cards': 3})
        states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False, to_string=lambda s: s.history_str())
        self.assertLen(states, 273)
if __name__ == '__main__':
    absltest.main()