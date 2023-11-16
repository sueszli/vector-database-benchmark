"""Tests for open_spiel.python.algorithms.sample_some_states."""
from absl.testing import absltest
from open_spiel.python.algorithms import sample_some_states
import pyspiel

class SampleSomeStatesTest(absltest.TestCase):

    def test_sampling_in_simple_games(self):
        if False:
            while True:
                i = 10
        matrix_mp_num_states = 1 + 2 + 4
        game = pyspiel.load_game_as_turn_based('matrix_mp')
        for n in range(1, matrix_mp_num_states + 1):
            states = sample_some_states.sample_some_states(game, max_states=n)
            self.assertLen(states, n)
        states = sample_some_states.sample_some_states(game, max_states=1)
        self.assertLen(states, 1)
        states = sample_some_states.sample_some_states(game, max_states=matrix_mp_num_states + 1)
        self.assertLen(states, matrix_mp_num_states)
        coordinated_mp_num_states = 1 + 2 + 4 + 8
        game = pyspiel.load_game_as_turn_based('coordinated_mp')
        for n in range(1, coordinated_mp_num_states + 1):
            states = sample_some_states.sample_some_states(game, max_states=n)
            self.assertLen(states, n)
        states = sample_some_states.sample_some_states(game, max_states=coordinated_mp_num_states + 1)
        self.assertLen(states, coordinated_mp_num_states)
if __name__ == '__main__':
    absltest.main()