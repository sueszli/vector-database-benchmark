"""Test Python bindings for game transforms."""
from absl.testing import absltest
import pyspiel

class RepeatedGameTest(absltest.TestCase):

    def test_create_repeated_game(self):
        if False:
            i = 10
            return i + 15
        'Test both create_repeated_game function signatures.'
        repeated_game = pyspiel.create_repeated_game('matrix_rps', {'num_repetitions': 10})
        assert repeated_game.utility_sum() == 0
        state = repeated_game.new_initial_state()
        for _ in range(10):
            state.apply_actions([0, 0])
        assert state.is_terminal()
        stage_game = pyspiel.load_game('matrix_mp')
        repeated_game = pyspiel.create_repeated_game(stage_game, {'num_repetitions': 5})
        state = repeated_game.new_initial_state()
        for _ in range(5):
            state.apply_actions([0, 0])
        assert state.is_terminal()
        stage_game = pyspiel.load_game('matrix_pd')
        repeated_game = pyspiel.create_repeated_game(stage_game, {'num_repetitions': 5})
        assert repeated_game.utility_sum() is None
if __name__ == '__main__':
    absltest.main()