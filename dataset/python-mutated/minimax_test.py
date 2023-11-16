"""Tests for open_spiel.python.algorithms.minimax."""
from absl.testing import absltest
from open_spiel.python.algorithms import minimax
import pyspiel

class MinimaxTest(absltest.TestCase):

    def test_compute_game_value(self):
        if False:
            for i in range(10):
                print('nop')
        tic_tac_toe = pyspiel.load_game('tic_tac_toe')
        (game_score, _) = minimax.alpha_beta_search(tic_tac_toe)
        self.assertEqual(0.0, game_score)

    def test_compute_game_value_with_evaluation_function(self):
        if False:
            for i in range(10):
                print('nop')
        tic_tac_toe = pyspiel.load_game('tic_tac_toe')
        (game_score, _) = minimax.alpha_beta_search(tic_tac_toe, value_function=lambda x: 0, maximum_depth=1)
        self.assertEqual(0.0, game_score)

    def test_win(self):
        if False:
            i = 10
            return i + 15
        tic_tac_toe = pyspiel.load_game('tic_tac_toe')
        state = tic_tac_toe.new_initial_state()
        state.apply_action(4)
        state.apply_action(1)
        (game_score, _) = minimax.alpha_beta_search(tic_tac_toe, state=state)
        self.assertEqual(1.0, game_score)

    def test_loss(self):
        if False:
            for i in range(10):
                print('nop')
        tic_tac_toe = pyspiel.load_game('tic_tac_toe')
        state = tic_tac_toe.new_initial_state()
        state.apply_action(5)
        state.apply_action(4)
        state.apply_action(3)
        state.apply_action(8)
        (game_score, _) = minimax.alpha_beta_search(tic_tac_toe, state=state)
        self.assertEqual(-1.0, game_score)
if __name__ == '__main__':
    absltest.main()