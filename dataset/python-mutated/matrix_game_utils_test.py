"""Tests the C++ matrix game utility methods exposed to Python."""
from absl.testing import absltest
from open_spiel.python.algorithms import lp_solver
import pyspiel

class MatrixGamesUtilsTest(absltest.TestCase):

    def test_num_deterministic_policies(self):
        if False:
            return 10
        game = pyspiel.load_game('kuhn_poker')
        self.assertEqual(pyspiel.num_deterministic_policies(game, 0), 64)
        self.assertEqual(pyspiel.num_deterministic_policies(game, 1), 64)
        game = pyspiel.load_game('leduc_poker')
        self.assertEqual(pyspiel.num_deterministic_policies(game, 0), -1)
        self.assertEqual(pyspiel.num_deterministic_policies(game, 1), -1)

    def test_extensive_to_matrix_game(self):
        if False:
            return 10
        kuhn_game = pyspiel.load_game('kuhn_poker')
        kuhn_matrix_game = pyspiel.extensive_to_matrix_game(kuhn_game)
        (unused_p0_strategy, unused_p1_strategy, p0_sol_val, p1_sol_val) = lp_solver.solve_zero_sum_matrix_game(kuhn_matrix_game)
        self.assertAlmostEqual(p0_sol_val, -1 / 18)
        self.assertAlmostEqual(p1_sol_val, +1 / 18)

    def test_extensive_to_matrix_game_type(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.extensive_to_matrix_game(pyspiel.load_game('kuhn_poker'))
        game_type = game.get_type()
        self.assertEqual(game_type.dynamics, pyspiel.GameType.Dynamics.SIMULTANEOUS)
        self.assertEqual(game_type.chance_mode, pyspiel.GameType.ChanceMode.DETERMINISTIC)
        self.assertEqual(game_type.information, pyspiel.GameType.Information.ONE_SHOT)
        self.assertEqual(game_type.utility, pyspiel.GameType.Utility.ZERO_SUM)

    def test_extensive_to_matrix_game_payoff_matrix(self):
        if False:
            while True:
                i = 10
        turn_based_game = pyspiel.load_game_as_turn_based('matrix_pd')
        matrix_game = pyspiel.extensive_to_matrix_game(turn_based_game)
        orig_game = pyspiel.load_matrix_game('matrix_pd')
        for row in range(orig_game.num_rows()):
            for col in range(orig_game.num_cols()):
                for player in range(2):
                    self.assertEqual(orig_game.player_utility(player, row, col), matrix_game.player_utility(player, row, col))
if __name__ == '__main__':
    absltest.main()