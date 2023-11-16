"""Tests for LP solvers."""
from absl.testing import absltest
from open_spiel.python.algorithms import sequence_form_lp
import pyspiel

class SFLPTest(absltest.TestCase):

    def test_rock_paper_scissors(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game_as_turn_based('matrix_rps')
        (val1, val2, _, _) = sequence_form_lp.solve_zero_sum_game(game)
        self.assertAlmostEqual(val1, 0)
        self.assertAlmostEqual(val2, 0)

    def test_kuhn_poker(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('kuhn_poker')
        (val1, val2, _, _) = sequence_form_lp.solve_zero_sum_game(game)
        self.assertAlmostEqual(val1, -1 / 18)
        self.assertAlmostEqual(val2, +1 / 18)

    def test_kuhn_poker_efg(self):
        if False:
            return 10
        game = pyspiel.load_efg_game(pyspiel.get_kuhn_poker_efg_data())
        (val1, val2, _, _) = sequence_form_lp.solve_zero_sum_game(game)
        self.assertAlmostEqual(val1, -1 / 18)
        self.assertAlmostEqual(val2, +1 / 18)

    def test_leduc_poker(self):
        if False:
            return 10
        game = pyspiel.load_game('leduc_poker')
        (val1, val2, _, _) = sequence_form_lp.solve_zero_sum_game(game)
        self.assertAlmostEqual(val1, -0.085606424078, places=6)
        self.assertAlmostEqual(val2, 0.085606424078, places=6)

    def test_iigoofspiel4(self):
        if False:
            for i in range(10):
                print('nop')
        game = pyspiel.load_game_as_turn_based('goofspiel', {'imp_info': True, 'num_cards': 4, 'points_order': 'descending'})
        (val1, val2, _, _) = sequence_form_lp.solve_zero_sum_game(game)
        self.assertAlmostEqual(val1, 0)
        self.assertAlmostEqual(val2, 0)
if __name__ == '__main__':
    absltest.main()