"""Tests the C++ nfg_game methods exposed to Python."""
from absl.testing import absltest
import pyspiel

class NFGGameTest(absltest.TestCase):

    def test_pd(self):
        if False:
            i = 10
            return i + 15
        pd_nfg_string = 'NFG 1 R "OpenSpiel export of matrix_pd()"\n{ "Player 0" "Player 1" } { 2 2 }\n\n5 5\n10 0\n0 10\n1 1\n'
        game = pyspiel.load_nfg_game(pd_nfg_string)
        self.assertEqual(game.player_utility(0, 0, 0), 5)
        self.assertEqual(game.player_utility(0, 1, 0), 10)
        self.assertEqual(game.player_utility(0, 0, 1), 0)
        self.assertEqual(game.player_utility(0, 1, 1), 1)
        self.assertEqual(game.player_utility(1, 0, 0), 5)
        self.assertEqual(game.player_utility(1, 1, 0), 0)
        self.assertEqual(game.player_utility(1, 0, 1), 10)
        self.assertEqual(game.player_utility(1, 1, 1), 1)

    def test_native_export_import(self):
        if False:
            return 10
        "Check that we can import games that we've exported.\n\n    We do not do any additional checking here, as these methods are already\n    being extensively tested in nfg_test.cc. The purpose of this test is only\n    to check that the python wrapping works.\n    "
        game_strings = ['matrix_rps', 'matrix_shapleys_game', 'matrix_pd', 'matrix_sh', 'blotto(players=2,coins=5,fields=3)', 'blotto(players=3,coins=5,fields=3)']
        for game_string in game_strings:
            game = pyspiel.load_game(game_string)
            nfg_text = pyspiel.game_to_nfg_string(game)
            nfg_game = pyspiel.load_nfg_game(nfg_text)
            self.assertIsNotNone(nfg_game)
if __name__ == '__main__':
    absltest.main()