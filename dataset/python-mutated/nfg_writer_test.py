"""Tests the C++ nfg_writer methods exposed to Python."""
from absl.testing import absltest
import pyspiel

class NFGWriterTest(absltest.TestCase):

    def test_rps(self):
        if False:
            return 10
        expected_rps_nfg = 'NFG 1 R "OpenSpiel export of matrix_rps()"\n{ "Player 0" "Player 1" } { 3 3 }\n\n0 0\n1 -1\n-1 1\n-1 1\n0 0\n1 -1\n1 -1\n-1 1\n0 0\n'
        game = pyspiel.load_game('matrix_rps')
        nfg_text = pyspiel.game_to_nfg_string(game)
        self.assertEqual(nfg_text, expected_rps_nfg)

    def test_pd(self):
        if False:
            for i in range(10):
                print('nop')
        expected_pd_nfg = 'NFG 1 R "OpenSpiel export of matrix_pd()"\n{ "Player 0" "Player 1" } { 2 2 }\n\n5 5\n10 0\n0 10\n1 1\n'
        game = pyspiel.load_game('matrix_pd')
        nfg_text = pyspiel.game_to_nfg_string(game)
        self.assertEqual(nfg_text, expected_pd_nfg)

    def test_mp3p(self):
        if False:
            while True:
                i = 10
        expected_mp3p_nfg = 'NFG 1 R "OpenSpiel export of matching_pennies_3p()"\n{ "Player 0" "Player 1" "Player 2" } { 2 2 2 }\n\n1 1 -1\n-1 1 1\n-1 -1 -1\n1 -1 1\n1 -1 1\n-1 -1 -1\n-1 1 1\n1 1 -1\n'
        game = pyspiel.load_game('matching_pennies_3p')
        nfg_text = pyspiel.game_to_nfg_string(game)
        self.assertEqual(nfg_text, expected_mp3p_nfg)
if __name__ == '__main__':
    absltest.main()