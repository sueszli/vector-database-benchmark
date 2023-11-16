"""Unit test for XinXin MCTS bot."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import evaluate_bots
import pyspiel
SEED = 12983641

class ISMCTSBotTest(absltest.TestCase):

    def xinxin_play_game(self, game):
        if False:
            print('Hello World!')
        bots = []
        for _ in range(4):
            bots.append(pyspiel.make_xinxin_bot(game.get_parameters()))
        evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)

    def test_basic_xinxin_selfplay(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('hearts')
        self.xinxin_play_game(game)
if __name__ == '__main__':
    np.random.seed(SEED)
    absltest.main()