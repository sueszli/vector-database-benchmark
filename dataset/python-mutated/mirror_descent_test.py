"""Tests for mirror descent."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

class MirrorDescentTest(parameterized.TestCase):

    @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'), ('cpp', 'mfg_crowd_modelling'))
    def test_fp(self, name):
        if False:
            print('Hello World!')
        'Checks if mirror descent works.'
        game = pyspiel.load_game(name)
        md = mirror_descent.MirrorDescent(game, value.TabularValueFunction(game))
        for _ in range(10):
            md.iteration()
        md_policy = md.get_policy()
        nash_conv_md = nash_conv.NashConv(game, md_policy)
        self.assertAlmostEqual(nash_conv_md.nash_conv(), 2.2730324915546056)
if __name__ == '__main__':
    absltest.main()