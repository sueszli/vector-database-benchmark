"""Tests for Munchausen Online Mirror Descent."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import munchausen_mirror_descent
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

class MunchausenMirrorDescentTest(parameterized.TestCase):

    @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'), ('cpp', 'mfg_crowd_modelling'))
    def test_run(self, name):
        if False:
            i = 10
            return i + 15
        'Checks if the algorithm works.'
        game = pyspiel.load_game(name)
        md = munchausen_mirror_descent.MunchausenMirrorDescent(game, value.TabularValueFunction(game))
        for _ in range(10):
            md.iteration()
        md_policy = md.get_policy()
        nash_conv_md = nash_conv.NashConv(game, md_policy)
        self.assertAlmostEqual(nash_conv_md.nash_conv(), 2.27366, places=5)
if __name__ == '__main__':
    absltest.main()