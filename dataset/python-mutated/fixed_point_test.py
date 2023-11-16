"""Tests for Fixed Point."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.mfg.algorithms import fixed_point
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

class FixedPointTest(parameterized.TestCase):

    @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'), ('cpp', 'mfg_crowd_modelling'))
    def test_run(self, name):
        if False:
            i = 10
            return i + 15
        'Checks if the algorithm works.'
        game = pyspiel.load_game(name)
        fixed_p = fixed_point.FixedPoint(game)
        for _ in range(10):
            fixed_p.iteration()
        fixed_p_policy = fixed_p.get_policy()
        nash_conv_fixed_p = nash_conv.NashConv(game, fixed_p_policy)
        self.assertAlmostEqual(nash_conv_fixed_p.nash_conv(), 55.745, places=3)

    @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'), ('cpp', 'mfg_crowd_modelling'))
    def test_softmax(self, name):
        if False:
            return 10
        'Checks the softmax policy.'
        game = pyspiel.load_game(name)
        fixed_p = fixed_point.FixedPoint(game, temperature=10.0)
        for _ in range(10):
            fixed_p.iteration()
        fixed_p_policy = fixed_p.get_policy()
        nash_conv_fixed_p = nash_conv.NashConv(game, fixed_p_policy)
        self.assertAlmostEqual(nash_conv_fixed_p.nash_conv(), 2.421, places=3)
if __name__ == '__main__':
    absltest.main()