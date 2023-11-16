"""Tests for nash conv."""
from absl.testing import absltest
from open_spiel.python import policy
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

class BestResponseTest(absltest.TestCase):

    def test_python_game(self):
        if False:
            return 10
        'Checks if the NashConv is consistent through time.'
        game = crowd_modelling.MFGCrowdModellingGame()
        uniform_policy = policy.UniformRandomPolicy(game)
        nash_conv_fp = nash_conv.NashConv(game, uniform_policy)
        self.assertAlmostEqual(nash_conv_fp.nash_conv(), 2.8135365543870385)

    def test_cpp_game(self):
        if False:
            print('Hello World!')
        'Checks if the NashConv is consistent through time.'
        game = pyspiel.load_game('mfg_crowd_modelling')
        uniform_policy = policy.UniformRandomPolicy(game)
        nash_conv_fp = nash_conv.NashConv(game, uniform_policy)
        self.assertAlmostEqual(nash_conv_fp.nash_conv(), 2.8135365543870385)
if __name__ == '__main__':
    absltest.main()