"""Tests for Boltzmann Policy Iteration."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import boltzmann_policy_iteration
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

class BoltzmannPolicyIterationTest(parameterized.TestCase):

    @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'), ('cpp', 'mfg_crowd_modelling'))
    def test_run(self, name):
        if False:
            i = 10
            return i + 15
        'Checks if the algorithm works.'
        game = pyspiel.load_game(name)
        bpi = boltzmann_policy_iteration.BoltzmannPolicyIteration(game, value.TabularValueFunction(game))
        for _ in range(10):
            bpi.iteration()
        bpi_policy = bpi.get_policy()
        nash_conv_bpi = nash_conv.NashConv(game, bpi_policy)
        self.assertAlmostEqual(nash_conv_bpi.nash_conv(), 2.75428, places=5)
if __name__ == '__main__':
    absltest.main()