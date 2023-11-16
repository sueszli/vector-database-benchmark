"""Tests for open_spiel.python.algorithms.cfr."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import outcome_sampling_mccfr
import pyspiel
SEED = 39823987

class OutcomeSamplingMCCFRTest(absltest.TestCase):

    def test_outcome_sampling_leduc_2p(self):
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        game = pyspiel.load_game('leduc_poker')
        os_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
        for _ in range(10000):
            os_solver.iteration()
        conv = exploitability.nash_conv(game, os_solver.average_policy())
        print('Leduc2P, conv = {}'.format(conv))
        self.assertLess(conv, 3.07)

    def test_outcome_sampling_kuhn_2p(self):
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        game = pyspiel.load_game('kuhn_poker')
        os_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
        for _ in range(10000):
            os_solver.iteration()
        conv = exploitability.nash_conv(game, os_solver.average_policy())
        print('Kuhn2P, conv = {}'.format(conv))
        self.assertLess(conv, 0.17)
        tabular_policy = os_solver.average_policy().to_tabular()
        conv2 = exploitability.nash_conv(game, tabular_policy)
        self.assertEqual(conv, conv2)

    def test_outcome_sampling_kuhn_3p(self):
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        game = pyspiel.load_game('kuhn_poker', {'players': 3})
        os_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(game)
        for _ in range(10000):
            os_solver.iteration()
        conv = exploitability.nash_conv(game, os_solver.average_policy())
        print('Kuhn3P, conv = {}'.format(conv))
        self.assertLess(conv, 0.22)
if __name__ == '__main__':
    absltest.main()