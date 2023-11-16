"""Tests for open_spiel.python.algorithms.cfr."""
from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr
import pyspiel
SEED = 39823987

class ExternalSamplingMCCFRTest(absltest.TestCase):

    def test_external_sampling_leduc_2p_simple(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(SEED)
        game = pyspiel.load_game('leduc_poker')
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.SIMPLE)
        for _ in range(10):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print('Leduc2P, conv = {}'.format(conv))
        self.assertLess(conv, 5)
        tabular_policy = es_solver.average_policy().to_tabular()
        conv2 = exploitability.nash_conv(game, tabular_policy)
        self.assertEqual(conv, conv2)

    def test_external_sampling_leduc_2p_full(self):
        if False:
            while True:
                i = 10
        np.random.seed(SEED)
        game = pyspiel.load_game('leduc_poker')
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.FULL)
        for _ in range(10):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print('Leduc2P, conv = {}'.format(conv))
        self.assertLess(conv, 5)

    def test_external_sampling_kuhn_2p_simple(self):
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        game = pyspiel.load_game('kuhn_poker')
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.SIMPLE)
        for _ in range(10):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print('Kuhn2P, conv = {}'.format(conv))
        self.assertLess(conv, 1)

    def test_external_sampling_kuhn_2p_full(self):
        if False:
            print('Hello World!')
        np.random.seed(SEED)
        game = pyspiel.load_game('kuhn_poker')
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.FULL)
        for _ in range(10):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print('Kuhn2P, conv = {}'.format(conv))
        self.assertLess(conv, 1)

    def disabled_test_external_sampling_liars_dice_2p_simple(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(SEED)
        game = pyspiel.load_game('liars_dice')
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.SIMPLE)
        for _ in range(1):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print("Liar's dice, conv = {}".format(conv))
        self.assertLess(conv, 2)

    def test_external_sampling_kuhn_3p_simple(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(SEED)
        game = pyspiel.load_game('kuhn_poker', {'players': 3})
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.SIMPLE)
        for _ in range(10):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print('Kuhn3P, conv = {}'.format(conv))
        self.assertLess(conv, 2)

    def test_external_sampling_kuhn_3p_full(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(SEED)
        game = pyspiel.load_game('kuhn_poker', {'players': 3})
        es_solver = external_sampling_mccfr.ExternalSamplingSolver(game, external_sampling_mccfr.AverageType.FULL)
        for _ in range(10):
            es_solver.iteration()
        conv = exploitability.nash_conv(game, es_solver.average_policy())
        print('Kuhn3P, conv = {}'.format(conv))
        self.assertLess(conv, 2)
if __name__ == '__main__':
    absltest.main()