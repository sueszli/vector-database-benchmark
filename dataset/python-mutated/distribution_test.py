"""Tests for distribution."""
from absl.testing import absltest
from open_spiel.python import policy
from open_spiel.python.mfg import games
from open_spiel.python.mfg.algorithms import distribution
import pyspiel

class DistributionTest(absltest.TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        game = pyspiel.load_game('python_mfg_crowd_modelling')
        uniform_policy = policy.UniformRandomPolicy(game)
        dist = distribution.DistributionPolicy(game, uniform_policy)
        state = game.new_initial_state().child(0)
        self.assertAlmostEqual(dist.value(state), 1 / game.size)

    def test_state_support_outside_distrib(self):
        if False:
            print('Hello World!')
        game = pyspiel.load_game('mfg_crowd_modelling_2d', {'initial_distribution': '[0|0]', 'initial_distribution_value': '[1.]'})
        uniform_policy = policy.UniformRandomPolicy(game)
        _ = distribution.DistributionPolicy(game, uniform_policy)

    def test_multi_pop(self):
        if False:
            while True:
                i = 10
        game = pyspiel.load_game('python_mfg_predator_prey')
        self.assertEqual(game.num_players(), 3)
        uniform_policy = policy.UniformRandomPolicy(game)
        dist = distribution.DistributionPolicy(game, uniform_policy)
        for pop in range(3):
            self.assertAlmostEqual(dist.value(game.new_initial_state_for_population(pop)), 1.0)
if __name__ == '__main__':
    absltest.main()