"""Tests for softmax_policy."""
from absl.testing import absltest
from absl.testing import parameterized
from open_spiel.python import policy
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms import softmax_policy
from open_spiel.python.mfg.games import crowd_modelling
import pyspiel

class SoftmaxPolicyTest(parameterized.TestCase):

    @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'), ('cpp', 'mfg_crowd_modelling'))
    def test_softmax(self, name):
        if False:
            print('Hello World!')
        'Check if the softmax policy works as expected.\n\n    The test checks that:\n    - uniform prior policy gives the same results than no prior.\n    - very high temperature gives almost a uniform policy.\n    - very low temperature gives almost a deterministic policy for the best\n    action.\n\n    Args:\n      name: Name of the game.\n    '
        game = pyspiel.load_game(name)
        uniform_policy = policy.UniformRandomPolicy(game)
        dist = distribution.DistributionPolicy(game, uniform_policy)
        br_value = best_response_value.BestResponse(game, dist, value.TabularValueFunction(game))
        br_init_val = br_value(game.new_initial_state())
        softmax_pi_uniform_prior = softmax_policy.SoftmaxPolicy(game, None, 1.0, br_value, uniform_policy).to_tabular()
        softmax_pi_uniform_prior_value = policy_value.PolicyValue(game, dist, softmax_pi_uniform_prior, value.TabularValueFunction(game))
        softmax_pi_uniform_prior_init_val = softmax_pi_uniform_prior_value(game.new_initial_state())
        softmax_pi_no_prior = softmax_policy.SoftmaxPolicy(game, None, 1.0, br_value, None)
        softmax_pi_no_prior_value = policy_value.PolicyValue(game, dist, softmax_pi_no_prior, value.TabularValueFunction(game))
        softmax_pi_no_prior_init_val = softmax_pi_no_prior_value(game.new_initial_state())
        self.assertAlmostEqual(softmax_pi_uniform_prior_init_val, softmax_pi_no_prior_init_val)
        uniform_policy = uniform_policy.to_tabular()
        uniform_value = policy_value.PolicyValue(game, dist, uniform_policy, value.TabularValueFunction(game))
        uniform_init_val = uniform_value(game.new_initial_state())
        softmax_pi_no_prior = softmax_policy.SoftmaxPolicy(game, None, 100000000, br_value, None)
        softmax_pi_no_prior_value = policy_value.PolicyValue(game, dist, softmax_pi_no_prior, value.TabularValueFunction(game))
        softmax_pi_no_prior_init_val = softmax_pi_no_prior_value(game.new_initial_state())
        self.assertAlmostEqual(uniform_init_val, softmax_pi_no_prior_init_val)
        softmax_pi_no_prior = softmax_policy.SoftmaxPolicy(game, None, 0.0001, br_value, None)
        softmax_pi_no_prior_value = policy_value.PolicyValue(game, dist, softmax_pi_no_prior, value.TabularValueFunction(game))
        softmax_pi_no_prior_init_val = softmax_pi_no_prior_value(game.new_initial_state())
        self.assertAlmostEqual(br_init_val, softmax_pi_no_prior_init_val)
if __name__ == '__main__':
    absltest.main()