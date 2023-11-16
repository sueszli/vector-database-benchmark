"""Implementation of Fictitious Play from Perrin & al.

Reference: https://arxiv.org/abs/2007.03458.
As presented, the Fictitious Play algorithm provides a robust approximation
scheme for Nash equilibrium by iteratively computing the best response
against the distribution induced by the average of the past best responses.
The provided formulation of Deep Fictitious Play mirrors this procedure,
but substitutes out the exact best reponse computation with an approximation
of best response values through a Reinforcement Learning approach (where
the RL method in question is a user-determined parameter for each iteration).

Policy is initialized to uniform policy.
Each iteration:
 1. Compute best response against policy
 2. Update policy as weighted average of best response and current policy
    (default learning rate is 1 / num_iterations + 1).

To use fictitious play one should initialize it and run multiple iterations:
fp = FictitiousPlay(game)
for _ in range(num_iterations):
  fp.iteration()
policy = fp.get_policy()
"""
import math
from typing import List, Optional
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import distribution as distribution_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.algorithms import softmax_policy
import pyspiel

class MergedPolicy(policy_std.Policy):
    """Merge several policies."""

    def __init__(self, game, player_ids: List[int], policies: List[policy_std.Policy], distributions: List[distribution_std.Distribution], weights: List[float]):
        if False:
            while True:
                i = 10
        'Initializes the merged policy.\n\n    Args:\n      game: The game to analyze.\n      player_ids: list of player ids for which this policy applies; each should\n        be in the range 0..game.num_players()-1.\n      policies: A `List[policy_std.Policy]` object.\n      distributions: A `List[distribution_std.Distribution]` object.\n      weights: A `List[float]` object. The elements should sum to 1.\n    '
        super().__init__(game, player_ids)
        self._policies = policies
        self._distributions = distributions
        self._weights = weights
        assert len(policies) == len(distributions), f'Length mismatch {len(policies)} != {len(distributions)}'
        assert len(policies) == len(weights), f'Length mismatch {len(policies)} != {len(weights)}'
        assert math.isclose(sum(weights), 1.0), f'Weights should sum to 1, but instead sum to {sum(weights)}'

    def action_probabilities(self, state, player_id=None):
        if False:
            print('Hello World!')
        action_prob = []
        legal = state.legal_actions()
        num_legal = len(legal)
        for a in legal:
            merged_pi = 0.0
            norm_merged_pi = 0.0
            for (p, d, w) in zip(self._policies, self._distributions, self._weights):
                try:
                    merged_pi += w * d(state) * p(state)[a]
                    norm_merged_pi += w * d(state)
                except (KeyError, ValueError):
                    pass
            if norm_merged_pi > 0.0:
                action_prob.append((a, merged_pi / norm_merged_pi))
            else:
                action_prob.append((a, 1.0 / num_legal))
        return dict(action_prob)

class FictitiousPlay(object):
    """Computes the value of a specified strategy."""

    def __init__(self, game: pyspiel.Game, lr: Optional[float]=None, temperature: Optional[float]=None):
        if False:
            return 10
        'Initializes the greedy policy.\n\n    Args:\n      game: The game to analyze.\n      lr: The learning rate of mirror descent. If None, at iteration i it will\n        be set to 1/i.\n      temperature: If set, then instead of the greedy policy a softmax policy\n        with the specified temperature will be used to update the policy at each\n        iteration.\n    '
        self._game = game
        self._lr = lr
        self._temperature = temperature
        self._policy = policy_std.UniformRandomPolicy(self._game)
        self._fp_step = 0

    def get_policy(self):
        if False:
            while True:
                i = 10
        return self._policy

    def iteration(self, br_policy=None, learning_rate=None):
        if False:
            while True:
                i = 10
        'Returns a new `TabularPolicy` equivalent to this policy.\n\n    Args:\n      br_policy: Policy to compute the best response value for each iteration.\n        If none provided, the exact value is computed.\n      learning_rate: The learning rate.\n    '
        self._fp_step += 1
        distrib = distribution.DistributionPolicy(self._game, self._policy)
        if br_policy:
            br_value = policy_value.PolicyValue(self._game, distrib, br_policy)
        else:
            br_value = best_response_value.BestResponse(self._game, distrib, value.TabularValueFunction(self._game))
        player_ids = list(range(self._game.num_players()))
        if self._temperature is None:
            pi = greedy_policy.GreedyPolicy(self._game, player_ids, br_value)
        else:
            pi = softmax_policy.SoftmaxPolicy(self._game, player_ids, self._temperature, br_value)
        pi = pi.to_tabular()
        distrib_pi = distribution.DistributionPolicy(self._game, pi)
        if learning_rate:
            weight = learning_rate
        else:
            weight = self._lr if self._lr else 1.0 / (self._fp_step + 1)
        if math.isclose(weight, 1.0):
            self._policy = pi
        else:
            self._policy = MergedPolicy(self._game, player_ids, [self._policy, pi], [distrib, distrib_pi], [1.0 - weight, weight]).to_tabular()