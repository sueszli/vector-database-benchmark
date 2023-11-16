"""Computes a softmax policy from a value function."""
from typing import Optional
import numpy as np
from open_spiel.python import policy
from open_spiel.python.mfg import value

class SoftmaxPolicy(policy.Policy):
    """Computes the softmax policy of a value function."""

    def __init__(self, game, player_ids, temperature: float, state_action_value: value.ValueFunction, prior_policy: Optional[policy.Policy]=None):
        if False:
            print('Hello World!')
        'Initializes the softmax policy.\n\n    Args:\n      game: The game to analyze.\n      player_ids: list of player ids for which this policy applies; each\n        should be in the range 0..game.num_players()-1.\n      temperature: float to scale the values (multiplied by 1/temperature).\n      state_action_value: A state-action value function.\n      prior_policy: Optional argument. Prior policy to scale the softmax\n        policy.\n    '
        super(SoftmaxPolicy, self).__init__(game, player_ids)
        self._state_action_value = state_action_value
        self._prior_policy = prior_policy
        self._temperature = temperature

    def action_probabilities(self, state, player_id=None):
        if False:
            return 10
        legal_actions = state.legal_actions()
        max_q = np.max([self._state_action_value(state, action) for action in legal_actions])
        exp_q = [np.exp((self._state_action_value(state, action) - max_q) / self._temperature) for action in legal_actions]
        if self._prior_policy is not None:
            prior_probs = self._prior_policy.action_probabilities(state)
            exp_q = [prior_probs.get(action, 0) * exp_q[i] for (i, action) in enumerate(legal_actions)]
        denom = sum(exp_q)
        smax_q = exp_q if denom == 0 else exp_q / denom
        return dict(zip(legal_actions, smax_q))