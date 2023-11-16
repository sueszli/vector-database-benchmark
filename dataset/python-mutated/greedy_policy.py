"""Computes a greedy policy from a value."""
import numpy as np
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import value

class GreedyPolicy(policy_std.Policy):
    """Computes the greedy policy of a value."""

    def __init__(self, game, player_ids, state_action_value: value.ValueFunction):
        if False:
            print('Hello World!')
        'Initializes the greedy policy.\n\n    Args:\n      game: The game to analyze.\n      player_ids: list of player ids for which this policy applies; each should\n        be in the range 0..game.num_players()-1.\n      state_action_value: A state-action value function.\n    '
        super(GreedyPolicy, self).__init__(game, player_ids)
        self._state_action_value = state_action_value

    def action_probabilities(self, state, player_id=None):
        if False:
            while True:
                i = 10
        q = [self._state_action_value(state, action) for action in state.legal_actions()]
        amax_q = [0.0 for _ in state.legal_actions()]
        amax_q[np.argmax(q)] = 1.0
        return dict(zip(state.legal_actions(), amax_q))