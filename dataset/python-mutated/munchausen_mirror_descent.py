"""Munchausen Online Mirror Descent."""
from typing import Dict, List, Optional
import numpy as np
from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import mirror_descent
import pyspiel

class ProjectedPolicyMunchausen(mirror_descent.ProjectedPolicy):
    """Project values on the policy simplex."""

    def __init__(self, game: pyspiel.Game, player_ids: List[int], state_value: value.ValueFunction, learning_rate: float, policy: policy_lib.Policy):
        if False:
            return 10
        'Initializes the projected policy.\n\n    Args:\n      game: The game to analyze.\n      player_ids: list of player ids for which this policy applies; each should\n        be in the range 0..game.num_players()-1.\n      state_value: The state value to project.\n      learning_rate: The learning rate.\n      policy: The policy to project.\n    '
        super().__init__(game, player_ids, state_value)
        self._learning_rate = learning_rate
        self._policy = policy

    def action_probabilities(self, state: pyspiel.State, player_id: Optional[int]=None) -> Dict[int, float]:
        if False:
            while True:
                i = 10
        del player_id
        action_logit = [(a, self._learning_rate * self.value(state, action=a) + np.log(p)) for (a, p) in self._policy.action_probabilities(state).items()]
        (action, logit) = zip(*action_logit)
        return dict(zip(action, mirror_descent.softmax_projection(logit)))

class MunchausenMirrorDescent(mirror_descent.MirrorDescent):
    """Munchausen Online Mirror Descent algorithm.

  This algorithm is equivalent to the online mirror descent algorithm but
  instead of summing value functions, it directly computes the cumulative
  Q-function using a penalty with respect to the previous policy.
  """

    def eval_state(self, state: pyspiel.State, learning_rate: float):
        if False:
            i = 10
            return i + 15
        'Evaluate the value of a state.'
        state_str = state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)
        if self._state_value.has(state_str):
            return self._state_value(state_str)
        v = self.get_state_value(state, learning_rate)
        self._state_value.set_value(state_str, v)
        return v

    def get_projected_policy(self) -> policy_lib.Policy:
        if False:
            while True:
                i = 10
        'Returns the projected policy.'
        return ProjectedPolicyMunchausen(self._game, list(range(self._game.num_players())), self._state_value, self._lr, self._policy)