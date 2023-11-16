"""Implementation of Nash Conv metric for a policy.

In the context of mean field games, the Nash Conv is the difference between:
- the value of a policy against the distribution of that policy,
- and the best response against the distribution of the policy.
"""
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import policy_value

class NashConv(object):
    """Computes the Nash Conv of a policy."""

    def __init__(self, game, policy: policy_std.Policy, root_state=None):
        if False:
            print('Hello World!')
        'Initializes the nash conv.\n\n    Args:\n      game: The game to analyze.\n      policy: A `policy.Policy` object.\n      root_state: The state of the game at which to start. If `None`, the game\n        root state is used.\n    '
        self._game = game
        self._policy = policy
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
        self._distrib = distribution.DistributionPolicy(self._game, self._policy, root_state=root_state)
        self._pi_value = policy_value.PolicyValue(self._game, self._distrib, self._policy, value.TabularValueFunction(self._game), root_state=root_state)
        self._br_value = best_response_value.BestResponse(self._game, self._distrib, value.TabularValueFunction(self._game), root_state=root_state)

    def nash_conv(self):
        if False:
            while True:
                i = 10
        'Returns the nash conv.\n\n    Returns:\n      A float representing the nash conv for the policy.\n    '
        return sum([self._br_value.eval_state(state) - self._pi_value.eval_state(state) for state in self._root_states])

    def br_values(self):
        if False:
            return 10
        'Returns the best response values to the policy distribution.\n\n    Returns:\n      A List[float] representing the best response values for a policy\n        distribution.\n    '
        return [self._br_value.eval_state(state) for state in self._root_states]

    @property
    def distribution(self) -> distribution.DistributionPolicy:
        if False:
            print('Hello World!')
        return self._distrib