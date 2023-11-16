"""Fixed Point."""
from typing import Optional
from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import softmax_policy
import pyspiel

class FixedPoint(object):
    """The fixed point algorithm.

  This algorithm is based on Banach-Picard iterations for the fixed point
  operator characterizing the Nash equilibrium. At each iteration, the policy is
  updated by computing a best response against the current mean-field or a
  regularized version that is obtained by taking a softmax with respect to the
  optimal Q-function, and the mean-field is updated by taking the mean-field
  induced by the current policy.
  """

    def __init__(self, game: pyspiel.Game, temperature: Optional[float]=None):
        if False:
            i = 10
            return i + 15
        'Initializes the algorithm.\n\n    Args:\n      game: The game to analyze.\n      temperature: If set, then instead of the greedy policy a softmax policy\n        with the specified temperature will be used to update the policy at each\n        iteration.\n    '
        self._game = game
        self._temperature = temperature
        self._policy = policy_lib.UniformRandomPolicy(self._game)
        self._distribution = distribution.DistributionPolicy(game, self._policy)

    def iteration(self):
        if False:
            print('Hello World!')
        'An itertion of Fixed Point.'
        distrib = distribution.DistributionPolicy(self._game, self._policy)
        br_value = best_response_value.BestResponse(self._game, distrib, value.TabularValueFunction(self._game))
        player_ids = list(range(self._game.num_players()))
        if self._temperature is None:
            self._policy = greedy_policy.GreedyPolicy(self._game, player_ids, br_value)
        else:
            self._policy = softmax_policy.SoftmaxPolicy(self._game, player_ids, self._temperature, br_value)
        self._distribution = distribution.DistributionPolicy(self._game, self._policy)

    def get_policy(self) -> policy_lib.Policy:
        if False:
            for i in range(10):
                print('nop')
        return self._policy

    @property
    def distribution(self) -> distribution.DistributionPolicy:
        if False:
            return 10
        return self._distribution