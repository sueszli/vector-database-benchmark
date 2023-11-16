"""Boltzmann Policy Iteration."""
from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg.algorithms import mirror_descent

class BoltzmannPolicyIteration(mirror_descent.MirrorDescent):
    """Boltzmann Policy Iteration algorithm.

  In this algorithm, at each iteration, we update the policy by first computing
  the Q-function that evaluates the current policy, and then take a softmax.
  This corresponds to using Online Mirror Descent algorithm without summing
  Q-functions but simply taking the latest Q-function.
  """

    def get_projected_policy(self) -> policy_lib.Policy:
        if False:
            for i in range(10):
                print('nop')
        'Returns the projected policy.'
        return mirror_descent.ProjectedPolicy(self._game, list(range(self._game.num_players())), self._state_value, coeff=self._lr)