"""Python implementation of the CFR-BR algorithm."""
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel
_CFRSolverBase = cfr._CFRSolverBase
_update_current_policy = cfr._update_current_policy
_apply_regret_matching_plus_reset = cfr._apply_regret_matching_plus_reset

class CFRBRSolver(_CFRSolverBase):
    """Implements the Counterfactual Regret Minimization (CFR-BR) algorithm.

  This is Counterfactual Regret Minimization against Best Response, from
  Michael Johanson and al., 2012, Finding Optimal Abstract Strategies in
  Extensive-Form Games,
  https://poker.cs.ualberta.ca/publications/AAAI12-cfrbr.pdf).

  The algorithm
  computes an approximate Nash policy for n-player zero-sum games, but the
  implementation is currently restricted to 2-player.

  It uses an exact Best Response and full tree traversal.

  One iteration for a n-player game consists of the following:

  - Compute the BR of each player against the rest of the players.
  - Then, for each player p sequentially (from player 0 to N-1):
    - Compute the conterfactual reach probabilities and action values for player
      p, playing against the set of the BR for all other players.
    - Update the player `p` policy using these values.

  CFR-BR should converge with high probability (see the paper), but we can also
  compute the time-averaged strategy.

  The implementation reuses the `action_values_vs_best_response` module and
  thus uses TabularPolicies. This will run only for smallish games.
  """

    def __init__(self, game, linear_averaging=False, regret_matching_plus=False):
        if False:
            return 10
        'Initializer.\n\n    Args:\n      game: The `pyspiel.Game` to run on.\n      linear_averaging: Whether to use linear averaging, i.e.\n        cumulative_policy[info_state][action] += (\n          iteration_number * reach_prob * action_prob)\n\n        or not:\n\n        cumulative_policy[info_state][action] += reach_prob * action_prob\n      regret_matching_plus: Whether to use Regret Matching+:\n        cumulative_regrets = max(cumulative_regrets + regrets, 0)\n        or simply regret matching:\n        cumulative_regrets = cumulative_regrets + regrets\n    '
        if game.num_players() != 2:
            raise ValueError('Game {} does not have {} players.'.format(game, 2))
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, "CFR requires sequential games. If you're trying to run it " + 'on a simultaneous (or normal-form) game, please first transform it ' + 'using turn_based_simultaneous_game.'
        super(CFRBRSolver, self).__init__(game, alternating_updates=True, linear_averaging=linear_averaging, regret_matching_plus=regret_matching_plus)
        self._best_responses = {i: None for i in range(game.num_players())}

    def _compute_best_responses(self):
        if False:
            return 10
        'Computes each player best-response against the pool of other players.'

        def policy_fn(state):
            if False:
                i = 10
                return i + 15
            key = state.information_state_string()
            return self._get_infostate_policy(key)
        current_policy = policy.tabular_policy_from_callable(self._game, policy_fn)
        for player_id in range(self._game.num_players()):
            self._best_responses[player_id] = exploitability.best_response(self._game, current_policy, player_id)

    def evaluate_and_update_policy(self):
        if False:
            while True:
                i = 10
        'Performs a single step of policy evaluation and policy improvement.'
        self._iteration += 1
        self._compute_best_responses()
        for player in range(self._num_players):
            policies = []
            for p in range(self._num_players):
                policies.append(lambda infostate_str, p=p: {self._best_responses[p]['best_response_action'][infostate_str]: 1})
            policies[player] = self._get_infostate_policy
            self._compute_counterfactual_regret_for_player(state=self._root_node, policies=policies, reach_probabilities=np.ones(self._num_players + 1), player=player)
            if self._regret_matching_plus:
                _apply_regret_matching_plus_reset(self._info_state_nodes)
        _update_current_policy(self._current_policy, self._info_state_nodes)