"""Python base module for the implementations of Monte Carlo Counterfactual Regret Minimization."""
import numpy as np
from open_spiel.python import policy
REGRET_INDEX = 0
AVG_POLICY_INDEX = 1

class AveragePolicy(policy.Policy):
    """A policy object representing the average policy for MCCFR algorithms."""

    def __init__(self, game, player_ids, infostates):
        if False:
            while True:
                i = 10
        super().__init__(game, player_ids)
        self._infostates = infostates

    def action_probabilities(self, state, player_id=None):
        if False:
            print('Hello World!')
        'Returns the MCCFR average policy for a player in a state.\n\n    If the policy is not defined for the provided state, a uniform\n    random policy is returned.\n\n    Args:\n      state: A `pyspiel.State` object.\n      player_id: Optional, the player id for which we want an action. Optional\n        unless this is a simultaneous state at which multiple players can act.\n\n    Returns:\n      A `dict` of `{action: probability}` for the specified player in the\n      supplied state. If the policy is defined for the state, this\n      will contain the average MCCFR strategy defined for that state.\n      Otherwise, it will contain all legal actions, each with the same\n      probability, equal to 1 / num_legal_actions.\n    '
        if player_id is None:
            player_id = state.current_player()
        legal_actions = state.legal_actions()
        info_state_key = state.information_state_string(player_id)
        retrieved_infostate = self._infostates.get(info_state_key, None)
        if retrieved_infostate is None:
            return {a: 1 / len(legal_actions) for a in legal_actions}
        avstrat = retrieved_infostate[AVG_POLICY_INDEX] / retrieved_infostate[AVG_POLICY_INDEX].sum()
        return {legal_actions[i]: avstrat[i] for i in range(len(legal_actions))}

class MCCFRSolverBase(object):
    """A base class for both outcome MCCFR and external MCCFR."""

    def __init__(self, game):
        if False:
            print('Hello World!')
        self._game = game
        self._infostates = {}
        self._num_players = game.num_players()

    def _lookup_infostate_info(self, info_state_key, num_legal_actions):
        if False:
            return 10
        'Looks up an information set table for the given key.\n\n    Args:\n      info_state_key: information state key (string identifier).\n      num_legal_actions: number of legal actions at this information state.\n\n    Returns:\n      A list of:\n        - the average regrets as a numpy array of shape [num_legal_actions]\n        - the average strategy as a numpy array of shape\n        [num_legal_actions].\n          The average is weighted using `my_reach`\n    '
        retrieved_infostate = self._infostates.get(info_state_key, None)
        if retrieved_infostate is not None:
            return retrieved_infostate
        self._infostates[info_state_key] = [np.ones(num_legal_actions, dtype=np.float64) / 1000000.0, np.ones(num_legal_actions, dtype=np.float64) / 1000000.0]
        return self._infostates[info_state_key]

    def _add_regret(self, info_state_key, action_idx, amount):
        if False:
            i = 10
            return i + 15
        self._infostates[info_state_key][REGRET_INDEX][action_idx] += amount

    def _add_avstrat(self, info_state_key, action_idx, amount):
        if False:
            for i in range(10):
                print('nop')
        self._infostates[info_state_key][AVG_POLICY_INDEX][action_idx] += amount

    def average_policy(self):
        if False:
            i = 10
            return i + 15
        'Computes the average policy, containing the policy for all players.\n\n    Returns:\n      An average policy instance that should only be used during\n      the lifetime of solver object.\n    '
        return AveragePolicy(self._game, list(range(self._num_players)), self._infostates)

    def _regret_matching(self, regrets, num_legal_actions):
        if False:
            for i in range(10):
                print('nop')
        'Applies regret matching to get a policy.\n\n    Args:\n      regrets: numpy array of regrets for each action.\n      num_legal_actions: number of legal actions at this state.\n\n    Returns:\n      numpy array of the policy indexed by the index of legal action in the\n      list.\n    '
        positive_regrets = np.maximum(regrets, np.zeros(num_legal_actions, dtype=np.float64))
        sum_pos_regret = positive_regrets.sum()
        if sum_pos_regret <= 0:
            return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
        else:
            return positive_regrets / sum_pos_regret