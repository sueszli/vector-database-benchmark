"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""
import enum
import numpy as np
from open_spiel.python.algorithms import mccfr
import pyspiel

class AverageType(enum.Enum):
    SIMPLE = 0
    FULL = 1

class ExternalSamplingSolver(mccfr.MCCFRSolverBase):
    """An implementation of external sampling MCCFR."""

    def __init__(self, game, average_type=AverageType.SIMPLE):
        if False:
            while True:
                i = 10
        super().__init__(game)
        self._average_type = average_type
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, "MCCFR requires sequential games. If you're trying to run it " + 'on a simultaneous (or normal-form) game, please first transform it ' + 'using turn_based_simultaneous_game.'

    def iteration(self):
        if False:
            i = 10
            return i + 15
        'Performs one iteration of external sampling.\n\n    An iteration consists of one episode for each player as the update\n    player.\n    '
        for player in range(self._num_players):
            self._update_regrets(self._game.new_initial_state(), player)
        if self._average_type == AverageType.FULL:
            reach_probs = np.ones(self._num_players, dtype=np.float64)
            self._full_update_average(self._game.new_initial_state(), reach_probs)

    def _full_update_average(self, state, reach_probs):
        if False:
            for i in range(10):
                print('nop')
        'Performs a full update average.\n\n    Args:\n      state: the open spiel state to run from\n      reach_probs: array containing the probability of reaching the state\n        from the players point of view\n    '
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action in state.legal_actions():
                self._full_update_average(state.child(action), reach_probs)
            return
        sum_reach_probs = np.sum(reach_probs)
        if sum_reach_probs == 0:
            return
        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        infostate_info = self._lookup_infostate_info(info_state_key, num_legal_actions)
        policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX], num_legal_actions)
        for action_idx in range(num_legal_actions):
            new_reach_probs = np.copy(reach_probs)
            new_reach_probs[cur_player] *= policy[action_idx]
            self._full_update_average(state.child(legal_actions[action_idx]), new_reach_probs)
        for action_idx in range(num_legal_actions):
            self._add_avstrat(info_state_key, action_idx, reach_probs[cur_player] * policy[action_idx])

    def _update_regrets(self, state, player):
        if False:
            i = 10
            return i + 15
        'Runs an episode of external sampling.\n\n    Args:\n      state: the open spiel state to run from\n      player: the player to update regrets for\n\n    Returns:\n      value: is the value of the state in the game\n      obtained as the weighted average of the values\n      of the children\n    '
        if state.is_terminal():
            return state.player_return(player)
        if state.is_chance_node():
            (outcomes, probs) = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            return self._update_regrets(state.child(outcome), player)
        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        infostate_info = self._lookup_infostate_info(info_state_key, num_legal_actions)
        policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX], num_legal_actions)
        value = 0
        child_values = np.zeros(num_legal_actions, dtype=np.float64)
        if cur_player != player:
            action_idx = np.random.choice(np.arange(num_legal_actions), p=policy)
            value = self._update_regrets(state.child(legal_actions[action_idx]), player)
        else:
            for action_idx in range(num_legal_actions):
                child_values[action_idx] = self._update_regrets(state.child(legal_actions[action_idx]), player)
                value += policy[action_idx] * child_values[action_idx]
        if cur_player == player:
            for action_idx in range(num_legal_actions):
                self._add_regret(info_state_key, action_idx, child_values[action_idx] - value)
        if self._average_type == AverageType.SIMPLE and cur_player == (player + 1) % self._num_players:
            for action_idx in range(num_legal_actions):
                self._add_avstrat(info_state_key, action_idx, policy[action_idx])
        return value