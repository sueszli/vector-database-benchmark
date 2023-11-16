"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""
import numpy as np
from open_spiel.python.algorithms import mccfr
import pyspiel

class OutcomeSamplingSolver(mccfr.MCCFRSolverBase):
    """An implementation of outcome sampling MCCFR."""

    def __init__(self, game):
        if False:
            while True:
                i = 10
        super().__init__(game)
        self._expl = 0.6
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, "MCCFR requires sequential games. If you're trying to run it " + 'on a simultaneous (or normal-form) game, please first transform it ' + 'using turn_based_simultaneous_game.'

    def iteration(self):
        if False:
            for i in range(10):
                print('nop')
        'Performs one iteration of outcome sampling.\n\n    An iteration consists of one episode for each player as the update\n    player.\n    '
        for update_player in range(self._num_players):
            state = self._game.new_initial_state()
            self._episode(state, update_player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

    def _baseline(self, state, info_state, aidx):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def _baseline_corrected_child_value(self, state, info_state, sampled_aidx, aidx, child_value, sample_prob):
        if False:
            print('Hello World!')
        baseline = self._baseline(state, info_state, aidx)
        if aidx == sampled_aidx:
            return baseline + (child_value - baseline) / sample_prob
        else:
            return baseline

    def _episode(self, state, update_player, my_reach, opp_reach, sample_reach):
        if False:
            i = 10
            return i + 15
        'Runs an episode of outcome sampling.\n\n    Args:\n      state: the open spiel state to run from (will be modified in-place).\n      update_player: the player to update regrets for (the other players\n        update average strategies)\n      my_reach: reach probability of the update player\n      opp_reach: reach probability of all the opponents (including chance)\n      sample_reach: reach probability of the sampling (behavior) policy\n\n    Returns:\n      util is a real value representing the utility of the update player\n    '
        if state.is_terminal():
            return state.player_return(update_player)
        if state.is_chance_node():
            (outcomes, probs) = zip(*state.chance_outcomes())
            aidx = np.random.choice(range(len(outcomes)), p=probs)
            state.apply_action(outcomes[aidx])
            return self._episode(state, update_player, my_reach, probs[aidx] * opp_reach, probs[aidx] * sample_reach)
        cur_player = state.current_player()
        info_state_key = state.information_state_string(cur_player)
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        infostate_info = self._lookup_infostate_info(info_state_key, num_legal_actions)
        policy = self._regret_matching(infostate_info[mccfr.REGRET_INDEX], num_legal_actions)
        if cur_player == update_player:
            uniform_policy = np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
            sample_policy = self._expl * uniform_policy + (1.0 - self._expl) * policy
        else:
            sample_policy = policy
        sampled_aidx = np.random.choice(range(num_legal_actions), p=sample_policy)
        state.apply_action(legal_actions[sampled_aidx])
        if cur_player == update_player:
            new_my_reach = my_reach * policy[sampled_aidx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_aidx]
        new_sample_reach = sample_reach * sample_policy[sampled_aidx]
        child_value = self._episode(state, update_player, new_my_reach, new_opp_reach, new_sample_reach)
        child_values = np.zeros(num_legal_actions, dtype=np.float64)
        for aidx in range(num_legal_actions):
            child_values[aidx] = self._baseline_corrected_child_value(state, infostate_info, sampled_aidx, aidx, child_value, sample_policy[aidx])
        value_estimate = 0
        for aidx in range(num_legal_actions):
            value_estimate += policy[aidx] * child_values[aidx]
        if cur_player == update_player:
            cf_value = value_estimate * opp_reach / sample_reach
            for aidx in range(num_legal_actions):
                cf_action_value = child_values[aidx] * opp_reach / sample_reach
                self._add_regret(info_state_key, aidx, cf_action_value - cf_value)
            for aidx in range(num_legal_actions):
                increment = my_reach * policy[aidx] / sample_reach
                self._add_avstrat(info_state_key, aidx, increment)
        return value_estimate