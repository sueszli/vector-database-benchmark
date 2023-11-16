"""Policy aggregator.

A joint policy is a list of `num_players` policies.
This files enables to compute mixtures of such joint-policies to get a new
policy.
"""
import copy
import itertools
from open_spiel.python import policy

def _aggregate_at_state(joint_policies, state, player):
    if False:
        while True:
            i = 10
    'Returns {action: prob} for `player` in `state` for all joint policies.\n\n  Args:\n    joint_policies: List of joint policies.\n    state: Openspiel State\n    player: Current Player\n\n  Returns:\n    {action: prob} for `player` in `state` for all joint policies.\n  '
    return [joint_policy[player].action_probabilities(state, player_id=player) for joint_policy in joint_policies]

class _DictPolicy(policy.Policy):
    """A callable policy class."""

    def __init__(self, game, policies_as_dict):
        if False:
            while True:
                i = 10
        'Constructs a policy function.\n\n    Arguments:\n      game: OpenSpiel game.\n      policies_as_dict: A list of `num_players` policy objects {action: prob}.\n    '
        self._game = game
        self._game_type = game.get_type()
        self._policies_as_dict = policies_as_dict

    def _state_key(self, state, player_id=None):
        if False:
            while True:
                i = 10
        'Returns the key to use to look up this (state, player_id) pair.'
        if self._game_type.provides_information_state_string:
            if player_id is None:
                return state.information_state_string()
            else:
                return state.information_state_string(player_id)
        elif self._game_type.provides_observation_string:
            if player_id is None:
                return state.observation_string()
            else:
                return state.observation_string(player_id)
        else:
            return str(state)

    @property
    def policies(self):
        if False:
            for i in range(10):
                print('nop')
        return self._policies_as_dict

    def action_probabilities(self, state, player_id=None):
        if False:
            print('Hello World!')
        'Returns the policy for a player in a state.\n\n    Args:\n      state: A `pyspiel.State` object.\n      player_id: Optional, the player id for whom we want an action. Optional\n        unless this is a simultaneous state at which multiple players can act.\n\n    Returns:\n      A `dict` of `{action: probability}` for the specified player in the\n      supplied state.\n    '
        state_key = self._state_key(state, player_id=player_id)
        if player_id is None:
            player_id = state.current_player()
        return self._policies_as_dict[player_id][state_key]

class JointPolicyAggregator(object):
    """Main aggregator object."""

    def __init__(self, game, epsilon=1e-40):
        if False:
            return 10
        self._game = game
        self._game_type = game.get_type()
        self._num_players = self._game.num_players()
        self._joint_policies = None
        self._policy = {}
        self._epsilon = epsilon

    def _state_key(self, state, player_id=None):
        if False:
            i = 10
            return i + 15
        'Returns the key to use to look up this (state, player) pair.'
        if self._game_type.provides_information_state_string:
            if player_id is None:
                return state.information_state_string()
            else:
                return state.information_state_string(player_id)
        elif self._game_type.provides_observation_string:
            if player_id is None:
                return state.observation()
            else:
                return state.observation(player_id)
        else:
            return str(state)

    def aggregate(self, pids, joint_policies, weights):
        if False:
            print('Hello World!')
        "Computes the weighted-mixture of the joint policies.\n\n    Let P of shape [num_players] be the joint policy, and W some weights.\n    Let N be the number of policies (i.e. len(policies)).\n    We return the policy P' such that for all state `s`:\n\n    P[s] ~ \\sum_{i=0}^{N-1} (policies[i][player(s)](s) * weights[i] *\n                             reach_prob(s, policies[i]))\n\n    Arguments:\n      pids: Spiel player ids of the players the strategies belong to.\n      joint_policies: List of list of policies (One list per joint strategy)\n      weights: List of weights to attach to each joint strategy.\n\n    Returns:\n      A _DictPolicy, a callable object representing the policy.\n    "
        aggr_policies = []
        self._joint_policies = joint_policies
        for pid in pids:
            aggr_policies.append(self._sub_aggregate(pid, weights))
        return _DictPolicy(self._game, aggr_policies)

    def _sub_aggregate(self, pid, weights):
        if False:
            for i in range(10):
                print('nop')
        'Aggregate the list of policies for one player.\n\n    Arguments:\n      pid: Spiel player id of the player the strategies belong to.\n      weights: List of weights to attach to each joint strategy.\n\n    Returns:\n      A _DictPolicy, a callable object representing the policy.\n    '
        self._policy = {}
        state = self._game.new_initial_state()
        self._rec_aggregate(pid, state, copy.deepcopy(weights))
        for key in self._policy:
            (actions, probabilities) = zip(*self._policy[key].items())
            new_probs = [prob + self._epsilon for prob in probabilities]
            denom = sum(new_probs)
            for i in range(len(actions)):
                self._policy[key][actions[i]] = new_probs[i] / denom
        return self._policy

    def _rec_aggregate(self, pid, state, my_reaches):
        if False:
            while True:
                i = 10
        'Recursively traverse game tree to compute aggregate policy.'
        if state.is_terminal():
            return
        if state.is_simultaneous_node():
            policies = _aggregate_at_state(self._joint_policies, state, pid)
            state_key = self._state_key(state, pid)
            self._policy[state_key] = {}
            used_moves = state.legal_actions(pid)
            for uid in used_moves:
                new_reaches = copy.deepcopy(my_reaches)
                for i in range(len(policies)):
                    new_reaches[i] *= policies[i].get(uid, 0)
                    if uid in self._policy[state_key].keys():
                        self._policy[state_key][uid] += new_reaches[i]
                    else:
                        self._policy[state_key][uid] = new_reaches[i]
            num_players = self._game.num_players()
            all_other_used_moves = []
            for player in range(num_players):
                if player != pid:
                    all_other_used_moves.append(state.legal_actions(player))
            other_joint_actions = itertools.product(*all_other_used_moves)
            for other_joint_action in other_joint_actions:
                for uid in used_moves:
                    new_reaches = copy.deepcopy(my_reaches)
                    for i in range(len(policies)):
                        new_reaches[i] *= policies[i].get(uid, 0)
                    joint_action = list(other_joint_action[:pid] + (uid,) + other_joint_action[pid:])
                    new_state = state.clone()
                    new_state.apply_actions(joint_action)
                    self._rec_aggregate(pid, new_state, new_reaches)
            return
        if state.is_chance_node():
            for action in state.legal_actions():
                new_state = state.child(action)
                self._rec_aggregate(pid, new_state, my_reaches)
            return
        current_player = state.current_player()
        state_key = self._state_key(state, current_player)
        action_probabilities_list = _aggregate_at_state(self._joint_policies, state, current_player)
        if pid == current_player:
            if state_key not in self._policy:
                self._policy[state_key] = {}
        for action in state.legal_actions():
            new_reaches = copy.deepcopy(my_reaches)
            if pid == current_player:
                for (idx, state_action_probs) in enumerate(action_probabilities_list):
                    new_reaches[idx] *= state_action_probs.get(action, 0)
                    if action in self._policy[state_key].keys():
                        self._policy[state_key][action] += new_reaches[idx]
                    else:
                        self._policy[state_key][action] = new_reaches[idx]
            self._rec_aggregate(pid, state.child(action), new_reaches)