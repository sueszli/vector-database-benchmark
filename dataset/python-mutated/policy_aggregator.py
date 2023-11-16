"""Policy aggregator.

Turns a weighted sum of N policies into a realization-equivalent single
policy by sweeping over the state space.
"""
import copy
import itertools
from open_spiel.python import policy

class PolicyFunction(policy.Policy):
    """A callable policy class."""

    def __init__(self, pids, policies, game):
        if False:
            for i in range(10):
                print('nop')
        'Construct a policy function.\n\n    Arguments:\n      pids: spiel player id of players these policies belong to.\n      policies: a list of dictionaries of keys (stringified binary observations)\n        to a list of probabilities for each move uid (between 0 and max_moves -\n        1).\n      game: OpenSpiel game.\n    '
        super().__init__(game, pids)
        self._policies = policies
        self._game_type = game.get_type()

    def _state_key(self, state, player_id=None):
        if False:
            i = 10
            return i + 15
        'Returns the key to use to look up this (state, player_id) pair.'
        if self._game_type.provides_information_state_string:
            if player_id is None:
                return state.information_state_string()
            else:
                return state.information_state_string(player_id)
        elif self._game_type.provides_observation_tensor:
            if player_id is None:
                return state.observation_tensor()
            else:
                return state.observation_tensor(player_id)
        else:
            return str(state)

    @property
    def policy(self):
        if False:
            while True:
                i = 10
        return self._policies

    def action_probabilities(self, state, player_id=None):
        if False:
            return 10
        'Returns the policy for a player in a state.\n\n    Args:\n      state: A `pyspiel.State` object.\n      player_id: Optional, the player id for whom we want an action. Optional\n        unless this is a simultaneous state at which multiple players can act.\n\n    Returns:\n      A `dict` of `{action: probability}` for the specified player in the\n      supplied state.\n    '
        state_key = self._state_key(state, player_id=player_id)
        if state.is_simultaneous_node():
            assert player_id >= 0
            return self._policies[player_id][state_key]
        if player_id is None:
            player_id = state.current_player()
        return self._policies[player_id][state_key]

class PolicyPool(object):
    """Transforms a list of list of policies (One list per player) to callable."""

    def __init__(self, policies):
        if False:
            for i in range(10):
                print('nop')
        'Transforms a list of list of policies (One list per player) to callable.\n\n    Args:\n      policies: List of list of policies.\n    '
        self._policies = policies

    def __call__(self, state, player):
        if False:
            while True:
                i = 10
        return [a.action_probabilities(state, player_id=player) for a in self._policies[player]]

class PolicyAggregator(object):
    """Main aggregator object."""

    def __init__(self, game, epsilon=1e-40):
        if False:
            for i in range(10):
                print('nop')
        self._game = game
        self._game_type = game.get_type()
        self._num_players = self._game.num_players()
        self._policy_pool = None
        self._weights = None
        self._policy = {}
        self._epsilon = epsilon

    def _state_key(self, state, player_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the key to use to look up this (state, player) pair.'
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

    def aggregate(self, pids, policies, weights):
        if False:
            for i in range(10):
                print('nop')
        'Aggregate the list of policies for each player.\n\n    Arguments:\n      pids: the spiel player ids of the players the strategies belong to.\n      policies: List of list of policies (One list per player)\n      weights: the list of weights to attach to each policy.\n\n    Returns:\n      A PolicyFunction, a callable object representing the policy.\n    '
        aggr_policies = []
        for pid in pids:
            aggr_policies.append(self._sub_aggregate(pid, policies, weights))
        return PolicyFunction(pids, aggr_policies, self._game)

    def _sub_aggregate(self, pid, policies, weights):
        if False:
            for i in range(10):
                print('nop')
        'Aggregate the list of policies for one player.\n\n    Arguments:\n      pid: the spiel player id of the player the strategies belong to.\n      policies: List of list of policies (One list per player)\n      weights: the list of weights to attach to each policy.\n\n    Returns:\n      A PolicyFunction, a callable object representing the policy.\n    '
        self._policy_pool = PolicyPool(policies)
        assert self._policy_pool is not None
        self._weights = weights
        self._policy = {}
        state = self._game.new_initial_state()
        my_reaches = weights[:]
        self._rec_aggregate(pid, state, my_reaches)
        for key in self._policy:
            (actions, probabilities) = zip(*self._policy[key].items())
            new_probs = [prob + self._epsilon for prob in probabilities]
            denom = sum(new_probs)
            for i in range(len(actions)):
                self._policy[key][actions[i]] = new_probs[i] / denom
        return self._policy

    def _rec_aggregate(self, pid, state, my_reaches):
        if False:
            for i in range(10):
                print('nop')
        'Recursively traverse game tree to compute aggregate policy.'
        if state.is_terminal():
            return
        elif state.is_simultaneous_node():
            policies = self._policy_pool(state, pid)
            state_key = self._state_key(state, pid)
            self._policy[state_key] = {}
            used_moves = state.legal_actions(pid)
            for uid in used_moves:
                new_reaches = copy.deepcopy(my_reaches)
                for i in range(len(policies)):
                    new_reaches[pid][i] *= policies[i].get(uid, 0)
                    if uid in self._policy[state_key].keys():
                        self._policy[state_key][uid] += new_reaches[pid][i]
                    else:
                        self._policy[state_key][uid] = new_reaches[pid][i]
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
                        new_reaches[pid][i] *= policies[i].get(uid, 0)
                    joint_action = list(other_joint_action[:pid] + (uid,) + other_joint_action[pid:])
                    new_state = state.clone()
                    new_state.apply_actions(joint_action)
                    self._rec_aggregate(pid, new_state, new_reaches)
            return
        elif state.is_chance_node():
            (outcomes, _) = zip(*state.chance_outcomes())
            for i in range(0, len(outcomes)):
                outcome = outcomes[i]
                new_state = state.clone()
                new_state.apply_action(outcome)
                self._rec_aggregate(pid, new_state, my_reaches)
            return
        else:
            turn_player = state.current_player()
            state_key = self._state_key(state, turn_player)
            legal_policies = self._policy_pool(state, turn_player)
            if pid == turn_player:
                if state_key not in self._policy:
                    self._policy[state_key] = {}
            used_moves = state.legal_actions(turn_player)
            for uid in used_moves:
                new_reaches = copy.deepcopy(my_reaches)
                if pid == turn_player:
                    for i in range(len(legal_policies)):
                        new_reaches[turn_player][i] *= legal_policies[i].get(uid, 0)
                        if uid in self._policy[state_key].keys():
                            self._policy[state_key][uid] += new_reaches[turn_player][i]
                        else:
                            self._policy[state_key][uid] = new_reaches[turn_player][i]
                new_state = state.clone()
                new_state.apply_action(uid)
                self._rec_aggregate(pid, new_state, new_reaches)