"""Computes a Best-Response policy.

The goal if this file is to be the main entry-point for BR APIs in Python.

TODO(author2): Also include computation using the more efficient C++
`TabularBestResponse` implementation.
"""
import collections
import itertools
import numpy as np
from open_spiel.python import games
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import noisy_policy
from open_spiel.python.algorithms import policy_utils
import pyspiel

def _memoize_method(key_fn=lambda x: x):
    if False:
        return 10
    'Memoize a single-arg instance method using an on-object cache.'

    def memoizer(method):
        if False:
            while True:
                i = 10
        cache_name = 'cache_' + method.__name__

        def wrap(self, arg):
            if False:
                return 10
            key = key_fn(arg)
            cache = vars(self).setdefault(cache_name, {})
            if key not in cache:
                cache[key] = method(self, arg)
            return cache[key]
        return wrap
    return memoizer

def compute_states_and_info_states_if_none(game, all_states=None, state_to_information_state=None):
    if False:
        while True:
            i = 10
    'Returns all_states and/or state_to_information_state for the game.\n\n  To recompute everything, pass in None for both all_states and\n  state_to_information_state. Otherwise, this function will use the passed in\n  values to reconstruct either of them.\n\n  Args:\n    game: The open_spiel game.\n    all_states: The result of calling get_all_states.get_all_states. Cached for\n      improved performance.\n    state_to_information_state: A dict mapping state.history_str() to\n      state.information_state for every state in the game. Cached for improved\n      performance.\n  '
    if all_states is None:
        all_states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=False, include_chance_states=False)
    if state_to_information_state is None:
        state_to_information_state = {state: all_states[state].information_state_string() for state in all_states}
    return (all_states, state_to_information_state)

class BestResponsePolicy(openspiel_policy.Policy):
    """Computes the best response to a specified strategy."""

    def __init__(self, game, player_id, policy, root_state=None, cut_threshold=0.0):
        if False:
            print('Hello World!')
        'Initializes the best-response calculation.\n\n    Args:\n      game: The game to analyze.\n      player_id: The player id of the best-responder.\n      policy: A `policy.Policy` object.\n      root_state: The state of the game at which to start analysis. If `None`,\n        the game root state is used.\n      cut_threshold: The probability to cut when calculating the value.\n        Increasing this value will trade off accuracy for speed.\n    '
        self._num_players = game.num_players()
        self._player_id = player_id
        self._policy = policy
        if root_state is None:
            root_state = game.new_initial_state()
        self._root_state = root_state
        self.infosets = self.info_sets(root_state)
        self._cut_threshold = cut_threshold

    def info_sets(self, state):
        if False:
            while True:
                i = 10
        'Returns a dict of infostatekey to list of (state, cf_probability).'
        infosets = collections.defaultdict(list)
        for (s, p) in self.decision_nodes(state):
            infosets[s.information_state_string(self._player_id)].append((s, p))
        return dict(infosets)

    def decision_nodes(self, parent_state):
        if False:
            print('Hello World!')
        'Yields a (state, cf_prob) pair for each descendant decision node.'
        if not parent_state.is_terminal():
            if parent_state.current_player() == self._player_id or parent_state.is_simultaneous_node():
                yield (parent_state, 1.0)
            for (action, p_action) in self.transitions(parent_state):
                for (state, p_state) in self.decision_nodes(openspiel_policy.child(parent_state, action)):
                    yield (state, p_state * p_action)

    def joint_action_probabilities_counterfactual(self, state):
        if False:
            for i in range(10):
                print('nop')
        "Get list of action, probability tuples for simultaneous node.\n\n    Counterfactual reach probabilities exclude the best-responder's actions,\n    the sum of the probabilities is equal to the number of actions of the\n    player _player_id.\n    Args:\n      state: the current state of the game.\n\n    Returns:\n      list of action, probability tuples. An action is a tuple of individual\n        actions for each player of the game.\n    "
        (actions_per_player, probs_per_player) = openspiel_policy.joint_action_probabilities_aux(state, self._policy)
        probs_per_player[self._player_id] = [1.0 for _ in probs_per_player[self._player_id]]
        return [(list(actions), np.prod(probs)) for (actions, probs) in zip(itertools.product(*actions_per_player), itertools.product(*probs_per_player))]

    def transitions(self, state):
        if False:
            i = 10
            return i + 15
        'Returns a list of (action, cf_prob) pairs from the specified state.'
        if state.current_player() == self._player_id:
            return [(action, 1.0) for action in state.legal_actions()]
        elif state.is_chance_node():
            return state.chance_outcomes()
        elif state.is_simultaneous_node():
            return self.joint_action_probabilities_counterfactual(state)
        else:
            return list(self._policy.action_probabilities(state).items())

    @_memoize_method(key_fn=lambda state: state.history_str())
    def value(self, state):
        if False:
            print('Hello World!')
        'Returns the value of the specified state to the best-responder.'
        if state.is_terminal():
            return state.player_return(self._player_id)
        elif state.current_player() == self._player_id or state.is_simultaneous_node():
            action = self.best_response_action(state.information_state_string(self._player_id))
            return self.q_value(state, action)
        else:
            return sum((p * self.q_value(state, a) for (a, p) in self.transitions(state) if p > self._cut_threshold))

    def q_value(self, state, action):
        if False:
            i = 10
            return i + 15
        'Returns the value of the (state, action) to the best-responder.'
        if state.is_simultaneous_node():

            def q_value_sim(sim_state, sim_actions):
                if False:
                    print('Hello World!')
                child = sim_state.clone()
                sim_actions[self._player_id] = action
                child.apply_actions(sim_actions)
                return self.value(child)
            (actions, probabilities) = zip(*self.transitions(state))
            return sum((p * q_value_sim(state, a) for (a, p) in zip(actions, probabilities / sum(probabilities)) if p > self._cut_threshold))
        else:
            return self.value(state.child(action))

    @_memoize_method()
    def best_response_action(self, infostate):
        if False:
            while True:
                i = 10
        'Returns the best response for this information state.'
        infoset = self.infosets[infostate]
        return max(infoset[0][0].legal_actions(self._player_id), key=lambda a: sum((cf_p * self.q_value(s, a) for (s, cf_p) in infoset)))

    def action_probabilities(self, state, player_id=None):
        if False:
            while True:
                i = 10
        'Returns the policy for a player in a state.\n\n    Args:\n      state: A `pyspiel.State` object.\n      player_id: Optional, the player id for whom we want an action. Optional\n        unless this is a simultaneous state at which multiple players can act.\n\n    Returns:\n      A `dict` of `{action: probability}` for the specified player in the\n      supplied state.\n    '
        if player_id is None:
            if state.is_simultaneous_node():
                player_id = self._player_id
            else:
                player_id = state.current_player()
        return {self.best_response_action(state.information_state_string(player_id)): 1}

class CPPBestResponsePolicy(openspiel_policy.Policy):
    """Computes best response action_probabilities using open_spiel's C++ backend.

     May have better performance than best_response.py for large games.
  """

    def __init__(self, game, best_responder_id, policy, all_states=None, state_to_information_state=None, best_response_processor=None, cut_threshold=0.0):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      game: The game to analyze.\n      best_responder_id: The player id of the best-responder.\n      policy: A policy.Policy object representing the joint policy, taking a\n        state and returning a list of (action, probability) pairs. This could be\n        aggr_policy, for instance.\n      all_states: The result of calling get_all_states.get_all_states. Cached\n        for improved performance.\n      state_to_information_state: A dict mapping state.history_str to\n        state.information_state for every state in the game. Cached for improved\n        performance.\n      best_response_processor: A TabularBestResponse object, used for processing\n        the best response actions.\n      cut_threshold: The probability to cut when calculating the value.\n        Increasing this value will trade off accuracy for speed.\n    '
        (self.all_states, self.state_to_information_state) = compute_states_and_info_states_if_none(game, all_states, state_to_information_state)
        policy_to_dict = policy_utils.policy_to_dict(policy, game, self.all_states, self.state_to_information_state)
        if not best_response_processor:
            best_response_processor = pyspiel.TabularBestResponse(game, best_responder_id, policy_to_dict)
        self._policy = policy
        self.game = game
        self.best_responder_id = best_responder_id
        self.tabular_best_response_map = best_response_processor.get_best_response_actions()
        self._cut_threshold = cut_threshold

    def decision_nodes(self, parent_state):
        if False:
            for i in range(10):
                print('nop')
        'Yields a (state, cf_prob) pair for each descendant decision node.'
        if not parent_state.is_terminal():
            if parent_state.current_player() == self.best_responder_id:
                yield (parent_state, 1.0)
            for (action, p_action) in self.transitions(parent_state):
                for (state, p_state) in self.decision_nodes(parent_state.child(action)):
                    yield (state, p_state * p_action)

    def transitions(self, state):
        if False:
            while True:
                i = 10
        'Returns a list of (action, cf_prob) pairs from the specified state.'
        if state.current_player() == self.best_responder_id:
            return [(action, 1.0) for action in state.legal_actions()]
        elif state.is_chance_node():
            return state.chance_outcomes()
        else:
            return list(self._policy.action_probabilities(state).items())

    @_memoize_method(key_fn=lambda state: state.history_str())
    def value(self, state):
        if False:
            while True:
                i = 10
        'Returns the value of the specified state to the best-responder.'
        if state.is_terminal():
            return state.player_return(self.best_responder_id)
        elif state.current_player() == self.best_responder_id:
            action = self.best_response_action(state.information_state_string(self.best_responder_id))
            return self.q_value(state, action)
        else:
            return sum((p * self.q_value(state, a) for (a, p) in self.transitions(state) if p > self._cut_threshold))

    def q_value(self, state, action):
        if False:
            while True:
                i = 10
        'Returns the value of the (state, action) to the best-responder.'
        return self.value(state.child(action))

    @_memoize_method()
    def best_response_action(self, infostate):
        if False:
            while True:
                i = 10
        'Returns the best response for this information state.'
        action = self.tabular_best_response_map[infostate]
        return action

    def action_probabilities(self, state, player_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the policy for a player in a state.\n\n    Args:\n      state: A `pyspiel.State` object.\n      player_id: Optional, the player id for whom we want an action. Optional\n        unless this is a simultabeous state at which multiple players can act.\n\n    Returns:\n      A `dict` of `{action: probability}` for the specified player in the\n      supplied state.\n    '
        if state.current_player() == self.best_responder_id:
            probs = {action_id: 0.0 for action_id in state.legal_actions()}
            info_state = self.state_to_information_state[state.history_str()]
            probs[self.tabular_best_response_map[info_state]] = 1.0
            return probs
        return self._policy.action_probabilities(state, player_id)

    @property
    def policy(self):
        if False:
            return 10
        return self._policy

    def copy_with_noise(self, alpha=0.0, beta=0.0):
        if False:
            i = 10
            return i + 15
        "Copies this policy and adds noise, making it a Noisy Best Response.\n\n    The policy's new probabilities P' on each state s become\n    P'(s) = alpha * epsilon + (1-alpha) * P(s)\n\n    With P the former policy's probabilities, and epsilon ~ Softmax(beta *\n    Uniform)\n\n    Args:\n      alpha: First mixture component\n      beta: Softmax 1/temperature component\n\n    Returns:\n      Noisy copy of best response.\n    "
        return noisy_policy.NoisyPolicy(self, alpha, beta, self.all_states)