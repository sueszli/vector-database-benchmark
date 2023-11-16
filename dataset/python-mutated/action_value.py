"""Q-values and reach probabilities computation."""
import collections
import numpy as np
_CalculatorReturn = collections.namedtuple('_CalculatorReturn', ['root_node_values', 'action_values', 'counterfactual_reach_probs', 'player_reach_probs', 'sum_cfr_reach_by_action_value'])

class TreeWalkCalculator(object):
    """Class to orchestrate the calculation.

  This performs a full history tree walk and computes several statistics,
  available as attributes.

  Attributes:
    weighted_action_values: A dictionary mapping (player,information state
      string) to a dictionary mapping each action to a vector of the sum of
      (reward * prob) reward taking that action for each player. To get the
      action-values, one will need to normalize by `info_state_prob`.
    info_state_prob:  A dictionary mapping (player,information state string) to
      the reach probability of this info_state.
    info_state_player_prob: Same as info_state_prob for the player reach
      probability.
    info_state_cf_prob: Same as info_state_prob for the counterfactual reach
      probability to get to that state, i.e. the sum over histories, of the
      product of the opponents probabilities of actions leading to the history.
    info_state_chance_prob: Same as above, for the chance probability to get
      into that state.
    info_state_cf_prob_by_q_sum: A dictionary mapping (player,information state
      string) to a vector of shape `[num_actions]`, that store for each action
      the cumulative \\sum_{h \\in x} cfr_reach(h) * Q(h, a)
    root_values: The values at the root node [for player 0, for player 1].
  """

    def __init__(self, game):
        if False:
            i = 10
            return i + 15
        if not game.get_type().provides_information_state_string:
            raise ValueError('Only game which provide the information_state_string are supported, as this is being used in the key to identify states.')
        self._game = game
        self._num_players = game.num_players()
        self._num_actions = game.num_distinct_actions()
        self.weighted_action_values = None
        self.info_state_prob = None
        self.info_state_player_prob = None
        self.info_state_cf_prob = None
        self.info_state_chance_prob = None
        self.info_state_cf_prob_by_q_sum = None
        self.root_values = None

    def _get_action_values(self, state, policies, reach_probabilities):
        if False:
            return 10
        'Computes the value of the state given the policies for both players.\n\n    Args:\n      state: The state to start analysis from.\n      policies: List of `policy.Policy` objects, one per player.\n      reach_probabilities: A numpy array of shape `[num_players + 1]`.\n        reach_probabilities[i] is the product of the player i action\n        probabilities along the current trajectory. Note that\n        reach_probabilities[-1] corresponds to the chance player. Initially, it\n        should be called with np.ones(self._num_players + 1) at the root node.\n\n    Returns:\n      The value of the root state to each player.\n\n    Side-effects - populates:\n      `self.weighted_action_values[(player, infostate)][action]`.\n      `self.info_state_prob[(player, infostate)]`.\n      `self.info_state_cf_prob[(player, infostate)]`.\n      `self.info_state_chance_prob[(player, infostate)]`.\n\n    We use `(player, infostate)` as a key in case the same infostate is shared\n    by multiple players, e.g. in a simultaneous-move game.\n    '
        if state.is_terminal():
            return np.array(state.returns())
        current_player = state.current_player()
        is_chance = state.is_chance_node()
        if not is_chance:
            key = (current_player, state.information_state_string())
            reach_prob = np.prod(reach_probabilities)
            opponent_probability = np.prod(reach_probabilities[:current_player]) * np.prod(reach_probabilities[current_player + 1:-1])
            self.info_state_cf_prob[key] += reach_probabilities[-1] * opponent_probability
            self.info_state_prob[key] += reach_prob
            self.info_state_chance_prob[key] += reach_probabilities[-1]
            self.info_state_player_prob[key] = reach_probabilities[current_player]
        value = np.zeros(len(policies))
        if is_chance:
            action_to_prob = dict(state.chance_outcomes())
        else:
            action_to_prob = policies[current_player].action_probabilities(state)
        for action in state.legal_actions():
            prob = action_to_prob.get(action, 0)
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= prob
            child = state.child(action)
            child_value = self._get_action_values(child, policies, reach_probabilities=new_reach_probabilities)
            if not is_chance:
                self.weighted_action_values[key][action] += child_value * reach_prob
                self.info_state_cf_prob_by_q_sum[key][action] += child_value[current_player] * opponent_probability * reach_probabilities[-1]
            value += child_value * prob
        return value

    def compute_all_states_action_values(self, policies):
        if False:
            for i in range(10):
                print('nop')
        "Computes action values per state for the player.\n\n    The internal state is fully re-created when calling this method, thus it's\n    safe to use one object to perform several tree-walks using different\n    policies, and to extract the results using for example\n    `calculator.infor_state_prob` to take ownership of the dictionary.\n\n    Args:\n      policies: List of `policy.Policy` objects, one per player. As the policy\n        will be accessed using `policies[i]`, it can also be a dictionary\n        mapping player_id to a `policy.Policy` object.\n    "
        assert len(policies) == self._num_players
        self.weighted_action_values = collections.defaultdict(lambda : collections.defaultdict(lambda : np.zeros(self._num_players)))
        self.info_state_prob = collections.defaultdict(float)
        self.info_state_player_prob = collections.defaultdict(float)
        self.info_state_cf_prob = collections.defaultdict(float)
        self.info_state_chance_prob = collections.defaultdict(float)
        self.info_state_cf_prob_by_q_sum = collections.defaultdict(lambda : np.zeros(self._num_actions))
        self.root_values = self._get_action_values(self._game.new_initial_state(), policies, reach_probabilities=np.ones(self._num_players + 1))

    def _get_tabular_statistics(self, keys):
        if False:
            i = 10
            return i + 15
        'Returns tabular numpy arrays of the resulting stastistics.\n\n    Args:\n      keys: A list of the (player, info_state_str) keys to use to return the\n        tabular numpy array of results.\n    '
        action_values = []
        cfrp = []
        player_reach_probs = []
        sum_cfr_reach_by_action_value = []
        for key in keys:
            player = key[0]
            av = self.weighted_action_values[key]
            norm_prob = self.info_state_prob[key]
            action_values.append([av[a][player] / norm_prob if a in av and norm_prob > 0 else 0 for a in range(self._num_actions)])
            cfrp.append(self.info_state_cf_prob[key])
            player_reach_probs.append(self.info_state_player_prob[key])
            sum_cfr_reach_by_action_value.append(self.info_state_cf_prob_by_q_sum[key])
        return _CalculatorReturn(root_node_values=self.root_values, action_values=action_values, counterfactual_reach_probs=cfrp, player_reach_probs=player_reach_probs, sum_cfr_reach_by_action_value=sum_cfr_reach_by_action_value)

    def get_tabular_statistics(self, tabular_policy):
        if False:
            while True:
                i = 10
        'Returns tabular numpy arrays of the resulting stastistics.\n\n    This function should be called after `compute_all_states_action_values`.\n    Optionally, one can directly call the object to perform both actions.\n\n    Args:\n      tabular_policy: A `policy.TabularPolicy` object, used to get the ordering\n        of the states in the tabular numpy array.\n    '
        keys = []
        for (player_id, player_states) in enumerate(tabular_policy.states_per_player):
            keys += [(player_id, s) for s in player_states]
        return self._get_tabular_statistics(keys)

    def __call__(self, policies, tabular_policy):
        if False:
            print('Hello World!')
        "Computes action values per state for the player.\n\n    The internal state is fully re-created when calling this method, thus it's\n    safe to use one object to perform several tree-walks using different\n    policies, and to extract the results using for example\n    `calculator.infor_state_prob` to take ownership of the dictionary.\n\n    Args:\n      policies: List of `policy.Policy` objects, one per player.\n      tabular_policy: A `policy.TabularPolicy` object, used to get the ordering\n        of the states in the tabular numpy array.\n\n    Returns:\n      A `_CalculatorReturn` namedtuple. See its docstring for the details.\n    "
        self.compute_all_states_action_values(policies)
        return self.get_tabular_statistics(tabular_policy)

    def get_root_node_values(self, policies):
        if False:
            print('Hello World!')
        'Gets root values only.\n\n    This speeds up calculation in two ways:\n\n    1. It only searches nodes with positive probability.\n    2. It does not populate a large dictionary of meta information.\n\n    Args:\n      policies: List of `policy.Policy` objects, one per player.\n\n    Returns:\n      A numpy array of shape [num_players] of the root value.\n    '
        return self._get_action_values_only(self._game.new_initial_state(), policies, reach_probabilities=np.ones(self._num_players + 1))

    def _get_action_values_only(self, state, policies, reach_probabilities):
        if False:
            for i in range(10):
                print('nop')
        'Computes the value of the state given the policies for both players.\n\n    Args:\n      state: The state to start analysis from.\n      policies: List of `policy.Policy` objects, one per player.\n      reach_probabilities: A numpy array of shape `[num_players + 1]`.\n        reach_probabilities[i] is the product of the player i action\n        probabilities along the current trajectory. Note that\n        reach_probabilities[-1] corresponds to the chance player. Initially, it\n        should be called with np.ones(self._num_players + 1) at the root node.\n\n    Returns:\n      A numpy array of shape [num_players] of the root value.\n    '
        if state.is_terminal():
            return np.array(state.returns())
        current_player = state.current_player()
        is_chance = state.is_chance_node()
        value = np.zeros(len(policies))
        if is_chance:
            action_to_prob = dict(state.chance_outcomes())
        else:
            action_to_prob = policies[current_player].action_probabilities(state)
        for action in state.legal_actions():
            prob = action_to_prob.get(action, 0)
            if prob == 0.0:
                continue
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= prob
            child = state.child(action)
            child_value = self._get_action_values_only(child, policies, reach_probabilities=new_reach_probabilities)
            value += child_value * prob
        return value