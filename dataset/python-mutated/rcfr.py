"""Regression counterfactual regret minimization (RCFR) [Waugh et al., 2015; Morrill, 2016].

In contrast to (tabular) counterfactual regret minimization (CFR)
[Zinkevich et al., 2007], RCFR replaces the table of regrets that generate the
current policy profile with a profile of regression models. The average
policy is still tracked exactly with a full game-size table. The exploitability
of the average policy in zero-sum games decreases as the model accuracy and
the number of iterations increase [Waugh et al., 2015; Morrill, 2016]. As long
as the regression model errors decrease across iterations, the average policy
converges toward a Nash equilibrium in zero-sum games.

# References

Dustin Morrill. Using Regret Estimation to Solve Games Compactly.
    M.Sc. thesis, Computing Science Department, University of Alberta,
    Apr 1, 2016, Edmonton Alberta, Canada.
Kevin Waugh, Dustin Morrill, J. Andrew Bagnell, and Michael Bowling.
    Solving Games with Functional Regret Estimation. At the Twenty-Ninth AAAI
    Conference on Artificial Intelligence, January 25-29, 2015, Austin Texas,
    USA. Pages 2138-2145.
Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione.
    Regret Minimization in Games with Incomplete Information.
    At Advances in Neural Information Processing Systems 20 (NeurIPS). 2007.
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def tensor_to_matrix(tensor):
    if False:
        print('Hello World!')
    'Converts `tensor` to a matrix (a rank-2 tensor) or raises an exception.\n\n  Args:\n    tensor: The tensor to convert.\n\n  Returns:\n    A TensorFlow matrix (rank-2 `tf.Tensor`).\n  Raises:\n    ValueError: If `tensor` cannot be trivially converted to a matrix, i.e.\n      `tensor` has a rank > 2.\n  '
    tensor = tf.convert_to_tensor(tensor)
    rank = tensor.shape.rank
    if rank > 2:
        raise ValueError('Tensor {} cannot be converted into a matrix as it is rank {} > 2.'.format(tensor, rank))
    elif rank < 2:
        num_columns = 1 if rank == 0 else tensor.shape[0].value
        tensor = tf.reshape(tensor, [1, num_columns])
    return tensor

def with_one_hot_action_features(state_features, legal_actions, num_distinct_actions):
    if False:
        while True:
            i = 10
    'Constructs features for each sequence by extending state features.\n\n  Sequences features are constructed by concatenating one-hot features\n  indicating each action to the information state features and stacking them.\n\n  Args:\n    state_features: The features for the information state alone. Must be a\n      `tf.Tensor` with a rank less than or equal to (if batched) 2.\n    legal_actions: The list of legal actions in this state. Determines the\n      number of rows in the returned feature matrix.\n    num_distinct_actions: The number of globally distinct actions in the game.\n      Determines the length of the action feature vector concatenated onto the\n      state features.\n\n  Returns:\n    A `tf.Tensor` feature matrix with one row for each sequence and # state\n    features plus `num_distinct_actions`-columns.\n\n  Raises:\n    ValueError: If `state_features` has a rank > 2.\n  '
    state_features = tensor_to_matrix(state_features)
    with_action_features = []
    for action in legal_actions:
        action_features = tf.one_hot([action], num_distinct_actions)
        action_features = tf.tile(action_features, [tf.shape(state_features)[0], 1])
        all_features = tf.concat([state_features, action_features], axis=1)
        with_action_features.append(all_features)
    return tf.concat(with_action_features, axis=0)

def sequence_features(state, num_distinct_actions):
    if False:
        return 10
    "The sequence features at `state`.\n\n  Features are constructed by concatenating `state`'s normalized feature\n  vector with one-hot vectors indicating each action (see\n  `with_one_hot_action_features`).\n\n  Args:\n    state: An OpenSpiel `State`.\n    num_distinct_actions: The number of globally distinct actions in `state`'s\n      game.\n\n  Returns:\n    A `tf.Tensor` feature matrix with one row for each sequence.\n  "
    return with_one_hot_action_features(state.information_state_tensor(), state.legal_actions(), num_distinct_actions)

def num_features(game):
    if False:
        while True:
            i = 10
    'Returns the number of features returned by `sequence_features`.\n\n  Args:\n    game: An OpenSpiel `Game`.\n  '
    return game.information_state_tensor_size() + game.num_distinct_actions()

class RootStateWrapper(object):
    """Analyzes the subgame at a given root state.

  It enumerates features for each player sequence, creates a mapping between
  information states to sequence index offsets, and caches terminal values
  in a dictionary with history string keys.

  Properties:
    root: An OpenSpiel `State`.
    sequence_features: A `list` of sequence feature matrices, one for each
      player. This list uses depth-first, information state-major ordering, so
      sequences are grouped by information state. I.e. the first legal action
      in the first state has index 0, the second action in the same information
      state has index 1, the third action will have index 3, and so on.
      Sequences in the next information state descendant of the first action
      will begin indexing its sequences at the number of legal actions in the
      ancestor information state.
    num_player_sequences: The number of sequences for each player.
    info_state_to_sequence_idx: A `dict` mapping each information state string
      to the `sequence_features` index of the first sequence in the
      corresponding information state.
    terminal_values: A `dict` mapping history strings to terminal values for
      each player.
  """

    def __init__(self, state):
        if False:
            while True:
                i = 10
        self.root = state
        self._num_distinct_actions = len(state.legal_actions_mask(0))
        self.sequence_features = [[] for _ in range(state.num_players())]
        self.num_player_sequences = [0] * state.num_players()
        self.info_state_to_sequence_idx = {}
        self.terminal_values = {}
        self._walk_descendants(state)
        self.sequence_features = [tf.concat(rows, axis=0) for rows in self.sequence_features]

    def _walk_descendants(self, state):
        if False:
            return 10
        'Records information about `state` and its descendants.'
        if state.is_terminal():
            self.terminal_values[state.history_str()] = np.array(state.returns())
            return
        elif state.is_chance_node():
            for (action, _) in state.chance_outcomes():
                self._walk_descendants(state.child(action))
            return
        player = state.current_player()
        info_state = state.information_state_string(player)
        actions = state.legal_actions()
        if info_state not in self.info_state_to_sequence_idx:
            n = self.num_player_sequences[player]
            self.info_state_to_sequence_idx[info_state] = n
            self.sequence_features[player].append(sequence_features(state, self._num_distinct_actions))
            self.num_player_sequences[player] += len(actions)
        for action in actions:
            self._walk_descendants(state.child(action))

    def sequence_weights_to_policy(self, sequence_weights, state):
        if False:
            print('Hello World!')
        "Returns a behavioral policy at `state` from sequence weights.\n\n    Args:\n      sequence_weights: An array of non-negative weights, one for each of\n        `state.current_player()`'s sequences in `state`'s game.\n      state: An OpenSpiel `State` that represents an information state in an\n        alternating-move game.\n\n    Returns:\n      A `np.array<double>` probability distribution representing the policy in\n      `state` encoded by `sequence_weights`. Weights corresponding to actions\n      in `state` are normalized by their sum.\n\n    Raises:\n      ValueError: If there are too few sequence weights at `state`.\n    "
        info_state = state.information_state_string()
        sequence_offset = self.info_state_to_sequence_idx[info_state]
        actions = state.legal_actions()
        sequence_idx_end = sequence_offset + len(actions)
        weights = sequence_weights[sequence_offset:sequence_idx_end]
        if len(weights) < len(actions):
            raise ValueError('Invalid policy: Policy {player} at sequence offset {sequence_offset} has only {policy_len} elements but there are {num_actions} legal actions.'.format(player=state.current_player(), sequence_offset=sequence_offset, policy_len=len(weights), num_actions=len(actions)))
        return normalized_by_sum(weights)

    def sequence_weights_to_policy_fn(self, player_sequence_weights):
        if False:
            return 10
        "Returns a policy function based on sequence weights for each player.\n\n    Args:\n      player_sequence_weights: A list of weight arrays, one for each player.\n        Each array should have a weight for each of that player's sequences in\n        `state`'s game.\n\n    Returns:\n      A `State` -> `np.array<double>` function. The output of this function is\n        a probability distribution that represents the policy at the given\n        `State` encoded by `player_sequence_weights` according to\n        `sequence_weights_to_policy`.\n    "

        def policy_fn(state):
            if False:
                for i in range(10):
                    print('nop')
            player = state.current_player()
            return self.sequence_weights_to_policy(player_sequence_weights[player], state)
        return policy_fn

    def sequence_weights_to_tabular_profile(self, player_sequence_weights):
        if False:
            i = 10
            return i + 15
        'Returns the tabular profile-form of `player_sequence_weights`.'
        return sequence_weights_to_tabular_profile(self.root, self.sequence_weights_to_policy_fn(player_sequence_weights))

    def counterfactual_regrets_and_reach_weights(self, regret_player, reach_weight_player, *sequence_weights):
        if False:
            while True:
                i = 10
        'Returns counterfactual regrets and reach weights as a tuple.\n\n    Args:\n      regret_player: The player for whom counterfactual regrets are computed.\n      reach_weight_player: The player for whom reach weights are computed.\n      *sequence_weights: A list of non-negative sequence weights for each player\n        determining the policy profile. Behavioral policies are generated by\n        normalizing sequence weights corresponding to actions in each\n        information state by their sum.\n\n    Returns:\n      The counterfactual regrets and reach weights as an `np.array`-`np.array`\n        tuple.\n\n    Raises:\n      ValueError: If there are too few sequence weights at any information state\n        for any player.\n    '
        num_players = len(sequence_weights)
        regrets = np.zeros(self.num_player_sequences[regret_player])
        reach_weights = np.zeros(self.num_player_sequences[reach_weight_player])

        def _walk_descendants(state, reach_probabilities, chance_reach_probability):
            if False:
                print('Hello World!')
            "Compute `state`'s counterfactual regrets and reach weights.\n\n      Args:\n        state: An OpenSpiel `State`.\n        reach_probabilities: The probability that each player plays to reach\n          `state`'s history.\n        chance_reach_probability: The probability that all chance outcomes in\n          `state`'s history occur.\n\n      Returns:\n        The counterfactual value of `state`'s history.\n      Raises:\n        ValueError if there are too few sequence weights at any information\n        state for any player.\n      "
            if state.is_terminal():
                player_reach = np.prod(reach_probabilities[:regret_player]) * np.prod(reach_probabilities[regret_player + 1:])
                counterfactual_reach_prob = player_reach * chance_reach_probability
                u = self.terminal_values[state.history_str()]
                return u[regret_player] * counterfactual_reach_prob
            elif state.is_chance_node():
                v = 0.0
                for (action, action_prob) in state.chance_outcomes():
                    v += _walk_descendants(state.child(action), reach_probabilities, chance_reach_probability * action_prob)
                return v
            player = state.current_player()
            info_state = state.information_state_string(player)
            sequence_idx_offset = self.info_state_to_sequence_idx[info_state]
            actions = state.legal_actions(player)
            sequence_idx_end = sequence_idx_offset + len(actions)
            my_sequence_weights = sequence_weights[player][sequence_idx_offset:sequence_idx_end]
            if len(my_sequence_weights) < len(actions):
                raise ValueError('Invalid policy: Policy {player} at sequence offset {sequence_idx_offset} has only {policy_len} elements but there are {num_actions} legal actions.'.format(player=player, sequence_idx_offset=sequence_idx_offset, policy_len=len(my_sequence_weights), num_actions=len(actions)))
            policy = normalized_by_sum(my_sequence_weights)
            action_values = np.zeros(len(actions))
            state_value = 0.0
            is_reach_weight_player_node = player == reach_weight_player
            is_regret_player_node = player == regret_player
            reach_prob = reach_probabilities[player]
            for (action_idx, action) in enumerate(actions):
                action_prob = policy[action_idx]
                next_reach_prob = reach_prob * action_prob
                if is_reach_weight_player_node:
                    reach_weight_player_plays_down_this_line = next_reach_prob > 0
                    if not reach_weight_player_plays_down_this_line:
                        continue
                    sequence_idx = sequence_idx_offset + action_idx
                    reach_weights[sequence_idx] += next_reach_prob
                reach_probabilities[player] = next_reach_prob
                action_value = _walk_descendants(state.child(action), reach_probabilities, chance_reach_probability)
                if is_regret_player_node:
                    state_value = state_value + action_prob * action_value
                else:
                    state_value = state_value + action_value
                action_values[action_idx] = action_value
            reach_probabilities[player] = reach_prob
            if is_regret_player_node:
                regrets[sequence_idx_offset:sequence_idx_end] += action_values - state_value
            return state_value
        _walk_descendants(self.root, np.ones(num_players), 1.0)
        return (regrets, reach_weights)

def normalized_by_sum(v, axis=0, mutate=False):
    if False:
        print('Hello World!')
    'Divides each element of `v` along `axis` by the sum of `v` along `axis`.\n\n  Assumes `v` is non-negative. Sets of `v` elements along `axis` that sum to\n  zero are normalized to `1 / v.shape[axis]` (a uniform distribution).\n\n  Args:\n    v: Non-negative array of values.\n    axis: An integer axis.\n    mutate: Whether or not to store the result in `v`.\n\n  Returns:\n    The normalized array.\n  '
    v = np.asarray(v)
    denominator = v.sum(axis=axis, keepdims=True)
    denominator_is_zero = denominator == 0
    denominator += v.shape[axis] * denominator_is_zero
    if mutate:
        v += denominator_is_zero
        v /= denominator
    else:
        v = (v + denominator_is_zero) / denominator
    return v

def relu(v):
    if False:
        return 10
    'Returns the element-wise maximum between `v` and 0.'
    return np.maximum(v, 0)

def _descendant_states(state, depth_limit, depth, include_terminals, include_chance_states):
    if False:
        print('Hello World!')
    'Recursive descendant state generator.\n\n  Decision states are always yielded.\n\n  Args:\n    state: The current state.\n    depth_limit: The descendant depth limit. Zero will ensure only\n      `initial_state` is generated and negative numbers specify the absence of a\n      limit.\n    depth: The current descendant depth.\n    include_terminals: Whether or not to include terminal states.\n    include_chance_states: Whether or not to include chance states.\n\n  Yields:\n    `State`, a state that is `initial_state` or one of its descendants.\n  '
    if state.is_terminal():
        if include_terminals:
            yield state
        return
    if depth > depth_limit >= 0:
        return
    if not state.is_chance_node() or include_chance_states:
        yield state
    for action in state.legal_actions():
        state_for_search = state.child(action)
        for substate in _descendant_states(state_for_search, depth_limit, depth + 1, include_terminals, include_chance_states):
            yield substate

def all_states(initial_state, depth_limit=-1, include_terminals=False, include_chance_states=False):
    if False:
        print('Hello World!')
    'Generates states from `initial_state`.\n\n  Generates the set of states that includes only the `initial_state` and its\n  descendants that satisfy the inclusion criteria specified by the remaining\n  parameters. Decision states are always included.\n\n  Args:\n    initial_state: The initial state from which to generate states.\n    depth_limit: The descendant depth limit. Zero will ensure only\n      `initial_state` is generated and negative numbers specify the absence of a\n      limit. Defaults to no limit.\n    include_terminals: Whether or not to include terminal states. Defaults to\n      `False`.\n    include_chance_states: Whether or not to include chance states. Defaults to\n      `False`.\n\n  Returns:\n    A generator that yields the `initial_state` and its descendants that\n    satisfy the inclusion criteria specified by the remaining parameters.\n  '
    return _descendant_states(state=initial_state, depth_limit=depth_limit, depth=0, include_terminals=include_terminals, include_chance_states=include_chance_states)

def sequence_weights_to_tabular_profile(root, policy_fn):
    if False:
        for i in range(10):
            print('nop')
    'Returns the `dict` of `list`s of action-prob pairs-form of `policy_fn`.'
    tabular_policy = {}
    players = range(root.num_players())
    for state in all_states(root):
        for player in players:
            legal_actions = state.legal_actions(player)
            if len(legal_actions) < 1:
                continue
            info_state = state.information_state_string(player)
            if info_state in tabular_policy:
                continue
            my_policy = policy_fn(state)
            tabular_policy[info_state] = list(zip(legal_actions, my_policy))
    return tabular_policy

@tf.function
def feedforward_evaluate(layers, x, use_skip_connections=False, hidden_are_factored=False):
    if False:
        while True:
            i = 10
    'Evaluates `layers` as a feedforward neural network on `x`.\n\n  Args:\n    layers: The neural network layers (`tf.Tensor` -> `tf.Tensor` callables).\n    x: The array-like input to evaluate. Must be trivially convertible to a\n      matrix (tensor rank <= 2).\n    use_skip_connections: Whether or not to use skip connections between layers.\n      If the layer input has too few features to be added to the layer output,\n      then the end of input is padded with zeros. If it has too many features,\n      then the input is truncated.\n    hidden_are_factored: Whether or not hidden logical layers are factored into\n      two separate linear transformations stored as adjacent elements of\n      `layers`.\n\n  Returns:\n    The `tf.Tensor` evaluation result.\n\n  Raises:\n    ValueError: If `x` has a rank greater than 2.\n  '
    x = tensor_to_matrix(x)
    i = 0
    while i < len(layers) - 1:
        y = layers[i](x)
        i += 1
        if hidden_are_factored:
            y = layers[i](y)
            i += 1
        if use_skip_connections:
            my_num_features = x.shape[1].value
            padding = y.shape[1].value - my_num_features
            if padding > 0:
                zeros = tf.zeros([tf.shape(x)[0], padding])
                x = tf.concat([x, zeros], axis=1)
            elif padding < 0:
                x = tf.strided_slice(x, [0, 0], [tf.shape(x)[0], y.shape[1].value])
            y = x + y
        x = y
    return layers[-1](x)

class DeepRcfrModel(object):
    """A flexible deep feedforward RCFR model class.

  Properties:
    layers: The `tf.keras.Layer` layers describing this  model.
    trainable_variables: The trainable `tf.Variable`s in this model's `layers`.
    losses: This model's layer specific losses (e.g. regularizers).
  """

    def __init__(self, game, num_hidden_units, num_hidden_layers=1, num_hidden_factors=0, hidden_activation=tf.nn.relu, use_skip_connections=False, regularizer=None):
        if False:
            print('Hello World!')
        'Creates a new `DeepRcfrModel.\n\n    Args:\n      game: The OpenSpiel game being solved.\n      num_hidden_units: The number of units in each hidden layer.\n      num_hidden_layers: The number of hidden layers. Defaults to 1.\n      num_hidden_factors: The number of hidden factors or the matrix rank of the\n        layer. If greater than zero, hidden layers will be split into two\n        separate linear transformations, the first with\n        `num_hidden_factors`-columns and the second with\n        `num_hidden_units`-columns. The result is that the logical hidden layer\n        is a rank-`num_hidden_units` matrix instead of a rank-`num_hidden_units`\n        matrix. When `num_hidden_units < num_hidden_units`, this is effectively\n        implements weight sharing. Defaults to 0.\n      hidden_activation: The activation function to apply over hidden layers.\n        Defaults to `tf.nn.relu`.\n      use_skip_connections: Whether or not to apply skip connections (layer\n        output = layer(x) + x) on hidden layers. Zero padding or truncation is\n        used to match the number of columns on layer inputs and outputs.\n      regularizer: A regularizer to apply to each layer. Defaults to `None`.\n    '
        self._use_skip_connections = use_skip_connections
        self._hidden_are_factored = num_hidden_factors > 0
        self.layers = []
        for _ in range(num_hidden_layers):
            if self._hidden_are_factored:
                self.layers.append(tf.keras.layers.Dense(num_hidden_factors, use_bias=True, kernel_regularizer=regularizer))
            self.layers.append(tf.keras.layers.Dense(num_hidden_units, use_bias=True, activation=hidden_activation, kernel_regularizer=regularizer))
        self.layers.append(tf.keras.layers.Dense(1, use_bias=True, kernel_regularizer=regularizer))
        x = tf.zeros([1, num_features(game)])
        for layer in self.layers:
            x = layer(x)
        self.trainable_variables = sum([layer.trainable_variables for layer in self.layers], [])
        self.losses = sum([layer.losses for layer in self.layers], [])

    def __call__(self, x):
        if False:
            while True:
                i = 10
        'Evaluates this model on `x`.'
        return feedforward_evaluate(layers=self.layers, x=x, use_skip_connections=self._use_skip_connections, hidden_are_factored=self._hidden_are_factored)

class _RcfrSolver(object):
    """An abstract RCFR solver class.

  Requires that subclasses implement `evaluate_and_update_policy`.
  """

    def __init__(self, game, models, truncate_negative=False, session=None):
        if False:
            print('Hello World!')
        'Creates a new `_RcfrSolver`.\n\n    Args:\n      game: An OpenSpiel `Game`.\n      models: Current policy models (optimizable array-like -> `tf.Tensor`\n        callables) for both players.\n      truncate_negative: Whether or not to truncate negative (approximate)\n        cumulative regrets to zero to implement RCFR+. Defaults to `False`.\n      session: A TensorFlow `Session` to convert sequence weights from\n        `tf.Tensor`s produced by `models` to `np.array`s. If `None`, it is\n        assumed that eager mode is enabled. Defaults to `None`.\n    '
        self._game = game
        self._models = models
        self._truncate_negative = truncate_negative
        self._root_wrapper = RootStateWrapper(game.new_initial_state())
        self._session = session
        self._cumulative_seq_probs = [np.zeros(n) for n in self._root_wrapper.num_player_sequences]

    def _sequence_weights(self, player=None):
        if False:
            return 10
        'Returns regret-like weights for each sequence as an `np.array`.\n\n    Negative weights are truncated to zero.\n\n    Args:\n      player: The player to compute weights for, or both if `player` is `None`.\n        Defaults to `None`.\n    '
        if player is None:
            return [self._sequence_weights(player) for player in range(self._game.num_players())]
        else:
            tensor = tf.nn.relu(tf.squeeze(self._models[player](self._root_wrapper.sequence_features[player])))
            return tensor.numpy() if self._session is None else self._session(tensor)

    def evaluate_and_update_policy(self, train_fn):
        if False:
            while True:
                i = 10
        'Performs a single step of policy evaluation and policy improvement.\n\n    Args:\n      train_fn: A (model, `tf.data.Dataset`) function that trains the given\n        regression model to accurately reproduce the x to y mapping given x-y\n        data.\n\n    Raises:\n      NotImplementedError: If not overridden by child class.\n    '
        raise NotImplementedError()

    def current_policy(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the current policy profile.\n\n    Returns:\n      A `dict<info state, list<Action, probability>>` that maps info state\n      strings to `Action`-probability pairs describing each player's policy.\n    "
        return self._root_wrapper.sequence_weights_to_tabular_profile(self._sequence_weights())

    def average_policy(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns the average of all policies iterated.\n\n    This average policy converges toward a Nash policy as the number of\n    iterations increases as long as the regret prediction error decreases\n    continually [Morrill, 2016].\n\n    The policy is computed using the accumulated policy probabilities computed\n    using `evaluate_and_update_policy`.\n\n    Returns:\n      A `dict<info state, list<Action, probability>>` that maps info state\n      strings to (Action, probability) pairs describing each player's policy.\n    "
        return self._root_wrapper.sequence_weights_to_tabular_profile(self._cumulative_seq_probs)

    def _previous_player(self, player):
        if False:
            for i in range(10):
                print('nop')
        'The previous player in the turn ordering.'
        return player - 1 if player > 0 else self._game.num_players() - 1

    def _average_policy_update_player(self, regret_player):
        if False:
            print('Hello World!')
        'The player for whom the average policy should be updated.'
        return self._previous_player(regret_player)

class RcfrSolver(_RcfrSolver):
    """RCFR with an effectively infinite regret data buffer.

  Exact or bootstrapped cumulative regrets are stored as if an infinitely
  large data buffer. The average strategy is updated and stored in a full
  game-size table. Reproduces the RCFR versions used in experiments by
  Waugh et al. [2015] and Morrill [2016] except that this class does not
  restrict the user to regression tree models.
  """

    def __init__(self, game, models, bootstrap=None, truncate_negative=False, session=None):
        if False:
            while True:
                i = 10
        self._bootstrap = bootstrap
        super(RcfrSolver, self).__init__(game, models, truncate_negative=truncate_negative, session=session)
        self._regret_targets = [np.zeros(n) for n in self._root_wrapper.num_player_sequences]

    def evaluate_and_update_policy(self, train_fn):
        if False:
            for i in range(10):
                print('nop')
        'Performs a single step of policy evaluation and policy improvement.\n\n    Args:\n      train_fn: A (model, `tf.data.Dataset`) function that trains the given\n        regression model to accurately reproduce the x to y mapping given x-y\n        data.\n    '
        sequence_weights = self._sequence_weights()
        player_seq_features = self._root_wrapper.sequence_features
        for regret_player in range(self._game.num_players()):
            seq_prob_player = self._average_policy_update_player(regret_player)
            (regrets, seq_probs) = self._root_wrapper.counterfactual_regrets_and_reach_weights(regret_player, seq_prob_player, *sequence_weights)
            if self._bootstrap:
                self._regret_targets[regret_player][:] = sequence_weights[regret_player]
            if self._truncate_negative:
                regrets = np.maximum(-relu(self._regret_targets[regret_player]), regrets)
            self._regret_targets[regret_player] += regrets
            self._cumulative_seq_probs[seq_prob_player] += seq_probs
            targets = tf.expand_dims(self._regret_targets[regret_player], axis=1)
            data = tf.data.Dataset.from_tensor_slices((player_seq_features[regret_player], targets))
            regret_player_model = self._models[regret_player]
            train_fn(regret_player_model, data)
            sequence_weights[regret_player] = self._sequence_weights(regret_player)

class ReservoirBuffer(object):
    """A generic reservoir buffer data structure.

  After every insertion, its contents represents a `size`-size uniform
  random sample from the stream of candidates that have been encountered.
  """

    def __init__(self, size):
        if False:
            i = 10
            return i + 15
        self.size = size
        self.num_elements = 0
        self._buffer = np.full([size], None, dtype=object)
        self._num_candidates = 0

    @property
    def buffer(self):
        if False:
            for i in range(10):
                print('nop')
        return self._buffer[:self.num_elements]

    def insert(self, candidate):
        if False:
            return 10
        'Consider this `candidate` for inclusion in this sampling buffer.'
        self._num_candidates += 1
        if self.num_elements < self.size:
            self._buffer[self.num_elements] = candidate
            self.num_elements += 1
            return
        idx = np.random.choice(self._num_candidates)
        if idx < self.size:
            self._buffer[idx] = candidate

    def insert_all(self, candidates):
        if False:
            print('Hello World!')
        'Consider all `candidates` for inclusion in this sampling buffer.'
        for candidate in candidates:
            self.insert(candidate)

    def num_available_spaces(self):
        if False:
            i = 10
            return i + 15
        'The number of freely available spaces in this buffer.'
        return self.size - self.num_elements

class ReservoirRcfrSolver(_RcfrSolver):
    """RCFR with a reservoir buffer for storing regret data.

  The average strategy is updated and stored in a full game-size table.
  """

    def __init__(self, game, models, buffer_size, truncate_negative=False, session=None):
        if False:
            i = 10
            return i + 15
        self._buffer_size = buffer_size
        super(ReservoirRcfrSolver, self).__init__(game, models, truncate_negative=truncate_negative, session=session)
        self._reservoirs = [ReservoirBuffer(self._buffer_size) for _ in range(game.num_players())]

    def evaluate_and_update_policy(self, train_fn):
        if False:
            i = 10
            return i + 15
        'Performs a single step of policy evaluation and policy improvement.\n\n    Args:\n      train_fn: A (model, `tf.data.Dataset`) function that trains the given\n        regression model to accurately reproduce the x to y mapping given x-y\n        data.\n    '
        sequence_weights = self._sequence_weights()
        player_seq_features = self._root_wrapper.sequence_features
        for regret_player in range(self._game.num_players()):
            seq_prob_player = self._average_policy_update_player(regret_player)
            (regrets, seq_probs) = self._root_wrapper.counterfactual_regrets_and_reach_weights(regret_player, seq_prob_player, *sequence_weights)
            if self._truncate_negative:
                regrets = np.maximum(-relu(sequence_weights[regret_player]), regrets)
            next_data = list(zip(player_seq_features[regret_player], tf.expand_dims(regrets, 1)))
            self._reservoirs[regret_player].insert_all(next_data)
            self._cumulative_seq_probs[seq_prob_player] += seq_probs
            my_buffer = tuple((tf.stack(a) for a in zip(*self._reservoirs[regret_player].buffer)))
            data = tf.data.Dataset.from_tensor_slices(my_buffer)
            regret_player_model = self._models[regret_player]
            train_fn(regret_player_model, data)
            sequence_weights[regret_player] = self._sequence_weights(regret_player)