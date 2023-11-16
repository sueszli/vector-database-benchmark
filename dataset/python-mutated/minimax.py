"""Implements the min-max algorithm with alpha-beta pruning.

Solves perfect play for deterministic, 2-players, perfect-information 0-sum
games.

See for example https://en.wikipedia.org/wiki/Alpha-beta_pruning
"""
import pyspiel

def _alpha_beta(state, depth, alpha, beta, value_function, maximizing_player_id):
    if False:
        return 10
    'An alpha-beta algorithm.\n\n  Implements a min-max algorithm with alpha-beta pruning.\n  See for example https://en.wikipedia.org/wiki/Alpha-beta_pruning\n\n  Arguments:\n    state: The current state node of the game.\n    depth: The maximum depth for the min/max search.\n    alpha: best value that the MAX player can guarantee (if the value is <= than\n      alpha, the MAX player will avoid it).\n    beta: the best value that the MIN currently can guarantee (if the value is\n      >= than beta, the MIN player will avoid it).\n    value_function: An optional function mapping a Spiel `State` to a numerical\n      value, to be used as the value of the maximizing player for a node when we\n      reach `maximum_depth` and the node is not terminal.\n    maximizing_player_id: The id of the MAX player. The other player is assumed\n      to be MIN.\n\n  Returns:\n    A tuple of the optimal value of the sub-game starting in state\n    (given alpha/beta) and the move that achieved it.\n\n  Raises:\n    NotImplementedError: If we reach the maximum depth. Given we have no value\n      function for a non-terminal node, we cannot break early.\n  '
    if state.is_terminal():
        return (state.player_return(maximizing_player_id), None)
    if depth == 0 and value_function is None:
        raise NotImplementedError('We assume we can walk the full depth of the tree. Try increasing the maximum_depth or provide a value_function.')
    if depth == 0:
        return (value_function(state), None)
    player = state.current_player()
    best_action = -1
    if player == maximizing_player_id:
        value = -float('inf')
        for action in state.legal_actions():
            child_state = state.clone()
            child_state.apply_action(action)
            (child_value, _) = _alpha_beta(child_state, depth - 1, alpha, beta, value_function, maximizing_player_id)
            if child_value > value:
                value = child_value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return (value, best_action)
    else:
        value = float('inf')
        for action in state.legal_actions():
            child_state = state.clone()
            child_state.apply_action(action)
            (child_value, _) = _alpha_beta(child_state, depth - 1, alpha, beta, value_function, maximizing_player_id)
            if child_value < value:
                value = child_value
                best_action = action
            beta = min(beta, value)
            if alpha >= beta:
                break
        return (value, best_action)

def alpha_beta_search(game, state=None, value_function=None, maximum_depth=30, maximizing_player_id=None):
    if False:
        i = 10
        return i + 15
    'Solves deterministic, 2-players, perfect-information 0-sum game.\n\n  For small games only! Please use keyword arguments for optional arguments.\n\n  Arguments:\n    game: The game to analyze, as returned by `load_game`.\n    state: The state to run from, as returned by `game.new_initial_state()`.  If\n      none is specified, then the initial state is assumed.\n    value_function: An optional function mapping a Spiel `State` to a numerical\n      value, to be used as the value of the maximizing player for a node when we\n      reach `maximum_depth` and the node is not terminal.\n    maximum_depth: The maximum depth to search over. When this depth is reached,\n      an exception will be raised.\n    maximizing_player_id: The id of the MAX player. The other player is assumed\n      to be MIN. The default (None) will suppose the player at the root to be\n      the MAX player.\n\n  Returns:\n    A tuple containing the value of the game for the maximizing player when\n    both player play optimally, and the action that achieves this value.\n  '
    game_info = game.get_type()
    if game.num_players() != 2:
        raise ValueError('Game must be a 2-player game')
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError('The game must be a Deterministic one, not {}'.format(game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError('The game must be a perfect information one, not {}'.format(game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError('The game must be turn-based, not {}'.format(game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError('The game must be 0-sum, not {}'.format(game.utility))
    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()
    return _alpha_beta(state.clone(), maximum_depth, alpha=-float('inf'), beta=float('inf'), value_function=value_function, maximizing_player_id=maximizing_player_id)

def expectiminimax(state, depth, value_function, maximizing_player_id):
    if False:
        for i in range(10):
            print('nop')
    'Runs expectiminimax until the specified depth.\n\n  See https://en.wikipedia.org/wiki/Expectiminimax for details.\n\n  Arguments:\n    state: The state to start the search from.\n    depth: The depth of the search (not counting chance nodes).\n    value_function: A value function, taking in a state and returning a value,\n      in terms of the maximizing_player_id.\n    maximizing_player_id: The player running the search (current player at root\n      of the search tree).\n\n  Returns:\n    A tuple (value, best_action) representing the value to the maximizing player\n    and the best action that achieves that value. None is returned as the best\n    action at chance nodes, the depth limit, and terminals.\n  '
    if state.is_terminal():
        return (state.player_return(maximizing_player_id), None)
    if depth == 0:
        return (value_function(state), None)
    if state.is_chance_node():
        value = 0
        for (outcome, prob) in state.chance_outcomes():
            child_state = state.clone()
            child_state.apply_action(outcome)
            (child_value, _) = expectiminimax(child_state, depth, value_function, maximizing_player_id)
            value += prob * child_value
        return (value, None)
    elif state.current_player() == maximizing_player_id:
        value = -float('inf')
        for action in state.legal_actions():
            child_state = state.clone()
            child_state.apply_action(action)
            (child_value, _) = expectiminimax(child_state, depth - 1, value_function, maximizing_player_id)
            if child_value > value:
                value = child_value
                best_action = action
        return (value, best_action)
    else:
        value = float('inf')
        for action in state.legal_actions():
            child_state = state.clone()
            child_state.apply_action(action)
            (child_value, _) = expectiminimax(child_state, depth - 1, value_function, maximizing_player_id)
            if child_value < value:
                value = child_value
                best_action = action
        return (value, best_action)