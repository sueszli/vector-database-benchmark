"""Value iteration algorithm for solving a game."""
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import lp_solver
import pyspiel

def _get_future_states(possibilities, state, reach=1.0):
    if False:
        while True:
            i = 10
    'Does a lookahead over chance nodes to all next states after (s,a).\n\n  Also works if there are no chance nodes (i.e. base case).\n\n  Arguments:\n    possibilities:  an empty list, that will be filled with (str(next_state),\n      transition probability) pairs for all possible next states\n    state: the state following some s.apply_action(a), can be a chance node\n    reach: chance reach probability of getting to this point from (s,a)\n  Returns: nothing.\n  '
    if not state.is_chance_node() or state.is_terminal():
        possibilities.append((str(state), reach))
    else:
        assert state.is_chance_node()
        for (outcome, prob) in state.chance_outcomes():
            next_state = state.child(outcome)
            _get_future_states(possibilities, next_state, reach * prob)

def _add_transition(transitions, key, state):
    if False:
        i = 10
        return i + 15
    'Adds action transitions from given state.'
    if state.is_simultaneous_node():
        for p0action in state.legal_actions(0):
            for p1action in state.legal_actions(1):
                next_state = state.clone()
                next_state.apply_actions([p0action, p1action])
                possibilities = []
                _get_future_states(possibilities, next_state)
                transitions[key, p0action, p1action] = possibilities
    else:
        for action in state.legal_actions():
            next_state = state.child(action)
            possibilities = []
            _get_future_states(possibilities, next_state)
            transitions[key, action] = possibilities

def _initialize_maps(states, values, transitions):
    if False:
        i = 10
        return i + 15
    'Initialize the value and transition maps.'
    for (key, state) in states.items():
        if state.is_terminal():
            values[key] = state.player_return(0)
        else:
            values[key] = 0
            _add_transition(transitions, key, state)

def value_iteration(game, depth_limit, threshold, cyclic_game=False):
    if False:
        print('Hello World!')
    'Solves for the optimal value function of a game.\n\n  For small games only! Solves the game using value iteration,\n  with the maximum error for the value function less than threshold.\n  This algorithm works for sequential 1-player games or 2-player zero-sum\n  games, with or without chance nodes.\n\n  Arguments:\n    game: The game to analyze, as returned by `load_game`.\n    depth_limit: How deeply to analyze the game tree. Negative means no limit, 0\n      means root-only, etc.\n    threshold: Maximum error for state values..\n    cyclic_game: set to True if the game has cycles (from state A we can get to\n      state B, and from state B we can get back to state A).\n\n  Returns:\n    A `dict` with string keys and float values, mapping string encoding of\n    states to the values of those states.\n  '
    assert game.num_players() in (1, 2), 'Game must be a 1-player or 2-player game'
    if game.num_players() == 2:
        assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM, '2-player games must be zero sum games'
    assert game.get_type().information == pyspiel.GameType.Information.ONE_SHOT or game.get_type().information == pyspiel.GameType.Information.PERFECT_INFORMATION
    states = get_all_states.get_all_states(game, depth_limit, True, False, to_string=str, stop_if_encountered=cyclic_game)
    values = {}
    transitions = {}
    _initialize_maps(states, values, transitions)
    error = threshold + 1
    min_utility = game.min_utility()
    while error > threshold:
        error = 0
        for (key, state) in states.items():
            if state.is_terminal():
                continue
            elif state.is_simultaneous_node():
                p0_utils = []
                p1_utils = []
                row = 0
                for p0action in state.legal_actions(0):
                    p0_utils.append([])
                    p1_utils.append([])
                    for p1action in state.legal_actions(1):
                        next_states = transitions[key, p0action, p1action]
                        joint_q_value = sum((p * values[next_state] for (next_state, p) in next_states))
                        p0_utils[row].append(joint_q_value)
                        p1_utils[row].append(-joint_q_value)
                    row += 1
                stage_game = pyspiel.create_matrix_game(p0_utils, p1_utils)
                solution = lp_solver.solve_zero_sum_matrix_game(stage_game)
                value = solution[2]
            else:
                player = state.current_player()
                value = min_utility if player == 0 else -min_utility
                for action in state.legal_actions():
                    next_states = transitions[key, action]
                    q_value = sum((p * values[next_state] for (next_state, p) in next_states))
                    if player == 0:
                        value = max(value, q_value)
                    else:
                        value = min(value, q_value)
            error = max(abs(values[key] - value), error)
            values[key] = value
    return values