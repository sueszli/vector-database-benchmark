"""Example algorithm to get all states from a game.

The algorithm does not support mean field games where the game evolution depends
on the mean field distribution.
"""
import itertools
from open_spiel.python import games
import pyspiel

def _get_subgames_states(state, all_states, depth_limit, depth, include_terminals, include_chance_states, include_mean_field_states, to_string, stop_if_encountered):
    if False:
        return 10
    'Extract non-chance states for a subgame into the all_states dict.'
    if state.is_terminal():
        if include_terminals:
            state_str = to_string(state)
            if state_str not in all_states:
                all_states[state_str] = state.clone()
        return
    if depth > depth_limit >= 0:
        return
    is_mean_field = state.current_player() == pyspiel.PlayerId.MEAN_FIELD
    if state.is_chance_node() and include_chance_states or (is_mean_field and include_mean_field_states) or (not (state.is_chance_node() or is_mean_field)):
        state_str = to_string(state)
        if state_str not in all_states:
            all_states[state_str] = state.clone()
        elif stop_if_encountered:
            return
    if is_mean_field:
        support = state.distribution_support()
        state_for_search = state.clone()
        support_length = len(support)
        state_for_search.update_distribution([1.0 / support_length for _ in range(support_length)])
        _get_subgames_states(state_for_search, all_states, depth_limit, depth + 1, include_terminals, include_chance_states, include_mean_field_states, to_string, stop_if_encountered)
    elif state.is_simultaneous_node():
        joint_legal_actions = [state.legal_actions(player) for player in range(state.get_game().num_players())]
        for joint_actions in itertools.product(*joint_legal_actions):
            state_for_search = state.clone()
            state_for_search.apply_actions(list(joint_actions))
            _get_subgames_states(state_for_search, all_states, depth_limit, depth + 1, include_terminals, include_chance_states, include_mean_field_states, to_string, stop_if_encountered)
    else:
        for action in state.legal_actions():
            state_for_search = state.child(action)
            _get_subgames_states(state_for_search, all_states, depth_limit, depth + 1, include_terminals, include_chance_states, include_mean_field_states, to_string, stop_if_encountered)

def get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False, include_mean_field_states=False, to_string=lambda s: s.history_str(), stop_if_encountered=True):
    if False:
        i = 10
        return i + 15
    'Gets all states in the game, indexed by their string representation.\n\n  For small games only! Useful for methods that solve the  games explicitly,\n  i.e. value iteration. Use this default implementation with caution as it does\n  a recursive tree walk of the game and could easily fill up memory for larger\n  games or games with long horizons.\n\n  Currently only works for sequential games.\n\n  Arguments:\n    game: The game to analyze, as returned by `load_game`.\n    depth_limit: How deeply to analyze the game tree. Negative means no limit, 0\n      means root-only, etc.\n    include_terminals: If True, include terminal states.\n    include_chance_states: If True, include chance node states.\n    include_mean_field_states: If True, include mean field node states.\n    to_string: The serialization function. We expect this to be\n      `lambda s: s.history_str()` as this enforces perfect recall, but for\n        historical reasons, using `str` is also supported, but the goal is to\n        remove this argument.\n    stop_if_encountered: if this is set, do not keep recursively adding states\n      if this state is already in the list. This allows support for games that\n      have cycles.\n\n  Returns:\n    A `dict` with `to_string(state)` keys and `pyspiel.State` values containing\n    all states encountered traversing the game tree up to the specified depth.\n  '
    root_states = game.new_initial_states()
    all_states = dict()
    for root in root_states:
        _get_subgames_states(state=root, all_states=all_states, depth_limit=depth_limit, depth=0, include_terminals=include_terminals, include_chance_states=include_chance_states, include_mean_field_states=include_mean_field_states, to_string=to_string, stop_if_encountered=stop_if_encountered)
    if not all_states:
        raise ValueError('GetSubgameStates returned 0 states!')
    return all_states