"""Example algorithm to sample some states from a game."""
import random
import pyspiel

def sample_some_states(game, max_states=100, make_distribution_fn=lambda states: [1 / len(states)] * len(states)):
    if False:
        while True:
            i = 10
    'Samples some states in the game.\n\n  This can be run for large games, in contrast to `get_all_states`. It is useful\n  for tests that need to check a predicate only on a subset of the game, since\n  generating the whole game is infeasible.\n\n  Currently only works for sequential games. For simultaneous games and mean\n  field games it returns only the initial state.\n\n  The algorithm maintains a list of states and repeatedly picks a random state\n  from the list to expand until enough states have been sampled.\n\n  Arguments:\n    game: The game to analyze, as returned by `load_game`.\n    max_states: The maximum number of states to return. Negative means no limit.\n    make_distribution_fn: Function that takes a list of states and returns a\n      corresponding distribution (as a list of floats). Only used for mean field\n      games.\n\n  Returns:\n    A `list` of `pyspiel.State`.\n  '
    if game.get_type().dynamics in [pyspiel.GameType.Dynamics.SIMULTANEOUS, pyspiel.GameType.Dynamics.MEAN_FIELD]:
        return [game.new_initial_state()]
    states = []
    unexplored_actions = []
    indexes_with_unexplored_actions = set()

    def add_state(state):
        if False:
            print('Hello World!')
        states.append(state)
        if state.is_terminal():
            unexplored_actions.append(None)
        else:
            indexes_with_unexplored_actions.add(len(states) - 1)
            unexplored_actions.append(set(state.legal_actions()))

    def expand_random_state():
        if False:
            return 10
        index = random.choice(list(indexes_with_unexplored_actions))
        state = states[index]
        if state.is_mean_field_node():
            child = state.clone()
            child.update_distribution(make_distribution_fn(child.distribution_support()))
            indexes_with_unexplored_actions.remove(index)
            return child
        else:
            actions = unexplored_actions[index]
            assert actions, f'Empty actions for state {state}'
            action = random.choice(list(actions))
            actions.remove(action)
            if not actions:
                indexes_with_unexplored_actions.remove(index)
            return state.child(action)
    add_state(game.new_initial_state())
    while len(states) < max_states and indexes_with_unexplored_actions:
        add_state(expand_random_state())
    if not states:
        raise ValueError('get_some_states sampled 0 states!')
    return states