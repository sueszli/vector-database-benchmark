"""Computes the value of a given policy."""
from typing import List, Union
import numpy as np
from open_spiel.python import policy

def _transitions(state, policies):
    if False:
        for i in range(10):
            print('nop')
    'Returns iterator over (action, prob) from the given state.'
    if state.is_chance_node():
        return state.chance_outcomes()
    elif state.is_simultaneous_node():
        return policy.joint_action_probabilities(state, policies)
    else:
        player = state.current_player()
        return policies[player].action_probabilities(state).items()

def policy_value(state, policies: Union[List[policy.Policy], policy.Policy], probability_threshold: float=0):
    if False:
        for i in range(10):
            print('nop')
    'Returns the expected values for the state for players following `policies`.\n\n  Computes the expected value of the`state` for each player, assuming player `i`\n  follows the policy given in `policies[i]`.\n\n  Args:\n    state: A `pyspiel.State`.\n    policies: A `list` of `policy.Policy` objects, one per player for sequential\n      games, one policy for simulatenous games.\n    probability_threshold: only sum over entries with prob greater than this\n      (default: 0).\n\n  Returns:\n    A `numpy.array` containing the expected value for each player.\n  '
    if state.is_terminal():
        return np.array(state.returns())
    else:
        return sum((prob * policy_value(policy.child(state, action), policies) for (action, prob) in _transitions(state, policies) if prob > probability_threshold))