"""
Implements a markov chain. Chains are described using a dictionary:

    my_chain = {
        'A': {'A': 0.6,
              'E': 0.4},
        'E': {'A': 0.7,
              'E': 0.3}
    }
"""
import random

def __choose_state(state_map):
    if False:
        i = 10
        return i + 15
    '\n    Choose the next state randomly\n    '
    choice = random.random()
    probability_reached = 0
    for (state, probability) in state_map.items():
        probability_reached += probability
        if probability_reached > choice:
            return state
    return None

def next_state(chain, current_state):
    if False:
        while True:
            i = 10
    '\n    Given a markov-chain, randomly chooses the next state given the current state.\n    '
    next_state_map = chain.get(current_state)
    return __choose_state(next_state_map)

def iterating_markov_chain(chain, state):
    if False:
        i = 10
        return i + 15
    '\n    Yield a sequence of states given a markov chain and the initial state\n    '
    while True:
        state = next_state(chain, state)
        yield state