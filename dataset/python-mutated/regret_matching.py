"""Regret-Matching Algorithm.

This is an N-player implementation of the regret-matching algorithm described in
Hart & Mas-Colell 2000:
https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00153
"""
import numpy as np
from open_spiel.python.algorithms import nfg_utils
INITIAL_REGRET_DENOM = 1000000.0

def _partial_multi_dot(player_payoff_tensor, strategies, index_avoided):
    if False:
        i = 10
        return i + 15
    "Computes a generalized dot product avoiding one dimension.\n\n  This is used to directly get the expected return of a given action, given\n  other players' strategies, for the player indexed by index_avoided.\n  Note that the numpy.dot function is used to compute this product, as it ended\n  up being (Slightly) faster in performance tests than np.tensordot. Using the\n  reduce function proved slower for both np.dot and np.tensordot.\n\n  Args:\n    player_payoff_tensor: payoff tensor for player[index_avoided], of dimension\n      (dim(vector[0]), dim(vector[1]), ..., dim(vector[-1])).\n    strategies: Meta strategy probabilities for each player.\n    index_avoided: Player for which we do not compute the dot product.\n\n  Returns:\n    Vector of expected returns for each action of player [the player indexed by\n      index_avoided].\n  "
    new_axis_order = [index_avoided] + [i for i in range(len(strategies)) if i != index_avoided]
    accumulator = np.transpose(player_payoff_tensor, new_axis_order)
    for i in range(len(strategies) - 1, -1, -1):
        if i != index_avoided:
            accumulator = np.dot(accumulator, strategies[i])
    return accumulator

def _regret_matching_step(payoff_tensors, strategies, regrets, gamma):
    if False:
        while True:
            i = 10
    'Does one step of the projected replicator dynamics algorithm.\n\n  Args:\n    payoff_tensors: List of payoff tensors for each player.\n    strategies: List of the strategies used by each player.\n    regrets: List of cumulative regrets used by each player.\n    gamma: Minimum exploratory probability term.\n\n  Returns:\n    A list of updated strategies for each player.\n  '
    new_strategies = []
    for player in range(len(payoff_tensors)):
        current_payoff_tensor = payoff_tensors[player]
        current_strategy = strategies[player]
        values_per_strategy = _partial_multi_dot(current_payoff_tensor, strategies, player)
        average_return = np.dot(values_per_strategy, current_strategy)
        regrets[player] += values_per_strategy - average_return
        updated_strategy = regrets[player].copy()
        updated_strategy[updated_strategy < 0] = 0.0
        sum_regret = updated_strategy.sum()
        uniform_strategy = np.ones(len(updated_strategy)) / len(updated_strategy)
        if sum_regret > 0:
            updated_strategy /= sum_regret
            updated_strategy = gamma * uniform_strategy + (1 - gamma) * updated_strategy
        else:
            updated_strategy = uniform_strategy
        new_strategies.append(updated_strategy)
    return new_strategies

def regret_matching(payoff_tensors, initial_strategies=None, iterations=int(100000.0), gamma=1e-06, average_over_last_n_strategies=None, **unused_kwargs):
    if False:
        print('Hello World!')
    'Runs regret-matching for the stated number of iterations.\n\n  Args:\n    payoff_tensors: List of payoff tensors for each player.\n    initial_strategies: Initial list of the strategies used by each player, if\n      any. Could be used to speed up the search by providing a good initial\n      solution.\n    iterations: Number of algorithmic steps to take before returning an answer.\n    gamma: Minimum exploratory probability term.\n    average_over_last_n_strategies: Running average window size for average\n      policy computation. If None, use the whole trajectory.\n    **unused_kwargs: Convenient way of exposing an API compatible with other\n      methods with possibly different arguments.\n\n  Returns:\n    RM-computed strategies.\n  '
    number_players = len(payoff_tensors)
    action_space_shapes = payoff_tensors[0].shape
    new_strategies = initial_strategies or [np.ones(action_space_shapes[k]) / action_space_shapes[k] for k in range(number_players)]
    regrets = [np.ones(action_space_shapes[k]) / INITIAL_REGRET_DENOM for k in range(number_players)]
    averager = nfg_utils.StrategyAverager(number_players, action_space_shapes, average_over_last_n_strategies)
    averager.append(new_strategies)
    for _ in range(iterations):
        new_strategies = _regret_matching_step(payoff_tensors, new_strategies, regrets, gamma)
        averager.append(new_strategies)
    return averager.average_strategies()