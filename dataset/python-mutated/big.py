"""Big tensor games."""
from absl import logging
import numpy as np
from open_spiel.python.algorithms.adidas_utils.helpers import misc

class TensorGame(object):
    """Tensor Game."""

    def __init__(self, pt, seed=None):
        if False:
            return 10
        'Ctor. Inits payoff tensor (players x actions x ... np.array).\n\n    Args:\n      pt: payoff tensor, np.array\n      seed: seed for random number generator, used if computing best responses\n    '
        if np.any(pt < 0.0):
            raise ValueError('Payoff tensor must contain non-negative values')
        self.pt = pt
        self.seed = seed
        self.random = np.random.RandomState(seed)

    def num_players(self):
        if False:
            i = 10
            return i + 15
        return self.pt.shape[0]

    def num_strategies(self):
        if False:
            print('Hello World!')
        return self.pt.shape[1:]

    def payoff_tensor(self):
        if False:
            return 10
        return self.pt

    def get_payoffs_for_strategies(self, policies):
        if False:
            for i in range(10):
                print('nop')
        'Return vector of payoffs for all players given list of strategies.\n\n    Args:\n      policies: list of integers indexing strategies for each player\n    Returns:\n      np.array (length num players) of payoffs\n    '
        return self.pt[tuple([slice(None)] + policies)]

    def best_response(self, mixed_strategy, return_exp=False):
        if False:
            i = 10
            return i + 15
        'Return best response and its superiority over the current strategy.\n\n    Args:\n      mixed_strategy: np.ndarray (distribution over strategies)\n      return_exp: bool, whether to return how much best response exploits the\n        given mixed strategy (default is False)\n    Returns:\n      br: int, index of strategy (ties split randomly)\n      exp: u(br) - u(mixed_strategy)\n    '
        logging.warn('Assumes symmetric game! Returns br for player 0.')
        gradient = misc.pt_reduce(self.pt[0], [mixed_strategy] * self.num_players(), [0])
        br = misc.argmax(self.random, gradient)
        exp = gradient.max() - gradient.dot(mixed_strategy)
        if return_exp:
            return (br, exp)
        else:
            return br

    def best_population_response(self, dist, policies):
        if False:
            i = 10
            return i + 15
        'Returns the best response to the current population of policies.\n\n    Args:\n      dist: np.ndarray, distribution over policies\n      policies: list of integers indexing strategies for each player\n    Returns:\n      best response, exploitability tuple (see best_response)\n    '
        ns = self.num_strategies()
        mixed_strat = np.zeros(ns)
        for (pure_strat, prob) in zip(policies, dist):
            mixed_strat[pure_strat] += prob
        return self.best_response(mixed_strat)

class ElFarol(TensorGame):
    """N-Player, 2-Action symmetric game with unique symmetric Nash."""

    def __init__(self, n=2, c=0.5, B=0, S=1, G=2, seed=None):
        if False:
            while True:
                i = 10
        "Ctor. Initializes payoff tensor (N x (2,) * N np.array).\n\n    See Section 3.1, The El Farol Stage Game in\n    http://www.econ.ed.ac.uk/papers/id186_esedps.pdf\n\n    action 0: go to bar\n    action 1: avoid bar\n\n    Args:\n      n: int, number of players\n      c: float, threshold for `crowded' as a fraction of number of players\n      B: float, payoff for going to a crowded bar\n      S: float, payoff for staying at home\n      G: float, payoff for going to an uncrowded bar\n      seed: seed for random number generator, used if computing best responses\n    "
        assert G > S > B, 'Game parameters must satisfy G > S > B.'
        pt = np.zeros((n,) + (2,) * n)
        for idx in np.ndindex(pt.shape):
            p = idx[0]
            a = idx[1:]
            a_i = a[p]
            go_to_bar = a_i < 1
            crowded = n - 1 - sum(a) + a_i >= c * n
            if go_to_bar and (not crowded):
                pt[idx] = G
            elif go_to_bar and crowded:
                pt[idx] = B
            else:
                pt[idx] = S
        super().__init__(pt, seed)