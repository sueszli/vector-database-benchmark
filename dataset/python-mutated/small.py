"""Small matrix games."""
from absl import logging
import numpy as np
from open_spiel.python.algorithms.adidas_utils.helpers import misc

class MatrixGame(object):
    """Matrix Game."""

    def __init__(self, pt, seed=None):
        if False:
            for i in range(10):
                print('nop')
        'Ctor. Inits payoff tensor (players x actions x ... np.array).\n\n    Args:\n      pt: payoff tensor, np.array\n      seed: seed for random number generator, used if computing best responses\n    '
        if np.any(pt < 0.0):
            raise ValueError('Payoff tensor must contain non-negative values')
        self.pt = pt
        self.seed = seed
        self.random = np.random.RandomState(seed)

    def num_players(self):
        if False:
            return 10
        return self.pt.shape[0]

    def num_strategies(self):
        if False:
            print('Hello World!')
        return self.pt.shape[1:]

    def payoff_tensor(self):
        if False:
            print('Hello World!')
        return self.pt

    def get_payoffs_for_strategies(self, policies):
        if False:
            return 10
        'Return vector of payoffs for all players given list of strategies.\n\n    Args:\n      policies: list of integers indexing strategies for each player\n    Returns:\n      np.array (length num players) of payoffs\n    '
        return self.pt[:, policies[0], policies[1]]

    def best_response(self, mixed_strategy, return_exp=False):
        if False:
            for i in range(10):
                print('nop')
        'Return best response and its superiority over the current strategy.\n\n    Args:\n      mixed_strategy: np.ndarray (distribution over strategies)\n      return_exp: bool, whether to return how much best response exploits the\n        given mixed strategy (default is False)\n    Returns:\n      br: int, index of strategy (ties split randomly)\n      exp: u(br) - u(mixed_strategy)\n    '
        logging.warn('Assumes symmetric game! Returns br for player 0.')
        gradient = self.pt[0].dot(mixed_strategy)
        br = misc.argmax(self.random, gradient)
        exp = gradient.max() - gradient.dot(mixed_strategy)
        if return_exp:
            return (br, exp)
        else:
            return br

    def best_population_response(self, dist, policies):
        if False:
            while True:
                i = 10
        'Returns the best response to the current population of policies.\n\n    Args:\n      dist: np.ndarray, distribution over policies\n      policies: list of integers indexing strategies for each player\n    Returns:\n      best response, exploitability tuple (see best_response)\n    '
        ns = self.num_strategies()
        mixed_strat = np.zeros(ns)
        for (pure_strat, prob) in zip(policies, dist):
            mixed_strat[pure_strat] += prob
        return self.best_response(mixed_strat)

class BiasedGame(MatrixGame):
    """2-Player, 3-Action symmetric game with biased stochastic best responses."""

    def __init__(self, seed=None):
        if False:
            i = 10
            return i + 15
        'Ctor. Initializes payoff tensor (2 x 3 x 3 np.array).\n\n    Args:\n      seed: seed for random number generator, used if computing best responses\n    '
        pt_r = np.array([[0, 0, 0], [1, -2, 0.5], [-2, 1, -1]]) + 2.0
        pt_c = pt_r.T
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        pt /= pt.max()
        super().__init__(pt, seed)

class PrisonersDilemma(MatrixGame):
    """2-Player, 2-Action symmetric prisoner's dilemma."""

    def __init__(self, seed=None):
        if False:
            while True:
                i = 10
        'Ctor. Initializes payoff tensor (2 x 2 x 2 np.array).\n\n    Args:\n      seed: seed for random number generator, used if computing best responses\n    '
        pt_r = np.array([[-1, -3], [0, -2]])
        pt_r -= pt_r.min()
        pt_c = pt_r.T
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        pt /= pt.max()
        super().__init__(pt, seed)

class RockPaperScissors(MatrixGame):
    """2-Player, 3-Action symmetric RPS."""

    def __init__(self, weights=None, seed=None):
        if False:
            return 10
        'Ctor. Initializes payoff tensor (2 x 3 x 3 np.array).\n\n    Args:\n      weights: list of weights (floats) for [rock, paper, scissors]\n      seed: seed for random number generator, used if computing best responses\n    '
        if weights is None:
            weights = np.ones(3)
        (r, p, s) = weights
        pt_r = np.array([[0, -p, r], [p, 0, -s], [-r, s, 0]])
        pt_r -= pt_r.min()
        pt_c = pt_r.T
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        super().__init__(pt, seed)

class SpiralGame(MatrixGame):
    """2-Player, 3-Action symmetric game with spiral dynamics on simplex."""

    def __init__(self, center=None, seed=None):
        if False:
            print('Hello World!')
        'Ctor. Initializes payoff tensor (2 x 3 x 3 np.array).\n\n    Args:\n      center: center of cycle given in [x, y, z] Euclidean coordinates\n      seed: seed for random number generator, used if computing best responses\n    '
        if center is None:
            center = np.ones(3) / 3.0
        elif not (np.sum(center) <= 1 + 1e-08 and np.all(center >= -1e-08)):
            raise ValueError('center must lie on simplex')
        self.center = center
        center = center.reshape((3, 1))
        transform = np.array([[0.5, -0.5, 0], [-0.5, -0.5, 1], [1, 1, 1]]).T
        transform /= np.linalg.norm(transform, axis=0)
        transform_inv = np.linalg.inv(transform)
        cycle = 0.1 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        pt_r = transform.dot(cycle.dot(transform_inv))
        pt_r -= pt_r.dot(center)
        if pt_r.min() < 0:
            pt_r -= pt_r.min()
        pt_c = pt_r.T
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        super().__init__(pt, seed)

class MatchingPennies(MatrixGame):
    """2-Player, 2-Action non-symmetric matching pennies."""

    def __init__(self, bias=1.0, seed=None):
        if False:
            print('Hello World!')
        'Ctor. Initializes payoff tensor (2 x 2 x 2 np.array).\n\n    Args:\n      bias: float, rewards one action (bias) more than the other (1)\n      seed: seed for random number generator, used if computing best responses\n    '
        pt_r = np.array([[1, -1], [-1, bias]])
        pt_c = (-pt_r).T
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        pt -= pt.min()
        pt /= pt.max()
        super().__init__(pt, seed)

class Shapleys(MatrixGame):
    """2-Player, 3-Action non-symmetric Shapleys game."""

    def __init__(self, beta=1.0, seed=None):
        if False:
            while True:
                i = 10
        'Ctor. Initializes payoff tensor (2 x 2 x 2 np.array).\n\n    See Eqn 4 in https://arxiv.org/pdf/1308.4049.pdf.\n\n    Args:\n      beta: float, modifies the game so that the utilities @ Nash are now\n        u_1(Nash) = (1 + beta) / 3 and u_2(Nash) = (1 - beta) / 3\n        where Nash is the joint uniform distribution\n      seed: seed for random number generator, used if computing best responses\n    '
        pt_r = np.array([[1, 0, beta], [beta, 1, 0], [0, beta, 1]])
        pt_c = np.array([[-beta, 1, 0], [0, -beta, 1], [1, 0, -beta]])
        pt = np.stack((pt_r, pt_c), axis=0).astype(float)
        pt -= pt.min()
        pt /= pt.max()
        super().__init__(pt, seed)