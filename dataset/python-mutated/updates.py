"""Generic solver for non-symmetric games."""
from absl import logging
import numpy as np
from scipy import special
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability

class Solver(object):
    """Generic Solver."""

    def __init__(self, proj_grad=True, euclidean=False, rnd_init=False, seed=None):
        if False:
            while True:
                i = 10
        'Ctor.'
        self.num_players = None
        self.proj_grad = proj_grad
        self.rnd_init = rnd_init
        self.lrs = (None,)
        self.has_aux = False
        self.euclidean = euclidean
        if euclidean:
            self.update = self.euc_descent_step
        else:
            self.update = self.mirror_descent_step
        self.seed = seed
        self.random = np.random.RandomState(seed)

    def init_vars(self, num_strats, num_players):
        if False:
            for i in range(10):
                print('nop')
        'Initialize solver parameters.'
        self.num_players = num_players
        if len(num_strats) != num_players:
            raise ValueError('Must specify num strategies for each player')
        init_dist = []
        for num_strats_i in num_strats:
            if self.rnd_init:
                init_dist_i = self.random.rand(num_strats_i)
            else:
                init_dist_i = np.ones(num_strats_i)
            init_dist_i /= init_dist_i.sum()
            init_dist.append(init_dist_i)
        return (init_dist,)

    def compute_gradients(self, params, payoff_matrices):
        if False:
            for i in range(10):
                print('nop')
        'Compute and return gradients for all parameters.\n\n    Args:\n      params: e.g., tuple of params (dist,)\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n        values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n        are sorted and arrays should be indexed in the same order\n    Returns:\n      eg., tuple of gradients (grad_dist,)\n    '
        raise NotImplementedError('Should be implemented by specific solver.')

    def exploitability(self, params, payoff_tensor):
        if False:
            print('Hello World!')
        'Compute and return exploitability that solver is minimizing.\n\n    Args:\n      params: e.g., tuple of params (dist,)\n      payoff_tensor: (n x A1 x ... x An) np.array, payoffs for each joint\n        action. can also be list of (A1 x ... x An) np.arrays\n    Returns:\n      float, exploitability of current dist\n    '
        return exploitability.unreg_exploitability(params, payoff_tensor)

    def euc_descent_step(self, params, grads, t):
        if False:
            print('Hello World!')
        'Projected gradient descent on exploitability using Euclidean projection.\n\n    Args:\n      params: tuple of variables to be updated (dist,)\n      grads: tuple of variable gradients (grad_dist,)\n      t: int, solver iteration (unused)\n    Returns:\n      new_params: tuple of update params (new_dist,)\n    '
        del t
        lr_dist = self.lrs[0]
        new_params = []
        for (dist_i, dist_grad_i) in zip(params[0], grads[0]):
            new_dist_i = dist_i - lr_dist * dist_grad_i
            new_dist_i = simplex.euclidean_projection_onto_simplex(new_dist_i)
            new_params.append(new_dist_i)
        return (new_params,)

    def mirror_descent_step(self, params, grads, t):
        if False:
            i = 10
            return i + 15
        'Entropic mirror descent on exploitability.\n\n    Args:\n      params: tuple of variables to be updated (dist - a list of np.arrays)\n      grads: tuple of variable gradients (grad_dist - a list of np.arrays)\n      t: int, solver iteration (unused)\n    Returns:\n      new_params: tuple of update params (new_dist)\n    '
        del t
        lr_dist = self.lrs[0]
        new_params = []
        for (dist_i, dist_grad_i) in zip(params[0], grads[0]):
            new_dist_i = np.clip(dist_i, 0, np.inf)
            new_dist_i = special.softmax(np.log(new_dist_i) - lr_dist * dist_grad_i)
            new_params.append(new_dist_i)
        return (new_params,)