"""Policy Gradient (PG)."""
from absl import logging
import numpy as np
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import updates

class Solver(updates.Solver):
    """PG Solver."""

    def __init__(self, proj_grad=True, euclidean=False, lrs=(0.1,), rnd_init=False, seed=None, **kwargs):
        if False:
            return 10
        'Ctor.'
        del kwargs
        super().__init__(proj_grad, euclidean, rnd_init, seed)
        self.lrs = lrs

    def compute_gradients(self, params, payoff_matrices):
        if False:
            for i in range(10):
                print('nop')
        'Compute and return gradients for all parameters.\n\n    Args:\n      params: tuple of params (dist,), see pg.gradients\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n          values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n          are sorted and arrays should be indexed in the same order\n    Returns:\n      tuple of gradients (grad_dist,), see pg.gradients\n      unregularized exploitability (stochastic estimate)\n      unregularized exploitability (stochastic estimate) *duplicate\n    '
        return gradients(*params, payoff_matrices, self.num_players, self.proj_grad)

    def exploitability(self, params, payoff_matrices):
        if False:
            return 10
        'Policy gradient does not minimize any exploitability so return NaN.\n\n    Args:\n      params: tuple of params (dist,)\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n    Returns:\n      np.NaN\n    '
        return np.NaN

def gradients(dist, payoff_matrices, num_players, proj_grad=True):
    if False:
        while True:
            i = 10
    'Computes exploitablity gradient.\n\n  Args:\n    dist: list of 1-d np.arrays, current estimate of nash distribution\n    payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n        values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n        are sorted and arrays should be indexed in the same order\n    num_players: int, number of players, in case payoff_matrices is abbreviated\n    proj_grad: bool, if True, projects dist gradient onto simplex\n  Returns:\n    gradient of payoff w.r.t. (dist) as tuple\n    unregularized exploitability (stochastic estimate)\n    unregularized exploitability (stochastic estimate) *duplicate\n  '
    grad_dist = []
    unreg_exp = []
    for i in range(num_players):
        nabla_i = np.zeros_like(dist[i])
        for j in range(num_players):
            if j == i:
                continue
            if i < j:
                hess_i_ij = payoff_matrices[i, j][0]
            else:
                hess_i_ij = payoff_matrices[j, i][1].T
            nabla_ij = hess_i_ij.dot(dist[j])
            nabla_i += nabla_ij / float(num_players - 1)
        grad_dist_i = -nabla_i
        if proj_grad:
            grad_dist_i = simplex.project_grad(grad_dist_i)
        grad_dist.append(nabla_i)
        unreg_exp.append(np.max(nabla_i) - nabla_i.dot(dist[i]))
    return ((grad_dist,), np.mean(unreg_exp), np.mean(unreg_exp))