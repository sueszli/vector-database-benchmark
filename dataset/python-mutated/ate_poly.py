"""Adaptive Tsallis Entropy (ATE) Stochastic Approximate Nash Solver."""
import itertools
from absl import logging
import numpy as np
from scipy import special
from open_spiel.python.algorithms.adidas_utils.helpers import misc
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability as exp

class Solver(object):
    """ATE Solver that constructs a polymatrix approximation to the full game."""

    def __init__(self, p=1.0, proj_grad=True, euclidean=False, cheap=False, lrs=(0.01, 0.1), rnd_init=False, seed=None, **kwargs):
        if False:
            return 10
        'Ctor.'
        del kwargs
        if p < 0.0 or p > 1.0:
            raise ValueError('p must be in [0, 1]')
        self.num_strats = None
        self.num_players = None
        self.p = p
        self.proj_grad = proj_grad
        self.cheap = cheap
        self.rnd_init = rnd_init
        self.lrs = lrs
        self.has_aux = True
        self.aux_errors = []
        self.euclidean = euclidean
        if euclidean:
            self.update = self.euc_descent_step
        else:
            self.update = self.mirror_descent_step
        self.seed = seed
        self.random = np.random.RandomState(seed)

    def init_vars(self, num_strats, num_players):
        if False:
            print('Hello World!')
        'Initialize solver parameters.'
        self.num_strats = num_strats
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
        init_y = self.init_polymatrix(num_strats, num_players)
        return (init_dist, init_y)

    def init_polymatrix(self, num_strats, num_players):
        if False:
            print('Hello World!')
        'Initialize all pairwise bimatrix games to zero and return as dict.'
        init_pm = dict()
        for (i, j) in itertools.combinations(range(num_players), 2):
            init_pm[i, j] = np.zeros((2, num_strats[i], num_strats[j]))
        return init_pm

    def record_aux_errors(self, grads):
        if False:
            for i in range(10):
                print('nop')
        'Record errors for the auxiliary variables.'
        grad_y = grads[1]
        grad_y_flat = np.concatenate([np.ravel(g) for g in grad_y.values()])
        self.aux_errors.append([np.linalg.norm(grad_y_flat)])

    def compute_gradients(self, params, payoff_matrices):
        if False:
            i = 10
            return i + 15
        'Compute and return gradients (and exploitabilities) for all parameters.\n\n    Args:\n      params: tuple of params (dist, y), see ate.gradients\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n    Returns:\n      tuple of gradients (grad_dist, grad_y), see ate.gradients\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    '
        return self.gradients(*params, payoff_matrices, self.p, self.proj_grad)

    def exploitability(self, params, payoff_matrices):
        if False:
            return 10
        'Compute and return tsallis entropy regularized exploitability.\n\n    Args:\n      params: tuple of params (dist, y), see ate.gradients\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n    Returns:\n      float, exploitability of current dist\n    '
        return exp.ate_exploitability(params, payoff_matrices, self.p)

    def euc_descent_step(self, params, grads, t):
        if False:
            return 10
        'Projected gradient descent on exploitability using Euclidean projection.\n\n    Args:\n      params: tuple of variables to be updated (dist, y)\n      grads: tuple of variable gradients (grad_dist, grad_y)\n      t: int, solver iteration (unused)\n    Returns:\n      new_params: tuple of update params (new_dist, new_y)\n    '
        (lr_dist, lr_y) = self.lrs
        new_dist = []
        for (dist_i, dist_grad_i) in zip(params[0], grads[0]):
            new_dist_i = dist_i - lr_dist * dist_grad_i
            new_dist_i = simplex.euclidean_projection_onto_simplex(new_dist_i)
            new_dist.append(new_dist_i)
        lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
        new_y = dict()
        for (i, j) in params[1]:
            y_ij = params[1][i, j]
            y_grad_ij = grads[1][i, j]
            new_y_ij = y_ij - lr_y * y_grad_ij
            new_y_ij = np.clip(new_y_ij, 0.0, np.inf)
            new_y[i, j] = new_y_ij
        return (new_dist, new_y)

    def mirror_descent_step(self, params, grads, t):
        if False:
            print('Hello World!')
        'Entropic mirror descent on exploitability.\n\n    Args:\n      params: tuple of variables to be updated (dist, y)\n      grads: tuple of variable gradients (grad_dist, grad_y)\n      t: int, solver iteration (unused)\n    Returns:\n      new_params: tuple of update params (new_dist, new_y)\n    '
        (lr_dist, lr_y) = self.lrs
        new_dist = []
        for (dist_i, dist_grad_i) in zip(params[0], grads[0]):
            new_dist_i = np.log(np.clip(dist_i, 0.0, np.inf)) - lr_dist * dist_grad_i
            new_dist_i = special.softmax(new_dist_i)
            new_dist.append(new_dist_i)
        lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
        new_y = dict()
        for (i, j) in params[1]:
            y_ij = params[1][i, j]
            y_grad_ij = grads[1][i, j]
            new_y_ij = y_ij - lr_y * y_grad_ij
            new_y_ij = np.clip(new_y_ij, 0.0, np.inf)
            new_y[i, j] = new_y_ij
        return (new_dist, new_y)

    def gradients(self, dist, y, payoff_matrices, p=1, proj_grad=True):
        if False:
            while True:
                i = 10
        "Computes exploitablity gradient and aux variable gradients.\n\n    Args:\n      dist: list of 1-d np.arrays, current estimate of nash distribution\n      y: dict of 2-d np.arrays, current est. of players (i, j)'s payoff matrix\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n          values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n          are sorted and arrays should be indexed in the same order\n      p: float in [0, 1], Tsallis entropy-regularization --> 0 as p --> 0\n      proj_grad: bool, if True, projects dist gradient onto simplex\n    Returns:\n      gradient of exploitability w.r.t. (dist, y) as tuple\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    "
        policy_gradient = []
        other_player_fx = []
        grad_y = self.init_polymatrix(self.num_strats, self.num_players)
        unreg_exp = []
        reg_exp = []
        for i in range(self.num_players):
            nabla_i = np.zeros_like(dist[i])
            for j in range(self.num_players):
                if j == i:
                    continue
                if i < j:
                    hess_i_ij = payoff_matrices[i, j][0]
                    hess_i_ij_from_y = y[i, j][0]
                    grad_y[i, j][0] = hess_i_ij_from_y - hess_i_ij
                else:
                    hess_i_ij = payoff_matrices[j, i][1].T
                    hess_i_ij_from_y = y[j, i][1].T
                    grad_y[j, i][1] = hess_i_ij_from_y.T - hess_i_ij.T
                nabla_ij = hess_i_ij_from_y.dot(dist[j])
                nabla_i += nabla_ij / float(self.num_players - 1)
            if p > 0:
                power = 1.0 / float(p)
                s_i = np.linalg.norm(nabla_i, ord=power)
                if s_i == 0:
                    br_i = misc.uniform_dist(nabla_i)
                else:
                    br_i = (nabla_i / s_i) ** power
            else:
                power = np.inf
                s_i = np.linalg.norm(nabla_i, ord=power)
                br_i = np.zeros_like(dist[i])
                maxima_i = nabla_i == s_i
                br_i[maxima_i] = 1.0 / maxima_i.sum()
            policy_gradient_i = nabla_i - s_i * dist[i] ** p
            policy_gradient.append(policy_gradient_i)
            unreg_exp.append(np.max(nabla_i) - nabla_i.dot(dist[i]))
            br_i_inv_sparse = 1 - np.sum(br_i ** (p + 1))
            dist_i_inv_sparse = 1 - np.sum(dist[i] ** (p + 1))
            entr_br_i = s_i / (p + 1) * br_i_inv_sparse
            entr_dist_i = s_i / (p + 1) * dist_i_inv_sparse
            reg_exp.append(nabla_i.dot(br_i - dist[i]) + entr_br_i - entr_dist_i)
            entr_br_vec_i = br_i_inv_sparse * br_i ** (1 - p)
            entr_dist_vec_i = dist_i_inv_sparse * dist[i] ** (1 - p)
            other_player_fx_i = br_i - dist[i] + 1 / (p + 1) * (entr_br_vec_i - entr_dist_vec_i)
            other_player_fx.append(other_player_fx_i)
        grad_dist = []
        for i in range(self.num_players):
            grad_dist_i = -policy_gradient[i]
            for j in range(self.num_players):
                if j == i:
                    continue
                if i < j:
                    hess_j_ij_from_y = y[i, j][1]
                else:
                    hess_j_ij_from_y = y[j, i][0].T
                grad_dist_i += hess_j_ij_from_y.dot(other_player_fx[j])
            if proj_grad:
                grad_dist_i = simplex.project_grad(grad_dist_i)
            grad_dist.append(grad_dist_i)
        return ((grad_dist, grad_y), np.mean(unreg_exp), np.mean(reg_exp))