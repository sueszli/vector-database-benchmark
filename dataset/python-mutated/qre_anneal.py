"""Quantal Response Equilibrium (QRE) Stochastic Approximate Nash Solver."""
from absl import logging
import numpy as np
from scipy import special
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.nonsymmetric import exploitability as exp

class Solver(object):
    """QRE Solver."""

    def __init__(self, temperature=1.0, proj_grad=True, euclidean=False, cheap=False, lrs=(0.01, 0.1), exp_thresh=-1.0, rnd_init=False, seed=None, **kwargs):
        if False:
            return 10
        'Ctor.'
        del kwargs
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative')
        self.num_players = None
        self.temperature = temperature
        self.proj_grad = proj_grad
        self.cheap = cheap
        self.rnd_init = rnd_init
        self.lrs = lrs
        self.exp_thresh = exp_thresh
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
        init_y = [np.zeros_like(dist_i) for dist_i in init_dist]
        init_anneal_steps = 0
        return (init_dist, init_y, init_anneal_steps)

    def record_aux_errors(self, grads):
        if False:
            return 10
        'Record errors for the auxiliary variables.'
        grad_y = grads[1]
        grad_y_flat = np.concatenate([np.ravel(g) for g in grad_y])
        self.aux_errors.append([np.linalg.norm(grad_y_flat)])

    def compute_gradients(self, params, payoff_matrices):
        if False:
            while True:
                i = 10
        'Compute and return gradients (and exploitabilities) for all parameters.\n\n    Args:\n      params: tuple of params (dist, y, anneal_steps), see gradients\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n          values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n          are sorted and arrays should be indexed in the same order\n    Returns:\n      tuple of gradients (grad_dist, grad_y, grad_anneal_steps), see gradients\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    '
        if self.cheap:
            return self.cheap_gradients(self.random, *params, payoff_matrices, self.num_players, self.temperature, self.proj_grad)
        else:
            return self.gradients(*params, payoff_matrices, self.num_players, self.temperature, self.proj_grad)

    def exploitability(self, params, payoff_matrices):
        if False:
            i = 10
            return i + 15
        'Compute and return tsallis entropy regularized exploitability.\n\n    Args:\n      params: tuple of params (dist, y), see ate.gradients\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n          values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n          are sorted and arrays should be indexed in the same order\n    Returns:\n      float, exploitability of current dist\n    '
        return exp.qre_exploitability(params, payoff_matrices, self.temperature)

    def euc_descent_step(self, params, grads, t):
        if False:
            while True:
                i = 10
        'Projected gradient descent on exploitability using Euclidean projection.\n\n    Args:\n      params: tuple of variables to be updated (dist, y, anneal_steps)\n      grads: tuple of variable gradients (grad_dist, grad_y, grad_anneal_steps)\n      t: int, solver iteration (unused)\n    Returns:\n      new_params: tuple of update params (new_dist, new_y, new_anneal_steps)\n    '
        (lr_dist, lr_y) = self.lrs
        new_dist = []
        for (dist_i, dist_grad_i) in zip(params[0], grads[0]):
            new_dist_i = dist_i - lr_dist * dist_grad_i
            new_dist_i = simplex.euclidean_projection_onto_simplex(new_dist_i)
            new_dist.append(new_dist_i)
        lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
        new_y = []
        for (y_i, y_grad_i) in zip(params[1], grads[1]):
            new_y_i = y_i - lr_y * y_grad_i
            new_y_i = np.clip(new_y_i, 0.0, np.inf)
            new_y.append(new_y_i)
        new_anneal_steps = params[2] + grads[2]
        return (new_dist, new_y, new_anneal_steps)

    def mirror_descent_step(self, params, grads, t):
        if False:
            while True:
                i = 10
        'Entropic mirror descent on exploitability.\n\n    Args:\n      params: tuple of variables to be updated (dist, y, anneal_steps)\n      grads: tuple of variable gradients (grad_dist, grad_y, grad_anneal_steps)\n      t: int, solver iteration (unused)\n    Returns:\n      new_params: tuple of update params (new_dist, new_y, new_anneal_steps)\n    '
        (lr_dist, lr_y) = self.lrs
        new_dist = []
        for (dist_i, dist_grad_i) in zip(params[0], grads[0]):
            new_dist_i = np.log(np.clip(dist_i, 0.0, np.inf)) - lr_dist * dist_grad_i
            new_dist_i = special.softmax(new_dist_i)
            new_dist.append(new_dist_i)
        lr_y = np.clip(1 / float(t + 1), lr_y, np.inf)
        new_y = []
        for (y_i, y_grad_i) in zip(params[1], grads[1]):
            new_y_i = y_i - lr_y * y_grad_i
            new_y_i = np.clip(new_y_i, 0.0, np.inf)
            new_y.append(new_y_i)
        new_anneal_steps = params[2] + grads[2]
        return (new_dist, new_y, new_anneal_steps)

    def gradients(self, dist, y, anneal_steps, payoff_matrices, num_players, temperature=0.0, proj_grad=True):
        if False:
            print('Hello World!')
        'Computes exploitablity gradient and aux variable gradients.\n\n    Args:\n      dist: list of 1-d np.arrays, current estimate of nash distribution\n      y: list 1-d np.arrays (same shape as dist), current est. of payoff\n        gradient\n      anneal_steps: int, elapsed num steps since last anneal\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n          values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n          are sorted and arrays should be indexed in the same order\n      num_players: int, number of players, in case payoff_matrices is\n        abbreviated\n      temperature: non-negative float, default 0.\n      proj_grad: bool, if True, projects dist gradient onto simplex\n    Returns:\n      gradient of exploitability w.r.t. (dist, y, anneal_steps) as tuple\n      unregularized exploitability (stochastic estimate)\n      shannon regularized exploitability (stochastic estimate)\n    '
        policy_gradient = []
        other_player_fx = []
        grad_y = []
        unreg_exp = []
        reg_exp = []
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
            grad_y.append(y[i] - nabla_i)
            if temperature >= 0.001:
                br_i = special.softmax(y[i] / temperature)
                br_i_mat = (np.diag(br_i) - np.outer(br_i, br_i)) / temperature
                br_i_policy_gradient = nabla_i - temperature * (np.log(br_i) + 1)
            else:
                power = np.inf
                s_i = np.linalg.norm(y[i], ord=power)
                br_i = np.zeros_like(dist[i])
                maxima_i = y[i] == s_i
                br_i[maxima_i] = 1.0 / maxima_i.sum()
                br_i_mat = np.zeros((br_i.size, br_i.size))
                br_i_policy_gradient = np.zeros_like(br_i)
            policy_gradient_i = nabla_i - temperature * (np.log(dist[i]) + 1)
            policy_gradient.append(policy_gradient_i)
            unreg_exp.append(np.max(y[i]) - y[i].dot(dist[i]))
            entr_br_i = temperature * special.entr(br_i).sum()
            entr_dist_i = temperature * special.entr(dist[i]).sum()
            reg_exp.append(y[i].dot(br_i - dist[i]) + entr_br_i - entr_dist_i)
            other_player_fx_i = br_i - dist[i] + br_i_mat.dot(br_i_policy_gradient)
            other_player_fx.append(other_player_fx_i)
        grad_dist = []
        for i in range(num_players):
            grad_dist_i = -policy_gradient[i]
            for j in range(num_players):
                if j == i:
                    continue
                if i < j:
                    hess_j_ij = payoff_matrices[i, j][1]
                else:
                    hess_j_ij = payoff_matrices[j, i][0].T
                grad_dist_i += hess_j_ij.dot(other_player_fx[j])
            if proj_grad:
                grad_dist_i = simplex.project_grad(grad_dist_i)
            grad_dist.append(grad_dist_i)
        unreg_exp_mean = np.mean(unreg_exp)
        reg_exp_mean = np.mean(reg_exp)
        (_, lr_y) = self.lrs
        if reg_exp_mean < self.exp_thresh and anneal_steps >= 1 / lr_y:
            self.temperature = np.clip(temperature / 2.0, 0.0, np.inf)
            grad_anneal_steps = -anneal_steps
        else:
            grad_anneal_steps = 1
        return ((grad_dist, grad_y, grad_anneal_steps), unreg_exp_mean, reg_exp_mean)

    def cheap_gradients(self, random, dist, y, anneal_steps, payoff_matrices, num_players, temperature=0.0, proj_grad=True):
        if False:
            return 10
        'Computes exploitablity gradient and aux variable gradients with samples.\n\n    This implementation takes payoff_matrices as input so technically uses\n    O(d^2) compute but only a single column of payoff_matrices is used to\n    perform the update so can be re-implemented in O(d) if needed.\n\n    Args:\n      random: random number generator, np.random.RandomState(seed)\n      dist: list of 1-d np.arrays, current estimate of nash distribution\n      y: list 1-d np.arrays (same shape as dist), current est. of payoff\n        gradient\n      anneal_steps: int, elapsed num steps since last anneal\n      payoff_matrices: dictionary with keys as tuples of agents (i, j) and\n          values of (2 x A x A) np.arrays, payoffs for each joint action. keys\n          are sorted and arrays should be indexed in the same order\n      num_players: int, number of players, in case payoff_matrices is\n        abbreviated\n      temperature: non-negative float, default 0.\n      proj_grad: bool, if True, projects dist gradient onto simplex\n    Returns:\n      gradient of exploitability w.r.t. (dist, y, anneal_steps) as tuple\n      unregularized exploitability (stochastic estimate)\n      shannon regularized exploitability (stochastic estimate)\n    '
        policy_gradient = []
        other_player_fx = []
        grad_y = []
        unreg_exp = []
        reg_exp = []
        for i in range(num_players):
            others = list(range(num_players))
            others.remove(i)
            j = np.random.choice(others)
            action_j = random.choice(dist[j].size, p=dist[j])
            if i < j:
                hess_i_ij = payoff_matrices[i, j][0]
            else:
                hess_i_ij = payoff_matrices[j, i][1].T
            nabla_i = hess_i_ij[:, action_j]
            grad_y.append(y[i] - nabla_i)
            if temperature >= 0.001:
                br_i = special.softmax(y[i] / temperature)
                br_i_mat = (np.diag(br_i) - np.outer(br_i, br_i)) / temperature
                br_i_policy_gradient = nabla_i - temperature * (np.log(br_i) + 1)
            else:
                power = np.inf
                s_i = np.linalg.norm(y[i], ord=power)
                br_i = np.zeros_like(dist[i])
                maxima_i = y[i] == s_i
                br_i[maxima_i] = 1.0 / maxima_i.sum()
                br_i_mat = np.zeros((br_i.size, br_i.size))
                br_i_policy_gradient = np.zeros_like(br_i)
            policy_gradient_i = nabla_i - temperature * (np.log(dist[i]) + 1)
            policy_gradient.append(policy_gradient_i)
            unreg_exp.append(np.max(y[i]) - y[i].dot(dist[i]))
            entr_br_i = temperature * special.entr(br_i).sum()
            entr_dist_i = temperature * special.entr(dist[i]).sum()
            reg_exp.append(y[i].dot(br_i - dist[i]) + entr_br_i - entr_dist_i)
            other_player_fx_i = br_i - dist[i] + br_i_mat.dot(br_i_policy_gradient)
            other_player_fx.append(other_player_fx_i)
        grad_dist = []
        for i in range(num_players):
            grad_dist_i = -policy_gradient[i]
            for j in range(num_players):
                if j == i:
                    continue
                if i < j:
                    hess_j_ij = payoff_matrices[i, j][1]
                else:
                    hess_j_ij = payoff_matrices[j, i][0].T
                action_u = random.choice(dist[j].size)
                other_player_fx_j = dist[j].size * other_player_fx[j][action_u]
                grad_dist_i += hess_j_ij[:, action_u] * other_player_fx_j
            if proj_grad:
                grad_dist_i = simplex.project_grad(grad_dist_i)
            grad_dist.append(grad_dist_i)
        unreg_exp_mean = np.mean(unreg_exp)
        reg_exp_mean = np.mean(reg_exp)
        (_, lr_y) = self.lrs
        if reg_exp_mean < self.exp_thresh and anneal_steps >= 1 / lr_y:
            self.temperature = np.clip(temperature / 2.0, 0.0, np.inf)
            grad_anneal_steps = -anneal_steps
        else:
            grad_anneal_steps = 1
        return ((grad_dist, grad_y, grad_anneal_steps), unreg_exp_mean, reg_exp_mean)