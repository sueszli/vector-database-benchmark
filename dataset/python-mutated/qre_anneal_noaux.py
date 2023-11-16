"""Quantal Response Equilibrium (QRE) Stochastic Approximate Nash Solver."""
from absl import logging
import numpy as np
from scipy import special
from open_spiel.python.algorithms.adidas_utils.helpers import simplex
from open_spiel.python.algorithms.adidas_utils.helpers.symmetric import exploitability as exp

class Solver(object):
    """QRE Solver without auxiliary y variable."""

    def __init__(self, temperature=1.0, proj_grad=True, euclidean=False, cheap=False, lrs=(0.01,), exp_thresh=-1.0, vr=True, rnd_init=False, seed=None, **kwargs):
        if False:
            while True:
                i = 10
        'Ctor.'
        del kwargs
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative')
        self.num_players = None
        self.temperature = temperature
        self.proj_grad = proj_grad
        self.cheap = cheap
        self.vr = vr
        self.pm_vr = None
        self.rnd_init = rnd_init
        self.lrs = lrs
        self.exp_thresh = exp_thresh
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
            i = 10
            return i + 15
        'Initialize solver parameters.'
        self.num_players = num_players
        if self.rnd_init:
            init_dist = self.random.rand(num_strats)
        else:
            init_dist = np.ones(num_strats)
        init_dist /= init_dist.sum()
        init_anneal_steps = 0
        if self.cheap and self.vr:
            self.pm_vr = np.zeros((num_strats, num_strats))
        return (init_dist, init_anneal_steps)

    def compute_gradients(self, params, payoff_matrices):
        if False:
            while True:
                i = 10
        'Compute and return gradients (and exploitabilities) for all parameters.\n\n    Args:\n      params: tuple of params (dist, anneal_steps), see gradients\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n    Returns:\n      tuple of gradients (grad_dist, grad_anneal_steps), see gradients\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    '
        if self.cheap and self.vr:
            (grads, pm_vr, exp_sto, exp_solver_sto) = self.cheap_gradients_vr(self.random, *params, payoff_matrices, self.num_players, self.pm_vr, self.temperature, self.proj_grad)
            self.pm_vr = pm_vr
            return (grads, exp_sto, exp_solver_sto)
        elif self.cheap and (not self.vr):
            return self.cheap_gradients(self.random, *params, payoff_matrices, self.num_players, self.temperature, self.proj_grad)
        else:
            return self.gradients(*params, payoff_matrices, self.num_players, self.temperature, self.proj_grad)

    def exploitability(self, params, payoff_matrices):
        if False:
            for i in range(10):
                print('nop')
        'Compute and return shannon entropy regularized exploitability.\n\n    Args:\n      params: tuple of params (dist, y), see qre.gradients\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n    Returns:\n      float, exploitability of current dist\n    '
        return exp.qre_exploitability(params, payoff_matrices, self.temperature)

    def gradients(self, dist, anneal_steps, payoff_matrices, num_players, temperature=0.0, proj_grad=True):
        if False:
            while True:
                i = 10
        'Computes exploitablity gradient and aux variable gradients.\n\n    Args:\n      dist: 1-d np.array, current estimate of nash distribution\n      anneal_steps: int, elapsed num steps since last anneal\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n      num_players: int, number of players, in case payoff_matrices is\n        abbreviated\n      temperature: non-negative float, default 0.\n      proj_grad: bool, if True, projects dist gradient onto simplex\n    Returns:\n      gradient of exploitability w.r.t. (dist, anneal_steps) as tuple\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    '
        y = nabla = payoff_matrices[0].dot(dist)
        if temperature >= 0.001:
            br = special.softmax(y / temperature)
            br_mat = (np.diag(br) - np.outer(br, br)) / temperature
            log_br_safe = np.clip(np.log(br), -100000.0, 0)
            br_policy_gradient = nabla - temperature * (log_br_safe + 1)
        else:
            power = np.inf
            s = np.linalg.norm(y, ord=power)
            br = np.zeros_like(dist)
            maxima = y == s
            br[maxima] = 1.0 / maxima.sum()
            br_mat = np.zeros((br.size, br.size))
            br_policy_gradient = np.zeros_like(br)
        unreg_exp = np.max(y) - y.dot(dist)
        entr_br = temperature * special.entr(br).sum()
        entr_dist = temperature * special.entr(dist).sum()
        reg_exp = y.dot(br - dist) + entr_br - entr_dist
        policy_gradient = np.array(nabla)
        if temperature > 0:
            log_dist_safe = np.clip(np.log(dist), -100000.0, 0)
            policy_gradient -= temperature * (log_dist_safe + 1)
        other_player_fx = br - dist + br_mat.dot(br_policy_gradient)
        other_player_fx_translated = payoff_matrices[1].dot(other_player_fx)
        grad_dist = -policy_gradient
        grad_dist += (num_players - 1) * other_player_fx_translated
        if proj_grad:
            grad_dist = simplex.project_grad(grad_dist)
        if reg_exp < self.exp_thresh:
            self.temperature = np.clip(temperature / 2.0, 0.0, np.inf)
            if self.temperature < 0.001:
                self.temperature = 0.0
            grad_anneal_steps = -anneal_steps
        else:
            grad_anneal_steps = 1
        return ((grad_dist, grad_anneal_steps), unreg_exp, reg_exp)

    def cheap_gradients(self, random, dist, anneal_steps, payoff_matrices, num_players, temperature=0.0, proj_grad=True):
        if False:
            while True:
                i = 10
        'Computes exploitablity gradient and aux variable gradients with samples.\n\n    This implementation takes payoff_matrices as input so technically uses\n    O(d^2) compute but only a single column of payoff_matrices is used to\n    perform the update so can be re-implemented in O(d) if needed.\n\n    Args:\n      random: random number generator, np.random.RandomState(seed)\n      dist: 1-d np.array, current estimate of nash distribution\n      anneal_steps: int, elapsed num steps since last anneal\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n      num_players: int, number of players, in case payoff_matrices is\n        abbreviated\n      temperature: non-negative float, default 0.\n      proj_grad: bool, if True, projects dist gradient onto simplex\n    Returns:\n      gradient of exploitability w.r.t. (dist, anneal_steps) as tuple\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    '
        del anneal_steps
        action_1 = random.choice(dist.size, p=dist)
        y = nabla = payoff_matrices[0][:, action_1]
        if temperature >= 0.001:
            br = special.softmax(y / temperature)
            br_mat = (np.diag(br) - np.outer(br, br)) / temperature
            br_policy_gradient = nabla - temperature * (np.log(br) + 1)
        else:
            power = np.inf
            s = np.linalg.norm(y, ord=power)
            br = np.zeros_like(dist)
            maxima = y == s
            br[maxima] = 1.0 / maxima.sum()
            br_mat = np.zeros((br.size, br.size))
            br_policy_gradient = np.zeros_like(br)
        unreg_exp = np.max(y) - y.dot(dist)
        entr_br = temperature * special.entr(br).sum()
        entr_dist = temperature * special.entr(dist).sum()
        reg_exp = y.dot(br - dist) + entr_br - entr_dist
        policy_gradient = nabla - temperature * (np.log(dist) + 1)
        other_player_fx = br - dist + br_mat.dot(br_policy_gradient)
        action_u = random.choice(dist.size)
        other_player_fx = dist.size * other_player_fx[action_u]
        other_player_fx_translat = payoff_matrices[1, :, action_u] * other_player_fx
        grad_dist = -policy_gradient + (num_players - 1) * other_player_fx_translat
        if proj_grad:
            grad_dist = simplex.project_grad(grad_dist)
        return ((grad_dist, None), unreg_exp, reg_exp)

    def cheap_gradients_vr(self, random, dist, anneal_steps, payoff_matrices, num_players, pm_vr, temperature=0.0, proj_grad=True, version=0):
        if False:
            for i in range(10):
                print('nop')
        'Computes exploitablity gradient and aux variable gradients with samples.\n\n    This implementation takes payoff_matrices as input so technically uses\n    O(d^2) compute but only a single column of payoff_matrices is used to\n    perform the update so can be re-implemented in O(d) if needed.\n\n    Args:\n      random: random number generator, np.random.RandomState(seed)\n      dist: 1-d np.array, current estimate of nash distribution\n      anneal_steps: int, elapsed num steps since last anneal\n      payoff_matrices: (>=2 x A x A) np.array, payoffs for each joint action\n      num_players: int, number of players, in case payoff_matrices is\n        abbreviated\n      pm_vr: approximate payoff_matrix for variance reduction\n      temperature: non-negative float, default 0.\n      proj_grad: bool, if True, projects dist gradient onto simplex\n      version: int, default 0, two options for variance reduction\n    Returns:\n      gradient of exploitability w.r.t. (dist, anneal_steps) as tuple\n      unregularized exploitability (stochastic estimate)\n      tsallis regularized exploitability (stochastic estimate)\n    '
        del anneal_steps
        if pm_vr is None:
            raise ValueError('pm_vr must be np.array of shape (num_strats,) * 2')
        if not isinstance(version, int) or version < 0 or version > 1:
            raise ValueError('version must be non-negative int < 2')
        action_1 = random.choice(dist.size, p=dist)
        y = nabla = payoff_matrices[0][:, action_1]
        if temperature >= 0.001:
            br = special.softmax(y / temperature)
            br_mat = (np.diag(br) - np.outer(br, br)) / temperature
            br_policy_gradient = nabla - temperature * (np.log(br) + 1)
        else:
            power = np.inf
            s = np.linalg.norm(y, ord=power)
            br = np.zeros_like(dist)
            maxima = y == s
            br[maxima] = 1.0 / maxima.sum()
            br_mat = np.zeros((br.size, br.size))
            br_policy_gradient = np.zeros_like(br)
        unreg_exp = np.max(y) - y.dot(dist)
        entr_br = temperature * special.entr(br).sum()
        entr_dist = temperature * special.entr(dist).sum()
        reg_exp = y.dot(br - dist) + entr_br - entr_dist
        policy_gradient = nabla - temperature * (np.log(dist) + 1)
        other_player_fx = br - dist + br_mat.dot(br_policy_gradient)
        if version == 0:
            other_player_fx_translated = pm_vr.dot(other_player_fx)
            action_u = random.choice(dist.size)
            other_player_fx = other_player_fx[action_u]
            m = dist.size
            pm_mod = m * (payoff_matrices[1, :, action_u] - pm_vr[:, action_u])
            other_player_fx_translated += pm_mod * other_player_fx
        elif version == 1:
            other_player_fx_translated = np.sum(pm_vr, axis=1)
            action_u = random.choice(dist.size)
            other_player_fx = other_player_fx[action_u]
            pm_mod = dist.size * payoff_matrices[1, :, action_u]
            r = dist.size * pm_vr[:, action_u]
            other_player_fx_translated += pm_mod * other_player_fx - r
        grad_dist = -policy_gradient
        grad_dist += (num_players - 1) * other_player_fx_translated
        if proj_grad:
            grad_dist = simplex.project_grad(grad_dist)
        if version == 0:
            pm_vr[:, action_u] = payoff_matrices[1, :, action_u]
        elif version == 1:
            pm_vr[:, action_u] = payoff_matrices[1, :, action_u] * other_player_fx
        return ((grad_dist, None), pm_vr, unreg_exp, reg_exp)

    def euc_descent_step(self, params, grads, t):
        if False:
            print('Hello World!')
        'Projected gradient descent on exploitability using Euclidean projection.\n\n    Args:\n      params: tuple of variables to be updated (dist, anneal_steps)\n      grads: tuple of variable gradients (grad_dist, grad_anneal_steps)\n      t: int, solver iteration\n    Returns:\n      new_params: tuple of update params (new_dist, new_anneal_steps)\n    '
        del t
        lr_dist = self.lrs[0]
        new_params = [params[0] - lr_dist * grads[0]]
        new_params = euc_project(*new_params)
        new_params += (params[1] + grads[1],)
        return new_params

    def mirror_descent_step(self, params, grads, t):
        if False:
            print('Hello World!')
        'Entropic mirror descent on exploitability.\n\n    Args:\n      params: tuple of variables to be updated (dist, anneal_steps)\n      grads: tuple of variable gradients (grad_dist, grad_anneal_steps)\n      t: int, solver iteration\n    Returns:\n      new_params: tuple of update params (new_dist, new_anneal_steps)\n    '
        del t
        lr_dist = self.lrs[0]
        new_params = [np.log(np.clip(params[0], 0, np.inf)) - lr_dist * grads[0]]
        new_params = mirror_project(*new_params)
        new_params += (params[1] + grads[1],)
        return new_params

def euc_project(dist):
    if False:
        i = 10
        return i + 15
    'Project variables onto their feasible sets (euclidean proj for dist).\n\n  Args:\n    dist: 1-d np.array, current estimate of nash distribution\n  Returns:\n    projected variables (dist,) as tuple\n  '
    dist = simplex.euclidean_projection_onto_simplex(dist)
    return (dist,)

def mirror_project(dist):
    if False:
        for i in range(10):
            print('nop')
    'Project variables onto their feasible sets (softmax for dist).\n\n  Args:\n    dist: 1-d np.array, current estimate of nash distribution\n  Returns:\n    projected variables (dist,) as tuple\n  '
    dist = special.softmax(dist)
    return (dist,)