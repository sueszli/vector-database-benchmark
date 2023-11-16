from __future__ import print_function, division
import time
from copy import deepcopy
import numpy as np
from pymanopt.solvers.linesearch import LineSearchAdaptive
from pymanopt.solvers.solver import Solver
from pymanopt import tools
BetaTypes = tools.make_enum('BetaTypes', 'FletcherReeves PolakRibiere HestenesStiefel HagerZhang'.split())

class ConjugateGradientMS(Solver):
    """
    Module containing conjugate gradient algorithm based on
    conjugategradient.m from the manopt MATLAB package.
    """

    def __init__(self, beta_type=BetaTypes.HestenesStiefel, orth_value=np.inf, linesearch=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Instantiate gradient solver class.\n\n        Args:\n            beta_type (object): Conjugate gradient beta rule used to construct the new search direction.\n            orth_value (float): Parameter for Powell's restart strategy. An infinite value disables this strategy.\n                See in code formula for the specific criterion used.\n            - linesearch (object): The linesearch method to used.\n        "
        super(ConjugateGradientMS, self).__init__(*args, **kwargs)
        self._beta_type = beta_type
        self._orth_value = orth_value
        if linesearch is None:
            self._linesearch = LineSearchAdaptive()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    def solve(self, problem, x=None, reuselinesearch=False, compute_stats=None):
        if False:
            for i in range(10):
                print('nop')
        'Perform optimization using nonlinear conjugate gradient method with\n        linesearch.\n\n        This method first computes the gradient of obj w.r.t. arg, and then\n        optimizes by moving in a direction that is conjugate to all previous\n        search directions.\n\n        Args:\n            problem (object): Pymanopt problem setup using the Problem class, this must\n                have a .manifold attribute specifying the manifold to optimize\n                over, as well as a cost and enough information to compute\n                the gradient of that cost.\n            x (numpy.ndarray): Optional parameter. Starting point on the manifold. If none\n                then a starting point will be randomly generated.\n            reuselinesearch (bool): Whether to reuse the previous linesearch object. Allows to\n                use information from a previous solve run.\n\n        Returns:\n            numpy.ndarray: Local minimum of obj, or if algorithm terminated before convergence x will be the point at which it terminated.\n        '
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad
        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch
        if verbosity >= 1:
            print('Optimizing...')
        if verbosity >= 2:
            print(' iter\t\t   cost val\t    grad. norm')
        iter = 0
        stats = {}
        stepsize = np.nan
        cumulative_time = 0.0
        time0 = time.time()
        t0 = time.time()
        if x is None:
            x = man.rand()
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
        Pgrad = problem.precon(x, grad)
        gradPgrad = man.inner(x, grad, Pgrad)
        desc_dir = -Pgrad
        time_iter = time.time() - t0
        cumulative_time += time_iter
        self._start_optlog(extraiterfields=['gradnorm'], solverparams={'beta_type': self._beta_type, 'orth_value': self._orth_value, 'linesearcher': linesearch})
        while True:
            if verbosity >= 2:
                print('%5d\t%+.16e\t%.8e' % (iter, cost, gradnorm))
            if compute_stats is not None:
                compute_stats(x, [iter, cost, gradnorm, cumulative_time], stats)
            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)
            t0 = time.time()
            stop_reason = self._check_stopping_criterion(time.time() - cumulative_time, gradnorm=gradnorm, iter=iter + 1, stepsize=stepsize)
            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break
            df0 = man.inner(x, grad, desc_dir)
            if df0 >= 0:
                if verbosity >= 3:
                    print('Conjugate gradient info: got an ascent direction (df0 = %.2f), reset to the (preconditioned) steepest descent direction.' % df0)
                desc_dir = -Pgrad
                df0 = -gradPgrad
            (stepsize, newx) = linesearch.search(objective, man, x, desc_dir, cost, df0)
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradnorm = man.norm(newx, newgrad)
            Pnewgrad = problem.precon(newx, newgrad)
            newgradPnewgrad = man.inner(newx, newgrad, Pnewgrad)
            oldgrad = man.transp(x, newx, grad)
            orth_grads = man.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                desc_dir = -Pnewgrad
            else:
                desc_dir = man.transp(x, newx, desc_dir)
                if self._beta_type == BetaTypes.FletcherReeves:
                    beta = newgradPnewgrad / gradPgrad
                elif self._beta_type == BetaTypes.PolakRibiere:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    beta = max(0, ip_diff / gradPgrad)
                elif self._beta_type == BetaTypes.HestenesStiefel:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    try:
                        beta = max(0, ip_diff / man.inner(newx, diff, desc_dir))
                    except ZeroDivisionError:
                        beta = 1
                elif self._beta_type == BetaTypes.HagerZhang:
                    diff = newgrad - oldgrad
                    Poldgrad = man.transp(x, newx, Pgrad)
                    Pdiff = Pnewgrad - Poldgrad
                    deno = man.inner(newx, diff, desc_dir)
                    numo = man.inner(newx, diff, Pnewgrad)
                    numo -= 2 * man.inner(newx, diff, Pdiff) * man.inner(newx, desc_dir, newgrad) / deno
                    beta = numo / deno
                    desc_dir_norm = man.norm(newx, desc_dir)
                    eta_HZ = -1 / (desc_dir_norm * min(0.01, gradnorm))
                    beta = max(beta, eta_HZ)
                else:
                    types = ', '.join(['BetaTypes.%s' % t for t in BetaTypes._fields])
                    raise ValueError('Unknown beta_type %s. Should be one of %s.' % (self._beta_type, types))
                desc_dir = -Pnewgrad + beta * desc_dir
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradnorm = newgradnorm
            gradPgrad = newgradPnewgrad
            iter += 1
            time_iter = time.time() - t0
            cumulative_time += time_iter
        if self._logverbosity <= 0:
            return (x, stats)
        else:
            self._stop_optlog(x, cost, stop_reason, time0, stepsize=stepsize, gradnorm=gradnorm, iter=iter)
            return (x, stats, self._optlog)