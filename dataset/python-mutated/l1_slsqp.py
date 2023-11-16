"""
Holds files for l1 regularization of LikelihoodModel, using
scipy.optimize.slsqp
"""
import numpy as np
from scipy.optimize import fmin_slsqp
import statsmodels.base.l1_solvers_common as l1_solvers_common

def fit_l1_slsqp(f, score, start_params, args, kwargs, disp=False, maxiter=1000, callback=None, retall=False, full_output=False, hess=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solve the l1 regularized problem using scipy.optimize.fmin_slsqp().\n\n    Specifically:  We convert the convex but non-smooth problem\n\n    .. math:: \\min_\\beta f(\\beta) + \\sum_k\\alpha_k |\\beta_k|\n\n    via the transformation to the smooth, convex, constrained problem in twice\n    as many variables (adding the "added variables" :math:`u_k`)\n\n    .. math:: \\min_{\\beta,u} f(\\beta) + \\sum_k\\alpha_k u_k,\n\n    subject to\n\n    .. math:: -u_k \\leq \\beta_k \\leq u_k.\n\n    Parameters\n    ----------\n    All the usual parameters from LikelhoodModel.fit\n    alpha : non-negative scalar or numpy array (same size as parameters)\n        The weight multiplying the l1 penalty term\n    trim_mode : \'auto, \'size\', or \'off\'\n        If not \'off\', trim (set to zero) parameters that would have been zero\n            if the solver reached the theoretical minimum.\n        If \'auto\', trim params using the Theory above.\n        If \'size\', trim params if they have very small absolute value\n    size_trim_tol : float or \'auto\' (default = \'auto\')\n        For use when trim_mode === \'size\'\n    auto_trim_tol : float\n        For sue when trim_mode == \'auto\'.  Use\n    qc_tol : float\n        Print warning and do not allow auto trim when (ii) in "Theory" (above)\n        is violated by this much.\n    qc_verbose : bool\n        If true, print out a full QC report upon failure\n    acc : float (default 1e-6)\n        Requested accuracy as used by slsqp\n    '
    start_params = np.array(start_params).ravel('F')
    k_params = len(start_params)
    x0 = np.append(start_params, np.fabs(start_params))
    alpha = np.array(kwargs['alpha_rescaled']).ravel('F')
    alpha = alpha * np.ones(k_params)
    assert alpha.min() >= 0
    disp_slsqp = _get_disp_slsqp(disp, retall)
    acc = kwargs.setdefault('acc', 1e-10)
    func = lambda x_full: _objective_func(f, x_full, k_params, alpha, *args)
    f_ieqcons_wrap = lambda x_full: _f_ieqcons(x_full, k_params)
    fprime_wrap = lambda x_full: _fprime(score, x_full, k_params, alpha)
    fprime_ieqcons_wrap = lambda x_full: _fprime_ieqcons(x_full, k_params)
    results = fmin_slsqp(func, x0, f_ieqcons=f_ieqcons_wrap, fprime=fprime_wrap, acc=acc, iter=maxiter, disp=disp_slsqp, full_output=full_output, fprime_ieqcons=fprime_ieqcons_wrap)
    params = np.asarray(results[0][:k_params])
    qc_tol = kwargs['qc_tol']
    qc_verbose = kwargs['qc_verbose']
    passed = l1_solvers_common.qc_results(params, alpha, score, qc_tol, qc_verbose)
    trim_mode = kwargs['trim_mode']
    size_trim_tol = kwargs['size_trim_tol']
    auto_trim_tol = kwargs['auto_trim_tol']
    (params, trimmed) = l1_solvers_common.do_trim_params(params, k_params, alpha, score, passed, trim_mode, size_trim_tol, auto_trim_tol)
    if full_output:
        (x_full, fx, its, imode, smode) = results
        fopt = func(np.asarray(x_full))
        converged = imode == 0
        warnflag = str(imode) + ' ' + smode
        iterations = its
        gopt = float('nan')
        hopt = float('nan')
        retvals = {'fopt': fopt, 'converged': converged, 'iterations': iterations, 'gopt': gopt, 'hopt': hopt, 'trimmed': trimmed, 'warnflag': warnflag}
    if full_output:
        return (params, retvals)
    else:
        return params

def _get_disp_slsqp(disp, retall):
    if False:
        i = 10
        return i + 15
    if disp or retall:
        if disp:
            disp_slsqp = 1
        if retall:
            disp_slsqp = 2
    else:
        disp_slsqp = 0
    return disp_slsqp

def _objective_func(f, x_full, k_params, alpha, *args):
    if False:
        print('Hello World!')
    '\n    The regularized objective function\n    '
    x_params = x_full[:k_params]
    x_added = x_full[k_params:]
    return f(x_params, *args) + (alpha * x_added).sum()

def _fprime(score, x_full, k_params, alpha):
    if False:
        while True:
            i = 10
    '\n    The regularized derivative\n    '
    x_params = x_full[:k_params]
    return np.append(score(x_params), alpha)

def _f_ieqcons(x_full, k_params):
    if False:
        print('Hello World!')
    '\n    The inequality constraints.\n    '
    x_params = x_full[:k_params]
    x_added = x_full[k_params:]
    return np.append(x_params + x_added, x_added - x_params)

def _fprime_ieqcons(x_full, k_params):
    if False:
        print('Hello World!')
    '\n    Derivative of the inequality constraints\n    '
    I = np.eye(k_params)
    A = np.concatenate((I, I), axis=1)
    B = np.concatenate((-I, I), axis=1)
    C = np.concatenate((A, B), axis=0)
    return C