"""
Holds files for l1 regularization of LikelihoodModel, using cvxopt.
"""
import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common

def fit_l1_cvxopt_cp(f, score, start_params, args, kwargs, disp=False, maxiter=100, callback=None, retall=False, full_output=False, hess=None):
    if False:
        i = 10
        return i + 15
    '\n    Solve the l1 regularized problem using cvxopt.solvers.cp\n\n    Specifically:  We convert the convex but non-smooth problem\n\n    .. math:: \\min_\\beta f(\\beta) + \\sum_k\\alpha_k |\\beta_k|\n\n    via the transformation to the smooth, convex, constrained problem in twice\n    as many variables (adding the "added variables" :math:`u_k`)\n\n    .. math:: \\min_{\\beta,u} f(\\beta) + \\sum_k\\alpha_k u_k,\n\n    subject to\n\n    .. math:: -u_k \\leq \\beta_k \\leq u_k.\n\n    Parameters\n    ----------\n    All the usual parameters from LikelhoodModel.fit\n    alpha : non-negative scalar or numpy array (same size as parameters)\n        The weight multiplying the l1 penalty term\n    trim_mode : \'auto, \'size\', or \'off\'\n        If not \'off\', trim (set to zero) parameters that would have been zero\n            if the solver reached the theoretical minimum.\n        If \'auto\', trim params using the Theory above.\n        If \'size\', trim params if they have very small absolute value\n    size_trim_tol : float or \'auto\' (default = \'auto\')\n        For use when trim_mode === \'size\'\n    auto_trim_tol : float\n        For sue when trim_mode == \'auto\'.  Use\n    qc_tol : float\n        Print warning and do not allow auto trim when (ii) in "Theory" (above)\n        is violated by this much.\n    qc_verbose : bool\n        If true, print out a full QC report upon failure\n    abstol : float\n        absolute accuracy (default: 1e-7).\n    reltol : float\n        relative accuracy (default: 1e-6).\n    feastol : float\n        tolerance for feasibility conditions (default: 1e-7).\n    refinement : int\n        number of iterative refinement steps when solving KKT equations\n        (default: 1).\n    '
    from cvxopt import solvers, matrix
    start_params = np.array(start_params).ravel('F')
    k_params = len(start_params)
    x0 = np.append(start_params, np.fabs(start_params))
    x0 = matrix(x0, (2 * k_params, 1))
    alpha = np.array(kwargs['alpha_rescaled']).ravel('F')
    alpha = alpha * np.ones(k_params)
    assert alpha.min() >= 0
    f_0 = lambda x: _objective_func(f, x, k_params, alpha, *args)
    Df = lambda x: _fprime(score, x, k_params, alpha)
    G = _get_G(k_params)
    h = matrix(0.0, (2 * k_params, 1))
    H = lambda x, z: _hessian_wrapper(hess, x, z, k_params)

    def F(x=None, z=None):
        if False:
            while True:
                i = 10
        if x is None:
            return (0, x0)
        elif z is None:
            return (f_0(x), Df(x))
        else:
            return (f_0(x), Df(x), H(x, z))
    solvers.options['show_progress'] = disp
    solvers.options['maxiters'] = maxiter
    if 'abstol' in kwargs:
        solvers.options['abstol'] = kwargs['abstol']
    if 'reltol' in kwargs:
        solvers.options['reltol'] = kwargs['reltol']
    if 'feastol' in kwargs:
        solvers.options['feastol'] = kwargs['feastol']
    if 'refinement' in kwargs:
        solvers.options['refinement'] = kwargs['refinement']
    results = solvers.cp(F, G, h)
    x = np.asarray(results['x']).ravel()
    params = x[:k_params]
    qc_tol = kwargs['qc_tol']
    qc_verbose = kwargs['qc_verbose']
    passed = l1_solvers_common.qc_results(params, alpha, score, qc_tol, qc_verbose)
    trim_mode = kwargs['trim_mode']
    size_trim_tol = kwargs['size_trim_tol']
    auto_trim_tol = kwargs['auto_trim_tol']
    (params, trimmed) = l1_solvers_common.do_trim_params(params, k_params, alpha, score, passed, trim_mode, size_trim_tol, auto_trim_tol)
    if full_output:
        fopt = f_0(x)
        gopt = float('nan')
        hopt = float('nan')
        iterations = float('nan')
        converged = results['status'] == 'optimal'
        warnflag = results['status']
        retvals = {'fopt': fopt, 'converged': converged, 'iterations': iterations, 'gopt': gopt, 'hopt': hopt, 'trimmed': trimmed, 'warnflag': warnflag}
    else:
        x = np.array(results['x']).ravel()
        params = x[:k_params]
    if full_output:
        return (params, retvals)
    else:
        return params

def _objective_func(f, x, k_params, alpha, *args):
    if False:
        print('Hello World!')
    '\n    The regularized objective function.\n    '
    from cvxopt import matrix
    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    u = x_arr[k_params:]
    objective_func_arr = f(params, *args) + (alpha * u).sum()
    return matrix(objective_func_arr)

def _fprime(score, x, k_params, alpha):
    if False:
        while True:
            i = 10
    '\n    The regularized derivative.\n    '
    from cvxopt import matrix
    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    fprime_arr = np.append(score(params), alpha)
    return matrix(fprime_arr, (1, 2 * k_params))

def _get_G(k_params):
    if False:
        i = 10
        return i + 15
    '\n    The linear inequality constraint matrix.\n    '
    from cvxopt import matrix
    I = np.eye(k_params)
    A = np.concatenate((-I, -I), axis=1)
    B = np.concatenate((I, -I), axis=1)
    C = np.concatenate((A, B), axis=0)
    return matrix(C)

def _hessian_wrapper(hess, x, z, k_params):
    if False:
        print('Hello World!')
    '\n    Wraps the hessian up in the form for cvxopt.\n\n    cvxopt wants the hessian of the objective function and the constraints.\n        Since our constraints are linear, this part is all zeros.\n    '
    from cvxopt import matrix
    x_arr = np.asarray(x)
    params = x_arr[:k_params].ravel()
    zh_x = np.asarray(z[0]) * hess(params)
    zero_mat = np.zeros(zh_x.shape)
    A = np.concatenate((zh_x, zero_mat), axis=1)
    B = np.concatenate((zero_mat, zero_mat), axis=1)
    zh_x_ext = np.concatenate((A, B), axis=0)
    return matrix(zh_x_ext, (2 * k_params, 2 * k_params))