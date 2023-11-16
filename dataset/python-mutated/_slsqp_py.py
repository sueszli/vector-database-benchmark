"""
This module implements the Sequential Least Squares Programming optimization
algorithm (SLSQP), originally developed by Dieter Kraft.
See http://www.netlib.org/toms/733

Functions
---------
.. autosummary::
   :toctree: generated/

    approx_jacobian
    fmin_slsqp

"""
__all__ = ['approx_jacobian', 'fmin_slsqp']
import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import zeros, array, linalg, append, concatenate, finfo, sqrt, vstack, isfinite, atleast_1d
from ._optimize import OptimizeResult, _check_unknown_options, _prepare_scalar_function, _clip_x_for_func, _check_clip_x
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import exp, inf
__docformat__ = 'restructuredtext en'
_epsilon = sqrt(finfo(float).eps)

def approx_jacobian(x, func, epsilon, *args):
    if False:
        return 10
    '\n    Approximate the Jacobian matrix of a callable function.\n\n    Parameters\n    ----------\n    x : array_like\n        The state vector at which to compute the Jacobian matrix.\n    func : callable f(x,*args)\n        The vector-valued function.\n    epsilon : float\n        The perturbation used to determine the partial derivatives.\n    args : sequence\n        Additional arguments passed to func.\n\n    Returns\n    -------\n    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length\n    of the outputs of `func`, and ``lenx`` is the number of elements in\n    `x`.\n\n    Notes\n    -----\n    The approximation is done using forward differences.\n\n    '
    jac = approx_derivative(func, x, method='2-point', abs_step=epsilon, args=args)
    return np.atleast_2d(jac)

def fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, args=(), iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=_epsilon, callback=None):
    if False:
        while True:
            i = 10
    "\n    Minimize a function using Sequential Least Squares Programming\n\n    Python interface function for the SLSQP Optimization subroutine\n    originally implemented by Dieter Kraft.\n\n    Parameters\n    ----------\n    func : callable f(x,*args)\n        Objective function.  Must return a scalar.\n    x0 : 1-D ndarray of float\n        Initial guess for the independent variable(s).\n    eqcons : list, optional\n        A list of functions of length n such that\n        eqcons[j](x,*args) == 0.0 in a successfully optimized\n        problem.\n    f_eqcons : callable f(x,*args), optional\n        Returns a 1-D array in which each element must equal 0.0 in a\n        successfully optimized problem. If f_eqcons is specified,\n        eqcons is ignored.\n    ieqcons : list, optional\n        A list of functions of length n such that\n        ieqcons[j](x,*args) >= 0.0 in a successfully optimized\n        problem.\n    f_ieqcons : callable f(x,*args), optional\n        Returns a 1-D ndarray in which each element must be greater or\n        equal to 0.0 in a successfully optimized problem. If\n        f_ieqcons is specified, ieqcons is ignored.\n    bounds : list, optional\n        A list of tuples specifying the lower and upper bound\n        for each independent variable [(xl0, xu0),(xl1, xu1),...]\n        Infinite values will be interpreted as large floating values.\n    fprime : callable `f(x,*args)`, optional\n        A function that evaluates the partial derivatives of func.\n    fprime_eqcons : callable `f(x,*args)`, optional\n        A function of the form `f(x, *args)` that returns the m by n\n        array of equality constraint normals. If not provided,\n        the normals will be approximated. The array returned by\n        fprime_eqcons should be sized as ( len(eqcons), len(x0) ).\n    fprime_ieqcons : callable `f(x,*args)`, optional\n        A function of the form `f(x, *args)` that returns the m by n\n        array of inequality constraint normals. If not provided,\n        the normals will be approximated. The array returned by\n        fprime_ieqcons should be sized as ( len(ieqcons), len(x0) ).\n    args : sequence, optional\n        Additional arguments passed to func and fprime.\n    iter : int, optional\n        The maximum number of iterations.\n    acc : float, optional\n        Requested accuracy.\n    iprint : int, optional\n        The verbosity of fmin_slsqp :\n\n        * iprint <= 0 : Silent operation\n        * iprint == 1 : Print summary upon completion (default)\n        * iprint >= 2 : Print status of each iterate and summary\n    disp : int, optional\n        Overrides the iprint interface (preferred).\n    full_output : bool, optional\n        If False, return only the minimizer of func (default).\n        Otherwise, output final objective function and summary\n        information.\n    epsilon : float, optional\n        The step size for finite-difference derivative estimates.\n    callback : callable, optional\n        Called after each iteration, as ``callback(x)``, where ``x`` is the\n        current parameter vector.\n\n    Returns\n    -------\n    out : ndarray of float\n        The final minimizer of func.\n    fx : ndarray of float, if full_output is true\n        The final value of the objective function.\n    its : int, if full_output is true\n        The number of iterations.\n    imode : int, if full_output is true\n        The exit mode from the optimizer (see below).\n    smode : string, if full_output is true\n        Message describing the exit mode from the optimizer.\n\n    See also\n    --------\n    minimize: Interface to minimization algorithms for multivariate\n        functions. See the 'SLSQP' `method` in particular.\n\n    Notes\n    -----\n    Exit modes are defined as follows ::\n\n        -1 : Gradient evaluation required (g & a)\n         0 : Optimization terminated successfully\n         1 : Function evaluation required (f & c)\n         2 : More equality constraints than independent variables\n         3 : More than 3*n iterations in LSQ subproblem\n         4 : Inequality constraints incompatible\n         5 : Singular matrix E in LSQ subproblem\n         6 : Singular matrix C in LSQ subproblem\n         7 : Rank-deficient equality constraint subproblem HFTI\n         8 : Positive directional derivative for linesearch\n         9 : Iteration limit reached\n\n    Examples\n    --------\n    Examples are given :ref:`in the tutorial <tutorial-sqlsp>`.\n\n    "
    if disp is not None:
        iprint = disp
    opts = {'maxiter': iter, 'ftol': acc, 'iprint': iprint, 'disp': iprint != 0, 'eps': epsilon, 'callback': callback}
    cons = ()
    cons += tuple(({'type': 'eq', 'fun': c, 'args': args} for c in eqcons))
    cons += tuple(({'type': 'ineq', 'fun': c, 'args': args} for c in ieqcons))
    if f_eqcons:
        cons += ({'type': 'eq', 'fun': f_eqcons, 'jac': fprime_eqcons, 'args': args},)
    if f_ieqcons:
        cons += ({'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons, 'args': args},)
    res = _minimize_slsqp(func, x0, args, jac=fprime, bounds=bounds, constraints=cons, **opts)
    if full_output:
        return (res['x'], res['fun'], res['nit'], res['status'], res['message'])
    else:
        return res['x']

def _minimize_slsqp(func, x0, args=(), jac=None, bounds=None, constraints=(), maxiter=100, ftol=1e-06, iprint=1, disp=False, eps=_epsilon, callback=None, finite_diff_rel_step=None, **unknown_options):
    if False:
        return 10
    "\n    Minimize a scalar function of one or more variables using Sequential\n    Least Squares Programming (SLSQP).\n\n    Options\n    -------\n    ftol : float\n        Precision goal for the value of f in the stopping criterion.\n    eps : float\n        Step size used for numerical approximation of the Jacobian.\n    disp : bool\n        Set to True to print convergence messages. If False,\n        `verbosity` is ignored and set to 0.\n    maxiter : int\n        Maximum number of iterations.\n    finite_diff_rel_step : None or array_like, optional\n        If `jac in ['2-point', '3-point', 'cs']` the relative step size to\n        use for numerical approximation of `jac`. The absolute step\n        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,\n        possibly adjusted to fit into the bounds. For ``method='3-point'``\n        the sign of `h` is ignored. If None (default) then step is selected\n        automatically.\n    "
    _check_unknown_options(unknown_options)
    iter = maxiter - 1
    acc = ftol
    epsilon = eps
    if not disp:
        iprint = 0
    xp = array_namespace(x0)
    x0 = atleast_nd(x0, ndim=1, xp=xp)
    dtype = xp.float64
    if xp.isdtype(x0.dtype, 'real floating'):
        dtype = x0.dtype
    x = xp.reshape(xp.astype(x0, dtype), -1)
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)
    x = np.clip(x, new_bounds[0], new_bounds[1])
    if isinstance(constraints, dict):
        constraints = (constraints,)
    cons = {'eq': (), 'ineq': ()}
    for (ic, con) in enumerate(constraints):
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)
        cjac = con.get('jac')
        if cjac is None:

            def cjac_factory(fun):
                if False:
                    while True:
                        i = 10

                def cjac(x, *args):
                    if False:
                        print('Hello World!')
                    x = _check_clip_x(x, new_bounds)
                    if jac in ['2-point', '3-point', 'cs']:
                        return approx_derivative(fun, x, method=jac, args=args, rel_step=finite_diff_rel_step, bounds=new_bounds)
                    else:
                        return approx_derivative(fun, x, method='2-point', abs_step=epsilon, args=args, bounds=new_bounds)
                return cjac
            cjac = cjac_factory(con['fun'])
        cons[ctype] += ({'fun': con['fun'], 'jac': cjac, 'args': con.get('args', ())},)
    exit_modes = {-1: 'Gradient evaluation required (g & a)', 0: 'Optimization terminated successfully', 1: 'Function evaluation required (f & c)', 2: 'More equality constraints than independent variables', 3: 'More than 3*n iterations in LSQ subproblem', 4: 'Inequality constraints incompatible', 5: 'Singular matrix E in LSQ subproblem', 6: 'Singular matrix C in LSQ subproblem', 7: 'Rank-deficient equality constraint subproblem HFTI', 8: 'Positive directional derivative for linesearch', 9: 'Iteration limit reached'}
    meq = sum(map(len, [atleast_1d(c['fun'](x, *c['args'])) for c in cons['eq']]))
    mieq = sum(map(len, [atleast_1d(c['fun'](x, *c['args'])) for c in cons['ineq']]))
    m = meq + mieq
    la = array([1, m]).max()
    n = len(x)
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3 * n1 + m) * (n1 + 1) + (n1 - meq + 1) * (mineq + 2) + 2 * mineq + (n1 + mineq) * (n1 - meq) + 2 * meq + n1 + (n + 1) * n // 2 + 2 * m + 3 * n + 3 * n1 + 1
    len_jw = mineq
    w = zeros(len_w)
    jw = zeros(len_jw)
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = array([(_arr_to_scalar(l), _arr_to_scalar(u)) for (l, u) in bounds], float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not compatible with that of x0.')
        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]
        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' % ', '.join((str(b) for b in bnderr)))
        (xl, xu) = (bnds[:, 0], bnds[:, 1])
        infbnd = ~isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan
    sf = _prepare_scalar_function(func, x, jac=jac, args=args, epsilon=eps, finite_diff_rel_step=finite_diff_rel_step, bounds=new_bounds)
    wrapped_fun = _clip_x_for_func(sf.fun, new_bounds)
    wrapped_grad = _clip_x_for_func(sf.grad, new_bounds)
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(iter, int)
    majiter_prev = 0
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float)
    h2 = array(0, float)
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)
    if iprint >= 2:
        print('%5s %5s %16s %16s' % ('NIT', 'FC', 'OBJFUN', 'GNORM'))
    fx = wrapped_fun(x)
    g = append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)
    while 1:
        slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw, alpha, f0, gs, h1, h2, h3, h4, t, t0, tol, iexact, incons, ireset, itermx, line, n1, n2, n3)
        if mode == 1:
            fx = wrapped_fun(x)
            c = _eval_constraint(x, cons)
        if mode == -1:
            g = append(wrapped_grad(x), 0.0)
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)
        if majiter > majiter_prev:
            if callback is not None:
                callback(np.copy(x))
            if iprint >= 2:
                print('%5i %5i % 16.6E % 16.6E' % (majiter, sf.nfev, fx, linalg.norm(g)))
        if abs(mode) != 1:
            break
        majiter_prev = int(majiter)
    if iprint >= 1:
        print(exit_modes[int(mode)] + '    (Exit mode ' + str(mode) + ')')
        print('            Current function value:', fx)
        print('            Iterations:', majiter)
        print('            Function evaluations:', sf.nfev)
        print('            Gradient evaluations:', sf.ngev)
    return OptimizeResult(x=x, fun=fx, jac=g[:-1], nit=int(majiter), nfev=sf.nfev, njev=sf.ngev, status=int(mode), message=exit_modes[int(mode)], success=mode == 0)

def _eval_constraint(x, cons):
    if False:
        i = 10
        return i + 15
    if cons['eq']:
        c_eq = concatenate([atleast_1d(con['fun'](x, *con['args'])) for con in cons['eq']])
    else:
        c_eq = zeros(0)
    if cons['ineq']:
        c_ieq = concatenate([atleast_1d(con['fun'](x, *con['args'])) for con in cons['ineq']])
    else:
        c_ieq = zeros(0)
    c = concatenate((c_eq, c_ieq))
    return c

def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    if False:
        for i in range(10):
            print('nop')
    if cons['eq']:
        a_eq = vstack([con['jac'](x, *con['args']) for con in cons['eq']])
    else:
        a_eq = zeros((meq, n))
    if cons['ineq']:
        a_ieq = vstack([con['jac'](x, *con['args']) for con in cons['ineq']])
    else:
        a_ieq = zeros((mieq, n))
    if m == 0:
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))
    a = concatenate((a, zeros([la, 1])), 1)
    return a