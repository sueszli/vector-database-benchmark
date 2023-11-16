"""
Interface to Constrained Optimization By Linear Approximation

Functions
---------
.. autosummary::
   :toctree: generated/

    fmin_cobyla

"""
import functools
from threading import RLock
import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import OptimizeResult, _check_unknown_options, _prepare_scalar_function
try:
    from itertools import izip
except ImportError:
    izip = zip
__all__ = ['fmin_cobyla']
_module_lock = RLock()

def synchronized(func):
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        with _module_lock:
            return func(*args, **kwargs)
    return wrapper

@synchronized
def fmin_cobyla(func, x0, cons, args=(), consargs=None, rhobeg=1.0, rhoend=0.0001, maxfun=1000, disp=None, catol=0.0002, *, callback=None):
    if False:
        return 10
    '\n    Minimize a function using the Constrained Optimization By Linear\n    Approximation (COBYLA) method. This method wraps a FORTRAN\n    implementation of the algorithm.\n\n    Parameters\n    ----------\n    func : callable\n        Function to minimize. In the form func(x, \\*args).\n    x0 : ndarray\n        Initial guess.\n    cons : sequence\n        Constraint functions; must all be ``>=0`` (a single function\n        if only 1 constraint). Each function takes the parameters `x`\n        as its first argument, and it can return either a single number or\n        an array or list of numbers.\n    args : tuple, optional\n        Extra arguments to pass to function.\n    consargs : tuple, optional\n        Extra arguments to pass to constraint functions (default of None means\n        use same extra arguments as those passed to func).\n        Use ``()`` for no extra arguments.\n    rhobeg : float, optional\n        Reasonable initial changes to the variables.\n    rhoend : float, optional\n        Final accuracy in the optimization (not precisely guaranteed). This\n        is a lower bound on the size of the trust region.\n    disp : {0, 1, 2, 3}, optional\n        Controls the frequency of output; 0 implies no output.\n    maxfun : int, optional\n        Maximum number of function evaluations.\n    catol : float, optional\n        Absolute tolerance for constraint violations.\n    callback : callable, optional\n        Called after each iteration, as ``callback(x)``, where ``x`` is the\n        current parameter vector.\n\n    Returns\n    -------\n    x : ndarray\n        The argument that minimises `f`.\n\n    See also\n    --------\n    minimize: Interface to minimization algorithms for multivariate\n        functions. See the \'COBYLA\' `method` in particular.\n\n    Notes\n    -----\n    This algorithm is based on linear approximations to the objective\n    function and each constraint. We briefly describe the algorithm.\n\n    Suppose the function is being minimized over k variables. At the\n    jth iteration the algorithm has k+1 points v_1, ..., v_(k+1),\n    an approximate solution x_j, and a radius RHO_j.\n    (i.e., linear plus a constant) approximations to the objective\n    function and constraint functions such that their function values\n    agree with the linear approximation on the k+1 points v_1,.., v_(k+1).\n    This gives a linear program to solve (where the linear approximations\n    of the constraint functions are constrained to be non-negative).\n\n    However, the linear approximations are likely only good\n    approximations near the current simplex, so the linear program is\n    given the further requirement that the solution, which\n    will become x_(j+1), must be within RHO_j from x_j. RHO_j only\n    decreases, never increases. The initial RHO_j is rhobeg and the\n    final RHO_j is rhoend. In this way COBYLA\'s iterations behave\n    like a trust region algorithm.\n\n    Additionally, the linear program may be inconsistent, or the\n    approximation may give poor improvement. For details about\n    how these issues are resolved, as well as how the points v_i are\n    updated, refer to the source code or the references below.\n\n\n    References\n    ----------\n    Powell M.J.D. (1994), "A direct search optimization method that models\n    the objective and constraint functions by linear interpolation.", in\n    Advances in Optimization and Numerical Analysis, eds. S. Gomez and\n    J-P Hennart, Kluwer Academic (Dordrecht), pp. 51-67\n\n    Powell M.J.D. (1998), "Direct search algorithms for optimization\n    calculations", Acta Numerica 7, 287-336\n\n    Powell M.J.D. (2007), "A view of algorithms for optimization without\n    derivatives", Cambridge University Technical Report DAMTP 2007/NA03\n\n\n    Examples\n    --------\n    Minimize the objective function f(x,y) = x*y subject\n    to the constraints x**2 + y**2 < 1 and y > 0::\n\n        >>> def objective(x):\n        ...     return x[0]*x[1]\n        ...\n        >>> def constr1(x):\n        ...     return 1 - (x[0]**2 + x[1]**2)\n        ...\n        >>> def constr2(x):\n        ...     return x[1]\n        ...\n        >>> from scipy.optimize import fmin_cobyla\n        >>> fmin_cobyla(objective, [0.0, 0.1], [constr1, constr2], rhoend=1e-7)\n        array([-0.70710685,  0.70710671])\n\n    The exact solution is (-sqrt(2)/2, sqrt(2)/2).\n\n\n\n    '
    err = 'cons must be a sequence of callable functions or a single callable function.'
    try:
        len(cons)
    except TypeError as e:
        if callable(cons):
            cons = [cons]
        else:
            raise TypeError(err) from e
    else:
        for thisfunc in cons:
            if not callable(thisfunc):
                raise TypeError(err)
    if consargs is None:
        consargs = args
    con = tuple(({'type': 'ineq', 'fun': c, 'args': consargs} for c in cons))
    opts = {'rhobeg': rhobeg, 'tol': rhoend, 'disp': disp, 'maxiter': maxfun, 'catol': catol, 'callback': callback}
    sol = _minimize_cobyla(func, x0, args, constraints=con, **opts)
    if disp and (not sol['success']):
        print(f'COBYLA failed to find a solution: {sol.message}')
    return sol['x']

@synchronized
def _minimize_cobyla(fun, x0, args=(), constraints=(), rhobeg=1.0, tol=0.0001, maxiter=1000, disp=False, catol=0.0002, callback=None, bounds=None, **unknown_options):
    if False:
        return 10
    '\n    Minimize a scalar function of one or more variables using the\n    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.\n\n    Options\n    -------\n    rhobeg : float\n        Reasonable initial changes to the variables.\n    tol : float\n        Final accuracy in the optimization (not precisely guaranteed).\n        This is a lower bound on the size of the trust region.\n    disp : bool\n        Set to True to print convergence messages. If False,\n        `verbosity` is ignored as set to 0.\n    maxiter : int\n        Maximum number of function evaluations.\n    catol : float\n        Tolerance (absolute) for constraint violations\n\n    '
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    rhoend = tol
    iprint = int(bool(disp))
    if isinstance(constraints, dict):
        constraints = (constraints,)
    if bounds:
        i_lb = np.isfinite(bounds.lb)
        if np.any(i_lb):

            def lb_constraint(x, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                return x[i_lb] - bounds.lb[i_lb]
            constraints.append({'type': 'ineq', 'fun': lb_constraint})
        i_ub = np.isfinite(bounds.ub)
        if np.any(i_ub):

            def ub_constraint(x):
                if False:
                    print('Hello World!')
                return bounds.ub[i_ub] - x[i_ub]
            constraints.append({'type': 'ineq', 'fun': ub_constraint})
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
            if ctype != 'ineq':
                raise ValueError("Constraints of type '%s' not handled by COBYLA." % con['type'])
        if 'fun' not in con:
            raise KeyError('Constraint %d has no function defined.' % ic)
        if 'args' not in con:
            con['args'] = ()
    cons_lengths = []
    for c in constraints:
        f = c['fun'](x0, *c['args'])
        try:
            cons_length = len(f)
        except TypeError:
            cons_length = 1
        cons_lengths.append(cons_length)
    m = sum(cons_lengths)

    def _jac(x, *args):
        if False:
            while True:
                i = 10
        return None
    sf = _prepare_scalar_function(fun, x0, args=args, jac=_jac)

    def calcfc(x, con):
        if False:
            print('Hello World!')
        f = sf.fun(x)
        i = 0
        for (size, c) in izip(cons_lengths, constraints):
            con[i:i + size] = c['fun'](x, *c['args'])
            i += size
        return f

    def wrapped_callback(x):
        if False:
            while True:
                i = 10
        if callback is not None:
            callback(np.copy(x))
    info = np.zeros(4, np.float64)
    (xopt, info) = cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg, rhoend=rhoend, iprint=iprint, maxfun=maxfun, dinfo=info, callback=wrapped_callback)
    if info[3] > catol:
        info[0] = 4
    return OptimizeResult(x=xopt, status=int(info[0]), success=info[0] == 1, message={1: 'Optimization terminated successfully.', 2: 'Maximum number of function evaluations has been exceeded.', 3: 'Rounding errors are becoming damaging in COBYLA subroutine.', 4: 'Did not converge to a solution satisfying the constraints. See `maxcv` for magnitude of violation.', 5: 'NaN result encountered.'}.get(info[0], 'Unknown exit status.'), nfev=int(info[1]), fun=info[2], maxcv=info[3])