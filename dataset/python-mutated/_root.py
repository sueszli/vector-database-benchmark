"""
Unified interfaces to root finding algorithms.

Functions
---------
- root : find a root of a vector function.
"""
__all__ = ['root']
import numpy as np
from warnings import warn
from ._optimize import MemoizeJac, OptimizeResult, _check_unknown_options
from ._minpack_py import _root_hybr, leastsq
from ._spectral import _root_df_sane
from . import _nonlin as nonlin
ROOT_METHODS = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']

def root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None):
    if False:
        while True:
            i = 10
    "\n    Find a root of a vector function.\n\n    Parameters\n    ----------\n    fun : callable\n        A vector function to find a root of.\n    x0 : ndarray\n        Initial guess.\n    args : tuple, optional\n        Extra arguments passed to the objective function and its Jacobian.\n    method : str, optional\n        Type of solver. Should be one of\n\n            - 'hybr'             :ref:`(see here) <optimize.root-hybr>`\n            - 'lm'               :ref:`(see here) <optimize.root-lm>`\n            - 'broyden1'         :ref:`(see here) <optimize.root-broyden1>`\n            - 'broyden2'         :ref:`(see here) <optimize.root-broyden2>`\n            - 'anderson'         :ref:`(see here) <optimize.root-anderson>`\n            - 'linearmixing'     :ref:`(see here) <optimize.root-linearmixing>`\n            - 'diagbroyden'      :ref:`(see here) <optimize.root-diagbroyden>`\n            - 'excitingmixing'   :ref:`(see here) <optimize.root-excitingmixing>`\n            - 'krylov'           :ref:`(see here) <optimize.root-krylov>`\n            - 'df-sane'          :ref:`(see here) <optimize.root-dfsane>`\n\n    jac : bool or callable, optional\n        If `jac` is a Boolean and is True, `fun` is assumed to return the\n        value of Jacobian along with the objective function. If False, the\n        Jacobian will be estimated numerically.\n        `jac` can also be a callable returning the Jacobian of `fun`. In\n        this case, it must accept the same arguments as `fun`.\n    tol : float, optional\n        Tolerance for termination. For detailed control, use solver-specific\n        options.\n    callback : function, optional\n        Optional callback function. It is called on every iteration as\n        ``callback(x, f)`` where `x` is the current solution and `f`\n        the corresponding residual. For all methods but 'hybr' and 'lm'.\n    options : dict, optional\n        A dictionary of solver options. E.g., `xtol` or `maxiter`, see\n        :obj:`show_options()` for details.\n\n    Returns\n    -------\n    sol : OptimizeResult\n        The solution represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the algorithm exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n    See also\n    --------\n    show_options : Additional options accepted by the solvers\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    'method' parameter. The default method is *hybr*.\n\n    Method *hybr* uses a modification of the Powell hybrid method as\n    implemented in MINPACK [1]_.\n\n    Method *lm* solves the system of nonlinear equations in a least squares\n    sense using a modification of the Levenberg-Marquardt algorithm as\n    implemented in MINPACK [1]_.\n\n    Method *df-sane* is a derivative-free spectral method. [3]_\n\n    Methods *broyden1*, *broyden2*, *anderson*, *linearmixing*,\n    *diagbroyden*, *excitingmixing*, *krylov* are inexact Newton methods,\n    with backtracking or full line searches [2]_. Each method corresponds\n    to a particular Jacobian approximations.\n\n    - Method *broyden1* uses Broyden's first Jacobian approximation, it is\n      known as Broyden's good method.\n    - Method *broyden2* uses Broyden's second Jacobian approximation, it\n      is known as Broyden's bad method.\n    - Method *anderson* uses (extended) Anderson mixing.\n    - Method *Krylov* uses Krylov approximation for inverse Jacobian. It\n      is suitable for large-scale problem.\n    - Method *diagbroyden* uses diagonal Broyden Jacobian approximation.\n    - Method *linearmixing* uses a scalar Jacobian approximation.\n    - Method *excitingmixing* uses a tuned diagonal Jacobian\n      approximation.\n\n    .. warning::\n\n        The algorithms implemented for methods *diagbroyden*,\n        *linearmixing* and *excitingmixing* may be useful for specific\n        problems, but whether they will work may depend strongly on the\n        problem.\n\n    .. versionadded:: 0.11.0\n\n    References\n    ----------\n    .. [1] More, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.\n       1980. User Guide for MINPACK-1.\n    .. [2] C. T. Kelley. 1995. Iterative Methods for Linear and Nonlinear\n       Equations. Society for Industrial and Applied Mathematics.\n       <https://archive.siam.org/books/kelley/fr16/>\n    .. [3] W. La Cruz, J.M. Martinez, M. Raydan. Math. Comp. 75, 1429 (2006).\n\n    Examples\n    --------\n    The following functions define a system of nonlinear equations and its\n    jacobian.\n\n    >>> import numpy as np\n    >>> def fun(x):\n    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,\n    ...             0.5 * (x[1] - x[0])**3 + x[1]]\n\n    >>> def jac(x):\n    ...     return np.array([[1 + 1.5 * (x[0] - x[1])**2,\n    ...                       -1.5 * (x[0] - x[1])**2],\n    ...                      [-1.5 * (x[1] - x[0])**2,\n    ...                       1 + 1.5 * (x[1] - x[0])**2]])\n\n    A solution can be obtained as follows.\n\n    >>> from scipy import optimize\n    >>> sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')\n    >>> sol.x\n    array([ 0.8411639,  0.1588361])\n\n    **Large problem**\n\n    Suppose that we needed to solve the following integrodifferential\n    equation on the square :math:`[0,1]\\times[0,1]`:\n\n    .. math::\n\n       \\nabla^2 P = 10 \\left(\\int_0^1\\int_0^1\\cosh(P)\\,dx\\,dy\\right)^2\n\n    with :math:`P(x,1) = 1` and :math:`P=0` elsewhere on the boundary of\n    the square.\n\n    The solution can be found using the ``method='krylov'`` solver:\n\n    >>> from scipy import optimize\n    >>> # parameters\n    >>> nx, ny = 75, 75\n    >>> hx, hy = 1./(nx-1), 1./(ny-1)\n\n    >>> P_left, P_right = 0, 0\n    >>> P_top, P_bottom = 1, 0\n\n    >>> def residual(P):\n    ...    d2x = np.zeros_like(P)\n    ...    d2y = np.zeros_like(P)\n    ...\n    ...    d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx\n    ...    d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx\n    ...    d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx\n    ...\n    ...    d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy\n    ...    d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy\n    ...    d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy\n    ...\n    ...    return d2x + d2y - 10*np.cosh(P).mean()**2\n\n    >>> guess = np.zeros((nx, ny), float)\n    >>> sol = optimize.root(residual, guess, method='krylov')\n    >>> print('Residual: %g' % abs(residual(sol.x)).max())\n    Residual: 5.7972e-06  # may vary\n\n    >>> import matplotlib.pyplot as plt\n    >>> x, y = np.mgrid[0:1:(nx*1j), 0:1:(ny*1j)]\n    >>> plt.pcolormesh(x, y, sol.x, shading='gouraud')\n    >>> plt.colorbar()\n    >>> plt.show()\n\n    "
    if not isinstance(args, tuple):
        args = (args,)
    meth = method.lower()
    if options is None:
        options = {}
    if callback is not None and meth in ('hybr', 'lm'):
        warn('Method %s does not accept callback.' % method, RuntimeWarning)
    if not callable(jac) and meth in ('hybr', 'lm'):
        if bool(jac):
            fun = MemoizeJac(fun)
            jac = fun.derivative
        else:
            jac = None
    if tol is not None:
        options = dict(options)
        if meth in ('hybr', 'lm'):
            options.setdefault('xtol', tol)
        elif meth in ('df-sane',):
            options.setdefault('ftol', tol)
        elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov'):
            options.setdefault('xtol', tol)
            options.setdefault('xatol', np.inf)
            options.setdefault('ftol', np.inf)
            options.setdefault('fatol', np.inf)
    if meth == 'hybr':
        sol = _root_hybr(fun, x0, args=args, jac=jac, **options)
    elif meth == 'lm':
        sol = _root_leastsq(fun, x0, args=args, jac=jac, **options)
    elif meth == 'df-sane':
        _warn_jac_unused(jac, method)
        sol = _root_df_sane(fun, x0, args=args, callback=callback, **options)
    elif meth in ('broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov'):
        _warn_jac_unused(jac, method)
        sol = _root_nonlin_solve(fun, x0, args=args, jac=jac, _method=meth, _callback=callback, **options)
    else:
        raise ValueError('Unknown solver %s' % method)
    return sol

def _warn_jac_unused(jac, method):
    if False:
        print('Hello World!')
    if jac is not None:
        warn(f'Method {method} does not use the jacobian (jac).', RuntimeWarning)

def _root_leastsq(fun, x0, args=(), jac=None, col_deriv=0, xtol=1.49012e-08, ftol=1.49012e-08, gtol=0.0, maxiter=0, eps=0.0, factor=100, diag=None, **unknown_options):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solve for least squares with Levenberg-Marquardt\n\n    Options\n    -------\n    col_deriv : bool\n        non-zero to specify that the Jacobian function computes derivatives\n        down the columns (faster, because there is no transpose operation).\n    ftol : float\n        Relative error desired in the sum of squares.\n    xtol : float\n        Relative error desired in the approximate solution.\n    gtol : float\n        Orthogonality desired between the function vector and the columns\n        of the Jacobian.\n    maxiter : int\n        The maximum number of calls to the function. If zero, then\n        100*(N+1) is the maximum where N is the number of elements in x0.\n    epsfcn : float\n        A suitable step length for the forward-difference approximation of\n        the Jacobian (for Dfun=None). If epsfcn is less than the machine\n        precision, it is assumed that the relative errors in the functions\n        are of the order of the machine precision.\n    factor : float\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.\n    diag : sequence\n        N positive entries that serve as a scale factors for the variables.\n    '
    _check_unknown_options(unknown_options)
    (x, cov_x, info, msg, ier) = leastsq(fun, x0, args=args, Dfun=jac, full_output=True, col_deriv=col_deriv, xtol=xtol, ftol=ftol, gtol=gtol, maxfev=maxiter, epsfcn=eps, factor=factor, diag=diag)
    sol = OptimizeResult(x=x, message=msg, status=ier, success=ier in (1, 2, 3, 4), cov_x=cov_x, fun=info.pop('fvec'), method='lm')
    sol.update(info)
    return sol

def _root_nonlin_solve(fun, x0, args=(), jac=None, _callback=None, _method=None, nit=None, disp=False, maxiter=None, ftol=None, fatol=None, xtol=None, xatol=None, tol_norm=None, line_search='armijo', jac_options=None, **unknown_options):
    if False:
        while True:
            i = 10
    _check_unknown_options(unknown_options)
    f_tol = fatol
    f_rtol = ftol
    x_tol = xatol
    x_rtol = xtol
    verbose = disp
    if jac_options is None:
        jac_options = dict()
    jacobian = {'broyden1': nonlin.BroydenFirst, 'broyden2': nonlin.BroydenSecond, 'anderson': nonlin.Anderson, 'linearmixing': nonlin.LinearMixing, 'diagbroyden': nonlin.DiagBroyden, 'excitingmixing': nonlin.ExcitingMixing, 'krylov': nonlin.KrylovJacobian}[_method]
    if args:
        if jac is True:

            def f(x):
                if False:
                    i = 10
                    return i + 15
                return fun(x, *args)[0]
        else:

            def f(x):
                if False:
                    while True:
                        i = 10
                return fun(x, *args)
    else:
        f = fun
    (x, info) = nonlin.nonlin_solve(f, x0, jacobian=jacobian(**jac_options), iter=nit, verbose=verbose, maxiter=maxiter, f_tol=f_tol, f_rtol=f_rtol, x_tol=x_tol, x_rtol=x_rtol, tol_norm=tol_norm, line_search=line_search, callback=_callback, full_output=True, raise_exception=False)
    sol = OptimizeResult(x=x, method=_method)
    sol.update(info)
    return sol

def _root_broyden1_doc():
    if False:
        for i in range(10):
            print('nop')
    "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n            alpha : float, optional\n                Initial guess for the Jacobian is (-1/alpha).\n            reduction_method : str or tuple, optional\n                Method used in ensuring that the rank of the Broyden\n                matrix stays low. Can either be a string giving the\n                name of the method, or a tuple of the form ``(method,\n                param1, param2, ...)`` that gives the name of the\n                method and values for additional parameters.\n\n                Methods available:\n\n                    - ``restart``\n                        Drop all matrix columns. Has no\n                        extra parameters.\n                    - ``simple``\n                        Drop oldest matrix column. Has no\n                        extra parameters.\n                    - ``svd``\n                        Keep only the most significant SVD\n                        components.\n\n                        Extra parameters:\n\n                            - ``to_retain``\n                                Number of SVD components to\n                                retain when rank reduction is done.\n                                Default is ``max_rank - 2``.\n            max_rank : int, optional\n                Maximum rank for the Broyden matrix.\n                Default is infinity (i.e., no rank reduction).\n\n    Examples\n    --------\n    >>> def func(x):\n    ...     return np.cos(x) + x[::-1] - [1, 2, 3, 4]\n    ...\n    >>> from scipy import optimize\n    >>> res = optimize.root(func, [1, 1, 1, 1], method='broyden1', tol=1e-14)\n    >>> x = res.x\n    >>> x\n    array([4.04674914, 3.91158389, 2.71791677, 1.61756251])\n    >>> np.cos(x) + x[::-1]\n    array([1., 2., 3., 4.])\n\n    "
    pass

def _root_broyden2_doc():
    if False:
        for i in range(10):
            print('nop')
    "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            Initial guess for the Jacobian is (-1/alpha).\n        reduction_method : str or tuple, optional\n            Method used in ensuring that the rank of the Broyden\n            matrix stays low. Can either be a string giving the\n            name of the method, or a tuple of the form ``(method,\n            param1, param2, ...)`` that gives the name of the\n            method and values for additional parameters.\n\n            Methods available:\n\n                - ``restart``\n                    Drop all matrix columns. Has no\n                    extra parameters.\n                - ``simple``\n                    Drop oldest matrix column. Has no\n                    extra parameters.\n                - ``svd``\n                    Keep only the most significant SVD\n                    components.\n\n                    Extra parameters:\n\n                        - ``to_retain``\n                            Number of SVD components to\n                            retain when rank reduction is done.\n                            Default is ``max_rank - 2``.\n        max_rank : int, optional\n            Maximum rank for the Broyden matrix.\n            Default is infinity (i.e., no rank reduction).\n    "
    pass

def _root_anderson_doc():
    if False:
        print('Hello World!')
    "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            Initial guess for the Jacobian is (-1/alpha).\n        M : float, optional\n            Number of previous vectors to retain. Defaults to 5.\n        w0 : float, optional\n            Regularization parameter for numerical stability.\n            Compared to unity, good values of the order of 0.01.\n    "
    pass

def _root_linearmixing_doc():
    if False:
        i = 10
        return i + 15
    "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, ``NoConvergence`` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            initial guess for the jacobian is (-1/alpha).\n    "
    pass

def _root_diagbroyden_doc():
    if False:
        while True:
            i = 10
    "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            initial guess for the jacobian is (-1/alpha).\n    "
    pass

def _root_excitingmixing_doc():
    if False:
        while True:
            i = 10
    "\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, 'armijo' (default), 'wolfe'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        'armijo'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        alpha : float, optional\n            Initial Jacobian approximation is (-1/alpha).\n        alphamax : float, optional\n            The entries of the diagonal Jacobian are kept in the range\n            ``[alpha, alphamax]``.\n    "
    pass

def _root_krylov_doc():
    if False:
        while True:
            i = 10
    '\n    Options\n    -------\n    nit : int, optional\n        Number of iterations to make. If omitted (default), make as many\n        as required to meet tolerances.\n    disp : bool, optional\n        Print status to stdout on every iteration.\n    maxiter : int, optional\n        Maximum number of iterations to make. If more are needed to\n        meet convergence, `NoConvergence` is raised.\n    ftol : float, optional\n        Relative tolerance for the residual. If omitted, not used.\n    fatol : float, optional\n        Absolute tolerance (in max-norm) for the residual.\n        If omitted, default is 6e-6.\n    xtol : float, optional\n        Relative minimum step size. If omitted, not used.\n    xatol : float, optional\n        Absolute minimum step size, as determined from the Jacobian\n        approximation. If the step size is smaller than this, optimization\n        is terminated as successful. If omitted, not used.\n    tol_norm : function(vector) -> scalar, optional\n        Norm to use in convergence check. Default is the maximum norm.\n    line_search : {None, \'armijo\' (default), \'wolfe\'}, optional\n        Which type of a line search to use to determine the step size in\n        the direction given by the Jacobian approximation. Defaults to\n        \'armijo\'.\n    jac_options : dict, optional\n        Options for the respective Jacobian approximation.\n\n        rdiff : float, optional\n            Relative step size to use in numerical differentiation.\n        method : str or callable, optional\n            Krylov method to use to approximate the Jacobian.  Can be a string,\n            or a function implementing the same interface as the iterative\n            solvers in `scipy.sparse.linalg`. If a string, needs to be one of:\n            ``\'lgmres\'``, ``\'gmres\'``, ``\'bicgstab\'``, ``\'cgs\'``, ``\'minres\'``,\n            ``\'tfqmr\'``.\n\n            The default is `scipy.sparse.linalg.lgmres`.\n        inner_M : LinearOperator or InverseJacobian\n            Preconditioner for the inner Krylov iteration.\n            Note that you can use also inverse Jacobians as (adaptive)\n            preconditioners. For example,\n\n            >>> jac = BroydenFirst()\n            >>> kjac = KrylovJacobian(inner_M=jac.inverse).\n\n            If the preconditioner has a method named \'update\', it will\n            be called as ``update(x, f)`` after each nonlinear step,\n            with ``x`` giving the current point, and ``f`` the current\n            function value.\n        inner_tol, inner_maxiter, ...\n            Parameters to pass on to the "inner" Krylov solver.\n            See `scipy.sparse.linalg.gmres` for details.\n        outer_k : int, optional\n            Size of the subspace kept across LGMRES nonlinear\n            iterations.\n\n            See `scipy.sparse.linalg.lgmres` for details.\n    '
    pass