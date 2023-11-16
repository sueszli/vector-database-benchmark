import warnings
from . import _minpack
import numpy as np
from numpy import atleast_1d, triu, shape, transpose, zeros, prod, greater, asarray, inf, finfo, inexact, issubdtype, dtype
from scipy import linalg
from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError
from scipy._lib._util import _asarray_validated, _lazywhere, _contains_nan
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._optimize import OptimizeResult, _check_unknown_options, OptimizeWarning
from ._lsq import least_squares
from ._lsq.least_squares import prepare_bounds
from scipy.optimize._minimize import Bounds
error = _minpack.error
from numpy import dot, eye, take
from numpy.linalg import inv
__all__ = ['fsolve', 'leastsq', 'fixed_point', 'curve_fit']

def _check_func(checker, argname, thefunc, x0, args, numinputs, output_shape=None):
    if False:
        while True:
            i = 10
    res = atleast_1d(thefunc(*(x0[:numinputs],) + args))
    if output_shape is not None and shape(res) != output_shape:
        if output_shape[0] != 1:
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = "{}: there is a mismatch between the input and output shape of the '{}' argument".format(checker, argname)
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += " '%s'." % func_name
            else:
                msg += '.'
            msg += f'Shape should be {output_shape} but it is {shape(res)}.'
            raise TypeError(msg)
    if issubdtype(res.dtype, inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return (shape(res), dt)

def fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the roots of a function.\n\n    Return the roots of the (non-linear) equations defined by\n    ``func(x) = 0`` given a starting estimate.\n\n    Parameters\n    ----------\n    func : callable ``f(x, *args)``\n        A function that takes at least one (possibly vector) argument,\n        and returns a value of the same length.\n    x0 : ndarray\n        The starting estimate for the roots of ``func(x) = 0``.\n    args : tuple, optional\n        Any extra arguments to `func`.\n    fprime : callable ``f(x, *args)``, optional\n        A function to compute the Jacobian of `func` with derivatives\n        across the rows. By default, the Jacobian will be estimated.\n    full_output : bool, optional\n        If True, return optional outputs.\n    col_deriv : bool, optional\n        Specify whether the Jacobian function computes derivatives down\n        the columns (faster, because there is no transpose operation).\n    xtol : float, optional\n        The calculation will terminate if the relative error between two\n        consecutive iterates is at most `xtol`.\n    maxfev : int, optional\n        The maximum number of calls to the function. If zero, then\n        ``100*(N+1)`` is the maximum where N is the number of elements\n        in `x0`.\n    band : tuple, optional\n        If set to a two-sequence containing the number of sub- and\n        super-diagonals within the band of the Jacobi matrix, the\n        Jacobi matrix is considered banded (only for ``fprime=None``).\n    epsfcn : float, optional\n        A suitable step length for the forward-difference\n        approximation of the Jacobian (for ``fprime=None``). If\n        `epsfcn` is less than the machine precision, it is assumed\n        that the relative errors in the functions are of the order of\n        the machine precision.\n    factor : float, optional\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``). Should be in the interval\n        ``(0.1, 100)``.\n    diag : sequence, optional\n        N positive entries that serve as a scale factors for the\n        variables.\n\n    Returns\n    -------\n    x : ndarray\n        The solution (or the result of the last iteration for\n        an unsuccessful call).\n    infodict : dict\n        A dictionary of optional outputs with the keys:\n\n        ``nfev``\n            number of function calls\n        ``njev``\n            number of Jacobian calls\n        ``fvec``\n            function evaluated at the output\n        ``fjac``\n            the orthogonal matrix, q, produced by the QR\n            factorization of the final approximate Jacobian\n            matrix, stored column wise\n        ``r``\n            upper triangular matrix produced by QR factorization\n            of the same matrix\n        ``qtf``\n            the vector ``(transpose(q) * fvec)``\n\n    ier : int\n        An integer flag.  Set to 1 if a solution was found, otherwise refer\n        to `mesg` for more information.\n    mesg : str\n        If no solution is found, `mesg` details the cause of failure.\n\n    See Also\n    --------\n    root : Interface to root finding algorithms for multivariate\n           functions. See the ``method='hybr'`` in particular.\n\n    Notes\n    -----\n    ``fsolve`` is a wrapper around MINPACK's hybrd and hybrj algorithms.\n\n    Examples\n    --------\n    Find a solution to the system of equations:\n    ``x0*cos(x1) = 4,  x1*x0 - x1 = 5``.\n\n    >>> import numpy as np\n    >>> from scipy.optimize import fsolve\n    >>> def func(x):\n    ...     return [x[0] * np.cos(x[1]) - 4,\n    ...             x[1] * x[0] - x[1] - 5]\n    >>> root = fsolve(func, [1, 1])\n    >>> root\n    array([6.50409711, 0.90841421])\n    >>> np.isclose(func(root), [0.0, 0.0])  # func(root) should be almost 0.0.\n    array([ True,  True])\n\n    "
    options = {'col_deriv': col_deriv, 'xtol': xtol, 'maxfev': maxfev, 'band': band, 'eps': epsfcn, 'factor': factor, 'diag': diag}
    res = _root_hybr(func, x0, args, jac=fprime, **options)
    if full_output:
        x = res['x']
        info = {k: res.get(k) for k in ('nfev', 'njev', 'fjac', 'r', 'qtf') if k in res}
        info['fvec'] = res['fun']
        return (x, info, res['status'], res['message'])
    else:
        status = res['status']
        msg = res['message']
        if status == 0:
            raise TypeError(msg)
        elif status == 1:
            pass
        elif status in [2, 3, 4, 5]:
            warnings.warn(msg, RuntimeWarning)
        else:
            raise TypeError(msg)
        return res['x']

def _root_hybr(func, x0, args=(), jac=None, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, eps=None, factor=100, diag=None, **unknown_options):
    if False:
        print('Hello World!')
    "\n    Find the roots of a multivariate function using MINPACK's hybrd and\n    hybrj routines (modified Powell method).\n\n    Options\n    -------\n    col_deriv : bool\n        Specify whether the Jacobian function computes derivatives down\n        the columns (faster, because there is no transpose operation).\n    xtol : float\n        The calculation will terminate if the relative error between two\n        consecutive iterates is at most `xtol`.\n    maxfev : int\n        The maximum number of calls to the function. If zero, then\n        ``100*(N+1)`` is the maximum where N is the number of elements\n        in `x0`.\n    band : tuple\n        If set to a two-sequence containing the number of sub- and\n        super-diagonals within the band of the Jacobi matrix, the\n        Jacobi matrix is considered banded (only for ``fprime=None``).\n    eps : float\n        A suitable step length for the forward-difference\n        approximation of the Jacobian (for ``fprime=None``). If\n        `eps` is less than the machine precision, it is assumed\n        that the relative errors in the functions are of the order of\n        the machine precision.\n    factor : float\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``). Should be in the interval\n        ``(0.1, 100)``.\n    diag : sequence\n        N positive entries that serve as a scale factors for the\n        variables.\n\n    "
    _check_unknown_options(unknown_options)
    epsfcn = eps
    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    (shape, dtype) = _check_func('fsolve', 'func', func, x0, args, n, (n,))
    if epsfcn is None:
        epsfcn = finfo(dtype).eps
    Dfun = jac
    if Dfun is None:
        if band is None:
            (ml, mu) = (-10, -10)
        else:
            (ml, mu) = band[:2]
        if maxfev == 0:
            maxfev = 200 * (n + 1)
        retval = _minpack._hybrd(func, x0, args, 1, xtol, maxfev, ml, mu, epsfcn, factor, diag)
    else:
        _check_func('fsolve', 'fprime', Dfun, x0, args, n, (n, n))
        if maxfev == 0:
            maxfev = 100 * (n + 1)
        retval = _minpack._hybrj(func, Dfun, x0, args, 1, col_deriv, xtol, maxfev, factor, diag)
    (x, status) = (retval[0], retval[-1])
    errors = {0: 'Improper input parameters were entered.', 1: 'The solution converged.', 2: 'The number of calls to function has reached maxfev = %d.' % maxfev, 3: 'xtol=%f is too small, no further improvement in the approximate\n  solution is possible.' % xtol, 4: 'The iteration is not making good progress, as measured by the \n  improvement from the last five Jacobian evaluations.', 5: 'The iteration is not making good progress, as measured by the \n  improvement from the last ten iterations.', 'unknown': 'An error occurred.'}
    info = retval[1]
    info['fun'] = info.pop('fvec')
    sol = OptimizeResult(x=x, success=status == 1, status=status, method='hybr')
    sol.update(info)
    try:
        sol['message'] = errors[status]
    except KeyError:
        sol['message'] = errors['unknown']
    return sol
LEASTSQ_SUCCESS = [1, 2, 3, 4]
LEASTSQ_FAILURE = [5, 6, 7, 8]

def leastsq(func, x0, args=(), Dfun=None, full_output=False, col_deriv=False, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    if False:
        i = 10
        return i + 15
    '\n    Minimize the sum of squares of a set of equations.\n\n    ::\n\n        x = arg min(sum(func(y)**2,axis=0))\n                 y\n\n    Parameters\n    ----------\n    func : callable\n        Should take at least one (possibly length ``N`` vector) argument and\n        returns ``M`` floating point numbers. It must not return NaNs or\n        fitting might fail. ``M`` must be greater than or equal to ``N``.\n    x0 : ndarray\n        The starting estimate for the minimization.\n    args : tuple, optional\n        Any extra arguments to func are placed in this tuple.\n    Dfun : callable, optional\n        A function or method to compute the Jacobian of func with derivatives\n        across the rows. If this is None, the Jacobian will be estimated.\n    full_output : bool, optional\n        If ``True``, return all optional outputs (not just `x` and `ier`).\n    col_deriv : bool, optional\n        If ``True``, specify that the Jacobian function computes derivatives\n        down the columns (faster, because there is no transpose operation).\n    ftol : float, optional\n        Relative error desired in the sum of squares.\n    xtol : float, optional\n        Relative error desired in the approximate solution.\n    gtol : float, optional\n        Orthogonality desired between the function vector and the columns of\n        the Jacobian.\n    maxfev : int, optional\n        The maximum number of calls to the function. If `Dfun` is provided,\n        then the default `maxfev` is 100*(N+1) where N is the number of elements\n        in x0, otherwise the default `maxfev` is 200*(N+1).\n    epsfcn : float, optional\n        A variable used in determining a suitable step length for the forward-\n        difference approximation of the Jacobian (for Dfun=None).\n        Normally the actual step length will be sqrt(epsfcn)*x\n        If epsfcn is less than the machine precision, it is assumed that the\n        relative errors are of the order of the machine precision.\n    factor : float, optional\n        A parameter determining the initial step bound\n        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.\n    diag : sequence, optional\n        N positive entries that serve as a scale factors for the variables.\n\n    Returns\n    -------\n    x : ndarray\n        The solution (or the result of the last iteration for an unsuccessful\n        call).\n    cov_x : ndarray\n        The inverse of the Hessian. `fjac` and `ipvt` are used to construct an\n        estimate of the Hessian. A value of None indicates a singular matrix,\n        which means the curvature in parameters `x` is numerically flat. To\n        obtain the covariance matrix of the parameters `x`, `cov_x` must be\n        multiplied by the variance of the residuals -- see curve_fit. Only\n        returned if `full_output` is ``True``.\n    infodict : dict\n        a dictionary of optional outputs with the keys:\n\n        ``nfev``\n            The number of function calls\n        ``fvec``\n            The function evaluated at the output\n        ``fjac``\n            A permutation of the R matrix of a QR\n            factorization of the final approximate\n            Jacobian matrix, stored column wise.\n            Together with ipvt, the covariance of the\n            estimate can be approximated.\n        ``ipvt``\n            An integer array of length N which defines\n            a permutation matrix, p, such that\n            fjac*p = q*r, where r is upper triangular\n            with diagonal elements of nonincreasing\n            magnitude. Column j of p is column ipvt(j)\n            of the identity matrix.\n        ``qtf``\n            The vector (transpose(q) * fvec).\n\n        Only returned if `full_output` is ``True``.\n    mesg : str\n        A string message giving information about the cause of failure.\n        Only returned if `full_output` is ``True``.\n    ier : int\n        An integer flag. If it is equal to 1, 2, 3 or 4, the solution was\n        found. Otherwise, the solution was not found. In either case, the\n        optional output variable \'mesg\' gives more information.\n\n    See Also\n    --------\n    least_squares : Newer interface to solve nonlinear least-squares problems\n        with bounds on the variables. See ``method=\'lm\'`` in particular.\n\n    Notes\n    -----\n    "leastsq" is a wrapper around MINPACK\'s lmdif and lmder algorithms.\n\n    cov_x is a Jacobian approximation to the Hessian of the least squares\n    objective function.\n    This approximation assumes that the objective function is based on the\n    difference between some observed target data (ydata) and a (non-linear)\n    function of the parameters `f(xdata, params)` ::\n\n           func(params) = ydata - f(xdata, params)\n\n    so that the objective function is ::\n\n           min   sum((ydata - f(xdata, params))**2, axis=0)\n         params\n\n    The solution, `x`, is always a 1-D array, regardless of the shape of `x0`,\n    or whether `x0` is a scalar.\n\n    Examples\n    --------\n    >>> from scipy.optimize import leastsq\n    >>> def func(x):\n    ...     return 2*(x-3)**2+1\n    >>> leastsq(func, 0)\n    (array([2.99999999]), 1)\n\n    '
    x0 = asarray(x0).flatten()
    n = len(x0)
    if not isinstance(args, tuple):
        args = (args,)
    (shape, dtype) = _check_func('leastsq', 'func', func, x0, args, n)
    m = shape[0]
    if n > m:
        raise TypeError(f'Improper input: func input vector length N={n} must not exceed func output vector length M={m}')
    if epsfcn is None:
        epsfcn = finfo(dtype).eps
    if Dfun is None:
        if maxfev == 0:
            maxfev = 200 * (n + 1)
        retval = _minpack._lmdif(func, x0, args, full_output, ftol, xtol, gtol, maxfev, epsfcn, factor, diag)
    else:
        if col_deriv:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
        if maxfev == 0:
            maxfev = 100 * (n + 1)
        retval = _minpack._lmder(func, Dfun, x0, args, full_output, col_deriv, ftol, xtol, gtol, maxfev, factor, diag)
    errors = {0: ['Improper input parameters.', TypeError], 1: ['Both actual and predicted relative reductions in the sum of squares\n  are at most %f' % ftol, None], 2: ['The relative error between two consecutive iterates is at most %f' % xtol, None], 3: ['Both actual and predicted relative reductions in the sum of squares\n  are at most {:f} and the relative error between two consecutive iterates is at \n  most {:f}'.format(ftol, xtol), None], 4: ['The cosine of the angle between func(x) and any column of the\n  Jacobian is at most %f in absolute value' % gtol, None], 5: ['Number of calls to function has reached maxfev = %d.' % maxfev, ValueError], 6: ['ftol=%f is too small, no further reduction in the sum of squares\n  is possible.' % ftol, ValueError], 7: ['xtol=%f is too small, no further improvement in the approximate\n  solution is possible.' % xtol, ValueError], 8: ['gtol=%f is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision.' % gtol, ValueError]}
    info = retval[-1]
    if full_output:
        cov_x = None
        if info in LEASTSQ_SUCCESS:
            perm = retval[1]['ipvt'] - 1
            n = len(perm)
            r = triu(transpose(retval[1]['fjac'])[:n, :])
            inv_triu = linalg.get_lapack_funcs('trtri', (r,))
            try:
                (invR, trtri_info) = inv_triu(r)
                if trtri_info != 0:
                    raise LinAlgError(f'trtri returned info {trtri_info}')
                invR[perm] = invR.copy()
                cov_x = invR @ invR.T
            except (LinAlgError, ValueError):
                pass
        return (retval[0], cov_x) + retval[1:-1] + (errors[info][0], info)
    else:
        if info in LEASTSQ_FAILURE:
            warnings.warn(errors[info][0], RuntimeWarning)
        elif info == 0:
            raise errors[info][1](errors[info][0])
        return (retval[0], info)

def _lightweight_memoizer(f):
    if False:
        return 10

    def _memoized_func(params):
        if False:
            print('Hello World!')
        if _memoized_func.skip_lookup:
            return f(params)
        if np.all(_memoized_func.last_params == params):
            return _memoized_func.last_val
        elif _memoized_func.last_params is not None:
            _memoized_func.skip_lookup = True
        val = f(params)
        if _memoized_func.last_params is None:
            _memoized_func.last_params = np.copy(params)
            _memoized_func.last_val = val
        return val
    _memoized_func.last_params = None
    _memoized_func.last_val = None
    _memoized_func.skip_lookup = False
    return _memoized_func

def _wrap_func(func, xdata, ydata, transform):
    if False:
        print('Hello World!')
    if transform is None:

        def func_wrapped(params):
            if False:
                print('Hello World!')
            return func(xdata, *params) - ydata
    elif transform.size == 1 or transform.ndim == 1:

        def func_wrapped(params):
            if False:
                return 10
            return transform * (func(xdata, *params) - ydata)
    else:

        def func_wrapped(params):
            if False:
                i = 10
                return i + 15
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)
    return func_wrapped

def _wrap_jac(jac, xdata, transform):
    if False:
        print('Hello World!')
    if transform is None:

        def jac_wrapped(params):
            if False:
                print('Hello World!')
            return jac(xdata, *params)
    elif transform.ndim == 1:

        def jac_wrapped(params):
            if False:
                return 10
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
    else:

        def jac_wrapped(params):
            if False:
                for i in range(10):
                    print('nop')
            return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)
    return jac_wrapped

def _initialize_feasible(lb, ub):
    if False:
        for i in range(10):
            print('nop')
    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)
    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])
    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1
    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1
    return p0

def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-np.inf, np.inf), method=None, jac=None, *, full_output=False, nan_policy=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Use non-linear least squares to fit a function, f, to data.\n\n    Assumes ``ydata = f(xdata, *params) + eps``.\n\n    Parameters\n    ----------\n    f : callable\n        The model function, f(x, ...). It must take the independent\n        variable as the first argument and the parameters to fit as\n        separate remaining arguments.\n    xdata : array_like\n        The independent variable where the data is measured.\n        Should usually be an M-length sequence or an (k,M)-shaped array for\n        functions with k predictors, and each element should be float\n        convertible if it is an array like object.\n    ydata : array_like\n        The dependent data, a length M array - nominally ``f(xdata, ...)``.\n    p0 : array_like, optional\n        Initial guess for the parameters (length N). If None, then the\n        initial values will all be 1 (if the number of parameters for the\n        function can be determined using introspection, otherwise a\n        ValueError is raised).\n    sigma : None or scalar or M-length sequence or MxM array, optional\n        Determines the uncertainty in `ydata`. If we define residuals as\n        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`\n        depends on its number of dimensions:\n\n            - A scalar or 1-D `sigma` should contain values of standard deviations of\n              errors in `ydata`. In this case, the optimized function is\n              ``chisq = sum((r / sigma) ** 2)``.\n\n            - A 2-D `sigma` should contain the covariance matrix of\n              errors in `ydata`. In this case, the optimized function is\n              ``chisq = r.T @ inv(sigma) @ r``.\n\n              .. versionadded:: 0.19\n\n        None (default) is equivalent of 1-D `sigma` filled with ones.\n    absolute_sigma : bool, optional\n        If True, `sigma` is used in an absolute sense and the estimated parameter\n        covariance `pcov` reflects these absolute values.\n\n        If False (default), only the relative magnitudes of the `sigma` values matter.\n        The returned parameter covariance matrix `pcov` is based on scaling\n        `sigma` by a constant factor. This constant is set by demanding that the\n        reduced `chisq` for the optimal parameters `popt` when using the\n        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to\n        match the sample variance of the residuals after the fit. Default is False.\n        Mathematically,\n        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``\n    check_finite : bool, optional\n        If True, check that the input arrays do not contain nans of infs,\n        and raise a ValueError if they do. Setting this parameter to\n        False may silently produce nonsensical results if the input arrays\n        do contain nans. Default is True if `nan_policy` is not specified\n        explicitly and False otherwise.\n    bounds : 2-tuple of array_like or `Bounds`, optional\n        Lower and upper bounds on parameters. Defaults to no bounds.\n        There are two ways to specify the bounds:\n\n            - Instance of `Bounds` class.\n\n            - 2-tuple of array_like: Each element of the tuple must be either\n              an array with the length equal to the number of parameters, or a\n              scalar (in which case the bound is taken to be the same for all\n              parameters). Use ``np.inf`` with an appropriate sign to disable\n              bounds on all or some parameters.\n\n    method : {'lm', 'trf', 'dogbox'}, optional\n        Method to use for optimization. See `least_squares` for more details.\n        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are\n        provided. The method 'lm' won't work when the number of observations\n        is less than the number of variables, use 'trf' or 'dogbox' in this\n        case.\n\n        .. versionadded:: 0.17\n    jac : callable, string or None, optional\n        Function with signature ``jac(x, ...)`` which computes the Jacobian\n        matrix of the model function with respect to parameters as a dense\n        array_like structure. It will be scaled according to provided `sigma`.\n        If None (default), the Jacobian will be estimated numerically.\n        String keywords for 'trf' and 'dogbox' methods can be used to select\n        a finite difference scheme, see `least_squares`.\n\n        .. versionadded:: 0.18\n    full_output : boolean, optional\n        If True, this function returns additioal information: `infodict`,\n        `mesg`, and `ier`.\n\n        .. versionadded:: 1.9\n    nan_policy : {'raise', 'omit', None}, optional\n        Defines how to handle when input contains nan.\n        The following options are available (default is None):\n\n          * 'raise': throws an error\n          * 'omit': performs the calculations ignoring nan values\n          * None: no special handling of NaNs is performed\n            (except what is done by check_finite); the behavior when NaNs\n            are present is implementation-dependent and may change.\n\n        Note that if this value is specified explicitly (not None),\n        `check_finite` will be set as False.\n\n        .. versionadded:: 1.11\n    **kwargs\n        Keyword arguments passed to `leastsq` for ``method='lm'`` or\n        `least_squares` otherwise.\n\n    Returns\n    -------\n    popt : array\n        Optimal values for the parameters so that the sum of the squared\n        residuals of ``f(xdata, *popt) - ydata`` is minimized.\n    pcov : 2-D array\n        The estimated approximate covariance of popt. The diagonals provide\n        the variance of the parameter estimate. To compute one standard\n        deviation errors on the parameters, use\n        ``perr = np.sqrt(np.diag(pcov))``. Note that the relationship between\n        `cov` and parameter error estimates is derived based on a linear\n        approximation to the model function around the optimum [1].\n        When this approximation becomes inaccurate, `cov` may not provide an\n        accurate measure of uncertainty.\n\n        How the `sigma` parameter affects the estimated covariance\n        depends on `absolute_sigma` argument, as described above.\n\n        If the Jacobian matrix at the solution doesn't have a full rank, then\n        'lm' method returns a matrix filled with ``np.inf``, on the other hand\n        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute\n        the covariance matrix. Covariance matrices with large condition numbers\n        (e.g. computed with `numpy.linalg.cond`) may indicate that results are\n        unreliable.\n    infodict : dict (returned only if `full_output` is True)\n        a dictionary of optional outputs with the keys:\n\n        ``nfev``\n            The number of function calls. Methods 'trf' and 'dogbox' do not\n            count function calls for numerical Jacobian approximation,\n            as opposed to 'lm' method.\n        ``fvec``\n            The residual values evaluated at the solution, for a 1-D `sigma`\n            this is ``(f(x, *popt) - ydata)/sigma``.\n        ``fjac``\n            A permutation of the R matrix of a QR\n            factorization of the final approximate\n            Jacobian matrix, stored column wise.\n            Together with ipvt, the covariance of the\n            estimate can be approximated.\n            Method 'lm' only provides this information.\n        ``ipvt``\n            An integer array of length N which defines\n            a permutation matrix, p, such that\n            fjac*p = q*r, where r is upper triangular\n            with diagonal elements of nonincreasing\n            magnitude. Column j of p is column ipvt(j)\n            of the identity matrix.\n            Method 'lm' only provides this information.\n        ``qtf``\n            The vector (transpose(q) * fvec).\n            Method 'lm' only provides this information.\n\n        .. versionadded:: 1.9\n    mesg : str (returned only if `full_output` is True)\n        A string message giving information about the solution.\n\n        .. versionadded:: 1.9\n    ier : int (returnned only if `full_output` is True)\n        An integer flag. If it is equal to 1, 2, 3 or 4, the solution was\n        found. Otherwise, the solution was not found. In either case, the\n        optional output variable `mesg` gives more information.\n\n        .. versionadded:: 1.9\n\n    Raises\n    ------\n    ValueError\n        if either `ydata` or `xdata` contain NaNs, or if incompatible options\n        are used.\n\n    RuntimeError\n        if the least-squares minimization fails.\n\n    OptimizeWarning\n        if covariance of the parameters can not be estimated.\n\n    See Also\n    --------\n    least_squares : Minimize the sum of squares of nonlinear functions.\n    scipy.stats.linregress : Calculate a linear least squares regression for\n                             two sets of measurements.\n\n    Notes\n    -----\n    Users should ensure that inputs `xdata`, `ydata`, and the output of `f`\n    are ``float64``, or else the optimization may return incorrect results.\n\n    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm\n    through `leastsq`. Note that this algorithm can only deal with\n    unconstrained problems.\n\n    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to\n    the docstring of `least_squares` for more information.\n\n    References\n    ----------\n    [1] K. Vugrin et al. Confidence region estimation techniques for nonlinear\n        regression in groundwater flow: Three case studies. Water Resources\n        Research, Vol. 43, W03423, :doi:`10.1029/2005WR004804`\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.optimize import curve_fit\n\n    >>> def func(x, a, b, c):\n    ...     return a * np.exp(-b * x) + c\n\n    Define the data to be fit with some noise:\n\n    >>> xdata = np.linspace(0, 4, 50)\n    >>> y = func(xdata, 2.5, 1.3, 0.5)\n    >>> rng = np.random.default_rng()\n    >>> y_noise = 0.2 * rng.normal(size=xdata.size)\n    >>> ydata = y + y_noise\n    >>> plt.plot(xdata, ydata, 'b-', label='data')\n\n    Fit for the parameters a, b, c of the function `func`:\n\n    >>> popt, pcov = curve_fit(func, xdata, ydata)\n    >>> popt\n    array([2.56274217, 1.37268521, 0.47427475])\n    >>> plt.plot(xdata, func(xdata, *popt), 'r-',\n    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))\n\n    Constrain the optimization to the region of ``0 <= a <= 3``,\n    ``0 <= b <= 1`` and ``0 <= c <= 0.5``:\n\n    >>> popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))\n    >>> popt\n    array([2.43736712, 1.        , 0.34463856])\n    >>> plt.plot(xdata, func(xdata, *popt), 'g--',\n    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))\n\n    >>> plt.xlabel('x')\n    >>> plt.ylabel('y')\n    >>> plt.legend()\n    >>> plt.show()\n\n    For reliable results, the model `func` should not be overparametrized;\n    redundant parameters can cause unreliable covariance matrices and, in some\n    cases, poorer quality fits. As a quick check of whether the model may be\n    overparameterized, calculate the condition number of the covariance matrix:\n\n    >>> np.linalg.cond(pcov)\n    34.571092161547405  # may vary\n\n    The value is small, so it does not raise much concern. If, however, we were\n    to add a fourth parameter ``d`` to `func` with the same effect as ``a``:\n\n    >>> def func(x, a, b, c, d):\n    ...     return a * d * np.exp(-b * x) + c  # a and d are redundant\n    >>> popt, pcov = curve_fit(func, xdata, ydata)\n    >>> np.linalg.cond(pcov)\n    1.13250718925596e+32  # may vary\n\n    Such a large value is cause for concern. The diagonal elements of the\n    covariance matrix, which is related to uncertainty of the fit, gives more\n    information:\n\n    >>> np.diag(pcov)\n    array([1.48814742e+29, 3.78596560e-02, 5.39253738e-03, 2.76417220e+28])  # may vary\n\n    Note that the first and last terms are much larger than the other elements,\n    suggesting that the optimal values of these parameters are ambiguous and\n    that only one of these parameters is needed in the model.\n\n    "
    if p0 is None:
        sig = _getfullargspec(f)
        args = sig.args
        if len(args) < 2:
            raise ValueError('Unable to determine number of fit parameters.')
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(p0)
        n = p0.size
    if isinstance(bounds, Bounds):
        (lb, ub) = (bounds.lb, bounds.ub)
    else:
        (lb, ub) = prepare_bounds(bounds, n)
    if p0 is None:
        p0 = _initialize_feasible(lb, ub)
    bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
    if method is None:
        if bounded_problem:
            method = 'trf'
        else:
            method = 'lm'
    if method == 'lm' and bounded_problem:
        raise ValueError("Method 'lm' only works for unconstrained problems. Use 'trf' or 'dogbox' instead.")
    if check_finite is None:
        check_finite = True if nan_policy is None else False
    if check_finite:
        ydata = np.asarray_chkfinite(ydata, float)
    else:
        ydata = np.asarray(ydata, float)
    if isinstance(xdata, (list, tuple, np.ndarray)):
        if check_finite:
            xdata = np.asarray_chkfinite(xdata, float)
        else:
            xdata = np.asarray(xdata, float)
    if ydata.size == 0:
        raise ValueError('`ydata` must not be empty!')
    if not check_finite and nan_policy is not None:
        if nan_policy == 'propagate':
            raise ValueError("`nan_policy='propagate'` is not supported by this function.")
        policies = [None, 'raise', 'omit']
        (x_contains_nan, nan_policy) = _contains_nan(xdata, nan_policy, policies=policies)
        (y_contains_nan, nan_policy) = _contains_nan(ydata, nan_policy, policies=policies)
        if (x_contains_nan or y_contains_nan) and nan_policy == 'omit':
            has_nan = np.isnan(xdata)
            has_nan = has_nan.any(axis=tuple(range(has_nan.ndim - 1)))
            has_nan |= np.isnan(ydata)
            xdata = xdata[..., ~has_nan]
            ydata = ydata[~has_nan]
    if sigma is not None:
        sigma = np.asarray(sigma)
        if sigma.size == 1 or sigma.shape == (ydata.size,):
            transform = 1.0 / sigma
        elif sigma.shape == (ydata.size, ydata.size):
            try:
                transform = cholesky(sigma, lower=True)
            except LinAlgError as e:
                raise ValueError('`sigma` must be positive definite.') from e
        else:
            raise ValueError('`sigma` has incorrect shape.')
    else:
        transform = None
    func = _lightweight_memoizer(_wrap_func(f, xdata, ydata, transform))
    if callable(jac):
        jac = _lightweight_memoizer(_wrap_jac(jac, xdata, transform))
    elif jac is None and method != 'lm':
        jac = '2-point'
    if 'args' in kwargs:
        raise ValueError("'args' is not a supported keyword argument.")
    if method == 'lm':
        if ydata.size != 1 and n > ydata.size:
            raise TypeError(f'The number of func parameters={n} must not exceed the number of data points={ydata.size}')
        res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
        (popt, pcov, infodict, errmsg, ier) = res
        ysize = len(infodict['fvec'])
        cost = np.sum(infodict['fvec'] ** 2)
        if ier not in [1, 2, 3, 4]:
            raise RuntimeError('Optimal parameters not found: ' + errmsg)
    else:
        if 'max_nfev' not in kwargs:
            kwargs['max_nfev'] = kwargs.pop('maxfev', None)
        res = least_squares(func, p0, jac=jac, bounds=bounds, method=method, **kwargs)
        if not res.success:
            raise RuntimeError('Optimal parameters not found: ' + res.message)
        infodict = dict(nfev=res.nfev, fvec=res.fun)
        ier = res.status
        errmsg = res.message
        ysize = len(res.fun)
        cost = 2 * res.cost
        popt = res.x
        (_, s, VT) = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
    warn_cov = False
    if pcov is None or np.isnan(pcov).any():
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
            warn_cov = True
    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated', category=OptimizeWarning)
    if full_output:
        return (popt, pcov, infodict, errmsg, ier)
    else:
        return (popt, pcov)

def check_gradient(fcn, Dfcn, x0, args=(), col_deriv=0):
    if False:
        i = 10
        return i + 15
    'Perform a simple check on the gradient for correctness.\n\n    '
    x = atleast_1d(x0)
    n = len(x)
    x = x.reshape((n,))
    fvec = atleast_1d(fcn(x, *args))
    m = len(fvec)
    fvec = fvec.reshape((m,))
    ldfjac = m
    fjac = atleast_1d(Dfcn(x, *args))
    fjac = fjac.reshape((m, n))
    if col_deriv == 0:
        fjac = transpose(fjac)
    xp = zeros((n,), float)
    err = zeros((m,), float)
    fvecp = None
    _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 1, err)
    fvecp = atleast_1d(fcn(xp, *args))
    fvecp = fvecp.reshape((m,))
    _minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 2, err)
    good = prod(greater(err, 0.5), axis=0)
    return (good, err)

def _del2(p0, p1, d):
    if False:
        i = 10
        return i + 15
    return p0 - np.square(p1 - p0) / d

def _relerr(actual, desired):
    if False:
        return 10
    return (actual - desired) / desired

def _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel):
    if False:
        print('Hello World!')
    p0 = x0
    for i in range(maxiter):
        p1 = func(p0, *args)
        if use_accel:
            p2 = func(p1, *args)
            d = p2 - 2.0 * p1 + p0
            p = _lazywhere(d != 0, (p0, p1, d), f=_del2, fillvalue=p2)
        else:
            p = p1
        relerr = _lazywhere(p0 != 0, (p, p0), f=_relerr, fillvalue=p)
        if np.all(np.abs(relerr) < xtol):
            return p
        p0 = p
    msg = 'Failed to converge after %d iterations, value is %s' % (maxiter, p)
    raise RuntimeError(msg)

def fixed_point(func, x0, args=(), xtol=1e-08, maxiter=500, method='del2'):
    if False:
        print('Hello World!')
    '\n    Find a fixed point of the function.\n\n    Given a function of one or more variables and a starting point, find a\n    fixed point of the function: i.e., where ``func(x0) == x0``.\n\n    Parameters\n    ----------\n    func : function\n        Function to evaluate.\n    x0 : array_like\n        Fixed point of function.\n    args : tuple, optional\n        Extra arguments to `func`.\n    xtol : float, optional\n        Convergence tolerance, defaults to 1e-08.\n    maxiter : int, optional\n        Maximum number of iterations, defaults to 500.\n    method : {"del2", "iteration"}, optional\n        Method of finding the fixed-point, defaults to "del2",\n        which uses Steffensen\'s Method with Aitken\'s ``Del^2``\n        convergence acceleration [1]_. The "iteration" method simply iterates\n        the function until convergence is detected, without attempting to\n        accelerate the convergence.\n\n    References\n    ----------\n    .. [1] Burden, Faires, "Numerical Analysis", 5th edition, pg. 80\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import optimize\n    >>> def func(x, c1, c2):\n    ...    return np.sqrt(c1/(x+c2))\n    >>> c1 = np.array([10,12.])\n    >>> c2 = np.array([3, 5.])\n    >>> optimize.fixed_point(func, [1.2, 1.3], args=(c1,c2))\n    array([ 1.4920333 ,  1.37228132])\n\n    '
    use_accel = {'del2': True, 'iteration': False}[method]
    x0 = _asarray_validated(x0, as_inexact=True)
    return _fixed_point_helper(func, x0, args, xtol, maxiter, use_accel)