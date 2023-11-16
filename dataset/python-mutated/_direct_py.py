from __future__ import annotations
from typing import Any, Callable, Iterable, TYPE_CHECKING
import numpy as np
from scipy.optimize import OptimizeResult
from ._constraints import old_bound_to_new, Bounds
from ._direct import direct as _direct
if TYPE_CHECKING:
    import numpy.typing as npt
__all__ = ['direct']
ERROR_MESSAGES = ('Number of function evaluations done is larger than maxfun={}', 'Number of iterations is larger than maxiter={}', 'u[i] < l[i] for some i', 'maxfun is too large', 'Initialization failed', 'There was an error in the creation of the sample points', 'An error occurred while the function was sampled', 'Maximum number of levels has been reached.', 'Forced stop', 'Invalid arguments', 'Out of memory')
SUCCESS_MESSAGES = ('The best function value found is within a relative error={} of the (known) global optimum f_min', 'The volume of the hyperrectangle containing the lowest function value found is below vol_tol={}', 'The side length measure of the hyperrectangle containing the lowest function value found is below len_tol={}')

def direct(func: Callable[[npt.ArrayLike, tuple[Any]], float], bounds: Iterable | Bounds, *, args: tuple=(), eps: float=0.0001, maxfun: int | None=None, maxiter: int=1000, locally_biased: bool=True, f_min: float=-np.inf, f_min_rtol: float=0.0001, vol_tol: float=1e-16, len_tol: float=1e-06, callback: Callable[[npt.ArrayLike], None] | None=None) -> OptimizeResult:
    if False:
        i = 10
        return i + 15
    "\n    Finds the global minimum of a function using the\n    DIRECT algorithm.\n\n    Parameters\n    ----------\n    func : callable\n        The objective function to be minimized.\n        ``func(x, *args) -> float``\n        where ``x`` is an 1-D array with shape (n,) and ``args`` is a tuple of\n        the fixed parameters needed to completely specify the function.\n    bounds : sequence or `Bounds`\n        Bounds for variables. There are two ways to specify the bounds:\n\n        1. Instance of `Bounds` class.\n        2. ``(min, max)`` pairs for each element in ``x``.\n\n    args : tuple, optional\n        Any additional fixed parameters needed to\n        completely specify the objective function.\n    eps : float, optional\n        Minimal required difference of the objective function values\n        between the current best hyperrectangle and the next potentially\n        optimal hyperrectangle to be divided. In consequence, `eps` serves as a\n        tradeoff between local and global search: the smaller, the more local\n        the search becomes. Default is 1e-4.\n    maxfun : int or None, optional\n        Approximate upper bound on objective function evaluations.\n        If `None`, will be automatically set to ``1000 * N`` where ``N``\n        represents the number of dimensions. Will be capped if necessary to\n        limit DIRECT's RAM usage to app. 1GiB. This will only occur for very\n        high dimensional problems and excessive `max_fun`. Default is `None`.\n    maxiter : int, optional\n        Maximum number of iterations. Default is 1000.\n    locally_biased : bool, optional\n        If `True` (default), use the locally biased variant of the\n        algorithm known as DIRECT_L. If `False`, use the original unbiased\n        DIRECT algorithm. For hard problems with many local minima,\n        `False` is recommended.\n    f_min : float, optional\n        Function value of the global optimum. Set this value only if the\n        global optimum is known. Default is ``-np.inf``, so that this\n        termination criterion is deactivated.\n    f_min_rtol : float, optional\n        Terminate the optimization once the relative error between the\n        current best minimum `f` and the supplied global minimum `f_min`\n        is smaller than `f_min_rtol`. This parameter is only used if\n        `f_min` is also set. Must lie between 0 and 1. Default is 1e-4.\n    vol_tol : float, optional\n        Terminate the optimization once the volume of the hyperrectangle\n        containing the lowest function value is smaller than `vol_tol`\n        of the complete search space. Must lie between 0 and 1.\n        Default is 1e-16.\n    len_tol : float, optional\n        If `locally_biased=True`, terminate the optimization once half of\n        the normalized maximal side length of the hyperrectangle containing\n        the lowest function value is smaller than `len_tol`.\n        If `locally_biased=False`, terminate the optimization once half of\n        the normalized diagonal of the hyperrectangle containing the lowest\n        function value is smaller than `len_tol`. Must lie between 0 and 1.\n        Default is 1e-6.\n    callback : callable, optional\n        A callback function with signature ``callback(xk)`` where ``xk``\n        represents the best function value found so far.\n\n    Returns\n    -------\n    res : OptimizeResult\n        The optimization result represented as a ``OptimizeResult`` object.\n        Important attributes are: ``x`` the solution array, ``success`` a\n        Boolean flag indicating if the optimizer exited successfully and\n        ``message`` which describes the cause of the termination. See\n        `OptimizeResult` for a description of other attributes.\n\n    Notes\n    -----\n    DIviding RECTangles (DIRECT) is a deterministic global\n    optimization algorithm capable of minimizing a black box function with\n    its variables subject to lower and upper bound constraints by sampling\n    potential solutions in the search space [1]_. The algorithm starts by\n    normalising the search space to an n-dimensional unit hypercube.\n    It samples the function at the center of this hypercube and at 2n\n    (n is the number of variables) more points, 2 in each coordinate\n    direction. Using these function values, DIRECT then divides the\n    domain into hyperrectangles, each having exactly one of the sampling\n    points as its center. In each iteration, DIRECT chooses, using the `eps`\n    parameter which defaults to 1e-4, some of the existing hyperrectangles\n    to be further divided. This division process continues until either the\n    maximum number of iterations or maximum function evaluations allowed\n    are exceeded, or the hyperrectangle containing the minimal value found\n    so far becomes small enough. If `f_min` is specified, the optimization\n    will stop once this function value is reached within a relative tolerance.\n    The locally biased variant of DIRECT (originally called DIRECT_L) [2]_ is\n    used by default. It makes the search more locally biased and more\n    efficient for cases with only a few local minima.\n\n    A note about termination criteria: `vol_tol` refers to the volume of the\n    hyperrectangle containing the lowest function value found so far. This\n    volume decreases exponentially with increasing dimensionality of the\n    problem. Therefore `vol_tol` should be decreased to avoid premature\n    termination of the algorithm for higher dimensions. This does not hold\n    for `len_tol`: it refers either to half of the maximal side length\n    (for ``locally_biased=True``) or half of the diagonal of the\n    hyperrectangle (for ``locally_biased=False``).\n\n    This code is based on the DIRECT 2.0.4 Fortran code by Gablonsky et al. at\n    https://ctk.math.ncsu.edu/SOFTWARE/DIRECTv204.tar.gz .\n    This original version was initially converted via f2c and then cleaned up\n    and reorganized by Steven G. Johnson, August 2007, for the NLopt project.\n    The `direct` function wraps the C implementation.\n\n    .. versionadded:: 1.9.0\n\n    References\n    ----------\n    .. [1] Jones, D.R., Perttunen, C.D. & Stuckman, B.E. Lipschitzian\n        optimization without the Lipschitz constant. J Optim Theory Appl\n        79, 157-181 (1993).\n    .. [2] Gablonsky, J., Kelley, C. A Locally-Biased form of the DIRECT\n        Algorithm. Journal of Global Optimization 21, 27-37 (2001).\n\n    Examples\n    --------\n    The following example is a 2-D problem with four local minima: minimizing\n    the Styblinski-Tang function\n    (https://en.wikipedia.org/wiki/Test_functions_for_optimization).\n\n    >>> from scipy.optimize import direct, Bounds\n    >>> def styblinski_tang(pos):\n    ...     x, y = pos\n    ...     return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)\n    >>> bounds = Bounds([-4., -4.], [4., 4.])\n    >>> result = direct(styblinski_tang, bounds)\n    >>> result.x, result.fun, result.nfev\n    array([-2.90321597, -2.90321597]), -78.3323279095383, 2011\n\n    The correct global minimum was found but with a huge number of function\n    evaluations (2011). Loosening the termination tolerances `vol_tol` and\n    `len_tol` can be used to stop DIRECT earlier.\n\n    >>> result = direct(styblinski_tang, bounds, len_tol=1e-3)\n    >>> result.x, result.fun, result.nfev\n    array([-2.9044353, -2.9044353]), -78.33230330754142, 207\n\n    "
    if not isinstance(bounds, Bounds):
        if isinstance(bounds, list) or isinstance(bounds, tuple):
            (lb, ub) = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
        else:
            message = 'bounds must be a sequence or instance of Bounds class'
            raise ValueError(message)
    lb = np.ascontiguousarray(bounds.lb, dtype=np.float64)
    ub = np.ascontiguousarray(bounds.ub, dtype=np.float64)
    if not np.all(lb < ub):
        raise ValueError('Bounds are not consistent min < max')
    if np.any(np.isinf(lb)) or np.any(np.isinf(ub)):
        raise ValueError('Bounds must not be inf.')
    if vol_tol < 0 or vol_tol > 1:
        raise ValueError('vol_tol must be between 0 and 1.')
    if len_tol < 0 or len_tol > 1:
        raise ValueError('len_tol must be between 0 and 1.')
    if f_min_rtol < 0 or f_min_rtol > 1:
        raise ValueError('f_min_rtol must be between 0 and 1.')
    if maxfun is None:
        maxfun = 1000 * lb.shape[0]
    if not isinstance(maxfun, int):
        raise ValueError('maxfun must be of type int.')
    if maxfun < 0:
        raise ValueError('maxfun must be > 0.')
    if not isinstance(maxiter, int):
        raise ValueError('maxiter must be of type int.')
    if maxiter < 0:
        raise ValueError('maxiter must be > 0.')
    if not isinstance(locally_biased, bool):
        raise ValueError('locally_biased must be True or False.')

    def _func_wrap(x, args=None):
        if False:
            for i in range(10):
                print('nop')
        x = np.asarray(x)
        if args is None:
            f = func(x)
        else:
            f = func(x, *args)
        return np.asarray(f).item()
    (x, fun, ret_code, nfev, nit) = _direct(_func_wrap, np.asarray(lb), np.asarray(ub), args, False, eps, maxfun, maxiter, locally_biased, f_min, f_min_rtol, vol_tol, len_tol, callback)
    format_val = (maxfun, maxiter, f_min_rtol, vol_tol, len_tol)
    if ret_code > 2:
        message = SUCCESS_MESSAGES[ret_code - 3].format(format_val[ret_code - 1])
    elif 0 < ret_code <= 2:
        message = ERROR_MESSAGES[ret_code - 1].format(format_val[ret_code - 1])
    elif 0 > ret_code > -100:
        message = ERROR_MESSAGES[abs(ret_code) + 1]
    else:
        message = ERROR_MESSAGES[ret_code + 99]
    return OptimizeResult(x=np.asarray(x), fun=fun, status=ret_code, success=ret_code > 2, message=message, nfev=nfev, nit=nit)