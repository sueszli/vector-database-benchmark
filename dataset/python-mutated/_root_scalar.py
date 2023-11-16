"""
Unified interfaces to root finding algorithms for real or complex
scalar functions.

Functions
---------
- root : find a root of a scalar function.
"""
import numpy as np
from . import _zeros_py as optzeros
from ._numdiff import approx_derivative
__all__ = ['root_scalar']
ROOT_SCALAR_METHODS = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748', 'newton', 'secant', 'halley']

class MemoizeDer:
    """Decorator that caches the value and derivative(s) of function each
    time it is called.

    This is a simplistic memoizer that calls and caches a single value
    of `f(x, *args)`.
    It assumes that `args` does not change between invocations.
    It supports the use case of a root-finder where `args` is fixed,
    `x` changes, and only rarely, if at all, does x assume the same value
    more than once."""

    def __init__(self, fun):
        if False:
            i = 10
            return i + 15
        self.fun = fun
        self.vals = None
        self.x = None
        self.n_calls = 0

    def __call__(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        'Calculate f or use cached value if available'
        if self.vals is None or x != self.x:
            fg = self.fun(x, *args)
            self.x = x
            self.n_calls += 1
            self.vals = fg[:]
        return self.vals[0]

    def fprime(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        "Calculate f' or use a cached value if available"
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[1]

    def fprime2(self, x, *args):
        if False:
            return 10
        "Calculate f'' or use a cached value if available"
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[2]

    def ncalls(self):
        if False:
            for i in range(10):
                print('nop')
        return self.n_calls

def root_scalar(f, args=(), method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find a root of a scalar function.\n\n    Parameters\n    ----------\n    f : callable\n        A function to find a root of.\n    args : tuple, optional\n        Extra arguments passed to the objective function and its derivative(s).\n    method : str, optional\n        Type of solver.  Should be one of\n\n            - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`\n            - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`\n            - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`\n            - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`\n            - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`\n            - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`\n            - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`\n            - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`\n\n    bracket: A sequence of 2 floats, optional\n        An interval bracketing a root.  `f(x, *args)` must have different\n        signs at the two endpoints.\n    x0 : float, optional\n        Initial guess.\n    x1 : float, optional\n        A second guess.\n    fprime : bool or callable, optional\n        If `fprime` is a boolean and is True, `f` is assumed to return the\n        value of the objective function and of the derivative.\n        `fprime` can also be a callable returning the derivative of `f`. In\n        this case, it must accept the same arguments as `f`.\n    fprime2 : bool or callable, optional\n        If `fprime2` is a boolean and is True, `f` is assumed to return the\n        value of the objective function and of the\n        first and second derivatives.\n        `fprime2` can also be a callable returning the second derivative of `f`.\n        In this case, it must accept the same arguments as `f`.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    options : dict, optional\n        A dictionary of solver options. E.g., ``k``, see\n        :obj:`show_options()` for details.\n\n    Returns\n    -------\n    sol : RootResults\n        The solution represented as a ``RootResults`` object.\n        Important attributes are: ``root`` the solution , ``converged`` a\n        boolean flag indicating if the algorithm exited successfully and\n        ``flag`` which describes the cause of the termination. See\n        `RootResults` for a description of other attributes.\n\n    See also\n    --------\n    show_options : Additional options accepted by the solvers\n    root : Find a root of a vector function.\n\n    Notes\n    -----\n    This section describes the available solvers that can be selected by the\n    'method' parameter.\n\n    The default is to use the best method available for the situation\n    presented.\n    If a bracket is provided, it may use one of the bracketing methods.\n    If a derivative and an initial value are specified, it may\n    select one of the derivative-based methods.\n    If no method is judged applicable, it will raise an Exception.\n\n    Arguments for each method are as follows (x=required, o=optional).\n\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    |                    method                     | f | args | bracket | x0 | x1 | fprime | fprime2 | xtol | rtol | maxiter | options |\n    +===============================================+===+======+=========+====+====+========+=========+======+======+=========+=========+\n    | :ref:`bisect <optimize.root_scalar-bisect>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`brentq <optimize.root_scalar-brentq>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`brenth <optimize.root_scalar-brenth>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`ridder <optimize.root_scalar-ridder>`   | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`toms748 <optimize.root_scalar-toms748>` | x |  o   |    x    |    |    |        |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`secant <optimize.root_scalar-secant>`   | x |  o   |         | x  | o  |        |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`newton <optimize.root_scalar-newton>`   | x |  o   |         | x  |    |   o    |         |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n    | :ref:`halley <optimize.root_scalar-halley>`   | x |  o   |         | x  |    |   x    |    x    |  o   |  o   |    o    |   o     |\n    +-----------------------------------------------+---+------+---------+----+----+--------+---------+------+------+---------+---------+\n\n    Examples\n    --------\n\n    Find the root of a simple cubic\n\n    >>> from scipy import optimize\n    >>> def f(x):\n    ...     return (x**3 - 1)  # only one real root at x = 1\n\n    >>> def fprime(x):\n    ...     return 3*x**2\n\n    The `brentq` method takes as input a bracket\n\n    >>> sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')\n    >>> sol.root, sol.iterations, sol.function_calls\n    (1.0, 10, 11)\n\n    The `newton` method takes as input a single point and uses the\n    derivative(s).\n\n    >>> sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')\n    >>> sol.root, sol.iterations, sol.function_calls\n    (1.0, 11, 22)\n\n    The function can provide the value and derivative(s) in a single call.\n\n    >>> def f_p_pp(x):\n    ...     return (x**3 - 1), 3*x**2, 6*x\n\n    >>> sol = optimize.root_scalar(\n    ...     f_p_pp, x0=0.2, fprime=True, method='newton'\n    ... )\n    >>> sol.root, sol.iterations, sol.function_calls\n    (1.0, 11, 11)\n\n    >>> sol = optimize.root_scalar(\n    ...     f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley'\n    ... )\n    >>> sol.root, sol.iterations, sol.function_calls\n    (1.0, 7, 8)\n\n\n    "
    if not isinstance(args, tuple):
        args = (args,)
    if options is None:
        options = {}
    is_memoized = False
    if fprime2 is not None and (not callable(fprime2)):
        if bool(fprime2):
            f = MemoizeDer(f)
            is_memoized = True
            fprime2 = f.fprime2
            fprime = f.fprime
        else:
            fprime2 = None
    if fprime is not None and (not callable(fprime)):
        if bool(fprime):
            f = MemoizeDer(f)
            is_memoized = True
            fprime = f.fprime
        else:
            fprime = None
    kwargs = {}
    for k in ['xtol', 'rtol', 'maxiter']:
        v = locals().get(k)
        if v is not None:
            kwargs[k] = v
    if options:
        kwargs.update(options)
    kwargs.update(full_output=True, disp=False)
    if not method:
        if bracket:
            method = 'brentq'
        elif x0 is not None:
            if fprime:
                if fprime2:
                    method = 'halley'
                else:
                    method = 'newton'
            elif x1 is not None:
                method = 'secant'
            else:
                method = 'newton'
    if not method:
        raise ValueError('Unable to select a solver as neither bracket nor starting point provided.')
    meth = method.lower()
    map2underlying = {'halley': 'newton', 'secant': 'newton'}
    try:
        methodc = getattr(optzeros, map2underlying.get(meth, meth))
    except AttributeError as e:
        raise ValueError('Unknown solver %s' % meth) from e
    if meth in ['bisect', 'ridder', 'brentq', 'brenth', 'toms748']:
        if not isinstance(bracket, (list, tuple, np.ndarray)):
            raise ValueError('Bracket needed for %s' % method)
        (a, b) = bracket[:2]
        try:
            (r, sol) = methodc(f, a, b, args=args, **kwargs)
        except ValueError as e:
            if hasattr(e, '_x'):
                sol = optzeros.RootResults(root=e._x, iterations=np.nan, function_calls=e._function_calls, flag=str(e), method=method)
            else:
                raise
    elif meth in ['secant']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        (r, sol) = methodc(f, x0, args=args, fprime=None, fprime2=None, x1=x1, **kwargs)
    elif meth in ['newton']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if not fprime:

            def fprime(x, *args):
                if False:
                    i = 10
                    return i + 15
                return approx_derivative(f, x, method='2-point', args=args)[0]
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        (r, sol) = methodc(f, x0, args=args, fprime=fprime, fprime2=None, **kwargs)
    elif meth in ['halley']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if not fprime:
            raise ValueError('fprime must be specified for %s' % method)
        if not fprime2:
            raise ValueError('fprime2 must be specified for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        (r, sol) = methodc(f, x0, args=args, fprime=fprime, fprime2=fprime2, **kwargs)
    else:
        raise ValueError('Unknown solver %s' % method)
    if is_memoized:
        n_calls = f.n_calls
        sol.function_calls = n_calls
    return sol

def _root_scalar_brentq_doc():
    if False:
        i = 10
        return i + 15
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    bracket: A sequence of 2 floats, optional\n        An interval bracketing a root.  `f(x, *args)` must have different\n        signs at the two endpoints.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    options: dict, optional\n        Specifies any method-specific options not covered above\n\n    '
    pass

def _root_scalar_brenth_doc():
    if False:
        i = 10
        return i + 15
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    bracket: A sequence of 2 floats, optional\n        An interval bracketing a root.  `f(x, *args)` must have different\n        signs at the two endpoints.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass

def _root_scalar_toms748_doc():
    if False:
        while True:
            i = 10
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    bracket: A sequence of 2 floats, optional\n        An interval bracketing a root.  `f(x, *args)` must have different\n        signs at the two endpoints.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass

def _root_scalar_secant_doc():
    if False:
        return 10
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    x0 : float, required\n        Initial guess.\n    x1 : float, required\n        A second guess.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass

def _root_scalar_newton_doc():
    if False:
        return 10
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function and its derivative.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    x0 : float, required\n        Initial guess.\n    fprime : bool or callable, optional\n        If `fprime` is a boolean and is True, `f` is assumed to return the\n        value of derivative along with the objective function.\n        `fprime` can also be a callable returning the derivative of `f`. In\n        this case, it must accept the same arguments as `f`.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass

def _root_scalar_halley_doc():
    if False:
        while True:
            i = 10
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function and its derivatives.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    x0 : float, required\n        Initial guess.\n    fprime : bool or callable, required\n        If `fprime` is a boolean and is True, `f` is assumed to return the\n        value of derivative along with the objective function.\n        `fprime` can also be a callable returning the derivative of `f`. In\n        this case, it must accept the same arguments as `f`.\n    fprime2 : bool or callable, required\n        If `fprime2` is a boolean and is True, `f` is assumed to return the\n        value of 1st and 2nd derivatives along with the objective function.\n        `fprime2` can also be a callable returning the 2nd derivative of `f`.\n        In this case, it must accept the same arguments as `f`.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass

def _root_scalar_ridder_doc():
    if False:
        while True:
            i = 10
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    bracket: A sequence of 2 floats, optional\n        An interval bracketing a root.  `f(x, *args)` must have different\n        signs at the two endpoints.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass

def _root_scalar_bisect_doc():
    if False:
        i = 10
        return i + 15
    '\n    Options\n    -------\n    args : tuple, optional\n        Extra arguments passed to the objective function.\n    bracket: A sequence of 2 floats, optional\n        An interval bracketing a root.  `f(x, *args)` must have different\n        signs at the two endpoints.\n    xtol : float, optional\n        Tolerance (absolute) for termination.\n    rtol : float, optional\n        Tolerance (relative) for termination.\n    maxiter : int, optional\n        Maximum number of iterations.\n    options: dict, optional\n        Specifies any method-specific options not covered above.\n\n    '
    pass