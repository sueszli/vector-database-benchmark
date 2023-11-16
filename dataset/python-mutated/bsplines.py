from sympy.core import S, sympify
from sympy.core.symbol import Dummy, symbols
from sympy.functions import Piecewise, piecewise_fold
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from functools import lru_cache

def _ivl(cond, x):
    if False:
        print('Hello World!')
    "return the interval corresponding to the condition\n\n    Conditions in spline's Piecewise give the range over\n    which an expression is valid like (lo <= x) & (x <= hi).\n    This function returns (lo, hi).\n    "
    if isinstance(cond, And) and len(cond.args) == 2:
        (a, b) = cond.args
        if a.lts == x:
            (a, b) = (b, a)
        return (a.lts, b.gts)
    raise TypeError('unexpected cond type: %s' % cond)

def _add_splines(c, b1, d, b2, x):
    if False:
        print('Hello World!')
    'Construct c*b1 + d*b2.'
    if S.Zero in (b1, c):
        rv = piecewise_fold(d * b2)
    elif S.Zero in (b2, d):
        rv = piecewise_fold(c * b1)
    else:
        new_args = []
        p1 = piecewise_fold(c * b1)
        p2 = piecewise_fold(d * b2)
        p2args = list(p2.args[:-1])
        for arg in p1.args[:-1]:
            expr = arg.expr
            cond = arg.cond
            lower = _ivl(cond, x)[0]
            for (i, arg2) in enumerate(p2args):
                expr2 = arg2.expr
                cond2 = arg2.cond
                (lower_2, upper_2) = _ivl(cond2, x)
                if cond2 == cond:
                    expr += expr2
                    del p2args[i]
                    break
                elif lower_2 < lower and upper_2 <= lower:
                    new_args.append(arg2)
                    del p2args[i]
                    break
            new_args.append((expr, cond))
        new_args.extend(p2args)
        new_args.append((0, True))
        rv = Piecewise(*new_args, evaluate=False)
    return rv.expand()

@lru_cache(maxsize=128)
def bspline_basis(d, knots, n, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    The $n$-th B-spline at $x$ of degree $d$ with knots.\n\n    Explanation\n    ===========\n\n    B-Splines are piecewise polynomials of degree $d$. They are defined on a\n    set of knots, which is a sequence of integers or floats.\n\n    Examples\n    ========\n\n    The 0th degree splines have a value of 1 on a single interval:\n\n        >>> from sympy import bspline_basis\n        >>> from sympy.abc import x\n        >>> d = 0\n        >>> knots = tuple(range(5))\n        >>> bspline_basis(d, knots, 0, x)\n        Piecewise((1, (x >= 0) & (x <= 1)), (0, True))\n\n    For a given ``(d, knots)`` there are ``len(knots)-d-1`` B-splines\n    defined, that are indexed by ``n`` (starting at 0).\n\n    Here is an example of a cubic B-spline:\n\n        >>> bspline_basis(3, tuple(range(5)), 0, x)\n        Piecewise((x**3/6, (x >= 0) & (x <= 1)),\n                  (-x**3/2 + 2*x**2 - 2*x + 2/3,\n                  (x >= 1) & (x <= 2)),\n                  (x**3/2 - 4*x**2 + 10*x - 22/3,\n                  (x >= 2) & (x <= 3)),\n                  (-x**3/6 + 2*x**2 - 8*x + 32/3,\n                  (x >= 3) & (x <= 4)),\n                  (0, True))\n\n    By repeating knot points, you can introduce discontinuities in the\n    B-splines and their derivatives:\n\n        >>> d = 1\n        >>> knots = (0, 0, 2, 3, 4)\n        >>> bspline_basis(d, knots, 0, x)\n        Piecewise((1 - x/2, (x >= 0) & (x <= 2)), (0, True))\n\n    It is quite time consuming to construct and evaluate B-splines. If\n    you need to evaluate a B-spline many times, it is best to lambdify them\n    first:\n\n        >>> from sympy import lambdify\n        >>> d = 3\n        >>> knots = tuple(range(10))\n        >>> b0 = bspline_basis(d, knots, 0, x)\n        >>> f = lambdify(x, b0)\n        >>> y = f(0.5)\n\n    Parameters\n    ==========\n\n    d : integer\n        degree of bspline\n\n    knots : list of integer values\n        list of knots points of bspline\n\n    n : integer\n        $n$-th B-spline\n\n    x : symbol\n\n    See Also\n    ========\n\n    bspline_basis_set\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/B-spline\n\n    '
    xvar = x
    x = Dummy()
    knots = tuple((sympify(k) for k in knots))
    d = int(d)
    n = int(n)
    n_knots = len(knots)
    n_intervals = n_knots - 1
    if n + d + 1 > n_intervals:
        raise ValueError('n + d + 1 must not exceed len(knots) - 1')
    if d == 0:
        result = Piecewise((S.One, Interval(knots[n], knots[n + 1]).contains(x)), (0, True))
    elif d > 0:
        denom = knots[n + d + 1] - knots[n + 1]
        if denom != S.Zero:
            B = (knots[n + d + 1] - x) / denom
            b2 = bspline_basis(d - 1, knots, n + 1, x)
        else:
            b2 = B = S.Zero
        denom = knots[n + d] - knots[n]
        if denom != S.Zero:
            A = (x - knots[n]) / denom
            b1 = bspline_basis(d - 1, knots, n, x)
        else:
            b1 = A = S.Zero
        result = _add_splines(A, b1, B, b2, x)
    else:
        raise ValueError('degree must be non-negative: %r' % n)
    return result.xreplace({x: xvar})

def bspline_basis_set(d, knots, x):
    if False:
        while True:
            i = 10
    '\n    Return the ``len(knots)-d-1`` B-splines at *x* of degree *d*\n    with *knots*.\n\n    Explanation\n    ===========\n\n    This function returns a list of piecewise polynomials that are the\n    ``len(knots)-d-1`` B-splines of degree *d* for the given knots.\n    This function calls ``bspline_basis(d, knots, n, x)`` for different\n    values of *n*.\n\n    Examples\n    ========\n\n    >>> from sympy import bspline_basis_set\n    >>> from sympy.abc import x\n    >>> d = 2\n    >>> knots = range(5)\n    >>> splines = bspline_basis_set(d, knots, x)\n    >>> splines\n    [Piecewise((x**2/2, (x >= 0) & (x <= 1)),\n               (-x**2 + 3*x - 3/2, (x >= 1) & (x <= 2)),\n               (x**2/2 - 3*x + 9/2, (x >= 2) & (x <= 3)),\n               (0, True)),\n    Piecewise((x**2/2 - x + 1/2, (x >= 1) & (x <= 2)),\n              (-x**2 + 5*x - 11/2, (x >= 2) & (x <= 3)),\n              (x**2/2 - 4*x + 8, (x >= 3) & (x <= 4)),\n              (0, True))]\n\n    Parameters\n    ==========\n\n    d : integer\n        degree of bspline\n\n    knots : list of integers\n        list of knots points of bspline\n\n    x : symbol\n\n    See Also\n    ========\n\n    bspline_basis\n\n    '
    n_splines = len(knots) - d - 1
    return [bspline_basis(d, tuple(knots), i, x) for i in range(n_splines)]

def interpolating_spline(d, x, X, Y):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return spline of degree *d*, passing through the given *X*\n    and *Y* values.\n\n    Explanation\n    ===========\n\n    This function returns a piecewise function such that each part is\n    a polynomial of degree not greater than *d*. The value of *d*\n    must be 1 or greater and the values of *X* must be strictly\n    increasing.\n\n    Examples\n    ========\n\n    >>> from sympy import interpolating_spline\n    >>> from sympy.abc import x\n    >>> interpolating_spline(1, x, [1, 2, 4, 7], [3, 6, 5, 7])\n    Piecewise((3*x, (x >= 1) & (x <= 2)),\n            (7 - x/2, (x >= 2) & (x <= 4)),\n            (2*x/3 + 7/3, (x >= 4) & (x <= 7)))\n    >>> interpolating_spline(3, x, [-2, 0, 1, 3, 4], [4, 2, 1, 1, 3])\n    Piecewise((7*x**3/117 + 7*x**2/117 - 131*x/117 + 2, (x >= -2) & (x <= 1)),\n            (10*x**3/117 - 2*x**2/117 - 122*x/117 + 77/39, (x >= 1) & (x <= 4)))\n\n    Parameters\n    ==========\n\n    d : integer\n        Degree of Bspline strictly greater than equal to one\n\n    x : symbol\n\n    X : list of strictly increasing real values\n        list of X coordinates through which the spline passes\n\n    Y : list of real values\n        list of corresponding Y coordinates through which the spline passes\n\n    See Also\n    ========\n\n    bspline_basis_set, interpolating_poly\n\n    '
    from sympy.solvers.solveset import linsolve
    from sympy.matrices.dense import Matrix
    d = sympify(d)
    if not (d.is_Integer and d.is_positive):
        raise ValueError('Spline degree must be a positive integer, not %s.' % d)
    if len(X) != len(Y):
        raise ValueError('Number of X and Y coordinates must be the same.')
    if len(X) < d + 1:
        raise ValueError('Degree must be less than the number of control points.')
    if not all((a < b for (a, b) in zip(X, X[1:]))):
        raise ValueError('The x-coordinates must be strictly increasing.')
    X = [sympify(i) for i in X]
    if d.is_odd:
        j = (d + 1) // 2
        interior_knots = X[j:-j]
    else:
        j = d // 2
        interior_knots = [(a + b) / 2 for (a, b) in zip(X[j:-j - 1], X[j + 1:-j])]
    knots = [X[0]] * (d + 1) + list(interior_knots) + [X[-1]] * (d + 1)
    basis = bspline_basis_set(d, knots, x)
    A = [[b.subs(x, v) for b in basis] for v in X]
    coeff = linsolve((Matrix(A), Matrix(Y)), symbols('c0:{}'.format(len(X)), cls=Dummy))
    coeff = list(coeff)[0]
    intervals = {c for b in basis for (e, c) in b.args if c != True}
    ival = [_ivl(c, x) for c in intervals]
    com = zip(ival, intervals)
    com = sorted(com, key=lambda x: x[0])
    intervals = [y for (x, y) in com]
    basis_dicts = [{c: e for (e, c) in b.args} for b in basis]
    spline = []
    for i in intervals:
        piece = sum([c * d.get(i, S.Zero) for (c, d) in zip(coeff, basis_dicts)], S.Zero)
        spline.append((piece, i))
    return Piecewise(*spline)