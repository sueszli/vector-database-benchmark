"""
Finite difference weights
=========================

This module implements an algorithm for efficient generation of finite
difference weights for ordinary differentials of functions for
derivatives from 0 (interpolation) up to arbitrary order.

The core algorithm is provided in the finite difference weight generating
function (``finite_diff_weights``), and two convenience functions are provided
for:

- estimating a derivative (or interpolate) directly from a series of points
    is also provided (``apply_finite_diff``).
- differentiating by using finite difference approximations
    (``differentiate_finite``).

"""
from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable

def finite_diff_weights(order, x_list, x0=S.One):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the finite difference weights for an arbitrarily spaced\n    one-dimensional grid (``x_list``) for derivatives at ``x0`` of order\n    0, 1, ..., up to ``order`` using a recursive formula. Order of accuracy\n    is at least ``len(x_list) - order``, if ``x_list`` is defined correctly.\n\n    Parameters\n    ==========\n\n    order: int\n        Up to what derivative order weights should be calculated.\n        0 corresponds to interpolation.\n    x_list: sequence\n        Sequence of (unique) values for the independent variable.\n        It is useful (but not necessary) to order ``x_list`` from\n        nearest to furthest from ``x0``; see examples below.\n    x0: Number or Symbol\n        Root or value of the independent variable for which the finite\n        difference weights should be generated. Default is ``S.One``.\n\n    Returns\n    =======\n\n    list\n        A list of sublists, each corresponding to coefficients for\n        increasing derivative order, and each containing lists of\n        coefficients for increasing subsets of x_list.\n\n    Examples\n    ========\n\n    >>> from sympy import finite_diff_weights, S\n    >>> res = finite_diff_weights(1, [-S(1)/2, S(1)/2, S(3)/2, S(5)/2], 0)\n    >>> res\n    [[[1, 0, 0, 0],\n      [1/2, 1/2, 0, 0],\n      [3/8, 3/4, -1/8, 0],\n      [5/16, 15/16, -5/16, 1/16]],\n     [[0, 0, 0, 0],\n      [-1, 1, 0, 0],\n      [-1, 1, 0, 0],\n      [-23/24, 7/8, 1/8, -1/24]]]\n    >>> res[0][-1]  # FD weights for 0th derivative, using full x_list\n    [5/16, 15/16, -5/16, 1/16]\n    >>> res[1][-1]  # FD weights for 1st derivative\n    [-23/24, 7/8, 1/8, -1/24]\n    >>> res[1][-2]  # FD weights for 1st derivative, using x_list[:-1]\n    [-1, 1, 0, 0]\n    >>> res[1][-1][0]  # FD weight for 1st deriv. for x_list[0]\n    -23/24\n    >>> res[1][-1][1]  # FD weight for 1st deriv. for x_list[1], etc.\n    7/8\n\n    Each sublist contains the most accurate formula at the end.\n    Note, that in the above example ``res[1][1]`` is the same as ``res[1][2]``.\n    Since res[1][2] has an order of accuracy of\n    ``len(x_list[:3]) - order = 3 - 1 = 2``, the same is true for ``res[1][1]``!\n\n    >>> res = finite_diff_weights(1, [S(0), S(1), -S(1), S(2), -S(2)], 0)[1]\n    >>> res\n    [[0, 0, 0, 0, 0],\n     [-1, 1, 0, 0, 0],\n     [0, 1/2, -1/2, 0, 0],\n     [-1/2, 1, -1/3, -1/6, 0],\n     [0, 2/3, -2/3, -1/12, 1/12]]\n    >>> res[0]  # no approximation possible, using x_list[0] only\n    [0, 0, 0, 0, 0]\n    >>> res[1]  # classic forward step approximation\n    [-1, 1, 0, 0, 0]\n    >>> res[2]  # classic centered approximation\n    [0, 1/2, -1/2, 0, 0]\n    >>> res[3:]  # higher order approximations\n    [[-1/2, 1, -1/3, -1/6, 0], [0, 2/3, -2/3, -1/12, 1/12]]\n\n    Let us compare this to a differently defined ``x_list``. Pay attention to\n    ``foo[i][k]`` corresponding to the gridpoint defined by ``x_list[k]``.\n\n    >>> foo = finite_diff_weights(1, [-S(2), -S(1), S(0), S(1), S(2)], 0)[1]\n    >>> foo\n    [[0, 0, 0, 0, 0],\n     [-1, 1, 0, 0, 0],\n     [1/2, -2, 3/2, 0, 0],\n     [1/6, -1, 1/2, 1/3, 0],\n     [1/12, -2/3, 0, 2/3, -1/12]]\n    >>> foo[1]  # not the same and of lower accuracy as res[1]!\n    [-1, 1, 0, 0, 0]\n    >>> foo[2]  # classic double backward step approximation\n    [1/2, -2, 3/2, 0, 0]\n    >>> foo[4]  # the same as res[4]\n    [1/12, -2/3, 0, 2/3, -1/12]\n\n    Note that, unless you plan on using approximations based on subsets of\n    ``x_list``, the order of gridpoints does not matter.\n\n    The capability to generate weights at arbitrary points can be\n    used e.g. to minimize Runge\'s phenomenon by using Chebyshev nodes:\n\n    >>> from sympy import cos, symbols, pi, simplify\n    >>> N, (h, x) = 4, symbols(\'h x\')\n    >>> x_list = [x+h*cos(i*pi/(N)) for i in range(N,-1,-1)] # chebyshev nodes\n    >>> print(x_list)\n    [-h + x, -sqrt(2)*h/2 + x, x, sqrt(2)*h/2 + x, h + x]\n    >>> mycoeffs = finite_diff_weights(1, x_list, 0)[1][4]\n    >>> [simplify(c) for c in  mycoeffs] #doctest: +NORMALIZE_WHITESPACE\n    [(h**3/2 + h**2*x - 3*h*x**2 - 4*x**3)/h**4,\n    (-sqrt(2)*h**3 - 4*h**2*x + 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,\n    (6*h**2*x - 8*x**3)/h**4,\n    (sqrt(2)*h**3 - 4*h**2*x - 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,\n    (-h**3/2 + h**2*x + 3*h*x**2 - 4*x**3)/h**4]\n\n    Notes\n    =====\n\n    If weights for a finite difference approximation of 3rd order\n    derivative is wanted, weights for 0th, 1st and 2nd order are\n    calculated "for free", so are formulae using subsets of ``x_list``.\n    This is something one can take advantage of to save computational cost.\n    Be aware that one should define ``x_list`` from nearest to furthest from\n    ``x0``. If not, subsets of ``x_list`` will yield poorer approximations,\n    which might not grand an order of accuracy of ``len(x_list) - order``.\n\n    See also\n    ========\n\n    sympy.calculus.finite_diff.apply_finite_diff\n\n    References\n    ==========\n\n    .. [1] Generation of Finite Difference Formulas on Arbitrarily Spaced\n            Grids, Bengt Fornberg; Mathematics of computation; 51; 184;\n            (1988); 699-706; doi:10.1090/S0025-5718-1988-0935077-0\n\n    '
    order = S(order)
    if not order.is_number:
        raise ValueError('Cannot handle symbolic order.')
    if order < 0:
        raise ValueError('Negative derivative order illegal.')
    if int(order) != order:
        raise ValueError('Non-integer order illegal')
    M = order
    N = len(x_list) - 1
    delta = [[[0 for nu in range(N + 1)] for n in range(N + 1)] for m in range(M + 1)]
    delta[0][0][0] = S.One
    c1 = S.One
    for n in range(1, N + 1):
        c2 = S.One
        for nu in range(n):
            c3 = x_list[n] - x_list[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n][n - 1][nu] = 0
            for m in range(min(n, M) + 1):
                delta[m][n][nu] = (x_list[n] - x0) * delta[m][n - 1][nu] - m * delta[m - 1][n - 1][nu]
                delta[m][n][nu] /= c3
        for m in range(min(n, M) + 1):
            delta[m][n][n] = c1 / c2 * (m * delta[m - 1][n - 1][n - 1] - (x_list[n - 1] - x0) * delta[m][n - 1][n - 1])
        c1 = c2
    return delta

def apply_finite_diff(order, x_list, y_list, x0=S.Zero):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculates the finite difference approximation of\n    the derivative of requested order at ``x0`` from points\n    provided in ``x_list`` and ``y_list``.\n\n    Parameters\n    ==========\n\n    order: int\n        order of derivative to approximate. 0 corresponds to interpolation.\n    x_list: sequence\n        Sequence of (unique) values for the independent variable.\n    y_list: sequence\n        The function value at corresponding values for the independent\n        variable in x_list.\n    x0: Number or Symbol\n        At what value of the independent variable the derivative should be\n        evaluated. Defaults to 0.\n\n    Returns\n    =======\n\n    sympy.core.add.Add or sympy.core.numbers.Number\n        The finite difference expression approximating the requested\n        derivative order at ``x0``.\n\n    Examples\n    ========\n\n    >>> from sympy import apply_finite_diff\n    >>> cube = lambda arg: (1.0*arg)**3\n    >>> xlist = range(-3,3+1)\n    >>> apply_finite_diff(2, xlist, map(cube, xlist), 2) - 12 # doctest: +SKIP\n    -3.55271367880050e-15\n\n    we see that the example above only contain rounding errors.\n    apply_finite_diff can also be used on more abstract objects:\n\n    >>> from sympy import IndexedBase, Idx\n    >>> x, y = map(IndexedBase, 'xy')\n    >>> i = Idx('i')\n    >>> x_list, y_list = zip(*[(x[i+j], y[i+j]) for j in range(-1,2)])\n    >>> apply_finite_diff(1, x_list, y_list, x[i])\n    ((x[i + 1] - x[i])/(-x[i - 1] + x[i]) - 1)*y[i]/(x[i + 1] - x[i]) -\n    (x[i + 1] - x[i])*y[i - 1]/((x[i + 1] - x[i - 1])*(-x[i - 1] + x[i])) +\n    (-x[i - 1] + x[i])*y[i + 1]/((x[i + 1] - x[i - 1])*(x[i + 1] - x[i]))\n\n    Notes\n    =====\n\n    Order = 0 corresponds to interpolation.\n    Only supply so many points you think makes sense\n    to around x0 when extracting the derivative (the function\n    need to be well behaved within that region). Also beware\n    of Runge's phenomenon.\n\n    See also\n    ========\n\n    sympy.calculus.finite_diff.finite_diff_weights\n\n    References\n    ==========\n\n    Fortran 90 implementation with Python interface for numerics: finitediff_\n\n    .. _finitediff: https://github.com/bjodah/finitediff\n\n    "
    N = len(x_list) - 1
    if len(x_list) != len(y_list):
        raise ValueError('x_list and y_list not equal in length.')
    delta = finite_diff_weights(order, x_list, x0)
    derivative = 0
    for nu in range(len(x_list)):
        derivative += delta[order][N][nu] * y_list[nu]
    return derivative

def _as_finite_diff(derivative, points=1, x0=None, wrt=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns an approximation of a derivative of a function in\n    the form of a finite difference formula. The expression is a\n    weighted sum of the function at a number of discrete values of\n    (one of) the independent variable(s).\n\n    Parameters\n    ==========\n\n    derivative: a Derivative instance\n\n    points: sequence or coefficient, optional\n        If sequence: discrete values (length >= order+1) of the\n        independent variable used for generating the finite\n        difference weights.\n        If it is a coefficient, it will be used as the step-size\n        for generating an equidistant sequence of length order+1\n        centered around ``x0``. default: 1 (step-size 1)\n\n    x0: number or Symbol, optional\n        the value of the independent variable (``wrt``) at which the\n        derivative is to be approximated. Default: same as ``wrt``.\n\n    wrt: Symbol, optional\n        "with respect to" the variable for which the (partial)\n        derivative is to be approximated for. If not provided it\n        is required that the Derivative is ordinary. Default: ``None``.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Function, exp, sqrt, Symbol\n    >>> from sympy.calculus.finite_diff import _as_finite_diff\n    >>> x, h = symbols(\'x h\')\n    >>> f = Function(\'f\')\n    >>> _as_finite_diff(f(x).diff(x))\n    -f(x - 1/2) + f(x + 1/2)\n\n    The default step size and number of points are 1 and ``order + 1``\n    respectively. We can change the step size by passing a symbol\n    as a parameter:\n\n    >>> _as_finite_diff(f(x).diff(x), h)\n    -f(-h/2 + x)/h + f(h/2 + x)/h\n\n    We can also specify the discretized values to be used in a sequence:\n\n    >>> _as_finite_diff(f(x).diff(x), [x, x+h, x+2*h])\n    -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)\n\n    The algorithm is not restricted to use equidistant spacing, nor\n    do we need to make the approximation around ``x0``, but we can get\n    an expression estimating the derivative at an offset:\n\n    >>> e, sq2 = exp(1), sqrt(2)\n    >>> xl = [x-h, x+h, x+e*h]\n    >>> _as_finite_diff(f(x).diff(x, 1), xl, x+h*sq2)\n    2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/((-h + E*h)*(h + E*h)) +\n    (-(-sqrt(2)*h + h)/(2*h) - (-sqrt(2)*h + E*h)/(2*h))*f(-h + x)/(h + E*h) +\n    (-(h + sqrt(2)*h)/(2*h) + (-sqrt(2)*h + E*h)/(2*h))*f(h + x)/(-h + E*h)\n\n    Partial derivatives are also supported:\n\n    >>> y = Symbol(\'y\')\n    >>> d2fdxdy=f(x,y).diff(x,y)\n    >>> _as_finite_diff(d2fdxdy, wrt=x)\n    -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)\n\n    See also\n    ========\n\n    sympy.calculus.finite_diff.apply_finite_diff\n    sympy.calculus.finite_diff.finite_diff_weights\n\n    '
    if derivative.is_Derivative:
        pass
    elif derivative.is_Atom:
        return derivative
    else:
        return derivative.fromiter([_as_finite_diff(ar, points, x0, wrt) for ar in derivative.args], **derivative.assumptions0)
    if wrt is None:
        old = None
        for v in derivative.variables:
            if old is v:
                continue
            derivative = _as_finite_diff(derivative, points, x0, v)
            old = v
        return derivative
    order = derivative.variables.count(wrt)
    if x0 is None:
        x0 = wrt
    if not iterable(points):
        if getattr(points, 'is_Function', False) and wrt in points.args:
            points = points.subs(wrt, x0)
        if order % 2 == 0:
            points = [x0 + points * i for i in range(-order // 2, order // 2 + 1)]
        else:
            points = [x0 + points * S(i) / 2 for i in range(-order, order + 1, 2)]
    others = [wrt, 0]
    for v in set(derivative.variables):
        if v == wrt:
            continue
        others += [v, derivative.variables.count(v)]
    if len(points) < order + 1:
        raise ValueError('Too few points for order %d' % order)
    return apply_finite_diff(order, points, [Derivative(derivative.expr.subs({wrt: x}), *others) for x in points], x0)

def differentiate_finite(expr, *symbols, points=1, x0=None, wrt=None, evaluate=False):
    if False:
        for i in range(10):
            print('nop')
    " Differentiate expr and replace Derivatives with finite differences.\n\n    Parameters\n    ==========\n\n    expr : expression\n    \\*symbols : differentiate with respect to symbols\n    points: sequence, coefficient or undefined function, optional\n        see ``Derivative.as_finite_difference``\n    x0: number or Symbol, optional\n        see ``Derivative.as_finite_difference``\n    wrt: Symbol, optional\n        see ``Derivative.as_finite_difference``\n\n    Examples\n    ========\n\n    >>> from sympy import sin, Function, differentiate_finite\n    >>> from sympy.abc import x, y, h\n    >>> f, g = Function('f'), Function('g')\n    >>> differentiate_finite(f(x)*g(x), x, points=[x-h, x+h])\n    -f(-h + x)*g(-h + x)/(2*h) + f(h + x)*g(h + x)/(2*h)\n\n    ``differentiate_finite`` works on any expression, including the expressions\n    with embedded derivatives:\n\n    >>> differentiate_finite(f(x) + sin(x), x, 2)\n    -2*f(x) + f(x - 1) + f(x + 1) - 2*sin(x) + sin(x - 1) + sin(x + 1)\n    >>> differentiate_finite(f(x, y), x, y)\n    f(x - 1/2, y - 1/2) - f(x - 1/2, y + 1/2) - f(x + 1/2, y - 1/2) + f(x + 1/2, y + 1/2)\n    >>> differentiate_finite(f(x)*g(x).diff(x), x)\n    (-g(x) + g(x + 1))*f(x + 1/2) - (g(x) - g(x - 1))*f(x - 1/2)\n\n    To make finite difference with non-constant discretization step use\n    undefined functions:\n\n    >>> dx = Function('dx')\n    >>> differentiate_finite(f(x)*g(x).diff(x), points=dx(x))\n    -(-g(x - dx(x)/2 - dx(x - dx(x)/2)/2)/dx(x - dx(x)/2) +\n    g(x - dx(x)/2 + dx(x - dx(x)/2)/2)/dx(x - dx(x)/2))*f(x - dx(x)/2)/dx(x) +\n    (-g(x + dx(x)/2 - dx(x + dx(x)/2)/2)/dx(x + dx(x)/2) +\n    g(x + dx(x)/2 + dx(x + dx(x)/2)/2)/dx(x + dx(x)/2))*f(x + dx(x)/2)/dx(x)\n\n    "
    if any((term.is_Derivative for term in list(preorder_traversal(expr)))):
        evaluate = False
    Dexpr = expr.diff(*symbols, evaluate=evaluate)
    if evaluate:
        sympy_deprecation_warning('\n        The evaluate flag to differentiate_finite() is deprecated.\n\n        evaluate=True expands the intermediate derivatives before computing\n        differences, but this usually not what you want, as it does not\n        satisfy the product rule.\n        ', deprecated_since_version='1.5', active_deprecations_target='deprecated-differentiate_finite-evaluate')
        return Dexpr.replace(lambda arg: arg.is_Derivative, lambda arg: arg.as_finite_difference(points=points, x0=x0, wrt=wrt))
    else:
        DFexpr = Dexpr.as_finite_difference(points=points, x0=x0, wrt=wrt)
        return DFexpr.replace(lambda arg: isinstance(arg, Subs), lambda arg: arg.expr.as_finite_difference(points=points, x0=arg.point[0], wrt=arg.variables[0]))