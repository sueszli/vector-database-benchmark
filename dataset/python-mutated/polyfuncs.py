"""High-level polynomials manipulation functions. """
from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import PolificationFailed, ComputationFailed, MultivariatePolynomialError, OptionError
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import symmetric_poly, interpolating_poly
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public

@public
def symmetrize(F, *gens, **args):
    if False:
        print('Hello World!')
    '\n    Rewrite a polynomial in terms of elementary symmetric polynomials.\n\n    A symmetric polynomial is a multivariate polynomial that remains invariant\n    under any variable permutation, i.e., if `f = f(x_1, x_2, \\dots, x_n)`,\n    then `f = f(x_{i_1}, x_{i_2}, \\dots, x_{i_n})`, where\n    `(i_1, i_2, \\dots, i_n)` is a permutation of `(1, 2, \\dots, n)` (an\n    element of the group `S_n`).\n\n    Returns a tuple of symmetric polynomials ``(f1, f2, ..., fn)`` such that\n    ``f = f1 + f2 + ... + fn``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polyfuncs import symmetrize\n    >>> from sympy.abc import x, y\n\n    >>> symmetrize(x**2 + y**2)\n    (-2*x*y + (x + y)**2, 0)\n\n    >>> symmetrize(x**2 + y**2, formal=True)\n    (s1**2 - 2*s2, 0, [(s1, x + y), (s2, x*y)])\n\n    >>> symmetrize(x**2 - y**2)\n    (-2*x*y + (x + y)**2, -2*y**2)\n\n    >>> symmetrize(x**2 - y**2, formal=True)\n    (s1**2 - 2*s2, -2*y**2, [(s1, x + y), (s2, x*y)])\n\n    '
    allowed_flags(args, ['formal', 'symbols'])
    iterable = True
    if not hasattr(F, '__iter__'):
        iterable = False
        F = [F]
    (R, F) = sring(F, *gens, **args)
    gens = R.symbols
    opt = build_options(gens, args)
    symbols = opt.symbols
    symbols = [next(symbols) for i in range(len(gens))]
    result = []
    for f in F:
        (p, r, m) = f.symmetrize()
        result.append((p.as_expr(*symbols), r.as_expr(*gens)))
    polys = [(s, g.as_expr()) for (s, (_, g)) in zip(symbols, m)]
    if not opt.formal:
        for (i, (sym, non_sym)) in enumerate(result):
            result[i] = (sym.subs(polys), non_sym)
    if not iterable:
        (result,) = result
    if not opt.formal:
        return result
    elif iterable:
        return (result, polys)
    else:
        return result + (polys,)

@public
def horner(f, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Rewrite a polynomial in Horner form.\n\n    Among other applications, evaluation of a polynomial at a point is optimal\n    when it is applied using the Horner scheme ([1]).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polyfuncs import horner\n    >>> from sympy.abc import x, y, a, b, c, d, e\n\n    >>> horner(9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5)\n    x*(x*(x*(9*x + 8) + 7) + 6) + 5\n\n    >>> horner(a*x**4 + b*x**3 + c*x**2 + d*x + e)\n    e + x*(d + x*(c + x*(a*x + b)))\n\n    >>> f = 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y\n\n    >>> horner(f, wrt=x)\n    x*(x*y*(4*y + 2) + y*(2*y + 1))\n\n    >>> horner(f, wrt=y)\n    y*(x*y*(4*x + 2) + x*(2*x + 1))\n\n    References\n    ==========\n    [1] - https://en.wikipedia.org/wiki/Horner_scheme\n\n    '
    allowed_flags(args, [])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr
    (form, gen) = (S.Zero, F.gen)
    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = form * gen + coeff
    else:
        (F, gens) = (Poly(F, gen), gens[1:])
        for coeff in F.all_coeffs():
            form = form * gen + horner(coeff, *gens, **args)
    return form

@public
def interpolate(data, x):
    if False:
        return 10
    '\n    Construct an interpolating polynomial for the data points\n    evaluated at point x (which can be symbolic or numeric).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polyfuncs import interpolate\n    >>> from sympy.abc import a, b, x\n\n    A list is interpreted as though it were paired with a range starting\n    from 1:\n\n    >>> interpolate([1, 4, 9, 16], x)\n    x**2\n\n    This can be made explicit by giving a list of coordinates:\n\n    >>> interpolate([(1, 1), (2, 4), (3, 9)], x)\n    x**2\n\n    The (x, y) coordinates can also be given as keys and values of a\n    dictionary (and the points need not be equispaced):\n\n    >>> interpolate([(-1, 2), (1, 2), (2, 5)], x)\n    x**2 + 1\n    >>> interpolate({-1: 2, 1: 2, 2: 5}, x)\n    x**2 + 1\n\n    If the interpolation is going to be used only once then the\n    value of interest can be passed instead of passing a symbol:\n\n    >>> interpolate([1, 4, 9], 5)\n    25\n\n    Symbolic coordinates are also supported:\n\n    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]\n    [(1, a), (2, b), (3, -a + 2*b)]\n    '
    n = len(data)
    if isinstance(data, dict):
        if x in data:
            return S(data[x])
        (X, Y) = list(zip(*data.items()))
    elif isinstance(data[0], tuple):
        (X, Y) = list(zip(*data))
        if x in X:
            return S(Y[X.index(x)])
    else:
        if x in range(1, n + 1):
            return S(data[x - 1])
        Y = list(data)
        X = list(range(1, n + 1))
    try:
        return interpolating_poly(n, x, X, Y).expand()
    except ValueError:
        d = Dummy()
        return interpolating_poly(n, d, X, Y).expand().subs(d, x)

@public
def rational_interpolate(data, degnum, X=symbols('x')):
    if False:
        i = 10
        return i + 15
    '\n    Returns a rational interpolation, where the data points are element of\n    any integral domain.\n\n    The first argument  contains the data (as a list of coordinates). The\n    ``degnum`` argument is the degree in the numerator of the rational\n    function. Setting it too high will decrease the maximal degree in the\n    denominator for the same amount of data.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polyfuncs import rational_interpolate\n\n    >>> data = [(1, -210), (2, -35), (3, 105), (4, 231), (5, 350), (6, 465)]\n    >>> rational_interpolate(data, 2)\n    (105*x**2 - 525)/(x + 1)\n\n    Values do not need to be integers:\n\n    >>> from sympy import sympify\n    >>> x = [1, 2, 3, 4, 5, 6]\n    >>> y = sympify("[-1, 0, 2, 22/5, 7, 68/7]")\n    >>> rational_interpolate(zip(x, y), 2)\n    (3*x**2 - 7*x + 2)/(x + 1)\n\n    The symbol for the variable can be changed if needed:\n    >>> from sympy import symbols\n    >>> z = symbols(\'z\')\n    >>> rational_interpolate(data, 2, X=z)\n    (105*z**2 - 525)/(z + 1)\n\n    References\n    ==========\n\n    .. [1] Algorithm is adapted from:\n           http://axiom-wiki.newsynthesis.org/RationalInterpolation\n\n    '
    from sympy.matrices.dense import ones
    (xdata, ydata) = list(zip(*data))
    k = len(xdata) - degnum - 1
    if k < 0:
        raise OptionError('Too few values for the required degree.')
    c = ones(degnum + k + 1, degnum + k + 2)
    for j in range(max(degnum, k)):
        for i in range(degnum + k + 1):
            c[i, j + 1] = c[i, j] * xdata[i]
    for j in range(k + 1):
        for i in range(degnum + k + 1):
            c[i, degnum + k + 1 - j] = -c[i, k - j] * ydata[i]
    r = c.nullspace()[0]
    return sum((r[i] * X ** i for i in range(degnum + 1))) / sum((r[i + degnum + 1] * X ** i for i in range(k + 1)))

@public
def viete(f, roots=None, *gens, **args):
    if False:
        return 10
    "\n    Generate Viete's formulas for ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polyfuncs import viete\n    >>> from sympy import symbols\n\n    >>> x, a, b, c, r1, r2 = symbols('x,a:c,r1:3')\n\n    >>> viete(a*x**2 + b*x + c, [r1, r2], x)\n    [(r1 + r2, -b/a), (r1*r2, c/a)]\n\n    "
    allowed_flags(args, [])
    if isinstance(roots, Basic):
        (gens, roots) = ((roots,) + gens, None)
    try:
        (f, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('viete', 1, exc)
    if f.is_multivariate:
        raise MultivariatePolynomialError('multivariate polynomials are not allowed')
    n = f.degree()
    if n < 1:
        raise ValueError("Cannot derive Viete's formulas for a constant polynomial")
    if roots is None:
        roots = numbered_symbols('r', start=1)
    roots = take(roots, n)
    if n != len(roots):
        raise ValueError('required %s roots, got %s' % (n, len(roots)))
    (lc, coeffs) = (f.LC(), f.all_coeffs())
    (result, sign) = ([], -1)
    for (i, coeff) in enumerate(coeffs[1:]):
        poly = symmetric_poly(i + 1, roots)
        coeff = sign * (coeff / lc)
        result.append((poly, coeff))
        sign = -sign
    return result