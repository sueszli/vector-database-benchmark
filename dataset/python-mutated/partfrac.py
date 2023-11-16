"""Algorithms for partial fraction decomposition of rational functions. """
from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public

@xthreaded
@public
def apart(f, x=None, full=False, **options):
    if False:
        i = 10
        return i + 15
    "\n    Compute partial fraction decomposition of a rational function.\n\n    Given a rational function ``f``, computes the partial fraction\n    decomposition of ``f``. Two algorithms are available: One is based on the\n    undertermined coefficients method, the other is Bronstein's full partial\n    fraction decomposition algorithm.\n\n    The undetermined coefficients method (selected by ``full=False``) uses\n    polynomial factorization (and therefore accepts the same options as\n    factor) for the denominator. Per default it works over the rational\n    numbers, therefore decomposition of denominators with non-rational roots\n    (e.g. irrational, complex roots) is not supported by default (see options\n    of factor).\n\n    Bronstein's algorithm can be selected by using ``full=True`` and allows a\n    decomposition of denominators with non-rational roots. A human-readable\n    result can be obtained via ``doit()`` (see examples below).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.partfrac import apart\n    >>> from sympy.abc import x, y\n\n    By default, using the undetermined coefficients method:\n\n    >>> apart(y/(x + 2)/(x + 1), x)\n    -y/(x + 2) + y/(x + 1)\n\n    The undetermined coefficients method does not provide a result when the\n    denominators roots are not rational:\n\n    >>> apart(y/(x**2 + x + 1), x)\n    y/(x**2 + x + 1)\n\n    You can choose Bronstein's algorithm by setting ``full=True``:\n\n    >>> apart(y/(x**2 + x + 1), x, full=True)\n    RootSum(_w**2 + _w + 1, Lambda(_a, (-2*_a*y/3 - y/3)/(-_a + x)))\n\n    Calling ``doit()`` yields a human-readable result:\n\n    >>> apart(y/(x**2 + x + 1), x, full=True).doit()\n    (-y/3 - 2*y*(-1/2 - sqrt(3)*I/2)/3)/(x + 1/2 + sqrt(3)*I/2) + (-y/3 -\n        2*y*(-1/2 + sqrt(3)*I/2)/3)/(x + 1/2 - sqrt(3)*I/2)\n\n\n    See Also\n    ========\n\n    apart_list, assemble_partfrac_list\n    "
    allowed_flags(options, [])
    f = sympify(f)
    if f.is_Atom:
        return f
    else:
        (P, Q) = f.as_numer_denom()
    _options = options.copy()
    options = set_defaults(options, extension=True)
    try:
        ((P, Q), opt) = parallel_poly_from_expr((P, Q), x, **options)
    except PolynomialError as msg:
        if f.is_commutative:
            raise PolynomialError(msg)
        if f.is_Mul:
            (c, nc) = f.args_cnc(split_1=False)
            nc = f.func(*nc)
            if c:
                c = apart(f.func._from_args(c), x=x, full=full, **_options)
                return c * nc
            else:
                return nc
        elif f.is_Add:
            c = []
            nc = []
            for i in f.args:
                if i.is_commutative:
                    c.append(i)
                else:
                    try:
                        nc.append(apart(i, x=x, full=full, **_options))
                    except NotImplementedError:
                        nc.append(i)
            return apart(f.func(*c), x=x, full=full, **_options) + f.func(*nc)
        else:
            reps = []
            pot = preorder_traversal(f)
            next(pot)
            for e in pot:
                try:
                    reps.append((e, apart(e, x=x, full=full, **_options)))
                    pot.skip()
                except NotImplementedError:
                    pass
            return f.xreplace(dict(reps))
    if P.is_multivariate:
        fc = f.cancel()
        if fc != f:
            return apart(fc, x=x, full=full, **_options)
        raise NotImplementedError('multivariate partial fraction decomposition')
    (common, P, Q) = P.cancel(Q)
    (poly, P) = P.div(Q, auto=True)
    (P, Q) = P.rat_clear_denoms(Q)
    if Q.degree() <= 1:
        partial = P / Q
    elif not full:
        partial = apart_undetermined_coeffs(P, Q)
    else:
        partial = apart_full_decomposition(P, Q)
    terms = S.Zero
    for term in Add.make_args(partial):
        if term.has(RootSum):
            terms += term
        else:
            terms += factor(term)
    return common * (poly.as_expr() + terms)

def apart_undetermined_coeffs(P, Q):
    if False:
        for i in range(10):
            print('nop')
    'Partial fractions via method of undetermined coefficients. '
    X = numbered_symbols(cls=Dummy)
    (partial, symbols) = ([], [])
    (_, factors) = Q.factor_list()
    for (f, k) in factors:
        (n, q) = (f.degree(), Q)
        for i in range(1, k + 1):
            (coeffs, q) = (take(X, n), q.quo(f))
            partial.append((coeffs, q, f, i))
            symbols.extend(coeffs)
    dom = Q.get_domain().inject(*symbols)
    F = Poly(0, Q.gen, domain=dom)
    for (i, (coeffs, q, f, k)) in enumerate(partial):
        h = Poly(coeffs, Q.gen, domain=dom)
        partial[i] = (h, f, k)
        q = q.set_domain(dom)
        F += h * q
    (system, result) = ([], S.Zero)
    for ((k,), coeff) in F.terms():
        system.append(coeff - P.nth(k))
    from sympy.solvers import solve
    solution = solve(system, symbols)
    for (h, f, k) in partial:
        h = h.as_expr().subs(solution)
        result += h / f.as_expr() ** k
    return result

def apart_full_decomposition(P, Q):
    if False:
        print('Hello World!')
    "\n    Bronstein's full partial fraction decomposition algorithm.\n\n    Given a univariate rational function ``f``, performing only GCD\n    operations over the algebraic closure of the initial ground domain\n    of definition, compute full partial fraction decomposition with\n    fractions having linear denominators.\n\n    Note that no factorization of the initial denominator of ``f`` is\n    performed. The final decomposition is formed in terms of a sum of\n    :class:`RootSum` instances.\n\n    References\n    ==========\n\n    .. [1] [Bronstein93]_\n\n    "
    return assemble_partfrac_list(apart_list(P / Q, P.gens[0]))

@public
def apart_list(f, x=None, dummies=None, **options):
    if False:
        while True:
            i = 10
    "\n    Compute partial fraction decomposition of a rational function\n    and return the result in structured form.\n\n    Given a rational function ``f`` compute the partial fraction decomposition\n    of ``f``. Only Bronstein's full partial fraction decomposition algorithm\n    is supported by this method. The return value is highly structured and\n    perfectly suited for further algorithmic treatment rather than being\n    human-readable. The function returns a tuple holding three elements:\n\n    * The first item is the common coefficient, free of the variable `x` used\n      for decomposition. (It is an element of the base field `K`.)\n\n    * The second item is the polynomial part of the decomposition. This can be\n      the zero polynomial. (It is an element of `K[x]`.)\n\n    * The third part itself is a list of quadruples. Each quadruple\n      has the following elements in this order:\n\n      - The (not necessarily irreducible) polynomial `D` whose roots `w_i` appear\n        in the linear denominator of a bunch of related fraction terms. (This item\n        can also be a list of explicit roots. However, at the moment ``apart_list``\n        never returns a result this way, but the related ``assemble_partfrac_list``\n        function accepts this format as input.)\n\n      - The numerator of the fraction, written as a function of the root `w`\n\n      - The linear denominator of the fraction *excluding its power exponent*,\n        written as a function of the root `w`.\n\n      - The power to which the denominator has to be raised.\n\n    On can always rebuild a plain expression by using the function ``assemble_partfrac_list``.\n\n    Examples\n    ========\n\n    A first example:\n\n    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list\n    >>> from sympy.abc import x, t\n\n    >>> f = (2*x**3 - 2*x) / (x**2 - 2*x + 1)\n    >>> pfd = apart_list(f)\n    >>> pfd\n    (1,\n    Poly(2*x + 4, x, domain='ZZ'),\n    [(Poly(_w - 1, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1)])\n\n    >>> assemble_partfrac_list(pfd)\n    2*x + 4 + 4/(x - 1)\n\n    Second example:\n\n    >>> f = (-2*x - 2*x**2) / (3*x**2 - 6*x)\n    >>> pfd = apart_list(f)\n    >>> pfd\n    (-1,\n    Poly(2/3, x, domain='QQ'),\n    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 2), Lambda(_a, -_a + x), 1)])\n\n    >>> assemble_partfrac_list(pfd)\n    -2/3 - 2/(x - 2)\n\n    Another example, showing symbolic parameters:\n\n    >>> pfd = apart_list(t/(x**2 + x + t), x)\n    >>> pfd\n    (1,\n    Poly(0, x, domain='ZZ[t]'),\n    [(Poly(_w**2 + _w + t, _w, domain='ZZ[t]'),\n    Lambda(_a, -2*_a*t/(4*t - 1) - t/(4*t - 1)),\n    Lambda(_a, -_a + x),\n    1)])\n\n    >>> assemble_partfrac_list(pfd)\n    RootSum(_w**2 + _w + t, Lambda(_a, (-2*_a*t/(4*t - 1) - t/(4*t - 1))/(-_a + x)))\n\n    This example is taken from Bronstein's original paper:\n\n    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)\n    >>> pfd = apart_list(f)\n    >>> pfd\n    (1,\n    Poly(0, x, domain='ZZ'),\n    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),\n    (Poly(_w**2 - 1, _w, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),\n    (Poly(_w + 1, _w, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])\n\n    >>> assemble_partfrac_list(pfd)\n    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)\n\n    See also\n    ========\n\n    apart, assemble_partfrac_list\n\n    References\n    ==========\n\n    .. [1] [Bronstein93]_\n\n    "
    allowed_flags(options, [])
    f = sympify(f)
    if f.is_Atom:
        return f
    else:
        (P, Q) = f.as_numer_denom()
    options = set_defaults(options, extension=True)
    ((P, Q), opt) = parallel_poly_from_expr((P, Q), x, **options)
    if P.is_multivariate:
        raise NotImplementedError('multivariate partial fraction decomposition')
    (common, P, Q) = P.cancel(Q)
    (poly, P) = P.div(Q, auto=True)
    (P, Q) = P.rat_clear_denoms(Q)
    polypart = poly
    if dummies is None:

        def dummies(name):
            if False:
                i = 10
                return i + 15
            d = Dummy(name)
            while True:
                yield d
        dummies = dummies('w')
    rationalpart = apart_list_full_decomposition(P, Q, dummies)
    return (common, polypart, rationalpart)

def apart_list_full_decomposition(P, Q, dummygen):
    if False:
        i = 10
        return i + 15
    "\n    Bronstein's full partial fraction decomposition algorithm.\n\n    Given a univariate rational function ``f``, performing only GCD\n    operations over the algebraic closure of the initial ground domain\n    of definition, compute full partial fraction decomposition with\n    fractions having linear denominators.\n\n    Note that no factorization of the initial denominator of ``f`` is\n    performed. The final decomposition is formed in terms of a sum of\n    :class:`RootSum` instances.\n\n    References\n    ==========\n\n    .. [1] [Bronstein93]_\n\n    "
    (f, x, U) = (P / Q, P.gen, [])
    u = Function('u')(x)
    a = Dummy('a')
    partial = []
    for (d, n) in Q.sqf_list_include(all=True):
        b = d.as_expr()
        U += [u.diff(x, n - 1)]
        h = cancel(f * b ** n) / u ** n
        (H, subs) = ([h], [])
        for j in range(1, n):
            H += [H[-1].diff(x) / j]
        for j in range(1, n + 1):
            subs += [(U[j - 1], b.diff(x, j) / j)]
        for j in range(0, n):
            (P, Q) = cancel(H[j]).as_numer_denom()
            for i in range(0, j + 1):
                P = P.subs(*subs[j - i])
            Q = Q.subs(*subs[0])
            P = Poly(P, x)
            Q = Poly(Q, x)
            G = P.gcd(d)
            D = d.quo(G)
            (B, g) = Q.half_gcdex(D)
            b = (P * B.quo(g)).rem(D)
            Dw = D.subs(x, next(dummygen))
            numer = Lambda(a, b.as_expr().subs(x, a))
            denom = Lambda(a, x - a)
            exponent = n - j
            partial.append((Dw, numer, denom, exponent))
    return partial

@public
def assemble_partfrac_list(partial_list):
    if False:
        i = 10
        return i + 15
    'Reassemble a full partial fraction decomposition\n    from a structured result obtained by the function ``apart_list``.\n\n    Examples\n    ========\n\n    This example is taken from Bronstein\'s original paper:\n\n    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list\n    >>> from sympy.abc import x\n\n    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)\n    >>> pfd = apart_list(f)\n    >>> pfd\n    (1,\n    Poly(0, x, domain=\'ZZ\'),\n    [(Poly(_w - 2, _w, domain=\'ZZ\'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),\n    (Poly(_w**2 - 1, _w, domain=\'ZZ\'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),\n    (Poly(_w + 1, _w, domain=\'ZZ\'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])\n\n    >>> assemble_partfrac_list(pfd)\n    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)\n\n    If we happen to know some roots we can provide them easily inside the structure:\n\n    >>> pfd = apart_list(2/(x**2-2))\n    >>> pfd\n    (1,\n    Poly(0, x, domain=\'ZZ\'),\n    [(Poly(_w**2 - 2, _w, domain=\'ZZ\'),\n    Lambda(_a, _a/2),\n    Lambda(_a, -_a + x),\n    1)])\n\n    >>> pfda = assemble_partfrac_list(pfd)\n    >>> pfda\n    RootSum(_w**2 - 2, Lambda(_a, _a/(-_a + x)))/2\n\n    >>> pfda.doit()\n    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))\n\n    >>> from sympy import Dummy, Poly, Lambda, sqrt\n    >>> a = Dummy("a")\n    >>> pfd = (1, Poly(0, x, domain=\'ZZ\'), [([sqrt(2),-sqrt(2)], Lambda(a, a/2), Lambda(a, -a + x), 1)])\n\n    >>> assemble_partfrac_list(pfd)\n    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))\n\n    See Also\n    ========\n\n    apart, apart_list\n    '
    common = partial_list[0]
    polypart = partial_list[1]
    pfd = polypart.as_expr()
    for (r, nf, df, ex) in partial_list[2]:
        if isinstance(r, Poly):
            (an, nu) = (nf.variables, nf.expr)
            (ad, de) = (df.variables, df.expr)
            de = de.subs(ad[0], an[0])
            func = Lambda(tuple(an), nu / de ** ex)
            pfd += RootSum(r, func, auto=False, quadratic=False)
        else:
            for root in r:
                pfd += nf(root) / df(root) ** ex
    return common * pfd