"""Minimal polynomials for algebraic numbers."""
from functools import reduce
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import I, Rational, pi, _illegal
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets
from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import NotAlgebraic, GeneratorsError
from sympy.polys.polytools import Poly, PurePoly, invert, factor_list, groebner, resultant, degree, poly_from_expr, parallel_poly_from_expr, lcm
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import numbered_symbols, public, sift

def _choose_factor(factors, x, v, dom=QQ, prec=200, bound=5):
    if False:
        return 10
    '\n    Return a factor having root ``v``\n    It is assumed that one of the factors has root ``v``.\n    '
    if isinstance(factors[0], tuple):
        factors = [f[0] for f in factors]
    if len(factors) == 1:
        return factors[0]
    prec1 = 10
    points = {}
    symbols = dom.symbols if hasattr(dom, 'symbols') else []
    while prec1 <= prec:
        fe = [f.as_expr().xreplace({x: v}) for f in factors]
        if v.is_number:
            fe = [f.n(prec) for f in fe]
        for n in subsets(range(bound), k=len(symbols), repetition=True):
            for (s, i) in zip(symbols, n):
                points[s] = i
            candidates = [(abs(f.subs(points).n(prec1)), i) for (i, f) in enumerate(fe)]
            if any((i in _illegal for (i, _) in candidates)):
                continue
            can = sorted(candidates)
            ((a, ix), (b, _)) = can[:2]
            if b > a * 10 ** 6:
                return factors[ix]
        prec1 *= 2
    raise NotImplementedError('multiple candidates for the minimal polynomial of %s' % v)

def _is_sum_surds(p):
    if False:
        return 10
    args = p.args if p.is_Add else [p]
    for y in args:
        if not ((y ** 2).is_Rational and y.is_extended_real):
            return False
    return True

def _separate_sq(p):
    if False:
        i = 10
        return i + 15
    '\n    helper function for ``_minimal_polynomial_sq``\n\n    It selects a rational ``g`` such that the polynomial ``p``\n    consists of a sum of terms whose surds squared have gcd equal to ``g``\n    and a sum of terms with surds squared prime with ``g``;\n    then it takes the field norm to eliminate ``sqrt(g)``\n\n    See simplify.simplify.split_surds and polytools.sqf_norm.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt\n    >>> from sympy.abc import x\n    >>> from sympy.polys.numberfields.minpoly import _separate_sq\n    >>> p= -x + sqrt(2) + sqrt(3) + sqrt(7)\n    >>> p = _separate_sq(p); p\n    -x**2 + 2*sqrt(3)*x + 2*sqrt(7)*x - 2*sqrt(21) - 8\n    >>> p = _separate_sq(p); p\n    -x**4 + 4*sqrt(7)*x**3 - 32*x**2 + 8*sqrt(7)*x + 20\n    >>> p = _separate_sq(p); p\n    -x**8 + 48*x**6 - 536*x**4 + 1728*x**2 - 400\n\n    '

    def is_sqrt(expr):
        if False:
            i = 10
            return i + 15
        return expr.is_Pow and expr.exp is S.Half
    a = []
    for y in p.args:
        if not y.is_Mul:
            if is_sqrt(y):
                a.append((S.One, y ** 2))
            elif y.is_Atom:
                a.append((y, S.One))
            elif y.is_Pow and y.exp.is_integer:
                a.append((y, S.One))
            else:
                raise NotImplementedError
        else:
            (T, F) = sift(y.args, is_sqrt, binary=True)
            a.append((Mul(*F), Mul(*T) ** 2))
    a.sort(key=lambda z: z[1])
    if a[-1][1] is S.One:
        return p
    surds = [z for (y, z) in a]
    for i in range(len(surds)):
        if surds[i] != 1:
            break
    from sympy.simplify.radsimp import _split_gcd
    (g, b1, b2) = _split_gcd(*surds[i:])
    a1 = []
    a2 = []
    for (y, z) in a:
        if z in b1:
            a1.append(y * z ** S.Half)
        else:
            a2.append(y * z ** S.Half)
    p1 = Add(*a1)
    p2 = Add(*a2)
    p = _mexpand(p1 ** 2) - _mexpand(p2 ** 2)
    return p

def _minimal_polynomial_sq(p, n, x):
    if False:
        return 10
    '\n    Returns the minimal polynomial for the ``nth-root`` of a sum of surds\n    or ``None`` if it fails.\n\n    Parameters\n    ==========\n\n    p : sum of surds\n    n : positive integer\n    x : variable of the returned polynomial\n\n    Examples\n    ========\n\n    >>> from sympy.polys.numberfields.minpoly import _minimal_polynomial_sq\n    >>> from sympy import sqrt\n    >>> from sympy.abc import x\n    >>> q = 1 + sqrt(2) + sqrt(3)\n    >>> _minimal_polynomial_sq(q, 3, x)\n    x**12 - 4*x**9 - 4*x**6 + 16*x**3 - 8\n\n    '
    p = sympify(p)
    n = sympify(n)
    if not n.is_Integer or not n > 0 or (not _is_sum_surds(p)):
        return None
    pn = p ** Rational(1, n)
    p -= x
    while 1:
        p1 = _separate_sq(p)
        if p1 is p:
            p = p1.subs({x: x ** n})
            break
        else:
            p = p1
    if n == 1:
        p1 = Poly(p)
        if p.coeff(x ** p1.degree(x)) < 0:
            p = -p
        p = p.primitive()[1]
        return p
    factors = factor_list(p)[1]
    result = _choose_factor(factors, x, pn)
    return result

def _minpoly_op_algebraic_element(op, ex1, ex2, x, dom, mp1=None, mp2=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    return the minimal polynomial for ``op(ex1, ex2)``\n\n    Parameters\n    ==========\n\n    op : operation ``Add`` or ``Mul``\n    ex1, ex2 : expressions for the algebraic elements\n    x : indeterminate of the polynomials\n    dom: ground domain\n    mp1, mp2 : minimal polynomials for ``ex1`` and ``ex2`` or None\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt, Add, Mul, QQ\n    >>> from sympy.polys.numberfields.minpoly import _minpoly_op_algebraic_element\n    >>> from sympy.abc import x, y\n    >>> p1 = sqrt(sqrt(2) + 1)\n    >>> p2 = sqrt(sqrt(2) - 1)\n    >>> _minpoly_op_algebraic_element(Mul, p1, p2, x, QQ)\n    x - 1\n    >>> q1 = sqrt(y)\n    >>> q2 = 1 / y\n    >>> _minpoly_op_algebraic_element(Add, q1, q2, x, QQ.frac_field(y))\n    x**2*y**2 - 2*x*y - y**3 + 1\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Resultant\n    .. [2] I.M. Isaacs, Proc. Amer. Math. Soc. 25 (1970), 638\n           "Degrees of sums in a separable field extension".\n\n    '
    y = Dummy(str(x))
    if mp1 is None:
        mp1 = _minpoly_compose(ex1, x, dom)
    if mp2 is None:
        mp2 = _minpoly_compose(ex2, y, dom)
    else:
        mp2 = mp2.subs({x: y})
    if op is Add:
        if dom == QQ:
            (R, X) = ring('X', QQ)
            p1 = R(dict_from_expr(mp1)[0])
            p2 = R(dict_from_expr(mp2)[0])
        else:
            ((p1, p2), _) = parallel_poly_from_expr((mp1, x - y), x, y)
            r = p1.compose(p2)
            mp1a = r.as_expr()
    elif op is Mul:
        mp1a = _muly(mp1, x, y)
    else:
        raise NotImplementedError('option not available')
    if op is Mul or dom != QQ:
        r = resultant(mp1a, mp2, gens=[y, x])
    else:
        r = rs_compose_add(p1, p2)
        r = expr_from_dict(r.as_expr_dict(), x)
    deg1 = degree(mp1, x)
    deg2 = degree(mp2, y)
    if op is Mul and deg1 == 1 or deg2 == 1:
        return r
    r = Poly(r, x, domain=dom)
    (_, factors) = r.factor_list()
    res = _choose_factor(factors, x, op(ex1, ex2), dom)
    return res.as_expr()

def _invertx(p, x):
    if False:
        while True:
            i = 10
    '\n    Returns ``expand_mul(x**degree(p, x)*p.subs(x, 1/x))``\n    '
    p1 = poly_from_expr(p, x)[0]
    n = degree(p1)
    a = [c * x ** (n - i) for ((i,), c) in p1.terms()]
    return Add(*a)

def _muly(p, x, y):
    if False:
        return 10
    '\n    Returns ``_mexpand(y**deg*p.subs({x:x / y}))``\n    '
    p1 = poly_from_expr(p, x)[0]
    n = degree(p1)
    a = [c * x ** i * y ** (n - i) for ((i,), c) in p1.terms()]
    return Add(*a)

def _minpoly_pow(ex, pw, x, dom, mp=None):
    if False:
        while True:
            i = 10
    '\n    Returns ``minpoly(ex**pw, x)``\n\n    Parameters\n    ==========\n\n    ex : algebraic element\n    pw : rational number\n    x : indeterminate of the polynomial\n    dom: ground domain\n    mp : minimal polynomial of ``p``\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt, QQ, Rational\n    >>> from sympy.polys.numberfields.minpoly import _minpoly_pow, minpoly\n    >>> from sympy.abc import x, y\n    >>> p = sqrt(1 + sqrt(2))\n    >>> _minpoly_pow(p, 2, x, QQ)\n    x**2 - 2*x - 1\n    >>> minpoly(p**2, x)\n    x**2 - 2*x - 1\n    >>> _minpoly_pow(y, Rational(1, 3), x, QQ.frac_field(y))\n    x**3 - y\n    >>> minpoly(y**Rational(1, 3), x)\n    x**3 - y\n\n    '
    pw = sympify(pw)
    if not mp:
        mp = _minpoly_compose(ex, x, dom)
    if not pw.is_rational:
        raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)
    if pw < 0:
        if mp == x:
            raise ZeroDivisionError('%s is zero' % ex)
        mp = _invertx(mp, x)
        if pw == -1:
            return mp
        pw = -pw
        ex = 1 / ex
    y = Dummy(str(x))
    mp = mp.subs({x: y})
    (n, d) = pw.as_numer_denom()
    res = Poly(resultant(mp, x ** d - y ** n, gens=[y]), x, domain=dom)
    (_, factors) = res.factor_list()
    res = _choose_factor(factors, x, ex ** pw, dom)
    return res.as_expr()

def _minpoly_add(x, dom, *a):
    if False:
        while True:
            i = 10
    '\n    returns ``minpoly(Add(*a), dom, x)``\n    '
    mp = _minpoly_op_algebraic_element(Add, a[0], a[1], x, dom)
    p = a[0] + a[1]
    for px in a[2:]:
        mp = _minpoly_op_algebraic_element(Add, p, px, x, dom, mp1=mp)
        p = p + px
    return mp

def _minpoly_mul(x, dom, *a):
    if False:
        for i in range(10):
            print('nop')
    '\n    returns ``minpoly(Mul(*a), dom, x)``\n    '
    mp = _minpoly_op_algebraic_element(Mul, a[0], a[1], x, dom)
    p = a[0] * a[1]
    for px in a[2:]:
        mp = _minpoly_op_algebraic_element(Mul, p, px, x, dom, mp1=mp)
        p = p * px
    return mp

def _minpoly_sin(ex, x):
    if False:
        return 10
    '\n    Returns the minimal polynomial of ``sin(ex)``\n    see https://mathworld.wolfram.com/TrigonometryAngles.html\n    '
    (c, a) = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            n = c.q
            q = sympify(n)
            if q.is_prime:
                a = dup_chebyshevt(n, ZZ)
                return Add(*[x ** (n - i - 1) * a[i] for i in range(n)])
            if c.p == 1:
                if q == 9:
                    return 64 * x ** 6 - 96 * x ** 4 + 36 * x ** 2 - 3
            if n % 2 == 1:
                a = dup_chebyshevt(n, ZZ)
                a = [x ** (n - i) * a[i] for i in range(n + 1)]
                r = Add(*a)
                (_, factors) = factor_list(r)
                res = _choose_factor(factors, x, ex)
                return res
            expr = ((1 - cos(2 * c * pi)) / 2) ** S.Half
            res = _minpoly_compose(expr, x, QQ)
            return res
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)

def _minpoly_cos(ex, x):
    if False:
        i = 10
        return i + 15
    '\n    Returns the minimal polynomial of ``cos(ex)``\n    see https://mathworld.wolfram.com/TrigonometryAngles.html\n    '
    (c, a) = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            if c.p == 1:
                if c.q == 7:
                    return 8 * x ** 3 - 4 * x ** 2 - 4 * x + 1
                if c.q == 9:
                    return 8 * x ** 3 - 6 * x - 1
            elif c.p == 2:
                q = sympify(c.q)
                if q.is_prime:
                    s = _minpoly_sin(ex, x)
                    return _mexpand(s.subs({x: sqrt((1 - x) / 2)}))
            n = int(c.q)
            a = dup_chebyshevt(n, ZZ)
            a = [x ** (n - i) * a[i] for i in range(n + 1)]
            r = Add(*a) - (-1) ** c.p
            (_, factors) = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)

def _minpoly_tan(ex, x):
    if False:
        return 10
    '\n    Returns the minimal polynomial of ``tan(ex)``\n    see https://github.com/sympy/sympy/issues/21430\n    '
    (c, a) = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            c = c * 2
            n = int(c.q)
            a = n if c.p % 2 == 0 else 1
            terms = []
            for k in range((c.p + 1) % 2, n + 1, 2):
                terms.append(a * x ** k)
                a = -(a * (n - k - 1) * (n - k)) // ((k + 1) * (k + 2))
            r = Add(*terms)
            (_, factors) = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)

def _minpoly_exp(ex, x):
    if False:
        print('Hello World!')
    '\n    Returns the minimal polynomial of ``exp(ex)``\n    '
    (c, a) = ex.args[0].as_coeff_Mul()
    if a == I * pi:
        if c.is_rational:
            q = sympify(c.q)
            if c.p == 1 or c.p == -1:
                if q == 3:
                    return x ** 2 - x + 1
                if q == 4:
                    return x ** 4 + 1
                if q == 6:
                    return x ** 4 - x ** 2 + 1
                if q == 8:
                    return x ** 8 + 1
                if q == 9:
                    return x ** 6 - x ** 3 + 1
                if q == 10:
                    return x ** 8 - x ** 6 + x ** 4 - x ** 2 + 1
                if q.is_prime:
                    s = 0
                    for i in range(q):
                        s += (-x) ** i
                    return s
            factors = [cyclotomic_poly(i, x) for i in divisors(2 * q)]
            mp = _choose_factor(factors, x, ex)
            return mp
        else:
            raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)

def _minpoly_rootof(ex, x):
    if False:
        print('Hello World!')
    '\n    Returns the minimal polynomial of a ``CRootOf`` object.\n    '
    p = ex.expr
    p = p.subs({ex.poly.gens[0]: x})
    (_, factors) = factor_list(p, x)
    result = _choose_factor(factors, x, ex)
    return result

def _minpoly_compose(ex, x, dom):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the minimal polynomial of an algebraic element\n    using operations on minimal polynomials\n\n    Examples\n    ========\n\n    >>> from sympy import minimal_polynomial, sqrt, Rational\n    >>> from sympy.abc import x, y\n    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=True)\n    x**2 - 2*x - 1\n    >>> minimal_polynomial(sqrt(y) + 1/y, x, compose=True)\n    x**2*y**2 - 2*x*y - y**3 + 1\n\n    '
    if ex.is_Rational:
        return ex.q * x - ex.p
    if ex is I:
        (_, factors) = factor_list(x ** 2 + 1, x, domain=dom)
        return x ** 2 + 1 if len(factors) == 1 else x - I
    if ex is S.GoldenRatio:
        (_, factors) = factor_list(x ** 2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x ** 2 - x - 1
        else:
            return _choose_factor(factors, x, (1 + sqrt(5)) / 2, dom=dom)
    if ex is S.TribonacciConstant:
        (_, factors) = factor_list(x ** 3 - x ** 2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x ** 3 - x ** 2 - x - 1
        else:
            fac = (1 + cbrt(19 - 3 * sqrt(33)) + cbrt(19 + 3 * sqrt(33))) / 3
            return _choose_factor(factors, x, fac, dom=dom)
    if hasattr(dom, 'symbols') and ex in dom.symbols:
        return x - ex
    if dom.is_QQ and _is_sum_surds(ex):
        ex -= x
        while 1:
            ex1 = _separate_sq(ex)
            if ex1 is ex:
                return ex
            else:
                ex = ex1
    if ex.is_Add:
        res = _minpoly_add(x, dom, *ex.args)
    elif ex.is_Mul:
        f = Factors(ex).factors
        r = sift(f.items(), lambda itx: itx[0].is_Rational and itx[1].is_Rational)
        if r[True] and dom == QQ:
            ex1 = Mul(*[bx ** ex for (bx, ex) in r[False] + r[None]])
            r1 = dict(r[True])
            dens = [y.q for y in r1.values()]
            lcmdens = reduce(lcm, dens, 1)
            neg1 = S.NegativeOne
            expn1 = r1.pop(neg1, S.Zero)
            nums = [base ** (y.p * lcmdens // y.q) for (base, y) in r1.items()]
            ex2 = Mul(*nums)
            mp1 = minimal_polynomial(ex1, x)
            mp2 = ex2.q * x ** lcmdens - ex2.p * neg1 ** (expn1 * lcmdens)
            ex2 = neg1 ** expn1 * ex2 ** Rational(1, lcmdens)
            res = _minpoly_op_algebraic_element(Mul, ex1, ex2, x, dom, mp1=mp1, mp2=mp2)
        else:
            res = _minpoly_mul(x, dom, *ex.args)
    elif ex.is_Pow:
        res = _minpoly_pow(ex.base, ex.exp, x, dom)
    elif ex.__class__ is sin:
        res = _minpoly_sin(ex, x)
    elif ex.__class__ is cos:
        res = _minpoly_cos(ex, x)
    elif ex.__class__ is tan:
        res = _minpoly_tan(ex, x)
    elif ex.__class__ is exp:
        res = _minpoly_exp(ex, x)
    elif ex.__class__ is CRootOf:
        res = _minpoly_rootof(ex, x)
    else:
        raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)
    return res

@public
def minimal_polynomial(ex, x=None, compose=True, polys=False, domain=None):
    if False:
        i = 10
        return i + 15
    '\n    Computes the minimal polynomial of an algebraic element.\n\n    Parameters\n    ==========\n\n    ex : Expr\n        Element or expression whose minimal polynomial is to be calculated.\n\n    x : Symbol, optional\n        Independent variable of the minimal polynomial\n\n    compose : boolean, optional (default=True)\n        Method to use for computing minimal polynomial. If ``compose=True``\n        (default) then ``_minpoly_compose`` is used, if ``compose=False`` then\n        groebner bases are used.\n\n    polys : boolean, optional (default=False)\n        If ``True`` returns a ``Poly`` object else an ``Expr`` object.\n\n    domain : Domain, optional\n        Ground domain\n\n    Notes\n    =====\n\n    By default ``compose=True``, the minimal polynomial of the subexpressions of ``ex``\n    are computed, then the arithmetic operations on them are performed using the resultant\n    and factorization.\n    If ``compose=False``, a bottom-up algorithm is used with ``groebner``.\n    The default algorithm stalls less frequently.\n\n    If no ground domain is given, it will be generated automatically from the expression.\n\n    Examples\n    ========\n\n    >>> from sympy import minimal_polynomial, sqrt, solve, QQ\n    >>> from sympy.abc import x, y\n\n    >>> minimal_polynomial(sqrt(2), x)\n    x**2 - 2\n    >>> minimal_polynomial(sqrt(2), x, domain=QQ.algebraic_field(sqrt(2)))\n    x - sqrt(2)\n    >>> minimal_polynomial(sqrt(2) + sqrt(3), x)\n    x**4 - 10*x**2 + 1\n    >>> minimal_polynomial(solve(x**3 + x + 3)[0], x)\n    x**3 + x + 3\n    >>> minimal_polynomial(sqrt(y), x)\n    x**2 - y\n\n    '
    ex = sympify(ex)
    if ex.is_number:
        ex = _mexpand(ex, recursive=True)
    for expr in preorder_traversal(ex):
        if expr.is_AlgebraicNumber:
            compose = False
            break
    if x is not None:
        (x, cls) = (sympify(x), Poly)
    else:
        (x, cls) = (Dummy('x'), PurePoly)
    if not domain:
        if ex.free_symbols:
            domain = FractionField(QQ, list(ex.free_symbols))
        else:
            domain = QQ
    if hasattr(domain, 'symbols') and x in domain.symbols:
        raise GeneratorsError('the variable %s is an element of the ground domain %s' % (x, domain))
    if compose:
        result = _minpoly_compose(ex, x, domain)
        result = result.primitive()[1]
        c = result.coeff(x ** degree(result, x))
        if c.is_negative:
            result = expand_mul(-result)
        return cls(result, x, field=True) if polys else result.collect(x)
    if not domain.is_QQ:
        raise NotImplementedError('groebner method only works for QQ')
    result = _minpoly_groebner(ex, x, cls)
    return cls(result, x, field=True) if polys else result.collect(x)

def _minpoly_groebner(ex, x, cls):
    if False:
        i = 10
        return i + 15
    '\n    Computes the minimal polynomial of an algebraic number\n    using Groebner bases\n\n    Examples\n    ========\n\n    >>> from sympy import minimal_polynomial, sqrt, Rational\n    >>> from sympy.abc import x\n    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=False)\n    x**2 - 2*x - 1\n\n    '
    generator = numbered_symbols('a', cls=Dummy)
    (mapping, symbols) = ({}, {})

    def update_mapping(ex, exp, base=None):
        if False:
            return 10
        a = next(generator)
        symbols[ex] = a
        if base is not None:
            mapping[ex] = a ** exp + base
        else:
            mapping[ex] = exp.as_expr(a)
        return a

    def bottom_up_scan(ex):
        if False:
            i = 10
            return i + 15
        '\n        Transform a given algebraic expression *ex* into a multivariate\n        polynomial, by introducing fresh variables with defining equations.\n\n        Explanation\n        ===========\n\n        The critical elements of the algebraic expression *ex* are root\n        extractions, instances of :py:class:`~.AlgebraicNumber`, and negative\n        powers.\n\n        When we encounter a root extraction or an :py:class:`~.AlgebraicNumber`\n        we replace this expression with a fresh variable ``a_i``, and record\n        the defining polynomial for ``a_i``. For example, if ``a_0**(1/3)``\n        occurs, we will replace it with ``a_1``, and record the new defining\n        polynomial ``a_1**3 - a_0``.\n\n        When we encounter a negative power we transform it into a positive\n        power by algebraically inverting the base. This means computing the\n        minimal polynomial in ``x`` for the base, inverting ``x`` modulo this\n        poly (which generates a new polynomial) and then substituting the\n        original base expression for ``x`` in this last polynomial.\n\n        We return the transformed expression, and we record the defining\n        equations for new symbols using the ``update_mapping()`` function.\n\n        '
        if ex.is_Atom:
            if ex is S.ImaginaryUnit:
                if ex not in mapping:
                    return update_mapping(ex, 2, 1)
                else:
                    return symbols[ex]
            elif ex.is_Rational:
                return ex
        elif ex.is_Add:
            return Add(*[bottom_up_scan(g) for g in ex.args])
        elif ex.is_Mul:
            return Mul(*[bottom_up_scan(g) for g in ex.args])
        elif ex.is_Pow:
            if ex.exp.is_Rational:
                if ex.exp < 0:
                    minpoly_base = _minpoly_groebner(ex.base, x, cls)
                    inverse = invert(x, minpoly_base).as_expr()
                    base_inv = inverse.subs(x, ex.base).expand()
                    if ex.exp == -1:
                        return bottom_up_scan(base_inv)
                    else:
                        ex = base_inv ** (-ex.exp)
                if not ex.exp.is_Integer:
                    (base, exp) = ((ex.base ** ex.exp.p).expand(), Rational(1, ex.exp.q))
                else:
                    (base, exp) = (ex.base, ex.exp)
                base = bottom_up_scan(base)
                expr = base ** exp
                if expr not in mapping:
                    if exp.is_Integer:
                        return expr.expand()
                    else:
                        return update_mapping(expr, 1 / exp, -base)
                else:
                    return symbols[expr]
        elif ex.is_AlgebraicNumber:
            if ex not in mapping:
                return update_mapping(ex, ex.minpoly_of_element())
            else:
                return symbols[ex]
        raise NotAlgebraic('%s does not seem to be an algebraic number' % ex)

    def simpler_inverse(ex):
        if False:
            return 10
        '\n        Returns True if it is more likely that the minimal polynomial\n        algorithm works better with the inverse\n        '
        if ex.is_Pow:
            if (1 / ex.exp).is_integer and ex.exp < 0:
                if ex.base.is_Add:
                    return True
        if ex.is_Mul:
            hit = True
            for p in ex.args:
                if p.is_Add:
                    return False
                if p.is_Pow:
                    if p.base.is_Add and p.exp > 0:
                        return False
            if hit:
                return True
        return False
    inverted = False
    ex = expand_multinomial(ex)
    if ex.is_AlgebraicNumber:
        return ex.minpoly_of_element().as_expr(x)
    elif ex.is_Rational:
        result = ex.q * x - ex.p
    else:
        inverted = simpler_inverse(ex)
        if inverted:
            ex = ex ** (-1)
        res = None
        if ex.is_Pow and (1 / ex.exp).is_Integer:
            n = 1 / ex.exp
            res = _minimal_polynomial_sq(ex.base, n, x)
        elif _is_sum_surds(ex):
            res = _minimal_polynomial_sq(ex, S.One, x)
        if res is not None:
            result = res
        if res is None:
            bus = bottom_up_scan(ex)
            F = [x - bus] + list(mapping.values())
            G = groebner(F, list(symbols.values()) + [x], order='lex')
            (_, factors) = factor_list(G[-1])
            result = _choose_factor(factors, x, ex)
    if inverted:
        result = _invertx(result, x)
        if result.coeff(x ** degree(result, x)) < 0:
            result = expand_mul(-result)
    return result

@public
def minpoly(ex, x=None, compose=True, polys=False, domain=None):
    if False:
        for i in range(10):
            print('nop')
    'This is a synonym for :py:func:`~.minimal_polynomial`.'
    return minimal_polynomial(ex, x=x, compose=compose, polys=polys, domain=domain)