"""Algorithms for computing symbolic roots of polynomials. """
import math
from functools import reduce
from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import Rational, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import PolynomialError, GeneratorsNeeded, DomainError, UnsolvableFactorError
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent
z = Symbol('z')

def roots_linear(f):
    if False:
        return 10
    'Returns a list of roots of a linear polynomial.'
    r = -f.nth(0) / f.nth(1)
    dom = f.get_domain()
    if not dom.is_Numerical:
        if dom.is_Composite:
            r = factor(r)
        else:
            from sympy.simplify.simplify import simplify
            r = simplify(r)
    return [r]

def roots_quadratic(f):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of roots of a quadratic polynomial. If the domain is ZZ\n    then the roots will be sorted with negatives coming before positives.\n    The ordering will be the same for any numerical coefficients as long as\n    the assumptions tested are correct, otherwise the ordering will not be\n    sorted (but will be canonical).\n    '
    (a, b, c) = f.all_coeffs()
    dom = f.get_domain()

    def _sqrt(d):
        if False:
            i = 10
            return i + 15
        co = []
        other = []
        for di in Mul.make_args(d):
            if di.is_Pow and di.exp.is_Integer and (di.exp % 2 == 0):
                co.append(Pow(di.base, di.exp // 2))
            else:
                other.append(di)
        if co:
            d = Mul(*other)
            co = Mul(*co)
            return co * sqrt(d)
        return sqrt(d)

    def _simplify(expr):
        if False:
            i = 10
            return i + 15
        if dom.is_Composite:
            return factor(expr)
        else:
            from sympy.simplify.simplify import simplify
            return simplify(expr)
    if c is S.Zero:
        (r0, r1) = (S.Zero, -b / a)
        if not dom.is_Numerical:
            r1 = _simplify(r1)
        elif r1.is_negative:
            (r0, r1) = (r1, r0)
    elif b is S.Zero:
        r = -c / a
        if not dom.is_Numerical:
            r = _simplify(r)
        R = _sqrt(r)
        r0 = -R
        r1 = R
    else:
        d = b ** 2 - 4 * a * c
        A = 2 * a
        B = -b / A
        if not dom.is_Numerical:
            d = _simplify(d)
            B = _simplify(B)
        D = factor_terms(_sqrt(d) / A)
        r0 = B - D
        r1 = B + D
        if a.is_negative:
            (r0, r1) = (r1, r0)
        elif not dom.is_Numerical:
            (r0, r1) = [expand_2arg(i) for i in (r0, r1)]
    return [r0, r1]

def roots_cubic(f, trig=False):
    if False:
        return 10
    'Returns a list of roots of a cubic polynomial.\n\n    References\n    ==========\n    [1] https://en.wikipedia.org/wiki/Cubic_function, General formula for roots,\n    (accessed November 17, 2014).\n    '
    if trig:
        (a, b, c, d) = f.all_coeffs()
        p = (3 * a * c - b ** 2) / (3 * a ** 2)
        q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
        D = 18 * a * b * c * d - 4 * b ** 3 * d + b ** 2 * c ** 2 - 4 * a * c ** 3 - 27 * a ** 2 * d ** 2
        if (D > 0) == True:
            rv = []
            for k in range(3):
                rv.append(2 * sqrt(-p / 3) * cos(acos(q / p * sqrt(-3 / p) * Rational(3, 2)) / 3 - k * pi * Rational(2, 3)))
            return [i - b / 3 / a for i in rv]
    (_, a, b, c) = f.monic().all_coeffs()
    if c is S.Zero:
        (x1, x2) = roots([1, a, b], multiple=True)
        return [x1, S.Zero, x2]
    p = b - a ** 2 / 3
    q = c - a * b / 3 + 2 * a ** 3 / 27
    pon3 = p / 3
    aon3 = a / 3
    u1 = None
    if p is S.Zero:
        if q is S.Zero:
            return [-aon3] * 3
        u1 = -root(q, 3) if q.is_positive else root(-q, 3)
    elif q is S.Zero:
        (y1, y2) = roots([1, 0, p], multiple=True)
        return [tmp - aon3 for tmp in [y1, S.Zero, y2]]
    elif q.is_real and q.is_negative:
        u1 = -root(-q / 2 + sqrt(q ** 2 / 4 + pon3 ** 3), 3)
    coeff = I * sqrt(3) / 2
    if u1 is None:
        u1 = S.One
        u2 = Rational(-1, 2) + coeff
        u3 = Rational(-1, 2) - coeff
        (b, c, d) = (a, b, c)
        D0 = b ** 2 - 3 * c
        D1 = 2 * b ** 3 - 9 * b * c + 27 * d
        C = root((D1 + sqrt(D1 ** 2 - 4 * D0 ** 3)) / 2, 3)
        return [-(b + uk * C + D0 / C / uk) / 3 for uk in [u1, u2, u3]]
    u2 = u1 * (Rational(-1, 2) + coeff)
    u3 = u1 * (Rational(-1, 2) - coeff)
    if p is S.Zero:
        return [u1 - aon3, u2 - aon3, u3 - aon3]
    soln = [-u1 + pon3 / u1 - aon3, -u2 + pon3 / u2 - aon3, -u3 + pon3 / u3 - aon3]
    return soln

def _roots_quartic_euler(p, q, r, a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Descartes-Euler solution of the quartic equation\n\n    Parameters\n    ==========\n\n    p, q, r: coefficients of ``x**4 + p*x**2 + q*x + r``\n    a: shift of the roots\n\n    Notes\n    =====\n\n    This is a helper function for ``roots_quartic``.\n\n    Look for solutions of the form ::\n\n      ``x1 = sqrt(R) - sqrt(A + B*sqrt(R))``\n      ``x2 = -sqrt(R) - sqrt(A - B*sqrt(R))``\n      ``x3 = -sqrt(R) + sqrt(A - B*sqrt(R))``\n      ``x4 = sqrt(R) + sqrt(A + B*sqrt(R))``\n\n    To satisfy the quartic equation one must have\n    ``p = -2*(R + A); q = -4*B*R; r = (R - A)**2 - B**2*R``\n    so that ``R`` must satisfy the Descartes-Euler resolvent equation\n    ``64*R**3 + 32*p*R**2 + (4*p**2 - 16*r)*R - q**2 = 0``\n\n    If the resolvent does not have a rational solution, return None;\n    in that case it is likely that the Ferrari method gives a simpler\n    solution.\n\n    Examples\n    ========\n\n    >>> from sympy import S\n    >>> from sympy.polys.polyroots import _roots_quartic_euler\n    >>> p, q, r = -S(64)/5, -S(512)/125, -S(1024)/3125\n    >>> _roots_quartic_euler(p, q, r, S(0))[0]\n    -sqrt(32*sqrt(5)/125 + 16/5) + 4*sqrt(5)/5\n    '
    x = Dummy('x')
    eq = 64 * x ** 3 + 32 * p * x ** 2 + (4 * p ** 2 - 16 * r) * x - q ** 2
    xsols = list(roots(Poly(eq, x), cubics=False).keys())
    xsols = [sol for sol in xsols if sol.is_rational and sol.is_nonzero]
    if not xsols:
        return None
    R = max(xsols)
    c1 = sqrt(R)
    B = -q * c1 / (4 * R)
    A = -R - p / 2
    c2 = sqrt(A + B)
    c3 = sqrt(A - B)
    return [c1 - c2 - a, -c1 - c3 - a, -c1 + c3 - a, c1 + c2 - a]

def roots_quartic(f):
    if False:
        print('Hello World!')
    "\n    Returns a list of roots of a quartic polynomial.\n\n    There are many references for solving quartic expressions available [1-5].\n    This reviewer has found that many of them require one to select from among\n    2 or more possible sets of solutions and that some solutions work when one\n    is searching for real roots but do not work when searching for complex roots\n    (though this is not always stated clearly). The following routine has been\n    tested and found to be correct for 0, 2 or 4 complex roots.\n\n    The quasisymmetric case solution [6] looks for quartics that have the form\n    `x**4 + A*x**3 + B*x**2 + C*x + D = 0` where `(C/A)**2 = D`.\n\n    Although no general solution that is always applicable for all\n    coefficients is known to this reviewer, certain conditions are tested\n    to determine the simplest 4 expressions that can be returned:\n\n      1) `f = c + a*(a**2/8 - b/2) == 0`\n      2) `g = d - a*(a*(3*a**2/256 - b/16) + c/4) = 0`\n      3) if `f != 0` and `g != 0` and `p = -d + a*c/4 - b**2/12` then\n        a) `p == 0`\n        b) `p != 0`\n\n    Examples\n    ========\n\n        >>> from sympy import Poly\n        >>> from sympy.polys.polyroots import roots_quartic\n\n        >>> r = roots_quartic(Poly('x**4-6*x**3+17*x**2-26*x+20'))\n\n        >>> # 4 complex roots: 1+-I*sqrt(3), 2+-I\n        >>> sorted(str(tmp.evalf(n=2)) for tmp in r)\n        ['1.0 + 1.7*I', '1.0 - 1.7*I', '2.0 + 1.0*I', '2.0 - 1.0*I']\n\n    References\n    ==========\n\n    1. http://mathforum.org/dr.math/faq/faq.cubic.equations.html\n    2. https://en.wikipedia.org/wiki/Quartic_function#Summary_of_Ferrari.27s_method\n    3. https://planetmath.org/encyclopedia/GaloisTheoreticDerivationOfTheQuarticFormula.html\n    4. https://people.bath.ac.uk/masjhd/JHD-CA.pdf\n    5. http://www.albmath.org/files/Math_5713.pdf\n    6. https://web.archive.org/web/20171002081448/http://www.statemaster.com/encyclopedia/Quartic-equation\n    7. https://eqworld.ipmnet.ru/en/solutions/ae/ae0108.pdf\n    "
    (_, a, b, c, d) = f.monic().all_coeffs()
    if not d:
        return [S.Zero] + roots([1, a, b, c], multiple=True)
    elif (c / a) ** 2 == d:
        (x, m) = (f.gen, c / a)
        g = Poly(x ** 2 + a * x + b - 2 * m, x)
        (z1, z2) = roots_quadratic(g)
        h1 = Poly(x ** 2 - z1 * x + m, x)
        h2 = Poly(x ** 2 - z2 * x + m, x)
        r1 = roots_quadratic(h1)
        r2 = roots_quadratic(h2)
        return r1 + r2
    else:
        a2 = a ** 2
        e = b - 3 * a2 / 8
        f = _mexpand(c + a * (a2 / 8 - b / 2))
        aon4 = a / 4
        g = _mexpand(d - aon4 * (a * (3 * a2 / 64 - b / 4) + c))
        if f.is_zero:
            (y1, y2) = [sqrt(tmp) for tmp in roots([1, e, g], multiple=True)]
            return [tmp - aon4 for tmp in [-y1, -y2, y1, y2]]
        if g.is_zero:
            y = [S.Zero] + roots([1, 0, e, f], multiple=True)
            return [tmp - aon4 for tmp in y]
        else:
            sols = _roots_quartic_euler(e, f, g, aon4)
            if sols:
                return sols
            p = -e ** 2 / 12 - g
            q = -e ** 3 / 108 + e * g / 3 - f ** 2 / 8
            TH = Rational(1, 3)

            def _ans(y):
                if False:
                    print('Hello World!')
                w = sqrt(e + 2 * y)
                arg1 = 3 * e + 2 * y
                arg2 = 2 * f / w
                ans = []
                for s in [-1, 1]:
                    root = sqrt(-(arg1 + s * arg2))
                    for t in [-1, 1]:
                        ans.append((s * w - t * root) / 2 - aon4)
                return ans
            p = _mexpand(p)
            y1 = e * Rational(-5, 6) - q ** TH
            if p.is_zero:
                return _ans(y1)
            root = sqrt(q ** 2 / 4 + p ** 3 / 27)
            r = -q / 2 + root
            u = r ** TH
            y2 = e * Rational(-5, 6) + u - p / u / 3
            if fuzzy_not(p.is_zero):
                return _ans(y2)
            return [Piecewise((a1, Eq(p, 0)), (a2, True)) for (a1, a2) in zip(_ans(y1), _ans(y2))]

def roots_binomial(f):
    if False:
        print('Hello World!')
    'Returns a list of roots of a binomial polynomial. If the domain is ZZ\n    then the roots will be sorted with negatives coming before positives.\n    The ordering will be the same for any numerical coefficients as long as\n    the assumptions tested are correct, otherwise the ordering will not be\n    sorted (but will be canonical).\n    '
    n = f.degree()
    (a, b) = (f.nth(n), f.nth(0))
    base = -cancel(b / a)
    alpha = root(base, n)
    if alpha.is_number:
        alpha = alpha.expand(complex=True)
    neg = base.is_negative
    even = n % 2 == 0
    if neg:
        if even == True and (base + 1).is_positive:
            big = True
        else:
            big = False
    ks = []
    imax = n // 2
    if even:
        ks.append(imax)
        imax -= 1
    if not neg:
        ks.append(0)
    for i in range(imax, 0, -1):
        if neg:
            ks.extend([i, -i])
        else:
            ks.extend([-i, i])
    if neg:
        ks.append(0)
        if big:
            for i in range(0, len(ks), 2):
                pair = ks[i:i + 2]
                pair = list(reversed(pair))
    (roots, d) = ([], 2 * I * pi / n)
    for k in ks:
        zeta = exp(k * d).expand(complex=True)
        roots.append((alpha * zeta).expand(power_base=False))
    return roots

def _inv_totient_estimate(m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find ``(L, U)`` such that ``L <= phi^-1(m) <= U``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polyroots import _inv_totient_estimate\n\n    >>> _inv_totient_estimate(192)\n    (192, 840)\n    >>> _inv_totient_estimate(400)\n    (400, 1750)\n\n    '
    primes = [d + 1 for d in divisors(m) if isprime(d + 1)]
    (a, b) = (1, 1)
    for p in primes:
        a *= p
        b *= p - 1
    L = m
    U = int(math.ceil(m * (float(a) / b)))
    P = p = 2
    primes = []
    while P <= U:
        p = nextprime(p)
        primes.append(p)
        P *= p
    P //= p
    b = 1
    for p in primes[:-1]:
        b *= p - 1
    U = int(math.ceil(m * (float(P) / b)))
    return (L, U)

def roots_cyclotomic(f, factor=False):
    if False:
        i = 10
        return i + 15
    'Compute roots of cyclotomic polynomials. '
    (L, U) = _inv_totient_estimate(f.degree())
    for n in range(L, U + 1):
        g = cyclotomic_poly(n, f.gen, polys=True)
        if f.expr == g.expr:
            break
    else:
        raise RuntimeError('failed to find index of a cyclotomic polynomial')
    roots = []
    if not factor:
        h = n // 2
        ks = [i for i in range(1, n + 1) if igcd(i, n) == 1]
        ks.sort(key=lambda x: (x, -1) if x <= h else (abs(x - n), 1))
        d = 2 * I * pi / n
        for k in reversed(ks):
            roots.append(exp(k * d).expand(complex=True))
    else:
        g = Poly(f, extension=root(-1, n))
        for (h, _) in ordered(g.factor_list()[1]):
            roots.append(-h.TC())
    return roots

def roots_quintic(f):
    if False:
        print('Hello World!')
    '\n    Calculate exact roots of a solvable irreducible quintic with rational coefficients.\n    Return an empty list if the quintic is reducible or not solvable.\n    '
    result = []
    (coeff_5, coeff_4, p_, q_, r_, s_) = f.all_coeffs()
    if not all((coeff.is_Rational for coeff in (coeff_5, coeff_4, p_, q_, r_, s_))):
        return result
    if coeff_5 != 1:
        f = Poly(f / coeff_5)
        (_, coeff_4, p_, q_, r_, s_) = f.all_coeffs()
    if coeff_4:
        p = p_ - 2 * coeff_4 * coeff_4 / 5
        q = q_ - 3 * coeff_4 * p_ / 5 + 4 * coeff_4 ** 3 / 25
        r = r_ - 2 * coeff_4 * q_ / 5 + 3 * coeff_4 ** 2 * p_ / 25 - 3 * coeff_4 ** 4 / 125
        s = s_ - coeff_4 * r_ / 5 + coeff_4 ** 2 * q_ / 25 - coeff_4 ** 3 * p_ / 125 + 4 * coeff_4 ** 5 / 3125
        x = f.gen
        f = Poly(x ** 5 + p * x ** 3 + q * x ** 2 + r * x + s)
    else:
        (p, q, r, s) = (p_, q_, r_, s_)
    quintic = PolyQuintic(f)
    if not f.is_irreducible:
        return result
    f20 = quintic.f20
    if f20.is_irreducible:
        return result
    for _factor in f20.factor_list()[1]:
        if _factor[0].is_linear:
            theta = _factor[0].root(0)
            break
    d = discriminant(f)
    delta = sqrt(d)
    (zeta1, zeta2, zeta3, zeta4) = quintic.zeta
    T = quintic.T(theta, d)
    tol = S(1e-10)
    alpha = T[1] + T[2] * delta
    alpha_bar = T[1] - T[2] * delta
    beta = T[3] + T[4] * delta
    beta_bar = T[3] - T[4] * delta
    disc = alpha ** 2 - 4 * beta
    disc_bar = alpha_bar ** 2 - 4 * beta_bar
    l0 = quintic.l0(theta)
    Stwo = S(2)
    l1 = _quintic_simplify((-alpha + sqrt(disc)) / Stwo)
    l4 = _quintic_simplify((-alpha - sqrt(disc)) / Stwo)
    l2 = _quintic_simplify((-alpha_bar + sqrt(disc_bar)) / Stwo)
    l3 = _quintic_simplify((-alpha_bar - sqrt(disc_bar)) / Stwo)
    order = quintic.order(theta, d)
    test = order * delta.n() - (l1.n() - l4.n()) * (l2.n() - l3.n())
    if not comp(test, 0, tol):
        (l2, l3) = (l3, l2)
    R1 = l0 + l1 * zeta1 + l2 * zeta2 + l3 * zeta3 + l4 * zeta4
    R2 = l0 + l3 * zeta1 + l1 * zeta2 + l4 * zeta3 + l2 * zeta4
    R3 = l0 + l2 * zeta1 + l4 * zeta2 + l1 * zeta3 + l3 * zeta4
    R4 = l0 + l4 * zeta1 + l3 * zeta2 + l2 * zeta3 + l1 * zeta4
    Res = [None, [None] * 5, [None] * 5, [None] * 5, [None] * 5]
    Res_n = [None, [None] * 5, [None] * 5, [None] * 5, [None] * 5]
    R1 = _quintic_simplify(R1)
    R2 = _quintic_simplify(R2)
    R3 = _quintic_simplify(R3)
    R4 = _quintic_simplify(R4)
    x0 = z ** (S(1) / 5)
    x1 = sqrt(2)
    x2 = sqrt(5)
    x3 = sqrt(5 - x2)
    x4 = I * x2
    x5 = x4 + I
    x6 = I * x0 / 4
    x7 = x1 * sqrt(x2 + 5)
    sol = [x0, -x6 * (x1 * x3 - x5), x6 * (x1 * x3 + x5), -x6 * (x4 + x7 - I), x6 * (-x4 + x7 + I)]
    R1 = R1.as_real_imag()
    R2 = R2.as_real_imag()
    R3 = R3.as_real_imag()
    R4 = R4.as_real_imag()
    for (i, s) in enumerate(sol):
        Res[1][i] = _quintic_simplify(s.xreplace({z: R1[0] + I * R1[1]}))
        Res[2][i] = _quintic_simplify(s.xreplace({z: R2[0] + I * R2[1]}))
        Res[3][i] = _quintic_simplify(s.xreplace({z: R3[0] + I * R3[1]}))
        Res[4][i] = _quintic_simplify(s.xreplace({z: R4[0] + I * R4[1]}))
    for i in range(1, 5):
        for j in range(5):
            Res_n[i][j] = Res[i][j].n()
            Res[i][j] = _quintic_simplify(Res[i][j])
    r1 = Res[1][0]
    r1_n = Res_n[1][0]
    for i in range(5):
        if comp(im(r1_n * Res_n[4][i]), 0, tol):
            r4 = Res[4][i]
            break
    (u, v) = quintic.uv(theta, d)
    testplus = (u + v * delta * sqrt(5)).n()
    testminus = (u - v * delta * sqrt(5)).n()
    r4_n = r4.n()
    r2 = r3 = None
    for i in range(5):
        r2temp_n = Res_n[2][i]
        for j in range(5):
            r3temp_n = Res_n[3][j]
            if comp((r1_n * r2temp_n ** 2 + r4_n * r3temp_n ** 2 - testplus).n(), 0, tol) and comp((r3temp_n * r1_n ** 2 + r2temp_n * r4_n ** 2 - testminus).n(), 0, tol):
                r2 = Res[2][i]
                r3 = Res[3][j]
                break
        if r2 is not None:
            break
    else:
        return []
    x1 = (r1 + r2 + r3 + r4) / 5
    x2 = (r1 * zeta4 + r2 * zeta3 + r3 * zeta2 + r4 * zeta1) / 5
    x3 = (r1 * zeta3 + r2 * zeta1 + r3 * zeta4 + r4 * zeta2) / 5
    x4 = (r1 * zeta2 + r2 * zeta4 + r3 * zeta1 + r4 * zeta3) / 5
    x5 = (r1 * zeta1 + r2 * zeta2 + r3 * zeta3 + r4 * zeta4) / 5
    result = [x1, x2, x3, x4, x5]
    saw = set()
    for r in result:
        r = r.n(2)
        if r in saw:
            return []
        saw.add(r)
    if coeff_4:
        result = [x - coeff_4 / 5 for x in result]
    return result

def _quintic_simplify(expr):
    if False:
        print('Hello World!')
    from sympy.simplify.simplify import powsimp
    expr = powsimp(expr)
    expr = cancel(expr)
    return together(expr)

def _integer_basis(poly):
    if False:
        print('Hello World!')
    "Compute coefficient basis for a polynomial over integers.\n\n    Returns the integer ``div`` such that substituting ``x = div*y``\n    ``p(x) = m*q(y)`` where the coefficients of ``q`` are smaller\n    than those of ``p``.\n\n    For example ``x**5 + 512*x + 1024 = 0``\n    with ``div = 4`` becomes ``y**5 + 2*y + 1 = 0``\n\n    Returns the integer ``div`` or ``None`` if there is no possible scaling.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import Poly\n    >>> from sympy.abc import x\n    >>> from sympy.polys.polyroots import _integer_basis\n    >>> p = Poly(x**5 + 512*x + 1024, x, domain='ZZ')\n    >>> _integer_basis(p)\n    4\n    "
    (monoms, coeffs) = list(zip(*poly.terms()))
    (monoms,) = list(zip(*monoms))
    coeffs = list(map(abs, coeffs))
    if coeffs[0] < coeffs[-1]:
        coeffs = list(reversed(coeffs))
        n = monoms[0]
        monoms = [n - i for i in reversed(monoms)]
    else:
        return None
    monoms = monoms[:-1]
    coeffs = coeffs[:-1]
    if len(monoms) == 1:
        r = Pow(coeffs[0], S.One / monoms[0])
        if r.is_Integer:
            return int(r)
        else:
            return None
    divs = reversed(divisors(gcd_list(coeffs))[1:])
    try:
        div = next(divs)
    except StopIteration:
        return None
    while True:
        for (monom, coeff) in zip(monoms, coeffs):
            if coeff % div ** monom != 0:
                try:
                    div = next(divs)
                except StopIteration:
                    return None
                else:
                    break
        else:
            return div

def preprocess_roots(poly):
    if False:
        print('Hello World!')
    'Try to get rid of symbolic coefficients from ``poly``. '
    coeff = S.One
    poly_func = poly.func
    try:
        (_, poly) = poly.clear_denoms(convert=True)
    except DomainError:
        return (coeff, poly)
    poly = poly.primitive()[1]
    poly = poly.retract()
    if poly.get_domain().is_Poly and all((c.is_term for c in poly.rep.coeffs())):
        poly = poly.inject()
        strips = list(zip(*poly.monoms()))
        gens = list(poly.gens[1:])
        (base, strips) = (strips[0], strips[1:])
        for (gen, strip) in zip(list(gens), strips):
            reverse = False
            if strip[0] < strip[-1]:
                strip = reversed(strip)
                reverse = True
            ratio = None
            for (a, b) in zip(base, strip):
                if not a and (not b):
                    continue
                elif not a or not b:
                    break
                elif b % a != 0:
                    break
                else:
                    _ratio = b // a
                    if ratio is None:
                        ratio = _ratio
                    elif ratio != _ratio:
                        break
            else:
                if reverse:
                    ratio = -ratio
                poly = poly.eval(gen, 1)
                coeff *= gen ** (-ratio)
                gens.remove(gen)
        if gens:
            poly = poly.eject(*gens)
    if poly.is_univariate and poly.get_domain().is_ZZ:
        basis = _integer_basis(poly)
        if basis is not None:
            n = poly.degree()

            def func(k, coeff):
                if False:
                    i = 10
                    return i + 15
                return coeff // basis ** (n - k[0])
            poly = poly.termwise(func)
            coeff *= basis
    if not isinstance(poly, poly_func):
        poly = poly_func(poly)
    return (coeff, poly)

@public
def roots(f, *gens, auto=True, cubics=True, trig=False, quartics=True, quintics=False, multiple=False, filter=None, predicate=None, strict=False, **flags):
    if False:
        for i in range(10):
            print('nop')
    "\n    Computes symbolic roots of a univariate polynomial.\n\n    Given a univariate polynomial f with symbolic coefficients (or\n    a list of the polynomial's coefficients), returns a dictionary\n    with its roots and their multiplicities.\n\n    Only roots expressible via radicals will be returned.  To get\n    a complete set of roots use RootOf class or numerical methods\n    instead. By default cubic and quartic formulas are used in\n    the algorithm. To disable them because of unreadable output\n    set ``cubics=False`` or ``quartics=False`` respectively. If cubic\n    roots are real but are expressed in terms of complex numbers\n    (casus irreducibilis [1]) the ``trig`` flag can be set to True to\n    have the solutions returned in terms of cosine and inverse cosine\n    functions.\n\n    To get roots from a specific domain set the ``filter`` flag with\n    one of the following specifiers: Z, Q, R, I, C. By default all\n    roots are returned (this is equivalent to setting ``filter='C'``).\n\n    By default a dictionary is returned giving a compact result in\n    case of multiple roots.  However to get a list containing all\n    those roots set the ``multiple`` flag to True; the list will\n    have identical roots appearing next to each other in the result.\n    (For a given Poly, the all_roots method will give the roots in\n    sorted numerical order.)\n\n    If the ``strict`` flag is True, ``UnsolvableFactorError`` will be\n    raised if the roots found are known to be incomplete (because\n    some roots are not expressible in radicals).\n\n    Examples\n    ========\n\n    >>> from sympy import Poly, roots, degree\n    >>> from sympy.abc import x, y\n\n    >>> roots(x**2 - 1, x)\n    {-1: 1, 1: 1}\n\n    >>> p = Poly(x**2-1, x)\n    >>> roots(p)\n    {-1: 1, 1: 1}\n\n    >>> p = Poly(x**2-y, x, y)\n\n    >>> roots(Poly(p, x))\n    {-sqrt(y): 1, sqrt(y): 1}\n\n    >>> roots(x**2 - y, x)\n    {-sqrt(y): 1, sqrt(y): 1}\n\n    >>> roots([1, 0, -1])\n    {-1: 1, 1: 1}\n\n    ``roots`` will only return roots expressible in radicals. If\n    the given polynomial has some or all of its roots inexpressible in\n    radicals, the result of ``roots`` will be incomplete or empty\n    respectively.\n\n    Example where result is incomplete:\n\n    >>> roots((x-1)*(x**5-x+1), x)\n    {1: 1}\n\n    In this case, the polynomial has an unsolvable quintic factor\n    whose roots cannot be expressed by radicals. The polynomial has a\n    rational root (due to the factor `(x-1)`), which is returned since\n    ``roots`` always finds all rational roots.\n\n    Example where result is empty:\n\n    >>> roots(x**7-3*x**2+1, x)\n    {}\n\n    Here, the polynomial has no roots expressible in radicals, so\n    ``roots`` returns an empty dictionary.\n\n    The result produced by ``roots`` is complete if and only if the\n    sum of the multiplicity of each root is equal to the degree of\n    the polynomial. If strict=True, UnsolvableFactorError will be\n    raised if the result is incomplete.\n\n    The result can be be checked for completeness as follows:\n\n    >>> f = x**3-2*x**2+1\n    >>> sum(roots(f, x).values()) == degree(f, x)\n    True\n    >>> f = (x-1)*(x**5-x+1)\n    >>> sum(roots(f, x).values()) == degree(f, x)\n    False\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_and_hyperbolic_solutions\n\n    "
    from sympy.polys.polytools import to_rational_coeffs
    flags = dict(flags)
    if isinstance(f, list):
        if gens:
            raise ValueError('redundant generators given')
        x = Dummy('x')
        (poly, i) = ({}, len(f) - 1)
        for coeff in f:
            (poly[i], i) = (sympify(coeff), i - 1)
        f = Poly(poly, x, field=True)
    else:
        try:
            F = Poly(f, *gens, **flags)
            if not isinstance(f, Poly) and (not F.gen.is_Symbol):
                raise PolynomialError('generator must be a Symbol')
            f = F
        except GeneratorsNeeded:
            if multiple:
                return []
            else:
                return {}
        else:
            n = f.degree()
            if f.length() == 2 and n > 2:
                (con, dep) = f.as_expr().as_independent(*f.gens)
                fcon = -(-con).factor()
                if fcon != con:
                    con = fcon
                    bases = []
                    for i in Mul.make_args(con):
                        if i.is_Pow:
                            (b, e) = i.as_base_exp()
                            if e.is_Integer and b.is_Add:
                                bases.append((b, Dummy(positive=True)))
                    if bases:
                        rv = roots(Poly((dep + con).xreplace(dict(bases)), *f.gens), *F.gens, auto=auto, cubics=cubics, trig=trig, quartics=quartics, quintics=quintics, multiple=multiple, filter=filter, predicate=predicate, **flags)
                        return {factor_terms(k.xreplace({v: k for (k, v) in bases})): v for (k, v) in rv.items()}
        if f.is_multivariate:
            raise PolynomialError('multivariate polynomials are not supported')

    def _update_dict(result, zeros, currentroot, k):
        if False:
            while True:
                i = 10
        if currentroot == S.Zero:
            if S.Zero in zeros:
                zeros[S.Zero] += k
            else:
                zeros[S.Zero] = k
        if currentroot in result:
            result[currentroot] += k
        else:
            result[currentroot] = k

    def _try_decompose(f):
        if False:
            return 10
        'Find roots using functional decomposition. '
        (factors, roots) = (f.decompose(), [])
        for currentroot in _try_heuristics(factors[0]):
            roots.append(currentroot)
        for currentfactor in factors[1:]:
            (previous, roots) = (list(roots), [])
            for currentroot in previous:
                g = currentfactor - Poly(currentroot, f.gen)
                for currentroot in _try_heuristics(g):
                    roots.append(currentroot)
        return roots

    def _try_heuristics(f):
        if False:
            while True:
                i = 10
        'Find roots using formulas and some tricks. '
        if f.is_ground:
            return []
        if f.is_monomial:
            return [S.Zero] * f.degree()
        if f.length() == 2:
            if f.degree() == 1:
                return list(map(cancel, roots_linear(f)))
            else:
                return roots_binomial(f)
        result = []
        for i in [-1, 1]:
            if not f.eval(i):
                f = f.quo(Poly(f.gen - i, f.gen))
                result.append(i)
                break
        n = f.degree()
        if n == 1:
            result += list(map(cancel, roots_linear(f)))
        elif n == 2:
            result += list(map(cancel, roots_quadratic(f)))
        elif f.is_cyclotomic:
            result += roots_cyclotomic(f)
        elif n == 3 and cubics:
            result += roots_cubic(f, trig=trig)
        elif n == 4 and quartics:
            result += roots_quartic(f)
        elif n == 5 and quintics:
            result += roots_quintic(f)
        return result
    dumgens = symbols('x:%d' % len(f.gens), cls=Dummy)
    f = f.per(f.rep, dumgens)
    ((k,), f) = f.terms_gcd()
    if not k:
        zeros = {}
    else:
        zeros = {S.Zero: k}
    (coeff, f) = preprocess_roots(f)
    if auto and f.get_domain().is_Ring:
        f = f.to_field()
    if f.get_domain().is_QQ_I:
        f = f.per(f.rep.convert(EX))
    rescale_x = None
    translate_x = None
    result = {}
    if not f.is_ground:
        dom = f.get_domain()
        if not dom.is_Exact and dom.is_Numerical:
            for r in f.nroots():
                _update_dict(result, zeros, r, 1)
        elif f.degree() == 1:
            _update_dict(result, zeros, roots_linear(f)[0], 1)
        elif f.length() == 2:
            roots_fun = roots_quadratic if f.degree() == 2 else roots_binomial
            for r in roots_fun(f):
                _update_dict(result, zeros, r, 1)
        else:
            (_, factors) = Poly(f.as_expr()).factor_list()
            if len(factors) == 1 and f.degree() == 2:
                for r in roots_quadratic(f):
                    _update_dict(result, zeros, r, 1)
            elif len(factors) == 1 and factors[0][1] == 1:
                if f.get_domain().is_EX:
                    res = to_rational_coeffs(f)
                    if res:
                        if res[0] is None:
                            (translate_x, f) = res[2:]
                        else:
                            (rescale_x, f) = (res[1], res[-1])
                        result = roots(f)
                        if not result:
                            for currentroot in _try_decompose(f):
                                _update_dict(result, zeros, currentroot, 1)
                    else:
                        for r in _try_heuristics(f):
                            _update_dict(result, zeros, r, 1)
                else:
                    for currentroot in _try_decompose(f):
                        _update_dict(result, zeros, currentroot, 1)
            else:
                for (currentfactor, k) in factors:
                    for r in _try_heuristics(Poly(currentfactor, f.gen, field=True)):
                        _update_dict(result, zeros, r, k)
    if coeff is not S.One:
        (_result, result) = (result, {})
        for (currentroot, k) in _result.items():
            result[coeff * currentroot] = k
    if filter not in [None, 'C']:
        handlers = {'Z': lambda r: r.is_Integer, 'Q': lambda r: r.is_Rational, 'R': lambda r: all((a.is_real for a in r.as_numer_denom())), 'I': lambda r: r.is_imaginary}
        try:
            query = handlers[filter]
        except KeyError:
            raise ValueError('Invalid filter: %s' % filter)
        for zero in dict(result).keys():
            if not query(zero):
                del result[zero]
    if predicate is not None:
        for zero in dict(result).keys():
            if not predicate(zero):
                del result[zero]
    if rescale_x:
        result1 = {}
        for (k, v) in result.items():
            result1[k * rescale_x] = v
        result = result1
    if translate_x:
        result1 = {}
        for (k, v) in result.items():
            result1[k + translate_x] = v
        result = result1
    result.update(zeros)
    if strict and sum(result.values()) < f.degree():
        raise UnsolvableFactorError(filldedent('\n            Strict mode: some factors cannot be solved in radicals, so\n            a complete list of solutions cannot be returned. Call\n            roots with strict=False to get solutions expressible in\n            radicals (if there are any).\n            '))
    if not multiple:
        return result
    else:
        zeros = []
        for zero in ordered(result):
            zeros.extend([zero] * result[zero])
        return zeros

def root_factors(f, *gens, filter=None, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns all factors of a univariate polynomial.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.polys.polyroots import root_factors\n\n    >>> root_factors(x**2 - y, x)\n    [x - sqrt(y), x + sqrt(y)]\n\n    '
    args = dict(args)
    F = Poly(f, *gens, **args)
    if not F.is_Poly:
        return [f]
    if F.is_multivariate:
        raise ValueError('multivariate polynomials are not supported')
    x = F.gens[0]
    zeros = roots(F, filter=filter)
    if not zeros:
        factors = [F]
    else:
        (factors, N) = ([], 0)
        for (r, n) in ordered(zeros.items()):
            (factors, N) = (factors + [Poly(x - r, x)] * n, N + n)
        if N < F.degree():
            G = reduce(lambda p, q: p * q, factors)
            factors.append(F.quo(G))
    if not isinstance(f, Poly):
        factors = [f.as_expr() for f in factors]
    return factors