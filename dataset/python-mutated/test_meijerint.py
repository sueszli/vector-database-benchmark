from sympy.core.function import expand_func
from sympy.core.numbers import I, Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.complexes import Abs, arg, re, unpolarify
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.hyperbolic import cosh, acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import cos, sin, sinc, asin
from sympy.functions.special.error_functions import erf, erfc
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals.integrals import Integral, integrate
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import simplify
from sympy.integrals.meijerint import _rewrite_single, _rewrite1, meijerint_indefinite, _inflate_g, _create_lookup_table, meijerint_definite, meijerint_inversion
from sympy.testing.pytest import slow
from sympy.core.random import verify_numerically, random_complex_number as randcplx
from sympy.abc import x, y, a, b, c, d, s, t, z

def test_rewrite_single():
    if False:
        i = 10
        return i + 15

    def t(expr, c, m):
        if False:
            for i in range(10):
                print('nop')
        e = _rewrite_single(meijerg([a], [b], [c], [d], expr), x)
        assert e is not None
        assert isinstance(e[0][0][2], meijerg)
        assert e[0][0][2].argument.as_coeff_mul(x) == (c, (m,))

    def tn(expr):
        if False:
            print('Hello World!')
        assert _rewrite_single(meijerg([a], [b], [c], [d], expr), x) is None
    t(x, 1, x)
    t(x ** 2, 1, x ** 2)
    t(x ** 2 + y * x ** 2, y + 1, x ** 2)
    tn(x ** 2 + x)
    tn(x ** y)

    def u(expr, x):
        if False:
            i = 10
            return i + 15
        from sympy.core.add import Add
        r = _rewrite_single(expr, x)
        e = Add(*[res[0] * res[2] for res in r[0]]).replace(exp_polar, exp)
        assert verify_numerically(e, expr, x)
    u(exp(-x) * sin(x), x)
    assert _rewrite_single(exp(x) * sin(x), x) == ([(-sqrt(2) / (2 * sqrt(pi)), 0, meijerg(((Rational(-1, 2), 0, Rational(1, 4), S.Half, Rational(3, 4)), (1,)), ((), (Rational(-1, 2), 0)), 64 * exp_polar(-4 * I * pi) / x ** 4))], True)

def test_rewrite1():
    if False:
        return 10
    assert _rewrite1(x ** 3 * meijerg([a], [b], [c], [d], x ** 2 + y * x ** 2) * 5, x) == (5, x ** 3, [(1, 0, meijerg([a], [b], [c], [d], x ** 2 * (y + 1)))], True)

def test_meijerint_indefinite_numerically():
    if False:
        print('Hello World!')

    def t(fac, arg):
        if False:
            while True:
                i = 10
        g = meijerg([a], [b], [c], [d], arg) * fac
        subs = {a: randcplx() / 10, b: randcplx() / 10 + I, c: randcplx(), d: randcplx()}
        integral = meijerint_indefinite(g, x)
        assert integral is not None
        assert verify_numerically(g.subs(subs), integral.diff(x).subs(subs), x)
    t(1, x)
    t(2, x)
    t(1, 2 * x)
    t(1, x ** 2)
    t(5, x ** S('3/2'))
    t(x ** 3, x)
    t(3 * x ** S('3/2'), 4 * x ** S('7/3'))

def test_meijerint_definite():
    if False:
        while True:
            i = 10
    (v, b) = meijerint_definite(x, x, 0, 0)
    assert v.is_zero and b is True
    (v, b) = meijerint_definite(x, x, oo, oo)
    assert v.is_zero and b is True

def test_inflate():
    if False:
        for i in range(10):
            print('nop')
    subs = {a: randcplx() / 10, b: randcplx() / 10 + I, c: randcplx(), d: randcplx(), y: randcplx() / 10}

    def t(a, b, arg, n):
        if False:
            return 10
        from sympy.core.mul import Mul
        m1 = meijerg(a, b, arg)
        m2 = Mul(*_inflate_g(m1, n))
        return verify_numerically(m1.subs(subs), m2.subs(subs), x, b=0.1, d=-0.1)
    assert t([[a], [b]], [[c], [d]], x, 3)
    assert t([[a, y], [b]], [[c], [d]], x, 3)
    assert t([[a], [b]], [[c, y], [d]], 2 * x ** 3, 3)

def test_recursive():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.symbol import symbols
    (a, b, c) = symbols('a b c', positive=True)
    r = exp(-(x - a) ** 2) * exp(-(x - b) ** 2)
    e = integrate(r, (x, 0, oo), meijerg=True)
    assert simplify(e.expand()) == sqrt(2) * sqrt(pi) * ((erf(sqrt(2) * (a + b) / 2) + 1) * exp(-a ** 2 / 2 + a * b - b ** 2 / 2)) / 4
    e = integrate(exp(-(x - a) ** 2) * exp(-(x - b) ** 2) * exp(c * x), (x, 0, oo), meijerg=True)
    assert simplify(e) == sqrt(2) * sqrt(pi) * (erf(sqrt(2) * (2 * a + 2 * b + c) / 4) + 1) * exp(-a ** 2 - b ** 2 + (2 * a + 2 * b + c) ** 2 / 8) / 4
    assert simplify(integrate(exp(-(x - a - b - c) ** 2), (x, 0, oo), meijerg=True)) == sqrt(pi) / 2 * (1 + erf(a + b + c))
    assert simplify(integrate(exp(-(x + a + b + c) ** 2), (x, 0, oo), meijerg=True)) == sqrt(pi) / 2 * (1 - erf(a + b + c))

@slow
def test_meijerint():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.function import expand
    from sympy.core.symbol import symbols
    (s, t, mu) = symbols('s t mu', real=True)
    assert integrate(meijerg([], [], [0], [], s * t) * meijerg([], [], [mu / 2], [-mu / 2], t ** 2 / 4), (t, 0, oo)).is_Piecewise
    s = symbols('s', positive=True)
    assert integrate(x ** s * meijerg([[], []], [[0], []], x), (x, 0, oo)) == gamma(s + 1)
    assert integrate(x ** s * meijerg([[], []], [[0], []], x), (x, 0, oo), meijerg=True) == gamma(s + 1)
    assert isinstance(integrate(x ** s * meijerg([[], []], [[0], []], x), (x, 0, oo), meijerg=False), Integral)
    assert meijerint_indefinite(exp(x), x) == exp(x)
    (a, b) = symbols('a b', positive=True)
    assert simplify(meijerint_definite(x ** a, x, 0, b)[0]) == b ** (a + 1) / (a + 1)
    assert meijerint_definite((x + 1) ** 3 * exp(-x), x, 0, oo) == (16, True)
    (sigma, mu) = symbols('sigma mu', positive=True)
    (i, c) = meijerint_definite(exp(-((x - mu) / (2 * sigma)) ** 2), x, 0, oo)
    assert simplify(i) == sqrt(pi) * sigma * (2 - erfc(mu / (2 * sigma)))
    assert c == True
    (i, _) = meijerint_definite(exp(-mu * x) * exp(sigma * x), x, 0, oo)
    assert simplify(i) == 1 / (mu - sigma)
    assert meijerint_definite(exp(x), x, -oo, 2) == (exp(2), True)
    assert expand(meijerint_definite(exp(x), x, 0, I)[0]) == exp(I) - 1
    assert expand(meijerint_definite(exp(-x), x, 0, x)[0]) == 1 - exp(-exp(I * arg(x)) * abs(x))
    assert meijerint_definite(exp(-x ** 2), x, -oo, oo) == (sqrt(pi), True)
    assert meijerint_definite(exp(-abs(x)), x, -oo, oo) == (2, True)
    assert meijerint_definite(exp(-(2 * x - 3) ** 2), x, -oo, oo) == (sqrt(pi) / 2, True)
    assert meijerint_definite(exp(-abs(2 * x - 3)), x, -oo, oo) == (1, True)
    assert meijerint_definite(exp(-((x - mu) / sigma) ** 2 / 2) / sqrt(2 * pi * sigma ** 2), x, -oo, oo) == (1, True)
    assert meijerint_definite(sinc(x) ** 2, x, -oo, oo) == (pi, True)
    assert meijerint_definite(exp(-x) * sin(x), x, 0, oo) == (S.Half, True)

    def res(n):
        if False:
            i = 10
            return i + 15
        return (1 / (1 + x ** 2)).diff(x, n).subs(x, 1) * (-1) ** n
    for n in range(6):
        assert integrate(exp(-x) * sin(x) * x ** n, (x, 0, oo), meijerg=True) == res(n)
    assert simplify(integrate(exp(-x) * sin(x + a), (x, 0, oo), meijerg=True)) == sqrt(2) * sin(a + pi / 4) / 2
    (a, b, s) = symbols('a b s')
    assert meijerint_definite(meijerg([], [], [a / 2], [-a / 2], x / 4) * meijerg([], [], [b / 2], [-b / 2], x / 4) * x ** (s - 1), x, 0, oo) == (4 * 2 ** (2 * s - 2) * gamma(-2 * s + 1) * gamma(a / 2 + b / 2 + s) / (gamma(-a / 2 + b / 2 - s + 1) * gamma(a / 2 - b / 2 - s + 1) * gamma(a / 2 + b / 2 - s + 1)), (re(s) < 1) & (re(s) < S(1) / 2) & (re(a) / 2 + re(b) / 2 + re(s) > 0))
    assert integrate(sin(x ** a) * sin(x ** b), (x, 0, oo), meijerg=True) == Integral(sin(x ** a) * sin(x ** b), (x, 0, oo))
    assert integrate(exp(-x ** 2) * log(x), (x, 0, oo), meijerg=True) == (sqrt(pi) * polygamma(0, S.Half) / 4).expand()
    from sympy.functions.special.gamma_functions import lowergamma
    n = symbols('n', integer=True)
    assert simplify(integrate(exp(-x) * x ** n, x, meijerg=True)) == lowergamma(n + 1, x)
    alpha = symbols('alpha', positive=True)
    assert meijerint_definite((2 - x) ** alpha * sin(alpha / x), x, 0, 2) == (sqrt(pi) * alpha * gamma(alpha + 1) * meijerg(((), (alpha / 2 + S.Half, alpha / 2 + 1)), ((0, 0, S.Half), (Rational(-1, 2),)), alpha ** 2 / 16) / 4, True)
    (a, s) = symbols('a s', positive=True)
    assert simplify(integrate(x ** s * exp(-a * x ** 2), (x, -oo, oo))) == a ** (-s / 2 - S.Half) * ((-1) ** s + 1) * gamma(s / 2 + S.Half) / 2

def test_bessel():
    if False:
        print('Hello World!')
    from sympy.functions.special.bessel import besseli, besselj
    assert simplify(integrate(besselj(a, z) * besselj(b, z) / z, (z, 0, oo), meijerg=True, conds='none')) == 2 * sin(pi * (a / 2 - b / 2)) / (pi * (a - b) * (a + b))
    assert simplify(integrate(besselj(a, z) * besselj(a, z) / z, (z, 0, oo), meijerg=True, conds='none')) == 1 / (2 * a)
    assert simplify(integrate(sin(z * x) * (x ** 2 - 1) ** (-(y + S.Half)), (x, 1, oo), meijerg=True, conds='none') * 2 / ((z / 2) ** y * sqrt(pi) * gamma(S.Half - y))) == besselj(y, z)
    assert integrate(x * besselj(0, x), x, meijerg=True) == x * besselj(1, x)
    assert integrate(x * besseli(0, x), x, meijerg=True) == x * besseli(1, x)
    assert integrate(besselj(1, x), x, meijerg=True) == -besselj(0, x)
    assert integrate(besselj(1, x) ** 2 / x, x, meijerg=True) == -(besselj(0, x) ** 2 + besselj(1, x) ** 2) / 2
    assert integrate(besselj(0, x) ** 2 / x ** 2, x, meijerg=True) == -2 * x * besselj(0, x) ** 2 - 2 * x * besselj(1, x) ** 2 + 2 * besselj(0, x) * besselj(1, x) - besselj(0, x) ** 2 / x
    assert integrate(besselj(0, x) * besselj(1, x), x, meijerg=True) == -besselj(0, x) ** 2 / 2
    assert integrate(x ** 2 * besselj(0, x) * besselj(1, x), x, meijerg=True) == x ** 2 * besselj(1, x) ** 2 / 2
    assert integrate(besselj(0, x) * besselj(1, x) / x, x, meijerg=True) == x * besselj(0, x) ** 2 + x * besselj(1, x) ** 2 - besselj(0, x) * besselj(1, x)
    assert integrate(besselj(1, x ** 2) * x, x, meijerg=True) == -besselj(0, x ** 2) / 2

def test_inversion():
    if False:
        while True:
            i = 10
    from sympy.functions.special.bessel import besselj
    from sympy.functions.special.delta_functions import Heaviside

    def inv(f):
        if False:
            print('Hello World!')
        return piecewise_fold(meijerint_inversion(f, s, t))
    assert inv(1 / (s ** 2 + 1)) == sin(t) * Heaviside(t)
    assert inv(s / (s ** 2 + 1)) == cos(t) * Heaviside(t)
    assert inv(exp(-s) / s) == Heaviside(t - 1)
    assert inv(1 / sqrt(1 + s ** 2)) == besselj(0, t) * Heaviside(t)
    assert meijerint_inversion(sqrt(s) / sqrt(1 + s ** 2), s, t) is None
    assert inv(exp(s ** 2)) is None
    assert meijerint_inversion(exp(-s ** 2), s, t) is None

def test_inversion_conditional_output():
    if False:
        i = 10
        return i + 15
    from sympy.core.symbol import Symbol
    from sympy.integrals.transforms import InverseLaplaceTransform
    a = Symbol('a', positive=True)
    F = sqrt(pi / a) * exp(-2 * sqrt(a) * sqrt(s))
    f = meijerint_inversion(F, s, t)
    assert not f.is_Piecewise
    b = Symbol('b', real=True)
    F = F.subs(a, b)
    f2 = meijerint_inversion(F, s, t)
    assert f2.is_Piecewise
    assert f2.args[0][0] == f.subs(a, b)
    assert f2.args[-1][1]
    ILT = InverseLaplaceTransform(F, s, t, None)
    assert f2.args[-1][0] == ILT or f2.args[-1][0] == ILT.as_integral

def test_inversion_exp_real_nonreal_shift():
    if False:
        print('Hello World!')
    from sympy.core.symbol import Symbol
    from sympy.functions.special.delta_functions import DiracDelta
    r = Symbol('r', real=True)
    c = Symbol('c', extended_real=False)
    a = 1 + 2 * I
    z = Symbol('z')
    assert not meijerint_inversion(exp(r * s), s, t).is_Piecewise
    assert meijerint_inversion(exp(a * s), s, t) is None
    assert meijerint_inversion(exp(c * s), s, t) is None
    f = meijerint_inversion(exp(z * s), s, t)
    assert f.is_Piecewise
    assert isinstance(f.args[0][0], DiracDelta)

@slow
def test_lookup_table():
    if False:
        print('Hello World!')
    from sympy.core.random import uniform, randrange
    from sympy.core.add import Add
    from sympy.integrals.meijerint import z as z_dummy
    table = {}
    _create_lookup_table(table)
    for (_, l) in table.items():
        for (formula, terms, cond, hint) in sorted(l, key=default_sort_key):
            subs = {}
            for ai in list(formula.free_symbols) + [z_dummy]:
                if hasattr(ai, 'properties') and ai.properties:
                    subs[ai] = randrange(1, 10)
                else:
                    subs[ai] = uniform(1.5, 2.0)
            if not isinstance(terms, list):
                terms = terms(subs)
            expanded = [hyperexpand(g) for (_, g) in terms]
            assert all((x.is_Piecewise or not x.has(meijerg) for x in expanded))
            expanded = Add(*[f * x for (f, x) in terms])
            (a, b) = (formula.n(subs=subs), expanded.n(subs=subs))
            r = min(abs(a), abs(b))
            if r < 1:
                assert abs(a - b).n() <= 1e-10
            else:
                assert (abs(a - b) / r).n() <= 1e-10

def test_branch_bug():
    if False:
        print('Hello World!')
    from sympy.functions.special.gamma_functions import lowergamma
    from sympy.simplify.powsimp import powdenest
    assert powdenest(integrate(erf(x ** 3), x, meijerg=True).diff(x), polar=True) == 2 * erf(x ** 3) * gamma(Rational(2, 3)) / 3 / gamma(Rational(5, 3))
    assert integrate(erf(x ** 3), x, meijerg=True) == 2 * x * erf(x ** 3) * gamma(Rational(2, 3)) / (3 * gamma(Rational(5, 3))) - 2 * gamma(Rational(2, 3)) * lowergamma(Rational(2, 3), x ** 6) / (3 * sqrt(pi) * gamma(Rational(5, 3)))

def test_linear_subs():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.bessel import besselj
    assert integrate(sin(x - 1), x, meijerg=True) == -cos(1 - x)
    assert integrate(besselj(1, x - 1), x, meijerg=True) == -besselj(0, 1 - x)

@slow
def test_probability():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.function import expand_mul
    from sympy.core.symbol import Symbol, symbols
    from sympy.simplify.gammasimp import gammasimp
    from sympy.simplify.powsimp import powsimp
    (mu1, mu2) = symbols('mu1 mu2', nonzero=True)
    (sigma1, sigma2) = symbols('sigma1 sigma2', positive=True)
    rate = Symbol('lambda', positive=True)

    def normal(x, mu, sigma):
        if False:
            while True:
                i = 10
        return 1 / sqrt(2 * pi * sigma ** 2) * exp(-(x - mu) ** 2 / 2 / sigma ** 2)

    def exponential(x, rate):
        if False:
            i = 10
            return i + 15
        return rate * exp(-rate * x)
    assert integrate(normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == 1
    assert integrate(x * normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == mu1
    assert integrate(x ** 2 * normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == mu1 ** 2 + sigma1 ** 2
    assert integrate(x ** 3 * normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True) == mu1 ** 3 + 3 * mu1 * sigma1 ** 2
    assert integrate(normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == 1
    assert integrate(x * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu1
    assert integrate(y * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu2
    assert integrate(x * y * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == mu1 * mu2
    assert integrate((x + y + 1) * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == 1 + mu1 + mu2
    assert integrate((x + y - 1) * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == -1 + mu1 + mu2
    i = integrate(x ** 2 * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True)
    assert not i.has(Abs)
    assert simplify(i) == mu1 ** 2 + sigma1 ** 2
    assert integrate(y ** 2 * normal(x, mu1, sigma1) * normal(y, mu2, sigma2), (x, -oo, oo), (y, -oo, oo), meijerg=True) == sigma2 ** 2 + mu2 ** 2
    assert integrate(exponential(x, rate), (x, 0, oo), meijerg=True) == 1
    assert integrate(x * exponential(x, rate), (x, 0, oo), meijerg=True) == 1 / rate
    assert integrate(x ** 2 * exponential(x, rate), (x, 0, oo), meijerg=True) == 2 / rate ** 2

    def E(expr):
        if False:
            i = 10
            return i + 15
        res1 = integrate(expr * exponential(x, rate) * normal(y, mu1, sigma1), (x, 0, oo), (y, -oo, oo), meijerg=True)
        res2 = integrate(expr * exponential(x, rate) * normal(y, mu1, sigma1), (y, -oo, oo), (x, 0, oo), meijerg=True)
        assert expand_mul(res1) == expand_mul(res2)
        return res1
    assert E(1) == 1
    assert E(x * y) == mu1 / rate
    assert E(x * y ** 2) == mu1 ** 2 / rate + sigma1 ** 2 / rate
    ans = sigma1 ** 2 + 1 / rate ** 2
    assert simplify(E((x + y + 1) ** 2) - E(x + y + 1) ** 2) == ans
    assert simplify(E((x + y - 1) ** 2) - E(x + y - 1) ** 2) == ans
    assert simplify(E((x + y) ** 2) - E(x + y) ** 2) == ans
    (alpha, beta) = symbols('alpha beta', positive=True)
    betadist = x ** (alpha - 1) * (1 + x) ** (-alpha - beta) * gamma(alpha + beta) / gamma(alpha) / gamma(beta)
    assert integrate(betadist, (x, 0, oo), meijerg=True) == 1
    i = integrate(x * betadist, (x, 0, oo), meijerg=True, conds='separate')
    assert (gammasimp(i[0]), i[1]) == (alpha / (beta - 1), 1 < beta)
    j = integrate(x ** 2 * betadist, (x, 0, oo), meijerg=True, conds='separate')
    assert j[1] == (beta > 2)
    assert gammasimp(j[0] - i[0] ** 2) == (alpha + beta - 1) * alpha / (beta - 2) / (beta - 1) ** 2
    (a, b) = symbols('a b', positive=True)
    betadist = x ** (a - 1) * (-x + 1) ** (b - 1) * gamma(a + b) / (gamma(a) * gamma(b))
    assert simplify(integrate(betadist, (x, 0, 1), meijerg=True)) == 1
    assert simplify(integrate(x * betadist, (x, 0, 1), meijerg=True)) == a / (a + b)
    assert simplify(integrate(x ** 2 * betadist, (x, 0, 1), meijerg=True)) == a * (a + 1) / (a + b) / (a + b + 1)
    assert simplify(integrate(x ** y * betadist, (x, 0, 1), meijerg=True)) == gamma(a + b) * gamma(a + y) / gamma(a) / gamma(a + b + y)
    k = Symbol('k', integer=True, positive=True)
    chi = 2 ** (1 - k / 2) * x ** (k - 1) * exp(-x ** 2 / 2) / gamma(k / 2)
    assert powsimp(integrate(chi, (x, 0, oo), meijerg=True)) == 1
    assert simplify(integrate(x * chi, (x, 0, oo), meijerg=True)) == sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
    assert simplify(integrate(x ** 2 * chi, (x, 0, oo), meijerg=True)) == k
    chisquared = 2 ** (-k / 2) / gamma(k / 2) * x ** (k / 2 - 1) * exp(-x / 2)
    assert powsimp(integrate(chisquared, (x, 0, oo), meijerg=True)) == 1
    assert simplify(integrate(x * chisquared, (x, 0, oo), meijerg=True)) == k
    assert simplify(integrate(x ** 2 * chisquared, (x, 0, oo), meijerg=True)) == k * (k + 2)
    assert gammasimp(integrate(((x - k) / sqrt(2 * k)) ** 3 * chisquared, (x, 0, oo), meijerg=True)) == 2 * sqrt(2) / sqrt(k)
    (a, b, p) = symbols('a b p', positive=True)
    dagum = a * p / x * (x / b) ** (a * p) / (1 + x ** a / b ** a) ** (p + 1)
    assert simplify(integrate(dagum, (x, 0, oo), meijerg=True)) == 1
    arg = x * dagum
    assert simplify(integrate(arg, (x, 0, oo), meijerg=True, conds='none')) == a * b * gamma(1 - 1 / a) * gamma(p + 1 + 1 / a) / ((a * p + 1) * gamma(p))
    assert simplify(integrate(x * arg, (x, 0, oo), meijerg=True, conds='none')) == a * b ** 2 * gamma(1 - 2 / a) * gamma(p + 1 + 2 / a) / ((a * p + 2) * gamma(p))
    (d1, d2) = symbols('d1 d2', positive=True)
    f = sqrt((d1 * x) ** d1 * d2 ** d2 / (d1 * x + d2) ** (d1 + d2)) / x / gamma(d1 / 2) / gamma(d2 / 2) * gamma((d1 + d2) / 2)
    assert simplify(integrate(f, (x, 0, oo), meijerg=True)) == 1
    assert simplify(integrate(x * f, (x, 0, oo), meijerg=True, conds='none')) == d2 / (d2 - 2)
    assert simplify(integrate(x ** 2 * f, (x, 0, oo), meijerg=True, conds='none')) == d2 ** 2 * (d1 + 2) / d1 / (d2 - 4) / (d2 - 2)
    (lamda, mu) = symbols('lamda mu', positive=True)
    dist = sqrt(lamda / 2 / pi) * x ** Rational(-3, 2) * exp(-lamda * (x - mu) ** 2 / x / 2 / mu ** 2)
    mysimp = lambda expr: simplify(expr.rewrite(exp))
    assert mysimp(integrate(dist, (x, 0, oo))) == 1
    assert mysimp(integrate(x * dist, (x, 0, oo))) == mu
    assert mysimp(integrate((x - mu) ** 2 * dist, (x, 0, oo))) == mu ** 3 / lamda
    assert mysimp(integrate((x - mu) ** 3 * dist, (x, 0, oo))) == 3 * mu ** 5 / lamda ** 2
    c = Symbol('c', positive=True)
    assert integrate(sqrt(c / 2 / pi) * exp(-c / 2 / (x - mu)) / (x - mu) ** S('3/2'), (x, mu, oo)) == 1
    (alpha, beta) = symbols('alpha beta', positive=True)
    distn = beta / alpha * x ** (beta - 1) / alpha ** (beta - 1) / (1 + x ** beta / alpha ** beta) ** 2
    assert simplify(integrate(distn, (x, 0, oo))) == 1
    assert simplify(integrate(x * distn, (x, 0, oo), conds='none')) == pi * alpha / beta / sin(pi / beta)
    assert simplify(integrate(x ** y * distn, (x, 0, oo), conds='none')) == pi * alpha ** y * y / beta / sin(pi * y / beta)
    k = Symbol('k', positive=True)
    n = Symbol('n', positive=True)
    distn = k / lamda * (x / lamda) ** (k - 1) * exp(-(x / lamda) ** k)
    assert simplify(integrate(distn, (x, 0, oo))) == 1
    assert simplify(integrate(x ** n * distn, (x, 0, oo))) == lamda ** n * gamma(1 + n / k)
    from sympy.functions.special.bessel import besseli
    (nu, sigma) = symbols('nu sigma', positive=True)
    rice = x / sigma ** 2 * exp(-(x ** 2 + nu ** 2) / 2 / sigma ** 2) * besseli(0, x * nu / sigma ** 2)
    assert integrate(rice, (x, 0, oo), meijerg=True) == 1
    mu = Symbol('mu', real=True)
    b = Symbol('b', positive=True)
    laplace = exp(-abs(x - mu) / b) / 2 / b
    assert integrate(laplace, (x, -oo, oo), meijerg=True) == 1
    assert integrate(x * laplace, (x, -oo, oo), meijerg=True) == mu
    assert integrate(x ** 2 * laplace, (x, -oo, oo), meijerg=True) == 2 * b ** 2 + mu ** 2
    k = Symbol('k', positive=True)
    assert gammasimp(expand_mul(integrate(log(x) * x ** (k - 1) * exp(-x) / gamma(k), (x, 0, oo)))) == polygamma(0, k)

@slow
def test_expint():
    if False:
        i = 10
        return i + 15
    ' Test various exponential integrals. '
    from sympy.core.symbol import Symbol
    from sympy.functions.elementary.hyperbolic import sinh
    from sympy.functions.special.error_functions import Chi, Ci, Ei, Shi, Si, expint
    assert simplify(unpolarify(integrate(exp(-z * x) / x ** y, (x, 1, oo), meijerg=True, conds='none').rewrite(expint).expand(func=True))) == expint(y, z)
    assert integrate(exp(-z * x) / x, (x, 1, oo), meijerg=True, conds='none').rewrite(expint).expand() == expint(1, z)
    assert integrate(exp(-z * x) / x ** 2, (x, 1, oo), meijerg=True, conds='none').rewrite(expint).expand() == expint(2, z).rewrite(Ei).rewrite(expint)
    assert integrate(exp(-z * x) / x ** 3, (x, 1, oo), meijerg=True, conds='none').rewrite(expint).expand() == expint(3, z).rewrite(Ei).rewrite(expint).expand()
    t = Symbol('t', positive=True)
    assert integrate(-cos(x) / x, (x, t, oo), meijerg=True).expand() == Ci(t)
    assert integrate(-sin(x) / x, (x, t, oo), meijerg=True).expand() == Si(t) - pi / 2
    assert integrate(sin(x) / x, (x, 0, z), meijerg=True) == Si(z)
    assert integrate(sinh(x) / x, (x, 0, z), meijerg=True) == Shi(z)
    assert integrate(exp(-x) / x, x, meijerg=True).expand().rewrite(expint) == I * pi - expint(1, x)
    assert integrate(exp(-x) / x ** 2, x, meijerg=True).rewrite(expint).expand() == expint(1, x) - exp(-x) / x - I * pi
    u = Symbol('u', polar=True)
    assert integrate(cos(u) / u, u, meijerg=True).expand().as_independent(u)[1] == Ci(u)
    assert integrate(cosh(u) / u, u, meijerg=True).expand().as_independent(u)[1] == Chi(u)
    assert integrate(expint(1, x), x, meijerg=True).rewrite(expint).expand() == x * expint(1, x) - exp(-x)
    assert integrate(expint(2, x), x, meijerg=True).rewrite(expint).expand() == -x ** 2 * expint(1, x) / 2 + x * exp(-x) / 2 - exp(-x) / 2
    assert simplify(unpolarify(integrate(expint(y, x), x, meijerg=True).rewrite(expint).expand(func=True))) == -expint(y + 1, x)
    assert integrate(Si(x), x, meijerg=True) == x * Si(x) + cos(x)
    assert integrate(Ci(u), u, meijerg=True).expand() == u * Ci(u) - sin(u)
    assert integrate(Shi(x), x, meijerg=True) == x * Shi(x) - cosh(x)
    assert integrate(Chi(u), u, meijerg=True).expand() == u * Chi(u) - sinh(u)
    assert integrate(Si(x) * exp(-x), (x, 0, oo), meijerg=True) == pi / 4
    assert integrate(expint(1, x) * sin(x), (x, 0, oo), meijerg=True) == log(2) / 2

def test_messy():
    if False:
        print('Hello World!')
    from sympy.functions.elementary.hyperbolic import acosh, acoth
    from sympy.functions.elementary.trigonometric import asin, atan
    from sympy.functions.special.bessel import besselj
    from sympy.functions.special.error_functions import Chi, E1, Shi, Si
    from sympy.integrals.transforms import fourier_transform, laplace_transform
    assert laplace_transform(Si(x), x, s, simplify=True) == ((-atan(s) + pi / 2) / s, 0, True)
    assert laplace_transform(Shi(x), x, s, simplify=True) == (acoth(s) / s, -oo, s ** 2 > 1)
    assert laplace_transform(Chi(x), x, s, simplify=True) == ((log(s ** (-2)) - log(1 - 1 / s ** 2)) / (2 * s), -oo, s ** 2 > 1)
    assert laplace_transform(besselj(a, x), x, s)[1:] == (0, (re(a) > -2) & (re(a) > -1))
    ans = fourier_transform(besselj(1, x) / x, x, s, noconds=False)
    assert (ans[0].factor(deep=True).expand(), ans[1]) == (Piecewise((0, (s > 1 / (2 * pi)) | (s < -1 / (2 * pi))), (2 * sqrt(-4 * pi ** 2 * s ** 2 + 1), True)), s > 0)
    assert integrate(E1(x) * besselj(0, x), (x, 0, oo), meijerg=True) == log(1 + sqrt(2))
    assert integrate(E1(x) * besselj(1, x), (x, 0, oo), meijerg=True) == log(S.Half + sqrt(2) / 2)
    assert integrate(1 / x / sqrt(1 - x ** 2), x, meijerg=True) == Piecewise((-acosh(1 / x), abs(x ** (-2)) > 1), (I * asin(1 / x), True))

def test_issue_6122():
    if False:
        i = 10
        return i + 15
    assert integrate(exp(-I * x ** 2), (x, -oo, oo), meijerg=True) == -I * sqrt(pi) * exp(I * pi / 4)

def test_issue_6252():
    if False:
        i = 10
        return i + 15
    expr = 1 / x / (a + b * x) ** Rational(1, 3)
    anti = integrate(expr, x, meijerg=True)
    assert not anti.has(hyper)

def test_issue_6348():
    if False:
        print('Hello World!')
    assert integrate(exp(I * x) / (1 + x ** 2), (x, -oo, oo)).simplify().rewrite(exp) == pi * exp(-1)

def test_fresnel():
    if False:
        for i in range(10):
            print('nop')
    from sympy.functions.special.error_functions import fresnelc, fresnels
    assert expand_func(integrate(sin(pi * x ** 2 / 2), x)) == fresnels(x)
    assert expand_func(integrate(cos(pi * x ** 2 / 2), x)) == fresnelc(x)

def test_issue_6860():
    if False:
        while True:
            i = 10
    assert meijerint_indefinite(x ** x ** x, x) is None

def test_issue_7337():
    if False:
        for i in range(10):
            print('nop')
    f = meijerint_indefinite(x * sqrt(2 * x + 3), x).together()
    assert f == sqrt(2 * x + 3) * (2 * x ** 2 + x - 3) / 5
    assert f._eval_interval(x, S.NegativeOne, S.One) == Rational(2, 5)

def test_issue_8368():
    if False:
        for i in range(10):
            print('nop')
    assert meijerint_indefinite(cosh(x) * exp(-x * t), x) == ((-t - 1) * exp(x) + (-t + 1) * exp(-x)) * exp(-t * x) / 2 / (t ** 2 - 1)

def test_issue_10211():
    if False:
        while True:
            i = 10
    from sympy.abc import h, w
    assert integrate(1 / sqrt((y - x) ** 2 + h ** 2) ** 3, (x, 0, w), (y, 0, w)) == 2 * sqrt(1 + w ** 2 / h ** 2) / h - 2 / h

def test_issue_11806():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.symbol import symbols
    (y, L) = symbols('y L', positive=True)
    assert integrate(1 / sqrt(x ** 2 + y ** 2) ** 3, (x, -L, L)) == 2 * L / (y ** 2 * sqrt(L ** 2 + y ** 2))

def test_issue_10681():
    if False:
        return 10
    from sympy.polys.domains.realfield import RR
    from sympy.abc import R, r
    f = integrate(r ** 2 * (R ** 2 - r ** 2) ** 0.5, r, meijerg=True)
    g = 1.0 / 3 * R ** 1.0 * r ** 3 * hyper((-0.5, Rational(3, 2)), (Rational(5, 2),), r ** 2 * exp_polar(2 * I * pi) / R ** 2)
    assert RR.almosteq((f / g).n(), 1.0, 1e-12)

def test_issue_13536():
    if False:
        for i in range(10):
            print('nop')
    from sympy.core.symbol import Symbol
    a = Symbol('a', positive=True)
    assert integrate(1 / x ** 2, (x, oo, a)) == -1 / a

def test_issue_6462():
    if False:
        print('Hello World!')
    from sympy.core.symbol import Symbol
    x = Symbol('x')
    n = Symbol('n')
    assert integrate(cos(x ** n) / x ** n, x, meijerg=True).subs(n, 2).equals(integrate(cos(x ** 2) / x ** 2, x, meijerg=True))

def test_indefinite_1_bug():
    if False:
        return 10
    assert integrate((b + t) ** (-a), t, meijerg=True) == -b ** (1 - a) * (1 + t / b) ** (1 - a) / (a - 1)

def test_pr_23583():
    if False:
        for i in range(10):
            print('nop')
    assert integrate(1 / sqrt((x - I) ** 2 - 1), meijerg=True) == Piecewise((acosh(x - I), Abs((x - I) ** 2) > 1), (-I * asin(x - I), True))