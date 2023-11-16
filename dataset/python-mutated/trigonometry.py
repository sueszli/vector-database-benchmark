from sympy.core import cacheit, Dummy, Ne, Integer, Rational, S, Wild
from sympy.functions import binomial, sin, cos, Piecewise, Abs
from .integrals import integrate

def _integer_instance(n):
    if False:
        return 10
    return isinstance(n, Integer)

@cacheit
def _pat_sincos(x):
    if False:
        while True:
            i = 10
    a = Wild('a', exclude=[x])
    (n, m) = [Wild(s, exclude=[x], properties=[_integer_instance]) for s in 'nm']
    pat = sin(a * x) ** n * cos(a * x) ** m
    return (pat, a, n, m)
_u = Dummy('u')

def trigintegrate(f, x, conds='piecewise'):
    if False:
        while True:
            i = 10
    '\n    Integrate f = Mul(trig) over x.\n\n    Examples\n    ========\n\n    >>> from sympy import sin, cos, tan, sec\n    >>> from sympy.integrals.trigonometry import trigintegrate\n    >>> from sympy.abc import x\n\n    >>> trigintegrate(sin(x)*cos(x), x)\n    sin(x)**2/2\n\n    >>> trigintegrate(sin(x)**2, x)\n    x/2 - sin(x)*cos(x)/2\n\n    >>> trigintegrate(tan(x)*sec(x), x)\n    1/cos(x)\n\n    >>> trigintegrate(sin(x)*tan(x), x)\n    -log(sin(x) - 1)/2 + log(sin(x) + 1)/2 - sin(x)\n\n    References\n    ==========\n\n    .. [1] https://en.wikibooks.org/wiki/Calculus/Integration_techniques\n\n    See Also\n    ========\n\n    sympy.integrals.integrals.Integral.doit\n    sympy.integrals.integrals.Integral\n    '
    (pat, a, n, m) = _pat_sincos(x)
    f = f.rewrite('sincos')
    M = f.match(pat)
    if M is None:
        return
    (n, m) = (M[n], M[m])
    if n.is_zero and m.is_zero:
        return x
    zz = x if n.is_zero else S.Zero
    a = M[a]
    if n.is_odd or m.is_odd:
        u = _u
        (n_, m_) = (n.is_odd, m.is_odd)
        if n_ and m_:
            if n < 0 and m > 0:
                m_ = True
                n_ = False
            elif m < 0 and n > 0:
                n_ = True
                m_ = False
            elif n < 0 and m < 0:
                n_ = n > m
                m_ = not n > m
            else:
                n_ = n < m
                m_ = not n < m
        if n_:
            ff = -(1 - u ** 2) ** ((n - 1) / 2) * u ** m
            uu = cos(a * x)
        elif m_:
            ff = u ** n * (1 - u ** 2) ** ((m - 1) / 2)
            uu = sin(a * x)
        fi = integrate(ff, u)
        fx = fi.subs(u, uu)
        if conds == 'piecewise':
            return Piecewise((fx / a, Ne(a, 0)), (zz, True))
        return fx / a
    n_ = Abs(n) > Abs(m)
    m_ = Abs(m) > Abs(n)
    res = S.Zero
    if n_:
        if m > 0:
            for i in range(0, m // 2 + 1):
                res += S.NegativeOne ** i * binomial(m // 2, i) * _sin_pow_integrate(n + 2 * i, x)
        elif m == 0:
            res = _sin_pow_integrate(n, x)
        else:
            res = Rational(-1, m + 1) * cos(x) ** (m + 1) * sin(x) ** (n - 1) + Rational(n - 1, m + 1) * trigintegrate(cos(x) ** (m + 2) * sin(x) ** (n - 2), x)
    elif m_:
        if n > 0:
            for i in range(0, n // 2 + 1):
                res += S.NegativeOne ** i * binomial(n // 2, i) * _cos_pow_integrate(m + 2 * i, x)
        elif n == 0:
            res = _cos_pow_integrate(m, x)
        else:
            res = Rational(1, n + 1) * cos(x) ** (m - 1) * sin(x) ** (n + 1) + Rational(m - 1, n + 1) * trigintegrate(cos(x) ** (m - 2) * sin(x) ** (n + 2), x)
    elif m == n:
        res = integrate((sin(2 * x) * S.Half) ** m, x)
    elif m == -n:
        if n < 0:
            res = Rational(1, n + 1) * cos(x) ** (m - 1) * sin(x) ** (n + 1) + Rational(m - 1, n + 1) * integrate(cos(x) ** (m - 2) * sin(x) ** (n + 2), x)
        else:
            res = Rational(-1, m + 1) * cos(x) ** (m + 1) * sin(x) ** (n - 1) + Rational(n - 1, m + 1) * integrate(cos(x) ** (m + 2) * sin(x) ** (n - 2), x)
    if conds == 'piecewise':
        return Piecewise((res.subs(x, a * x) / a, Ne(a, 0)), (zz, True))
    return res.subs(x, a * x) / a

def _sin_pow_integrate(n, x):
    if False:
        while True:
            i = 10
    if n > 0:
        if n == 1:
            return -cos(x)
        return Rational(-1, n) * cos(x) * sin(x) ** (n - 1) + Rational(n - 1, n) * _sin_pow_integrate(n - 2, x)
    if n < 0:
        if n == -1:
            return trigintegrate(1 / sin(x), x)
        return Rational(1, n + 1) * cos(x) * sin(x) ** (n + 1) + Rational(n + 2, n + 1) * _sin_pow_integrate(n + 2, x)
    else:
        return x

def _cos_pow_integrate(n, x):
    if False:
        for i in range(10):
            print('nop')
    if n > 0:
        if n == 1:
            return sin(x)
        return Rational(1, n) * sin(x) * cos(x) ** (n - 1) + Rational(n - 1, n) * _cos_pow_integrate(n - 2, x)
    if n < 0:
        if n == -1:
            return trigintegrate(1 / cos(x), x)
        return Rational(-1, n + 1) * sin(x) * cos(x) ** (n + 1) + Rational(n + 2, n + 1) * _cos_pow_integrate(n + 2, x)
    else:
        return x