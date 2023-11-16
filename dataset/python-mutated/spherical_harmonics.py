from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions import assoc_legendre
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
_x = Dummy('x')

class Ynm(Function):
    """
    Spherical harmonics defined as

    .. math::
        Y_n^m(\\theta, \\varphi) := \\sqrt{\\frac{(2n+1)(n-m)!}{4\\pi(n+m)!}}
                                  \\exp(i m \\varphi)
                                  \\mathrm{P}_n^m\\left(\\cos(\\theta)\\right)

    Explanation
    ===========

    ``Ynm()`` gives the spherical harmonic function of order $n$ and $m$
    in $\\theta$ and $\\varphi$, $Y_n^m(\\theta, \\varphi)$. The four
    parameters are as follows: $n \\geq 0$ an integer and $m$ an integer
    such that $-n \\leq m \\leq n$ holds. The two angles are real-valued
    with $\\theta \\in [0, \\pi]$ and $\\varphi \\in [0, 2\\pi]$.

    Examples
    ========

    >>> from sympy import Ynm, Symbol, simplify
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> Ynm(n, m, theta, phi)
    Ynm(n, m, theta, phi)

    Several symmetries are known, for the order:

    >>> Ynm(n, -m, theta, phi)
    (-1)**m*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    As well as for the angles:

    >>> Ynm(n, m, -theta, phi)
    Ynm(n, m, theta, phi)

    >>> Ynm(n, m, theta, -phi)
    exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    For specific integers $n$ and $m$ we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Ynm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))

    >>> simplify(Ynm(1, -1, theta, phi).expand(func=True))
    sqrt(6)*exp(-I*phi)*sin(theta)/(4*sqrt(pi))

    >>> simplify(Ynm(1, 0, theta, phi).expand(func=True))
    sqrt(3)*cos(theta)/(2*sqrt(pi))

    >>> simplify(Ynm(1, 1, theta, phi).expand(func=True))
    -sqrt(6)*exp(I*phi)*sin(theta)/(4*sqrt(pi))

    >>> simplify(Ynm(2, -2, theta, phi).expand(func=True))
    sqrt(30)*exp(-2*I*phi)*sin(theta)**2/(8*sqrt(pi))

    >>> simplify(Ynm(2, -1, theta, phi).expand(func=True))
    sqrt(30)*exp(-I*phi)*sin(2*theta)/(8*sqrt(pi))

    >>> simplify(Ynm(2, 0, theta, phi).expand(func=True))
    sqrt(5)*(3*cos(theta)**2 - 1)/(4*sqrt(pi))

    >>> simplify(Ynm(2, 1, theta, phi).expand(func=True))
    -sqrt(30)*exp(I*phi)*sin(2*theta)/(8*sqrt(pi))

    >>> simplify(Ynm(2, 2, theta, phi).expand(func=True))
    sqrt(30)*exp(2*I*phi)*sin(theta)**2/(8*sqrt(pi))

    We can differentiate the functions with respect
    to both angles:

    >>> from sympy import Ynm, Symbol, diff
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> diff(Ynm(n, m, theta, phi), theta)
    m*cot(theta)*Ynm(n, m, theta, phi) + sqrt((-m + n)*(m + n + 1))*exp(-I*phi)*Ynm(n, m + 1, theta, phi)

    >>> diff(Ynm(n, m, theta, phi), phi)
    I*m*Ynm(n, m, theta, phi)

    Further we can compute the complex conjugation:

    >>> from sympy import Ynm, Symbol, conjugate
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> conjugate(Ynm(n, m, theta, phi))
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    To get back the well known expressions in spherical
    coordinates, we use full expansion:

    >>> from sympy import Ynm, Symbol, expand_func
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> expand_func(Ynm(n, m, theta, phi))
    sqrt((2*n + 1)*factorial(-m + n)/factorial(m + n))*exp(I*m*phi)*assoc_legendre(n, m, cos(theta))/(2*sqrt(pi))

    See Also
    ========

    Ynm_c, Znm

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/
    .. [4] https://dlmf.nist.gov/14.30

    """

    @classmethod
    def eval(cls, n, m, theta, phi):
        if False:
            for i in range(10):
                print('nop')
        if m.could_extract_minus_sign():
            m = -m
            return S.NegativeOne ** m * exp(-2 * I * m * phi) * Ynm(n, m, theta, phi)
        if theta.could_extract_minus_sign():
            theta = -theta
            return Ynm(n, m, theta, phi)
        if phi.could_extract_minus_sign():
            phi = -phi
            return exp(-2 * I * m * phi) * Ynm(n, m, theta, phi)

    def _eval_expand_func(self, **hints):
        if False:
            return 10
        (n, m, theta, phi) = self.args
        rv = sqrt((2 * n + 1) / (4 * pi) * factorial(n - m) / factorial(n + m)) * exp(I * m * phi) * assoc_legendre(n, m, cos(theta))
        return rv.subs(sqrt(-cos(theta) ** 2 + 1), sin(theta))

    def fdiff(self, argindex=4):
        if False:
            i = 10
            return i + 15
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            (n, m, theta, phi) = self.args
            return m * cot(theta) * Ynm(n, m, theta, phi) + sqrt((n - m) * (n + m + 1)) * exp(-I * phi) * Ynm(n, m + 1, theta, phi)
        elif argindex == 4:
            (n, m, theta, phi) = self.args
            return I * m * Ynm(n, m, theta, phi)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_polynomial(self, n, m, theta, phi, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.expand(func=True)

    def _eval_rewrite_as_sin(self, n, m, theta, phi, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.rewrite(cos)

    def _eval_rewrite_as_cos(self, n, m, theta, phi, **kwargs):
        if False:
            i = 10
            return i + 15
        from sympy.simplify import simplify, trigsimp
        term = simplify(self.expand(func=True))
        term = term.xreplace({Abs(sin(theta)): sin(theta)})
        return simplify(trigsimp(term))

    def _eval_conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        (n, m, theta, phi) = self.args
        return S.NegativeOne ** m * self.func(n, -m, theta, phi)

    def as_real_imag(self, deep=True, **hints):
        if False:
            print('Hello World!')
        (n, m, theta, phi) = self.args
        re = sqrt((2 * n + 1) / (4 * pi) * factorial(n - m) / factorial(n + m)) * cos(m * phi) * assoc_legendre(n, m, cos(theta))
        im = sqrt((2 * n + 1) / (4 * pi) * factorial(n - m) / factorial(n + m)) * sin(m * phi) * assoc_legendre(n, m, cos(theta))
        return (re, im)

    def _eval_evalf(self, prec):
        if False:
            print('Hello World!')
        from mpmath import mp, workprec
        n = self.args[0]._to_mpmath(prec)
        m = self.args[1]._to_mpmath(prec)
        theta = self.args[2]._to_mpmath(prec)
        phi = self.args[3]._to_mpmath(prec)
        with workprec(prec):
            res = mp.spherharm(n, m, theta, phi)
        return Expr._from_mpmath(res, prec)

def Ynm_c(n, m, theta, phi):
    if False:
        for i in range(10):
            print('nop')
    '\n    Conjugate spherical harmonics defined as\n\n    .. math::\n        \\overline{Y_n^m(\\theta, \\varphi)} := (-1)^m Y_n^{-m}(\\theta, \\varphi).\n\n    Examples\n    ========\n\n    >>> from sympy import Ynm_c, Symbol, simplify\n    >>> from sympy.abc import n,m\n    >>> theta = Symbol("theta")\n    >>> phi = Symbol("phi")\n    >>> Ynm_c(n, m, theta, phi)\n    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)\n    >>> Ynm_c(n, m, -theta, phi)\n    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)\n\n    For specific integers $n$ and $m$ we can evaluate the harmonics\n    to more useful expressions:\n\n    >>> simplify(Ynm_c(0, 0, theta, phi).expand(func=True))\n    1/(2*sqrt(pi))\n    >>> simplify(Ynm_c(1, -1, theta, phi).expand(func=True))\n    sqrt(6)*exp(I*(-phi + 2*conjugate(phi)))*sin(theta)/(4*sqrt(pi))\n\n    See Also\n    ========\n\n    Ynm, Znm\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics\n    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html\n    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/\n\n    '
    return conjugate(Ynm(n, m, theta, phi))

class Znm(Function):
    """
    Real spherical harmonics defined as

    .. math::

        Z_n^m(\\theta, \\varphi) :=
        \\begin{cases}
          \\frac{Y_n^m(\\theta, \\varphi) + \\overline{Y_n^m(\\theta, \\varphi)}}{\\sqrt{2}} &\\quad m > 0 \\\\
          Y_n^m(\\theta, \\varphi) &\\quad m = 0 \\\\
          \\frac{Y_n^m(\\theta, \\varphi) - \\overline{Y_n^m(\\theta, \\varphi)}}{i \\sqrt{2}} &\\quad m < 0 \\\\
        \\end{cases}

    which gives in simplified form

    .. math::

        Z_n^m(\\theta, \\varphi) =
        \\begin{cases}
          \\frac{Y_n^m(\\theta, \\varphi) + (-1)^m Y_n^{-m}(\\theta, \\varphi)}{\\sqrt{2}} &\\quad m > 0 \\\\
          Y_n^m(\\theta, \\varphi) &\\quad m = 0 \\\\
          \\frac{Y_n^m(\\theta, \\varphi) - (-1)^m Y_n^{-m}(\\theta, \\varphi)}{i \\sqrt{2}} &\\quad m < 0 \\\\
        \\end{cases}

    Examples
    ========

    >>> from sympy import Znm, Symbol, simplify
    >>> from sympy.abc import n, m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Znm(n, m, theta, phi)
    Znm(n, m, theta, phi)

    For specific integers n and m we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Znm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Znm(1, 1, theta, phi).expand(func=True))
    -sqrt(3)*sin(theta)*cos(phi)/(2*sqrt(pi))
    >>> simplify(Znm(2, 1, theta, phi).expand(func=True))
    -sqrt(15)*sin(2*theta)*cos(phi)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Ynm_c

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """

    @classmethod
    def eval(cls, n, m, theta, phi):
        if False:
            i = 10
            return i + 15
        if m.is_positive:
            zz = (Ynm(n, m, theta, phi) + Ynm_c(n, m, theta, phi)) / sqrt(2)
            return zz
        elif m.is_zero:
            return Ynm(n, m, theta, phi)
        elif m.is_negative:
            zz = (Ynm(n, m, theta, phi) - Ynm_c(n, m, theta, phi)) / (sqrt(2) * I)
            return zz