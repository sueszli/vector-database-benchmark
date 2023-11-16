from typing import Type
from sympy import Interval, numer, Rational, solveset
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions import Abs
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import Matrix, ImmutableMatrix, ImmutableDenseMatrix, eye, ShapeError, zeros
from sympy.functions.elementary.exponential import exp, log
from sympy.matrices.expressions import MatMul, MatAdd
from sympy.polys import Poly, rootof
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel, degree
from sympy.series import limit
from sympy.utilities.misc import filldedent
from mpmath.libmp.libmpf import prec_to_dps
__all__ = ['TransferFunction', 'Series', 'MIMOSeries', 'Parallel', 'MIMOParallel', 'Feedback', 'MIMOFeedback', 'TransferFunctionMatrix', 'StateSpace', 'gbt', 'bilinear', 'forward_diff', 'backward_diff', 'phase_margin', 'gain_margin']

def _roots(poly, var):
    if False:
        return 10
    ' like roots, but works on higher-order polynomials. '
    r = roots(poly, var, multiple=True)
    n = degree(poly)
    if len(r) != n:
        r = [rootof(poly, var, k) for k in range(n)]
    return r

def gbt(tf, sample_per, alpha):
    if False:
        i = 10
        return i + 15
    '\n    Returns falling coefficients of H(z) from numerator and denominator.\n\n    Explanation\n    ===========\n\n    Where H(z) is the corresponding discretized transfer function,\n    discretized with the generalised bilinear transformation method.\n    H(z) is obtained from the continuous transfer function H(s)\n    by substituting $s(z) = \\frac{z-1}{T(\\alpha z + (1-\\alpha))}$ into H(s), where T is the\n    sample period.\n    Coefficients are falling, i.e. $H(z) = \\frac{az+b}{cz+d}$ is returned\n    as [a, b], [c, d].\n\n    Examples\n    ========\n\n    >>> from sympy.physics.control.lti import TransferFunction, gbt\n    >>> from sympy.abc import s, L, R, T\n\n    >>> tf = TransferFunction(1, s*L + R, s)\n    >>> numZ, denZ = gbt(tf, T, 0.5)\n    >>> numZ\n    [T/(2*(L + R*T/2)), T/(2*(L + R*T/2))]\n    >>> denZ\n    [1, (-L + R*T/2)/(L + R*T/2)]\n\n    >>> numZ, denZ = gbt(tf, T, 0)\n    >>> numZ\n    [T/L]\n    >>> denZ\n    [1, (-L + R*T)/L]\n\n    >>> numZ, denZ = gbt(tf, T, 1)\n    >>> numZ\n    [T/(L + R*T), 0]\n    >>> denZ\n    [1, -L/(L + R*T)]\n\n    >>> numZ, denZ = gbt(tf, T, 0.3)\n    >>> numZ\n    [3*T/(10*(L + 3*R*T/10)), 7*T/(10*(L + 3*R*T/10))]\n    >>> denZ\n    [1, (-L + 7*R*T/10)/(L + 3*R*T/10)]\n\n    References\n    ==========\n\n    .. [1] https://www.polyu.edu.hk/ama/profile/gfzhang/Research/ZCC09_IJC.pdf\n    '
    if not tf.is_SISO:
        raise NotImplementedError('Not implemented for MIMO systems.')
    T = sample_per
    s = tf.var
    z = s
    np = tf.num.as_poly(s).all_coeffs()
    dp = tf.den.as_poly(s).all_coeffs()
    alpha = Rational(alpha).limit_denominator(1000)
    N = max(len(np), len(dp)) - 1
    num = Add(*[T ** (N - i) * c * (z - 1) ** i * (alpha * z + 1 - alpha) ** (N - i) for (c, i) in zip(np[::-1], range(len(np)))])
    den = Add(*[T ** (N - i) * c * (z - 1) ** i * (alpha * z + 1 - alpha) ** (N - i) for (c, i) in zip(dp[::-1], range(len(dp)))])
    num_coefs = num.as_poly(z).all_coeffs()
    den_coefs = den.as_poly(z).all_coeffs()
    para = den_coefs[0]
    num_coefs = [coef / para for coef in num_coefs]
    den_coefs = [coef / para for coef in den_coefs]
    return (num_coefs, den_coefs)

def bilinear(tf, sample_per):
    if False:
        i = 10
        return i + 15
    '\n    Returns falling coefficients of H(z) from numerator and denominator.\n\n    Explanation\n    ===========\n\n    Where H(z) is the corresponding discretized transfer function,\n    discretized with the bilinear transform method.\n    H(z) is obtained from the continuous transfer function H(s)\n    by substituting $s(z) = \\frac{2}{T}\\frac{z-1}{z+1}$ into H(s), where T is the\n    sample period.\n    Coefficients are falling, i.e. $H(z) = \\frac{az+b}{cz+d}$ is returned\n    as [a, b], [c, d].\n\n    Examples\n    ========\n\n    >>> from sympy.physics.control.lti import TransferFunction, bilinear\n    >>> from sympy.abc import s, L, R, T\n\n    >>> tf = TransferFunction(1, s*L + R, s)\n    >>> numZ, denZ = bilinear(tf, T)\n    >>> numZ\n    [T/(2*(L + R*T/2)), T/(2*(L + R*T/2))]\n    >>> denZ\n    [1, (-L + R*T/2)/(L + R*T/2)]\n    '
    return gbt(tf, sample_per, S.Half)

def forward_diff(tf, sample_per):
    if False:
        print('Hello World!')
    '\n    Returns falling coefficients of H(z) from numerator and denominator.\n\n    Explanation\n    ===========\n\n    Where H(z) is the corresponding discretized transfer function,\n    discretized with the forward difference transform method.\n    H(z) is obtained from the continuous transfer function H(s)\n    by substituting $s(z) = \\frac{z-1}{T}$ into H(s), where T is the\n    sample period.\n    Coefficients are falling, i.e. $H(z) = \\frac{az+b}{cz+d}$ is returned\n    as [a, b], [c, d].\n\n    Examples\n    ========\n\n    >>> from sympy.physics.control.lti import TransferFunction, forward_diff\n    >>> from sympy.abc import s, L, R, T\n\n    >>> tf = TransferFunction(1, s*L + R, s)\n    >>> numZ, denZ = forward_diff(tf, T)\n    >>> numZ\n    [T/L]\n    >>> denZ\n    [1, (-L + R*T)/L]\n    '
    return gbt(tf, sample_per, S.Zero)

def backward_diff(tf, sample_per):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns falling coefficients of H(z) from numerator and denominator.\n\n    Explanation\n    ===========\n\n    Where H(z) is the corresponding discretized transfer function,\n    discretized with the backward difference transform method.\n    H(z) is obtained from the continuous transfer function H(s)\n    by substituting $s(z) =  \\frac{z-1}{Tz}$ into H(s), where T is the\n    sample period.\n    Coefficients are falling, i.e. $H(z) = \\frac{az+b}{cz+d}$ is returned\n    as [a, b], [c, d].\n\n    Examples\n    ========\n\n    >>> from sympy.physics.control.lti import TransferFunction, backward_diff\n    >>> from sympy.abc import s, L, R, T\n\n    >>> tf = TransferFunction(1, s*L + R, s)\n    >>> numZ, denZ = backward_diff(tf, T)\n    >>> numZ\n    [T/(L + R*T), 0]\n    >>> denZ\n    [1, -L/(L + R*T)]\n    '
    return gbt(tf, sample_per, S.One)

def phase_margin(system):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the phase margin of a continuous time system.\n    Only applicable to Transfer Functions which can generate valid bode plots.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When time delay terms are present in the system.\n\n    ValueError\n        When a SISO LTI system is not passed.\n\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.control import TransferFunction, phase_margin\n    >>> from sympy.abc import s\n\n    >>> tf = TransferFunction(1, s**3 + 2*s**2 + s, s)\n    >>> phase_margin(tf)\n    180*(-pi + atan((-1 + (-2*18**(1/3)/(9 + sqrt(93))**(1/3) + 12**(1/3)*(9 + sqrt(93))**(1/3))**2/36)/(-12**(1/3)*(9 + sqrt(93))**(1/3)/3 + 2*18**(1/3)/(3*(9 + sqrt(93))**(1/3)))))/pi + 180\n    >>> phase_margin(tf).n()\n    21.3863897518751\n\n    >>> tf1 = TransferFunction(s**3, s**2 + 5*s, s)\n    >>> phase_margin(tf1)\n    -180 + 180*(atan(sqrt(2)*(-51/10 - sqrt(101)/10)*sqrt(1 + sqrt(101))/(2*(sqrt(101)/2 + 51/2))) + pi)/pi\n    >>> phase_margin(tf1).n()\n    -25.1783920627277\n\n    >>> tf2 = TransferFunction(1, s + 1, s)\n    >>> phase_margin(tf2)\n    -180\n\n    See Also\n    ========\n\n    gain_margin\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Phase_margin\n\n    '
    from sympy.functions import arg
    if not isinstance(system, SISOLinearTimeInvariant):
        raise ValueError('Margins are only applicable for SISO LTI systems.')
    _w = Dummy('w', real=True)
    repl = I * _w
    expr = system.to_expr()
    len_free_symbols = len(expr.free_symbols)
    if expr.has(exp):
        raise NotImplementedError('Margins for systems with Time delay terms are not supported.')
    elif len_free_symbols > 1:
        raise ValueError('Extra degree of freedom found. Make sure that there are no free symbols in the dynamical system other than the variable of Laplace transform.')
    w_expr = expr.subs({system.var: repl})
    mag = 20 * log(Abs(w_expr), 10)
    mag_sol = list(solveset(mag, _w, Interval(0, oo, left_open=True)))
    if len(mag_sol) == 0:
        pm = S(-180)
    else:
        wcp = mag_sol[0]
        pm = ((arg(w_expr) * S(180) / pi).subs({_w: wcp}) + S(180)) % 360
    if pm >= 180:
        pm = pm - 360
    return pm

def gain_margin(system):
    if False:
        return 10
    '\n    Returns the gain margin of a continuous time system.\n    Only applicable to Transfer Functions which can generate valid bode plots.\n\n    Raises\n    ======\n\n    NotImplementedError\n        When time delay terms are present in the system.\n\n    ValueError\n        When a SISO LTI system is not passed.\n\n        When more than one free symbol is present in the system.\n        The only variable in the transfer function should be\n        the variable of the Laplace transform.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.control import TransferFunction, gain_margin\n    >>> from sympy.abc import s\n\n    >>> tf = TransferFunction(1, s**3 + 2*s**2 + s, s)\n    >>> gain_margin(tf)\n    20*log(2)/log(10)\n    >>> gain_margin(tf).n()\n    6.02059991327962\n\n    >>> tf1 = TransferFunction(s**3, s**2 + 5*s, s)\n    >>> gain_margin(tf1)\n    oo\n\n    See Also\n    ========\n\n    phase_margin\n\n    References\n    ==========\n\n    https://en.wikipedia.org/wiki/Bode_plot\n\n    '
    if not isinstance(system, SISOLinearTimeInvariant):
        raise ValueError('Margins are only applicable for SISO LTI systems.')
    _w = Dummy('w', real=True)
    repl = I * _w
    expr = system.to_expr()
    len_free_symbols = len(expr.free_symbols)
    if expr.has(exp):
        raise NotImplementedError('Margins for systems with Time delay terms are not supported.')
    elif len_free_symbols > 1:
        raise ValueError('Extra degree of freedom found. Make sure that there are no free symbols in the dynamical system other than the variable of Laplace transform.')
    w_expr = expr.subs({system.var: repl})
    mag = 20 * log(Abs(w_expr), 10)
    phase = w_expr
    phase_sol = list(solveset(numer(phase.as_real_imag()[1].cancel()), _w, Interval(0, oo, left_open=True)))
    if len(phase_sol) == 0:
        gm = oo
    else:
        wcg = phase_sol[0]
        gm = -mag.subs({_w: wcg})
    return gm

class LinearTimeInvariant(Basic, EvalfMixin):
    """A common class for all the Linear Time-Invariant Dynamical Systems."""
    _clstype: Type

    def __new__(cls, *system, **kwargs):
        if False:
            print('Hello World!')
        if cls is LinearTimeInvariant:
            raise NotImplementedError('The LTICommon class is not meant to be used directly.')
        return super(LinearTimeInvariant, cls).__new__(cls, *system, **kwargs)

    @classmethod
    def _check_args(cls, args):
        if False:
            print('Hello World!')
        if not args:
            raise ValueError('At least 1 argument must be passed.')
        if not all((isinstance(arg, cls._clstype) for arg in args)):
            raise TypeError(f'All arguments must be of type {cls._clstype}.')
        var_set = {arg.var for arg in args}
        if len(var_set) != 1:
            raise ValueError(filldedent(f'\n                All transfer functions should use the same complex variable\n                of the Laplace transform. {len(var_set)} different\n                values found.'))

    @property
    def is_SISO(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns `True` if the passed LTI system is SISO else returns False.'
        return self._is_SISO

class SISOLinearTimeInvariant(LinearTimeInvariant):
    """A common class for all the SISO Linear Time-Invariant Dynamical Systems."""
    _is_SISO = True

class MIMOLinearTimeInvariant(LinearTimeInvariant):
    """A common class for all the MIMO Linear Time-Invariant Dynamical Systems."""
    _is_SISO = False
SISOLinearTimeInvariant._clstype = SISOLinearTimeInvariant
MIMOLinearTimeInvariant._clstype = MIMOLinearTimeInvariant

def _check_other_SISO(func):
    if False:
        for i in range(10):
            print('nop')

    def wrapper(*args, **kwargs):
        if False:
            return 10
        if not isinstance(args[-1], SISOLinearTimeInvariant):
            return NotImplemented
        else:
            return func(*args, **kwargs)
    return wrapper

def _check_other_MIMO(func):
    if False:
        return 10

    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(args[-1], MIMOLinearTimeInvariant):
            return NotImplemented
        else:
            return func(*args, **kwargs)
    return wrapper

class TransferFunction(SISOLinearTimeInvariant):
    """
    A class for representing LTI (Linear, time-invariant) systems that can be strictly described
    by ratio of polynomials in the Laplace transform complex variable. The arguments
    are ``num``, ``den``, and ``var``, where ``num`` and ``den`` are numerator and
    denominator polynomials of the ``TransferFunction`` respectively, and the third argument is
    a complex variable of the Laplace transform used by these polynomials of the transfer function.
    ``num`` and ``den`` can be either polynomials or numbers, whereas ``var``
    has to be a :py:class:`~.Symbol`.

    Explanation
    ===========

    Generally, a dynamical system representing a physical model can be described in terms of Linear
    Ordinary Differential Equations like -

            $\\small{b_{m}y^{\\left(m\\right)}+b_{m-1}y^{\\left(m-1\\right)}+\\dots+b_{1}y^{\\left(1\\right)}+b_{0}y=
            a_{n}x^{\\left(n\\right)}+a_{n-1}x^{\\left(n-1\\right)}+\\dots+a_{1}x^{\\left(1\\right)}+a_{0}x}$

    Here, $x$ is the input signal and $y$ is the output signal and superscript on both is the order of derivative
    (not exponent). Derivative is taken with respect to the independent variable, $t$. Also, generally $m$ is greater
    than $n$.

    It is not feasible to analyse the properties of such systems in their native form therefore, we use
    mathematical tools like Laplace transform to get a better perspective. Taking the Laplace transform
    of both the sides in the equation (at zero initial conditions), we get -

            $\\small{\\mathcal{L}[b_{m}y^{\\left(m\\right)}+b_{m-1}y^{\\left(m-1\\right)}+\\dots+b_{1}y^{\\left(1\\right)}+b_{0}y]=
            \\mathcal{L}[a_{n}x^{\\left(n\\right)}+a_{n-1}x^{\\left(n-1\\right)}+\\dots+a_{1}x^{\\left(1\\right)}+a_{0}x]}$

    Using the linearity property of Laplace transform and also considering zero initial conditions
    (i.e. $\\small{y(0^{-}) = 0}$, $\\small{y'(0^{-}) = 0}$ and so on), the equation
    above gets translated to -

            $\\small{b_{m}\\mathcal{L}[y^{\\left(m\\right)}]+\\dots+b_{1}\\mathcal{L}[y^{\\left(1\\right)}]+b_{0}\\mathcal{L}[y]=
            a_{n}\\mathcal{L}[x^{\\left(n\\right)}]+\\dots+a_{1}\\mathcal{L}[x^{\\left(1\\right)}]+a_{0}\\mathcal{L}[x]}$

    Now, applying Derivative property of Laplace transform,

            $\\small{b_{m}s^{m}\\mathcal{L}[y]+\\dots+b_{1}s\\mathcal{L}[y]+b_{0}\\mathcal{L}[y]=
            a_{n}s^{n}\\mathcal{L}[x]+\\dots+a_{1}s\\mathcal{L}[x]+a_{0}\\mathcal{L}[x]}$

    Here, the superscript on $s$ is **exponent**. Note that the zero initial conditions assumption, mentioned above, is very important
    and cannot be ignored otherwise the dynamical system cannot be considered time-independent and the simplified equation above
    cannot be reached.

    Collecting $\\mathcal{L}[y]$ and $\\mathcal{L}[x]$ terms from both the sides and taking the ratio
    $\\frac{ \\mathcal{L}\\left\\{y\\right\\} }{ \\mathcal{L}\\left\\{x\\right\\} }$, we get the typical rational form of transfer
    function.

    The numerator of the transfer function is, therefore, the Laplace transform of the output signal
    (The signals are represented as functions of time) and similarly, the denominator
    of the transfer function is the Laplace transform of the input signal. It is also a convention
    to denote the input and output signal's Laplace transform with capital alphabets like shown below.

            $H(s) = \\frac{Y(s)}{X(s)} = \\frac{ \\mathcal{L}\\left\\{y(t)\\right\\} }{ \\mathcal{L}\\left\\{x(t)\\right\\} }$

    $s$, also known as complex frequency, is a complex variable in the Laplace domain. It corresponds to the
    equivalent variable $t$, in the time domain. Transfer functions are sometimes also referred to as the Laplace
    transform of the system's impulse response. Transfer function, $H$, is represented as a rational
    function in $s$ like,

            $H(s) =\\ \\frac{a_{n}s^{n}+a_{n-1}s^{n-1}+\\dots+a_{1}s+a_{0}}{b_{m}s^{m}+b_{m-1}s^{m-1}+\\dots+b_{1}s+b_{0}}$

    Parameters
    ==========

    num : Expr, Number
        The numerator polynomial of the transfer function.
    den : Expr, Number
        The denominator polynomial of the transfer function.
    var : Symbol
        Complex variable of the Laplace transform used by the
        polynomials of the transfer function.

    Raises
    ======

    TypeError
        When ``var`` is not a Symbol or when ``num`` or ``den`` is not a
        number or a polynomial.
    ValueError
        When ``den`` is zero.

    Examples
    ========

    >>> from sympy.abc import s, p, a
    >>> from sympy.physics.control.lti import TransferFunction
    >>> tf1 = TransferFunction(s + a, s**2 + s + 1, s)
    >>> tf1
    TransferFunction(a + s, s**2 + s + 1, s)
    >>> tf1.num
    a + s
    >>> tf1.den
    s**2 + s + 1
    >>> tf1.var
    s
    >>> tf1.args
    (a + s, s**2 + s + 1, s)

    Any complex variable can be used for ``var``.

    >>> tf2 = TransferFunction(a*p**3 - a*p**2 + s*p, p + a**2, p)
    >>> tf2
    TransferFunction(a*p**3 - a*p**2 + p*s, a**2 + p, p)
    >>> tf3 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)
    >>> tf3
    TransferFunction((p - 1)*(p + 3), (p - 1)*(p + 5), p)

    To negate a transfer function the ``-`` operator can be prepended:

    >>> tf4 = TransferFunction(-a + s, p**2 + s, p)
    >>> -tf4
    TransferFunction(a - s, p**2 + s, p)
    >>> tf5 = TransferFunction(s**4 - 2*s**3 + 5*s + 4, s + 4, s)
    >>> -tf5
    TransferFunction(-s**4 + 2*s**3 - 5*s - 4, s + 4, s)

    You can use a float or an integer (or other constants) as numerator and denominator:

    >>> tf6 = TransferFunction(1/2, 4, s)
    >>> tf6.num
    0.500000000000000
    >>> tf6.den
    4
    >>> tf6.var
    s
    >>> tf6.args
    (0.5, 4, s)

    You can take the integer power of a transfer function using the ``**`` operator:

    >>> tf7 = TransferFunction(s + a, s - a, s)
    >>> tf7**3
    TransferFunction((a + s)**3, (-a + s)**3, s)
    >>> tf7**0
    TransferFunction(1, 1, s)
    >>> tf8 = TransferFunction(p + 4, p - 3, p)
    >>> tf8**-1
    TransferFunction(p - 3, p + 4, p)

    Addition, subtraction, and multiplication of transfer functions can form
    unevaluated ``Series`` or ``Parallel`` objects.

    >>> tf9 = TransferFunction(s + 1, s**2 + s + 1, s)
    >>> tf10 = TransferFunction(s - p, s + 3, s)
    >>> tf11 = TransferFunction(4*s**2 + 2*s - 4, s - 1, s)
    >>> tf12 = TransferFunction(1 - s, s**2 + 4, s)
    >>> tf9 + tf10
    Parallel(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-p + s, s + 3, s))
    >>> tf10 - tf11
    Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(-4*s**2 - 2*s + 4, s - 1, s))
    >>> tf9 * tf10
    Series(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-p + s, s + 3, s))
    >>> tf10 - (tf9 + tf12)
    Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(-s - 1, s**2 + s + 1, s), TransferFunction(s - 1, s**2 + 4, s))
    >>> tf10 - (tf9 * tf12)
    Parallel(TransferFunction(-p + s, s + 3, s), Series(TransferFunction(-1, 1, s), TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(1 - s, s**2 + 4, s)))
    >>> tf11 * tf10 * tf9
    Series(TransferFunction(4*s**2 + 2*s - 4, s - 1, s), TransferFunction(-p + s, s + 3, s), TransferFunction(s + 1, s**2 + s + 1, s))
    >>> tf9 * tf11 + tf10 * tf12
    Parallel(Series(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(4*s**2 + 2*s - 4, s - 1, s)), Series(TransferFunction(-p + s, s + 3, s), TransferFunction(1 - s, s**2 + 4, s)))
    >>> (tf9 + tf12) * (tf10 + tf11)
    Series(Parallel(TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(1 - s, s**2 + 4, s)), Parallel(TransferFunction(-p + s, s + 3, s), TransferFunction(4*s**2 + 2*s - 4, s - 1, s)))

    These unevaluated ``Series`` or ``Parallel`` objects can convert into the
    resultant transfer function using ``.doit()`` method or by ``.rewrite(TransferFunction)``.

    >>> ((tf9 + tf10) * tf12).doit()
    TransferFunction((1 - s)*((-p + s)*(s**2 + s + 1) + (s + 1)*(s + 3)), (s + 3)*(s**2 + 4)*(s**2 + s + 1), s)
    >>> (tf9 * tf10 - tf11 * tf12).rewrite(TransferFunction)
    TransferFunction(-(1 - s)*(s + 3)*(s**2 + s + 1)*(4*s**2 + 2*s - 4) + (-p + s)*(s - 1)*(s + 1)*(s**2 + 4), (s - 1)*(s + 3)*(s**2 + 4)*(s**2 + s + 1), s)

    See Also
    ========

    Feedback, Series, Parallel

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Transfer_function
    .. [2] https://en.wikipedia.org/wiki/Laplace_transform

    """

    def __new__(cls, num, den, var):
        if False:
            while True:
                i = 10
        (num, den) = (_sympify(num), _sympify(den))
        if not isinstance(var, Symbol):
            raise TypeError('Variable input must be a Symbol.')
        if den == 0:
            raise ValueError('TransferFunction cannot have a zero denominator.')
        if (isinstance(num, Expr) and num.has(Symbol) or num.is_number) and (isinstance(den, Expr) and den.has(Symbol) or den.is_number):
            return super(TransferFunction, cls).__new__(cls, num, den, var)
        else:
            raise TypeError('Unsupported type for numerator or denominator of TransferFunction.')

    @classmethod
    def from_rational_expression(cls, expr, var=None):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new ``TransferFunction`` efficiently from a rational expression.\n\n        Parameters\n        ==========\n\n        expr : Expr, Number\n            The rational expression representing the ``TransferFunction``.\n        var : Symbol, optional\n            Complex variable of the Laplace transform used by the\n            polynomials of the transfer function.\n\n        Raises\n        ======\n\n        ValueError\n            When ``expr`` is of type ``Number`` and optional parameter ``var``\n            is not passed.\n\n            When ``expr`` has more than one variables and an optional parameter\n            ``var`` is not passed.\n        ZeroDivisionError\n            When denominator of ``expr`` is zero or it has ``ComplexInfinity``\n            in its numerator.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> expr1 = (s + 5)/(3*s**2 + 2*s + 1)\n        >>> tf1 = TransferFunction.from_rational_expression(expr1)\n        >>> tf1\n        TransferFunction(s + 5, 3*s**2 + 2*s + 1, s)\n        >>> expr2 = (a*p**3 - a*p**2 + s*p)/(p + a**2)  # Expr with more than one variables\n        >>> tf2 = TransferFunction.from_rational_expression(expr2, p)\n        >>> tf2\n        TransferFunction(a*p**3 - a*p**2 + p*s, a**2 + p, p)\n\n        In case of conflict between two or more variables in a expression, SymPy will\n        raise a ``ValueError``, if ``var`` is not passed by the user.\n\n        >>> tf = TransferFunction.from_rational_expression((a + a*s)/(s**2 + s + 1))\n        Traceback (most recent call last):\n        ...\n        ValueError: Conflicting values found for positional argument `var` ({a, s}). Specify it manually.\n\n        This can be corrected by specifying the ``var`` parameter manually.\n\n        >>> tf = TransferFunction.from_rational_expression((a + a*s)/(s**2 + s + 1), s)\n        >>> tf\n        TransferFunction(a*s + a, s**2 + s + 1, s)\n\n        ``var`` also need to be specified when ``expr`` is a ``Number``\n\n        >>> tf3 = TransferFunction.from_rational_expression(10, s)\n        >>> tf3\n        TransferFunction(10, 1, s)\n\n        '
        expr = _sympify(expr)
        if var is None:
            _free_symbols = expr.free_symbols
            _len_free_symbols = len(_free_symbols)
            if _len_free_symbols == 1:
                var = list(_free_symbols)[0]
            elif _len_free_symbols == 0:
                raise ValueError(filldedent('\n                    Positional argument `var` not found in the\n                    TransferFunction defined. Specify it manually.'))
            else:
                raise ValueError(filldedent('\n                    Conflicting values found for positional argument `var` ({}).\n                    Specify it manually.'.format(_free_symbols)))
        (_num, _den) = expr.as_numer_denom()
        if _den == 0 or _num.has(S.ComplexInfinity):
            raise ZeroDivisionError('TransferFunction cannot have a zero denominator.')
        return cls(_num, _den, var)

    @classmethod
    def from_coeff_lists(cls, num_list, den_list, var):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new ``TransferFunction`` efficiently from a list of coefficients.\n\n        Parameters\n        ==========\n\n        num_list : Sequence\n            Sequence comprising of numerator coefficients.\n        den_list : Sequence\n            Sequence comprising of denominator coefficients.\n        var : Symbol\n            Complex variable of the Laplace transform used by the\n            polynomials of the transfer function.\n\n        Raises\n        ======\n\n        ZeroDivisionError\n            When the constructed denominator is zero.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> num = [1, 0, 2]\n        >>> den = [3, 2, 2, 1]\n        >>> tf = TransferFunction.from_coeff_lists(num, den, s)\n        >>> tf\n        TransferFunction(s**2 + 2, 3*s**3 + 2*s**2 + 2*s + 1, s)\n\n        # Create a Transfer Function with more than one variable\n        >>> tf1 = TransferFunction.from_coeff_lists([p, 1], [2*p, 0, 4], s)\n        >>> tf1\n        TransferFunction(p*s + 1, 2*p*s**2 + 4, s)\n\n        '
        num_list = num_list[::-1]
        den_list = den_list[::-1]
        num_var_powers = [var ** i for i in range(len(num_list))]
        den_var_powers = [var ** i for i in range(len(den_list))]
        _num = sum((coeff * var_power for (coeff, var_power) in zip(num_list, num_var_powers)))
        _den = sum((coeff * var_power for (coeff, var_power) in zip(den_list, den_var_powers)))
        if _den == 0:
            raise ZeroDivisionError('TransferFunction cannot have a zero denominator.')
        return cls(_num, _den, var)

    @classmethod
    def from_zpk(cls, zeros, poles, gain, var):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new ``TransferFunction`` from given zeros, poles and gain.\n\n        Parameters\n        ==========\n\n        zeros : Sequence\n            Sequence comprising of zeros of transfer function.\n        poles : Sequence\n            Sequence comprising of poles of transfer function.\n        gain : Number, Symbol, Expression\n            A scalar value specifying gain of the model.\n        var : Symbol\n            Complex variable of the Laplace transform used by the\n            polynomials of the transfer function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, k\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> zeros = [1, 2, 3]\n        >>> poles = [6, 5, 4]\n        >>> gain = 7\n        >>> tf = TransferFunction.from_zpk(zeros, poles, gain, s)\n        >>> tf\n        TransferFunction(7*(s - 3)*(s - 2)*(s - 1), (s - 6)*(s - 5)*(s - 4), s)\n\n        # Create a Transfer Function with variable poles and zeros\n        >>> tf1 = TransferFunction.from_zpk([p, k], [p + k, p - k], 2, s)\n        >>> tf1\n        TransferFunction(2*(-k + s)*(-p + s), (-k - p + s)*(k - p + s), s)\n\n        # Complex poles or zeros are acceptable\n        >>> tf2 = TransferFunction.from_zpk([0], [1-1j, 1+1j, 2], -2, s)\n        >>> tf2\n        TransferFunction(-2*s, (s - 2)*(s - 1.0 - 1.0*I)*(s - 1.0 + 1.0*I), s)\n\n        '
        num_poly = 1
        den_poly = 1
        for zero in zeros:
            num_poly *= var - zero
        for pole in poles:
            den_poly *= var - pole
        return cls(gain * num_poly, den_poly, var)

    @property
    def num(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the numerator polynomial of the transfer function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> G1 = TransferFunction(s**2 + p*s + 3, s - 4, s)\n        >>> G1.num\n        p*s + s**2 + 3\n        >>> G2 = TransferFunction((p + 5)*(p - 3), (p - 3)*(p + 1), p)\n        >>> G2.num\n        (p - 3)*(p + 5)\n\n        '
        return self.args[0]

    @property
    def den(self):
        if False:
            return 10
        '\n        Returns the denominator polynomial of the transfer function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> G1 = TransferFunction(s + 4, p**3 - 2*p + 4, s)\n        >>> G1.den\n        p**3 - 2*p + 4\n        >>> G2 = TransferFunction(3, 4, s)\n        >>> G2.den\n        4\n\n        '
        return self.args[1]

    @property
    def var(self):
        if False:
            print('Hello World!')
        '\n        Returns the complex variable of the Laplace transform used by the polynomials of\n        the transfer function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)\n        >>> G1.var\n        p\n        >>> G2 = TransferFunction(0, s - 5, s)\n        >>> G2.var\n        s\n\n        '
        return self.args[2]

    def _eval_subs(self, old, new):
        if False:
            for i in range(10):
                print('nop')
        arg_num = self.num.subs(old, new)
        arg_den = self.den.subs(old, new)
        argnew = TransferFunction(arg_num, arg_den, self.var)
        return self if old == self.var else argnew

    def _eval_evalf(self, prec):
        if False:
            for i in range(10):
                print('nop')
        return TransferFunction(self.num._eval_evalf(prec), self.den._eval_evalf(prec), self.var)

    def _eval_simplify(self, **kwargs):
        if False:
            return 10
        tf = cancel(Mul(self.num, 1 / self.den, evaluate=False), expand=False).as_numer_denom()
        (num_, den_) = (tf[0], tf[1])
        return TransferFunction(num_, den_, self.var)

    def _eval_rewrite_as_StateSpace(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Returns the equivalent space space model of the transfer function model.\n        The state space model will be returned in the controllable cannonical form.\n\n        Unlike the space state to transfer function model conversion, the transfer function\n        to state space model conversion is not unique. There can be multiple state space\n        representations of a given transfer function model.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control import TransferFunction, StateSpace\n        >>> tf = TransferFunction(s**2 + 1, s**3 + 2*s + 10, s)\n        >>> tf.rewrite(StateSpace)\n        StateSpace(Matrix([\n        [  0,  1, 0],\n        [  0,  0, 1],\n        [-10, -2, 0]]), Matrix([\n        [0],\n        [0],\n        [1]]), Matrix([[1, 0, 1]]), Matrix([[0]]))\n\n        '
        if not self.is_proper:
            raise ValueError('Transfer Function must be proper.')
        num_poly = Poly(self.num, self.var)
        den_poly = Poly(self.den, self.var)
        n = den_poly.degree()
        num_coeffs = num_poly.all_coeffs()
        den_coeffs = den_poly.all_coeffs()
        diff = n - num_poly.degree()
        num_coeffs = [0] * diff + num_coeffs
        a = den_coeffs[1:]
        a_mat = Matrix([[-1 * coefficient / den_coeffs[0] for coefficient in reversed(a)]])
        vert = zeros(n - 1, 1)
        mat = eye(n - 1)
        A = vert.row_join(mat)
        A = A.col_join(a_mat)
        B = zeros(n, 1)
        B[n - 1] = 1
        i = n
        C = []
        while i > 0:
            C.append(num_coeffs[i] - den_coeffs[i] * num_coeffs[0])
            i -= 1
        C = Matrix([C])
        D = Matrix([num_coeffs[0]])
        return StateSpace(A, B, C, D)

    def expand(self):
        if False:
            while True:
                i = 10
        '\n        Returns the transfer function with numerator and denominator\n        in expanded form.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> G1 = TransferFunction((a - s)**2, (s**2 + a)**2, s)\n        >>> G1.expand()\n        TransferFunction(a**2 - 2*a*s + s**2, a**2 + 2*a*s**2 + s**4, s)\n        >>> G2 = TransferFunction((p + 3*b)*(p - b), (p - b)*(p + 2*b), p)\n        >>> G2.expand()\n        TransferFunction(-3*b**2 + 2*b*p + p**2, -2*b**2 + b*p + p**2, p)\n\n        '
        return TransferFunction(expand(self.num), expand(self.den), self.var)

    def dc_gain(self):
        if False:
            print('Hello World!')
        '\n        Computes the gain of the response as the frequency approaches zero.\n\n        The DC gain is infinite for systems with pure integrators.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> tf1 = TransferFunction(s + 3, s**2 - 9, s)\n        >>> tf1.dc_gain()\n        -1/3\n        >>> tf2 = TransferFunction(p**2, p - 3 + p**3, p)\n        >>> tf2.dc_gain()\n        0\n        >>> tf3 = TransferFunction(a*p**2 - b, s + b, s)\n        >>> tf3.dc_gain()\n        (a*p**2 - b)/b\n        >>> tf4 = TransferFunction(1, s, s)\n        >>> tf4.dc_gain()\n        oo\n\n        '
        m = Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
        return limit(m, self.var, 0)

    def poles(self):
        if False:
            print('Hello World!')
        '\n        Returns the poles of a transfer function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)\n        >>> tf1.poles()\n        [-5, 1]\n        >>> tf2 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)\n        >>> tf2.poles()\n        [I, I, -I, -I]\n        >>> tf3 = TransferFunction(s**2, a*s + p, s)\n        >>> tf3.poles()\n        [-p/a]\n\n        '
        return _roots(Poly(self.den, self.var), self.var)

    def zeros(self):
        if False:
            print('Hello World!')
        '\n        Returns the zeros of a transfer function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> tf1 = TransferFunction((p + 3)*(p - 1), (p - 1)*(p + 5), p)\n        >>> tf1.zeros()\n        [-3, 1]\n        >>> tf2 = TransferFunction((1 - s)**2, (s**2 + 1)**2, s)\n        >>> tf2.zeros()\n        [1, 1]\n        >>> tf3 = TransferFunction(s**2, a*s + p, s)\n        >>> tf3.zeros()\n        [0, 0]\n\n        '
        return _roots(Poly(self.num, self.var), self.var)

    def eval_frequency(self, other):
        if False:
            print('Hello World!')
        '\n        Returns the system response at any point in the real or complex plane.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy import I\n        >>> tf1 = TransferFunction(1, s**2 + 2*s + 1, s)\n        >>> omega = 0.1\n        >>> tf1.eval_frequency(I*omega)\n        1/(0.99 + 0.2*I)\n        >>> tf2 = TransferFunction(s**2, a*s + p, s)\n        >>> tf2.eval_frequency(2)\n        4/(2*a + p)\n        >>> tf2.eval_frequency(I*2)\n        -4/(2*I*a + p)\n        '
        arg_num = self.num.subs(self.var, other)
        arg_den = self.den.subs(self.var, other)
        argnew = TransferFunction(arg_num, arg_den, self.var).to_expr()
        return argnew.expand()

    def is_stable(self):
        if False:
            print('Hello World!')
        "\n        Returns True if the transfer function is asymptotically stable; else False.\n\n        This would not check the marginal or conditional stability of the system.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a\n        >>> from sympy import symbols\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> q, r = symbols('q, r', negative=True)\n        >>> tf1 = TransferFunction((1 - s)**2, (s + 1)**2, s)\n        >>> tf1.is_stable()\n        True\n        >>> tf2 = TransferFunction((1 - p)**2, (s**2 + 1)**2, s)\n        >>> tf2.is_stable()\n        False\n        >>> tf3 = TransferFunction(4, q*s - r, s)\n        >>> tf3.is_stable()\n        False\n        >>> tf4 = TransferFunction(p + 1, a*p - s**2, p)\n        >>> tf4.is_stable() is None   # Not enough info about the symbols to determine stability\n        True\n\n        "
        return fuzzy_and((pole.as_real_imag()[0].is_negative for pole in self.poles()))

    def __add__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, (TransferFunction, Series)):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            return Parallel(self, other)
        elif isinstance(other, Parallel):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            arg_list = list(other.args)
            return Parallel(self, *arg_list)
        else:
            raise ValueError('TransferFunction cannot be added with {}.'.format(type(other)))

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self + other

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, (TransferFunction, Series)):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            return Parallel(self, -other)
        elif isinstance(other, Parallel):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            arg_list = [-i for i in list(other.args)]
            return Parallel(self, *arg_list)
        else:
            raise ValueError('{} cannot be subtracted from a TransferFunction.'.format(type(other)))

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return -self + other

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, (TransferFunction, Parallel)):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            return Series(self, other)
        elif isinstance(other, Series):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            arg_list = list(other.args)
            return Series(self, *arg_list)
        else:
            raise ValueError('TransferFunction cannot be multiplied with {}.'.format(type(other)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        if False:
            return 10
        if isinstance(other, TransferFunction):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            return Series(self, TransferFunction(other.den, other.num, self.var))
        elif isinstance(other, Parallel) and len(other.args) == 2 and isinstance(other.args[0], TransferFunction) and isinstance(other.args[1], (Series, TransferFunction)):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    Both TransferFunction and Parallel should use the\n                    same complex variable of the Laplace transform.'))
            if other.args[1] == self:
                return Feedback(self, other.args[0])
            other_arg_list = list(other.args[1].args) if isinstance(other.args[1], Series) else other.args[1]
            if other_arg_list == other.args[1]:
                return Feedback(self, other_arg_list)
            elif self in other_arg_list:
                other_arg_list.remove(self)
            else:
                return Feedback(self, Series(*other_arg_list))
            if len(other_arg_list) == 1:
                return Feedback(self, *other_arg_list)
            else:
                return Feedback(self, Series(*other_arg_list))
        else:
            raise ValueError('TransferFunction cannot be divided by {}.'.format(type(other)))
    __rtruediv__ = __truediv__

    def __pow__(self, p):
        if False:
            print('Hello World!')
        p = sympify(p)
        if not p.is_Integer:
            raise ValueError('Exponent must be an integer.')
        if p is S.Zero:
            return TransferFunction(1, 1, self.var)
        elif p > 0:
            (num_, den_) = (self.num ** p, self.den ** p)
        else:
            p = abs(p)
            (num_, den_) = (self.den ** p, self.num ** p)
        return TransferFunction(num_, den_, self.var)

    def __neg__(self):
        if False:
            print('Hello World!')
        return TransferFunction(-self.num, self.den, self.var)

    @property
    def is_proper(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if degree of the numerator polynomial is less than\n        or equal to degree of the denominator polynomial, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)\n        >>> tf1.is_proper\n        False\n        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*p + 2, p)\n        >>> tf2.is_proper\n        True\n\n        '
        return degree(self.num, self.var) <= degree(self.den, self.var)

    @property
    def is_strictly_proper(self):
        if False:
            while True:
                i = 10
        '\n        Returns True if degree of the numerator polynomial is strictly less\n        than degree of the denominator polynomial, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf1.is_strictly_proper\n        False\n        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)\n        >>> tf2.is_strictly_proper\n        True\n\n        '
        return degree(self.num, self.var) < degree(self.den, self.var)

    @property
    def is_biproper(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if degree of the numerator polynomial is equal to\n        degree of the denominator polynomial, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf1.is_biproper\n        True\n        >>> tf2 = TransferFunction(p**2, p + a, p)\n        >>> tf2.is_biproper\n        False\n\n        '
        return degree(self.num, self.var) == degree(self.den, self.var)

    def to_expr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts a ``TransferFunction`` object to SymPy Expr.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction\n        >>> from sympy import Expr\n        >>> tf1 = TransferFunction(s, a*s**2 + 1, s)\n        >>> tf1.to_expr()\n        s/(a*s**2 + 1)\n        >>> isinstance(_, Expr)\n        True\n        >>> tf2 = TransferFunction(1, (p + 3*b)*(b - p), p)\n        >>> tf2.to_expr()\n        1/((b - p)*(3*b + p))\n        >>> tf3 = TransferFunction((s - 2)*(s - 3), (s - 1)*(s - 2)*(s - 3), s)\n        >>> tf3.to_expr()\n        ((s - 3)*(s - 2))/(((s - 3)*(s - 2)*(s - 1)))\n\n        '
        if self.num != 1:
            return Mul(self.num, Pow(self.den, -1, evaluate=False), evaluate=False)
        else:
            return Pow(self.den, -1, evaluate=False)

def _flatten_args(args, _cls):
    if False:
        while True:
            i = 10
    temp_args = []
    for arg in args:
        if isinstance(arg, _cls):
            temp_args.extend(arg.args)
        else:
            temp_args.append(arg)
    return tuple(temp_args)

def _dummify_args(_arg, var):
    if False:
        return 10
    dummy_dict = {}
    dummy_arg_list = []
    for arg in _arg:
        _s = Dummy()
        dummy_dict[_s] = var
        dummy_arg = arg.subs({var: _s})
        dummy_arg_list.append(dummy_arg)
    return (dummy_arg_list, dummy_dict)

class Series(SISOLinearTimeInvariant):
    """
    A class for representing a series configuration of SISO systems.

    Parameters
    ==========

    args : SISOLinearTimeInvariant
        SISO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``Series(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, SISO in this case.

    Examples
    ========

    >>> from sympy.abc import s, p, a, b
    >>> from sympy.physics.control.lti import TransferFunction, Series, Parallel
    >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
    >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
    >>> tf3 = TransferFunction(p**2, p + s, s)
    >>> S1 = Series(tf1, tf2)
    >>> S1
    Series(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s))
    >>> S1.var
    s
    >>> S2 = Series(tf2, Parallel(tf3, -tf1))
    >>> S2
    Series(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), Parallel(TransferFunction(p**2, p + s, s), TransferFunction(-a*p**2 - b*s, -p + s, s)))
    >>> S2.var
    s
    >>> S3 = Series(Parallel(tf1, tf2), Parallel(tf2, tf3))
    >>> S3
    Series(Parallel(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)), Parallel(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), TransferFunction(p**2, p + s, s)))
    >>> S3.var
    s

    You can get the resultant transfer function by using ``.doit()`` method:

    >>> S3 = Series(tf1, tf2, -tf3)
    >>> S3.doit()
    TransferFunction(-p**2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)
    >>> S4 = Series(tf2, Parallel(tf1, -tf3))
    >>> S4.doit()
    TransferFunction((s**3 - 2)*(-p**2*(-p + s) + (p + s)*(a*p**2 + b*s)), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)

    Notes
    =====

    All the transfer functions should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    MIMOSeries, Parallel, TransferFunction, Feedback

    """

    def __new__(cls, *args, evaluate=False):
        if False:
            while True:
                i = 10
        args = _flatten_args(args, Series)
        cls._check_args(args)
        obj = super().__new__(cls, *args)
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        if False:
            print('Hello World!')
        '\n        Returns the complex variable used by all the transfer functions.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, Series, Parallel\n        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)\n        >>> G2 = TransferFunction(p, 4 - p, p)\n        >>> G3 = TransferFunction(0, p**4 - 1, p)\n        >>> Series(G1, G2).var\n        p\n        >>> Series(-G3, Parallel(G1, G2)).var\n        p\n\n        '
        return self.args[0].var

    def doit(self, **hints):
        if False:
            return 10
        '\n        Returns the resultant transfer function obtained after evaluating\n        the transfer functions in series configuration.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Series\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)\n        >>> Series(tf2, tf1).doit()\n        TransferFunction((s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s)\n        >>> Series(-tf1, -tf2).doit()\n        TransferFunction((2 - s**3)*(-a*p**2 - b*s), (-p + s)*(s**4 + 5*s + 6), s)\n\n        '
        _num_arg = (arg.doit().num for arg in self.args)
        _den_arg = (arg.doit().den for arg in self.args)
        res_num = Mul(*_num_arg, evaluate=True)
        res_den = Mul(*_den_arg, evaluate=True)
        return TransferFunction(res_num, res_den, self.var)

    def _eval_rewrite_as_TransferFunction(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.doit()

    @_check_other_SISO
    def __add__(self, other):
        if False:
            return 10
        if isinstance(other, Parallel):
            arg_list = list(other.args)
            return Parallel(self, *arg_list)
        return Parallel(self, other)
    __radd__ = __add__

    @_check_other_SISO
    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self + -other

    def __rsub__(self, other):
        if False:
            i = 10
            return i + 15
        return -self + other

    @_check_other_SISO
    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        arg_list = list(self.args)
        return Series(*arg_list, other)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, TransferFunction):
            return Series(*self.args, TransferFunction(other.den, other.num, other.var))
        elif isinstance(other, Series):
            tf_self = self.rewrite(TransferFunction)
            tf_other = other.rewrite(TransferFunction)
            return tf_self / tf_other
        elif isinstance(other, Parallel) and len(other.args) == 2 and isinstance(other.args[0], TransferFunction) and isinstance(other.args[1], Series):
            if not self.var == other.var:
                raise ValueError(filldedent('\n                    All the transfer functions should use the same complex variable\n                    of the Laplace transform.'))
            self_arg_list = set(self.args)
            other_arg_list = set(other.args[1].args)
            res = list(self_arg_list ^ other_arg_list)
            if len(res) == 0:
                return Feedback(self, other.args[0])
            elif len(res) == 1:
                return Feedback(self, *res)
            else:
                return Feedback(self, Series(*res))
        else:
            raise ValueError('This transfer function expression is invalid.')

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return Series(TransferFunction(-1, 1, self.var), self)

    def to_expr(self):
        if False:
            while True:
                i = 10
        'Returns the equivalent ``Expr`` object.'
        return Mul(*(arg.to_expr() for arg in self.args), evaluate=False)

    @property
    def is_proper(self):
        if False:
            print('Hello World!')
        '\n        Returns True if degree of the numerator polynomial of the resultant transfer\n        function is less than or equal to degree of the denominator polynomial of\n        the same, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Series\n        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)\n        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*s + 2, s)\n        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)\n        >>> S1 = Series(-tf2, tf1)\n        >>> S1.is_proper\n        False\n        >>> S2 = Series(tf1, tf2, tf3)\n        >>> S2.is_proper\n        True\n\n        '
        return self.doit().is_proper

    @property
    def is_strictly_proper(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if degree of the numerator polynomial of the resultant transfer\n        function is strictly less than degree of the denominator polynomial of\n        the same, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Series\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(s**3 - 2, s**2 + 5*s + 6, s)\n        >>> tf3 = TransferFunction(1, s**2 + s + 1, s)\n        >>> S1 = Series(tf1, tf2)\n        >>> S1.is_strictly_proper\n        False\n        >>> S2 = Series(tf1, tf2, tf3)\n        >>> S2.is_strictly_proper\n        True\n\n        '
        return self.doit().is_strictly_proper

    @property
    def is_biproper(self):
        if False:
            print('Hello World!')
        '\n        Returns True if degree of the numerator polynomial of the resultant transfer\n        function is equal to degree of the denominator polynomial of\n        the same, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Series\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(p, s**2, s)\n        >>> tf3 = TransferFunction(s**2, 1, s)\n        >>> S1 = Series(tf1, -tf2)\n        >>> S1.is_biproper\n        False\n        >>> S2 = Series(tf2, tf3)\n        >>> S2.is_biproper\n        True\n\n        '
        return self.doit().is_biproper

def _mat_mul_compatible(*args):
    if False:
        i = 10
        return i + 15
    'To check whether shapes are compatible for matrix mul.'
    return all((args[i].num_outputs == args[i + 1].num_inputs for i in range(len(args) - 1)))

class MIMOSeries(MIMOLinearTimeInvariant):
    """
    A class for representing a series configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO systems in a series configuration.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOSeries(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        ``num_outputs`` of the MIMO system is not equal to the
        ``num_inputs`` of its adjacent MIMO system. (Matrix
        multiplication constraint, basically)
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import MIMOSeries, TransferFunctionMatrix
    >>> from sympy import Matrix, pprint
    >>> mat_a = Matrix([[5*s], [5]])  # 2 Outputs 1 Input
    >>> mat_b = Matrix([[5, 1/(6*s**2)]])  # 1 Output 2 Inputs
    >>> mat_c = Matrix([[1, s], [5/s, 1]])  # 2 Outputs 2 Inputs
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(mat_a, s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(mat_b, s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(mat_c, s)
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a)
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(1, 1, s), TransferFunction(s, 1, s)), (TransferFunction(5, s, s), TransferFunction(1, 1, s)))), TransferFunctionMatrix(((TransferFunction(5, 1, s), TransferFunction(1, 6*s**2, s)),)), TransferFunctionMatrix(((TransferFunction(5*s, 1, s),), (TransferFunction(5, 1, s),))))
    >>> pprint(_, use_unicode=False)  #  For Better Visualization
    [5*s]                 [1  s]
    [---]    [5   1  ]    [-  -]
    [ 1 ]    [-  ----]    [1  1]
    [   ]   *[1     2]   *[    ]
    [ 5 ]    [   6*s ]{t} [5  1]
    [ - ]                 [-  -]
    [ 1 ]{t}              [s  1]{t}
    >>> MIMOSeries(tfm_c, tfm_b, tfm_a).doit()
    TransferFunctionMatrix(((TransferFunction(150*s**4 + 25*s, 6*s**3, s), TransferFunction(150*s**4 + 5*s, 6*s**2, s)), (TransferFunction(150*s**3 + 25, 6*s**3, s), TransferFunction(150*s**3 + 5, 6*s**2, s))))
    >>> pprint(_, use_unicode=False)  # (2 Inputs -A-> 2 Outputs) -> (2 Inputs -B-> 1 Output) -> (1 Input -C-> 2 Outputs) is equivalent to (2 Inputs -Series Equivalent-> 2 Outputs).
    [     4              4      ]
    [150*s  + 25*s  150*s  + 5*s]
    [-------------  ------------]
    [        3             2    ]
    [     6*s           6*s     ]
    [                           ]
    [      3              3     ]
    [ 150*s  + 25    150*s  + 5 ]
    [ -----------    ---------- ]
    [        3             2    ]
    [     6*s           6*s     ]{t}

    Notes
    =====

    All the transfer function matrices should use the same complex variable ``var`` of the Laplace transform.

    ``MIMOSeries(A, B)`` is not equivalent to ``A*B``. It is always in the reverse order, that is ``B*A``.

    See Also
    ========

    Series, MIMOParallel

    """

    def __new__(cls, *args, evaluate=False):
        if False:
            for i in range(10):
                print('nop')
        cls._check_args(args)
        if _mat_mul_compatible(*args):
            obj = super().__new__(cls, *args)
        else:
            raise ValueError(filldedent('\n                Number of input signals do not match the number\n                of output signals of adjacent systems for some args.'))
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        if False:
            return 10
        '\n        Returns the complex variable used by all the transfer functions.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix\n        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)\n        >>> G2 = TransferFunction(p, 4 - p, p)\n        >>> G3 = TransferFunction(0, p**4 - 1, p)\n        >>> tfm_1 = TransferFunctionMatrix([[G1, G2, G3]])\n        >>> tfm_2 = TransferFunctionMatrix([[G1], [G2], [G3]])\n        >>> MIMOSeries(tfm_2, tfm_1).var\n        p\n\n        '
        return self.args[0].var

    @property
    def num_inputs(self):
        if False:
            while True:
                i = 10
        'Returns the number of input signals of the series system.'
        return self.args[0].num_inputs

    @property
    def num_outputs(self):
        if False:
            while True:
                i = 10
        'Returns the number of output signals of the series system.'
        return self.args[-1].num_outputs

    @property
    def shape(self):
        if False:
            return 10
        'Returns the shape of the equivalent MIMO system.'
        return (self.num_outputs, self.num_inputs)

    def doit(self, cancel=False, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the resultant transfer function matrix obtained after evaluating\n        the MIMO systems arranged in a series configuration.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, MIMOSeries, TransferFunctionMatrix\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)\n        >>> tfm1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf2]])\n        >>> tfm2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf1]])\n        >>> MIMOSeries(tfm2, tfm1).doit()\n        TransferFunctionMatrix(((TransferFunction(2*(-p + s)*(s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)**2*(s**4 + 5*s + 6)**2, s), TransferFunction((-p + s)**2*(s**3 - 2)*(a*p**2 + b*s) + (-p + s)*(a*p**2 + b*s)**2*(s**4 + 5*s + 6), (-p + s)**3*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2)**2*(s**4 + 5*s + 6) + (s**3 - 2)*(a*p**2 + b*s)*(s**4 + 5*s + 6)**2, (-p + s)*(s**4 + 5*s + 6)**3, s), TransferFunction(2*(s**3 - 2)*(a*p**2 + b*s), (-p + s)*(s**4 + 5*s + 6), s))))\n\n        '
        _arg = (arg.doit()._expr_mat for arg in reversed(self.args))
        if cancel:
            res = MatMul(*_arg, evaluate=True)
            return TransferFunctionMatrix.from_Matrix(res, self.var)
        (_dummy_args, _dummy_dict) = _dummify_args(_arg, self.var)
        res = MatMul(*_dummy_args, evaluate=True)
        temp_tfm = TransferFunctionMatrix.from_Matrix(res, self.var)
        return temp_tfm.subs(_dummy_dict)

    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.doit()

    @_check_other_MIMO
    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, MIMOParallel):
            arg_list = list(other.args)
            return MIMOParallel(self, *arg_list)
        return MIMOParallel(self, other)
    __radd__ = __add__

    @_check_other_MIMO
    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self + -other

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        return -self + other

    @_check_other_MIMO
    def __mul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, MIMOSeries):
            self_arg_list = list(self.args)
            other_arg_list = list(other.args)
            return MIMOSeries(*other_arg_list, *self_arg_list)
        arg_list = list(self.args)
        return MIMOSeries(other, *arg_list)

    def __neg__(self):
        if False:
            print('Hello World!')
        arg_list = list(self.args)
        arg_list[0] = -arg_list[0]
        return MIMOSeries(*arg_list)

class Parallel(SISOLinearTimeInvariant):
    """
    A class for representing a parallel configuration of SISO systems.

    Parameters
    ==========

    args : SISOLinearTimeInvariant
        SISO systems in a parallel arrangement.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``Parallel(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed.

    Examples
    ========

    >>> from sympy.abc import s, p, a, b
    >>> from sympy.physics.control.lti import TransferFunction, Parallel, Series
    >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)
    >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)
    >>> tf3 = TransferFunction(p**2, p + s, s)
    >>> P1 = Parallel(tf1, tf2)
    >>> P1
    Parallel(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s))
    >>> P1.var
    s
    >>> P2 = Parallel(tf2, Series(tf3, -tf1))
    >>> P2
    Parallel(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), Series(TransferFunction(p**2, p + s, s), TransferFunction(-a*p**2 - b*s, -p + s, s)))
    >>> P2.var
    s
    >>> P3 = Parallel(Series(tf1, tf2), Series(tf2, tf3))
    >>> P3
    Parallel(Series(TransferFunction(a*p**2 + b*s, -p + s, s), TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)), Series(TransferFunction(s**3 - 2, s**4 + 5*s + 6, s), TransferFunction(p**2, p + s, s)))
    >>> P3.var
    s

    You can get the resultant transfer function by using ``.doit()`` method:

    >>> Parallel(tf1, tf2, -tf3).doit()
    TransferFunction(-p**2*(-p + s)*(s**4 + 5*s + 6) + (-p + s)*(p + s)*(s**3 - 2) + (p + s)*(a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)
    >>> Parallel(tf2, Series(tf1, -tf3)).doit()
    TransferFunction(-p**2*(a*p**2 + b*s)*(s**4 + 5*s + 6) + (-p + s)*(p + s)*(s**3 - 2), (-p + s)*(p + s)*(s**4 + 5*s + 6), s)

    Notes
    =====

    All the transfer functions should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    Series, TransferFunction, Feedback

    """

    def __new__(cls, *args, evaluate=False):
        if False:
            for i in range(10):
                print('nop')
        args = _flatten_args(args, Parallel)
        cls._check_args(args)
        obj = super().__new__(cls, *args)
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        if False:
            return 10
        '\n        Returns the complex variable used by all the transfer functions.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, Parallel, Series\n        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)\n        >>> G2 = TransferFunction(p, 4 - p, p)\n        >>> G3 = TransferFunction(0, p**4 - 1, p)\n        >>> Parallel(G1, G2).var\n        p\n        >>> Parallel(-G3, Series(G1, G2)).var\n        p\n\n        '
        return self.args[0].var

    def doit(self, **hints):
        if False:
            while True:
                i = 10
        '\n        Returns the resultant transfer function obtained after evaluating\n        the transfer functions in parallel configuration.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Parallel\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)\n        >>> Parallel(tf2, tf1).doit()\n        TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)\n        >>> Parallel(-tf1, -tf2).doit()\n        TransferFunction((2 - s**3)*(-p + s) + (-a*p**2 - b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)\n\n        '
        _arg = (arg.doit().to_expr() for arg in self.args)
        res = Add(*_arg).as_numer_denom()
        return TransferFunction(*res, self.var)

    def _eval_rewrite_as_TransferFunction(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.doit()

    @_check_other_SISO
    def __add__(self, other):
        if False:
            return 10
        self_arg_list = list(self.args)
        return Parallel(*self_arg_list, other)
    __radd__ = __add__

    @_check_other_SISO
    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        return self + -other

    def __rsub__(self, other):
        if False:
            return 10
        return -self + other

    @_check_other_SISO
    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, Series):
            arg_list = list(other.args)
            return Series(self, *arg_list)
        return Series(self, other)

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        return Series(TransferFunction(-1, 1, self.var), self)

    def to_expr(self):
        if False:
            print('Hello World!')
        'Returns the equivalent ``Expr`` object.'
        return Add(*(arg.to_expr() for arg in self.args), evaluate=False)

    @property
    def is_proper(self):
        if False:
            return 10
        '\n        Returns True if degree of the numerator polynomial of the resultant transfer\n        function is less than or equal to degree of the denominator polynomial of\n        the same, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Parallel\n        >>> tf1 = TransferFunction(b*s**2 + p**2 - a*p + s, b - p**2, s)\n        >>> tf2 = TransferFunction(p**2 - 4*p, p**3 + 3*s + 2, s)\n        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)\n        >>> P1 = Parallel(-tf2, tf1)\n        >>> P1.is_proper\n        False\n        >>> P2 = Parallel(tf2, tf3)\n        >>> P2.is_proper\n        True\n\n        '
        return self.doit().is_proper

    @property
    def is_strictly_proper(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if degree of the numerator polynomial of the resultant transfer\n        function is strictly less than degree of the denominator polynomial of\n        the same, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Parallel\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)\n        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)\n        >>> P1 = Parallel(tf1, tf2)\n        >>> P1.is_strictly_proper\n        False\n        >>> P2 = Parallel(tf2, tf3)\n        >>> P2.is_strictly_proper\n        True\n\n        '
        return self.doit().is_strictly_proper

    @property
    def is_biproper(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if degree of the numerator polynomial of the resultant transfer\n        function is equal to degree of the denominator polynomial of\n        the same, else False.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, Parallel\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(p**2, p + s, s)\n        >>> tf3 = TransferFunction(s, s**2 + s + 1, s)\n        >>> P1 = Parallel(tf1, -tf2)\n        >>> P1.is_biproper\n        True\n        >>> P2 = Parallel(tf2, tf3)\n        >>> P2.is_biproper\n        False\n\n        '
        return self.doit().is_biproper

class MIMOParallel(MIMOLinearTimeInvariant):
    """
    A class for representing a parallel configuration of MIMO systems.

    Parameters
    ==========

    args : MIMOLinearTimeInvariant
        MIMO Systems in a parallel arrangement.
    evaluate : Boolean, Keyword
        When passed ``True``, returns the equivalent
        ``MIMOParallel(*args).doit()``. Set to ``False`` by default.

    Raises
    ======

    ValueError
        When no argument is passed.

        ``var`` attribute is not same for every system.

        All MIMO systems passed do not have same shape.
    TypeError
        Any of the passed ``*args`` has unsupported type

        A combination of SISO and MIMO systems is
        passed. There should be homogeneity in the
        type of systems passed, MIMO in this case.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunctionMatrix, MIMOParallel
    >>> from sympy import Matrix, pprint
    >>> expr_1 = 1/s
    >>> expr_2 = s/(s**2-1)
    >>> expr_3 = (2 + s)/(s**2 - 1)
    >>> expr_4 = 5
    >>> tfm_a = TransferFunctionMatrix.from_Matrix(Matrix([[expr_1, expr_2], [expr_3, expr_4]]), s)
    >>> tfm_b = TransferFunctionMatrix.from_Matrix(Matrix([[expr_2, expr_1], [expr_4, expr_3]]), s)
    >>> tfm_c = TransferFunctionMatrix.from_Matrix(Matrix([[expr_3, expr_4], [expr_1, expr_2]]), s)
    >>> MIMOParallel(tfm_a, tfm_b, tfm_c)
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(1, s, s), TransferFunction(s, s**2 - 1, s)), (TransferFunction(s + 2, s**2 - 1, s), TransferFunction(5, 1, s)))), TransferFunctionMatrix(((TransferFunction(s, s**2 - 1, s), TransferFunction(1, s, s)), (TransferFunction(5, 1, s), TransferFunction(s + 2, s**2 - 1, s)))), TransferFunctionMatrix(((TransferFunction(s + 2, s**2 - 1, s), TransferFunction(5, 1, s)), (TransferFunction(1, s, s), TransferFunction(s, s**2 - 1, s)))))
    >>> pprint(_, use_unicode=False)  #  For Better Visualization
    [  1       s   ]      [  s       1   ]      [s + 2     5   ]
    [  -     ------]      [------    -   ]      [------    -   ]
    [  s      2    ]      [ 2        s   ]      [ 2        1   ]
    [        s  - 1]      [s  - 1        ]      [s  - 1        ]
    [              ]    + [              ]    + [              ]
    [s + 2     5   ]      [  5     s + 2 ]      [  1       s   ]
    [------    -   ]      [  -     ------]      [  -     ------]
    [ 2        1   ]      [  1      2    ]      [  s      2    ]
    [s  - 1        ]{t}   [        s  - 1]{t}   [        s  - 1]{t}
    >>> MIMOParallel(tfm_a, tfm_b, tfm_c).doit()
    TransferFunctionMatrix(((TransferFunction(s**2 + s*(2*s + 2) - 1, s*(s**2 - 1), s), TransferFunction(2*s**2 + 5*s*(s**2 - 1) - 1, s*(s**2 - 1), s)), (TransferFunction(s**2 + s*(s + 2) + 5*s*(s**2 - 1) - 1, s*(s**2 - 1), s), TransferFunction(5*s**2 + 2*s - 3, s**2 - 1, s))))
    >>> pprint(_, use_unicode=False)
    [       2                              2       / 2    \\    ]
    [      s  + s*(2*s + 2) - 1         2*s  + 5*s*\\s  - 1/ - 1]
    [      --------------------         -----------------------]
    [             / 2    \\                       / 2    \\      ]
    [           s*\\s  - 1/                     s*\\s  - 1/      ]
    [                                                          ]
    [ 2                   / 2    \\             2               ]
    [s  + s*(s + 2) + 5*s*\\s  - 1/ - 1      5*s  + 2*s - 3     ]
    [---------------------------------      --------------     ]
    [              / 2    \\                      2             ]
    [            s*\\s  - 1/                     s  - 1         ]{t}

    Notes
    =====

    All the transfer function matrices should use the same complex variable
    ``var`` of the Laplace transform.

    See Also
    ========

    Parallel, MIMOSeries

    """

    def __new__(cls, *args, evaluate=False):
        if False:
            while True:
                i = 10
        args = _flatten_args(args, MIMOParallel)
        cls._check_args(args)
        if any((arg.shape != args[0].shape for arg in args)):
            raise TypeError('Shape of all the args is not equal.')
        obj = super().__new__(cls, *args)
        return obj.doit() if evaluate else obj

    @property
    def var(self):
        if False:
            return 10
        '\n        Returns the complex variable used by all the systems.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOParallel\n        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)\n        >>> G2 = TransferFunction(p, 4 - p, p)\n        >>> G3 = TransferFunction(0, p**4 - 1, p)\n        >>> G4 = TransferFunction(p**2, p**2 - 1, p)\n        >>> tfm_a = TransferFunctionMatrix([[G1, G2], [G3, G4]])\n        >>> tfm_b = TransferFunctionMatrix([[G2, G1], [G4, G3]])\n        >>> MIMOParallel(tfm_a, tfm_b).var\n        p\n\n        '
        return self.args[0].var

    @property
    def num_inputs(self):
        if False:
            while True:
                i = 10
        'Returns the number of input signals of the parallel system.'
        return self.args[0].num_inputs

    @property
    def num_outputs(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of output signals of the parallel system.'
        return self.args[0].num_outputs

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the shape of the equivalent MIMO system.'
        return (self.num_outputs, self.num_inputs)

    def doit(self, **hints):
        if False:
            print('Hello World!')
        '\n        Returns the resultant transfer function matrix obtained after evaluating\n        the MIMO systems arranged in a parallel configuration.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p, a, b\n        >>> from sympy.physics.control.lti import TransferFunction, MIMOParallel, TransferFunctionMatrix\n        >>> tf1 = TransferFunction(a*p**2 + b*s, s - p, s)\n        >>> tf2 = TransferFunction(s**3 - 2, s**4 + 5*s + 6, s)\n        >>> tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])\n        >>> tfm_2 = TransferFunctionMatrix([[tf2, tf1], [tf1, tf2]])\n        >>> MIMOParallel(tfm_1, tfm_2).doit()\n        TransferFunctionMatrix(((TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s), TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s)), (TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s), TransferFunction((-p + s)*(s**3 - 2) + (a*p**2 + b*s)*(s**4 + 5*s + 6), (-p + s)*(s**4 + 5*s + 6), s))))\n\n        '
        _arg = (arg.doit()._expr_mat for arg in self.args)
        res = MatAdd(*_arg, evaluate=True)
        return TransferFunctionMatrix.from_Matrix(res, self.var)

    def _eval_rewrite_as_TransferFunctionMatrix(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.doit()

    @_check_other_MIMO
    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        self_arg_list = list(self.args)
        return MIMOParallel(*self_arg_list, other)
    __radd__ = __add__

    @_check_other_MIMO
    def __sub__(self, other):
        if False:
            print('Hello World!')
        return self + -other

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return -self + other

    @_check_other_MIMO
    def __mul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, MIMOSeries):
            arg_list = list(other.args)
            return MIMOSeries(*arg_list, self)
        return MIMOSeries(other, self)

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        arg_list = [-arg for arg in list(self.args)]
        return MIMOParallel(*arg_list)

class Feedback(SISOLinearTimeInvariant):
    """
    A class for representing closed-loop feedback interconnection between two
    SISO input/output systems.

    The first argument, ``sys1``, is the feedforward part of the closed-loop
    system or in simple words, the dynamical model representing the process
    to be controlled. The second argument, ``sys2``, is the feedback system
    and controls the fed back signal to ``sys1``. Both ``sys1`` and ``sys2``
    can either be ``Series`` or ``TransferFunction`` objects.

    Parameters
    ==========

    sys1 : Series, TransferFunction
        The feedforward path system.
    sys2 : Series, TransferFunction, optional
        The feedback path system (often a feedback controller).
        It is the model sitting on the feedback path.

        If not specified explicitly, the sys2 is
        assumed to be unit (1.0) transfer function.
    sign : int, optional
        The sign of feedback. Can either be ``1``
        (for positive feedback) or ``-1`` (for negative feedback).
        Default value is `-1`.

    Raises
    ======

    ValueError
        When ``sys1`` and ``sys2`` are not using the
        same complex variable of the Laplace transform.

        When a combination of ``sys1`` and ``sys2`` yields
        zero denominator.

    TypeError
        When either ``sys1`` or ``sys2`` is not a ``Series`` or a
        ``TransferFunction`` object.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction, Feedback
    >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
    >>> controller = TransferFunction(5*s - 10, s + 7, s)
    >>> F1 = Feedback(plant, controller)
    >>> F1
    Feedback(TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s), TransferFunction(5*s - 10, s + 7, s), -1)
    >>> F1.var
    s
    >>> F1.args
    (TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s), TransferFunction(5*s - 10, s + 7, s), -1)

    You can get the feedforward and feedback path systems by using ``.sys1`` and ``.sys2`` respectively.

    >>> F1.sys1
    TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)
    >>> F1.sys2
    TransferFunction(5*s - 10, s + 7, s)

    You can get the resultant closed loop transfer function obtained by negative feedback
    interconnection using ``.doit()`` method.

    >>> F1.doit()
    TransferFunction((s + 7)*(s**2 - 4*s + 2)*(3*s**2 + 7*s - 3), ((s + 7)*(s**2 - 4*s + 2) + (5*s - 10)*(3*s**2 + 7*s - 3))*(s**2 - 4*s + 2), s)
    >>> G = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)
    >>> C = TransferFunction(5*s + 10, s + 10, s)
    >>> F2 = Feedback(G*C, TransferFunction(1, 1, s))
    >>> F2.doit()
    TransferFunction((s + 10)*(5*s + 10)*(s**2 + 2*s + 3)*(2*s**2 + 5*s + 1), (s + 10)*((s + 10)*(s**2 + 2*s + 3) + (5*s + 10)*(2*s**2 + 5*s + 1))*(s**2 + 2*s + 3), s)

    To negate a ``Feedback`` object, the ``-`` operator can be prepended:

    >>> -F1
    Feedback(TransferFunction(-3*s**2 - 7*s + 3, s**2 - 4*s + 2, s), TransferFunction(10 - 5*s, s + 7, s), -1)
    >>> -F2
    Feedback(Series(TransferFunction(-1, 1, s), TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s), TransferFunction(5*s + 10, s + 10, s)), TransferFunction(-1, 1, s), -1)

    See Also
    ========

    MIMOFeedback, Series, Parallel

    """

    def __new__(cls, sys1, sys2=None, sign=-1):
        if False:
            for i in range(10):
                print('nop')
        if not sys2:
            sys2 = TransferFunction(1, 1, sys1.var)
        if not (isinstance(sys1, (TransferFunction, Series)) and isinstance(sys2, (TransferFunction, Series))):
            raise TypeError('Unsupported type for `sys1` or `sys2` of Feedback.')
        if sign not in [-1, 1]:
            raise ValueError(filldedent('\n                Unsupported type for feedback. `sign` arg should\n                either be 1 (positive feedback loop) or -1\n                (negative feedback loop).'))
        if Mul(sys1.to_expr(), sys2.to_expr()).simplify() == sign:
            raise ValueError('The equivalent system will have zero denominator.')
        if sys1.var != sys2.var:
            raise ValueError(filldedent('\n                Both `sys1` and `sys2` should be using the\n                same complex variable.'))
        return super().__new__(cls, sys1, sys2, _sympify(sign))

    @property
    def sys1(self):
        if False:
            return 10
        '\n        Returns the feedforward system of the feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction, Feedback\n        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)\n        >>> controller = TransferFunction(5*s - 10, s + 7, s)\n        >>> F1 = Feedback(plant, controller)\n        >>> F1.sys1\n        TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)\n        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)\n        >>> C = TransferFunction(5*p + 10, p + 10, p)\n        >>> P = TransferFunction(1 - s, p + 2, p)\n        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)\n        >>> F2.sys1\n        TransferFunction(1, 1, p)\n\n        '
        return self.args[0]

    @property
    def sys2(self):
        if False:
            print('Hello World!')
        '\n        Returns the feedback controller of the feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction, Feedback\n        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)\n        >>> controller = TransferFunction(5*s - 10, s + 7, s)\n        >>> F1 = Feedback(plant, controller)\n        >>> F1.sys2\n        TransferFunction(5*s - 10, s + 7, s)\n        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)\n        >>> C = TransferFunction(5*p + 10, p + 10, p)\n        >>> P = TransferFunction(1 - s, p + 2, p)\n        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)\n        >>> F2.sys2\n        Series(TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p), TransferFunction(5*p + 10, p + 10, p), TransferFunction(1 - s, p + 2, p))\n\n        '
        return self.args[1]

    @property
    def var(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the complex variable of the Laplace transform used by all\n        the transfer functions involved in the feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction, Feedback\n        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)\n        >>> controller = TransferFunction(5*s - 10, s + 7, s)\n        >>> F1 = Feedback(plant, controller)\n        >>> F1.var\n        s\n        >>> G = TransferFunction(2*s**2 + 5*s + 1, p**2 + 2*p + 3, p)\n        >>> C = TransferFunction(5*p + 10, p + 10, p)\n        >>> P = TransferFunction(1 - s, p + 2, p)\n        >>> F2 = Feedback(TransferFunction(1, 1, p), G*C*P)\n        >>> F2.var\n        p\n\n        '
        return self.sys1.var

    @property
    def sign(self):
        if False:
            print('Hello World!')
        '\n        Returns the type of MIMO Feedback model. ``1``\n        for Positive and ``-1`` for Negative.\n        '
        return self.args[2]

    @property
    def sensitivity(self):
        if False:
            return 10
        '\n        Returns the sensitivity function of the feedback loop.\n\n        Sensitivity of a Feedback system is the ratio\n        of change in the open loop gain to the change in\n        the closed loop gain.\n\n        .. note::\n            This method would not return the complementary\n            sensitivity function.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, Feedback\n        >>> C = TransferFunction(5*p + 10, p + 10, p)\n        >>> P = TransferFunction(1 - p, p + 2, p)\n        >>> F_1 = Feedback(P, C)\n        >>> F_1.sensitivity\n        1/((1 - p)*(5*p + 10)/((p + 2)*(p + 10)) + 1)\n\n        '
        return 1 / (1 - self.sign * self.sys1.to_expr() * self.sys2.to_expr())

    def doit(self, cancel=False, expand=False, **hints):
        if False:
            print('Hello World!')
        '\n        Returns the resultant transfer function obtained by the\n        feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, Feedback\n        >>> plant = TransferFunction(3*s**2 + 7*s - 3, s**2 - 4*s + 2, s)\n        >>> controller = TransferFunction(5*s - 10, s + 7, s)\n        >>> F1 = Feedback(plant, controller)\n        >>> F1.doit()\n        TransferFunction((s + 7)*(s**2 - 4*s + 2)*(3*s**2 + 7*s - 3), ((s + 7)*(s**2 - 4*s + 2) + (5*s - 10)*(3*s**2 + 7*s - 3))*(s**2 - 4*s + 2), s)\n        >>> G = TransferFunction(2*s**2 + 5*s + 1, s**2 + 2*s + 3, s)\n        >>> F2 = Feedback(G, TransferFunction(1, 1, s))\n        >>> F2.doit()\n        TransferFunction((s**2 + 2*s + 3)*(2*s**2 + 5*s + 1), (s**2 + 2*s + 3)*(3*s**2 + 7*s + 4), s)\n\n        Use kwarg ``expand=True`` to expand the resultant transfer function.\n        Use ``cancel=True`` to cancel out the common terms in numerator and\n        denominator.\n\n        >>> F2.doit(cancel=True, expand=True)\n        TransferFunction(2*s**2 + 5*s + 1, 3*s**2 + 7*s + 4, s)\n        >>> F2.doit(expand=True)\n        TransferFunction(2*s**4 + 9*s**3 + 17*s**2 + 17*s + 3, 3*s**4 + 13*s**3 + 27*s**2 + 29*s + 12, s)\n\n        '
        arg_list = list(self.sys1.args) if isinstance(self.sys1, Series) else [self.sys1]
        (F_n, unit) = (self.sys1.doit(), TransferFunction(1, 1, self.sys1.var))
        if self.sign == -1:
            F_d = Parallel(unit, Series(self.sys2, *arg_list)).doit()
        else:
            F_d = Parallel(unit, -Series(self.sys2, *arg_list)).doit()
        _resultant_tf = TransferFunction(F_n.num * F_d.den, F_n.den * F_d.num, F_n.var)
        if cancel:
            _resultant_tf = _resultant_tf.simplify()
        if expand:
            _resultant_tf = _resultant_tf.expand()
        return _resultant_tf

    def _eval_rewrite_as_TransferFunction(self, num, den, sign, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.doit()

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return Feedback(-self.sys1, -self.sys2, self.sign)

def _is_invertible(a, b, sign):
    if False:
        while True:
            i = 10
    '\n    Checks whether a given pair of MIMO\n    systems passed is invertible or not.\n    '
    _mat = eye(a.num_outputs) - sign * a.doit()._expr_mat * b.doit()._expr_mat
    _det = _mat.det()
    return _det != 0

class MIMOFeedback(MIMOLinearTimeInvariant):
    """
    A class for representing closed-loop feedback interconnection between two
    MIMO input/output systems.

    Parameters
    ==========

    sys1 : MIMOSeries, TransferFunctionMatrix
        The MIMO system placed on the feedforward path.
    sys2 : MIMOSeries, TransferFunctionMatrix
        The system placed on the feedback path
        (often a feedback controller).
    sign : int, optional
        The sign of feedback. Can either be ``1``
        (for positive feedback) or ``-1`` (for negative feedback).
        Default value is `-1`.

    Raises
    ======

    ValueError
        When ``sys1`` and ``sys2`` are not using the
        same complex variable of the Laplace transform.

        Forward path model should have an equal number of inputs/outputs
        to the feedback path outputs/inputs.

        When product of ``sys1`` and ``sys2`` is not a square matrix.

        When the equivalent MIMO system is not invertible.

    TypeError
        When either ``sys1`` or ``sys2`` is not a ``MIMOSeries`` or a
        ``TransferFunctionMatrix`` object.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunctionMatrix, MIMOFeedback
    >>> plant_mat = Matrix([[1, 1/s], [0, 1]])
    >>> controller_mat = Matrix([[10, 0], [0, 10]])  # Constant Gain
    >>> plant = TransferFunctionMatrix.from_Matrix(plant_mat, s)
    >>> controller = TransferFunctionMatrix.from_Matrix(controller_mat, s)
    >>> feedback = MIMOFeedback(plant, controller)  # Negative Feedback (default)
    >>> pprint(feedback, use_unicode=False)
    /    [1  1]    [10  0 ]   \\-1   [1  1]
    |    [-  -]    [--  - ]   |     [-  -]
    |    [1  s]    [1   1 ]   |     [1  s]
    |I + [    ]   *[      ]   |   * [    ]
    |    [0  1]    [0   10]   |     [0  1]
    |    [-  -]    [-   --]   |     [-  -]
    \\    [1  1]{t} [1   1 ]{t}/     [1  1]{t}

    To get the equivalent system matrix, use either ``doit`` or ``rewrite`` method.

    >>> pprint(feedback.doit(), use_unicode=False)
    [1     1  ]
    [--  -----]
    [11  121*s]
    [         ]
    [0    1   ]
    [-    --  ]
    [1    11  ]{t}

    To negate the ``MIMOFeedback`` object, use ``-`` operator.

    >>> neg_feedback = -feedback
    >>> pprint(neg_feedback.doit(), use_unicode=False)
    [-1    -1  ]
    [---  -----]
    [11   121*s]
    [          ]
    [ 0    -1  ]
    [ -    --- ]
    [ 1    11  ]{t}

    See Also
    ========

    Feedback, MIMOSeries, MIMOParallel

    """

    def __new__(cls, sys1, sys2, sign=-1):
        if False:
            for i in range(10):
                print('nop')
        if not (isinstance(sys1, (TransferFunctionMatrix, MIMOSeries)) and isinstance(sys2, (TransferFunctionMatrix, MIMOSeries))):
            raise TypeError('Unsupported type for `sys1` or `sys2` of MIMO Feedback.')
        if sys1.num_inputs != sys2.num_outputs or sys1.num_outputs != sys2.num_inputs:
            raise ValueError(filldedent('\n                Product of `sys1` and `sys2` must\n                yield a square matrix.'))
        if sign not in (-1, 1):
            raise ValueError(filldedent('\n                Unsupported type for feedback. `sign` arg should\n                either be 1 (positive feedback loop) or -1\n                (negative feedback loop).'))
        if not _is_invertible(sys1, sys2, sign):
            raise ValueError('Non-Invertible system inputted.')
        if sys1.var != sys2.var:
            raise ValueError(filldedent('\n                Both `sys1` and `sys2` should be using the\n                same complex variable.'))
        return super().__new__(cls, sys1, sys2, _sympify(sign))

    @property
    def sys1(self):
        if False:
            print('Hello World!')
        '\n        Returns the system placed on the feedforward path of the MIMO feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy import pprint\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback\n        >>> tf1 = TransferFunction(s**2 + s + 1, s**2 - s + 1, s)\n        >>> tf2 = TransferFunction(1, s, s)\n        >>> tf3 = TransferFunction(1, 1, s)\n        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])\n        >>> sys2 = TransferFunctionMatrix([[tf3, tf3], [tf3, tf2]])\n        >>> F_1 = MIMOFeedback(sys1, sys2, 1)\n        >>> F_1.sys1\n        TransferFunctionMatrix(((TransferFunction(s**2 + s + 1, s**2 - s + 1, s), TransferFunction(1, s, s)), (TransferFunction(1, s, s), TransferFunction(s**2 + s + 1, s**2 - s + 1, s))))\n        >>> pprint(_, use_unicode=False)\n        [ 2                    ]\n        [s  + s + 1      1     ]\n        [----------      -     ]\n        [ 2              s     ]\n        [s  - s + 1            ]\n        [                      ]\n        [             2        ]\n        [    1       s  + s + 1]\n        [    -       ----------]\n        [    s        2        ]\n        [            s  - s + 1]{t}\n\n        '
        return self.args[0]

    @property
    def sys2(self):
        if False:
            while True:
                i = 10
        '\n        Returns the feedback controller of the MIMO feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy import pprint\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback\n        >>> tf1 = TransferFunction(s**2, s**3 - s + 1, s)\n        >>> tf2 = TransferFunction(1, s, s)\n        >>> tf3 = TransferFunction(1, 1, s)\n        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])\n        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])\n        >>> F_1 = MIMOFeedback(sys1, sys2)\n        >>> F_1.sys2\n        TransferFunctionMatrix(((TransferFunction(s**2, s**3 - s + 1, s), TransferFunction(1, 1, s)), (TransferFunction(1, 1, s), TransferFunction(1, s, s))))\n        >>> pprint(_, use_unicode=False)\n        [     2       ]\n        [    s       1]\n        [----------  -]\n        [ 3          1]\n        [s  - s + 1   ]\n        [             ]\n        [    1       1]\n        [    -       -]\n        [    1       s]{t}\n\n        '
        return self.args[1]

    @property
    def var(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the complex variable of the Laplace transform used by all\n        the transfer functions involved in the MIMO feedback loop.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback\n        >>> tf1 = TransferFunction(p, 1 - p, p)\n        >>> tf2 = TransferFunction(1, p, p)\n        >>> tf3 = TransferFunction(1, 1, p)\n        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])\n        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])\n        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback\n        >>> F_1.var\n        p\n\n        '
        return self.sys1.var

    @property
    def sign(self):
        if False:
            print('Hello World!')
        '\n        Returns the type of feedback interconnection of two models. ``1``\n        for Positive and ``-1`` for Negative.\n        '
        return self.args[2]

    @property
    def sensitivity(self):
        if False:
            print('Hello World!')
        '\n        Returns the sensitivity function matrix of the feedback loop.\n\n        Sensitivity of a closed-loop system is the ratio of change\n        in the open loop gain to the change in the closed loop gain.\n\n        .. note::\n            This method would not return the complementary\n            sensitivity function.\n\n        Examples\n        ========\n\n        >>> from sympy import pprint\n        >>> from sympy.abc import p\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback\n        >>> tf1 = TransferFunction(p, 1 - p, p)\n        >>> tf2 = TransferFunction(1, p, p)\n        >>> tf3 = TransferFunction(1, 1, p)\n        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf2, tf1]])\n        >>> sys2 = TransferFunctionMatrix([[tf1, tf3], [tf3, tf2]])\n        >>> F_1 = MIMOFeedback(sys1, sys2, 1)  # Positive feedback\n        >>> F_2 = MIMOFeedback(sys1, sys2)  # Negative feedback\n        >>> pprint(F_1.sensitivity, use_unicode=False)\n        [   4      3      2               5      4      2           ]\n        [- p  + 3*p  - 4*p  + 3*p - 1    p  - 2*p  + 3*p  - 3*p + 1 ]\n        [----------------------------  -----------------------------]\n        [  4      3      2              5      4      3      2      ]\n        [ p  + 3*p  - 8*p  + 8*p - 3   p  + 3*p  - 8*p  + 8*p  - 3*p]\n        [                                                           ]\n        [       4    3    2                  3      2               ]\n        [      p  - p  - p  + p           3*p  - 6*p  + 4*p - 1     ]\n        [ --------------------------    --------------------------  ]\n        [  4      3      2               4      3      2            ]\n        [ p  + 3*p  - 8*p  + 8*p - 3    p  + 3*p  - 8*p  + 8*p - 3  ]\n        >>> pprint(F_2.sensitivity, use_unicode=False)\n        [ 4      3      2           5      4      2          ]\n        [p  - 3*p  + 2*p  + p - 1  p  - 2*p  + 3*p  - 3*p + 1]\n        [------------------------  --------------------------]\n        [   4      3                   5      4      2       ]\n        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p  - p   ]\n        [                                                    ]\n        [     4    3    2               4      3             ]\n        [    p  - p  - p  + p        2*p  - 3*p  + 2*p - 1   ]\n        [  -------------------       ---------------------   ]\n        [   4      3                   4      3              ]\n        [  p  - 3*p  + 2*p - 1        p  - 3*p  + 2*p - 1    ]\n\n        '
        _sys1_mat = self.sys1.doit()._expr_mat
        _sys2_mat = self.sys2.doit()._expr_mat
        return (eye(self.sys1.num_inputs) - self.sign * _sys1_mat * _sys2_mat).inv()

    def doit(self, cancel=True, expand=False, **hints):
        if False:
            print('Hello World!')
        '\n        Returns the resultant transfer function matrix obtained by the\n        feedback interconnection.\n\n        Examples\n        ========\n\n        >>> from sympy import pprint\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, MIMOFeedback\n        >>> tf1 = TransferFunction(s, 1 - s, s)\n        >>> tf2 = TransferFunction(1, s, s)\n        >>> tf3 = TransferFunction(5, 1, s)\n        >>> tf4 = TransferFunction(s - 1, s, s)\n        >>> tf5 = TransferFunction(0, 1, s)\n        >>> sys1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])\n        >>> sys2 = TransferFunctionMatrix([[tf3, tf5], [tf5, tf5]])\n        >>> F_1 = MIMOFeedback(sys1, sys2, 1)\n        >>> pprint(F_1, use_unicode=False)\n        /    [  s      1  ]    [5  0]   \\-1   [  s      1  ]\n        |    [-----    -  ]    [-  -]   |     [-----    -  ]\n        |    [1 - s    s  ]    [1  1]   |     [1 - s    s  ]\n        |I - [            ]   *[    ]   |   * [            ]\n        |    [  5    s - 1]    [0  0]   |     [  5    s - 1]\n        |    [  -    -----]    [-  -]   |     [  -    -----]\n        \\    [  1      s  ]{t} [1  1]{t}/     [  1      s  ]{t}\n        >>> pprint(F_1.doit(), use_unicode=False)\n        [  -s           s - 1       ]\n        [-------     -----------    ]\n        [6*s - 1     s*(6*s - 1)    ]\n        [                           ]\n        [5*s - 5  (s - 1)*(6*s + 24)]\n        [-------  ------------------]\n        [6*s - 1     s*(6*s - 1)    ]{t}\n\n        If the user wants the resultant ``TransferFunctionMatrix`` object without\n        canceling the common factors then the ``cancel`` kwarg should be passed ``False``.\n\n        >>> pprint(F_1.doit(cancel=False), use_unicode=False)\n        [             s*(s - 1)                              s - 1               ]\n        [         -----------------                       -----------            ]\n        [         (1 - s)*(6*s - 1)                       s*(6*s - 1)            ]\n        [                                                                        ]\n        [s*(25*s - 25) + 5*(1 - s)*(6*s - 1)  s*(s - 1)*(6*s - 1) + s*(25*s - 25)]\n        [-----------------------------------  -----------------------------------]\n        [         (1 - s)*(6*s - 1)                        2                     ]\n        [                                                 s *(6*s - 1)           ]{t}\n\n        If the user wants the expanded form of the resultant transfer function matrix,\n        the ``expand`` kwarg should be passed as ``True``.\n\n        >>> pprint(F_1.doit(expand=True), use_unicode=False)\n        [  -s          s - 1      ]\n        [-------      --------    ]\n        [6*s - 1         2        ]\n        [             6*s  - s    ]\n        [                         ]\n        [            2            ]\n        [5*s - 5  6*s  + 18*s - 24]\n        [-------  ----------------]\n        [6*s - 1         2        ]\n        [             6*s  - s    ]{t}\n\n        '
        _mat = self.sensitivity * self.sys1.doit()._expr_mat
        _resultant_tfm = _to_TFM(_mat, self.var)
        if cancel:
            _resultant_tfm = _resultant_tfm.simplify()
        if expand:
            _resultant_tfm = _resultant_tfm.expand()
        return _resultant_tfm

    def _eval_rewrite_as_TransferFunctionMatrix(self, sys1, sys2, sign, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.doit()

    def __neg__(self):
        if False:
            while True:
                i = 10
        return MIMOFeedback(-self.sys1, -self.sys2, self.sign)

def _to_TFM(mat, var):
    if False:
        print('Hello World!')
    'Private method to convert ImmutableMatrix to TransferFunctionMatrix efficiently'
    to_tf = lambda expr: TransferFunction.from_rational_expression(expr, var)
    arg = [[to_tf(expr) for expr in row] for row in mat.tolist()]
    return TransferFunctionMatrix(arg)

class TransferFunctionMatrix(MIMOLinearTimeInvariant):
    """
    A class for representing the MIMO (multiple-input and multiple-output)
    generalization of the SISO (single-input and single-output) transfer function.

    It is a matrix of transfer functions (``TransferFunction``, SISO-``Series`` or SISO-``Parallel``).
    There is only one argument, ``arg`` which is also the compulsory argument.
    ``arg`` is expected to be strictly of the type list of lists
    which holds the transfer functions or reducible to transfer functions.

    Parameters
    ==========

    arg : Nested ``List`` (strictly).
        Users are expected to input a nested list of ``TransferFunction``, ``Series``
        and/or ``Parallel`` objects.

    Examples
    ========

    .. note::
        ``pprint()`` can be used for better visualization of ``TransferFunctionMatrix`` objects.

    >>> from sympy.abc import s, p, a
    >>> from sympy import pprint
    >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, Series, Parallel
    >>> tf_1 = TransferFunction(s + a, s**2 + s + 1, s)
    >>> tf_2 = TransferFunction(p**4 - 3*p + 2, s + p, s)
    >>> tf_3 = TransferFunction(3, s + 2, s)
    >>> tf_4 = TransferFunction(-a + p, 9*s - 9, s)
    >>> tfm_1 = TransferFunctionMatrix([[tf_1], [tf_2], [tf_3]])
    >>> tfm_1
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(3, s + 2, s),)))
    >>> tfm_1.var
    s
    >>> tfm_1.num_inputs
    1
    >>> tfm_1.num_outputs
    3
    >>> tfm_1.shape
    (3, 1)
    >>> tfm_1.args
    (((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(3, s + 2, s),)),)
    >>> tfm_2 = TransferFunctionMatrix([[tf_1, -tf_3], [tf_2, -tf_1], [tf_3, -tf_2]])
    >>> tfm_2
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(p**4 - 3*p + 2, p + s, s), TransferFunction(-a - s, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-p**4 + 3*p - 2, p + s, s))))
    >>> pprint(tfm_2, use_unicode=False)  # pretty-printing for better visualization
    [   a + s           -3       ]
    [ ----------       -----     ]
    [  2               s + 2     ]
    [ s  + s + 1                 ]
    [                            ]
    [ 4                          ]
    [p  - 3*p + 2      -a - s    ]
    [------------    ----------  ]
    [   p + s         2          ]
    [                s  + s + 1  ]
    [                            ]
    [                 4          ]
    [     3        - p  + 3*p - 2]
    [   -----      --------------]
    [   s + 2          p + s     ]{t}

    TransferFunctionMatrix can be transposed, if user wants to switch the input and output transfer functions

    >>> tfm_2.transpose()
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(p**4 - 3*p + 2, p + s, s), TransferFunction(3, s + 2, s)), (TransferFunction(-3, s + 2, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(-p**4 + 3*p - 2, p + s, s))))
    >>> pprint(_, use_unicode=False)
    [             4                          ]
    [  a + s     p  - 3*p + 2        3       ]
    [----------  ------------      -----     ]
    [ 2             p + s          s + 2     ]
    [s  + s + 1                              ]
    [                                        ]
    [                             4          ]
    [   -3          -a - s     - p  + 3*p - 2]
    [  -----      ----------   --------------]
    [  s + 2       2               p + s     ]
    [             s  + s + 1                 ]{t}

    >>> tf_5 = TransferFunction(5, s, s)
    >>> tf_6 = TransferFunction(5*s, (2 + s**2), s)
    >>> tf_7 = TransferFunction(5, (s*(2 + s**2)), s)
    >>> tf_8 = TransferFunction(5, 1, s)
    >>> tfm_3 = TransferFunctionMatrix([[tf_5, tf_6], [tf_7, tf_8]])
    >>> tfm_3
    TransferFunctionMatrix(((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)), (TransferFunction(5, s*(s**2 + 2), s), TransferFunction(5, 1, s))))
    >>> pprint(tfm_3, use_unicode=False)
    [    5        5*s  ]
    [    -       ------]
    [    s        2    ]
    [            s  + 2]
    [                  ]
    [    5         5   ]
    [----------    -   ]
    [  / 2    \\    1   ]
    [s*\\s  + 2/        ]{t}
    >>> tfm_3.var
    s
    >>> tfm_3.shape
    (2, 2)
    >>> tfm_3.num_outputs
    2
    >>> tfm_3.num_inputs
    2
    >>> tfm_3.args
    (((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)), (TransferFunction(5, s*(s**2 + 2), s), TransferFunction(5, 1, s))),)

    To access the ``TransferFunction`` at any index in the ``TransferFunctionMatrix``, use the index notation.

    >>> tfm_3[1, 0]  # gives the TransferFunction present at 2nd Row and 1st Col. Similar to that in Matrix classes
    TransferFunction(5, s*(s**2 + 2), s)
    >>> tfm_3[0, 0]  # gives the TransferFunction present at 1st Row and 1st Col.
    TransferFunction(5, s, s)
    >>> tfm_3[:, 0]  # gives the first column
    TransferFunctionMatrix(((TransferFunction(5, s, s),), (TransferFunction(5, s*(s**2 + 2), s),)))
    >>> pprint(_, use_unicode=False)
    [    5     ]
    [    -     ]
    [    s     ]
    [          ]
    [    5     ]
    [----------]
    [  / 2    \\]
    [s*\\s  + 2/]{t}
    >>> tfm_3[0, :]  # gives the first row
    TransferFunctionMatrix(((TransferFunction(5, s, s), TransferFunction(5*s, s**2 + 2, s)),))
    >>> pprint(_, use_unicode=False)
    [5   5*s  ]
    [-  ------]
    [s   2    ]
    [   s  + 2]{t}

    To negate a transfer function matrix, ``-`` operator can be prepended:

    >>> tfm_4 = TransferFunctionMatrix([[tf_2], [-tf_1], [tf_3]])
    >>> -tfm_4
    TransferFunctionMatrix(((TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(-3, s + 2, s),)))
    >>> tfm_5 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, -tf_1]])
    >>> -tfm_5
    TransferFunctionMatrix(((TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(-p**4 + 3*p - 2, p + s, s)), (TransferFunction(-3, s + 2, s), TransferFunction(a + s, s**2 + s + 1, s))))

    ``subs()`` returns the ``TransferFunctionMatrix`` object with the value substituted in the expression. This will not
    mutate your original ``TransferFunctionMatrix``.

    >>> tfm_2.subs(p, 2)  #  substituting p everywhere in tfm_2 with 2.
    TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(12, s + 2, s), TransferFunction(-a - s, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-12, s + 2, s))))
    >>> pprint(_, use_unicode=False)
    [  a + s        -3     ]
    [----------    -----   ]
    [ 2            s + 2   ]
    [s  + s + 1            ]
    [                      ]
    [    12        -a - s  ]
    [  -----     ----------]
    [  s + 2      2        ]
    [            s  + s + 1]
    [                      ]
    [    3          -12    ]
    [  -----       -----   ]
    [  s + 2       s + 2   ]{t}
    >>> pprint(tfm_2, use_unicode=False) # State of tfm_2 is unchanged after substitution
    [   a + s           -3       ]
    [ ----------       -----     ]
    [  2               s + 2     ]
    [ s  + s + 1                 ]
    [                            ]
    [ 4                          ]
    [p  - 3*p + 2      -a - s    ]
    [------------    ----------  ]
    [   p + s         2          ]
    [                s  + s + 1  ]
    [                            ]
    [                 4          ]
    [     3        - p  + 3*p - 2]
    [   -----      --------------]
    [   s + 2          p + s     ]{t}

    ``subs()`` also supports multiple substitutions.

    >>> tfm_2.subs({p: 2, a: 1})  # substituting p with 2 and a with 1
    TransferFunctionMatrix(((TransferFunction(s + 1, s**2 + s + 1, s), TransferFunction(-3, s + 2, s)), (TransferFunction(12, s + 2, s), TransferFunction(-s - 1, s**2 + s + 1, s)), (TransferFunction(3, s + 2, s), TransferFunction(-12, s + 2, s))))
    >>> pprint(_, use_unicode=False)
    [  s + 1        -3     ]
    [----------    -----   ]
    [ 2            s + 2   ]
    [s  + s + 1            ]
    [                      ]
    [    12        -s - 1  ]
    [  -----     ----------]
    [  s + 2      2        ]
    [            s  + s + 1]
    [                      ]
    [    3          -12    ]
    [  -----       -----   ]
    [  s + 2       s + 2   ]{t}

    Users can reduce the ``Series`` and ``Parallel`` elements of the matrix to ``TransferFunction`` by using
    ``doit()``.

    >>> tfm_6 = TransferFunctionMatrix([[Series(tf_3, tf_4), Parallel(tf_3, tf_4)]])
    >>> tfm_6
    TransferFunctionMatrix(((Series(TransferFunction(3, s + 2, s), TransferFunction(-a + p, 9*s - 9, s)), Parallel(TransferFunction(3, s + 2, s), TransferFunction(-a + p, 9*s - 9, s))),))
    >>> pprint(tfm_6, use_unicode=False)
    [-a + p    3    -a + p      3  ]
    [-------*-----  ------- + -----]
    [9*s - 9 s + 2  9*s - 9   s + 2]{t}
    >>> tfm_6.doit()
    TransferFunctionMatrix(((TransferFunction(-3*a + 3*p, (s + 2)*(9*s - 9), s), TransferFunction(27*s + (-a + p)*(s + 2) - 27, (s + 2)*(9*s - 9), s)),))
    >>> pprint(_, use_unicode=False)
    [    -3*a + 3*p     27*s + (-a + p)*(s + 2) - 27]
    [-----------------  ----------------------------]
    [(s + 2)*(9*s - 9)       (s + 2)*(9*s - 9)      ]{t}
    >>> tf_9 = TransferFunction(1, s, s)
    >>> tf_10 = TransferFunction(1, s**2, s)
    >>> tfm_7 = TransferFunctionMatrix([[Series(tf_9, tf_10), tf_9], [tf_10, Parallel(tf_9, tf_10)]])
    >>> tfm_7
    TransferFunctionMatrix(((Series(TransferFunction(1, s, s), TransferFunction(1, s**2, s)), TransferFunction(1, s, s)), (TransferFunction(1, s**2, s), Parallel(TransferFunction(1, s, s), TransferFunction(1, s**2, s)))))
    >>> pprint(tfm_7, use_unicode=False)
    [ 1      1   ]
    [----    -   ]
    [   2    s   ]
    [s*s         ]
    [            ]
    [ 1    1    1]
    [ --   -- + -]
    [  2    2   s]
    [ s    s     ]{t}
    >>> tfm_7.doit()
    TransferFunctionMatrix(((TransferFunction(1, s**3, s), TransferFunction(1, s, s)), (TransferFunction(1, s**2, s), TransferFunction(s**2 + s, s**3, s))))
    >>> pprint(_, use_unicode=False)
    [1     1   ]
    [--    -   ]
    [ 3    s   ]
    [s         ]
    [          ]
    [     2    ]
    [1   s  + s]
    [--  ------]
    [ 2     3  ]
    [s     s   ]{t}

    Addition, subtraction, and multiplication of transfer function matrices can form
    unevaluated ``Series`` or ``Parallel`` objects.

    - For addition and subtraction:
      All the transfer function matrices must have the same shape.

    - For multiplication (C = A * B):
      The number of inputs of the first transfer function matrix (A) must be equal to the
      number of outputs of the second transfer function matrix (B).

    Also, use pretty-printing (``pprint``) to analyse better.

    >>> tfm_8 = TransferFunctionMatrix([[tf_3], [tf_2], [-tf_1]])
    >>> tfm_9 = TransferFunctionMatrix([[-tf_3]])
    >>> tfm_10 = TransferFunctionMatrix([[tf_1], [tf_2], [tf_4]])
    >>> tfm_11 = TransferFunctionMatrix([[tf_4], [-tf_1]])
    >>> tfm_12 = TransferFunctionMatrix([[tf_4, -tf_1, tf_3], [-tf_2, -tf_4, -tf_3]])
    >>> tfm_8 + tfm_10
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),))), TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a + p, 9*s - 9, s),))))
    >>> pprint(_, use_unicode=False)
    [     3      ]      [   a + s    ]
    [   -----    ]      [ ---------- ]
    [   s + 2    ]      [  2         ]
    [            ]      [ s  + s + 1 ]
    [ 4          ]      [            ]
    [p  - 3*p + 2]      [ 4          ]
    [------------]    + [p  - 3*p + 2]
    [   p + s    ]      [------------]
    [            ]      [   p + s    ]
    [   -a - s   ]      [            ]
    [ ---------- ]      [   -a + p   ]
    [  2         ]      [  -------   ]
    [ s  + s + 1 ]{t}   [  9*s - 9   ]{t}
    >>> -tfm_10 - tfm_8
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(-a - s, s**2 + s + 1, s),), (TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a - p, 9*s - 9, s),))), TransferFunctionMatrix(((TransferFunction(-3, s + 2, s),), (TransferFunction(-p**4 + 3*p - 2, p + s, s),), (TransferFunction(a + s, s**2 + s + 1, s),))))
    >>> pprint(_, use_unicode=False)
    [    -a - s    ]      [     -3       ]
    [  ----------  ]      [    -----     ]
    [   2          ]      [    s + 2     ]
    [  s  + s + 1  ]      [              ]
    [              ]      [   4          ]
    [   4          ]      [- p  + 3*p - 2]
    [- p  + 3*p - 2]    + [--------------]
    [--------------]      [    p + s     ]
    [    p + s     ]      [              ]
    [              ]      [    a + s     ]
    [    a - p     ]      [  ----------  ]
    [   -------    ]      [   2          ]
    [   9*s - 9    ]{t}   [  s  + s + 1  ]{t}
    >>> tfm_12 * tfm_8
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),))), TransferFunctionMatrix(((TransferFunction(-a + p, 9*s - 9, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(3, s + 2, s)), (TransferFunction(-p**4 + 3*p - 2, p + s, s), TransferFunction(a - p, 9*s - 9, s), TransferFunction(-3, s + 2, s)))))
    >>> pprint(_, use_unicode=False)
                                           [     3      ]
                                           [   -----    ]
    [    -a + p        -a - s      3  ]    [   s + 2    ]
    [   -------      ----------  -----]    [            ]
    [   9*s - 9       2          s + 2]    [ 4          ]
    [                s  + s + 1       ]    [p  - 3*p + 2]
    [                                 ]   *[------------]
    [   4                             ]    [   p + s    ]
    [- p  + 3*p - 2    a - p      -3  ]    [            ]
    [--------------   -------    -----]    [   -a - s   ]
    [    p + s        9*s - 9    s + 2]{t} [ ---------- ]
                                           [  2         ]
                                           [ s  + s + 1 ]{t}
    >>> tfm_12 * tfm_8 * tfm_9
    MIMOSeries(TransferFunctionMatrix(((TransferFunction(-3, s + 2, s),),)), TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),))), TransferFunctionMatrix(((TransferFunction(-a + p, 9*s - 9, s), TransferFunction(-a - s, s**2 + s + 1, s), TransferFunction(3, s + 2, s)), (TransferFunction(-p**4 + 3*p - 2, p + s, s), TransferFunction(a - p, 9*s - 9, s), TransferFunction(-3, s + 2, s)))))
    >>> pprint(_, use_unicode=False)
                                           [     3      ]
                                           [   -----    ]
    [    -a + p        -a - s      3  ]    [   s + 2    ]
    [   -------      ----------  -----]    [            ]
    [   9*s - 9       2          s + 2]    [ 4          ]
    [                s  + s + 1       ]    [p  - 3*p + 2]    [ -3  ]
    [                                 ]   *[------------]   *[-----]
    [   4                             ]    [   p + s    ]    [s + 2]{t}
    [- p  + 3*p - 2    a - p      -3  ]    [            ]
    [--------------   -------    -----]    [   -a - s   ]
    [    p + s        9*s - 9    s + 2]{t} [ ---------- ]
                                           [  2         ]
                                           [ s  + s + 1 ]{t}
    >>> tfm_10 + tfm_8*tfm_9
    MIMOParallel(TransferFunctionMatrix(((TransferFunction(a + s, s**2 + s + 1, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a + p, 9*s - 9, s),))), MIMOSeries(TransferFunctionMatrix(((TransferFunction(-3, s + 2, s),),)), TransferFunctionMatrix(((TransferFunction(3, s + 2, s),), (TransferFunction(p**4 - 3*p + 2, p + s, s),), (TransferFunction(-a - s, s**2 + s + 1, s),)))))
    >>> pprint(_, use_unicode=False)
    [   a + s    ]      [     3      ]
    [ ---------- ]      [   -----    ]
    [  2         ]      [   s + 2    ]
    [ s  + s + 1 ]      [            ]
    [            ]      [ 4          ]
    [ 4          ]      [p  - 3*p + 2]    [ -3  ]
    [p  - 3*p + 2]    + [------------]   *[-----]
    [------------]      [   p + s    ]    [s + 2]{t}
    [   p + s    ]      [            ]
    [            ]      [   -a - s   ]
    [   -a + p   ]      [ ---------- ]
    [  -------   ]      [  2         ]
    [  9*s - 9   ]{t}   [ s  + s + 1 ]{t}

    These unevaluated ``Series`` or ``Parallel`` objects can convert into the
    resultant transfer function matrix using ``.doit()`` method or by
    ``.rewrite(TransferFunctionMatrix)``.

    >>> (-tfm_8 + tfm_10 + tfm_8*tfm_9).doit()
    TransferFunctionMatrix(((TransferFunction((a + s)*(s + 2)**3 - 3*(s + 2)**2*(s**2 + s + 1) - 9*(s + 2)*(s**2 + s + 1), (s + 2)**3*(s**2 + s + 1), s),), (TransferFunction((p + s)*(-3*p**4 + 9*p - 6), (p + s)**2*(s + 2), s),), (TransferFunction((-a + p)*(s + 2)*(s**2 + s + 1)**2 + (a + s)*(s + 2)*(9*s - 9)*(s**2 + s + 1) + (3*a + 3*s)*(9*s - 9)*(s**2 + s + 1), (s + 2)*(9*s - 9)*(s**2 + s + 1)**2, s),)))
    >>> (-tfm_12 * -tfm_8 * -tfm_9).rewrite(TransferFunctionMatrix)
    TransferFunctionMatrix(((TransferFunction(3*(-3*a + 3*p)*(p + s)*(s + 2)*(s**2 + s + 1)**2 + 3*(-3*a - 3*s)*(p + s)*(s + 2)*(9*s - 9)*(s**2 + s + 1) + 3*(a + s)*(s + 2)**2*(9*s - 9)*(-p**4 + 3*p - 2)*(s**2 + s + 1), (p + s)*(s + 2)**3*(9*s - 9)*(s**2 + s + 1)**2, s),), (TransferFunction(3*(-a + p)*(p + s)*(s + 2)**2*(-p**4 + 3*p - 2)*(s**2 + s + 1) + 3*(3*a + 3*s)*(p + s)**2*(s + 2)*(9*s - 9) + 3*(p + s)*(s + 2)*(9*s - 9)*(-3*p**4 + 9*p - 6)*(s**2 + s + 1), (p + s)**2*(s + 2)**3*(9*s - 9)*(s**2 + s + 1), s),)))

    See Also
    ========

    TransferFunction, MIMOSeries, MIMOParallel, Feedback

    """

    def __new__(cls, arg):
        if False:
            return 10
        expr_mat_arg = []
        try:
            var = arg[0][0].var
        except TypeError:
            raise ValueError(filldedent('\n                `arg` param in TransferFunctionMatrix should\n                strictly be a nested list containing TransferFunction\n                objects.'))
        for (row_index, row) in enumerate(arg):
            temp = []
            for (col_index, element) in enumerate(row):
                if not isinstance(element, SISOLinearTimeInvariant):
                    raise TypeError(filldedent('\n                        Each element is expected to be of\n                        type `SISOLinearTimeInvariant`.'))
                if var != element.var:
                    raise ValueError(filldedent('\n                        Conflicting value(s) found for `var`. All TransferFunction\n                        instances in TransferFunctionMatrix should use the same\n                        complex variable in Laplace domain.'))
                temp.append(element.to_expr())
            expr_mat_arg.append(temp)
        if isinstance(arg, (tuple, list, Tuple)):
            arg = Tuple(*(Tuple(*r, sympify=False) for r in arg), sympify=False)
        obj = super(TransferFunctionMatrix, cls).__new__(cls, arg)
        obj._expr_mat = ImmutableMatrix(expr_mat_arg)
        return obj

    @classmethod
    def from_Matrix(cls, matrix, var):
        if False:
            return 10
        '\n        Creates a new ``TransferFunctionMatrix`` efficiently from a SymPy Matrix of ``Expr`` objects.\n\n        Parameters\n        ==========\n\n        matrix : ``ImmutableMatrix`` having ``Expr``/``Number`` elements.\n        var : Symbol\n            Complex variable of the Laplace transform which will be used by the\n            all the ``TransferFunction`` objects in the ``TransferFunctionMatrix``.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunctionMatrix\n        >>> from sympy import Matrix, pprint\n        >>> M = Matrix([[s, 1/s], [1/(s+1), s]])\n        >>> M_tf = TransferFunctionMatrix.from_Matrix(M, s)\n        >>> pprint(M_tf, use_unicode=False)\n        [  s    1]\n        [  -    -]\n        [  1    s]\n        [        ]\n        [  1    s]\n        [-----  -]\n        [s + 1  1]{t}\n        >>> M_tf.elem_poles()\n        [[[], [0]], [[-1], []]]\n        >>> M_tf.elem_zeros()\n        [[[0], []], [[], [0]]]\n\n        '
        return _to_TFM(matrix, var)

    @property
    def var(self):
        if False:
            return 10
        '\n        Returns the complex variable used by all the transfer functions or\n        ``Series``/``Parallel`` objects in a transfer function matrix.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import p, s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix, Series, Parallel\n        >>> G1 = TransferFunction(p**2 + 2*p + 4, p - 6, p)\n        >>> G2 = TransferFunction(p, 4 - p, p)\n        >>> G3 = TransferFunction(0, p**4 - 1, p)\n        >>> G4 = TransferFunction(s + 1, s**2 + s + 1, s)\n        >>> S1 = Series(G1, G2)\n        >>> S2 = Series(-G3, Parallel(G2, -G1))\n        >>> tfm1 = TransferFunctionMatrix([[G1], [G2], [G3]])\n        >>> tfm1.var\n        p\n        >>> tfm2 = TransferFunctionMatrix([[-S1, -S2], [S1, S2]])\n        >>> tfm2.var\n        p\n        >>> tfm3 = TransferFunctionMatrix([[G4]])\n        >>> tfm3.var\n        s\n\n        '
        return self.args[0][0][0].var

    @property
    def num_inputs(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of inputs of the system.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix\n        >>> G1 = TransferFunction(s + 3, s**2 - 3, s)\n        >>> G2 = TransferFunction(4, s**2, s)\n        >>> G3 = TransferFunction(p**2 + s**2, p - 3, s)\n        >>> tfm_1 = TransferFunctionMatrix([[G2, -G1, G3], [-G2, -G1, -G3]])\n        >>> tfm_1.num_inputs\n        3\n\n        See Also\n        ========\n\n        num_outputs\n\n        '
        return self._expr_mat.shape[1]

    @property
    def num_outputs(self):
        if False:
            while True:
                i = 10
        '\n        Returns the number of outputs of the system.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunctionMatrix\n        >>> from sympy import Matrix\n        >>> M_1 = Matrix([[s], [1/s]])\n        >>> TFM = TransferFunctionMatrix.from_Matrix(M_1, s)\n        >>> print(TFM)\n        TransferFunctionMatrix(((TransferFunction(s, 1, s),), (TransferFunction(1, s, s),)))\n        >>> TFM.num_outputs\n        2\n\n        See Also\n        ========\n\n        num_inputs\n\n        '
        return self._expr_mat.shape[0]

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the shape of the transfer function matrix, that is, ``(# of outputs, # of inputs)``.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s, p\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix\n        >>> tf1 = TransferFunction(p**2 - 1, s**4 + s**3 - p, p)\n        >>> tf2 = TransferFunction(1 - p, p**2 - 3*p + 7, p)\n        >>> tf3 = TransferFunction(3, 4, p)\n        >>> tfm1 = TransferFunctionMatrix([[tf1, -tf2]])\n        >>> tfm1.shape\n        (1, 2)\n        >>> tfm2 = TransferFunctionMatrix([[-tf2, tf3], [tf1, -tf1]])\n        >>> tfm2.shape\n        (2, 2)\n\n        '
        return self._expr_mat.shape

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        neg = -self._expr_mat
        return _to_TFM(neg, self.var)

    @_check_other_MIMO
    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, MIMOParallel):
            return MIMOParallel(self, other)
        other_arg_list = list(other.args)
        return MIMOParallel(self, *other_arg_list)

    @_check_other_MIMO
    def __sub__(self, other):
        if False:
            return 10
        return self + -other

    @_check_other_MIMO
    def __mul__(self, other):
        if False:
            return 10
        if not isinstance(other, MIMOSeries):
            return MIMOSeries(other, self)
        other_arg_list = list(other.args)
        return MIMOSeries(*other_arg_list, self)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        trunc = self._expr_mat.__getitem__(key)
        if isinstance(trunc, ImmutableMatrix):
            return _to_TFM(trunc, self.var)
        return TransferFunction.from_rational_expression(trunc, self.var)

    def transpose(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the transpose of the ``TransferFunctionMatrix`` (switched input and output layers).'
        transposed_mat = self._expr_mat.transpose()
        return _to_TFM(transposed_mat, self.var)

    def elem_poles(self):
        if False:
            return 10
        '\n        Returns the poles of each element of the ``TransferFunctionMatrix``.\n\n        .. note::\n            Actual poles of a MIMO system are NOT the poles of individual elements.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix\n        >>> tf_1 = TransferFunction(3, (s + 1), s)\n        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)\n        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)\n        >>> tf_4 = TransferFunction(s + 2, s**2 + 5*s - 10, s)\n        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])\n        >>> tfm_1\n        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s + 2, s**2 + 5*s - 10, s))))\n        >>> tfm_1.elem_poles()\n        [[[-1], [-2, -1]], [[-2, -1], [-5/2 + sqrt(65)/2, -sqrt(65)/2 - 5/2]]]\n\n        See Also\n        ========\n\n        elem_zeros\n\n        '
        return [[element.poles() for element in row] for row in self.doit().args[0]]

    def elem_zeros(self):
        if False:
            return 10
        '\n        Returns the zeros of each element of the ``TransferFunctionMatrix``.\n\n        .. note::\n            Actual zeros of a MIMO system are NOT the zeros of individual elements.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix\n        >>> tf_1 = TransferFunction(3, (s + 1), s)\n        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)\n        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)\n        >>> tf_4 = TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s)\n        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])\n        >>> tfm_1\n        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s))))\n        >>> tfm_1.elem_zeros()\n        [[[], [-6]], [[-3], [4, 5]]]\n\n        See Also\n        ========\n\n        elem_poles\n\n        '
        return [[element.zeros() for element in row] for row in self.doit().args[0]]

    def eval_frequency(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluates system response of each transfer function in the ``TransferFunctionMatrix`` at any point in the real or complex plane.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import s\n        >>> from sympy.physics.control.lti import TransferFunction, TransferFunctionMatrix\n        >>> from sympy import I\n        >>> tf_1 = TransferFunction(3, (s + 1), s)\n        >>> tf_2 = TransferFunction(s + 6, (s + 1)*(s + 2), s)\n        >>> tf_3 = TransferFunction(s + 3, s**2 + 3*s + 2, s)\n        >>> tf_4 = TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s)\n        >>> tfm_1 = TransferFunctionMatrix([[tf_1, tf_2], [tf_3, tf_4]])\n        >>> tfm_1\n        TransferFunctionMatrix(((TransferFunction(3, s + 1, s), TransferFunction(s + 6, (s + 1)*(s + 2), s)), (TransferFunction(s + 3, s**2 + 3*s + 2, s), TransferFunction(s**2 - 9*s + 20, s**2 + 5*s - 10, s))))\n        >>> tfm_1.eval_frequency(2)\n        Matrix([\n        [   1, 2/3],\n        [5/12, 3/2]])\n        >>> tfm_1.eval_frequency(I*2)\n        Matrix([\n        [   3/5 - 6*I/5,                -I],\n        [3/20 - 11*I/20, -101/74 + 23*I/74]])\n        '
        mat = self._expr_mat.subs(self.var, other)
        return mat.expand()

    def _flat(self):
        if False:
            i = 10
            return i + 15
        'Returns flattened list of args in TransferFunctionMatrix'
        return [elem for tup in self.args[0] for elem in tup]

    def _eval_evalf(self, prec):
        if False:
            for i in range(10):
                print('nop')
        'Calls evalf() on each transfer function in the transfer function matrix'
        dps = prec_to_dps(prec)
        mat = self._expr_mat.applyfunc(lambda a: a.evalf(n=dps))
        return _to_TFM(mat, self.var)

    def _eval_simplify(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Simplifies the transfer function matrix'
        simp_mat = self._expr_mat.applyfunc(lambda a: cancel(a, expand=False))
        return _to_TFM(simp_mat, self.var)

    def expand(self, **hints):
        if False:
            print('Hello World!')
        'Expands the transfer function matrix'
        expand_mat = self._expr_mat.expand(**hints)
        return _to_TFM(expand_mat, self.var)

class StateSpace(LinearTimeInvariant):
    """
    State space model (ssm) of a linear, time invariant control system.

    Represents the standard state-space model with A, B, C, D as state-space matrices.
    This makes the linear control system:
        (1) x'(t) = A * x(t) + B * u(t);    x in R^n , u in R^k
        (2) y(t)  = C * x(t) + D * u(t);    y in R^m
    where u(t) is any input signal, y(t) the corresponding output, and x(t) the system's state.

    Parameters
    ==========

    A : Matrix
        The State matrix of the state space model.
    B : Matrix
        The Input-to-State matrix of the state space model.
    C : Matrix
        The State-to-Output matrix of the state space model.
    D : Matrix
        The Feedthrough matrix of the state space model.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.physics.control import StateSpace

    The easiest way to create a StateSpaceModel is via four matrices:

    >>> A = Matrix([[1, 2], [1, 0]])
    >>> B = Matrix([1, 1])
    >>> C = Matrix([[0, 1]])
    >>> D = Matrix([0])
    >>> StateSpace(A, B, C, D)
    StateSpace(Matrix([
    [1, 2],
    [1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 1]]), Matrix([[0]]))


    One can use less matrices. The rest will be filled with a minimum of zeros:

    >>> StateSpace(A, B)
    StateSpace(Matrix([
    [1, 2],
    [1, 0]]), Matrix([
    [1],
    [1]]), Matrix([[0, 0]]), Matrix([[0]]))


    See Also
    ========

    TransferFunction, TransferFunctionMatrix

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/State-space_representation
    .. [2] https://in.mathworks.com/help/control/ref/ss.html

    """

    def __new__(cls, A=None, B=None, C=None, D=None):
        if False:
            for i in range(10):
                print('nop')
        if A is None:
            A = zeros(1)
        if B is None:
            B = zeros(A.rows, 1)
        if C is None:
            C = zeros(1, A.cols)
        if D is None:
            D = zeros(C.rows, B.cols)
        A = _sympify(A)
        B = _sympify(B)
        C = _sympify(C)
        D = _sympify(D)
        if isinstance(A, ImmutableDenseMatrix) and isinstance(B, ImmutableDenseMatrix) and isinstance(C, ImmutableDenseMatrix) and isinstance(D, ImmutableDenseMatrix):
            if A.rows != A.cols:
                raise ShapeError('Matrix A must be a square matrix.')
            if A.rows != B.rows:
                raise ShapeError('Matrices A and B must have the same number of rows.')
            if C.rows != D.rows:
                raise ShapeError('Matrices C and D must have the same number of rows.')
            if A.cols != C.cols:
                raise ShapeError('Matrices A and C must have the same number of columns.')
            if B.cols != D.cols:
                raise ShapeError('Matrices B and D must have the same number of columns.')
            obj = super(StateSpace, cls).__new__(cls, A, B, C, D)
            obj._A = A
            obj._B = B
            obj._C = C
            obj._D = D
            num_outputs = D.rows
            num_inputs = D.cols
            if num_inputs == 1 and num_outputs == 1:
                obj._is_SISO = True
                obj._clstype = SISOLinearTimeInvariant
            else:
                obj._is_SISO = False
                obj._clstype = MIMOLinearTimeInvariant
            return obj
        else:
            raise TypeError('A, B, C and D inputs must all be sympy Matrices.')

    @property
    def state_matrix(self):
        if False:
            print('Hello World!')
        '\n        Returns the state matrix of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.state_matrix\n        Matrix([\n        [1, 2],\n        [1, 0]])\n\n        '
        return self._A

    @property
    def input_matrix(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the input matrix of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.input_matrix\n        Matrix([\n        [1],\n        [1]])\n\n        '
        return self._B

    @property
    def output_matrix(self):
        if False:
            return 10
        '\n        Returns the output matrix of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.output_matrix\n        Matrix([[0, 1]])\n\n        '
        return self._C

    @property
    def feedforward_matrix(self):
        if False:
            print('Hello World!')
        '\n        Returns the feedforward matrix of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.feedforward_matrix\n        Matrix([[0]])\n\n        '
        return self._D

    @property
    def num_states(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of states of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.num_states\n        2\n\n        '
        return self._A.rows

    @property
    def num_inputs(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of inputs of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.num_inputs\n        1\n\n        '
        return self._D.cols

    @property
    def num_outputs(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of outputs of the model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[1, 2], [1, 0]])\n        >>> B = Matrix([1, 1])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.num_outputs\n        1\n\n        '
        return self._D.rows

    def _eval_evalf(self, prec):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns state space model where numerical expressions are evaluated into floating point numbers.\n        '
        dps = prec_to_dps(prec)
        return StateSpace(self._A.evalf(n=dps), self._B.evalf(n=dps), self._C.evalf(n=dps), self._D.evalf(n=dps))

    def _eval_rewrite_as_TransferFunction(self, *args):
        if False:
            while True:
                i = 10
        '\n        Returns the equivalent Transfer Function of the state space model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import TransferFunction, StateSpace\n        >>> A = Matrix([[-5, -1], [3, -1]])\n        >>> B = Matrix([2, 5])\n        >>> C = Matrix([[1, 2]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.rewrite(TransferFunction)\n        [[TransferFunction(12*s + 59, s**2 + 6*s + 8, s)]]\n\n        '
        s = Symbol('s')
        n = self._A.shape[0]
        I = eye(n)
        G = self._C * (s * I - self._A).solve(self._B) + self._D
        G = G.simplify()
        to_tf = lambda expr: TransferFunction.from_rational_expression(expr, s)
        tf_mat = [[to_tf(expr) for expr in sublist] for sublist in G.tolist()]
        return tf_mat

    def __add__(self, other):
        if False:
            print('Hello World!')
        '\n        Add two State Space systems (parallel connection).\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A1 = Matrix([[1]])\n        >>> B1 = Matrix([[2]])\n        >>> C1 = Matrix([[-1]])\n        >>> D1 = Matrix([[-2]])\n        >>> A2 = Matrix([[-1]])\n        >>> B2 = Matrix([[-2]])\n        >>> C2 = Matrix([[1]])\n        >>> D2 = Matrix([[2]])\n        >>> ss1 = StateSpace(A1, B1, C1, D1)\n        >>> ss2 = StateSpace(A2, B2, C2, D2)\n        >>> ss1 + ss2\n        StateSpace(Matrix([\n        [1,  0],\n        [0, -1]]), Matrix([\n        [ 2],\n        [-2]]), Matrix([[-1, 1]]), Matrix([[0]]))\n\n        '
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            B = self._B
            C = self._C
            D = self._D.applyfunc(lambda element: element + other)
        else:
            if not isinstance(other, StateSpace):
                raise ValueError('Addition is only supported for 2 State Space models.')
            elif self.num_inputs != other.num_inputs or self.num_outputs != other.num_outputs:
                raise ShapeError('Systems with incompatible inputs and outputs cannot be added.')
            m1 = self._A.row_join(zeros(self._A.shape[0], other._A.shape[-1]))
            m2 = zeros(other._A.shape[0], self._A.shape[-1]).row_join(other._A)
            A = m1.col_join(m2)
            B = self._B.col_join(other._B)
            C = self._C.row_join(other._C)
            D = self._D + other._D
        return StateSpace(A, B, C, D)

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        '\n        Right add two State Space systems.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.control import StateSpace\n        >>> s = StateSpace()\n        >>> 5 + s\n        StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[5]]))\n\n        '
        return self + other

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        '\n        Subtract two State Space systems.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A1 = Matrix([[1]])\n        >>> B1 = Matrix([[2]])\n        >>> C1 = Matrix([[-1]])\n        >>> D1 = Matrix([[-2]])\n        >>> A2 = Matrix([[-1]])\n        >>> B2 = Matrix([[-2]])\n        >>> C2 = Matrix([[1]])\n        >>> D2 = Matrix([[2]])\n        >>> ss1 = StateSpace(A1, B1, C1, D1)\n        >>> ss2 = StateSpace(A2, B2, C2, D2)\n        >>> ss1 - ss2\n        StateSpace(Matrix([\n        [1,  0],\n        [0, -1]]), Matrix([\n        [ 2],\n        [-2]]), Matrix([[-1, -1]]), Matrix([[-4]]))\n\n        '
        return self + -other

    def __rsub__(self, other):
        if False:
            return 10
        '\n        Right subtract two tate Space systems.\n\n        Examples\n        ========\n\n        >>> from sympy.physics.control import StateSpace\n        >>> s = StateSpace()\n        >>> 5 - s\n        StateSpace(Matrix([[0]]), Matrix([[0]]), Matrix([[0]]), Matrix([[5]]))\n\n        '
        return other + -self

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the negation of the state space model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-5, -1], [3, -1]])\n        >>> B = Matrix([2, 5])\n        >>> C = Matrix([[1, 2]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> -ss\n        StateSpace(Matrix([\n        [-5, -1],\n        [ 3, -1]]), Matrix([\n        [2],\n        [5]]), Matrix([[-1, -2]]), Matrix([[0]]))\n\n        '
        return StateSpace(self._A, self._B, -self._C, -self._D)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Multiplication of two State Space systems (serial connection).\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-5, -1], [3, -1]])\n        >>> B = Matrix([2, 5])\n        >>> C = Matrix([[1, 2]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss*5\n        StateSpace(Matrix([\n        [-5, -1],\n        [ 3, -1]]), Matrix([\n        [2],\n        [5]]), Matrix([[5, 10]]), Matrix([[0]]))\n\n        '
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            B = self._B
            C = self._C.applyfunc(lambda element: element * other)
            D = self._D.applyfunc(lambda element: element * other)
        else:
            if not isinstance(other, StateSpace):
                raise ValueError('Multiplication is only supported for 2 State Space models.')
            elif self.num_inputs != other.num_outputs:
                raise ShapeError('Systems with incompatible inputs and outputs cannot be multiplied.')
            m1 = other._A.row_join(zeros(other._A.shape[0], self._A.shape[1]))
            m2 = (self._B * other._C).row_join(self._A)
            A = m1.col_join(m2)
            B = other._B.col_join(self._B * other._D)
            C = (self._D * other._C).row_join(self._C)
            D = self._D * other._D
        return StateSpace(A, B, C, D)

    def __rmul__(self, other):
        if False:
            return 10
        '\n        Right multiply two tate Space systems.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-5, -1], [3, -1]])\n        >>> B = Matrix([2, 5])\n        >>> C = Matrix([[1, 2]])\n        >>> D = Matrix([0])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> 5*ss\n        StateSpace(Matrix([\n        [-5, -1],\n        [ 3, -1]]), Matrix([\n        [10],\n        [25]]), Matrix([[1, 2]]), Matrix([[0]]))\n\n        '
        if isinstance(other, (int, float, complex, Symbol)):
            A = self._A
            C = self._C
            B = self._B.applyfunc(lambda element: element * other)
            D = self._D.applyfunc(lambda element: element * other)
            return StateSpace(A, B, C, D)
        else:
            return self * other

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        A_str = self._A.__repr__()
        B_str = self._B.__repr__()
        C_str = self._C.__repr__()
        D_str = self._D.__repr__()
        return f'StateSpace(\n{A_str},\n\n{B_str},\n\n{C_str},\n\n{D_str})'

    def append(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Returns the first model appended with the second model. The order is preserved.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A1 = Matrix([[1]])\n        >>> B1 = Matrix([[2]])\n        >>> C1 = Matrix([[-1]])\n        >>> D1 = Matrix([[-2]])\n        >>> A2 = Matrix([[-1]])\n        >>> B2 = Matrix([[-2]])\n        >>> C2 = Matrix([[1]])\n        >>> D2 = Matrix([[2]])\n        >>> ss1 = StateSpace(A1, B1, C1, D1)\n        >>> ss2 = StateSpace(A2, B2, C2, D2)\n        >>> ss1.append(ss2)\n        StateSpace(Matrix([\n        [1,  0],\n        [0, -1]]), Matrix([\n        [2,  0],\n        [0, -2]]), Matrix([\n        [-1, 0],\n        [ 0, 1]]), Matrix([\n        [-2, 0],\n        [ 0, 2]]))\n\n        '
        n = self.num_states + other.num_states
        m = self.num_inputs + other.num_inputs
        p = self.num_outputs + other.num_outputs
        A = zeros(n, n)
        B = zeros(n, m)
        C = zeros(p, n)
        D = zeros(p, m)
        A[:self.num_states, :self.num_states] = self._A
        A[self.num_states:, self.num_states:] = other._A
        B[:self.num_states, :self.num_inputs] = self._B
        B[self.num_states:, self.num_inputs:] = other._B
        C[:self.num_outputs, :self.num_states] = self._C
        C[self.num_outputs:, self.num_states:] = other._C
        D[:self.num_outputs, :self.num_inputs] = self._D
        D[self.num_outputs:, self.num_inputs:] = other._D
        return StateSpace(A, B, C, D)

    def observability_matrix(self):
        if False:
            return 10
        '\n        Returns the observability matrix of the state space model:\n            [C, C * A^1, C * A^2, .. , C * A^(n-1)]; A in R^(n x n), C in R^(m x k)\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-1.5, -2], [1, 0]])\n        >>> B = Matrix([0.5, 0])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([1])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ob = ss.observability_matrix()\n        >>> ob\n        Matrix([\n        [0, 1],\n        [1, 0]])\n\n        References\n        ==========\n        .. [1] https://in.mathworks.com/help/control/ref/statespacemodel.obsv.html\n\n        '
        n = self.num_states
        ob = self._C
        for i in range(1, n):
            ob = ob.col_join(self._C * self._A ** i)
        return ob

    def observable_subspace(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the observable subspace of the state space model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-1.5, -2], [1, 0]])\n        >>> B = Matrix([0.5, 0])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([1])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ob_subspace = ss.observable_subspace()\n        >>> ob_subspace\n        [Matrix([\n        [0],\n        [1]]), Matrix([\n        [1],\n        [0]])]\n\n        '
        return self.observability_matrix().columnspace()

    def is_observable(self):
        if False:
            print('Hello World!')
        '\n        Returns if the state space model is observable.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-1.5, -2], [1, 0]])\n        >>> B = Matrix([0.5, 0])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([1])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.is_observable()\n        True\n\n        '
        return self.observability_matrix().rank() == self.num_states

    def controllability_matrix(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the controllability matrix of the system:\n            [B, A * B, A^2 * B, .. , A^(n-1) * B]; A in R^(n x n), B in R^(n x m)\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-1.5, -2], [1, 0]])\n        >>> B = Matrix([0.5, 0])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([1])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.controllability_matrix()\n        Matrix([\n        [0.5, -0.75],\n        [  0,   0.5]])\n\n        References\n        ==========\n        .. [1] https://in.mathworks.com/help/control/ref/statespacemodel.ctrb.html\n\n        '
        co = self._B
        n = self._A.shape[0]
        for i in range(1, n):
            co = co.row_join(self._A ** i * self._B)
        return co

    def controllable_subspace(self):
        if False:
            print('Hello World!')
        '\n        Returns the controllable subspace of the state space model.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-1.5, -2], [1, 0]])\n        >>> B = Matrix([0.5, 0])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([1])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> co_subspace = ss.controllable_subspace()\n        >>> co_subspace\n        [Matrix([\n        [0.5],\n        [  0]]), Matrix([\n        [-0.75],\n        [  0.5]])]\n\n        '
        return self.controllability_matrix().columnspace()

    def is_controllable(self):
        if False:
            while True:
                i = 10
        '\n        Returns if the state space model is controllable.\n\n        Examples\n        ========\n\n        >>> from sympy import Matrix\n        >>> from sympy.physics.control import StateSpace\n        >>> A = Matrix([[-1.5, -2], [1, 0]])\n        >>> B = Matrix([0.5, 0])\n        >>> C = Matrix([[0, 1]])\n        >>> D = Matrix([1])\n        >>> ss = StateSpace(A, B, C, D)\n        >>> ss.is_controllable()\n        True\n\n        '
        return self.controllability_matrix().rank() == self.num_states