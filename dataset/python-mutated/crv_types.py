"""
Continuous Random Variables - Prebuilt variables

Contains
========
Arcsin
Benini
Beta
BetaNoncentral
BetaPrime
BoundedPareto
Cauchy
Chi
ChiNoncentral
ChiSquared
Dagum
Davis
Erlang
ExGaussian
Exponential
ExponentialPower
FDistribution
FisherZ
Frechet
Gamma
GammaInverse
Gumbel
Gompertz
Kumaraswamy
Laplace
Levy
LogCauchy
Logistic
LogLogistic
LogitNormal
LogNormal
Lomax
Maxwell
Moyal
Nakagami
Normal
Pareto
PowerFunction
QuadraticU
RaisedCosine
Rayleigh
Reciprocal
ShiftedGompertz
StudentT
Trapezoidal
Triangular
Uniform
UniformSum
VonMises
Wald
Weibull
WignerSemicircle
"""
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
from sympy.functions.special.bessel import besseli, besselj, besselk
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I, Rational, pi
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import erf, erfc, erfi, erfinv, expint
from sympy.functions.special.gamma_functions import gamma, lowergamma, uppergamma
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random
oo = S.Infinity
__all__ = ['ContinuousRV', 'Arcsin', 'Benini', 'Beta', 'BetaNoncentral', 'BetaPrime', 'BoundedPareto', 'Cauchy', 'Chi', 'ChiNoncentral', 'ChiSquared', 'Dagum', 'Davis', 'Erlang', 'ExGaussian', 'Exponential', 'ExponentialPower', 'FDistribution', 'FisherZ', 'Frechet', 'Gamma', 'GammaInverse', 'Gompertz', 'Gumbel', 'Kumaraswamy', 'Laplace', 'Levy', 'LogCauchy', 'Logistic', 'LogLogistic', 'LogitNormal', 'LogNormal', 'Lomax', 'Maxwell', 'Moyal', 'Nakagami', 'Normal', 'GaussianInverse', 'Pareto', 'PowerFunction', 'QuadraticU', 'RaisedCosine', 'Rayleigh', 'Reciprocal', 'StudentT', 'ShiftedGompertz', 'Trapezoidal', 'Triangular', 'Uniform', 'UniformSum', 'VonMises', 'Wald', 'Weibull', 'WignerSemicircle']

@is_random.register(MatrixBase)
def _(x):
    if False:
        for i in range(10):
            print('nop')
    return any((is_random(i) for i in x))

def rv(symbol, cls, args, **kwargs):
    if False:
        print('Hello World!')
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleContinuousPSpace(symbol, dist)
    if any((is_random(arg) for arg in args)):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value

class ContinuousDistributionHandmade(SingleContinuousDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=Interval(-oo, oo)):
        if False:
            return 10
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        if False:
            i = 10
            return i + 15
        x = Dummy('x')
        val = integrate(pdf(x), (x, set))
        _value_check(Eq(val, 1) != S.false, 'The pdf on the given set is incorrect.')

def ContinuousRV(symbol, density, set=Interval(-oo, oo), **kwargs):
    if False:
        return 10
    '\n    Create a Continuous Random Variable given the following:\n\n    Parameters\n    ==========\n\n    symbol : Symbol\n        Represents name of the random variable.\n    density : Expression containing symbol\n        Represents probability density function.\n    set : set/Interval\n        Represents the region where the pdf is valid, by default is real line.\n    check : bool\n        If True, it will check whether the given density\n        integrates to 1 over the given set. If False, it\n        will not perform this check. Default is False.\n\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Many common continuous random variable types are already implemented.\n    This function should be necessary only very rarely.\n\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol, sqrt, exp, pi\n    >>> from sympy.stats import ContinuousRV, P, E\n\n    >>> x = Symbol("x")\n\n    >>> pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)) # Normal distribution\n    >>> X = ContinuousRV(x, pdf)\n\n    >>> E(X)\n    0\n    >>> P(X>0)\n    1/2\n    '
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, ContinuousDistributionHandmade, (pdf, set), **kwargs)

class ArcsinDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        return Interval(self.a, self.b)

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (a, b) = (self.a, self.b)
        return 1 / (pi * sqrt((x - a) * (b - x)))

    def _cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (self.a, self.b)
        return Piecewise((S.Zero, x < a), (2 * asin(sqrt((x - a) / (b - a))) / pi, x <= b), (S.One, True))

def Arcsin(name, a=0, b=1):
    if False:
        return 10
    '\n    Create a Continuous Random Variable with an arcsin distribution.\n\n    The density of the arcsin distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{\\pi\\sqrt{(x-a)(b-x)}}\n\n    with :math:`x \\in (a,b)`. It must hold that :math:`-\\infty < a < b < \\infty`.\n\n    Parameters\n    ==========\n\n    a : Real number, the left interval boundary\n    b : Real number, the right interval boundary\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Arcsin, density, cdf\n    >>> from sympy import Symbol\n\n    >>> a = Symbol("a", real=True)\n    >>> b = Symbol("b", real=True)\n    >>> z = Symbol("z")\n\n    >>> X = Arcsin("x", a, b)\n\n    >>> density(X)(z)\n    1/(pi*sqrt((-a + z)*(b - z)))\n\n    >>> cdf(X)(z)\n    Piecewise((0, a > z),\n            (2*asin(sqrt((-a + z)/(-a + b)))/pi, b >= z),\n            (1, True))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Arcsine_distribution\n\n    '
    return rv(name, ArcsinDistribution, (a, b))

class BeniniDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'sigma')

    @staticmethod
    def check(alpha, beta, sigma):
        if False:
            print('Hello World!')
        _value_check(alpha > 0, 'Shape parameter Alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter Beta must be positive.')
        _value_check(sigma > 0, 'Scale parameter Sigma must be positive.')

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        return Interval(self.sigma, oo)

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta, sigma) = (self.alpha, self.beta, self.sigma)
        return exp(-alpha * log(x / sigma) - beta * log(x / sigma) ** 2) * (alpha / x + 2 * beta * log(x / sigma) / x)

    def _moment_generating_function(self, t):
        if False:
            return 10
        raise NotImplementedError('The moment generating function of the Benini distribution does not exist.')

def Benini(name, alpha, beta, sigma):
    if False:
        print('Hello World!')
    '\n    Create a Continuous Random Variable with a Benini distribution.\n\n    The density of the Benini distribution is given by\n\n    .. math::\n        f(x) := e^{-\\alpha\\log{\\frac{x}{\\sigma}}\n                -\\beta\\log^2\\left[{\\frac{x}{\\sigma}}\\right]}\n                \\left(\\frac{\\alpha}{x}+\\frac{2\\beta\\log{\\frac{x}{\\sigma}}}{x}\\right)\n\n    This is a heavy-tailed distribution and is also known as the log-Rayleigh\n    distribution.\n\n    Parameters\n    ==========\n\n    alpha : Real number, `\\alpha > 0`, a shape\n    beta : Real number, `\\beta > 0`, a shape\n    sigma : Real number, `\\sigma > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Benini, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> sigma = Symbol("sigma", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Benini("x", alpha, beta, sigma)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n    /                  /  z  \\\\             /  z  \\            2/  z  \\\n    |        2*beta*log|-----||  - alpha*log|-----| - beta*log  |-----|\n    |alpha             \\sigma/|             \\sigma/             \\sigma/\n    |----- + -----------------|*e\n    \\  z             z        /\n\n    >>> cdf(X)(z)\n    Piecewise((1 - exp(-alpha*log(z/sigma) - beta*log(z/sigma)**2), sigma <= z),\n            (0, True))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Benini_distribution\n    .. [2] https://reference.wolfram.com/legacy/v8/ref/BeniniDistribution.html\n\n    '
    return rv(name, BeniniDistribution, (alpha, beta, sigma))

class BetaDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')
    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta):
        if False:
            return 10
        _value_check(alpha > 0, 'Shape parameter Alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter Beta must be positive.')

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta) = (self.alpha, self.beta)
        return x ** (alpha - 1) * (1 - x) ** (beta - 1) / beta_fn(alpha, beta)

    def _characteristic_function(self, t):
        if False:
            return 10
        return hyper((self.alpha,), (self.alpha + self.beta,), I * t)

    def _moment_generating_function(self, t):
        if False:
            return 10
        return hyper((self.alpha,), (self.alpha + self.beta,), t)

def Beta(name, alpha, beta):
    if False:
        i = 10
        return i + 15
    '\n    Create a Continuous Random Variable with a Beta distribution.\n\n    The density of the Beta distribution is given by\n\n    .. math::\n        f(x) := \\frac{x^{\\alpha-1}(1-x)^{\\beta-1}} {\\mathrm{B}(\\alpha,\\beta)}\n\n    with :math:`x \\in [0,1]`.\n\n    Parameters\n    ==========\n\n    alpha : Real number, `\\alpha > 0`, a shape\n    beta : Real number, `\\beta > 0`, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Beta, density, E, variance\n    >>> from sympy import Symbol, simplify, pprint, factor\n\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Beta("x", alpha, beta)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n     alpha - 1        beta - 1\n    z         *(1 - z)\n    --------------------------\n          B(alpha, beta)\n\n    >>> simplify(E(X))\n    alpha/(alpha + beta)\n\n    >>> factor(simplify(variance(X)))\n    alpha*beta/((alpha + beta)**2*(alpha + beta + 1))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Beta_distribution\n    .. [2] https://mathworld.wolfram.com/BetaDistribution.html\n\n    '
    return rv(name, BetaDistribution, (alpha, beta))

class BetaNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'lamda')
    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta, lamda):
        if False:
            return 10
        _value_check(alpha > 0, 'Shape parameter Alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter Beta must be positive.')
        _value_check(lamda >= 0, 'Noncentrality parameter Lambda must be positive')

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (alpha, beta, lamda) = (self.alpha, self.beta, self.lamda)
        k = Dummy('k')
        return Sum(exp(-lamda / 2) * (lamda / 2) ** k * x ** (alpha + k - 1) * (1 - x) ** (beta - 1) / (factorial(k) * beta_fn(alpha + k, beta)), (k, 0, oo))

def BetaNoncentral(name, alpha, beta, lamda):
    if False:
        while True:
            i = 10
    '\n    Create a Continuous Random Variable with a Type I Noncentral Beta distribution.\n\n    The density of the Noncentral Beta distribution is given by\n\n    .. math::\n        f(x) := \\sum_{k=0}^\\infty e^{-\\lambda/2}\\frac{(\\lambda/2)^k}{k!}\n                \\frac{x^{\\alpha+k-1}(1-x)^{\\beta-1}}{\\mathrm{B}(\\alpha+k,\\beta)}\n\n    with :math:`x \\in [0,1]`.\n\n    Parameters\n    ==========\n\n    alpha : Real number, `\\alpha > 0`, a shape\n    beta : Real number, `\\beta > 0`, a shape\n    lamda : Real number, `\\lambda \\geq 0`, noncentrality parameter\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import BetaNoncentral, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> lamda = Symbol("lamda", nonnegative=True)\n    >>> z = Symbol("z")\n\n    >>> X = BetaNoncentral("x", alpha, beta, lamda)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n      oo\n    _____\n    \\    `\n     \\                                              -lamda\n      \\                          k                  -------\n       \\    k + alpha - 1 /lamda\\         beta - 1     2\n        )  z             *|-----| *(1 - z)        *e\n       /                  \\  2  /\n      /    ------------------------------------------------\n     /                  B(k + alpha, beta)*k!\n    /____,\n    k = 0\n\n    Compute cdf with specific \'x\', \'alpha\', \'beta\' and \'lamda\' values as follows:\n\n    >>> cdf(BetaNoncentral("x", 1, 1, 1), evaluate=False)(2).doit()\n    2*exp(1/2)\n\n    The argument evaluate=False prevents an attempt at evaluation\n    of the sum for general x, before the argument 2 is passed.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Noncentral_beta_distribution\n    .. [2] https://reference.wolfram.com/language/ref/NoncentralBetaDistribution.html\n\n    '
    return rv(name, BetaNoncentralDistribution, (alpha, beta, lamda))

class BetaPrimeDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    @staticmethod
    def check(alpha, beta):
        if False:
            i = 10
            return i + 15
        _value_check(alpha > 0, 'Shape parameter Alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter Beta must be positive.')
    set = Interval(0, oo)

    def pdf(self, x):
        if False:
            print('Hello World!')
        (alpha, beta) = (self.alpha, self.beta)
        return x ** (alpha - 1) * (1 + x) ** (-alpha - beta) / beta_fn(alpha, beta)

def BetaPrime(name, alpha, beta):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a Beta prime distribution.\n\n    The density of the Beta prime distribution is given by\n\n    .. math::\n        f(x) := \\frac{x^{\\alpha-1} (1+x)^{-\\alpha -\\beta}}{B(\\alpha,\\beta)}\n\n    with :math:`x > 0`.\n\n    Parameters\n    ==========\n\n    alpha : Real number, `\\alpha > 0`, a shape\n    beta : Real number, `\\beta > 0`, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import BetaPrime, density\n    >>> from sympy import Symbol, pprint\n\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = BetaPrime("x", alpha, beta)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n     alpha - 1        -alpha - beta\n    z         *(z + 1)\n    -------------------------------\n             B(alpha, beta)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Beta_prime_distribution\n    .. [2] https://mathworld.wolfram.com/BetaPrimeDistribution.html\n\n    '
    return rv(name, BetaPrimeDistribution, (alpha, beta))

class BoundedParetoDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'left', 'right')

    @property
    def set(self):
        if False:
            return 10
        return Interval(self.left, self.right)

    @staticmethod
    def check(alpha, left, right):
        if False:
            i = 10
            return i + 15
        _value_check(alpha.is_positive, 'Shape must be positive.')
        _value_check(left.is_positive, 'Left value should be positive.')
        _value_check(right > left, 'Right should be greater than left.')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (alpha, left, right) = (self.alpha, self.left, self.right)
        num = alpha * left ** alpha * x ** (-alpha - 1)
        den = 1 - (left / right) ** alpha
        return num / den

def BoundedPareto(name, alpha, left, right):
    if False:
        print('Hello World!')
    "\n    Create a continuous random variable with a Bounded Pareto distribution.\n\n    The density of the Bounded Pareto distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\alpha L^{\\alpha}x^{-\\alpha-1}}{1-(\\frac{L}{H})^{\\alpha}}\n\n    Parameters\n    ==========\n\n    alpha : Real Number, `\\alpha > 0`\n        Shape parameter\n    left : Real Number, `left > 0`\n        Location parameter\n    right : Real Number, `right > left`\n        Location parameter\n\n    Examples\n    ========\n\n    >>> from sympy.stats import BoundedPareto, density, cdf, E\n    >>> from sympy import symbols\n    >>> L, H = symbols('L, H', positive=True)\n    >>> X = BoundedPareto('X', 2, L, H)\n    >>> x = symbols('x')\n    >>> density(X)(x)\n    2*L**2/(x**3*(1 - L**2/H**2))\n    >>> cdf(X)(x)\n    Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) + H**2/(H**2 - L**2), L <= x), (0, True))\n    >>> E(X).simplify()\n    2*H*L/(H + L)\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution\n\n    "
    return rv(name, BoundedParetoDistribution, (alpha, left, right))

class CauchyDistribution(SingleContinuousDistribution):
    _argnames = ('x0', 'gamma')

    @staticmethod
    def check(x0, gamma):
        if False:
            print('Hello World!')
        _value_check(gamma > 0, 'Scale parameter Gamma must be positive.')
        _value_check(x0.is_real, 'Location parameter must be real.')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        return 1 / (pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))

    def _cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (x0, gamma) = (self.x0, self.gamma)
        return 1 / pi * atan((x - x0) / gamma) + S.Half

    def _characteristic_function(self, t):
        if False:
            return 10
        return exp(self.x0 * I * t - self.gamma * Abs(t))

    def _moment_generating_function(self, t):
        if False:
            return 10
        raise NotImplementedError('The moment generating function for the Cauchy distribution does not exist.')

    def _quantile(self, p):
        if False:
            return 10
        return self.x0 + self.gamma * tan(pi * (p - S.Half))

def Cauchy(name, x0, gamma):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with a Cauchy distribution.\n\n    The density of the Cauchy distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{\\pi \\gamma [1 + {(\\frac{x-x_0}{\\gamma})}^2]}\n\n    Parameters\n    ==========\n\n    x0 : Real number, the location\n    gamma : Real number, `\\gamma > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Cauchy, density\n    >>> from sympy import Symbol\n\n    >>> x0 = Symbol("x0")\n    >>> gamma = Symbol("gamma", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Cauchy("x", x0, gamma)\n\n    >>> density(X)(z)\n    1/(pi*gamma*(1 + (-x0 + z)**2/gamma**2))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Cauchy_distribution\n    .. [2] https://mathworld.wolfram.com/CauchyDistribution.html\n\n    '
    return rv(name, CauchyDistribution, (x0, gamma))

class ChiDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        if False:
            for i in range(10):
                print('nop')
        _value_check(k > 0, 'Number of degrees of freedom (k) must be positive.')
        _value_check(k.is_integer, 'Number of degrees of freedom (k) must be an integer.')
    set = Interval(0, oo)

    def pdf(self, x):
        if False:
            return 10
        return 2 ** (1 - self.k / 2) * x ** (self.k - 1) * exp(-x ** 2 / 2) / gamma(self.k / 2)

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        k = self.k
        part_1 = hyper((k / 2,), (S.Half,), -t ** 2 / 2)
        part_2 = I * t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (Rational(3, 2),), -t ** 2 / 2)
        return part_1 + part_2 * part_3

    def _moment_generating_function(self, t):
        if False:
            return 10
        k = self.k
        part_1 = hyper((k / 2,), (S.Half,), t ** 2 / 2)
        part_2 = t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (S(3) / 2,), t ** 2 / 2)
        return part_1 + part_2 * part_3

def Chi(name, k):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a Chi distribution.\n\n    The density of the Chi distribution is given by\n\n    .. math::\n        f(x) := \\frac{2^{1-k/2}x^{k-1}e^{-x^2/2}}{\\Gamma(k/2)}\n\n    with :math:`x \\geq 0`.\n\n    Parameters\n    ==========\n\n    k : Positive integer, The number of degrees of freedom\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Chi, density, E\n    >>> from sympy import Symbol, simplify\n\n    >>> k = Symbol("k", integer=True)\n    >>> z = Symbol("z")\n\n    >>> X = Chi("x", k)\n\n    >>> density(X)(z)\n    2**(1 - k/2)*z**(k - 1)*exp(-z**2/2)/gamma(k/2)\n\n    >>> simplify(E(X))\n    sqrt(2)*gamma(k/2 + 1/2)/gamma(k/2)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Chi_distribution\n    .. [2] https://mathworld.wolfram.com/ChiDistribution.html\n\n    '
    return rv(name, ChiDistribution, (k,))

class ChiNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'l')

    @staticmethod
    def check(k, l):
        if False:
            i = 10
            return i + 15
        _value_check(k > 0, 'Number of degrees of freedom (k) must be positive.')
        _value_check(k.is_integer, 'Number of degrees of freedom (k) must be an integer.')
        _value_check(l > 0, 'Shift parameter Lambda must be positive.')
    set = Interval(0, oo)

    def pdf(self, x):
        if False:
            print('Hello World!')
        (k, l) = (self.k, self.l)
        return exp(-(x ** 2 + l ** 2) / 2) * x ** k * l / (l * x) ** (k / 2) * besseli(k / 2 - 1, l * x)

def ChiNoncentral(name, k, l):
    if False:
        i = 10
        return i + 15
    '\n    Create a continuous random variable with a non-central Chi distribution.\n\n    Explanation\n    ===========\n\n    The density of the non-central Chi distribution is given by\n\n    .. math::\n        f(x) := \\frac{e^{-(x^2+\\lambda^2)/2} x^k\\lambda}\n                {(\\lambda x)^{k/2}} I_{k/2-1}(\\lambda x)\n\n    with `x \\geq 0`. Here, `I_\\nu (x)` is the\n    :ref:`modified Bessel function of the first kind <besseli>`.\n\n    Parameters\n    ==========\n\n    k : A positive Integer, $k > 0$\n        The number of degrees of freedom.\n    lambda : Real number, `\\lambda > 0`\n        Shift parameter.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import ChiNoncentral, density\n    >>> from sympy import Symbol\n\n    >>> k = Symbol("k", integer=True)\n    >>> l = Symbol("l")\n    >>> z = Symbol("z")\n\n    >>> X = ChiNoncentral("x", k, l)\n\n    >>> density(X)(z)\n    l*z**k*exp(-l**2/2 - z**2/2)*besseli(k/2 - 1, l*z)/(l*z)**(k/2)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Noncentral_chi_distribution\n    '
    return rv(name, ChiNoncentralDistribution, (k, l))

class ChiSquaredDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        if False:
            return 10
        _value_check(k > 0, 'Number of degrees of freedom (k) must be positive.')
        _value_check(k.is_integer, 'Number of degrees of freedom (k) must be an integer.')
    set = Interval(0, oo)

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        k = self.k
        return 1 / (2 ** (k / 2) * gamma(k / 2)) * x ** (k / 2 - 1) * exp(-x / 2)

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        k = self.k
        return Piecewise((S.One / gamma(k / 2) * lowergamma(k / 2, x / 2), x >= 0), (0, True))

    def _characteristic_function(self, t):
        if False:
            return 10
        return (1 - 2 * I * t) ** (-self.k / 2)

    def _moment_generating_function(self, t):
        if False:
            return 10
        return (1 - 2 * t) ** (-self.k / 2)

def ChiSquared(name, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with a Chi-squared distribution.\n\n    Explanation\n    ===========\n\n    The density of the Chi-squared distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{2^{\\frac{k}{2}}\\Gamma\\left(\\frac{k}{2}\\right)}\n                x^{\\frac{k}{2}-1} e^{-\\frac{x}{2}}\n\n    with :math:`x \\geq 0`.\n\n    Parameters\n    ==========\n\n    k : Positive integer\n        The number of degrees of freedom.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import ChiSquared, density, E, variance, moment\n    >>> from sympy import Symbol\n\n    >>> k = Symbol("k", integer=True, positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = ChiSquared("x", k)\n\n    >>> density(X)(z)\n    z**(k/2 - 1)*exp(-z/2)/(2**(k/2)*gamma(k/2))\n\n    >>> E(X)\n    k\n\n    >>> variance(X)\n    2*k\n\n    >>> moment(X, 3)\n    k**3 + 6*k**2 + 8*k\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Chi_squared_distribution\n    .. [2] https://mathworld.wolfram.com/Chi-SquaredDistribution.html\n    '
    return rv(name, ChiSquaredDistribution, (k,))

class DagumDistribution(SingleContinuousDistribution):
    _argnames = ('p', 'a', 'b')
    set = Interval(0, oo)

    @staticmethod
    def check(p, a, b):
        if False:
            while True:
                i = 10
        _value_check(p > 0, 'Shape parameter p must be positive.')
        _value_check(a > 0, 'Shape parameter a must be positive.')
        _value_check(b > 0, 'Scale parameter b must be positive.')

    def pdf(self, x):
        if False:
            return 10
        (p, a, b) = (self.p, self.a, self.b)
        return a * p / x * ((x / b) ** (a * p) / ((x / b) ** a + 1) ** (p + 1))

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        (p, a, b) = (self.p, self.a, self.b)
        return Piecewise(((S.One + (S(x) / b) ** (-a)) ** (-p), x >= 0), (S.Zero, True))

def Dagum(name, p, a, b):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a Dagum distribution.\n\n    Explanation\n    ===========\n\n    The density of the Dagum distribution is given by\n\n    .. math::\n        f(x) := \\frac{a p}{x} \\left( \\frac{\\left(\\tfrac{x}{b}\\right)^{a p}}\n                {\\left(\\left(\\tfrac{x}{b}\\right)^a + 1 \\right)^{p+1}} \\right)\n\n    with :math:`x > 0`.\n\n    Parameters\n    ==========\n\n    p : Real number\n        `p > 0`, a shape.\n    a : Real number\n        `a > 0`, a shape.\n    b : Real number\n        `b > 0`, a scale.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Dagum, density, cdf\n    >>> from sympy import Symbol\n\n    >>> p = Symbol("p", positive=True)\n    >>> a = Symbol("a", positive=True)\n    >>> b = Symbol("b", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Dagum("x", p, a, b)\n\n    >>> density(X)(z)\n    a*p*(z/b)**(a*p)*((z/b)**a + 1)**(-p - 1)/z\n\n    >>> cdf(X)(z)\n    Piecewise(((1 + (z/b)**(-a))**(-p), z >= 0), (0, True))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Dagum_distribution\n\n    '
    return rv(name, DagumDistribution, (p, a, b))

class DavisDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'n', 'mu')
    set = Interval(0, oo)

    @staticmethod
    def check(b, n, mu):
        if False:
            i = 10
            return i + 15
        _value_check(b > 0, 'Scale parameter b must be positive.')
        _value_check(n > 1, 'Shape parameter n must be above 1.')
        _value_check(mu > 0, 'Location parameter mu must be positive.')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (b, n, mu) = (self.b, self.n, self.mu)
        dividend = b ** n * (x - mu) ** (-1 - n)
        divisor = (exp(b / (x - mu)) - 1) * (gamma(n) * zeta(n))
        return dividend / divisor

def Davis(name, b, n, mu):
    if False:
        for i in range(10):
            print('nop')
    ' Create a continuous random variable with Davis distribution.\n\n    Explanation\n    ===========\n\n    The density of Davis distribution is given by\n\n    .. math::\n        f(x; \\mu; b, n) := \\frac{b^{n}(x - \\mu)^{1-n}}{ \\left( e^{\\frac{b}{x-\\mu}} - 1 \\right) \\Gamma(n)\\zeta(n)}\n\n    with :math:`x \\in [0,\\infty]`.\n\n    Davis distribution is a generalization of the Planck\'s law of radiation from statistical physics. It is used for modeling income distribution.\n\n    Parameters\n    ==========\n    b : Real number\n        `p > 0`, a scale.\n    n : Real number\n        `n > 1`, a shape.\n    mu : Real number\n        `mu > 0`, a location.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n    >>> from sympy.stats import Davis, density\n    >>> from sympy import Symbol\n    >>> b = Symbol("b", positive=True)\n    >>> n = Symbol("n", positive=True)\n    >>> mu = Symbol("mu", positive=True)\n    >>> z = Symbol("z")\n    >>> X = Davis("x", b, n, mu)\n    >>> density(X)(z)\n    b**n*(-mu + z)**(-n - 1)/((exp(b/(-mu + z)) - 1)*gamma(n)*zeta(n))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Davis_distribution\n    .. [2] https://reference.wolfram.com/language/ref/DavisDistribution.html\n\n    '
    return rv(name, DavisDistribution, (b, n, mu))

def Erlang(name, k, l):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with an Erlang distribution.\n\n    Explanation\n    ===========\n\n    The density of the Erlang distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\lambda^k x^{k-1} e^{-\\lambda x}}{(k-1)!}\n\n    with :math:`x \\in [0,\\infty]`.\n\n    Parameters\n    ==========\n\n    k : Positive integer\n    l : Real number, `\\lambda > 0`, the rate\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Erlang, density, cdf, E, variance\n    >>> from sympy import Symbol, simplify, pprint\n\n    >>> k = Symbol("k", integer=True, positive=True)\n    >>> l = Symbol("l", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Erlang("x", k, l)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n     k  k - 1  -l*z\n    l *z     *e\n    ---------------\n        Gamma(k)\n\n    >>> C = cdf(X)(z)\n    >>> pprint(C, use_unicode=False)\n    /lowergamma(k, l*z)\n    |------------------  for z > 0\n    <     Gamma(k)\n    |\n    \\        0           otherwise\n\n\n    >>> E(X)\n    k/l\n\n    >>> simplify(variance(X))\n    k/l**2\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Erlang_distribution\n    .. [2] https://mathworld.wolfram.com/ErlangDistribution.html\n\n    '
    return rv(name, GammaDistribution, (k, S.One / l))

class ExGaussianDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std', 'rate')
    set = Interval(-oo, oo)

    @staticmethod
    def check(mean, std, rate):
        if False:
            return 10
        _value_check(std > 0, 'Standard deviation of ExGaussian must be positive.')
        _value_check(rate > 0, 'Rate of ExGaussian must be positive.')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        (mean, std, rate) = (self.mean, self.std, self.rate)
        term1 = rate / 2
        term2 = exp(rate * (2 * mean + rate * std ** 2 - 2 * x) / 2)
        term3 = erfc((mean + rate * std ** 2 - x) / (sqrt(2) * std))
        return term1 * term2 * term3

    def _cdf(self, x):
        if False:
            i = 10
            return i + 15
        from sympy.stats import cdf
        (mean, std, rate) = (self.mean, self.std, self.rate)
        u = rate * (x - mean)
        v = rate * std
        GaussianCDF1 = cdf(Normal('x', 0, v))(u)
        GaussianCDF2 = cdf(Normal('x', v ** 2, v))(u)
        return GaussianCDF1 - exp(-u + v ** 2 / 2 + log(GaussianCDF2))

    def _characteristic_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        (mean, std, rate) = (self.mean, self.std, self.rate)
        term1 = (1 - I * t / rate) ** (-1)
        term2 = exp(I * mean * t - std ** 2 * t ** 2 / 2)
        return term1 * term2

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        (mean, std, rate) = (self.mean, self.std, self.rate)
        term1 = (1 - t / rate) ** (-1)
        term2 = exp(mean * t + std ** 2 * t ** 2 / 2)
        return term1 * term2

def ExGaussian(name, mean, std, rate):
    if False:
        i = 10
        return i + 15
    '\n    Create a continuous random variable with an Exponentially modified\n    Gaussian (EMG) distribution.\n\n    Explanation\n    ===========\n\n    The density of the exponentially modified Gaussian distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\lambda}{2}e^{\\frac{\\lambda}{2}(2\\mu+\\lambda\\sigma^2-2x)}\n            \\text{erfc}(\\frac{\\mu + \\lambda\\sigma^2 - x}{\\sqrt{2}\\sigma})\n\n    with $x > 0$. Note that the expected value is `1/\\lambda`.\n\n    Parameters\n    ==========\n\n    name : A string giving a name for this distribution\n    mean : A Real number, the mean of Gaussian component\n    std : A positive Real number,\n        :math: `\\sigma^2 > 0` the variance of Gaussian component\n    rate : A positive Real number,\n        :math: `\\lambda > 0` the rate of Exponential component\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import ExGaussian, density, cdf, E\n    >>> from sympy.stats import variance, skewness\n    >>> from sympy import Symbol, pprint, simplify\n\n    >>> mean = Symbol("mu")\n    >>> std = Symbol("sigma", positive=True)\n    >>> rate = Symbol("lamda", positive=True)\n    >>> z = Symbol("z")\n    >>> X = ExGaussian("x", mean, std, rate)\n\n    >>> pprint(density(X)(z), use_unicode=False)\n                 /           2             \\\n           lamda*\\lamda*sigma  + 2*mu - 2*z/\n           ---------------------------------     /  ___ /           2         \\\\\n                           2                     |\\/ 2 *\\lamda*sigma  + mu - z/|\n    lamda*e                                 *erfc|-----------------------------|\n                                                 \\           2*sigma           /\n    ----------------------------------------------------------------------------\n                                         2\n\n    >>> cdf(X)(z)\n    -(erf(sqrt(2)*(-lamda**2*sigma**2 + lamda*(-mu + z))/(2*lamda*sigma))/2 + 1/2)*exp(lamda**2*sigma**2/2 - lamda*(-mu + z)) + erf(sqrt(2)*(-mu + z)/(2*sigma))/2 + 1/2\n\n    >>> E(X)\n    (lamda*mu + 1)/lamda\n\n    >>> simplify(variance(X))\n    sigma**2 + lamda**(-2)\n\n    >>> simplify(skewness(X))\n    2/(lamda**2*sigma**2 + 1)**(3/2)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution\n    '
    return rv(name, ExGaussianDistribution, (mean, std, rate))

class ExponentialDistribution(SingleContinuousDistribution):
    _argnames = ('rate',)
    set = Interval(0, oo)

    @staticmethod
    def check(rate):
        if False:
            return 10
        _value_check(rate > 0, 'Rate must be positive.')

    def pdf(self, x):
        if False:
            print('Hello World!')
        return self.rate * exp(-self.rate * x)

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        return Piecewise((S.One - exp(-self.rate * x), x >= 0), (0, True))

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        rate = self.rate
        return rate / (rate - I * t)

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        rate = self.rate
        return rate / (rate - t)

    def _quantile(self, p):
        if False:
            while True:
                i = 10
        return -log(1 - p) / self.rate

def Exponential(name, rate):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with an Exponential distribution.\n\n    Explanation\n    ===========\n\n    The density of the exponential distribution is given by\n\n    .. math::\n        f(x) := \\lambda \\exp(-\\lambda x)\n\n    with $x > 0$. Note that the expected value is `1/\\lambda`.\n\n    Parameters\n    ==========\n\n    rate : A positive Real number, `\\lambda > 0`, the rate (or inverse scale/inverse mean)\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Exponential, density, cdf, E\n    >>> from sympy.stats import variance, std, skewness, quantile\n    >>> from sympy import Symbol\n\n    >>> l = Symbol("lambda", positive=True)\n    >>> z = Symbol("z")\n    >>> p = Symbol("p")\n    >>> X = Exponential("x", l)\n\n    >>> density(X)(z)\n    lambda*exp(-lambda*z)\n\n    >>> cdf(X)(z)\n    Piecewise((1 - exp(-lambda*z), z >= 0), (0, True))\n\n    >>> quantile(X)(p)\n    -log(1 - p)/lambda\n\n    >>> E(X)\n    1/lambda\n\n    >>> variance(X)\n    lambda**(-2)\n\n    >>> skewness(X)\n    2\n\n    >>> X = Exponential(\'x\', 10)\n\n    >>> density(X)(z)\n    10*exp(-10*z)\n\n    >>> E(X)\n    1/10\n\n    >>> std(X)\n    1/10\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Exponential_distribution\n    .. [2] https://mathworld.wolfram.com/ExponentialDistribution.html\n\n    '
    return rv(name, ExponentialDistribution, (rate,))

class ExponentialPowerDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'alpha', 'beta')
    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, alpha, beta):
        if False:
            return 10
        _value_check(alpha > 0, 'Scale parameter alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter beta must be positive.')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        (mu, alpha, beta) = (self.mu, self.alpha, self.beta)
        num = beta * exp(-(Abs(x - mu) / alpha) ** beta)
        den = 2 * alpha * gamma(1 / beta)
        return num / den

    def _cdf(self, x):
        if False:
            return 10
        (mu, alpha, beta) = (self.mu, self.alpha, self.beta)
        num = lowergamma(1 / beta, (Abs(x - mu) / alpha) ** beta)
        den = 2 * gamma(1 / beta)
        return sign(x - mu) * num / den + S.Half

def ExponentialPower(name, mu, alpha, beta):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Continuous Random Variable with Exponential Power distribution.\n    This distribution is known also as Generalized Normal\n    distribution version 1.\n\n    Explanation\n    ===========\n\n    The density of the Exponential Power distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\beta}{2\\alpha\\Gamma(\\frac{1}{\\beta})}\n            e^{{-(\\frac{|x - \\mu|}{\\alpha})^{\\beta}}}\n\n    with :math:`x \\in [ - \\infty, \\infty ]`.\n\n    Parameters\n    ==========\n\n    mu : Real number\n        A location.\n    alpha : Real number,`\\alpha > 0`\n        A  scale.\n    beta : Real number, `\\beta > 0`\n        A shape.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import ExponentialPower, density, cdf\n    >>> from sympy import Symbol, pprint\n    >>> z = Symbol("z")\n    >>> mu = Symbol("mu")\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> X = ExponentialPower("x", mu, alpha, beta)\n    >>> pprint(density(X)(z), use_unicode=False)\n                     beta\n           /|mu - z|\\\n          -|--------|\n           \\ alpha  /\n    beta*e\n    ---------------------\n                  / 1  \\\n     2*alpha*Gamma|----|\n                  \\beta/\n    >>> cdf(X)(z)\n    1/2 + lowergamma(1/beta, (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/(2*gamma(1/beta))\n\n    References\n    ==========\n\n    .. [1] https://reference.wolfram.com/language/ref/ExponentialPowerDistribution.html\n    .. [2] https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1\n\n    '
    return rv(name, ExponentialPowerDistribution, (mu, alpha, beta))

class FDistributionDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')
    set = Interval(0, oo)

    @staticmethod
    def check(d1, d2):
        if False:
            i = 10
            return i + 15
        _value_check((d1 > 0, d1.is_integer), 'Degrees of freedom d1 must be positive integer.')
        _value_check((d2 > 0, d2.is_integer), 'Degrees of freedom d2 must be positive integer.')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (d1, d2) = (self.d1, self.d2)
        return sqrt((d1 * x) ** d1 * d2 ** d2 / (d1 * x + d2) ** (d1 + d2)) / (x * beta_fn(d1 / 2, d2 / 2))

    def _moment_generating_function(self, t):
        if False:
            while True:
                i = 10
        raise NotImplementedError('The moment generating function for the F-distribution does not exist.')

def FDistribution(name, d1, d2):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a F distribution.\n\n    Explanation\n    ===========\n\n    The density of the F distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\sqrt{\\frac{(d_1 x)^{d_1} d_2^{d_2}}\n                {(d_1 x + d_2)^{d_1 + d_2}}}}\n                {x \\mathrm{B} \\left(\\frac{d_1}{2}, \\frac{d_2}{2}\\right)}\n\n    with :math:`x > 0`.\n\n    Parameters\n    ==========\n\n    d1 : `d_1 > 0`, where `d_1` is the degrees of freedom (`n_1 - 1`)\n    d2 : `d_2 > 0`, where `d_2` is the degrees of freedom (`n_2 - 1`)\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import FDistribution, density\n    >>> from sympy import Symbol, pprint\n\n    >>> d1 = Symbol("d1", positive=True)\n    >>> d2 = Symbol("d2", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = FDistribution("x", d1, d2)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n      d2\n      --    ______________________________\n      2    /       d1            -d1 - d2\n    d2  *\\/  (d1*z)  *(d1*z + d2)\n    --------------------------------------\n                    /d1  d2\\\n                 z*B|--, --|\n                    \\2   2 /\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/F-distribution\n    .. [2] https://mathworld.wolfram.com/F-Distribution.html\n\n    '
    return rv(name, FDistributionDistribution, (d1, d2))

class FisherZDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')
    set = Interval(-oo, oo)

    @staticmethod
    def check(d1, d2):
        if False:
            for i in range(10):
                print('nop')
        _value_check(d1 > 0, 'Degree of freedom d1 must be positive.')
        _value_check(d2 > 0, 'Degree of freedom d2 must be positive.')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        (d1, d2) = (self.d1, self.d2)
        return 2 * d1 ** (d1 / 2) * d2 ** (d2 / 2) / beta_fn(d1 / 2, d2 / 2) * exp(d1 * x) / (d1 * exp(2 * x) + d2) ** ((d1 + d2) / 2)

def FisherZ(name, d1, d2):
    if False:
        while True:
            i = 10
    '\n    Create a Continuous Random Variable with an Fisher\'s Z distribution.\n\n    Explanation\n    ===========\n\n    The density of the Fisher\'s Z distribution is given by\n\n    .. math::\n        f(x) := \\frac{2d_1^{d_1/2} d_2^{d_2/2}} {\\mathrm{B}(d_1/2, d_2/2)}\n                \\frac{e^{d_1z}}{\\left(d_1e^{2z}+d_2\\right)^{\\left(d_1+d_2\\right)/2}}\n\n\n    .. TODO - What is the difference between these degrees of freedom?\n\n    Parameters\n    ==========\n\n    d1 : `d_1 > 0`\n        Degree of freedom.\n    d2 : `d_2 > 0`\n        Degree of freedom.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import FisherZ, density\n    >>> from sympy import Symbol, pprint\n\n    >>> d1 = Symbol("d1", positive=True)\n    >>> d2 = Symbol("d2", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = FisherZ("x", d1, d2)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                                d1   d2\n        d1   d2               - -- - --\n        --   --                 2    2\n        2    2  /    2*z     \\           d1*z\n    2*d1  *d2  *\\d1*e    + d2/         *e\n    -----------------------------------------\n                     /d1  d2\\\n                    B|--, --|\n                     \\2   2 /\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Fisher%27s_z-distribution\n    .. [2] https://mathworld.wolfram.com/Fishersz-Distribution.html\n\n    '
    return rv(name, FisherZDistribution, (d1, d2))

class FrechetDistribution(SingleContinuousDistribution):
    _argnames = ('a', 's', 'm')
    set = Interval(0, oo)

    @staticmethod
    def check(a, s, m):
        if False:
            i = 10
            return i + 15
        _value_check(a > 0, 'Shape parameter alpha must be positive.')
        _value_check(s > 0, 'Scale parameter s must be positive.')

    def __new__(cls, a, s=1, m=0):
        if False:
            return 10
        (a, s, m) = list(map(sympify, (a, s, m)))
        return Basic.__new__(cls, a, s, m)

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (a, s, m) = (self.a, self.s, self.m)
        return a / s * ((x - m) / s) ** (-1 - a) * exp(-((x - m) / s) ** (-a))

    def _cdf(self, x):
        if False:
            return 10
        (a, s, m) = (self.a, self.s, self.m)
        return Piecewise((exp(-((x - m) / s) ** (-a)), x >= m), (S.Zero, True))

def Frechet(name, a, s=1, m=0):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a Frechet distribution.\n\n    Explanation\n    ===========\n\n    The density of the Frechet distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\alpha}{s} \\left(\\frac{x-m}{s}\\right)^{-1-\\alpha}\n                 e^{-(\\frac{x-m}{s})^{-\\alpha}}\n\n    with :math:`x \\geq m`.\n\n    Parameters\n    ==========\n\n    a : Real number, :math:`a \\in \\left(0, \\infty\\right)` the shape\n    s : Real number, :math:`s \\in \\left(0, \\infty\\right)` the scale\n    m : Real number, :math:`m \\in \\left(-\\infty, \\infty\\right)` the minimum\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Frechet, density, cdf\n    >>> from sympy import Symbol\n\n    >>> a = Symbol("a", positive=True)\n    >>> s = Symbol("s", positive=True)\n    >>> m = Symbol("m", real=True)\n    >>> z = Symbol("z")\n\n    >>> X = Frechet("x", a, s, m)\n\n    >>> density(X)(z)\n    a*((-m + z)/s)**(-a - 1)*exp(-1/((-m + z)/s)**a)/s\n\n    >>> cdf(X)(z)\n    Piecewise((exp(-1/((-m + z)/s)**a), m <= z), (0, True))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution\n\n    '
    return rv(name, FrechetDistribution, (a, s, m))

class GammaDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'theta')
    set = Interval(0, oo)

    @staticmethod
    def check(k, theta):
        if False:
            i = 10
            return i + 15
        _value_check(k > 0, 'k must be positive')
        _value_check(theta > 0, 'Theta must be positive')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (k, theta) = (self.k, self.theta)
        return x ** (k - 1) * exp(-x / theta) / (gamma(k) * theta ** k)

    def _cdf(self, x):
        if False:
            print('Hello World!')
        (k, theta) = (self.k, self.theta)
        return Piecewise((lowergamma(k, S(x) / theta) / gamma(k), x > 0), (S.Zero, True))

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        return (1 - self.theta * I * t) ** (-self.k)

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        return (1 - self.theta * t) ** (-self.k)

def Gamma(name, k, theta):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a Gamma distribution.\n\n    Explanation\n    ===========\n\n    The density of the Gamma distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{\\Gamma(k) \\theta^k} x^{k - 1} e^{-\\frac{x}{\\theta}}\n\n    with :math:`x \\in [0,1]`.\n\n    Parameters\n    ==========\n\n    k : Real number, `k > 0`, a shape\n    theta : Real number, `\\theta > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Gamma, density, cdf, E, variance\n    >>> from sympy import Symbol, pprint, simplify\n\n    >>> k = Symbol("k", positive=True)\n    >>> theta = Symbol("theta", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Gamma("x", k, theta)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                      -z\n                    -----\n         -k  k - 1  theta\n    theta  *z     *e\n    ---------------------\n           Gamma(k)\n\n    >>> C = cdf(X, meijerg=True)(z)\n    >>> pprint(C, use_unicode=False)\n    /            /     z  \\\n    |k*lowergamma|k, -----|\n    |            \\   theta/\n    <----------------------  for z >= 0\n    |     Gamma(k + 1)\n    |\n    \\          0             otherwise\n\n    >>> E(X)\n    k*theta\n\n    >>> V = simplify(variance(X))\n    >>> pprint(V, use_unicode=False)\n           2\n    k*theta\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Gamma_distribution\n    .. [2] https://mathworld.wolfram.com/GammaDistribution.html\n\n    '
    return rv(name, GammaDistribution, (k, theta))

class GammaInverseDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')
    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        if False:
            return 10
        _value_check(a > 0, 'alpha must be positive')
        _value_check(b > 0, 'beta must be positive')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (a, b) = (self.a, self.b)
        return b ** a / gamma(a) * x ** (-a - 1) * exp(-b / x)

    def _cdf(self, x):
        if False:
            i = 10
            return i + 15
        (a, b) = (self.a, self.b)
        return Piecewise((uppergamma(a, b / x) / gamma(a), x > 0), (S.Zero, True))

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        (a, b) = (self.a, self.b)
        return 2 * (-I * b * t) ** (a / 2) * besselk(a, sqrt(-4 * I * b * t)) / gamma(a)

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('The moment generating function for the gamma inverse distribution does not exist.')

def GammaInverse(name, a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with an inverse Gamma distribution.\n\n    Explanation\n    ===========\n\n    The density of the inverse Gamma distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} x^{-\\alpha - 1}\n                \\exp\\left(\\frac{-\\beta}{x}\\right)\n\n    with :math:`x > 0`.\n\n    Parameters\n    ==========\n\n    a : Real number, `a > 0`, a shape\n    b : Real number, `b > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GammaInverse, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> a = Symbol("a", positive=True)\n    >>> b = Symbol("b", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = GammaInverse("x", a, b)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                -b\n                ---\n     a  -a - 1   z\n    b *z      *e\n    ---------------\n       Gamma(a)\n\n    >>> cdf(X)(z)\n    Piecewise((uppergamma(a, b/z)/gamma(a), z > 0), (0, True))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution\n\n    '
    return rv(name, GammaInverseDistribution, (a, b))

class GumbelDistribution(SingleContinuousDistribution):
    _argnames = ('beta', 'mu', 'minimum')
    set = Interval(-oo, oo)

    @staticmethod
    def check(beta, mu, minimum):
        if False:
            print('Hello World!')
        _value_check(beta > 0, 'Scale parameter beta must be positive.')

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (beta, mu) = (self.beta, self.mu)
        z = (x - mu) / beta
        f_max = 1 / beta * exp(-z - exp(-z))
        f_min = 1 / beta * exp(z - exp(z))
        return Piecewise((f_min, self.minimum), (f_max, not self.minimum))

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        (beta, mu) = (self.beta, self.mu)
        z = (x - mu) / beta
        F_max = exp(-exp(-z))
        F_min = 1 - exp(-exp(z))
        return Piecewise((F_min, self.minimum), (F_max, not self.minimum))

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        cf_max = gamma(1 - I * self.beta * t) * exp(I * self.mu * t)
        cf_min = gamma(1 + I * self.beta * t) * exp(I * self.mu * t)
        return Piecewise((cf_min, self.minimum), (cf_max, not self.minimum))

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        mgf_max = gamma(1 - self.beta * t) * exp(self.mu * t)
        mgf_min = gamma(1 + self.beta * t) * exp(self.mu * t)
        return Piecewise((mgf_min, self.minimum), (mgf_max, not self.minimum))

def Gumbel(name, beta, mu, minimum=False):
    if False:
        return 10
    '\n    Create a Continuous Random Variable with Gumbel distribution.\n\n    Explanation\n    ===========\n\n    The density of the Gumbel distribution is given by\n\n    For Maximum\n\n    .. math::\n        f(x) := \\dfrac{1}{\\beta} \\exp \\left( -\\dfrac{x-\\mu}{\\beta}\n                - \\exp \\left( -\\dfrac{x - \\mu}{\\beta} \\right) \\right)\n\n    with :math:`x \\in [ - \\infty, \\infty ]`.\n\n    For Minimum\n\n    .. math::\n        f(x) := \\frac{e^{- e^{\\frac{- \\mu + x}{\\beta}} + \\frac{- \\mu + x}{\\beta}}}{\\beta}\n\n    with :math:`x \\in [ - \\infty, \\infty ]`.\n\n    Parameters\n    ==========\n\n    mu : Real number, `\\mu`, a location\n    beta : Real number, `\\beta > 0`, a scale\n    minimum : Boolean, by default ``False``, set to ``True`` for enabling minimum distribution\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Gumbel, density, cdf\n    >>> from sympy import Symbol\n    >>> x = Symbol("x")\n    >>> mu = Symbol("mu")\n    >>> beta = Symbol("beta", positive=True)\n    >>> X = Gumbel("x", beta, mu)\n    >>> density(X)(x)\n    exp(-exp(-(-mu + x)/beta) - (-mu + x)/beta)/beta\n    >>> cdf(X)(x)\n    exp(-exp(-(-mu + x)/beta))\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/GumbelDistribution.html\n    .. [2] https://en.wikipedia.org/wiki/Gumbel_distribution\n    .. [3] https://web.archive.org/web/20200628222206/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_max.html\n    .. [4] https://web.archive.org/web/20200628222212/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_min.html\n\n    '
    return rv(name, GumbelDistribution, (beta, mu, minimum))

class GompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')
    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        if False:
            i = 10
            return i + 15
        _value_check(b > 0, 'b must be positive')
        _value_check(eta > 0, 'eta must be positive')

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (eta, b) = (self.eta, self.b)
        return b * eta * exp(b * x) * exp(eta) * exp(-eta * exp(b * x))

    def _cdf(self, x):
        if False:
            return 10
        (eta, b) = (self.eta, self.b)
        return 1 - exp(eta) * exp(-eta * exp(b * x))

    def _moment_generating_function(self, t):
        if False:
            i = 10
            return i + 15
        (eta, b) = (self.eta, self.b)
        return eta * exp(eta) * expint(t / b, eta)

def Gompertz(name, b, eta):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Continuous Random Variable with Gompertz distribution.\n\n    Explanation\n    ===========\n\n    The density of the Gompertz distribution is given by\n\n    .. math::\n        f(x) := b \\eta e^{b x} e^{\\eta} \\exp \\left(-\\eta e^{bx} \\right)\n\n    with :math:`x \\in [0, \\infty)`.\n\n    Parameters\n    ==========\n\n    b : Real number, `b > 0`, a scale\n    eta : Real number, `\\eta > 0`, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Gompertz, density\n    >>> from sympy import Symbol\n\n    >>> b = Symbol("b", positive=True)\n    >>> eta = Symbol("eta", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Gompertz("x", b, eta)\n\n    >>> density(X)(z)\n    b*eta*exp(eta)*exp(b*z)*exp(-eta*exp(b*z))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Gompertz_distribution\n\n    '
    return rv(name, GompertzDistribution, (b, eta))

class KumaraswamyDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')
    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        if False:
            for i in range(10):
                print('nop')
        _value_check(a > 0, 'a must be positive')
        _value_check(b > 0, 'b must be positive')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (a, b) = (self.a, self.b)
        return a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)

    def _cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (self.a, self.b)
        return Piecewise((S.Zero, x < S.Zero), (1 - (1 - x ** a) ** b, x <= S.One), (S.One, True))

def Kumaraswamy(name, a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Continuous Random Variable with a Kumaraswamy distribution.\n\n    Explanation\n    ===========\n\n    The density of the Kumaraswamy distribution is given by\n\n    .. math::\n        f(x) := a b x^{a-1} (1-x^a)^{b-1}\n\n    with :math:`x \\in [0,1]`.\n\n    Parameters\n    ==========\n\n    a : Real number, `a > 0`, a shape\n    b : Real number, `b > 0`, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Kumaraswamy, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> a = Symbol("a", positive=True)\n    >>> b = Symbol("b", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Kumaraswamy("x", a, b)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                       b - 1\n         a - 1 /     a\\\n    a*b*z     *\\1 - z /\n\n    >>> cdf(X)(z)\n    Piecewise((0, z < 0), (1 - (1 - z**a)**b, z <= 1), (1, True))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Kumaraswamy_distribution\n\n    '
    return rv(name, KumaraswamyDistribution, (a, b))

class LaplaceDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'b')
    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, b):
        if False:
            print('Hello World!')
        _value_check(b > 0, 'Scale parameter b must be positive.')
        _value_check(mu.is_real, 'Location parameter mu should be real')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        (mu, b) = (self.mu, self.b)
        return 1 / (2 * b) * exp(-Abs(x - mu) / b)

    def _cdf(self, x):
        if False:
            return 10
        (mu, b) = (self.mu, self.b)
        return Piecewise((S.Half * exp((x - mu) / b), x < mu), (S.One - S.Half * exp(-(x - mu) / b), x >= mu))

    def _characteristic_function(self, t):
        if False:
            return 10
        return exp(self.mu * I * t) / (1 + self.b ** 2 * t ** 2)

    def _moment_generating_function(self, t):
        if False:
            return 10
        return exp(self.mu * t) / (1 - self.b ** 2 * t ** 2)

def Laplace(name, mu, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with a Laplace distribution.\n\n    Explanation\n    ===========\n\n    The density of the Laplace distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{2 b} \\exp \\left(-\\frac{|x-\\mu|}b \\right)\n\n    Parameters\n    ==========\n\n    mu : Real number or a list/matrix, the location (mean) or the\n        location vector\n    b : Real number or a positive definite matrix, representing a scale\n        or the covariance matrix.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Laplace, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> mu = Symbol("mu")\n    >>> b = Symbol("b", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Laplace("x", mu, b)\n\n    >>> density(X)(z)\n    exp(-Abs(mu - z)/b)/(2*b)\n\n    >>> cdf(X)(z)\n    Piecewise((exp((-mu + z)/b)/2, mu > z), (1 - exp((mu - z)/b)/2, True))\n\n    >>> L = Laplace(\'L\', [1, 2], [[1, 0], [0, 1]])\n    >>> pprint(density(L)(1, 2), use_unicode=False)\n     5        /     ____\\\n    e *besselk\\0, \\/ 35 /\n    ---------------------\n              pi\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Laplace_distribution\n    .. [2] https://mathworld.wolfram.com/LaplaceDistribution.html\n\n    '
    if isinstance(mu, (list, MatrixBase)) and isinstance(b, (list, MatrixBase)):
        from sympy.stats.joint_rv_types import MultivariateLaplace
        return MultivariateLaplace(name, mu, b)
    return rv(name, LaplaceDistribution, (mu, b))

class LevyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'c')

    @property
    def set(self):
        if False:
            while True:
                i = 10
        return Interval(self.mu, oo)

    @staticmethod
    def check(mu, c):
        if False:
            while True:
                i = 10
        _value_check(c > 0, 'c (scale parameter) must be positive')
        _value_check(mu.is_real, 'mu (location parameter) must be real')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (mu, c) = (self.mu, self.c)
        return sqrt(c / (2 * pi)) * exp(-c / (2 * (x - mu))) / (x - mu) ** (S.One + S.Half)

    def _cdf(self, x):
        if False:
            print('Hello World!')
        (mu, c) = (self.mu, self.c)
        return erfc(sqrt(c / (2 * (x - mu))))

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        (mu, c) = (self.mu, self.c)
        return exp(I * mu * t - sqrt(-2 * I * c * t))

    def _moment_generating_function(self, t):
        if False:
            return 10
        raise NotImplementedError('The moment generating function of Levy distribution does not exist.')

def Levy(name, mu, c):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a Levy distribution.\n\n    The density of the Levy distribution is given by\n\n    .. math::\n        f(x) := \\sqrt(\\frac{c}{2 \\pi}) \\frac{\\exp -\\frac{c}{2 (x - \\mu)}}{(x - \\mu)^{3/2}}\n\n    Parameters\n    ==========\n\n    mu : Real number\n        The location parameter.\n    c : Real number, `c > 0`\n        A scale parameter.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Levy, density, cdf\n    >>> from sympy import Symbol\n\n    >>> mu = Symbol("mu", real=True)\n    >>> c = Symbol("c", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Levy("x", mu, c)\n\n    >>> density(X)(z)\n    sqrt(2)*sqrt(c)*exp(-c/(-2*mu + 2*z))/(2*sqrt(pi)*(-mu + z)**(3/2))\n\n    >>> cdf(X)(z)\n    erfc(sqrt(c)*sqrt(1/(-2*mu + 2*z)))\n\n    References\n    ==========\n    .. [1] https://en.wikipedia.org/wiki/L%C3%A9vy_distribution\n    .. [2] https://mathworld.wolfram.com/LevyDistribution.html\n    '
    return rv(name, LevyDistribution, (mu, c))

class LogCauchyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')
    set = Interval.open(0, oo)

    @staticmethod
    def check(mu, sigma):
        if False:
            i = 10
            return i + 15
        _value_check((sigma > 0) != False, 'Scale parameter Gamma must be positive.')
        _value_check(mu.is_real != False, 'Location parameter must be real.')

    def pdf(self, x):
        if False:
            return 10
        (mu, sigma) = (self.mu, self.sigma)
        return 1 / (x * pi) * (sigma / ((log(x) - mu) ** 2 + sigma ** 2))

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        (mu, sigma) = (self.mu, self.sigma)
        return 1 / pi * atan((log(x) - mu) / sigma) + S.Half

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        raise NotImplementedError('The characteristic function for the Log-Cauchy distribution does not exist.')

    def _moment_generating_function(self, t):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('The moment generating function for the Log-Cauchy distribution does not exist.')

def LogCauchy(name, mu, sigma):
    if False:
        i = 10
        return i + 15
    '\n    Create a continuous random variable with a Log-Cauchy distribution.\n    The density of the Log-Cauchy distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{\\pi x} \\frac{\\sigma}{(log(x)-\\mu^2) + \\sigma^2}\n\n    Parameters\n    ==========\n\n    mu : Real number, the location\n\n    sigma : Real number, `\\sigma > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import LogCauchy, density, cdf\n    >>> from sympy import Symbol, S\n\n    >>> mu = 2\n    >>> sigma = S.One / 5\n    >>> z = Symbol("z")\n\n    >>> X = LogCauchy("x", mu, sigma)\n\n    >>> density(X)(z)\n    1/(5*pi*z*((log(z) - 2)**2 + 1/25))\n\n    >>> cdf(X)(z)\n    atan(5*log(z) - 10)/pi + 1/2\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Log-Cauchy_distribution\n    '
    return rv(name, LogCauchyDistribution, (mu, sigma))

class LogisticDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')
    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, s):
        if False:
            print('Hello World!')
        _value_check(s > 0, 'Scale parameter s must be positive.')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        (mu, s) = (self.mu, self.s)
        return exp(-(x - mu) / s) / (s * (1 + exp(-(x - mu) / s)) ** 2)

    def _cdf(self, x):
        if False:
            return 10
        (mu, s) = (self.mu, self.s)
        return S.One / (1 + exp(-(x - mu) / s))

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        return Piecewise((exp(I * t * self.mu) * pi * self.s * t / sinh(pi * self.s * t), Ne(t, 0)), (S.One, True))

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        return exp(self.mu * t) * beta_fn(1 - self.s * t, 1 + self.s * t)

    def _quantile(self, p):
        if False:
            print('Hello World!')
        return self.mu - self.s * log(-S.One + S.One / p)

def Logistic(name, mu, s):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a logistic distribution.\n\n    Explanation\n    ===========\n\n    The density of the logistic distribution is given by\n\n    .. math::\n        f(x) := \\frac{e^{-(x-\\mu)/s}} {s\\left(1+e^{-(x-\\mu)/s}\\right)^2}\n\n    Parameters\n    ==========\n\n    mu : Real number, the location (mean)\n    s : Real number, `s > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Logistic, density, cdf\n    >>> from sympy import Symbol\n\n    >>> mu = Symbol("mu", real=True)\n    >>> s = Symbol("s", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Logistic("x", mu, s)\n\n    >>> density(X)(z)\n    exp((mu - z)/s)/(s*(exp((mu - z)/s) + 1)**2)\n\n    >>> cdf(X)(z)\n    1/(exp((mu - z)/s) + 1)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Logistic_distribution\n    .. [2] https://mathworld.wolfram.com/LogisticDistribution.html\n\n    '
    return rv(name, LogisticDistribution, (mu, s))

class LogLogisticDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        if False:
            return 10
        _value_check(alpha > 0, 'Scale parameter Alpha must be positive.')
        _value_check(beta > 0, 'Shape parameter Beta must be positive.')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (a, b) = (self.alpha, self.beta)
        return b / a * (x / a) ** (b - 1) / (1 + (x / a) ** b) ** 2

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        (a, b) = (self.alpha, self.beta)
        return 1 / (1 + (x / a) ** (-b))

    def _quantile(self, p):
        if False:
            print('Hello World!')
        (a, b) = (self.alpha, self.beta)
        return a * (p / (1 - p)) ** (1 / b)

    def expectation(self, expr, var, **kwargs):
        if False:
            return 10
        (a, b) = self.args
        return Piecewise((S.NaN, b <= 1), (pi * a / (b * sin(pi / b)), True))

def LogLogistic(name, alpha, beta):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with a log-logistic distribution.\n    The distribution is unimodal when ``beta > 1``.\n\n    Explanation\n    ===========\n\n    The density of the log-logistic distribution is given by\n\n    .. math::\n        f(x) := \\frac{(\\frac{\\beta}{\\alpha})(\\frac{x}{\\alpha})^{\\beta - 1}}\n                {(1 + (\\frac{x}{\\alpha})^{\\beta})^2}\n\n    Parameters\n    ==========\n\n    alpha : Real number, `\\alpha > 0`, scale parameter and median of distribution\n    beta : Real number, `\\beta > 0`, a shape parameter\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import LogLogistic, density, cdf, quantile\n    >>> from sympy import Symbol, pprint\n\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> p = Symbol("p")\n    >>> z = Symbol("z", positive=True)\n\n    >>> X = LogLogistic("x", alpha, beta)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                  beta - 1\n           /  z  \\\n      beta*|-----|\n           \\alpha/\n    ------------------------\n                           2\n          /       beta    \\\n          |/  z  \\        |\n    alpha*||-----|     + 1|\n          \\\\alpha/        /\n\n    >>> cdf(X)(z)\n    1/(1 + (z/alpha)**(-beta))\n\n    >>> quantile(X)(p)\n    alpha*(p/(1 - p))**(1/beta)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Log-logistic_distribution\n\n    '
    return rv(name, LogLogisticDistribution, (alpha, beta))

class LogitNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')
    set = Interval.open(0, 1)

    @staticmethod
    def check(mu, s):
        if False:
            print('Hello World!')
        _value_check((s ** 2).is_real is not False and s ** 2 > 0, 'Squared scale parameter s must be positive.')
        _value_check(mu.is_real is not False, 'Location parameter must be real')

    def _logit(self, x):
        if False:
            print('Hello World!')
        return log(x / (1 - x))

    def pdf(self, x):
        if False:
            print('Hello World!')
        (mu, s) = (self.mu, self.s)
        return exp(-(self._logit(x) - mu) ** 2 / (2 * s ** 2)) * (S.One / sqrt(2 * pi * s ** 2)) * (1 / (x * (1 - x)))

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        (mu, s) = (self.mu, self.s)
        return S.One / 2 * (1 + erf((self._logit(x) - mu) / sqrt(2 * s ** 2)))

def LogitNormal(name, mu, s):
    if False:
        i = 10
        return i + 15
    '\n    Create a continuous random variable with a Logit-Normal distribution.\n\n    The density of the logistic distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{s \\sqrt{2 \\pi}} \\frac{1}{x(1 - x)} e^{- \\frac{(logit(x)  - \\mu)^2}{s^2}}\n        where logit(x) = \\log(\\frac{x}{1 - x})\n    Parameters\n    ==========\n\n    mu : Real number, the location (mean)\n    s : Real number, `s > 0`, a scale\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import LogitNormal, density, cdf\n    >>> from sympy import Symbol,pprint\n\n    >>> mu = Symbol("mu", real=True)\n    >>> s = Symbol("s", positive=True)\n    >>> z = Symbol("z")\n    >>> X = LogitNormal("x",mu,s)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                              2\n            /         /  z  \\\\\n           -|-mu + log|-----||\n            \\         \\1 - z//\n           ---------------------\n                       2\n      ___           2*s\n    \\/ 2 *e\n    ----------------------------\n            ____\n        2*\\/ pi *s*z*(1 - z)\n\n    >>> density(X)(z)\n    sqrt(2)*exp(-(-mu + log(z/(1 - z)))**2/(2*s**2))/(2*sqrt(pi)*s*z*(1 - z))\n\n    >>> cdf(X)(z)\n    erf(sqrt(2)*(-mu + log(z/(1 - z)))/(2*s))/2 + 1/2\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Logit-normal_distribution\n\n    '
    return rv(name, LogitNormalDistribution, (mu, s))

class LogNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')
    set = Interval(0, oo)

    @staticmethod
    def check(mean, std):
        if False:
            print('Hello World!')
        _value_check(std > 0, 'Parameter std must be positive.')

    def pdf(self, x):
        if False:
            return 10
        (mean, std) = (self.mean, self.std)
        return exp(-(log(x) - mean) ** 2 / (2 * std ** 2)) / (x * sqrt(2 * pi) * std)

    def _cdf(self, x):
        if False:
            return 10
        (mean, std) = (self.mean, self.std)
        return Piecewise((S.Half + S.Half * erf((log(x) - mean) / sqrt(2) / std), x > 0), (S.Zero, True))

    def _moment_generating_function(self, t):
        if False:
            return 10
        raise NotImplementedError('Moment generating function of the log-normal distribution is not defined.')

def LogNormal(name, mean, std):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a log-normal distribution.\n\n    Explanation\n    ===========\n\n    The density of the log-normal distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{x\\sqrt{2\\pi\\sigma^2}}\n                e^{-\\frac{\\left(\\ln x-\\mu\\right)^2}{2\\sigma^2}}\n\n    with :math:`x \\geq 0`.\n\n    Parameters\n    ==========\n\n    mu : Real number\n        The log-scale.\n    sigma : Real number\n        A shape. ($\\sigma^2 > 0$)\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import LogNormal, density\n    >>> from sympy import Symbol, pprint\n\n    >>> mu = Symbol("mu", real=True)\n    >>> sigma = Symbol("sigma", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = LogNormal("x", mu, sigma)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                          2\n           -(-mu + log(z))\n           -----------------\n                      2\n      ___      2*sigma\n    \\/ 2 *e\n    ------------------------\n            ____\n        2*\\/ pi *sigma*z\n\n\n    >>> X = LogNormal(\'x\', 0, 1) # Mean 0, standard deviation 1\n\n    >>> density(X)(z)\n    sqrt(2)*exp(-log(z)**2/2)/(2*sqrt(pi)*z)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Lognormal\n    .. [2] https://mathworld.wolfram.com/LogNormalDistribution.html\n\n    '
    return rv(name, LogNormalDistribution, (mean, std))

class LomaxDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'lamda')
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, lamda):
        if False:
            i = 10
            return i + 15
        _value_check(alpha.is_real, 'Shape parameter should be real.')
        _value_check(lamda.is_real, 'Scale parameter should be real.')
        _value_check(alpha.is_positive, 'Shape parameter should be positive.')
        _value_check(lamda.is_positive, 'Scale parameter should be positive.')

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        (lamba, alpha) = (self.lamda, self.alpha)
        return alpha / lamba * (S.One + x / lamba) ** (-alpha - 1)

def Lomax(name, alpha, lamda):
    if False:
        i = 10
        return i + 15
    "\n    Create a continuous random variable with a Lomax distribution.\n\n    Explanation\n    ===========\n\n    The density of the Lomax distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\alpha}{\\lambda}\\left[1+\\frac{x}{\\lambda}\\right]^{-(\\alpha+1)}\n\n    Parameters\n    ==========\n\n    alpha : Real Number, `\\alpha > 0`\n        Shape parameter\n    lamda : Real Number, `\\lambda > 0`\n        Scale parameter\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Lomax, density, cdf, E\n    >>> from sympy import symbols\n    >>> a, l = symbols('a, l', positive=True)\n    >>> X = Lomax('X', a, l)\n    >>> x = symbols('x')\n    >>> density(X)(x)\n    a*(1 + x/l)**(-a - 1)/l\n    >>> cdf(X)(x)\n    Piecewise((1 - 1/(1 + x/l)**a, x >= 0), (0, True))\n    >>> a = 2\n    >>> X = Lomax('X', a, l)\n    >>> E(X)\n    l\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Lomax_distribution\n\n    "
    return rv(name, LomaxDistribution, (alpha, lamda))

class MaxwellDistribution(SingleContinuousDistribution):
    _argnames = ('a',)
    set = Interval(0, oo)

    @staticmethod
    def check(a):
        if False:
            for i in range(10):
                print('nop')
        _value_check(a > 0, 'Parameter a must be positive.')

    def pdf(self, x):
        if False:
            print('Hello World!')
        a = self.a
        return sqrt(2 / pi) * x ** 2 * exp(-x ** 2 / (2 * a ** 2)) / a ** 3

    def _cdf(self, x):
        if False:
            print('Hello World!')
        a = self.a
        return erf(sqrt(2) * x / (2 * a)) - sqrt(2) * x * exp(-x ** 2 / (2 * a ** 2)) / (sqrt(pi) * a)

def Maxwell(name, a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with a Maxwell distribution.\n\n    Explanation\n    ===========\n\n    The density of the Maxwell distribution is given by\n\n    .. math::\n        f(x) := \\sqrt{\\frac{2}{\\pi}} \\frac{x^2 e^{-x^2/(2a^2)}}{a^3}\n\n    with :math:`x \\geq 0`.\n\n    .. TODO - what does the parameter mean?\n\n    Parameters\n    ==========\n\n    a : Real number, `a > 0`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Maxwell, density, E, variance\n    >>> from sympy import Symbol, simplify\n\n    >>> a = Symbol("a", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Maxwell("x", a)\n\n    >>> density(X)(z)\n    sqrt(2)*z**2*exp(-z**2/(2*a**2))/(sqrt(pi)*a**3)\n\n    >>> E(X)\n    2*sqrt(2)*a/sqrt(pi)\n\n    >>> simplify(variance(X))\n    a**2*(-8 + 3*pi)/pi\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Maxwell_distribution\n    .. [2] https://mathworld.wolfram.com/MaxwellDistribution.html\n\n    '
    return rv(name, MaxwellDistribution, (a,))

class MoyalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    @staticmethod
    def check(mu, sigma):
        if False:
            i = 10
            return i + 15
        _value_check(mu.is_real, 'Location parameter must be real.')
        _value_check(sigma.is_real and sigma > 0, 'Scale parameter must be real        and positive.')

    def pdf(self, x):
        if False:
            return 10
        (mu, sigma) = (self.mu, self.sigma)
        num = exp(-(exp(-(x - mu) / sigma) + (x - mu) / sigma) / 2)
        den = sqrt(2 * pi) * sigma
        return num / den

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        (mu, sigma) = (self.mu, self.sigma)
        term1 = exp(I * t * mu)
        term2 = 2 ** (-I * sigma * t) * gamma(Rational(1, 2) - I * t * sigma)
        return term1 * term2 / sqrt(pi)

    def _moment_generating_function(self, t):
        if False:
            while True:
                i = 10
        (mu, sigma) = (self.mu, self.sigma)
        term1 = exp(t * mu)
        term2 = 2 ** (-1 * sigma * t) * gamma(Rational(1, 2) - t * sigma)
        return term1 * term2 / sqrt(pi)

def Moyal(name, mu, sigma):
    if False:
        return 10
    '\n    Create a continuous random variable with a Moyal distribution.\n\n    Explanation\n    ===========\n\n    The density of the Moyal distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\exp-\\frac{1}{2}\\exp-\\frac{x-\\mu}{\\sigma}-\\frac{x-\\mu}{2\\sigma}}{\\sqrt{2\\pi}\\sigma}\n\n    with :math:`x \\in \\mathbb{R}`.\n\n    Parameters\n    ==========\n\n    mu : Real number\n        Location parameter\n    sigma : Real positive number\n        Scale parameter\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Moyal, density, cdf\n    >>> from sympy import Symbol, simplify\n    >>> mu = Symbol("mu", real=True)\n    >>> sigma = Symbol("sigma", positive=True, real=True)\n    >>> z = Symbol("z")\n    >>> X = Moyal("x", mu, sigma)\n    >>> density(X)(z)\n    sqrt(2)*exp(-exp((mu - z)/sigma)/2 - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)\n    >>> simplify(cdf(X)(z))\n    1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)\n\n    References\n    ==========\n\n    .. [1] https://reference.wolfram.com/language/ref/MoyalDistribution.html\n    .. [2] https://www.stat.rice.edu/~dobelman/textfiles/DistributionsHandbook.pdf\n\n    '
    return rv(name, MoyalDistribution, (mu, sigma))

class NakagamiDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'omega')
    set = Interval(0, oo)

    @staticmethod
    def check(mu, omega):
        if False:
            while True:
                i = 10
        _value_check(mu >= S.Half, 'Shape parameter mu must be greater than equal to 1/2.')
        _value_check(omega > 0, 'Spread parameter omega must be positive.')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (mu, omega) = (self.mu, self.omega)
        return 2 * mu ** mu / (gamma(mu) * omega ** mu) * x ** (2 * mu - 1) * exp(-mu / omega * x ** 2)

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        (mu, omega) = (self.mu, self.omega)
        return Piecewise((lowergamma(mu, mu / omega * x ** 2) / gamma(mu), x > 0), (S.Zero, True))

def Nakagami(name, mu, omega):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a Nakagami distribution.\n\n    Explanation\n    ===========\n\n    The density of the Nakagami distribution is given by\n\n    .. math::\n        f(x) := \\frac{2\\mu^\\mu}{\\Gamma(\\mu)\\omega^\\mu} x^{2\\mu-1}\n                \\exp\\left(-\\frac{\\mu}{\\omega}x^2 \\right)\n\n    with :math:`x > 0`.\n\n    Parameters\n    ==========\n\n    mu : Real number, `\\mu \\geq \\frac{1}{2}`, a shape\n    omega : Real number, `\\omega > 0`, the spread\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Nakagami, density, E, variance, cdf\n    >>> from sympy import Symbol, simplify, pprint\n\n    >>> mu = Symbol("mu", positive=True)\n    >>> omega = Symbol("omega", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Nakagami("x", mu, omega)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                                    2\n                               -mu*z\n                               -------\n        mu      -mu  2*mu - 1  omega\n    2*mu  *omega   *z        *e\n    ----------------------------------\n                Gamma(mu)\n\n    >>> simplify(E(X))\n    sqrt(mu)*sqrt(omega)*gamma(mu + 1/2)/gamma(mu + 1)\n\n    >>> V = simplify(variance(X))\n    >>> pprint(V, use_unicode=False)\n                        2\n             omega*Gamma (mu + 1/2)\n    omega - -----------------------\n            Gamma(mu)*Gamma(mu + 1)\n\n    >>> cdf(X)(z)\n    Piecewise((lowergamma(mu, mu*z**2/omega)/gamma(mu), z > 0),\n            (0, True))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Nakagami_distribution\n\n    '
    return rv(name, NakagamiDistribution, (mu, omega))

class NormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    @staticmethod
    def check(mean, std):
        if False:
            print('Hello World!')
        _value_check(std > 0, 'Standard deviation must be positive')

    def pdf(self, x):
        if False:
            return 10
        return exp(-(x - self.mean) ** 2 / (2 * self.std ** 2)) / (sqrt(2 * pi) * self.std)

    def _cdf(self, x):
        if False:
            print('Hello World!')
        (mean, std) = (self.mean, self.std)
        return erf(sqrt(2) * (-mean + x) / (2 * std)) / 2 + S.Half

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        (mean, std) = (self.mean, self.std)
        return exp(I * mean * t - std ** 2 * t ** 2 / 2)

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        (mean, std) = (self.mean, self.std)
        return exp(mean * t + std ** 2 * t ** 2 / 2)

    def _quantile(self, p):
        if False:
            return 10
        (mean, std) = (self.mean, self.std)
        return mean + std * sqrt(2) * erfinv(2 * p - 1)

def Normal(name, mean, std):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a Normal distribution.\n\n    Explanation\n    ===========\n\n    The density of the Normal distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{ -\\frac{(x-\\mu)^2}{2\\sigma^2} }\n\n    Parameters\n    ==========\n\n    mu : Real number or a list representing the mean or the mean vector\n    sigma : Real number or a positive definite square matrix,\n         :math:`\\sigma^2 > 0`, the variance\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Normal, density, E, std, cdf, skewness, quantile, marginal_distribution\n    >>> from sympy import Symbol, simplify, pprint\n\n    >>> mu = Symbol("mu")\n    >>> sigma = Symbol("sigma", positive=True)\n    >>> z = Symbol("z")\n    >>> y = Symbol("y")\n    >>> p = Symbol("p")\n    >>> X = Normal("x", mu, sigma)\n\n    >>> density(X)(z)\n    sqrt(2)*exp(-(-mu + z)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)\n\n    >>> C = simplify(cdf(X))(z) # it needs a little more help...\n    >>> pprint(C, use_unicode=False)\n       /  ___          \\\n       |\\/ 2 *(-mu + z)|\n    erf|---------------|\n       \\    2*sigma    /   1\n    -------------------- + -\n             2             2\n\n    >>> quantile(X)(p)\n    mu + sqrt(2)*sigma*erfinv(2*p - 1)\n\n    >>> simplify(skewness(X))\n    0\n\n    >>> X = Normal("x", 0, 1) # Mean 0, standard deviation 1\n    >>> density(X)(z)\n    sqrt(2)*exp(-z**2/2)/(2*sqrt(pi))\n\n    >>> E(2*X + 1)\n    1\n\n    >>> simplify(std(2*X + 1))\n    2\n\n    >>> m = Normal(\'X\', [1, 2], [[2, 1], [1, 2]])\n    >>> pprint(density(m)(y, z), use_unicode=False)\n              2          2\n             y    y*z   z\n           - -- + --- - -- + z - 1\n      ___    3     3    3\n    \\/ 3 *e\n    ------------------------------\n                 6*pi\n\n    >>> marginal_distribution(m, m[0])(1)\n     1/(2*sqrt(pi))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Normal_distribution\n    .. [2] https://mathworld.wolfram.com/NormalDistributionFunction.html\n\n    '
    if isinstance(mean, list) or (getattr(mean, 'is_Matrix', False) and isinstance(std, list)) or getattr(std, 'is_Matrix', False):
        from sympy.stats.joint_rv_types import MultivariateNormal
        return MultivariateNormal(name, mean, std)
    return rv(name, NormalDistribution, (mean, std))

class GaussianInverseDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'shape')

    @property
    def set(self):
        if False:
            return 10
        return Interval(0, oo)

    @staticmethod
    def check(mean, shape):
        if False:
            while True:
                i = 10
        _value_check(shape > 0, 'Shape parameter must be positive')
        _value_check(mean > 0, 'Mean must be positive')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (mu, s) = (self.mean, self.shape)
        return exp(-s * (x - mu) ** 2 / (2 * x * mu ** 2)) * sqrt(s / (2 * pi * x ** 3))

    def _cdf(self, x):
        if False:
            i = 10
            return i + 15
        from sympy.stats import cdf
        (mu, s) = (self.mean, self.shape)
        stdNormalcdf = cdf(Normal('x', 0, 1))
        first_term = stdNormalcdf(sqrt(s / x) * (x / mu - S.One))
        second_term = exp(2 * s / mu) * stdNormalcdf(-sqrt(s / x) * (x / mu + S.One))
        return first_term + second_term

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        (mu, s) = (self.mean, self.shape)
        return exp(s / mu * (1 - sqrt(1 - 2 * mu ** 2 * I * t / s)))

    def _moment_generating_function(self, t):
        if False:
            i = 10
            return i + 15
        (mu, s) = (self.mean, self.shape)
        return exp(s / mu * (1 - sqrt(1 - 2 * mu ** 2 * t / s)))

def GaussianInverse(name, mean, shape):
    if False:
        return 10
    '\n    Create a continuous random variable with an Inverse Gaussian distribution.\n    Inverse Gaussian distribution is also known as Wald distribution.\n\n    Explanation\n    ===========\n\n    The density of the Inverse Gaussian distribution is given by\n\n    .. math::\n        f(x) := \\sqrt{\\frac{\\lambda}{2\\pi x^3}} e^{-\\frac{\\lambda(x-\\mu)^2}{2x\\mu^2}}\n\n    Parameters\n    ==========\n\n    mu :\n        Positive number representing the mean.\n    lambda :\n        Positive number representing the shape parameter.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import GaussianInverse, density, E, std, skewness\n    >>> from sympy import Symbol, pprint\n\n    >>> mu = Symbol("mu", positive=True)\n    >>> lamda = Symbol("lambda", positive=True)\n    >>> z = Symbol("z", positive=True)\n    >>> X = GaussianInverse("x", mu, lamda)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n                                       2\n                      -lambda*(-mu + z)\n                      -------------------\n                                2\n      ___   ________        2*mu *z\n    \\/ 2 *\\/ lambda *e\n    -------------------------------------\n                    ____  3/2\n                2*\\/ pi *z\n\n    >>> E(X)\n    mu\n\n    >>> std(X).expand()\n    mu**(3/2)/sqrt(lambda)\n\n    >>> skewness(X).expand()\n    3*sqrt(mu)/sqrt(lambda)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution\n    .. [2] https://mathworld.wolfram.com/InverseGaussianDistribution.html\n\n    '
    return rv(name, GaussianInverseDistribution, (mean, shape))
Wald = GaussianInverse

class ParetoDistribution(SingleContinuousDistribution):
    _argnames = ('xm', 'alpha')

    @property
    def set(self):
        if False:
            for i in range(10):
                print('nop')
        return Interval(self.xm, oo)

    @staticmethod
    def check(xm, alpha):
        if False:
            for i in range(10):
                print('nop')
        _value_check(xm > 0, 'Xm must be positive')
        _value_check(alpha > 0, 'Alpha must be positive')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (xm, alpha) = (self.xm, self.alpha)
        return alpha * xm ** alpha / x ** (alpha + 1)

    def _cdf(self, x):
        if False:
            i = 10
            return i + 15
        (xm, alpha) = (self.xm, self.alpha)
        return Piecewise((S.One - xm ** alpha / x ** alpha, x >= xm), (0, True))

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        (xm, alpha) = (self.xm, self.alpha)
        return alpha * (-xm * t) ** alpha * uppergamma(-alpha, -xm * t)

    def _characteristic_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        (xm, alpha) = (self.xm, self.alpha)
        return alpha * (-I * xm * t) ** alpha * uppergamma(-alpha, -I * xm * t)

def Pareto(name, xm, alpha):
    if False:
        return 10
    '\n    Create a continuous random variable with the Pareto distribution.\n\n    Explanation\n    ===========\n\n    The density of the Pareto distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\alpha\\,x_m^\\alpha}{x^{\\alpha+1}}\n\n    with :math:`x \\in [x_m,\\infty]`.\n\n    Parameters\n    ==========\n\n    xm : Real number, `x_m > 0`, a scale\n    alpha : Real number, `\\alpha > 0`, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Pareto, density\n    >>> from sympy import Symbol\n\n    >>> xm = Symbol("xm", positive=True)\n    >>> beta = Symbol("beta", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Pareto("x", xm, beta)\n\n    >>> density(X)(z)\n    beta*xm**beta*z**(-beta - 1)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution\n    .. [2] https://mathworld.wolfram.com/ParetoDistribution.html\n\n    '
    return rv(name, ParetoDistribution, (xm, alpha))

class PowerFunctionDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'a', 'b')

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        return Interval(self.a, self.b)

    @staticmethod
    def check(alpha, a, b):
        if False:
            while True:
                i = 10
        _value_check(a.is_real, 'Continuous Boundary parameter should be real.')
        _value_check(b.is_real, 'Continuous Boundary parameter should be real.')
        _value_check(a < b, " 'a' the left Boundary must be smaller than 'b' the right Boundary.")
        _value_check(alpha.is_positive, 'Continuous Shape parameter should be positive.')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (alpha, a, b) = (self.alpha, self.a, self.b)
        num = alpha * (x - a) ** (alpha - 1)
        den = (b - a) ** alpha
        return num / den

def PowerFunction(name, alpha, a, b):
    if False:
        while True:
            i = 10
    '\n    Creates a continuous random variable with a Power Function Distribution.\n\n    Explanation\n    ===========\n\n    The density of PowerFunction distribution is given by\n\n    .. math::\n        f(x) := \\frac{{\\alpha}(x - a)^{\\alpha - 1}}{(b - a)^{\\alpha}}\n\n    with :math:`x \\in [a,b]`.\n\n    Parameters\n    ==========\n\n    alpha : Positive number, `0 < \\alpha`, the shape parameter\n    a : Real number, :math:`-\\infty < a`, the left boundary\n    b : Real number, :math:`a < b < \\infty`, the right boundary\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import PowerFunction, density, cdf, E, variance\n    >>> from sympy import Symbol\n    >>> alpha = Symbol("alpha", positive=True)\n    >>> a = Symbol("a", real=True)\n    >>> b = Symbol("b", real=True)\n    >>> z = Symbol("z")\n\n    >>> X = PowerFunction("X", 2, a, b)\n\n    >>> density(X)(z)\n    (-2*a + 2*z)/(-a + b)**2\n\n    >>> cdf(X)(z)\n    Piecewise((a**2/(a**2 - 2*a*b + b**2) - 2*a*z/(a**2 - 2*a*b + b**2) +\n    z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))\n\n    >>> alpha = 2\n    >>> a = 0\n    >>> b = 1\n    >>> Y = PowerFunction("Y", alpha, a, b)\n\n    >>> E(Y)\n    2/3\n\n    >>> variance(Y)\n    1/18\n\n    References\n    ==========\n\n    .. [1] https://web.archive.org/web/20200204081320/http://www.mathwave.com/help/easyfit/html/analyses/distributions/power_func.html\n\n    '
    return rv(name, PowerFunctionDistribution, (alpha, a, b))

class QuadraticUDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        if False:
            while True:
                i = 10
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        if False:
            print('Hello World!')
        _value_check(b > a, 'Parameter b must be in range (%s, oo).' % a)

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (self.a, self.b)
        alpha = 12 / (b - a) ** 3
        beta = (a + b) / 2
        return Piecewise((alpha * (x - beta) ** 2, And(a <= x, x <= b)), (S.Zero, True))

    def _moment_generating_function(self, t):
        if False:
            return 10
        (a, b) = (self.a, self.b)
        return -3 * (exp(a * t) * (4 + (a ** 2 + 2 * a * (-2 + b) + b ** 2) * t) - exp(b * t) * (4 + (-4 * b + (a + b) ** 2) * t)) / ((a - b) ** 3 * t ** 2)

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        (a, b) = (self.a, self.b)
        return -3 * I * (exp(I * a * t * exp(I * b * t)) * (4 * I - (-4 * b + (a + b) ** 2) * t)) / ((a - b) ** 3 * t ** 2)

def QuadraticU(name, a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Continuous Random Variable with a U-quadratic distribution.\n\n    Explanation\n    ===========\n\n    The density of the U-quadratic distribution is given by\n\n    .. math::\n        f(x) := \\alpha (x-\\beta)^2\n\n    with :math:`x \\in [a,b]`.\n\n    Parameters\n    ==========\n\n    a : Real number\n    b : Real number, :math:`a < b`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import QuadraticU, density\n    >>> from sympy import Symbol, pprint\n\n    >>> a = Symbol("a", real=True)\n    >>> b = Symbol("b", real=True)\n    >>> z = Symbol("z")\n\n    >>> X = QuadraticU("x", a, b)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n    /                2\n    |   /  a   b    \\\n    |12*|- - - - + z|\n    |   \\  2   2    /\n    <-----------------  for And(b >= z, a <= z)\n    |            3\n    |    (-a + b)\n    |\n    \\        0                 otherwise\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/U-quadratic_distribution\n\n    '
    return rv(name, QuadraticUDistribution, (a, b))

class RaisedCosineDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    @property
    def set(self):
        if False:
            return 10
        return Interval(self.mu - self.s, self.mu + self.s)

    @staticmethod
    def check(mu, s):
        if False:
            for i in range(10):
                print('nop')
        _value_check(s > 0, 's must be positive')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (mu, s) = (self.mu, self.s)
        return Piecewise(((1 + cos(pi * (x - mu) / s)) / (2 * s), And(mu - s <= x, x <= mu + s)), (S.Zero, True))

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        (mu, s) = (self.mu, self.s)
        return Piecewise((exp(-I * pi * mu / s) / 2, Eq(t, -pi / s)), (exp(I * pi * mu / s) / 2, Eq(t, pi / s)), (pi ** 2 * sin(s * t) * exp(I * mu * t) / (s * t * (pi ** 2 - s ** 2 * t ** 2)), True))

    def _moment_generating_function(self, t):
        if False:
            i = 10
            return i + 15
        (mu, s) = (self.mu, self.s)
        return pi ** 2 * sinh(s * t) * exp(mu * t) / (s * t * (pi ** 2 + s ** 2 * t ** 2))

def RaisedCosine(name, mu, s):
    if False:
        return 10
    '\n    Create a Continuous Random Variable with a raised cosine distribution.\n\n    Explanation\n    ===========\n\n    The density of the raised cosine distribution is given by\n\n    .. math::\n        f(x) := \\frac{1}{2s}\\left(1+\\cos\\left(\\frac{x-\\mu}{s}\\pi\\right)\\right)\n\n    with :math:`x \\in [\\mu-s,\\mu+s]`.\n\n    Parameters\n    ==========\n\n    mu : Real number\n    s : Real number, `s > 0`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import RaisedCosine, density\n    >>> from sympy import Symbol, pprint\n\n    >>> mu = Symbol("mu", real=True)\n    >>> s = Symbol("s", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = RaisedCosine("x", mu, s)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n    /   /pi*(-mu + z)\\\n    |cos|------------| + 1\n    |   \\     s      /\n    <---------------------  for And(z >= mu - s, z <= mu + s)\n    |         2*s\n    |\n    \\          0                        otherwise\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Raised_cosine_distribution\n\n    '
    return rv(name, RaisedCosineDistribution, (mu, s))

class RayleighDistribution(SingleContinuousDistribution):
    _argnames = ('sigma',)
    set = Interval(0, oo)

    @staticmethod
    def check(sigma):
        if False:
            i = 10
            return i + 15
        _value_check(sigma > 0, 'Scale parameter sigma must be positive.')

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        sigma = self.sigma
        return x / sigma ** 2 * exp(-x ** 2 / (2 * sigma ** 2))

    def _cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        sigma = self.sigma
        return 1 - exp(-(x ** 2 / (2 * sigma ** 2)))

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        sigma = self.sigma
        return 1 - sigma * t * exp(-sigma ** 2 * t ** 2 / 2) * sqrt(pi / 2) * (erfi(sigma * t / sqrt(2)) - I)

    def _moment_generating_function(self, t):
        if False:
            while True:
                i = 10
        sigma = self.sigma
        return 1 + sigma * t * exp(sigma ** 2 * t ** 2 / 2) * sqrt(pi / 2) * (erf(sigma * t / sqrt(2)) + 1)

def Rayleigh(name, sigma):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with a Rayleigh distribution.\n\n    Explanation\n    ===========\n\n    The density of the Rayleigh distribution is given by\n\n    .. math ::\n        f(x) := \\frac{x}{\\sigma^2} e^{-x^2/2\\sigma^2}\n\n    with :math:`x > 0`.\n\n    Parameters\n    ==========\n\n    sigma : Real number, `\\sigma > 0`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Rayleigh, density, E, variance\n    >>> from sympy import Symbol\n\n    >>> sigma = Symbol("sigma", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Rayleigh("x", sigma)\n\n    >>> density(X)(z)\n    z*exp(-z**2/(2*sigma**2))/sigma**2\n\n    >>> E(X)\n    sqrt(2)*sqrt(pi)*sigma/2\n\n    >>> variance(X)\n    -pi*sigma**2/2 + 2*sigma**2\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Rayleigh_distribution\n    .. [2] https://mathworld.wolfram.com/RayleighDistribution.html\n\n    '
    return rv(name, RayleighDistribution, (sigma,))

class ReciprocalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        if False:
            print('Hello World!')
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        if False:
            for i in range(10):
                print('nop')
        _value_check(a > 0, 'Parameter > 0. a = %s' % a)
        _value_check(a < b, 'Parameter b must be in range (%s, +oo]. b = %s' % (a, b))

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (self.a, self.b)
        return 1 / (x * (log(b) - log(a)))

def Reciprocal(name, a, b):
    if False:
        while True:
            i = 10
    "Creates a continuous random variable with a reciprocal distribution.\n\n\n    Parameters\n    ==========\n\n    a : Real number, :math:`0 < a`\n    b : Real number, :math:`a < b`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Reciprocal, density, cdf\n    >>> from sympy import symbols\n    >>> a, b, x = symbols('a, b, x', positive=True)\n    >>> R = Reciprocal('R', a, b)\n\n    >>> density(R)(x)\n    1/(x*(-log(a) + log(b)))\n    >>> cdf(R)(x)\n    Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))\n\n    Reference\n    =========\n\n    .. [1] https://en.wikipedia.org/wiki/Reciprocal_distribution\n\n    "
    return rv(name, ReciprocalDistribution, (a, b))

class ShiftedGompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')
    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        if False:
            print('Hello World!')
        _value_check(b > 0, 'b must be positive')
        _value_check(eta > 0, 'eta must be positive')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (b, eta) = (self.b, self.eta)
        return b * exp(-b * x) * exp(-eta * exp(-b * x)) * (1 + eta * (1 - exp(-b * x)))

def ShiftedGompertz(name, b, eta):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a Shifted Gompertz distribution.\n\n    Explanation\n    ===========\n\n    The density of the Shifted Gompertz distribution is given by\n\n    .. math::\n        f(x) := b e^{-b x} e^{-\\eta \\exp(-b x)} \\left[1 + \\eta(1 - e^(-bx)) \\right]\n\n    with :math:`x \\in [0, \\infty)`.\n\n    Parameters\n    ==========\n\n    b : Real number, `b > 0`, a scale\n    eta : Real number, `\\eta > 0`, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n    >>> from sympy.stats import ShiftedGompertz, density\n    >>> from sympy import Symbol\n\n    >>> b = Symbol("b", positive=True)\n    >>> eta = Symbol("eta", positive=True)\n    >>> x = Symbol("x")\n\n    >>> X = ShiftedGompertz("x", b, eta)\n\n    >>> density(X)(x)\n    b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution\n\n    '
    return rv(name, ShiftedGompertzDistribution, (b, eta))

class StudentTDistribution(SingleContinuousDistribution):
    _argnames = ('nu',)
    set = Interval(-oo, oo)

    @staticmethod
    def check(nu):
        if False:
            i = 10
            return i + 15
        _value_check(nu > 0, 'Degrees of freedom nu must be positive.')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        nu = self.nu
        return 1 / (sqrt(nu) * beta_fn(S.Half, nu / 2)) * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)

    def _cdf(self, x):
        if False:
            return 10
        nu = self.nu
        return S.Half + x * gamma((nu + 1) / 2) * hyper((S.Half, (nu + 1) / 2), (Rational(3, 2),), -x ** 2 / nu) / (sqrt(pi * nu) * gamma(nu / 2))

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('The moment generating function for the Student-T distribution is undefined.')

def StudentT(name, nu):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a student\'s t distribution.\n\n    Explanation\n    ===========\n\n    The density of the student\'s t distribution is given by\n\n    .. math::\n        f(x) := \\frac{\\Gamma \\left(\\frac{\\nu+1}{2} \\right)}\n                {\\sqrt{\\nu\\pi}\\Gamma \\left(\\frac{\\nu}{2} \\right)}\n                \\left(1+\\frac{x^2}{\\nu} \\right)^{-\\frac{\\nu+1}{2}}\n\n    Parameters\n    ==========\n\n    nu : Real number, `\\nu > 0`, the degrees of freedom\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import StudentT, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> nu = Symbol("nu", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = StudentT("x", nu)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n               nu   1\n             - -- - -\n               2    2\n     /     2\\\n     |    z |\n     |1 + --|\n     \\    nu/\n    -----------------\n      ____  /     nu\\\n    \\/ nu *B|1/2, --|\n            \\     2 /\n\n    >>> cdf(X)(z)\n    1/2 + z*gamma(nu/2 + 1/2)*hyper((1/2, nu/2 + 1/2), (3/2,),\n                                -z**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Student_t-distribution\n    .. [2] https://mathworld.wolfram.com/Studentst-Distribution.html\n\n    '
    return rv(name, StudentTDistribution, (nu,))

class TrapezoidalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c', 'd')

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        return Interval(self.a, self.d)

    @staticmethod
    def check(a, b, c, d):
        if False:
            for i in range(10):
                print('nop')
        _value_check(a < d, 'Lower bound parameter a < %s. a = %s' % (d, a))
        _value_check((a <= b, b < c), 'Level start parameter b must be in range [%s, %s). b = %s' % (a, c, b))
        _value_check((b < c, c <= d), 'Level end parameter c must be in range (%s, %s]. c = %s' % (b, d, c))
        _value_check(d >= c, 'Upper bound parameter d > %s. d = %s' % (c, d))

    def pdf(self, x):
        if False:
            while True:
                i = 10
        (a, b, c, d) = (self.a, self.b, self.c, self.d)
        return Piecewise((2 * (x - a) / ((b - a) * (d + c - a - b)), And(a <= x, x < b)), (2 / (d + c - a - b), And(b <= x, x < c)), (2 * (d - x) / ((d - c) * (d + c - a - b)), And(c <= x, x <= d)), (S.Zero, True))

def Trapezoidal(name, a, b, c, d):
    if False:
        i = 10
        return i + 15
    '\n    Create a continuous random variable with a trapezoidal distribution.\n\n    Explanation\n    ===========\n\n    The density of the trapezoidal distribution is given by\n\n    .. math::\n        f(x) := \\begin{cases}\n                  0 & \\mathrm{for\\ } x < a, \\\\\n                  \\frac{2(x-a)}{(b-a)(d+c-a-b)} & \\mathrm{for\\ } a \\le x < b, \\\\\n                  \\frac{2}{d+c-a-b} & \\mathrm{for\\ } b \\le x < c, \\\\\n                  \\frac{2(d-x)}{(d-c)(d+c-a-b)} & \\mathrm{for\\ } c \\le x < d, \\\\\n                  0 & \\mathrm{for\\ } d < x.\n                \\end{cases}\n\n    Parameters\n    ==========\n\n    a : Real number, :math:`a < d`\n    b : Real number, :math:`a \\le b < c`\n    c : Real number, :math:`b < c \\le d`\n    d : Real number\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Trapezoidal, density\n    >>> from sympy import Symbol, pprint\n\n    >>> a = Symbol("a")\n    >>> b = Symbol("b")\n    >>> c = Symbol("c")\n    >>> d = Symbol("d")\n    >>> z = Symbol("z")\n\n    >>> X = Trapezoidal("x", a,b,c,d)\n\n    >>> pprint(density(X)(z), use_unicode=False)\n    /        -2*a + 2*z\n    |-------------------------  for And(a <= z, b > z)\n    |(-a + b)*(-a - b + c + d)\n    |\n    |           2\n    |     --------------        for And(b <= z, c > z)\n    <     -a - b + c + d\n    |\n    |        2*d - 2*z\n    |-------------------------  for And(d >= z, c <= z)\n    |(-c + d)*(-a - b + c + d)\n    |\n    \\            0                     otherwise\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Trapezoidal_distribution\n\n    '
    return rv(name, TrapezoidalDistribution, (a, b, c, d))

class TriangularDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c')

    @property
    def set(self):
        if False:
            while True:
                i = 10
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b, c):
        if False:
            for i in range(10):
                print('nop')
        _value_check(b > a, 'Parameter b > %s. b = %s' % (a, b))
        _value_check((a <= c, c <= b), 'Parameter c must be in range [%s, %s]. c = %s' % (a, b, c))

    def pdf(self, x):
        if False:
            print('Hello World!')
        (a, b, c) = (self.a, self.b, self.c)
        return Piecewise((2 * (x - a) / ((b - a) * (c - a)), And(a <= x, x < c)), (2 / (b - a), Eq(x, c)), (2 * (b - x) / ((b - a) * (b - c)), And(c < x, x <= b)), (S.Zero, True))

    def _characteristic_function(self, t):
        if False:
            return 10
        (a, b, c) = (self.a, self.b, self.c)
        return -2 * ((b - c) * exp(I * a * t) - (b - a) * exp(I * c * t) + (c - a) * exp(I * b * t)) / ((b - a) * (c - a) * (b - c) * t ** 2)

    def _moment_generating_function(self, t):
        if False:
            i = 10
            return i + 15
        (a, b, c) = (self.a, self.b, self.c)
        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / ((b - a) * (c - a) * (b - c) * t ** 2)

def Triangular(name, a, b, c):
    if False:
        return 10
    '\n    Create a continuous random variable with a triangular distribution.\n\n    Explanation\n    ===========\n\n    The density of the triangular distribution is given by\n\n    .. math::\n        f(x) := \\begin{cases}\n                  0 & \\mathrm{for\\ } x < a, \\\\\n                  \\frac{2(x-a)}{(b-a)(c-a)} & \\mathrm{for\\ } a \\le x < c, \\\\\n                  \\frac{2}{b-a} & \\mathrm{for\\ } x = c, \\\\\n                  \\frac{2(b-x)}{(b-a)(b-c)} & \\mathrm{for\\ } c < x \\le b, \\\\\n                  0 & \\mathrm{for\\ } b < x.\n                \\end{cases}\n\n    Parameters\n    ==========\n\n    a : Real number, :math:`a \\in \\left(-\\infty, \\infty\\right)`\n    b : Real number, :math:`a < b`\n    c : Real number, :math:`a \\leq c \\leq b`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Triangular, density\n    >>> from sympy import Symbol, pprint\n\n    >>> a = Symbol("a")\n    >>> b = Symbol("b")\n    >>> c = Symbol("c")\n    >>> z = Symbol("z")\n\n    >>> X = Triangular("x", a,b,c)\n\n    >>> pprint(density(X)(z), use_unicode=False)\n    /    -2*a + 2*z\n    |-----------------  for And(a <= z, c > z)\n    |(-a + b)*(-a + c)\n    |\n    |       2\n    |     ------              for c = z\n    <     -a + b\n    |\n    |   2*b - 2*z\n    |----------------   for And(b >= z, c < z)\n    |(-a + b)*(b - c)\n    |\n    \\        0                otherwise\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Triangular_distribution\n    .. [2] https://mathworld.wolfram.com/TriangularDistribution.html\n\n    '
    return rv(name, TriangularDistribution, (a, b, c))

class UniformDistribution(SingleContinuousDistribution):
    _argnames = ('left', 'right')

    @property
    def set(self):
        if False:
            while True:
                i = 10
        return Interval(self.left, self.right)

    @staticmethod
    def check(left, right):
        if False:
            for i in range(10):
                print('nop')
        _value_check(left < right, 'Lower limit should be less than Upper limit.')

    def pdf(self, x):
        if False:
            return 10
        (left, right) = (self.left, self.right)
        return Piecewise((S.One / (right - left), And(left <= x, x <= right)), (S.Zero, True))

    def _cdf(self, x):
        if False:
            return 10
        (left, right) = (self.left, self.right)
        return Piecewise((S.Zero, x < left), ((x - left) / (right - left), x <= right), (S.One, True))

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        (left, right) = (self.left, self.right)
        return Piecewise(((exp(I * t * right) - exp(I * t * left)) / (I * t * (right - left)), Ne(t, 0)), (S.One, True))

    def _moment_generating_function(self, t):
        if False:
            while True:
                i = 10
        (left, right) = (self.left, self.right)
        return Piecewise(((exp(t * right) - exp(t * left)) / (t * (right - left)), Ne(t, 0)), (S.One, True))

    def expectation(self, expr, var, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['evaluate'] = True
        result = SingleContinuousDistribution.expectation(self, expr, var, **kwargs)
        result = result.subs({Max(self.left, self.right): self.right, Min(self.left, self.right): self.left})
        return result

def Uniform(name, left, right):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a continuous random variable with a uniform distribution.\n\n    Explanation\n    ===========\n\n    The density of the uniform distribution is given by\n\n    .. math::\n        f(x) := \\begin{cases}\n                  \\frac{1}{b - a} & \\text{for } x \\in [a,b]  \\\\\n                  0               & \\text{otherwise}\n                \\end{cases}\n\n    with :math:`x \\in [a,b]`.\n\n    Parameters\n    ==========\n\n    a : Real number, :math:`-\\infty < a`, the left boundary\n    b : Real number, :math:`a < b < \\infty`, the right boundary\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Uniform, density, cdf, E, variance\n    >>> from sympy import Symbol, simplify\n\n    >>> a = Symbol("a", negative=True)\n    >>> b = Symbol("b", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Uniform("x", a, b)\n\n    >>> density(X)(z)\n    Piecewise((1/(-a + b), (b >= z) & (a <= z)), (0, True))\n\n    >>> cdf(X)(z)\n    Piecewise((0, a > z), ((-a + z)/(-a + b), b >= z), (1, True))\n\n    >>> E(X)\n    a/2 + b/2\n\n    >>> simplify(variance(X))\n    a**2/12 - a*b/6 + b**2/12\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29\n    .. [2] https://mathworld.wolfram.com/UniformDistribution.html\n\n    '
    return rv(name, UniformDistribution, (left, right))

class UniformSumDistribution(SingleContinuousDistribution):
    _argnames = ('n',)

    @property
    def set(self):
        if False:
            while True:
                i = 10
        return Interval(0, self.n)

    @staticmethod
    def check(n):
        if False:
            for i in range(10):
                print('nop')
        _value_check((n > 0, n.is_integer), 'Parameter n must be positive integer.')

    def pdf(self, x):
        if False:
            return 10
        n = self.n
        k = Dummy('k')
        return 1 / factorial(n - 1) * Sum((-1) ** k * binomial(n, k) * (x - k) ** (n - 1), (k, 0, floor(x)))

    def _cdf(self, x):
        if False:
            print('Hello World!')
        n = self.n
        k = Dummy('k')
        return Piecewise((S.Zero, x < 0), (1 / factorial(n) * Sum((-1) ** k * binomial(n, k) * (x - k) ** n, (k, 0, floor(x))), x <= n), (S.One, True))

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        return ((exp(I * t) - 1) / (I * t)) ** self.n

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        return ((exp(t) - 1) / t) ** self.n

def UniformSum(name, n):
    if False:
        while True:
            i = 10
    '\n    Create a continuous random variable with an Irwin-Hall distribution.\n\n    Explanation\n    ===========\n\n    The probability distribution function depends on a single parameter\n    $n$ which is an integer.\n\n    The density of the Irwin-Hall distribution is given by\n\n    .. math ::\n        f(x) := \\frac{1}{(n-1)!}\\sum_{k=0}^{\\left\\lfloor x\\right\\rfloor}(-1)^k\n                \\binom{n}{k}(x-k)^{n-1}\n\n    Parameters\n    ==========\n\n    n : A positive integer, `n > 0`\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import UniformSum, density, cdf\n    >>> from sympy import Symbol, pprint\n\n    >>> n = Symbol("n", integer=True)\n    >>> z = Symbol("z")\n\n    >>> X = UniformSum("x", n)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n    floor(z)\n      ___\n      \\  `\n       \\         k         n - 1 /n\\\n        )    (-1) *(-k + z)     *| |\n       /                         \\k/\n      /__,\n     k = 0\n    --------------------------------\n                (n - 1)!\n\n    >>> cdf(X)(z)\n    Piecewise((0, z < 0), (Sum((-1)**_k*(-_k + z)**n*binomial(n, _k),\n                    (_k, 0, floor(z)))/factorial(n), n >= z), (1, True))\n\n\n    Compute cdf with specific \'x\' and \'n\' values as follows :\n    >>> cdf(UniformSum("x", 5), evaluate=False)(2).doit()\n    9/40\n\n    The argument evaluate=False prevents an attempt at evaluation\n    of the sum for general n, before the argument 2 is passed.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Uniform_sum_distribution\n    .. [2] https://mathworld.wolfram.com/UniformSumDistribution.html\n\n    '
    return rv(name, UniformSumDistribution, (n,))

class VonMisesDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'k')
    set = Interval(0, 2 * pi)

    @staticmethod
    def check(mu, k):
        if False:
            while True:
                i = 10
        _value_check(k > 0, 'k must be positive')

    def pdf(self, x):
        if False:
            print('Hello World!')
        (mu, k) = (self.mu, self.k)
        return exp(k * cos(x - mu)) / (2 * pi * besseli(0, k))

def VonMises(name, mu, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Continuous Random Variable with a von Mises distribution.\n\n    Explanation\n    ===========\n\n    The density of the von Mises distribution is given by\n\n    .. math::\n        f(x) := \\frac{e^{\\kappa\\cos(x-\\mu)}}{2\\pi I_0(\\kappa)}\n\n    with :math:`x \\in [0,2\\pi]`.\n\n    Parameters\n    ==========\n\n    mu : Real number\n        Measure of location.\n    k : Real number\n        Measure of concentration.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import VonMises, density\n    >>> from sympy import Symbol, pprint\n\n    >>> mu = Symbol("mu")\n    >>> k = Symbol("k", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = VonMises("x", mu, k)\n\n    >>> D = density(X)(z)\n    >>> pprint(D, use_unicode=False)\n         k*cos(mu - z)\n        e\n    ------------------\n    2*pi*besseli(0, k)\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Von_Mises_distribution\n    .. [2] https://mathworld.wolfram.com/vonMisesDistribution.html\n\n    '
    return rv(name, VonMisesDistribution, (mu, k))

class WeibullDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        if False:
            for i in range(10):
                print('nop')
        _value_check(alpha > 0, 'Alpha must be positive')
        _value_check(beta > 0, 'Beta must be positive')

    def pdf(self, x):
        if False:
            return 10
        (alpha, beta) = (self.alpha, self.beta)
        return beta * (x / alpha) ** (beta - 1) * exp(-(x / alpha) ** beta) / alpha

def Weibull(name, alpha, beta):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a Weibull distribution.\n\n    Explanation\n    ===========\n\n    The density of the Weibull distribution is given by\n\n    .. math::\n        f(x) := \\begin{cases}\n                  \\frac{k}{\\lambda}\\left(\\frac{x}{\\lambda}\\right)^{k-1}\n                  e^{-(x/\\lambda)^{k}} & x\\geq0\\\\\n                  0 & x<0\n                \\end{cases}\n\n    Parameters\n    ==========\n\n    lambda : Real number, $\\lambda > 0$, a scale\n    k : Real number, $k > 0$, a shape\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Weibull, density, E, variance\n    >>> from sympy import Symbol, simplify\n\n    >>> l = Symbol("lambda", positive=True)\n    >>> k = Symbol("k", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Weibull("x", l, k)\n\n    >>> density(X)(z)\n    k*(z/lambda)**(k - 1)*exp(-(z/lambda)**k)/lambda\n\n    >>> simplify(E(X))\n    lambda*gamma(1 + 1/k)\n\n    >>> simplify(variance(X))\n    lambda**2*(-gamma(1 + 1/k)**2 + gamma(1 + 2/k))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Weibull_distribution\n    .. [2] https://mathworld.wolfram.com/WeibullDistribution.html\n\n    '
    return rv(name, WeibullDistribution, (alpha, beta))

class WignerSemicircleDistribution(SingleContinuousDistribution):
    _argnames = ('R',)

    @property
    def set(self):
        if False:
            print('Hello World!')
        return Interval(-self.R, self.R)

    @staticmethod
    def check(R):
        if False:
            for i in range(10):
                print('nop')
        _value_check(R > 0, 'Radius R must be positive.')

    def pdf(self, x):
        if False:
            while True:
                i = 10
        R = self.R
        return 2 / (pi * R ** 2) * sqrt(R ** 2 - x ** 2)

    def _characteristic_function(self, t):
        if False:
            return 10
        return Piecewise((2 * besselj(1, self.R * t) / (self.R * t), Ne(t, 0)), (S.One, True))

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        return Piecewise((2 * besseli(1, self.R * t) / (self.R * t), Ne(t, 0)), (S.One, True))

def WignerSemicircle(name, R):
    if False:
        print('Hello World!')
    '\n    Create a continuous random variable with a Wigner semicircle distribution.\n\n    Explanation\n    ===========\n\n    The density of the Wigner semicircle distribution is given by\n\n    .. math::\n        f(x) := \\frac2{\\pi R^2}\\,\\sqrt{R^2-x^2}\n\n    with :math:`x \\in [-R,R]`.\n\n    Parameters\n    ==========\n\n    R : Real number, `R > 0`, the radius\n\n    Returns\n    =======\n\n    A RandomSymbol.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import WignerSemicircle, density, E\n    >>> from sympy import Symbol\n\n    >>> R = Symbol("R", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = WignerSemicircle("x", R)\n\n    >>> density(X)(z)\n    2*sqrt(R**2 - z**2)/(pi*R**2)\n\n    >>> E(X)\n    0\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Wigner_semicircle_distribution\n    .. [2] https://mathworld.wolfram.com/WignersSemicircleLaw.html\n\n    '
    return rv(name, WignerSemicircleDistribution, (R,))