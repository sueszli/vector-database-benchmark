"""

Contains
========
FlorySchulz
Geometric
Hermite
Logarithmic
NegativeBinomial
Poisson
Skellam
YuleSimon
Zeta
"""
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import polylog, zeta
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random
__all__ = ['FlorySchulz', 'Geometric', 'Hermite', 'Logarithmic', 'NegativeBinomial', 'Poisson', 'Skellam', 'YuleSimon', 'Zeta']

def rv(symbol, cls, *args, **kwargs):
    if False:
        while True:
            i = 10
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleDiscretePSpace(symbol, dist)
    if any((is_random(arg) for arg in args)):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value

class DiscreteDistributionHandmade(SingleDiscreteDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=S.Integers):
        if False:
            for i in range(10):
                print('nop')
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
        val = Sum(pdf(x), (x, set._inf, set._sup)).doit()
        _value_check(Eq(val, 1) != S.false, 'The pdf is incorrect on the given set.')

def DiscreteRV(symbol, density, set=S.Integers, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a Discrete Random Variable given the following:\n\n    Parameters\n    ==========\n\n    symbol : Symbol\n        Represents name of the random variable.\n    density : Expression containing symbol\n        Represents probability density function.\n    set : set\n        Represents the region where the pdf is valid, by default is real line.\n    check : bool\n        If True, it will check whether the given density\n        integrates to 1 over the given set. If False, it\n        will not perform this check. Default is False.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import DiscreteRV, P, E\n    >>> from sympy import Rational, Symbol\n    >>> x = Symbol('x')\n    >>> n = 10\n    >>> density = Rational(1, 10)\n    >>> X = DiscreteRV(x, density, set=set(range(n)))\n    >>> E(X)\n    9/2\n    >>> P(X>3)\n    3/5\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    "
    set = sympify(set)
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, DiscreteDistributionHandmade, pdf, set, **kwargs)

class FlorySchulzDistribution(SingleDiscreteDistribution):
    _argnames = ('a',)
    set = S.Naturals

    @staticmethod
    def check(a):
        if False:
            for i in range(10):
                print('nop')
        _value_check((0 < a, a < 1), 'a must be between 0 and 1')

    def pdf(self, k):
        if False:
            for i in range(10):
                print('nop')
        a = self.a
        return a ** 2 * k * (1 - a) ** (k - 1)

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        a = self.a
        return a ** 2 * exp(I * t) / (1 + (a - 1) * exp(I * t)) ** 2

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        a = self.a
        return a ** 2 * exp(t) / (1 + (a - 1) * exp(t)) ** 2

def FlorySchulz(name, a):
    if False:
        print('Hello World!')
    '\n    Create a discrete random variable with a FlorySchulz distribution.\n\n    The density of the FlorySchulz distribution is given by\n\n    .. math::\n        f(k) := (a^2) k (1 - a)^{k-1}\n\n    Parameters\n    ==========\n\n    a : A real number between 0 and 1\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import density, E, variance, FlorySchulz\n    >>> from sympy import Symbol, S\n\n    >>> a = S.One / 5\n    >>> z = Symbol("z")\n\n    >>> X = FlorySchulz("x", a)\n\n    >>> density(X)(z)\n    (5/4)**(1 - z)*z/25\n\n    >>> E(X)\n    9\n\n    >>> variance(X)\n    40\n\n    References\n    ==========\n\n    https://en.wikipedia.org/wiki/Flory%E2%80%93Schulz_distribution\n    '
    return rv(name, FlorySchulzDistribution, a)

class GeometricDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)
    set = S.Naturals

    @staticmethod
    def check(p):
        if False:
            return 10
        _value_check((0 < p, p <= 1), 'p must be between 0 and 1')

    def pdf(self, k):
        if False:
            return 10
        return (1 - self.p) ** (k - 1) * self.p

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        p = self.p
        return p * exp(I * t) / (1 - (1 - p) * exp(I * t))

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        p = self.p
        return p * exp(t) / (1 - (1 - p) * exp(t))

def Geometric(name, p):
    if False:
        return 10
    '\n    Create a discrete random variable with a Geometric distribution.\n\n    Explanation\n    ===========\n\n    The density of the Geometric distribution is given by\n\n    .. math::\n        f(k) := p (1 - p)^{k - 1}\n\n    Parameters\n    ==========\n\n    p : A probability between 0 and 1\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Geometric, density, E, variance\n    >>> from sympy import Symbol, S\n\n    >>> p = S.One / 5\n    >>> z = Symbol("z")\n\n    >>> X = Geometric("x", p)\n\n    >>> density(X)(z)\n    (5/4)**(1 - z)/5\n\n    >>> E(X)\n    5\n\n    >>> variance(X)\n    20\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Geometric_distribution\n    .. [2] https://mathworld.wolfram.com/GeometricDistribution.html\n\n    '
    return rv(name, GeometricDistribution, p)

class HermiteDistribution(SingleDiscreteDistribution):
    _argnames = ('a1', 'a2')
    set = S.Naturals0

    @staticmethod
    def check(a1, a2):
        if False:
            i = 10
            return i + 15
        _value_check(a1.is_nonnegative, 'Parameter a1 must be >= 0.')
        _value_check(a2.is_nonnegative, 'Parameter a2 must be >= 0.')

    def pdf(self, k):
        if False:
            print('Hello World!')
        (a1, a2) = (self.a1, self.a2)
        term1 = exp(-(a1 + a2))
        j = Dummy('j', integer=True)
        num = a1 ** (k - 2 * j) * a2 ** j
        den = factorial(k - 2 * j) * factorial(j)
        return term1 * Sum(num / den, (j, 0, k // 2)).doit()

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        (a1, a2) = (self.a1, self.a2)
        term1 = a1 * (exp(t) - 1)
        term2 = a2 * (exp(2 * t) - 1)
        return exp(term1 + term2)

    def _characteristic_function(self, t):
        if False:
            print('Hello World!')
        (a1, a2) = (self.a1, self.a2)
        term1 = a1 * (exp(I * t) - 1)
        term2 = a2 * (exp(2 * I * t) - 1)
        return exp(term1 + term2)

def Hermite(name, a1, a2):
    if False:
        while True:
            i = 10
    '\n    Create a discrete random variable with a Hermite distribution.\n\n    Explanation\n    ===========\n\n    The density of the Hermite distribution is given by\n\n    .. math::\n        f(x):= e^{-a_1 -a_2}\\sum_{j=0}^{\\left \\lfloor x/2 \\right \\rfloor}\n                    \\frac{a_{1}^{x-2j}a_{2}^{j}}{(x-2j)!j!}\n\n    Parameters\n    ==========\n\n    a1 : A Positive number greater than equal to 0.\n    a2 : A Positive number greater than equal to 0.\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Hermite, density, E, variance\n    >>> from sympy import Symbol\n\n    >>> a1 = Symbol("a1", positive=True)\n    >>> a2 = Symbol("a2", positive=True)\n    >>> x = Symbol("x")\n\n    >>> H = Hermite("H", a1=5, a2=4)\n\n    >>> density(H)(2)\n    33*exp(-9)/2\n\n    >>> E(H)\n    13\n\n    >>> variance(H)\n    21\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hermite_distribution\n\n    '
    return rv(name, HermiteDistribution, a1, a2)

class LogarithmicDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)
    set = S.Naturals

    @staticmethod
    def check(p):
        if False:
            while True:
                i = 10
        _value_check((p > 0, p < 1), 'p should be between 0 and 1')

    def pdf(self, k):
        if False:
            for i in range(10):
                print('nop')
        p = self.p
        return -1 * p ** k / (k * log(1 - p))

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        p = self.p
        return log(1 - p * exp(I * t)) / log(1 - p)

    def _moment_generating_function(self, t):
        if False:
            return 10
        p = self.p
        return log(1 - p * exp(t)) / log(1 - p)

def Logarithmic(name, p):
    if False:
        return 10
    '\n    Create a discrete random variable with a Logarithmic distribution.\n\n    Explanation\n    ===========\n\n    The density of the Logarithmic distribution is given by\n\n    .. math::\n        f(k) := \\frac{-p^k}{k \\ln{(1 - p)}}\n\n    Parameters\n    ==========\n\n    p : A value between 0 and 1\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Logarithmic, density, E, variance\n    >>> from sympy import Symbol, S\n\n    >>> p = S.One / 5\n    >>> z = Symbol("z")\n\n    >>> X = Logarithmic("x", p)\n\n    >>> density(X)(z)\n    -1/(5**z*z*log(4/5))\n\n    >>> E(X)\n    -1/(-4*log(5) + 8*log(2))\n\n    >>> variance(X)\n    -1/((-4*log(5) + 8*log(2))*(-2*log(5) + 4*log(2))) + 1/(-64*log(2)*log(5) + 64*log(2)**2 + 16*log(5)**2) - 10/(-32*log(5) + 64*log(2))\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution\n    .. [2] https://mathworld.wolfram.com/LogarithmicDistribution.html\n\n    '
    return rv(name, LogarithmicDistribution, p)

class NegativeBinomialDistribution(SingleDiscreteDistribution):
    _argnames = ('r', 'p')
    set = S.Naturals0

    @staticmethod
    def check(r, p):
        if False:
            i = 10
            return i + 15
        _value_check(r > 0, 'r should be positive')
        _value_check((p > 0, p < 1), 'p should be between 0 and 1')

    def pdf(self, k):
        if False:
            for i in range(10):
                print('nop')
        r = self.r
        p = self.p
        return binomial(k + r - 1, k) * (1 - p) ** r * p ** k

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        r = self.r
        p = self.p
        return ((1 - p) / (1 - p * exp(I * t))) ** r

    def _moment_generating_function(self, t):
        if False:
            while True:
                i = 10
        r = self.r
        p = self.p
        return ((1 - p) / (1 - p * exp(t))) ** r

def NegativeBinomial(name, r, p):
    if False:
        while True:
            i = 10
    '\n    Create a discrete random variable with a Negative Binomial distribution.\n\n    Explanation\n    ===========\n\n    The density of the Negative Binomial distribution is given by\n\n    .. math::\n        f(k) := \\binom{k + r - 1}{k} (1 - p)^r p^k\n\n    Parameters\n    ==========\n\n    r : A positive value\n    p : A value between 0 and 1\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import NegativeBinomial, density, E, variance\n    >>> from sympy import Symbol, S\n\n    >>> r = 5\n    >>> p = S.One / 5\n    >>> z = Symbol("z")\n\n    >>> X = NegativeBinomial("x", r, p)\n\n    >>> density(X)(z)\n    1024*binomial(z + 4, z)/(3125*5**z)\n\n    >>> E(X)\n    5/4\n\n    >>> variance(X)\n    25/16\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution\n    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html\n\n    '
    return rv(name, NegativeBinomialDistribution, r, p)

class PoissonDistribution(SingleDiscreteDistribution):
    _argnames = ('lamda',)
    set = S.Naturals0

    @staticmethod
    def check(lamda):
        if False:
            for i in range(10):
                print('nop')
        _value_check(lamda > 0, 'Lambda must be positive')

    def pdf(self, k):
        if False:
            while True:
                i = 10
        return self.lamda ** k / factorial(k) * exp(-self.lamda)

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        return exp(self.lamda * (exp(I * t) - 1))

    def _moment_generating_function(self, t):
        if False:
            print('Hello World!')
        return exp(self.lamda * (exp(t) - 1))

def Poisson(name, lamda):
    if False:
        while True:
            i = 10
    '\n    Create a discrete random variable with a Poisson distribution.\n\n    Explanation\n    ===========\n\n    The density of the Poisson distribution is given by\n\n    .. math::\n        f(k) := \\frac{\\lambda^{k} e^{- \\lambda}}{k!}\n\n    Parameters\n    ==========\n\n    lamda : Positive number, a rate\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Poisson, density, E, variance\n    >>> from sympy import Symbol, simplify\n\n    >>> rate = Symbol("lambda", positive=True)\n    >>> z = Symbol("z")\n\n    >>> X = Poisson("x", rate)\n\n    >>> density(X)(z)\n    lambda**z*exp(-lambda)/factorial(z)\n\n    >>> E(X)\n    lambda\n\n    >>> simplify(variance(X))\n    lambda\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution\n    .. [2] https://mathworld.wolfram.com/PoissonDistribution.html\n\n    '
    return rv(name, PoissonDistribution, lamda)

class SkellamDistribution(SingleDiscreteDistribution):
    _argnames = ('mu1', 'mu2')
    set = S.Integers

    @staticmethod
    def check(mu1, mu2):
        if False:
            print('Hello World!')
        _value_check(mu1 >= 0, 'Parameter mu1 must be >= 0')
        _value_check(mu2 >= 0, 'Parameter mu2 must be >= 0')

    def pdf(self, k):
        if False:
            while True:
                i = 10
        (mu1, mu2) = (self.mu1, self.mu2)
        term1 = exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2)
        term2 = besseli(k, 2 * sqrt(mu1 * mu2))
        return term1 * term2

    def _cdf(self, x):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError("Skellam doesn't have closed form for the CDF.")

    def _characteristic_function(self, t):
        if False:
            return 10
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(I * t) + mu2 * exp(-I * t))

    def _moment_generating_function(self, t):
        if False:
            return 10
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(t) + mu2 * exp(-t))

def Skellam(name, mu1, mu2):
    if False:
        i = 10
        return i + 15
    '\n    Create a discrete random variable with a Skellam distribution.\n\n    Explanation\n    ===========\n\n    The Skellam is the distribution of the difference N1 - N2\n    of two statistically independent random variables N1 and N2\n    each Poisson-distributed with respective expected values mu1 and mu2.\n\n    The density of the Skellam distribution is given by\n\n    .. math::\n        f(k) := e^{-(\\mu_1+\\mu_2)}(\\frac{\\mu_1}{\\mu_2})^{k/2}I_k(2\\sqrt{\\mu_1\\mu_2})\n\n    Parameters\n    ==========\n\n    mu1 : A non-negative value\n    mu2 : A non-negative value\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Skellam, density, E, variance\n    >>> from sympy import Symbol, pprint\n\n    >>> z = Symbol("z", integer=True)\n    >>> mu1 = Symbol("mu1", positive=True)\n    >>> mu2 = Symbol("mu2", positive=True)\n    >>> X = Skellam("x", mu1, mu2)\n\n    >>> pprint(density(X)(z), use_unicode=False)\n         z\n         -\n         2\n    /mu1\\   -mu1 - mu2        /       _____   _____\\\n    |---| *e          *besseli\\z, 2*\\/ mu1 *\\/ mu2 /\n    \\mu2/\n    >>> E(X)\n    mu1 - mu2\n    >>> variance(X).expand()\n    mu1 + mu2\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Skellam_distribution\n\n    '
    return rv(name, SkellamDistribution, mu1, mu2)

class YuleSimonDistribution(SingleDiscreteDistribution):
    _argnames = ('rho',)
    set = S.Naturals

    @staticmethod
    def check(rho):
        if False:
            print('Hello World!')
        _value_check(rho > 0, 'rho should be positive')

    def pdf(self, k):
        if False:
            print('Hello World!')
        rho = self.rho
        return rho * beta(k, rho + 1)

    def _cdf(self, x):
        if False:
            while True:
                i = 10
        return Piecewise((1 - floor(x) * beta(floor(x), self.rho + 1), x >= 1), (0, True))

    def _characteristic_function(self, t):
        if False:
            i = 10
            return i + 15
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(I * t)) * exp(I * t) / (rho + 1)

    def _moment_generating_function(self, t):
        if False:
            return 10
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(t)) * exp(t) / (rho + 1)

def YuleSimon(name, rho):
    if False:
        while True:
            i = 10
    '\n    Create a discrete random variable with a Yule-Simon distribution.\n\n    Explanation\n    ===========\n\n    The density of the Yule-Simon distribution is given by\n\n    .. math::\n        f(k) := \\rho B(k, \\rho + 1)\n\n    Parameters\n    ==========\n\n    rho : A positive value\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import YuleSimon, density, E, variance\n    >>> from sympy import Symbol, simplify\n\n    >>> p = 5\n    >>> z = Symbol("z")\n\n    >>> X = YuleSimon("x", p)\n\n    >>> density(X)(z)\n    5*beta(z, 6)\n\n    >>> simplify(E(X))\n    5/4\n\n    >>> simplify(variance(X))\n    25/48\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution\n\n    '
    return rv(name, YuleSimonDistribution, rho)

class ZetaDistribution(SingleDiscreteDistribution):
    _argnames = ('s',)
    set = S.Naturals

    @staticmethod
    def check(s):
        if False:
            while True:
                i = 10
        _value_check(s > 1, 's should be greater than 1')

    def pdf(self, k):
        if False:
            for i in range(10):
                print('nop')
        s = self.s
        return 1 / (k ** s * zeta(s))

    def _characteristic_function(self, t):
        if False:
            while True:
                i = 10
        return polylog(self.s, exp(I * t)) / zeta(self.s)

    def _moment_generating_function(self, t):
        if False:
            i = 10
            return i + 15
        return polylog(self.s, exp(t)) / zeta(self.s)

def Zeta(name, s):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a discrete random variable with a Zeta distribution.\n\n    Explanation\n    ===========\n\n    The density of the Zeta distribution is given by\n\n    .. math::\n        f(k) := \\frac{1}{k^s \\zeta{(s)}}\n\n    Parameters\n    ==========\n\n    s : A value greater than 1\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Zeta, density, E, variance\n    >>> from sympy import Symbol\n\n    >>> s = 5\n    >>> z = Symbol("z")\n\n    >>> X = Zeta("x", s)\n\n    >>> density(X)(z)\n    1/(z**5*zeta(5))\n\n    >>> E(X)\n    pi**4/(90*zeta(5))\n\n    >>> variance(X)\n    -pi**8/(8100*zeta(5)**2) + zeta(3)/zeta(5)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Zeta_distribution\n\n    '
    return rv(name, ZetaDistribution, s)