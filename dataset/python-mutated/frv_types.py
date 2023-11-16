"""
Finite Discrete Random Variables - Prebuilt variable types

Contains
========
FiniteRV
DiscreteUniform
Die
Bernoulli
Coin
Binomial
BetaBinomial
Hypergeometric
Rademacher
IdealSoliton
RobustSoliton
"""
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import Integer, Rational
from sympy.core.relational import Eq, Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Or
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import Intersection, Interval
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.rv import _value_check, Density, is_random
from sympy.utilities.iterables import multiset
from sympy.utilities.misc import filldedent
__all__ = ['FiniteRV', 'DiscreteUniform', 'Die', 'Bernoulli', 'Coin', 'Binomial', 'BetaBinomial', 'Hypergeometric', 'Rademacher', 'IdealSoliton', 'RobustSoliton']

def rv(name, cls, *args, **kwargs):
    if False:
        return 10
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleFinitePSpace(name, dist)
    if any((is_random(arg) for arg in args)):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(name, CompoundDistribution(dist))
    return pspace.value

class FiniteDistributionHandmade(SingleFiniteDistribution):

    @property
    def dict(self):
        if False:
            return 10
        return self.args[0]

    def pmf(self, x):
        if False:
            while True:
                i = 10
        x = Symbol('x')
        return Lambda(x, Piecewise(*[(v, Eq(k, x)) for (k, v) in self.dict.items()] + [(S.Zero, True)]))

    @property
    def set(self):
        if False:
            for i in range(10):
                print('nop')
        return set(self.dict.keys())

    @staticmethod
    def check(density):
        if False:
            return 10
        for p in density.values():
            _value_check((p >= 0, p <= 1), 'Probability at a point must be between 0 and 1.')
        val = sum(density.values())
        _value_check(Eq(val, 1) != S.false, 'Total Probability must be 1.')

def FiniteRV(name, density, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a Finite Random Variable given a dict representing the density.\n\n    Parameters\n    ==========\n\n    name : Symbol\n        Represents name of the random variable.\n    density : dict\n        Dictionary containing the pdf of finite distribution\n    check : bool\n        If True, it will check whether the given density\n        integrates to 1 over the given set. If False, it\n        will not perform this check. Default is False.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import FiniteRV, P, E\n\n    >>> density = {0: .1, 1: .2, 2: .3, 3: .4}\n    >>> X = FiniteRV('X', density)\n\n    >>> E(X)\n    2.00000000000000\n    >>> P(X >= 2)\n    0.700000000000000\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    "
    kwargs['check'] = kwargs.pop('check', False)
    return rv(name, FiniteDistributionHandmade, density, **kwargs)

class DiscreteUniformDistribution(SingleFiniteDistribution):

    @staticmethod
    def check(*args):
        if False:
            return 10
        if len(set(args)) != len(args):
            weights = multiset(args)
            n = Integer(len(args))
            for k in weights:
                weights[k] /= n
            raise ValueError(filldedent('\n                Repeated args detected but set expected. For a\n                distribution having different weights for each\n                item use the following:') + '\nS("FiniteRV(%s, %s)")' % ("'X'", weights))

    @property
    def p(self):
        if False:
            for i in range(10):
                print('nop')
        return Rational(1, len(self.args))

    @property
    @cacheit
    def dict(self):
        if False:
            print('Hello World!')
        return {k: self.p for k in self.set}

    @property
    def set(self):
        if False:
            return 10
        return set(self.args)

    def pmf(self, x):
        if False:
            while True:
                i = 10
        if x in self.args:
            return self.p
        else:
            return S.Zero

def DiscreteUniform(name, items):
    if False:
        print('Hello World!')
    "\n    Create a Finite Random Variable representing a uniform distribution over\n    the input set.\n\n    Parameters\n    ==========\n\n    items : list/tuple\n        Items over which Uniform distribution is to be made\n\n    Examples\n    ========\n\n    >>> from sympy.stats import DiscreteUniform, density\n    >>> from sympy import symbols\n\n    >>> X = DiscreteUniform('X', symbols('a b c')) # equally likely over a, b, c\n    >>> density(X).dict\n    {a: 1/3, b: 1/3, c: 1/3}\n\n    >>> Y = DiscreteUniform('Y', list(range(5))) # distribution over a range\n    >>> density(Y).dict\n    {0: 1/5, 1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Discrete_uniform_distribution\n    .. [2] https://mathworld.wolfram.com/DiscreteUniformDistribution.html\n\n    "
    return rv(name, DiscreteUniformDistribution, *items)

class DieDistribution(SingleFiniteDistribution):
    _argnames = ('sides',)

    @staticmethod
    def check(sides):
        if False:
            i = 10
            return i + 15
        _value_check((sides.is_positive, sides.is_integer), 'number of sides must be a positive integer.')

    @property
    def is_symbolic(self):
        if False:
            i = 10
            return i + 15
        return not self.sides.is_number

    @property
    def high(self):
        if False:
            print('Hello World!')
        return self.sides

    @property
    def low(self):
        if False:
            i = 10
            return i + 15
        return S.One

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.sides))
        return set(map(Integer, range(1, self.sides + 1)))

    def pmf(self, x):
        if False:
            i = 10
            return i + 15
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond = Ge(x, 1) & Le(x, self.sides) & Contains(x, S.Integers)
        return Piecewise((S.One / self.sides, cond), (S.Zero, True))

def Die(name, sides=6):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a Finite Random Variable representing a fair die.\n\n    Parameters\n    ==========\n\n    sides : Integer\n        Represents the number of sides of the Die, by default is 6\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Die, density\n    >>> from sympy import Symbol\n\n    >>> D6 = Die('D6', 6) # Six sided Die\n    >>> density(D6).dict\n    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}\n\n    >>> D4 = Die('D4', 4) # Four sided Die\n    >>> density(D4).dict\n    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}\n\n    >>> n = Symbol('n', positive=True, integer=True)\n    >>> Dn = Die('Dn', n) # n sided Die\n    >>> density(Dn).dict\n    Density(DieDistribution(n))\n    >>> density(Dn).dict.subs(n, 4).doit()\n    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}\n\n    Returns\n    =======\n\n    RandomSymbol\n    "
    return rv(name, DieDistribution, sides)

class BernoulliDistribution(SingleFiniteDistribution):
    _argnames = ('p', 'succ', 'fail')

    @staticmethod
    def check(p, succ, fail):
        if False:
            while True:
                i = 10
        _value_check((p >= 0, p <= 1), 'p should be in range [0, 1].')

    @property
    def set(self):
        if False:
            print('Hello World!')
        return {self.succ, self.fail}

    def pmf(self, x):
        if False:
            print('Hello World!')
        if isinstance(self.succ, Symbol) and isinstance(self.fail, Symbol):
            return Piecewise((self.p, x == self.succ), (1 - self.p, x == self.fail), (S.Zero, True))
        return Piecewise((self.p, Eq(x, self.succ)), (1 - self.p, Eq(x, self.fail)), (S.Zero, True))

def Bernoulli(name, p, succ=1, fail=0):
    if False:
        while True:
            i = 10
    "\n    Create a Finite Random Variable representing a Bernoulli process.\n\n    Parameters\n    ==========\n\n    p : Rational number between 0 and 1\n       Represents probability of success\n    succ : Integer/symbol/string\n       Represents event of success\n    fail : Integer/symbol/string\n       Represents event of failure\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Bernoulli, density\n    >>> from sympy import S\n\n    >>> X = Bernoulli('X', S(3)/4) # 1-0 Bernoulli variable, probability = 3/4\n    >>> density(X).dict\n    {0: 1/4, 1: 3/4}\n\n    >>> X = Bernoulli('X', S.Half, 'Heads', 'Tails') # A fair coin toss\n    >>> density(X).dict\n    {Heads: 1/2, Tails: 1/2}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Bernoulli_distribution\n    .. [2] https://mathworld.wolfram.com/BernoulliDistribution.html\n\n    "
    return rv(name, BernoulliDistribution, p, succ, fail)

def Coin(name, p=S.Half):
    if False:
        return 10
    '\n    Create a Finite Random Variable representing a Coin toss.\n\n    Parameters\n    ==========\n\n    p : Rational Number between 0 and 1\n      Represents probability of getting "Heads", by default is Half\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Coin, density\n    >>> from sympy import Rational\n\n    >>> C = Coin(\'C\') # A fair coin toss\n    >>> density(C).dict\n    {H: 1/2, T: 1/2}\n\n    >>> C2 = Coin(\'C2\', Rational(3, 5)) # An unfair coin\n    >>> density(C2).dict\n    {H: 3/5, T: 2/5}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    See Also\n    ========\n\n    sympy.stats.Binomial\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Coin_flipping\n\n    '
    return rv(name, BernoulliDistribution, p, 'H', 'T')

class BinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'p', 'succ', 'fail')

    @staticmethod
    def check(n, p, succ, fail):
        if False:
            for i in range(10):
                print('nop')
        _value_check((n.is_integer, n.is_nonnegative), "'n' must be nonnegative integer.")
        _value_check((p <= 1, p >= 0), 'p should be in range [0, 1].')

    @property
    def high(self):
        if False:
            for i in range(10):
                print('nop')
        return self.n

    @property
    def low(self):
        if False:
            while True:
                i = 10
        return S.Zero

    @property
    def is_symbolic(self):
        if False:
            i = 10
            return i + 15
        return not self.n.is_number

    @property
    def set(self):
        if False:
            while True:
                i = 10
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(self.dict.keys())

    def pmf(self, x):
        if False:
            return 10
        (n, p) = (self.n, self.p)
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond = Ge(x, 0) & Le(x, n) & Contains(x, S.Integers)
        return Piecewise((binomial(n, x) * p ** x * (1 - p) ** (n - x), cond), (S.Zero, True))

    @property
    @cacheit
    def dict(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_symbolic:
            return Density(self)
        return {k * self.succ + (self.n - k) * self.fail: self.pmf(k) for k in range(0, self.n + 1)}

def Binomial(name, n, p, succ=1, fail=0):
    if False:
        print('Hello World!')
    '\n    Create a Finite Random Variable representing a binomial distribution.\n\n    Parameters\n    ==========\n\n    n : Positive Integer\n      Represents number of trials\n    p : Rational Number between 0 and 1\n      Represents probability of success\n    succ : Integer/symbol/string\n      Represents event of success, by default is 1\n    fail : Integer/symbol/string\n      Represents event of failure, by default is 0\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Binomial, density\n    >>> from sympy import S, Symbol\n\n    >>> X = Binomial(\'X\', 4, S.Half) # Four "coin flips"\n    >>> density(X).dict\n    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}\n\n    >>> n = Symbol(\'n\', positive=True, integer=True)\n    >>> p = Symbol(\'p\', positive=True)\n    >>> X = Binomial(\'X\', n, S.Half) # n "coin flips"\n    >>> density(X).dict\n    Density(BinomialDistribution(n, 1/2, 1, 0))\n    >>> density(X).dict.subs(n, 4).doit()\n    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution\n    .. [2] https://mathworld.wolfram.com/BinomialDistribution.html\n\n    '
    return rv(name, BinomialDistribution, n, p, succ, fail)

class BetaBinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'alpha', 'beta')

    @staticmethod
    def check(n, alpha, beta):
        if False:
            return 10
        _value_check((n.is_integer, n.is_nonnegative), "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check(alpha > 0, "'alpha' must be: alpha > 0 . alpha = %s" % str(alpha))
        _value_check(beta > 0, "'beta' must be: beta > 0 . beta = %s" % str(beta))

    @property
    def high(self):
        if False:
            while True:
                i = 10
        return self.n

    @property
    def low(self):
        if False:
            for i in range(10):
                print('nop')
        return S.Zero

    @property
    def is_symbolic(self):
        if False:
            while True:
                i = 10
        return not self.n.is_number

    @property
    def set(self):
        if False:
            return 10
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(map(Integer, range(self.n + 1)))

    def pmf(self, k):
        if False:
            i = 10
            return i + 15
        (n, a, b) = (self.n, self.alpha, self.beta)
        return binomial(n, k) * beta_fn(k + a, n - k + b) / beta_fn(a, b)

def BetaBinomial(name, n, alpha, beta):
    if False:
        return 10
    "\n    Create a Finite Random Variable representing a Beta-binomial distribution.\n\n    Parameters\n    ==========\n\n    n : Positive Integer\n      Represents number of trials\n    alpha : Real positive number\n    beta : Real positive number\n\n    Examples\n    ========\n\n    >>> from sympy.stats import BetaBinomial, density\n\n    >>> X = BetaBinomial('X', 2, 1, 1)\n    >>> density(X).dict\n    {0: 1/3, 1: 2*beta(2, 2), 2: 1/3}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution\n    .. [2] https://mathworld.wolfram.com/BetaBinomialDistribution.html\n\n    "
    return rv(name, BetaBinomialDistribution, n, alpha, beta)

class HypergeometricDistribution(SingleFiniteDistribution):
    _argnames = ('N', 'm', 'n')

    @staticmethod
    def check(n, N, m):
        if False:
            while True:
                i = 10
        _value_check((N.is_integer, N.is_nonnegative), "'N' must be nonnegative integer. N = %s." % str(N))
        _value_check((n.is_integer, n.is_nonnegative), "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check((m.is_integer, m.is_nonnegative), "'m' must be nonnegative integer. m = %s." % str(m))

    @property
    def is_symbolic(self):
        if False:
            print('Hello World!')
        return not all((x.is_number for x in (self.N, self.m, self.n)))

    @property
    def high(self):
        if False:
            i = 10
            return i + 15
        return Piecewise((self.n, Lt(self.n, self.m) != False), (self.m, True))

    @property
    def low(self):
        if False:
            i = 10
            return i + 15
        return Piecewise((0, Gt(0, self.n + self.m - self.N) != False), (self.n + self.m - self.N, True))

    @property
    def set(self):
        if False:
            return 10
        (N, m, n) = (self.N, self.m, self.n)
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(self.low, self.high))
        return set(range(max(0, n + m - N), min(n, m) + 1))

    def pmf(self, k):
        if False:
            for i in range(10):
                print('nop')
        (N, m, n) = (self.N, self.m, self.n)
        return S(binomial(m, k) * binomial(N - m, n - k)) / binomial(N, n)

def Hypergeometric(name, N, m, n):
    if False:
        i = 10
        return i + 15
    "\n    Create a Finite Random Variable representing a hypergeometric distribution.\n\n    Parameters\n    ==========\n\n    N : Positive Integer\n      Represents finite population of size N.\n    m : Positive Integer\n      Represents number of trials with required feature.\n    n : Positive Integer\n      Represents numbers of draws.\n\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Hypergeometric, density\n\n    >>> X = Hypergeometric('X', 10, 5, 3) # 10 marbles, 5 white (success), 3 draws\n    >>> density(X).dict\n    {0: 1/12, 1: 5/12, 2: 5/12, 3: 1/12}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hypergeometric_distribution\n    .. [2] https://mathworld.wolfram.com/HypergeometricDistribution.html\n\n    "
    return rv(name, HypergeometricDistribution, N, m, n)

class RademacherDistribution(SingleFiniteDistribution):

    @property
    def set(self):
        if False:
            for i in range(10):
                print('nop')
        return {-1, 1}

    @property
    def pmf(self):
        if False:
            while True:
                i = 10
        k = Dummy('k')
        return Lambda(k, Piecewise((S.Half, Or(Eq(k, -1), Eq(k, 1))), (S.Zero, True)))

def Rademacher(name):
    if False:
        while True:
            i = 10
    "\n    Create a Finite Random Variable representing a Rademacher distribution.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Rademacher, density\n\n    >>> X = Rademacher('X')\n    >>> density(X).dict\n    {-1: 1/2, 1: 1/2}\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    See Also\n    ========\n\n    sympy.stats.Bernoulli\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Rademacher_distribution\n\n    "
    return rv(name, RademacherDistribution)

class IdealSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        if False:
            while True:
                i = 10
        _value_check(k.is_integer and k.is_positive, "'k' must be a positive integer.")

    @property
    def low(self):
        if False:
            print('Hello World!')
        return S.One

    @property
    def high(self):
        if False:
            return 10
        return self.k

    @property
    def set(self):
        if False:
            while True:
                i = 10
        return set(map(Integer, range(1, self.k + 1)))

    @property
    @cacheit
    def dict(self):
        if False:
            while True:
                i = 10
        if self.k.is_Symbol:
            return Density(self)
        d = {1: Rational(1, self.k)}
        d.update({i: Rational(1, i * (i - 1)) for i in range(2, self.k + 1)})
        return d

    def pmf(self, x):
        if False:
            return 10
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        return Piecewise((1 / self.k, cond1), (1 / (x * (x - 1)), cond2), (S.Zero, True))

def IdealSoliton(name, k):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a Finite Random Variable of Ideal Soliton Distribution\n\n    Parameters\n    ==========\n\n    k : Positive Integer\n        Represents the number of input symbols in an LT (Luby Transform) code.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import IdealSoliton, density, P, E\n    >>> sol = IdealSoliton('sol', 5)\n    >>> density(sol).dict\n    {1: 1/5, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20}\n    >>> density(sol).set\n    {1, 2, 3, 4, 5}\n\n    >>> from sympy import Symbol\n    >>> k = Symbol('k', positive=True, integer=True)\n    >>> sol = IdealSoliton('sol', k)\n    >>> density(sol).dict\n    Density(IdealSolitonDistribution(k))\n    >>> density(sol).dict.subs(k, 10).doit()\n    {1: 1/10, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20, 6: 1/30, 7: 1/42, 8: 1/56, 9: 1/72, 10: 1/90}\n\n    >>> E(sol.subs(k, 10))\n    7381/2520\n\n    >>> P(sol.subs(k, 4) > 2)\n    1/4\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Ideal_distribution\n    .. [2] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf\n\n    "
    return rv(name, IdealSolitonDistribution, k)

class RobustSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k', 'delta', 'c')

    @staticmethod
    def check(k, delta, c):
        if False:
            i = 10
            return i + 15
        _value_check(k.is_integer and k.is_positive, "'k' must be a positive integer")
        _value_check(Gt(delta, 0) and Le(delta, 1), "'delta' must be a real number in the interval (0,1)")
        _value_check(c.is_positive, "'c' must be a positive real number.")

    @property
    def R(self):
        if False:
            i = 10
            return i + 15
        return self.c * log(self.k / self.delta) * self.k ** 0.5

    @property
    def Z(self):
        if False:
            i = 10
            return i + 15
        z = 0
        for i in Range(1, round(self.k / self.R)):
            z += 1 / i
        z += log(self.R / self.delta)
        return 1 + z * self.R / self.k

    @property
    def low(self):
        if False:
            return 10
        return S.One

    @property
    def high(self):
        if False:
            while True:
                i = 10
        return self.k

    @property
    def set(self):
        if False:
            i = 10
            return i + 15
        return set(map(Integer, range(1, self.k + 1)))

    @property
    def is_symbolic(self):
        if False:
            print('Hello World!')
        return not (self.k.is_number and self.c.is_number and self.delta.is_number)

    def pmf(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or 'RandomSymbol' not %s" % type(x))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        rho = Piecewise((Rational(1, self.k), cond1), (Rational(1, x * (x - 1)), cond2), (S.Zero, True))
        cond1 = Ge(x, 1) & Le(x, round(self.k / self.R) - 1)
        cond2 = Eq(x, round(self.k / self.R))
        tau = Piecewise((self.R / (self.k * x), cond1), (self.R * log(self.R / self.delta) / self.k, cond2), (S.Zero, True))
        return (rho + tau) / self.Z

def RobustSoliton(name, k, delta, c):
    if False:
        return 10
    "\n    Create a Finite Random Variable of Robust Soliton Distribution\n\n    Parameters\n    ==========\n\n    k : Positive Integer\n        Represents the number of input symbols in an LT (Luby Transform) code.\n    delta : Positive Rational Number\n            Represents the failure probability. Must be in the interval (0,1).\n    c : Positive Rational Number\n        Constant of proportionality. Values close to 1 are recommended\n\n    Examples\n    ========\n\n    >>> from sympy.stats import RobustSoliton, density, P, E\n    >>> robSol = RobustSoliton('robSol', 5, 0.5, 0.01)\n    >>> density(robSol).dict\n    {1: 0.204253668152708, 2: 0.490631107897393, 3: 0.165210624506162, 4: 0.0834387731899302, 5: 0.0505633404760675}\n    >>> density(robSol).set\n    {1, 2, 3, 4, 5}\n\n    >>> from sympy import Symbol\n    >>> k = Symbol('k', positive=True, integer=True)\n    >>> c = Symbol('c', positive=True)\n    >>> robSol = RobustSoliton('robSol', k, 0.5, c)\n    >>> density(robSol).dict\n    Density(RobustSolitonDistribution(k, 0.5, c))\n    >>> density(robSol).dict.subs(k, 10).subs(c, 0.03).doit()\n    {1: 0.116641095387194, 2: 0.467045731687165, 3: 0.159984123349381, 4: 0.0821431680681869, 5: 0.0505765646770100,\n    6: 0.0345781523420719, 7: 0.0253132820710503, 8: 0.0194459129233227, 9: 0.0154831166726115, 10: 0.0126733075238887}\n\n    >>> E(robSol.subs(k, 10).subs(c, 0.05))\n    2.91358846104106\n\n    >>> P(robSol.subs(k, 4).subs(c, 0.1) > 2)\n    0.243650614389834\n\n    Returns\n    =======\n\n    RandomSymbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Robust_distribution\n    .. [2] https://www.inference.org.uk/mackay/itprnn/ps/588.596.pdf\n    .. [3] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf\n\n    "
    return rv(name, RobustSolitonDistribution, k, delta, c)