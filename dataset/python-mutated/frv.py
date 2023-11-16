"""
Finite Discrete Random Variables Module

See Also
========
sympy.stats.frv_types
sympy.stats.rv
sympy.stats.crv
"""
from itertools import product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import I, nan
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And, Or
from sympy.sets.sets import Intersection
from sympy.core.containers import Dict
from sympy.core.logic import Logic
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet
from sympy.stats.rv import RandomDomain, ProductDomain, ConditionalDomain, PSpace, IndependentProductPSpace, SinglePSpace, random_symbols, sumsets, rv_subs, NamedArgsMixin, Density, Distribution

class FiniteDensity(dict):
    """
    A domain with Finite Density.
    """

    def __call__(self, item):
        if False:
            i = 10
            return i + 15
        '\n        Make instance of a class callable.\n\n        If item belongs to current instance of a class, return it.\n\n        Otherwise, return 0.\n        '
        item = sympify(item)
        if item in self:
            return self[item]
        else:
            return 0

    @property
    def dict(self):
        if False:
            while True:
                i = 10
        '\n        Return item as dictionary.\n        '
        return dict(self)

class FiniteDomain(RandomDomain):
    """
    A domain with discrete finite support

    Represented using a FiniteSet.
    """
    is_Finite = True

    @property
    def symbols(self):
        if False:
            while True:
                i = 10
        return FiniteSet((sym for (sym, val) in self.elements))

    @property
    def elements(self):
        if False:
            return 10
        return self.args[0]

    @property
    def dict(self):
        if False:
            i = 10
            return i + 15
        return FiniteSet(*[Dict(dict(el)) for el in self.elements])

    def __contains__(self, other):
        if False:
            print('Hello World!')
        return other in self.elements

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self.elements.__iter__()

    def as_boolean(self):
        if False:
            for i in range(10):
                print('nop')
        return Or(*[And(*[Eq(sym, val) for (sym, val) in item]) for item in self])

class SingleFiniteDomain(FiniteDomain):
    """
    A FiniteDomain over a single symbol/set

    Example: The possibilities of a *single* die roll.
    """

    def __new__(cls, symbol, set):
        if False:
            print('Hello World!')
        if not isinstance(set, FiniteSet) and (not isinstance(set, Intersection)):
            set = FiniteSet(*set)
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[0]

    @property
    def symbols(self):
        if False:
            return 10
        return FiniteSet(self.symbol)

    @property
    def set(self):
        if False:
            print('Hello World!')
        return self.args[1]

    @property
    def elements(self):
        if False:
            while True:
                i = 10
        return FiniteSet(*[frozenset(((self.symbol, elem),)) for elem in self.set])

    def __iter__(self):
        if False:
            print('Hello World!')
        return (frozenset(((self.symbol, elem),)) for elem in self.set)

    def __contains__(self, other):
        if False:
            while True:
                i = 10
        (sym, val) = tuple(other)[0]
        return sym == self.symbol and val in self.set

class ProductFiniteDomain(ProductDomain, FiniteDomain):
    """
    A Finite domain consisting of several other FiniteDomains

    Example: The possibilities of the rolls of three independent dice
    """

    def __iter__(self):
        if False:
            while True:
                i = 10
        proditer = product(*self.domains)
        return (sumsets(items) for items in proditer)

    @property
    def elements(self):
        if False:
            print('Hello World!')
        return FiniteSet(*self)

class ConditionalFiniteDomain(ConditionalDomain, ProductFiniteDomain):
    """
    A FiniteDomain that has been restricted by a condition

    Example: The possibilities of a die roll under the condition that the
    roll is even.
    """

    def __new__(cls, domain, condition):
        if False:
            i = 10
            return i + 15
        '\n        Create a new instance of ConditionalFiniteDomain class\n        '
        if condition is True:
            return domain
        cond = rv_subs(condition)
        return Basic.__new__(cls, domain, cond)

    def _test(self, elem):
        if False:
            i = 10
            return i + 15
        '\n        Test the value. If value is boolean, return it. If value is equality\n        relational (two objects are equal), return it with left-hand side\n        being equal to right-hand side. Otherwise, raise ValueError exception.\n        '
        val = self.condition.xreplace(dict(elem))
        if val in [True, False]:
            return val
        elif val.is_Equality:
            return val.lhs == val.rhs
        raise ValueError('Undecidable if %s' % str(val))

    def __contains__(self, other):
        if False:
            return 10
        return other in self.fulldomain and self._test(other)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return (elem for elem in self.fulldomain if self._test(elem))

    @property
    def set(self):
        if False:
            while True:
                i = 10
        if isinstance(self.fulldomain, SingleFiniteDomain):
            return FiniteSet(*[elem for elem in self.fulldomain.set if frozenset(((self.fulldomain.symbol, elem),)) in self])
        else:
            raise NotImplementedError('Not implemented on multi-dimensional conditional domain')

    def as_boolean(self):
        if False:
            return 10
        return FiniteDomain.as_boolean(self)

class SingleFiniteDistribution(Distribution, NamedArgsMixin):

    def __new__(cls, *args):
        if False:
            return 10
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        if False:
            while True:
                i = 10
        pass

    @property
    @cacheit
    def dict(self):
        if False:
            return 10
        if self.is_symbolic:
            return Density(self)
        return {k: self.pmf(k) for k in self.set}

    def pmf(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    @property
    def set(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError()
    values = property(lambda self: self.dict.values)
    items = property(lambda self: self.dict.items)
    is_symbolic = property(lambda self: False)
    __iter__ = property(lambda self: self.dict.__iter__)
    __getitem__ = property(lambda self: self.dict.__getitem__)

    def __call__(self, *args):
        if False:
            i = 10
            return i + 15
        return self.pmf(*args)

    def __contains__(self, other):
        if False:
            while True:
                i = 10
        return other in self.set

class FinitePSpace(PSpace):
    """
    A Finite Probability Space

    Represents the probabilities of a finite number of events.
    """
    is_Finite = True

    def __new__(cls, domain, density):
        if False:
            while True:
                i = 10
        density = {sympify(key): sympify(val) for (key, val) in density.items()}
        public_density = Dict(density)
        obj = PSpace.__new__(cls, domain, public_density)
        obj._density = density
        return obj

    def prob_of(self, elem):
        if False:
            while True:
                i = 10
        elem = sympify(elem)
        density = self._density
        if isinstance(list(density.keys())[0], FiniteSet):
            return density.get(elem, S.Zero)
        return density.get(tuple(elem)[0][1], S.Zero)

    def where(self, condition):
        if False:
            i = 10
            return i + 15
        assert all((r.symbol in self.symbols for r in random_symbols(condition)))
        return ConditionalFiniteDomain(self.domain, condition)

    def compute_density(self, expr):
        if False:
            i = 10
            return i + 15
        expr = rv_subs(expr, self.values)
        d = FiniteDensity()
        for elem in self.domain:
            val = expr.xreplace(dict(elem))
            prob = self.prob_of(elem)
            d[val] = d.get(val, S.Zero) + prob
        return d

    @cacheit
    def compute_cdf(self, expr):
        if False:
            while True:
                i = 10
        d = self.compute_density(expr)
        cum_prob = S.Zero
        cdf = []
        for key in sorted(d):
            prob = d[key]
            cum_prob += prob
            cdf.append((key, cum_prob))
        return dict(cdf)

    @cacheit
    def sorted_cdf(self, expr, python_float=False):
        if False:
            for i in range(10):
                print('nop')
        cdf = self.compute_cdf(expr)
        items = list(cdf.items())
        sorted_items = sorted(items, key=lambda val_cumprob: val_cumprob[1])
        if python_float:
            sorted_items = [(v, float(cum_prob)) for (v, cum_prob) in sorted_items]
        return sorted_items

    @cacheit
    def compute_characteristic_function(self, expr):
        if False:
            while True:
                i = 10
        d = self.compute_density(expr)
        t = Dummy('t', real=True)
        return Lambda(t, sum((exp(I * k * t) * v for (k, v) in d.items())))

    @cacheit
    def compute_moment_generating_function(self, expr):
        if False:
            i = 10
            return i + 15
        d = self.compute_density(expr)
        t = Dummy('t', real=True)
        return Lambda(t, sum((exp(k * t) * v for (k, v) in d.items())))

    def compute_expectation(self, expr, rvs=None, **kwargs):
        if False:
            while True:
                i = 10
        rvs = rvs or self.values
        expr = rv_subs(expr, rvs)
        probs = [self.prob_of(elem) for elem in self.domain]
        if isinstance(expr, (Logic, Relational)):
            parse_domain = [tuple(elem)[0][1] for elem in self.domain]
            bools = [expr.xreplace(dict(elem)) for elem in self.domain]
        else:
            parse_domain = [expr.xreplace(dict(elem)) for elem in self.domain]
            bools = [True for elem in self.domain]
        return sum([Piecewise((prob * elem, blv), (S.Zero, True)) for (prob, elem, blv) in zip(probs, parse_domain, bools)])

    def compute_quantile(self, expr):
        if False:
            while True:
                i = 10
        cdf = self.compute_cdf(expr)
        p = Dummy('p', real=True)
        set = ((nan, (p < 0) | (p > 1)),)
        for (key, value) in cdf.items():
            set = set + ((key, p <= value),)
        return Lambda(p, Piecewise(*set))

    def probability(self, condition):
        if False:
            i = 10
            return i + 15
        cond_symbols = frozenset((rs.symbol for rs in random_symbols(condition)))
        cond = rv_subs(condition)
        if not cond_symbols.issubset(self.symbols):
            raise ValueError('Cannot compare foreign random symbols, %s' % str(cond_symbols - self.symbols))
        if isinstance(condition, Relational) and (not cond.free_symbols.issubset(self.domain.free_symbols)):
            rv = condition.lhs if isinstance(condition.rhs, Symbol) else condition.rhs
            return sum((Piecewise((self.prob_of(elem), condition.subs(rv, list(elem)[0][1])), (S.Zero, True)) for elem in self.domain))
        return sympify(sum((self.prob_of(elem) for elem in self.where(condition))))

    def conditional_space(self, condition):
        if False:
            while True:
                i = 10
        domain = self.where(condition)
        prob = self.probability(condition)
        density = {key: val / prob for (key, val) in self._density.items() if domain._test(key)}
        return FinitePSpace(domain, density)

    def sample(self, size=(), library='scipy', seed=None):
        if False:
            return 10
        '\n        Internal sample method\n\n        Returns dictionary mapping RandomSymbol to realization value.\n        '
        return {self.value: self.distribution.sample(size, library, seed)}

class SingleFinitePSpace(SinglePSpace, FinitePSpace):
    """
    A single finite probability space

    Represents the probabilities of a set of random events that can be
    attributed to a single variable/symbol.

    This class is implemented by many of the standard FiniteRV types such as
    Die, Bernoulli, Coin, etc....
    """

    @property
    def domain(self):
        if False:
            while True:
                i = 10
        return SingleFiniteDomain(self.symbol, self.distribution.set)

    @property
    def _is_symbolic(self):
        if False:
            i = 10
            return i + 15
        '\n        Helper property to check if the distribution\n        of the random variable is having symbolic\n        dimension.\n        '
        return self.distribution.is_symbolic

    @property
    def distribution(self):
        if False:
            i = 10
            return i + 15
        return self.args[1]

    def pmf(self, expr):
        if False:
            while True:
                i = 10
        return self.distribution.pmf(expr)

    @property
    @cacheit
    def _density(self):
        if False:
            print('Hello World!')
        return {FiniteSet((self.symbol, val)): prob for (val, prob) in self.distribution.dict.items()}

    @cacheit
    def compute_characteristic_function(self, expr):
        if False:
            i = 10
            return i + 15
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            return Lambda(t, Sum(d(ki) * exp(I * ki * t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_characteristic_function(expr)

    @cacheit
    def compute_moment_generating_function(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            return Lambda(t, Sum(d(ki) * exp(ki * t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_moment_generating_function(expr)

    def compute_quantile(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if self._is_symbolic:
            raise NotImplementedError('Computing quantile for random variables with symbolic dimension because the bounds of searching the required value is undetermined.')
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_quantile(expr)

    def compute_density(self, expr):
        if False:
            while True:
                i = 10
        if self._is_symbolic:
            rv = list(random_symbols(expr))[0]
            k = Dummy('k', integer=True)
            cond = True if not isinstance(expr, (Relational, Logic)) else expr.subs(rv, k)
            return Lambda(k, Piecewise((self.pmf(k), And(k >= self.args[1].low, k <= self.args[1].high, cond)), (S.Zero, True)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_density(expr)

    def compute_cdf(self, expr):
        if False:
            i = 10
            return i + 15
        if self._is_symbolic:
            d = self.compute_density(expr)
            k = Dummy('k')
            ki = Dummy('ki')
            return Lambda(k, Sum(d(ki), (ki, self.args[1].low, k)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_cdf(expr)

    def compute_expectation(self, expr, rvs=None, **kwargs):
        if False:
            while True:
                i = 10
        if self._is_symbolic:
            rv = random_symbols(expr)[0]
            k = Dummy('k', integer=True)
            expr = expr.subs(rv, k)
            cond = True if not isinstance(expr, (Relational, Logic)) else expr
            func = self.pmf(k) * k if cond != True else self.pmf(k) * expr
            return Sum(Piecewise((func, cond), (S.Zero, True)), (k, self.distribution.low, self.distribution.high)).doit()
        expr = _sympify(expr)
        expr = rv_subs(expr, rvs)
        return FinitePSpace(self.domain, self.distribution).compute_expectation(expr, rvs, **kwargs)

    def probability(self, condition):
        if False:
            for i in range(10):
                print('nop')
        if self._is_symbolic:
            raise NotImplementedError('Currently, probability queries are not supported for random variables with symbolic sized distributions.')
        condition = rv_subs(condition)
        return FinitePSpace(self.domain, self.distribution).probability(condition)

    def conditional_space(self, condition):
        if False:
            return 10
        '\n        This method is used for transferring the\n        computation to probability method because\n        conditional space of random variables with\n        symbolic dimensions is currently not possible.\n        '
        if self._is_symbolic:
            self
        domain = self.where(condition)
        prob = self.probability(condition)
        density = {key: val / prob for (key, val) in self._density.items() if domain._test(key)}
        return FinitePSpace(domain, density)

class ProductFinitePSpace(IndependentProductPSpace, FinitePSpace):
    """
    A collection of several independent finite probability spaces
    """

    @property
    def domain(self):
        if False:
            i = 10
            return i + 15
        return ProductFiniteDomain(*[space.domain for space in self.spaces])

    @property
    @cacheit
    def _density(self):
        if False:
            return 10
        proditer = product(*[iter(space._density.items()) for space in self.spaces])
        d = {}
        for items in proditer:
            (elems, probs) = list(zip(*items))
            elem = sumsets(elems)
            prob = Mul(*probs)
            d[elem] = d.get(elem, S.Zero) + prob
        return Dict(d)

    @property
    @cacheit
    def density(self):
        if False:
            return 10
        return Dict(self._density)

    def probability(self, condition):
        if False:
            for i in range(10):
                print('nop')
        return FinitePSpace.probability(self, condition)

    def compute_density(self, expr):
        if False:
            while True:
                i = 10
        return FinitePSpace.compute_density(self, expr)