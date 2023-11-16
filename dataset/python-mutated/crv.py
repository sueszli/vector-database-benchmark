"""
Continuous Random Variables Module

See Also
========
sympy.stats.crv_types
sympy.stats.rv
sympy.stats.frv
"""
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda, PoleError
from sympy.core.numbers import I, nan, oo
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, symbols
from sympy.core.sympify import _sympify, sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import Integral, integrate
from sympy.logic.boolalg import And, Or
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import poly
from sympy.series.series import series
from sympy.sets.sets import FiniteSet, Intersection, Interval, Union
from sympy.solvers.solveset import solveset
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats.rv import RandomDomain, SingleDomain, ConditionalDomain, is_random, ProductDomain, PSpace, SinglePSpace, random_symbols, NamedArgsMixin, Distribution

class ContinuousDomain(RandomDomain):
    """
    A domain with continuous support

    Represented using symbols and Intervals.
    """
    is_Continuous = True

    def as_boolean(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Not Implemented for generic Domains')

class SingleContinuousDomain(ContinuousDomain, SingleDomain):
    """
    A univariate domain with continuous support

    Represented using a single symbol and interval.
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if variables is None:
            variables = self.symbols
        if not variables:
            return expr
        if frozenset(variables) != frozenset(self.symbols):
            raise ValueError('Values should be equal')
        return Integral(expr, (self.symbol, self.set), **kwargs)

    def as_boolean(self):
        if False:
            while True:
                i = 10
        return self.set.as_relational(self.symbol)

class ProductContinuousDomain(ProductDomain, ContinuousDomain):
    """
    A collection of independent domains with continuous support
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if False:
            return 10
        if variables is None:
            variables = self.symbols
        for domain in self.domains:
            domain_vars = frozenset(variables) & frozenset(domain.symbols)
            if domain_vars:
                expr = domain.compute_expectation(expr, domain_vars, **kwargs)
        return expr

    def as_boolean(self):
        if False:
            while True:
                i = 10
        return And(*[domain.as_boolean() for domain in self.domains])

class ConditionalContinuousDomain(ContinuousDomain, ConditionalDomain):
    """
    A domain with continuous support that has been further restricted by a
    condition such as $x > 3$.
    """

    def compute_expectation(self, expr, variables=None, **kwargs):
        if False:
            return 10
        if variables is None:
            variables = self.symbols
        if not variables:
            return expr
        fullintgrl = self.fulldomain.compute_expectation(expr, variables)
        (integrand, limits) = (fullintgrl.function, list(fullintgrl.limits))
        conditions = [self.condition]
        while conditions:
            cond = conditions.pop()
            if cond.is_Boolean:
                if isinstance(cond, And):
                    conditions.extend(cond.args)
                elif isinstance(cond, Or):
                    raise NotImplementedError('Or not implemented here')
            elif cond.is_Relational:
                if cond.is_Equality:
                    integrand *= DiracDelta(cond.lhs - cond.rhs)
                else:
                    symbols = cond.free_symbols & set(self.symbols)
                    if len(symbols) != 1:
                        raise NotImplementedError('Multivariate Inequalities not yet implemented')
                    symbol = symbols.pop()
                    for (i, limit) in enumerate(limits):
                        if limit[0] == symbol:
                            cintvl = reduce_rational_inequalities_wrap(cond, symbol)
                            lintvl = Interval(limit[1], limit[2])
                            intvl = cintvl.intersect(lintvl)
                            limits[i] = (symbol, intvl.left, intvl.right)
            else:
                raise TypeError('Condition %s is not a relational or Boolean' % cond)
        return Integral(integrand, *limits, **kwargs)

    def as_boolean(self):
        if False:
            return 10
        return And(self.fulldomain.as_boolean(), self.condition)

    @property
    def set(self):
        if False:
            return 10
        if len(self.symbols) == 1:
            return self.fulldomain.set & reduce_rational_inequalities_wrap(self.condition, tuple(self.symbols)[0])
        else:
            raise NotImplementedError('Set of Conditional Domain not Implemented')

class ContinuousDistribution(Distribution):

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.pdf(*args)

class SingleContinuousDistribution(ContinuousDistribution, NamedArgsMixin):
    """ Continuous distribution of a single variable.

    Explanation
    ===========

    Serves as superclass for Normal/Exponential/UniformDistribution etc....

    Represented by parameters for each of the specific classes.  E.g
    NormalDistribution is represented by a mean and standard deviation.

    Provides methods for pdf, cdf, and sampling.

    See Also
    ========

    sympy.stats.crv_types.*
    """
    set = Interval(-oo, oo)

    def __new__(cls, *args):
        if False:
            i = 10
            return i + 15
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        if False:
            for i in range(10):
                print('nop')
        pass

    @cacheit
    def compute_cdf(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Compute the CDF from the PDF.\n\n        Returns a Lambda.\n        '
        (x, z) = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.set.start
        pdf = self.pdf(x)
        cdf = integrate(pdf.doit(), (x, left_bound, z), **kwargs)
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    def _cdf(self, x):
        if False:
            i = 10
            return i + 15
        return None

    def cdf(self, x, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Cumulative density function '
        if len(kwargs) == 0:
            cdf = self._cdf(x)
            if cdf is not None:
                return cdf
        return self.compute_cdf(**kwargs)(x)

    @cacheit
    def compute_characteristic_function(self, **kwargs):
        if False:
            return 10
        ' Compute the characteristic function from the PDF.\n\n        Returns a Lambda.\n        '
        (x, t) = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        cf = integrate(exp(I * t * x) * pdf, (x, self.set))
        return Lambda(t, cf)

    def _characteristic_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        return None

    def characteristic_function(self, t, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Characteristic function '
        if len(kwargs) == 0:
            cf = self._characteristic_function(t)
            if cf is not None:
                return cf
        return self.compute_characteristic_function(**kwargs)(t)

    @cacheit
    def compute_moment_generating_function(self, **kwargs):
        if False:
            while True:
                i = 10
        ' Compute the moment generating function from the PDF.\n\n        Returns a Lambda.\n        '
        (x, t) = symbols('x, t', real=True, cls=Dummy)
        pdf = self.pdf(x)
        mgf = integrate(exp(t * x) * pdf, (x, self.set))
        return Lambda(t, mgf)

    def _moment_generating_function(self, t):
        if False:
            for i in range(10):
                print('nop')
        return None

    def moment_generating_function(self, t, **kwargs):
        if False:
            return 10
        ' Moment generating function '
        if not kwargs:
            mgf = self._moment_generating_function(t)
            if mgf is not None:
                return mgf
        return self.compute_moment_generating_function(**kwargs)(t)

    def expectation(self, expr, var, evaluate=True, **kwargs):
        if False:
            return 10
        ' Expectation of expression over distribution '
        if evaluate:
            try:
                p = poly(expr, var)
                if p.is_zero:
                    return S.Zero
                t = Dummy('t', real=True)
                mgf = self._moment_generating_function(t)
                if mgf is None:
                    return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
                deg = p.degree()
                taylor = poly(series(mgf, t, 0, deg + 1).removeO(), t)
                result = 0
                for k in range(deg + 1):
                    result += p.coeff_monomial(var ** k) * taylor.coeff_monomial(t ** k) * factorial(k)
                return result
            except PolynomialError:
                return integrate(expr * self.pdf(var), (var, self.set), **kwargs)
        else:
            return Integral(expr * self.pdf(var), (var, self.set), **kwargs)

    @cacheit
    def compute_quantile(self, **kwargs):
        if False:
            return 10
        ' Compute the Quantile from the PDF.\n\n        Returns a Lambda.\n        '
        (x, p) = symbols('x, p', real=True, cls=Dummy)
        left_bound = self.set.start
        pdf = self.pdf(x)
        cdf = integrate(pdf, (x, left_bound, x), **kwargs)
        quantile = solveset(cdf - p, x, self.set)
        return Lambda(p, Piecewise((quantile, (p >= 0) & (p <= 1)), (nan, True)))

    def _quantile(self, x):
        if False:
            while True:
                i = 10
        return None

    def quantile(self, x, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Cumulative density function '
        if len(kwargs) == 0:
            quantile = self._quantile(x)
            if quantile is not None:
                return quantile
        return self.compute_quantile(**kwargs)(x)

class ContinuousPSpace(PSpace):
    """ Continuous Probability Space

    Represents the likelihood of an event space defined over a continuum.

    Represented with a ContinuousDomain and a PDF (Lambda-Like)
    """
    is_Continuous = True
    is_real = True

    @property
    def pdf(self):
        if False:
            while True:
                i = 10
        return self.density(*self.domain.symbols)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        if False:
            while True:
                i = 10
        if rvs is None:
            rvs = self.values
        else:
            rvs = frozenset(rvs)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})
        domain_symbols = frozenset((rv.symbol for rv in rvs))
        return self.domain.compute_expectation(self.pdf * expr, domain_symbols, **kwargs)

    def compute_density(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if expr in self.values:
            randomsymbols = tuple(set(self.values) - frozenset([expr]))
            symbols = tuple((rs.symbol for rs in randomsymbols))
            pdf = self.domain.compute_expectation(self.pdf, symbols, **kwargs)
            return Lambda(expr.symbol, pdf)
        z = Dummy('z', real=True)
        return Lambda(z, self.compute_expectation(DiracDelta(expr - z), **kwargs))

    @cacheit
    def compute_cdf(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not self.domain.set.is_Interval:
            raise ValueError('CDF not well defined on multivariate expressions')
        d = self.compute_density(expr, **kwargs)
        (x, z) = symbols('x, z', real=True, cls=Dummy)
        left_bound = self.domain.set.start
        cdf = integrate(d(x), (x, left_bound, z), **kwargs)
        cdf = Piecewise((cdf, z >= left_bound), (0, True))
        return Lambda(z, cdf)

    @cacheit
    def compute_characteristic_function(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not self.domain.set.is_Interval:
            raise NotImplementedError('Characteristic function of multivariate expressions not implemented')
        d = self.compute_density(expr, **kwargs)
        (x, t) = symbols('x, t', real=True, cls=Dummy)
        cf = integrate(exp(I * t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, cf)

    @cacheit
    def compute_moment_generating_function(self, expr, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.domain.set.is_Interval:
            raise NotImplementedError('Moment generating function of multivariate expressions not implemented')
        d = self.compute_density(expr, **kwargs)
        (x, t) = symbols('x, t', real=True, cls=Dummy)
        mgf = integrate(exp(t * x) * d(x), (x, -oo, oo), **kwargs)
        return Lambda(t, mgf)

    @cacheit
    def compute_quantile(self, expr, **kwargs):
        if False:
            return 10
        if not self.domain.set.is_Interval:
            raise ValueError('Quantile not well defined on multivariate expressions')
        d = self.compute_cdf(expr, **kwargs)
        x = Dummy('x', real=True)
        p = Dummy('p', positive=True)
        quantile = solveset(d(x) - p, x, self.set)
        return Lambda(p, quantile)

    def probability(self, condition, **kwargs):
        if False:
            print('Hello World!')
        z = Dummy('z', real=True)
        cond_inv = False
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        try:
            domain = self.where(condition)
            rv = [rv for rv in self.values if rv.symbol == domain.symbol][0]
            pdf = self.compute_density(rv, **kwargs)
            if domain.set is S.EmptySet or isinstance(domain.set, FiniteSet):
                return S.Zero if not cond_inv else S.One
            if isinstance(domain.set, Union):
                return sum((Integral(pdf(z), (z, subset), **kwargs) for subset in domain.set.args if isinstance(subset, Interval)))
            return Integral(pdf(z), (z, domain.set), **kwargs)
        except NotImplementedError:
            from sympy.stats.rv import density
            expr = condition.lhs - condition.rhs
            if not is_random(expr):
                dens = self.density
                comp = condition.rhs
            else:
                dens = density(expr, **kwargs)
                comp = 0
            if not isinstance(dens, ContinuousDistribution):
                from sympy.stats.crv_types import ContinuousDistributionHandmade
                dens = ContinuousDistributionHandmade(dens, set=self.domain.set)
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, comp))
            return result if not cond_inv else S.One - result

    def where(self, condition):
        if False:
            for i in range(10):
                print('nop')
        rvs = frozenset(random_symbols(condition))
        if not (len(rvs) == 1 and rvs.issubset(self.values)):
            raise NotImplementedError('Multiple continuous random variables not supported')
        rv = tuple(rvs)[0]
        interval = reduce_rational_inequalities_wrap(condition, rv)
        interval = interval.intersect(self.domain.set)
        return SingleContinuousDomain(rv.symbol, interval)

    def conditional_space(self, condition, normalize=True, **kwargs):
        if False:
            while True:
                i = 10
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        domain = ConditionalContinuousDomain(self.domain, condition)
        if normalize:
            replacement = {rv: Dummy(str(rv)) for rv in self.symbols}
            norm = domain.compute_expectation(self.pdf, **kwargs)
            pdf = self.pdf / norm.xreplace(replacement)
            density = Lambda(tuple(domain.symbols), pdf)
        return ContinuousPSpace(domain, density)

class SingleContinuousPSpace(ContinuousPSpace, SinglePSpace):
    """
    A continuous probability space over a single univariate variable.

    These consist of a Symbol and a SingleContinuousDistribution

    This class is normally accessed through the various random variable
    functions, Normal, Exponential, Uniform, etc....
    """

    @property
    def set(self):
        if False:
            print('Hello World!')
        return self.distribution.set

    @property
    def domain(self):
        if False:
            return 10
        return SingleContinuousDomain(sympify(self.symbol), self.set)

    def sample(self, size=(), library='scipy', seed=None):
        if False:
            i = 10
            return i + 15
        '\n        Internal sample method.\n\n        Returns dictionary mapping RandomSymbol to realization value.\n        '
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        if False:
            i = 10
            return i + 15
        rvs = rvs or (self.value,)
        if self.value not in rvs:
            return expr
        expr = _sympify(expr)
        expr = expr.xreplace({rv: rv.symbol for rv in rvs})
        x = self.value.symbol
        try:
            return self.distribution.expectation(expr, x, evaluate=evaluate, **kwargs)
        except PoleError:
            return Integral(expr * self.pdf, (x, self.set), **kwargs)

    def compute_cdf(self, expr, **kwargs):
        if False:
            while True:
                i = 10
        if expr == self.value:
            z = Dummy('z', real=True)
            return Lambda(z, self.distribution.cdf(z, **kwargs))
        else:
            return ContinuousPSpace.compute_cdf(self, expr, **kwargs)

    def compute_characteristic_function(self, expr, **kwargs):
        if False:
            while True:
                i = 10
        if expr == self.value:
            t = Dummy('t', real=True)
            return Lambda(t, self.distribution.characteristic_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_characteristic_function(self, expr, **kwargs)

    def compute_moment_generating_function(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if expr == self.value:
            t = Dummy('t', real=True)
            return Lambda(t, self.distribution.moment_generating_function(t, **kwargs))
        else:
            return ContinuousPSpace.compute_moment_generating_function(self, expr, **kwargs)

    def compute_density(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if expr == self.value:
            return self.density
        y = Dummy('y', real=True)
        gs = solveset(expr - y, self.value, S.Reals)
        if isinstance(gs, Intersection) and S.Reals in gs.args:
            gs = list(gs.args[1])
        if not gs:
            raise ValueError('Can not solve %s for %s' % (expr, self.value))
        fx = self.compute_density(self.value)
        fy = sum((fx(g) * abs(g.diff(y)) for g in gs))
        return Lambda(y, fy)

    def compute_quantile(self, expr, **kwargs):
        if False:
            while True:
                i = 10
        if expr == self.value:
            p = Dummy('p', real=True)
            return Lambda(p, self.distribution.quantile(p, **kwargs))
        else:
            return ContinuousPSpace.compute_quantile(self, expr, **kwargs)

def _reduce_inequalities(conditions, var, **kwargs):
    if False:
        print('Hello World!')
    try:
        return reduce_rational_inequalities(conditions, var, **kwargs)
    except PolynomialError:
        raise ValueError('Reduction of condition failed %s\n' % conditions[0])

def reduce_rational_inequalities_wrap(condition, var):
    if False:
        for i in range(10):
            print('nop')
    if condition.is_Relational:
        return _reduce_inequalities([[condition]], var, relational=False)
    if isinstance(condition, Or):
        return Union(*[_reduce_inequalities([[arg]], var, relational=False) for arg in condition.args])
    if isinstance(condition, And):
        intervals = [_reduce_inequalities([[arg]], var, relational=False) for arg in condition.args]
        I = intervals[0]
        for i in intervals:
            I = I.intersect(i)
        return I