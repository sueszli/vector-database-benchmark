from sympy.sets import FiniteSet
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import FallingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.integrals.integrals import Integral
from sympy.solvers.solveset import solveset
from .rv import probability, expectation, density, where, given, pspace, cdf, PSpace, characteristic_function, sample, sample_iter, random_symbols, independent, dependent, sampling_density, moment_generating_function, quantile, is_random, sample_stochastic_process
__all__ = ['P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf', 'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std', 'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'median', 'independent', 'random_symbols', 'correlation', 'factorial_moment', 'moment', 'cmoment', 'sampling_density', 'moment_generating_function', 'smoment', 'quantile', 'sample_stochastic_process']

def moment(X, n, c=0, condition=None, *, evaluate=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the nth moment of a random expression about c.\n\n    .. math::\n        moment(X, c, n) = E((X-c)^{n})\n\n    Default value of c is 0.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Die, moment, E\n    >>> X = Die('X', 6)\n    >>> moment(X, 1, 6)\n    -5/2\n    >>> moment(X, 2)\n    91/6\n    >>> moment(X, 1) == E(X)\n    True\n    "
    from sympy.stats.symbolic_probability import Moment
    if evaluate:
        return Moment(X, n, c, condition).doit()
    return Moment(X, n, c, condition).rewrite(Integral)

def variance(X, condition=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Variance of a random expression.\n\n    .. math::\n        variance(X) = E((X-E(X))^{2})\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Die, Bernoulli, variance\n    >>> from sympy import simplify, Symbol\n\n    >>> X = Die('X', 6)\n    >>> p = Symbol('p')\n    >>> B = Bernoulli('B', p, 1, 0)\n\n    >>> variance(2*X)\n    35/3\n\n    >>> simplify(variance(B))\n    p*(1 - p)\n    "
    if is_random(X) and pspace(X) == PSpace():
        from sympy.stats.symbolic_probability import Variance
        return Variance(X, condition)
    return cmoment(X, 2, condition, **kwargs)

def standard_deviation(X, condition=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Standard Deviation of a random expression\n\n    .. math::\n        std(X) = \\sqrt(E((X-E(X))^{2}))\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Bernoulli, std\n    >>> from sympy import Symbol, simplify\n\n    >>> p = Symbol('p')\n    >>> B = Bernoulli('B', p, 1, 0)\n\n    >>> simplify(std(B))\n    sqrt(p*(1 - p))\n    "
    return sqrt(variance(X, condition, **kwargs))
std = standard_deviation

def entropy(expr, condition=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Calculuates entropy of a probability distribution.\n\n    Parameters\n    ==========\n\n    expression : the random expression whose entropy is to be calculated\n    condition : optional, to specify conditions on random expression\n    b: base of the logarithm, optional\n       By default, it is taken as Euler's number\n\n    Returns\n    =======\n\n    result : Entropy of the expression, a constant\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Normal, Die, entropy\n    >>> X = Normal('X', 0, 1)\n    >>> entropy(X)\n    log(2)/2 + 1/2 + log(pi)/2\n\n    >>> D = Die('D', 4)\n    >>> entropy(D)\n    log(4)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Entropy_%28information_theory%29\n    .. [2] https://www.crmarsh.com/static/pdf/Charles_Marsh_Continuous_Entropy.pdf\n    .. [3] https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf\n    "
    pdf = density(expr, condition, **kwargs)
    base = kwargs.get('b', exp(1))
    if isinstance(pdf, dict):
        return sum([-prob * log(prob, base) for prob in pdf.values()])
    return expectation(-log(pdf(expr), base))

def covariance(X, Y, condition=None, **kwargs):
    if False:
        return 10
    "\n    Covariance of two random expressions.\n\n    Explanation\n    ===========\n\n    The expectation that the two variables will rise and fall together\n\n    .. math::\n        covariance(X,Y) = E((X-E(X)) (Y-E(Y)))\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Exponential, covariance\n    >>> from sympy import Symbol\n\n    >>> rate = Symbol('lambda', positive=True, real=True)\n    >>> X = Exponential('X', rate)\n    >>> Y = Exponential('Y', rate)\n\n    >>> covariance(X, X)\n    lambda**(-2)\n    >>> covariance(X, Y)\n    0\n    >>> covariance(X, Y + rate*X)\n    1/lambda\n    "
    if is_random(X) and pspace(X) == PSpace() or (is_random(Y) and pspace(Y) == PSpace()):
        from sympy.stats.symbolic_probability import Covariance
        return Covariance(X, Y, condition)
    return expectation((X - expectation(X, condition, **kwargs)) * (Y - expectation(Y, condition, **kwargs)), condition, **kwargs)

def correlation(X, Y, condition=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Correlation of two random expressions, also known as correlation\n    coefficient or Pearson's correlation.\n\n    Explanation\n    ===========\n\n    The normalized expectation that the two variables will rise\n    and fall together\n\n    .. math::\n        correlation(X,Y) = E((X-E(X))(Y-E(Y)) / (\\sigma_x  \\sigma_y))\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Exponential, correlation\n    >>> from sympy import Symbol\n\n    >>> rate = Symbol('lambda', positive=True, real=True)\n    >>> X = Exponential('X', rate)\n    >>> Y = Exponential('Y', rate)\n\n    >>> correlation(X, X)\n    1\n    >>> correlation(X, Y)\n    0\n    >>> correlation(X, Y + rate*X)\n    1/sqrt(1 + lambda**(-2))\n    "
    return covariance(X, Y, condition, **kwargs) / (std(X, condition, **kwargs) * std(Y, condition, **kwargs))

def cmoment(X, n, condition=None, *, evaluate=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the nth central moment of a random expression about its mean.\n\n    .. math::\n        cmoment(X, n) = E((X - E(X))^{n})\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Die, cmoment, variance\n    >>> X = Die('X', 6)\n    >>> cmoment(X, 3)\n    0\n    >>> cmoment(X, 2)\n    35/12\n    >>> cmoment(X, 2) == variance(X)\n    True\n    "
    from sympy.stats.symbolic_probability import CentralMoment
    if evaluate:
        return CentralMoment(X, n, condition).doit()
    return CentralMoment(X, n, condition).rewrite(Integral)

def smoment(X, n, condition=None, **kwargs):
    if False:
        return 10
    "\n    Return the nth Standardized moment of a random expression.\n\n    .. math::\n        smoment(X, n) = E(((X - \\mu)/\\sigma_X)^{n})\n\n    Examples\n    ========\n\n    >>> from sympy.stats import skewness, Exponential, smoment\n    >>> from sympy import Symbol\n    >>> rate = Symbol('lambda', positive=True, real=True)\n    >>> Y = Exponential('Y', rate)\n    >>> smoment(Y, 4)\n    9\n    >>> smoment(Y, 4) == smoment(3*Y, 4)\n    True\n    >>> smoment(Y, 3) == skewness(Y)\n    True\n    "
    sigma = std(X, condition, **kwargs)
    return (1 / sigma) ** n * cmoment(X, n, condition, **kwargs)

def skewness(X, condition=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Measure of the asymmetry of the probability distribution.\n\n    Explanation\n    ===========\n\n    Positive skew indicates that most of the values lie to the right of\n    the mean.\n\n    .. math::\n        skewness(X) = E(((X - E(X))/\\sigma_X)^{3})\n\n    Parameters\n    ==========\n\n    condition : Expr containing RandomSymbols\n            A conditional expression. skewness(X, X>0) is skewness of X given X > 0\n\n    Examples\n    ========\n\n    >>> from sympy.stats import skewness, Exponential, Normal\n    >>> from sympy import Symbol\n    >>> X = Normal('X', 0, 1)\n    >>> skewness(X)\n    0\n    >>> skewness(X, X > 0) # find skewness given X > 0\n    (-sqrt(2)/sqrt(pi) + 4*sqrt(2)/pi**(3/2))/(1 - 2/pi)**(3/2)\n\n    >>> rate = Symbol('lambda', positive=True, real=True)\n    >>> Y = Exponential('Y', rate)\n    >>> skewness(Y)\n    2\n    "
    return smoment(X, 3, condition=condition, **kwargs)

def kurtosis(X, condition=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Characterizes the tails/outliers of a probability distribution.\n\n    Explanation\n    ===========\n\n    Kurtosis of any univariate normal distribution is 3. Kurtosis less than\n    3 means that the distribution produces fewer and less extreme outliers\n    than the normal distribution.\n\n    .. math::\n        kurtosis(X) = E(((X - E(X))/\\sigma_X)^{4})\n\n    Parameters\n    ==========\n\n    condition : Expr containing RandomSymbols\n            A conditional expression. kurtosis(X, X>0) is kurtosis of X given X > 0\n\n    Examples\n    ========\n\n    >>> from sympy.stats import kurtosis, Exponential, Normal\n    >>> from sympy import Symbol\n    >>> X = Normal('X', 0, 1)\n    >>> kurtosis(X)\n    3\n    >>> kurtosis(X, X > 0) # find kurtosis given X > 0\n    (-4/pi - 12/pi**2 + 3)/(1 - 2/pi)**2\n\n    >>> rate = Symbol('lamda', positive=True, real=True)\n    >>> Y = Exponential('Y', rate)\n    >>> kurtosis(Y)\n    9\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Kurtosis\n    .. [2] https://mathworld.wolfram.com/Kurtosis.html\n    "
    return smoment(X, 4, condition=condition, **kwargs)

def factorial_moment(X, n, condition=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    The factorial moment is a mathematical quantity defined as the expectation\n    or average of the falling factorial of a random variable.\n\n    .. math::\n        factorial-moment(X, n) = E(X(X - 1)(X - 2)...(X - n + 1))\n\n    Parameters\n    ==========\n\n    n: A natural number, n-th factorial moment.\n\n    condition : Expr containing RandomSymbols\n            A conditional expression.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import factorial_moment, Poisson, Binomial\n    >>> from sympy import Symbol, S\n    >>> lamda = Symbol('lamda')\n    >>> X = Poisson('X', lamda)\n    >>> factorial_moment(X, 2)\n    lamda**2\n    >>> Y = Binomial('Y', 2, S.Half)\n    >>> factorial_moment(Y, 2)\n    1/2\n    >>> factorial_moment(Y, 2, Y > 1) # find factorial moment for Y > 1\n    2\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Factorial_moment\n    .. [2] https://mathworld.wolfram.com/FactorialMoment.html\n    "
    return expectation(FallingFactorial(X, n), condition=condition, **kwargs)

def median(X, evaluate=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Calculuates the median of the probability distribution.\n\n    Explanation\n    ===========\n\n    Mathematically, median of Probability distribution is defined as all those\n    values of `m` for which the following condition is satisfied\n\n    .. math::\n        P(X\\leq m) \\geq  \\frac{1}{2} \\text{ and} \\text{ } P(X\\geq m)\\geq \\frac{1}{2}\n\n    Parameters\n    ==========\n\n    X: The random expression whose median is to be calculated.\n\n    Returns\n    =======\n\n    The FiniteSet or an Interval which contains the median of the\n    random expression.\n\n    Examples\n    ========\n\n    >>> from sympy.stats import Normal, Die, median\n    >>> N = Normal('N', 3, 1)\n    >>> median(N)\n    {3}\n    >>> D = Die('D')\n    >>> median(D)\n    {3, 4}\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Median#Probability_distributions\n\n    "
    if not is_random(X):
        return X
    from sympy.stats.crv import ContinuousPSpace
    from sympy.stats.drv import DiscretePSpace
    from sympy.stats.frv import FinitePSpace
    if isinstance(pspace(X), FinitePSpace):
        cdf = pspace(X).compute_cdf(X)
        result = []
        for (key, value) in cdf.items():
            if value >= Rational(1, 2) and 1 - value + pspace(X).probability(Eq(X, key)) >= Rational(1, 2):
                result.append(key)
        return FiniteSet(*result)
    if isinstance(pspace(X), (ContinuousPSpace, DiscretePSpace)):
        cdf = pspace(X).compute_cdf(X)
        x = Dummy('x')
        result = solveset(piecewise_fold(cdf(x) - Rational(1, 2)), x, pspace(X).set)
        return result
    raise NotImplementedError('The median of %s is not implemented.' % str(pspace(X)))

def coskewness(X, Y, Z, condition=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculates the co-skewness of three random variables.\n\n    Explanation\n    ===========\n\n    Mathematically Coskewness is defined as\n\n    .. math::\n        coskewness(X,Y,Z)=\\frac{E[(X-E[X]) * (Y-E[Y]) * (Z-E[Z])]} {\\sigma_{X}\\sigma_{Y}\\sigma_{Z}}\n\n    Parameters\n    ==========\n\n    X : RandomSymbol\n            Random Variable used to calculate coskewness\n    Y : RandomSymbol\n            Random Variable used to calculate coskewness\n    Z : RandomSymbol\n            Random Variable used to calculate coskewness\n    condition : Expr containing RandomSymbols\n            A conditional expression\n\n    Examples\n    ========\n\n    >>> from sympy.stats import coskewness, Exponential, skewness\n    >>> from sympy import symbols\n    >>> p = symbols('p', positive=True)\n    >>> X = Exponential('X', p)\n    >>> Y = Exponential('Y', 2*p)\n    >>> coskewness(X, Y, Y)\n    0\n    >>> coskewness(X, Y + X, Y + 2*X)\n    16*sqrt(85)/85\n    >>> coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3)\n    9*sqrt(170)/85\n    >>> coskewness(Y, Y, Y) == skewness(Y)\n    True\n    >>> coskewness(X, Y + p*X, Y + 2*p*X)\n    4/(sqrt(1 + 1/(4*p**2))*sqrt(4 + 1/(4*p**2)))\n\n    Returns\n    =======\n\n    coskewness : The coskewness of the three random variables\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Coskewness\n\n    "
    num = expectation((X - expectation(X, condition, **kwargs)) * (Y - expectation(Y, condition, **kwargs)) * (Z - expectation(Z, condition, **kwargs)), condition, **kwargs)
    den = std(X, condition, **kwargs) * std(Y, condition, **kwargs) * std(Z, condition, **kwargs)
    return num / den
P = probability
E = expectation
H = entropy