from functools import singledispatch
from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, CauchyDistribution, ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, GaussianInverseDistribution
from sympy.stats.drv_types import PoissonDistribution, GeometricDistribution, NegativeBinomialDistribution
from sympy.stats.frv_types import BinomialDistribution, BernoulliDistribution
try:
    import pymc
except ImportError:
    pymc = import_module('pymc3')

@singledispatch
def do_sample_pymc(dist):
    if False:
        i = 10
        return i + 15
    return None

@do_sample_pymc.register(BetaDistribution)
def _(dist: BetaDistribution):
    if False:
        for i in range(10):
            print('nop')
    return pymc.Beta('X', alpha=float(dist.alpha), beta=float(dist.beta))

@do_sample_pymc.register(CauchyDistribution)
def _(dist: CauchyDistribution):
    if False:
        for i in range(10):
            print('nop')
    return pymc.Cauchy('X', alpha=float(dist.x0), beta=float(dist.gamma))

@do_sample_pymc.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution):
    if False:
        return 10
    return pymc.ChiSquared('X', nu=float(dist.k))

@do_sample_pymc.register(ExponentialDistribution)
def _(dist: ExponentialDistribution):
    if False:
        while True:
            i = 10
    return pymc.Exponential('X', lam=float(dist.rate))

@do_sample_pymc.register(GammaDistribution)
def _(dist: GammaDistribution):
    if False:
        i = 10
        return i + 15
    return pymc.Gamma('X', alpha=float(dist.k), beta=1 / float(dist.theta))

@do_sample_pymc.register(LogNormalDistribution)
def _(dist: LogNormalDistribution):
    if False:
        print('Hello World!')
    return pymc.Lognormal('X', mu=float(dist.mean), sigma=float(dist.std))

@do_sample_pymc.register(NormalDistribution)
def _(dist: NormalDistribution):
    if False:
        while True:
            i = 10
    return pymc.Normal('X', float(dist.mean), float(dist.std))

@do_sample_pymc.register(GaussianInverseDistribution)
def _(dist: GaussianInverseDistribution):
    if False:
        i = 10
        return i + 15
    return pymc.Wald('X', mu=float(dist.mean), lam=float(dist.shape))

@do_sample_pymc.register(ParetoDistribution)
def _(dist: ParetoDistribution):
    if False:
        for i in range(10):
            print('nop')
    return pymc.Pareto('X', alpha=float(dist.alpha), m=float(dist.xm))

@do_sample_pymc.register(UniformDistribution)
def _(dist: UniformDistribution):
    if False:
        return 10
    return pymc.Uniform('X', lower=float(dist.left), upper=float(dist.right))

@do_sample_pymc.register(GeometricDistribution)
def _(dist: GeometricDistribution):
    if False:
        return 10
    return pymc.Geometric('X', p=float(dist.p))

@do_sample_pymc.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution):
    if False:
        while True:
            i = 10
    return pymc.NegativeBinomial('X', mu=float(dist.p * dist.r / (1 - dist.p)), alpha=float(dist.r))

@do_sample_pymc.register(PoissonDistribution)
def _(dist: PoissonDistribution):
    if False:
        return 10
    return pymc.Poisson('X', mu=float(dist.lamda))

@do_sample_pymc.register(BernoulliDistribution)
def _(dist: BernoulliDistribution):
    if False:
        while True:
            i = 10
    return pymc.Bernoulli('X', p=float(dist.p))

@do_sample_pymc.register(BinomialDistribution)
def _(dist: BinomialDistribution):
    if False:
        for i in range(10):
            print('nop')
    return pymc.Binomial('X', n=int(dist.n), p=float(dist.p))