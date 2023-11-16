import numpy as np
from .common import Benchmark, safe_import
with safe_import():
    from scipy import stats
with safe_import():
    from scipy.stats import sampling
with safe_import():
    from scipy import special

class contdist1:

    def __init__(self):
        if False:
            return 10
        self.mode = 1 / 3

    def pdf(self, x):
        if False:
            print('Hello World!')
        return 12 * x * (1 - x) ** 2

    def dpdf(self, x):
        if False:
            return 10
        return 12 * ((1 - x) ** 2 - 2 * x * (1 - x))

    def cdf(self, x):
        if False:
            print('Hello World!')
        return 12 * (x ** 2 / 2 - x ** 3 / 3 + x ** 4 / 4)

    def support(self):
        if False:
            return 10
        return (0, 1)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'beta(2, 3)'

class contdist2:

    def __init__(self):
        if False:
            return 10
        self.mode = 0

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        return 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x * x)

    def dpdf(self, x):
        if False:
            print('Hello World!')
        return 1.0 / np.sqrt(2 * np.pi) * -x * np.exp(-0.5 * x * x)

    def cdf(self, x):
        if False:
            return 10
        return special.ndtr(x)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'norm(0, 1)'

class contdist3:

    def __init__(self, shift=0.0):
        if False:
            return 10
        self.shift = shift
        self.mode = shift

    def pdf(self, x):
        if False:
            while True:
                i = 10
        x -= self.shift
        y = 1.0 / (abs(x) + 1.0)
        return y * y

    def dpdf(self, x):
        if False:
            return 10
        x -= self.shift
        y = 1.0 / (abs(x) + 1.0)
        y = 2.0 * y * y * y
        return y if x < 0.0 else -y

    def cdf(self, x):
        if False:
            return 10
        x -= self.shift
        if x <= 0.0:
            return 0.5 / (1.0 - x)
        return 1.0 - 0.5 / (1.0 + x)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'sqrtlinshft({self.shift})'

class contdist4:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.mode = 0

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 0.05 + 0.45 * (1 + np.sin(2 * np.pi * x))

    def dpdf(self, x):
        if False:
            while True:
                i = 10
        return 0.2 * 0.45 * (2 * np.pi) * np.cos(2 * np.pi * x)

    def cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 0.05 * (x + 1) + 0.9 * (1.0 + 2.0 * np.pi * (1 + x) - np.cos(2.0 * np.pi * x)) / (4.0 * np.pi)

    def support(self):
        if False:
            for i in range(10):
                print('nop')
        return (-1, 1)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'sin2'

class contdist5:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.mode = 0

    def pdf(self, x):
        if False:
            return 10
        return 0.2 * (0.05 + 0.45 * (1 + np.sin(2 * np.pi * x)))

    def dpdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 0.2 * 0.45 * (2 * np.pi) * np.cos(2 * np.pi * x)

    def cdf(self, x):
        if False:
            i = 10
            return i + 15
        return x / 10.0 + 0.5 + 0.09 / (2 * np.pi) * (np.cos(10 * np.pi) - np.cos(2 * np.pi * x))

    def support(self):
        if False:
            i = 10
            return i + 15
        return (-5, 5)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'sin10'
allcontdists = [contdist1(), contdist2(), contdist3(), contdist3(10000.0), contdist4(), contdist5()]

class TransformedDensityRejection(Benchmark):
    param_names = ['dist', 'c']
    params = [allcontdists, [0.0, -0.5]]

    def setup(self, dist, c):
        if False:
            return 10
        self.urng = np.random.default_rng(333207820760031151694751813924901565029)
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                self.rng = sampling.TransformedDensityRejection(dist, c=c, random_state=self.urng)
            except sampling.UNURANError:
                raise NotImplementedError(f'{dist} not T-concave for c={c}')

    def time_tdr_setup(self, dist, c):
        if False:
            i = 10
            return i + 15
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sampling.TransformedDensityRejection(dist, c=c, random_state=self.urng)

    def time_tdr_rvs(self, dist, c):
        if False:
            return 10
        self.rng.rvs(100000)

class SimpleRatioUniforms(Benchmark):
    param_names = ['dist', 'cdf_at_mode']
    params = [allcontdists, [0, 1]]

    def setup(self, dist, cdf_at_mode):
        if False:
            for i in range(10):
                print('nop')
        self.urng = np.random.default_rng(333207820760031151694751813924901565029)
        try:
            if cdf_at_mode:
                cdf_at_mode = dist.cdf(dist.mode)
            else:
                cdf_at_mode = None
            self.rng = sampling.SimpleRatioUniforms(dist, mode=dist.mode, cdf_at_mode=cdf_at_mode, random_state=self.urng)
        except sampling.UNURANError:
            raise NotImplementedError(f'{dist} not T-concave')

    def time_srou_setup(self, dist, cdf_at_mode):
        if False:
            i = 10
            return i + 15
        if cdf_at_mode:
            cdf_at_mode = dist.cdf(dist.mode)
        else:
            cdf_at_mode = None
        sampling.SimpleRatioUniforms(dist, mode=dist.mode, cdf_at_mode=cdf_at_mode, random_state=self.urng)

    def time_srou_rvs(self, dist, cdf_at_mode):
        if False:
            return 10
        self.rng.rvs(100000)

class NumericalInversePolynomial(Benchmark):
    param_names = ['dist']
    params = [allcontdists]

    def setup(self, dist):
        if False:
            i = 10
            return i + 15
        self.urng = np.random.default_rng(236881457201010196970795409533145269457)
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                self.rng = sampling.NumericalInversePolynomial(dist, random_state=self.urng)
            except sampling.UNURANError:
                raise NotImplementedError(f'setup failed for {dist}')

    def time_pinv_setup(self, dist):
        if False:
            while True:
                i = 10
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sampling.NumericalInversePolynomial(dist, random_state=self.urng)

    def time_pinv_rvs(self, dist):
        if False:
            print('Hello World!')
        self.rng.rvs(100000)

class NumericalInverseHermite(Benchmark):
    param_names = ['dist', 'order']
    params = [allcontdists, [3, 5]]

    def setup(self, dist, order):
        if False:
            for i in range(10):
                print('nop')
        self.urng = np.random.default_rng(236881457201010196970795409533145269457)
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            try:
                self.rng = sampling.NumericalInverseHermite(dist, order=order, random_state=self.urng)
            except sampling.UNURANError:
                raise NotImplementedError(f'setup failed for {dist}')

    def time_hinv_setup(self, dist, order):
        if False:
            return 10
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sampling.NumericalInverseHermite(dist, order=order, random_state=self.urng)

    def time_hinv_rvs(self, dist, order):
        if False:
            return 10
        self.rng.rvs(100000)

class DiscreteAliasUrn(Benchmark):
    param_names = ['distribution']
    params = [[['nhypergeom', (20, 7, 1)], ['hypergeom', (30, 12, 6)], ['nchypergeom_wallenius', (140, 80, 60, 0.5)], ['binom', (5, 0.4)]]]

    def setup(self, distribution):
        if False:
            while True:
                i = 10
        (distname, params) = distribution
        dist = getattr(stats, distname)
        domain = dist.support(*params)
        self.urng = np.random.default_rng(63522142853038270735835366608715032679)
        x = np.arange(domain[0], domain[1] + 1)
        self.pv = dist.pmf(x, *params)
        self.rng = sampling.DiscreteAliasUrn(self.pv, random_state=self.urng)

    def time_dau_setup(self, distribution):
        if False:
            while True:
                i = 10
        sampling.DiscreteAliasUrn(self.pv, random_state=self.urng)

    def time_dau_rvs(self, distribution):
        if False:
            while True:
                i = 10
        self.rng.rvs(100000)

class DiscreteGuideTable(Benchmark):
    param_names = ['distribution']
    params = [[['nhypergeom', (20, 7, 1)], ['hypergeom', (30, 12, 6)], ['nchypergeom_wallenius', (140, 80, 60, 0.5)], ['binom', (5, 0.4)]]]

    def setup(self, distribution):
        if False:
            print('Hello World!')
        (distname, params) = distribution
        dist = getattr(stats, distname)
        domain = dist.support(*params)
        self.urng = np.random.default_rng(63522142853038270735835366608715032679)
        x = np.arange(domain[0], domain[1] + 1)
        self.pv = dist.pmf(x, *params)
        self.rng = sampling.DiscreteGuideTable(self.pv, random_state=self.urng)

    def time_dgt_setup(self, distribution):
        if False:
            print('Hello World!')
        sampling.DiscreteGuideTable(self.pv, random_state=self.urng)

    def time_dgt_rvs(self, distribution):
        if False:
            while True:
                i = 10
        self.rng.rvs(100000)