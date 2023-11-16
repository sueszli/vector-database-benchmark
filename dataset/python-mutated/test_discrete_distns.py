import pytest
import itertools
from scipy.stats import betabinom, hypergeom, nhypergeom, bernoulli, boltzmann, skellam, zipf, zipfian, binom, nbinom, nchypergeom_fisher, nchypergeom_wallenius, randint
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose, suppress_warnings
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad

@pytest.mark.parametrize('k, M, n, N, expected, rtol', [(3, 10, 4, 5, 0.9761904761904762, 1e-15), (107, 10000, 3000, 215, 0.9999999997226765, 1e-15), (10, 10000, 3000, 215, 2.681682217692179e-21, 5e-11)])
def test_hypergeom_cdf(k, M, n, N, expected, rtol):
    if False:
        while True:
            i = 10
    p = hypergeom.cdf(k, M, n, N)
    assert_allclose(p, expected, rtol=rtol)

@pytest.mark.parametrize('k, M, n, N, expected, rtol', [(25, 10000, 3000, 215, 0.9999999999052958, 1e-15), (125, 10000, 3000, 215, 1.4416781705752128e-18, 5e-11)])
def test_hypergeom_sf(k, M, n, N, expected, rtol):
    if False:
        while True:
            i = 10
    p = hypergeom.sf(k, M, n, N)
    assert_allclose(p, expected, rtol=rtol)

def test_hypergeom_logpmf():
    if False:
        print('Hello World!')
    k = 5
    N = 50
    K = 10
    n = 5
    logpmf1 = hypergeom.logpmf(k, N, K, n)
    logpmf2 = hypergeom.logpmf(n - k, N, N - K, n)
    logpmf3 = hypergeom.logpmf(K - k, N, K, N - n)
    logpmf4 = hypergeom.logpmf(k, N, n, K)
    assert_almost_equal(logpmf1, logpmf2, decimal=12)
    assert_almost_equal(logpmf1, logpmf3, decimal=12)
    assert_almost_equal(logpmf1, logpmf4, decimal=12)
    k = 1
    N = 10
    K = 7
    n = 1
    hypergeom_logpmf = hypergeom.logpmf(k, N, K, n)
    bernoulli_logpmf = bernoulli.logpmf(k, K / N)
    assert_almost_equal(hypergeom_logpmf, bernoulli_logpmf, decimal=12)

def test_nhypergeom_pmf():
    if False:
        for i in range(10):
            print('nop')
    (M, n, r) = (45, 13, 8)
    k = 6
    NHG = nhypergeom.pmf(k, M, n, r)
    HG = hypergeom.pmf(k, M, n, k + r - 1) * (M - n - (r - 1)) / (M - (k + r - 1))
    assert_allclose(HG, NHG, rtol=1e-10)

def test_nhypergeom_pmfcdf():
    if False:
        while True:
            i = 10
    M = 8
    n = 3
    r = 4
    support = np.arange(n + 1)
    pmf = nhypergeom.pmf(support, M, n, r)
    cdf = nhypergeom.cdf(support, M, n, r)
    assert_allclose(pmf, [1 / 14, 3 / 14, 5 / 14, 5 / 14], rtol=1e-13)
    assert_allclose(cdf, [1 / 14, 4 / 14, 9 / 14, 1.0], rtol=1e-13)

def test_nhypergeom_r0():
    if False:
        return 10
    M = 10
    n = 3
    r = 0
    pmf = nhypergeom.pmf([[0, 1, 2, 0], [1, 2, 0, 3]], M, n, r)
    assert_allclose(pmf, [[1, 0, 0, 1], [0, 0, 1, 0]], rtol=1e-13)

def test_nhypergeom_rvs_shape():
    if False:
        print('Hello World!')
    x = nhypergeom.rvs(22, [7, 8, 9], [[12], [13]], size=(5, 1, 2, 3))
    assert x.shape == (5, 1, 2, 3)

def test_nhypergeom_accuracy():
    if False:
        print('Hello World!')
    np.random.seed(0)
    x = nhypergeom.rvs(22, 7, 11, size=100)
    np.random.seed(0)
    p = np.random.uniform(size=100)
    y = nhypergeom.ppf(p, 22, 7, 11)
    assert_equal(x, y)

def test_boltzmann_upper_bound():
    if False:
        print('Hello World!')
    k = np.arange(-3, 5)
    N = 1
    p = boltzmann.pmf(k, 0.123, N)
    expected = k == 0
    assert_equal(p, expected)
    lam = np.log(2)
    N = 3
    p = boltzmann.pmf(k, lam, N)
    expected = [0, 0, 0, 4 / 7, 2 / 7, 1 / 7, 0, 0]
    assert_allclose(p, expected, rtol=1e-13)
    c = boltzmann.cdf(k, lam, N)
    expected = [0, 0, 0, 4 / 7, 6 / 7, 1, 1, 1]
    assert_allclose(c, expected, rtol=1e-13)

def test_betabinom_a_and_b_unity():
    if False:
        while True:
            i = 10
    n = 20
    k = np.arange(n + 1)
    p = betabinom(n, 1, 1).pmf(k)
    expected = np.repeat(1 / (n + 1), n + 1)
    assert_almost_equal(p, expected)

@pytest.mark.parametrize('dtypes', itertools.product(*[(int, float)] * 3))
def test_betabinom_stats_a_and_b_integers_gh18026(dtypes):
    if False:
        while True:
            i = 10
    (n_type, a_type, b_type) = dtypes
    (n, a, b) = (n_type(10), a_type(2), b_type(3))
    assert_allclose(betabinom.stats(n, a, b, moments='k'), -0.6904761904761907)

def test_betabinom_bernoulli():
    if False:
        for i in range(10):
            print('nop')
    a = 2.3
    b = 0.63
    k = np.arange(2)
    p = betabinom(1, a, b).pmf(k)
    expected = bernoulli(a / (a + b)).pmf(k)
    assert_almost_equal(p, expected)

def test_issue_10317():
    if False:
        print('Hello World!')
    (alpha, n, p) = (0.9, 10, 1)
    assert_equal(nbinom.interval(confidence=alpha, n=n, p=p), (0, 0))

def test_issue_11134():
    if False:
        print('Hello World!')
    (alpha, n, p) = (0.95, 10, 0)
    assert_equal(binom.interval(confidence=alpha, n=n, p=p), (0, 0))

def test_issue_7406():
    if False:
        i = 10
        return i + 15
    np.random.seed(0)
    assert_equal(binom.ppf(np.random.rand(10), 0, 0.5), 0)
    assert_equal(binom.ppf(0, 0, 0.5), -1)
    assert_equal(binom.ppf(1, 0, 0.5), 0)

def test_issue_5122():
    if False:
        while True:
            i = 10
    p = 0
    n = np.random.randint(100, size=10)
    x = 0
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, -1)
    x = np.linspace(0.01, 0.99, 10)
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, 0)
    x = 1
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, n)

def test_issue_1603():
    if False:
        i = 10
        return i + 15
    assert_equal(binom(1000, np.logspace(-3, -100)).ppf(0.01), 0)

def test_issue_5503():
    if False:
        i = 10
        return i + 15
    p = 0.5
    x = np.logspace(3, 14, 12)
    assert_allclose(binom.cdf(x, 2 * x, p), 0.5, atol=0.01)

@pytest.mark.parametrize('x, n, p, cdf_desired', [(300, 1000, 3 / 10, 0.5155935198141199), (3000, 10000, 3 / 10, 0.504932983819297), (30000, 100000, 3 / 10, 0.5015600059172642), (300000, 1000000, 3 / 10, 0.5004933190666696), (3000000, 10000000, 3 / 10, 0.5001560012458526), (30000000, 100000000, 3 / 10, 0.5000493319273523), (30010000, 100000000, 3 / 10, 0.9854538401657079), (29990000, 100000000, 3 / 10, 0.014550171779852687), (29950000, 100000000, 3 / 10, 5.022509634874321e-28)])
def test_issue_5503pt2(x, n, p, cdf_desired):
    if False:
        print('Hello World!')
    assert_allclose(binom.cdf(x, n, p), cdf_desired)

def test_issue_5503pt3():
    if False:
        i = 10
        return i + 15
    assert_allclose(binom.cdf(2, 10 ** 12, 10 ** (-12)), 0.9196986029286978)

def test_issue_6682():
    if False:
        print('Hello World!')
    assert_allclose(nbinom.sf(250, 50, 32.0 / 63.0), 1.460458510976452e-35)

def test_boost_divide_by_zero_issue_15101():
    if False:
        print('Hello World!')
    n = 1000
    p = 0.01
    k = 996
    assert_allclose(binom.pmf(k, n, p), 0.0)

def test_skellam_gh11474():
    if False:
        print('Hello World!')
    mu = [1, 10, 100, 1000, 5000, 5050, 5100, 5250, 6000]
    cdf = skellam.cdf(0, mu, mu)
    cdf_expected = [0.6542541612768356, 0.5448901559424127, 0.514113579974558, 0.5044605891382528, 0.501994736335045, 0.5019848365953181, 0.5019750827993392, 0.501946662180506, 0.5018209330219539]
    assert_allclose(cdf, cdf_expected)

class TestZipfian:

    def test_zipfian_asymptotic(self):
        if False:
            while True:
                i = 10
        a = 6.5
        N = 10000000
        k = np.arange(1, 21)
        assert_allclose(zipfian.pmf(k, a, N), zipf.pmf(k, a))
        assert_allclose(zipfian.cdf(k, a, N), zipf.cdf(k, a))
        assert_allclose(zipfian.sf(k, a, N), zipf.sf(k, a))
        assert_allclose(zipfian.stats(a, N, moments='msvk'), zipf.stats(a, moments='msvk'))

    def test_zipfian_continuity(self):
        if False:
            i = 10
            return i + 15
        (alt1, agt1) = (0.99999999, 1.00000001)
        N = 30
        k = np.arange(1, N + 1)
        assert_allclose(zipfian.pmf(k, alt1, N), zipfian.pmf(k, agt1, N), rtol=5e-07)
        assert_allclose(zipfian.cdf(k, alt1, N), zipfian.cdf(k, agt1, N), rtol=5e-07)
        assert_allclose(zipfian.sf(k, alt1, N), zipfian.sf(k, agt1, N), rtol=5e-07)
        assert_allclose(zipfian.stats(alt1, N, moments='msvk'), zipfian.stats(agt1, N, moments='msvk'), rtol=5e-07)

    def test_zipfian_R(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(0)
        k = np.random.randint(1, 20, size=10)
        a = np.random.rand(10) * 10 + 1
        n = np.random.randint(1, 100, size=10)
        pmf = [0.008076972, 2.950214e-05, 0.9799333, 3.216601e-06, 0.0003158895, 3.412497e-05, 4.350472e-10, 2.405773e-06, 5.860662e-06, 0.0001053948]
        cdf = [0.8964133, 0.9998666, 0.9799333, 0.9999995, 0.9998584, 0.9999458, 1.0, 0.999992, 0.9999977, 0.9998498]
        assert_allclose(zipfian.pmf(k, a, n)[1:], pmf[1:], rtol=1e-06)
        assert_allclose(zipfian.cdf(k, a, n)[1:], cdf[1:], rtol=5e-05)
    np.random.seed(0)
    naive_tests = np.vstack((np.logspace(-2, 1, 10), np.random.randint(2, 40, 10))).T

    @pytest.mark.parametrize('a, n', naive_tests)
    def test_zipfian_naive(self, a, n):
        if False:
            return 10

        @np.vectorize
        def Hns(n, s):
            if False:
                i = 10
                return i + 15
            'Naive implementation of harmonic sum'
            return (1 / np.arange(1, n + 1) ** s).sum()

        @np.vectorize
        def pzip(k, a, n):
            if False:
                print('Hello World!')
            'Naive implementation of zipfian pmf'
            if k < 1 or k > n:
                return 0.0
            else:
                return 1 / k ** a / Hns(n, a)
        k = np.arange(n + 1)
        pmf = pzip(k, a, n)
        cdf = np.cumsum(pmf)
        mean = np.average(k, weights=pmf)
        var = np.average((k - mean) ** 2, weights=pmf)
        std = var ** 0.5
        skew = np.average(((k - mean) / std) ** 3, weights=pmf)
        kurtosis = np.average(((k - mean) / std) ** 4, weights=pmf) - 3
        assert_allclose(zipfian.pmf(k, a, n), pmf)
        assert_allclose(zipfian.cdf(k, a, n), cdf)
        assert_allclose(zipfian.stats(a, n, moments='mvsk'), [mean, var, skew, kurtosis])

class TestNCH:
    np.random.seed(2)
    shape = (2, 4, 3)
    max_m = 100
    m1 = np.random.randint(1, max_m, size=shape)
    m2 = np.random.randint(1, max_m, size=shape)
    N = m1 + m2
    n = randint.rvs(0, N, size=N.shape)
    xl = np.maximum(0, n - m2)
    xu = np.minimum(n, m1)
    x = randint.rvs(xl, xu, size=xl.shape)
    odds = np.random.rand(*x.shape) * 2

    @pytest.mark.parametrize('dist_name', ['nchypergeom_fisher', 'nchypergeom_wallenius'])
    def test_nch_hypergeom(self, dist_name):
        if False:
            for i in range(10):
                print('nop')
        dists = {'nchypergeom_fisher': nchypergeom_fisher, 'nchypergeom_wallenius': nchypergeom_wallenius}
        dist = dists[dist_name]
        (x, N, m1, n) = (self.x, self.N, self.m1, self.n)
        assert_allclose(dist.pmf(x, N, m1, n, odds=1), hypergeom.pmf(x, N, m1, n))

    def test_nchypergeom_fisher_naive(self):
        if False:
            for i in range(10):
                print('nop')
        (x, N, m1, n, odds) = (self.x, self.N, self.m1, self.n, self.odds)

        @np.vectorize
        def pmf_mean_var(x, N, m1, n, w):
            if False:
                print('Hello World!')
            m2 = N - m1
            xl = np.maximum(0, n - m2)
            xu = np.minimum(n, m1)

            def f(x):
                if False:
                    return 10
                t1 = special_binom(m1, x)
                t2 = special_binom(m2, n - x)
                return t1 * t2 * w ** x

            def P(k):
                if False:
                    for i in range(10):
                        print('nop')
                return sum((f(y) * y ** k for y in range(xl, xu + 1)))
            P0 = P(0)
            P1 = P(1)
            P2 = P(2)
            pmf = f(x) / P0
            mean = P1 / P0
            var = P2 / P0 - (P1 / P0) ** 2
            return (pmf, mean, var)
        (pmf, mean, var) = pmf_mean_var(x, N, m1, n, odds)
        assert_allclose(nchypergeom_fisher.pmf(x, N, m1, n, odds), pmf)
        assert_allclose(nchypergeom_fisher.stats(N, m1, n, odds, moments='m'), mean)
        assert_allclose(nchypergeom_fisher.stats(N, m1, n, odds, moments='v'), var)

    def test_nchypergeom_wallenius_naive(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(2)
        shape = (2, 4, 3)
        max_m = 100
        m1 = np.random.randint(1, max_m, size=shape)
        m2 = np.random.randint(1, max_m, size=shape)
        N = m1 + m2
        n = randint.rvs(0, N, size=N.shape)
        xl = np.maximum(0, n - m2)
        xu = np.minimum(n, m1)
        x = randint.rvs(xl, xu, size=xl.shape)
        w = np.random.rand(*x.shape) * 2

        def support(N, m1, n, w):
            if False:
                i = 10
                return i + 15
            m2 = N - m1
            xl = np.maximum(0, n - m2)
            xu = np.minimum(n, m1)
            return (xl, xu)

        @np.vectorize
        def mean(N, m1, n, w):
            if False:
                for i in range(10):
                    print('nop')
            m2 = N - m1
            (xl, xu) = support(N, m1, n, w)

            def fun(u):
                if False:
                    i = 10
                    return i + 15
                return u / m1 + (1 - (n - u) / m2) ** w - 1
            return root_scalar(fun, bracket=(xl, xu)).root
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message='invalid value encountered in mean')
            assert_allclose(nchypergeom_wallenius.mean(N, m1, n, w), mean(N, m1, n, w), rtol=0.02)

        @np.vectorize
        def variance(N, m1, n, w):
            if False:
                print('Hello World!')
            m2 = N - m1
            u = mean(N, m1, n, w)
            a = u * (m1 - u)
            b = (n - u) * (u + m2 - n)
            return N * a * b / ((N - 1) * (m1 * b + m2 * a))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message='invalid value encountered in mean')
            assert_allclose(nchypergeom_wallenius.stats(N, m1, n, w, moments='v'), variance(N, m1, n, w), rtol=0.05)

        @np.vectorize
        def pmf(x, N, m1, n, w):
            if False:
                i = 10
                return i + 15
            m2 = N - m1
            (xl, xu) = support(N, m1, n, w)

            def integrand(t):
                if False:
                    print('Hello World!')
                D = w * (m1 - x) + (m2 - (n - x))
                res = (1 - t ** (w / D)) ** x * (1 - t ** (1 / D)) ** (n - x)
                return res

            def f(x):
                if False:
                    print('Hello World!')
                t1 = special_binom(m1, x)
                t2 = special_binom(m2, n - x)
                the_integral = quad(integrand, 0, 1, epsrel=1e-16, epsabs=1e-16)
                return t1 * t2 * the_integral[0]
            return f(x)
        pmf0 = pmf(x, N, m1, n, w)
        pmf1 = nchypergeom_wallenius.pmf(x, N, m1, n, w)
        (atol, rtol) = (1e-06, 1e-06)
        i = np.abs(pmf1 - pmf0) < atol + rtol * np.abs(pmf0)
        assert i.sum() > np.prod(shape) / 2
        for (N, m1, n, w) in zip(N[~i], m1[~i], n[~i], w[~i]):
            m2 = N - m1
            (xl, xu) = support(N, m1, n, w)
            x = np.arange(xl, xu + 1)
            assert pmf(x, N, m1, n, w).sum() < 0.5
            assert_allclose(nchypergeom_wallenius.pmf(x, N, m1, n, w).sum(), 1)

    def test_wallenius_against_mpmath(self):
        if False:
            print('Hello World!')
        M = 50
        n = 30
        N = 20
        odds = 2.25
        sup = np.arange(21)
        pmf = np.array([3.699003068656875e-20, 5.89398584245431e-17, 2.1594437742911123e-14, 3.221458044649955e-12, 2.4658279241205077e-10, 1.0965862603981212e-08, 3.057890479665704e-07, 5.622818831643761e-06, 7.056482841531681e-05, 0.000618899425358671, 0.003854172932571669, 0.01720592676256026, 0.05528844897093792, 0.12772363313574242, 0.21065898367825722, 0.24465958845359234, 0.1955114898110033, 0.10355390084949237, 0.03414490375225675, 0.006231989845775931, 0.0004715577304677075])
        mean = 14.808018384813426
        var = 2.6085975877923717
        assert_allclose(nchypergeom_wallenius.pmf(sup, M, n, N, odds), pmf, rtol=1e-13, atol=1e-13)
        assert_allclose(nchypergeom_wallenius.mean(M, n, N, odds), mean, rtol=1e-13)
        assert_allclose(nchypergeom_wallenius.var(M, n, N, odds), var, rtol=1e-11)

    @pytest.mark.parametrize('dist_name', ['nchypergeom_fisher', 'nchypergeom_wallenius'])
    def test_rvs_shape(self, dist_name):
        if False:
            for i in range(10):
                print('nop')
        dists = {'nchypergeom_fisher': nchypergeom_fisher, 'nchypergeom_wallenius': nchypergeom_wallenius}
        dist = dists[dist_name]
        x = dist.rvs(50, 30, [[10], [20]], [0.5, 1.0, 2.0], size=(5, 1, 2, 3))
        assert x.shape == (5, 1, 2, 3)

@pytest.mark.parametrize('mu, q, expected', [[10, 120, -1.240089881791596e-38], [1500, 0, -86.61466680572661]])
def test_nbinom_11465(mu, q, expected):
    if False:
        while True:
            i = 10
    size = 20
    (n, p) = (size, size / (size + mu))
    assert_allclose(nbinom.logcdf(q, n, p), expected)

def test_gh_17146():
    if False:
        for i in range(10):
            print('nop')
    x = np.linspace(0, 1, 11)
    p = 0.8
    pmf = bernoulli(p).pmf(x)
    i = x % 1 == 0
    assert_allclose(pmf[-1], p)
    assert_allclose(pmf[0], 1 - p)
    assert_equal(pmf[~i], 0)