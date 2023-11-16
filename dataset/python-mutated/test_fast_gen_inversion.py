import pytest
import warnings
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, suppress_warnings
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats

def test_bad_args():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='loc must be scalar'):
        FastGeneratorInversion(stats.norm(loc=(1.2, 1.3)))
    with pytest.raises(ValueError, match='scale must be scalar'):
        FastGeneratorInversion(stats.norm(scale=[1.5, 5.7]))
    with pytest.raises(ValueError, match="'test' cannot be used to seed"):
        FastGeneratorInversion(stats.norm(), random_state='test')
    msg = 'Each of the 1 shape parameters must be a scalar'
    with pytest.raises(ValueError, match=msg):
        FastGeneratorInversion(stats.gamma([1.3, 2.5]))
    with pytest.raises(ValueError, match='`dist` must be a frozen'):
        FastGeneratorInversion('xy')
    with pytest.raises(ValueError, match="Distribution 'truncnorm' is not"):
        FastGeneratorInversion(stats.truncnorm(1.3, 4.5))

def test_random_state():
    if False:
        print('Hello World!')
    gen = FastGeneratorInversion(stats.norm(), random_state=68734509)
    x1 = gen.rvs(size=10)
    gen.random_state = 68734509
    x2 = gen.rvs(size=10)
    assert_array_equal(x1, x2)
    urng = np.random.default_rng(20375857)
    gen = FastGeneratorInversion(stats.norm(), random_state=urng)
    x1 = gen.rvs(size=10)
    gen.random_state = np.random.default_rng(20375857)
    x2 = gen.rvs(size=10)
    assert_array_equal(x1, x2)
    urng = np.random.RandomState(2364)
    gen = FastGeneratorInversion(stats.norm(), random_state=urng)
    x1 = gen.rvs(size=10)
    gen.random_state = np.random.RandomState(2364)
    x2 = gen.rvs(size=10)
    assert_array_equal(x1, x2)
    gen = FastGeneratorInversion(stats.norm(), random_state=68734509)
    x1 = gen.rvs(size=10)
    _ = gen.evaluate_error(size=5)
    x2 = gen.rvs(size=10)
    gen.random_state = 68734509
    x3 = gen.rvs(size=20)
    assert_array_equal(x2, x3[10:])
dists_with_params = [('alpha', (3.5,)), ('anglit', ()), ('argus', (3.5,)), ('argus', (5.1,)), ('beta', (1.5, 0.9)), ('cosine', ()), ('betaprime', (2.5, 3.3)), ('bradford', (1.2,)), ('burr', (1.3, 2.4)), ('burr12', (0.7, 1.2)), ('cauchy', ()), ('chi2', (3.5,)), ('chi', (4.5,)), ('crystalball', (0.7, 1.2)), ('expon', ()), ('gamma', (1.5,)), ('gennorm', (2.7,)), ('gumbel_l', ()), ('gumbel_r', ()), ('hypsecant', ()), ('invgauss', (3.1,)), ('invweibull', (1.5,)), ('laplace', ()), ('logistic', ()), ('maxwell', ()), ('moyal', ()), ('norm', ()), ('pareto', (1.3,)), ('powerlaw', (7.6,)), ('rayleigh', ()), ('semicircular', ()), ('t', (5.7,)), ('wald', ()), ('weibull_max', (2.4,)), ('weibull_min', (1.2,))]

@pytest.mark.parametrize('distname, args', dists_with_params)
def test_rvs_and_ppf(distname, args):
    if False:
        while True:
            i = 10
    urng = np.random.default_rng(9807324628097097)
    rng1 = getattr(stats, distname)(*args)
    rvs1 = rng1.rvs(size=500, random_state=urng)
    rng2 = FastGeneratorInversion(rng1, random_state=urng)
    rvs2 = rng2.rvs(size=500)
    assert stats.cramervonmises_2samp(rvs1, rvs2).pvalue > 0.01
    q = [0.001, 0.1, 0.5, 0.9, 0.999]
    assert_allclose(rng1.ppf(q), rng2.ppf(q), atol=1e-10)

@pytest.mark.parametrize('distname, args', dists_with_params)
def test_u_error(distname, args):
    if False:
        return 10
    dist = getattr(stats, distname)(*args)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        rng = FastGeneratorInversion(dist)
    (u_error, x_error) = rng.evaluate_error(size=10000, random_state=9807324628097097, x_error=False)
    assert u_error <= 1e-10

@pytest.mark.xfail(reason='geninvgauss CDF is not accurate')
def test_geninvgauss_uerror():
    if False:
        while True:
            i = 10
    dist = stats.geninvgauss(3.2, 1.5)
    rng = FastGeneratorInversion(dist)
    err = rng.evaluate_error(size=10000, random_state=67982)
    assert err[0] < 1e-10

@pytest.mark.parametrize('distname, args', [('beta', (0.11, 0.11))])
def test_error_extreme_params(distname, args):
    if False:
        for i in range(10):
            print('nop')
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        dist = getattr(stats, distname)(*args)
        rng = FastGeneratorInversion(dist)
    (u_error, x_error) = rng.evaluate_error(size=10000, random_state=980732462809709732623, x_error=True)
    if u_error >= 2.5 * 1e-10:
        assert x_error < 1e-09

def test_evaluate_error_inputs():
    if False:
        return 10
    gen = FastGeneratorInversion(stats.norm())
    with pytest.raises(ValueError, match='size must be an integer'):
        gen.evaluate_error(size=3.5)
    with pytest.raises(ValueError, match='size must be an integer'):
        gen.evaluate_error(size=(3, 3))

def test_rvs_ppf_loc_scale():
    if False:
        return 10
    (loc, scale) = (3.5, 2.3)
    dist = stats.norm(loc=loc, scale=scale)
    rng = FastGeneratorInversion(dist, random_state=1234)
    r = rng.rvs(size=1000)
    r_rescaled = (r - loc) / scale
    assert stats.cramervonmises(r_rescaled, 'norm').pvalue > 0.01
    q = [0.001, 0.1, 0.5, 0.9, 0.999]
    assert_allclose(rng._ppf(q), rng.ppf(q), atol=1e-10)

def test_domain():
    if False:
        return 10
    rng = FastGeneratorInversion(stats.norm(), domain=(-1, 1))
    r = rng.rvs(size=100)
    assert -1 <= r.min() < r.max() <= 1
    (loc, scale) = (3.5, 1.3)
    dist = stats.norm(loc=loc, scale=scale)
    rng = FastGeneratorInversion(dist, domain=(-1.5, 2))
    r = rng.rvs(size=100)
    (lb, ub) = (loc - scale * 1.5, loc + scale * 2)
    assert lb <= r.min() < r.max() <= ub

@pytest.mark.parametrize('distname, args, expected', [('beta', (3.5, 2.5), (0, 1)), ('norm', (), (-np.inf, np.inf))])
def test_support(distname, args, expected):
    if False:
        return 10
    dist = getattr(stats, distname)(*args)
    rng = FastGeneratorInversion(dist)
    assert_array_equal(rng.support(), expected)
    rng.loc = 1
    rng.scale = 2
    assert_array_equal(rng.support(), 1 + 2 * np.array(expected))

@pytest.mark.parametrize('distname, args', [('beta', (3.5, 2.5)), ('norm', ())])
def test_support_truncation(distname, args):
    if False:
        for i in range(10):
            print('nop')
    dist = getattr(stats, distname)(*args)
    rng = FastGeneratorInversion(dist, domain=(0.5, 0.7))
    assert_array_equal(rng.support(), (0.5, 0.7))
    rng.loc = 1
    rng.scale = 2
    assert_array_equal(rng.support(), (1 + 2 * 0.5, 1 + 2 * 0.7))

def test_domain_shift_truncation():
    if False:
        for i in range(10):
            print('nop')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rng = FastGeneratorInversion(stats.norm(), domain=(1, 2))
    r = rng.rvs(size=100)
    assert 1 <= r.min() < r.max() <= 2

def test_non_rvs_methods_with_domain():
    if False:
        while True:
            i = 10
    rng = FastGeneratorInversion(stats.norm(), domain=(2.3, 3.2))
    trunc_norm = stats.truncnorm(2.3, 3.2)
    x = (2.0, 2.4, 3.0, 3.4)
    p = (0.01, 0.5, 0.99)
    assert_allclose(rng._cdf(x), trunc_norm.cdf(x))
    assert_allclose(rng._ppf(p), trunc_norm.ppf(p))
    (loc, scale) = (2, 3)
    rng.loc = 2
    rng.scale = 3
    trunc_norm = stats.truncnorm(2.3, 3.2, loc=loc, scale=scale)
    x = np.array(x) * scale + loc
    assert_allclose(rng._cdf(x), trunc_norm.cdf(x))
    assert_allclose(rng._ppf(p), trunc_norm.ppf(p))
    rng = FastGeneratorInversion(stats.beta(2.5, 3.5), domain=(0.3, 0.7))
    rng.loc = 2
    rng.scale = 2.5
    assert_array_equal(rng.support(), (2.75, 3.75))
    x = np.array([2.74, 2.76, 3.74, 3.76])
    y_cdf = rng._cdf(x)
    assert_array_equal((y_cdf[0], y_cdf[3]), (0, 1))
    assert np.min(y_cdf[1:3]) > 0
    assert_allclose(rng._ppf(y_cdf), (2.75, 2.76, 3.74, 3.75))

def test_non_rvs_methods_without_domain():
    if False:
        print('Hello World!')
    norm_dist = stats.norm()
    rng = FastGeneratorInversion(norm_dist)
    x = np.linspace(-3, 3, num=10)
    p = (0.01, 0.5, 0.99)
    assert_allclose(rng._cdf(x), norm_dist.cdf(x))
    assert_allclose(rng._ppf(p), norm_dist.ppf(p))
    (loc, scale) = (0.5, 1.3)
    rng.loc = loc
    rng.scale = scale
    norm_dist = stats.norm(loc=loc, scale=scale)
    assert_allclose(rng._cdf(x), norm_dist.cdf(x))
    assert_allclose(rng._ppf(p), norm_dist.ppf(p))

@pytest.mark.parametrize('domain, x', [(None, 0.5), ((0, 1), 0.5), ((0, 1), 1.5)])
def test_scalar_inputs(domain, x):
    if False:
        print('Hello World!')
    ' pdf, cdf etc should map scalar values to scalars. check with and\n    w/o domain since domain impacts pdf, cdf etc\n    Take x inside and outside of domain '
    rng = FastGeneratorInversion(stats.norm(), domain=domain)
    assert np.isscalar(rng._cdf(x))
    assert np.isscalar(rng._ppf(0.5))

def test_domain_argus_large_chi():
    if False:
        return 10
    (chi, lb, ub) = (5.5, 0.25, 0.75)
    rng = FastGeneratorInversion(stats.argus(chi), domain=(lb, ub))
    rng.random_state = 4574
    r = rng.rvs(size=500)
    assert lb <= r.min() < r.max() <= ub
    cdf = stats.argus(chi).cdf
    prob = cdf(ub) - cdf(lb)
    assert stats.cramervonmises(r, lambda x: cdf(x) / prob).pvalue > 0.05

def test_setting_loc_scale():
    if False:
        print('Hello World!')
    rng = FastGeneratorInversion(stats.norm(), random_state=765765864)
    r1 = rng.rvs(size=1000)
    rng.loc = 3.0
    rng.scale = 2.5
    r2 = rng.rvs(1000)
    assert stats.cramervonmises_2samp(r1, (r2 - 3) / 2.5).pvalue > 0.05
    rng.loc = 0
    rng.scale = 1
    r2 = rng.rvs(1000)
    assert stats.cramervonmises_2samp(r1, r2).pvalue > 0.05

def test_ignore_shape_range():
    if False:
        print('Hello World!')
    msg = 'No generator is defined for the shape parameters'
    with pytest.raises(ValueError, match=msg):
        rng = FastGeneratorInversion(stats.t(0.03))
    rng = FastGeneratorInversion(stats.t(0.03), ignore_shape_range=True)
    (u_err, _) = rng.evaluate_error(size=1000, random_state=234)
    assert u_err >= 1e-06

@pytest.mark.xfail_on_32bit('NumericalInversePolynomial.qrvs fails for Win 32-bit')
class TestQRVS:

    def test_input_validation(self):
        if False:
            return 10
        gen = FastGeneratorInversion(stats.norm())
        match = '`qmc_engine` must be an instance of...'
        with pytest.raises(ValueError, match=match):
            gen.qrvs(qmc_engine=0)
        match = '`d` must be consistent with dimension of `qmc_engine`.'
        with pytest.raises(ValueError, match=match):
            gen.qrvs(d=3, qmc_engine=stats.qmc.Halton(2))
    qrngs = [None, stats.qmc.Sobol(1, seed=0), stats.qmc.Halton(3, seed=0)]
    sizes = [(None, tuple()), (1, (1,)), (4, (4,)), ((4,), (4,)), ((2, 4), (2, 4))]
    ds = [(None, tuple()), (1, tuple()), (3, (3,))]

    @pytest.mark.parametrize('qrng', qrngs)
    @pytest.mark.parametrize('size_in, size_out', sizes)
    @pytest.mark.parametrize('d_in, d_out', ds)
    def test_QRVS_shape_consistency(self, qrng, size_in, size_out, d_in, d_out):
        if False:
            while True:
                i = 10
        gen = FastGeneratorInversion(stats.norm())
        if d_in is not None and qrng is not None and (qrng.d != d_in):
            match = '`d` must be consistent with dimension of `qmc_engine`.'
            with pytest.raises(ValueError, match=match):
                gen.qrvs(size_in, d=d_in, qmc_engine=qrng)
            return
        if d_in is None and qrng is not None and (qrng.d != 1):
            d_out = (qrng.d,)
        shape_expected = size_out + d_out
        qrng2 = deepcopy(qrng)
        qrvs = gen.qrvs(size=size_in, d=d_in, qmc_engine=qrng)
        if size_in is not None:
            assert qrvs.shape == shape_expected
        if qrng2 is not None:
            uniform = qrng2.random(np.prod(size_in) or 1)
            qrvs2 = stats.norm.ppf(uniform).reshape(shape_expected)
            assert_allclose(qrvs, qrvs2, atol=1e-12)

    def test_QRVS_size_tuple(self):
        if False:
            return 10
        gen = FastGeneratorInversion(stats.norm())
        size = (3, 4)
        d = 5
        qrng = stats.qmc.Halton(d, seed=0)
        qrng2 = stats.qmc.Halton(d, seed=0)
        uniform = qrng2.random(np.prod(size))
        qrvs = gen.qrvs(size=size, d=d, qmc_engine=qrng)
        qrvs2 = stats.norm.ppf(uniform)
        for i in range(d):
            sample = qrvs[..., i]
            sample2 = qrvs2[:, i].reshape(size)
            assert_allclose(sample, sample2, atol=1e-12)

def test_burr_overflow():
    if False:
        print('Hello World!')
    args = (1.89128135, 0.30195177)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        gen = FastGeneratorInversion(stats.burr(*args))
    (u_error, _) = gen.evaluate_error(random_state=4326)
    assert u_error <= 1e-10