import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_entropy_vect_scale, check_edge_support, check_named_args, check_random_state_property, check_meth_dtype, check_ppf_dtype, check_cmplx_deriv, check_pickling, check_rvs_broadcast, check_freezing, check_munp_expect
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
'\nTest all continuous distributions.\n\nParameters were chosen for those distributions that pass the\nKolmogorov-Smirnov test.  This provides safe parameters for each\ndistributions so that we can perform further testing of class methods.\n\nThese tests currently check only/mostly for serious errors and exceptions,\nnot for numerically exact results.\n'
DECIMAL = 5
_IS_32BIT = sys.maxsize < 2 ** 32
distslow = ['recipinvgauss', 'vonmises', 'kappa4', 'vonmises_line', 'gausshyper', 'norminvgauss', 'geninvgauss', 'genhyperbolic', 'truncnorm', 'truncweibull_min']
distxslow = ['studentized_range', 'kstwo', 'ksone', 'wrapcauchy', 'genexpon']
distxslow_test_moments = ['studentized_range', 'vonmises', 'vonmises_line', 'ksone', 'kstwo', 'recipinvgauss', 'genexpon']
skip_fit_test_mle = ['exponpow', 'exponweib', 'gausshyper', 'genexpon', 'halfgennorm', 'gompertz', 'johnsonsb', 'johnsonsu', 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'mielke', 'ncf', 'nct', 'powerlognorm', 'powernorm', 'recipinvgauss', 'trapezoid', 'vonmises', 'vonmises_line', 'levy_stable', 'rv_histogram_instance', 'studentized_range']
slow_fit_test_mm = ['argus', 'exponpow', 'exponweib', 'gausshyper', 'genexpon', 'genhalflogistic', 'halfgennorm', 'gompertz', 'johnsonsb', 'kappa4', 'kstwobign', 'recipinvgauss', 'trapezoid', 'truncexpon', 'vonmises', 'vonmises_line', 'studentized_range']
fail_fit_test_mm = ['alpha', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'crystalball', 'f', 'fisk', 'foldcauchy', 'genextreme', 'genpareto', 'halfcauchy', 'invgamma', 'jf_skew_t', 'kappa3', 'levy', 'levy_l', 'loglaplace', 'lomax', 'mielke', 'nakagami', 'ncf', 'skewcauchy', 't', 'tukeylambda', 'invweibull', 'rel_breitwigner'] + ['genhyperbolic', 'johnsonsu', 'ksone', 'kstwo', 'nct', 'pareto', 'powernorm', 'powerlognorm'] + ['pearson3']
skip_fit_test = {'MLE': skip_fit_test_mle, 'MM': slow_fit_test_mm + fail_fit_test_mm}
skip_fit_fix_test_mle = ['burr', 'exponpow', 'exponweib', 'gausshyper', 'genexpon', 'halfgennorm', 'gompertz', 'johnsonsb', 'johnsonsu', 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'levy_stable', 'mielke', 'ncf', 'ncx2', 'powerlognorm', 'powernorm', 'rdist', 'recipinvgauss', 'trapezoid', 'truncpareto', 'vonmises', 'vonmises_line', 'studentized_range']
fail_fit_fix_test_mm = ['alpha', 'betaprime', 'burr', 'burr12', 'cauchy', 'crystalball', 'f', 'fisk', 'foldcauchy', 'genextreme', 'genpareto', 'halfcauchy', 'invgamma', 'jf_skew_t', 'kappa3', 'levy', 'levy_l', 'loglaplace', 'lomax', 'mielke', 'nakagami', 'ncf', 'nct', 'skewcauchy', 't', 'truncpareto', 'invweibull'] + ['genhyperbolic', 'johnsonsu', 'ksone', 'kstwo', 'pareto', 'powernorm', 'powerlognorm'] + ['pearson3']
skip_fit_fix_test = {'MLE': skip_fit_fix_test_mle, 'MM': slow_fit_test_mm + fail_fit_fix_test_mm}
fails_cmplx = {'argus', 'beta', 'betaprime', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'f', 'foldcauchy', 'gamma', 'gausshyper', 'gengamma', 'genhyperbolic', 'geninvgauss', 'gennorm', 'genpareto', 'halfcauchy', 'halfgennorm', 'invgamma', 'jf_skew_t', 'ksone', 'kstwo', 'kstwobign', 'levy_l', 'loggamma', 'logistic', 'loguniform', 'maxwell', 'nakagami', 'ncf', 'nct', 'ncx2', 'norminvgauss', 'pearson3', 'powerlaw', 'rdist', 'reciprocal', 'rice', 'skewnorm', 't', 'truncweibull_min', 'tukeylambda', 'vonmises', 'vonmises_line', 'rv_histogram_instance', 'truncnorm', 'studentized_range', 'johnsonsb', 'halflogistic', 'rel_breitwigner'}
histogram_test_instances = []
case1 = {'a': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9], 'bins': 8}
case2 = {'a': [1, 1], 'bins': [0, 1, 10]}
for (case, density) in itertools.product([case1, case2], [True, False]):
    _hist = np.histogram(**case, density=density)
    _rv_hist = stats.rv_histogram(_hist, density=density)
    histogram_test_instances.append((_rv_hist, tuple()))

def cases_test_cont_basic():
    if False:
        while True:
            i = 10
    for (distname, arg) in distcont[:] + histogram_test_instances:
        if distname == 'levy_stable':
            continue
        elif distname in distslow:
            yield pytest.param(distname, arg, marks=pytest.mark.slow)
        elif distname in distxslow:
            yield pytest.param(distname, arg, marks=pytest.mark.xslow)
        else:
            yield (distname, arg)

@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
@pytest.mark.parametrize('sn, n_fit_samples', [(500, 200)])
def test_cont_basic(distname, arg, sn, n_fit_samples):
    if False:
        i = 10
        return i + 15
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'
    rng = np.random.RandomState(765456)
    rvs = distfn.rvs(*arg, size=sn, random_state=rng)
    (m, v) = distfn.stats(*arg)
    if distname not in {'laplace_asymmetric'}:
        check_sample_meanvar_(m, v, rvs)
    check_cdf_ppf(distfn, arg, distname)
    check_sf_isf(distfn, arg, distname)
    check_cdf_sf(distfn, arg, distname)
    check_ppf_isf(distfn, arg, distname)
    check_pdf(distfn, arg, distname)
    check_pdf_logpdf(distfn, arg, distname)
    check_pdf_logpdf_at_endpoints(distfn, arg, distname)
    check_cdf_logcdf(distfn, arg, distname)
    check_sf_logsf(distfn, arg, distname)
    check_ppf_broadcast(distfn, arg, distname)
    alpha = 0.01
    if distname == 'rv_histogram_instance':
        check_distribution_rvs(distfn.cdf, arg, alpha, rvs)
    elif distname != 'geninvgauss':
        check_distribution_rvs(distname, arg, alpha, rvs)
    locscale_defaults = (0, 1)
    meths = [distfn.pdf, distfn.logpdf, distfn.cdf, distfn.logcdf, distfn.logsf]
    spec_x = {'weibull_max': -0.5, 'levy_l': -0.5, 'pareto': 1.5, 'truncpareto': 3.2, 'tukeylambda': 0.3, 'rv_histogram_instance': 5.0}
    x = spec_x.get(distname, 0.5)
    if distname == 'invweibull':
        arg = (1,)
    elif distname == 'ksone':
        arg = (3,)
    check_named_args(distfn, x, arg, locscale_defaults, meths)
    check_random_state_property(distfn, arg)
    if distname in ['rel_breitwigner'] and _IS_32BIT:
        pytest.skip('fails on Linux 32-bit')
    else:
        check_pickling(distfn, arg)
    check_freezing(distfn, arg)
    if distname not in ['kstwobign', 'kstwo', 'ncf']:
        check_entropy(distfn, arg, distname)
    if distfn.numargs == 0:
        check_vecentropy(distfn, arg)
    if distfn.__class__._entropy != stats.rv_continuous._entropy and distname != 'vonmises':
        check_private_entropy(distfn, arg, stats.rv_continuous)
    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The occurrence of roundoff error')
        sup.filter(IntegrationWarning, 'Extremely bad integrand')
        sup.filter(RuntimeWarning, 'invalid value')
        check_entropy_vect_scale(distfn, arg)
    check_retrieving_support(distfn, arg)
    check_edge_support(distfn, arg)
    check_meth_dtype(distfn, arg, meths)
    check_ppf_dtype(distfn, arg)
    if distname not in fails_cmplx:
        check_cmplx_deriv(distfn, arg)
    if distname != 'truncnorm':
        check_ppf_private(distfn, arg, distname)
    for method in ['MLE', 'MM']:
        if distname not in skip_fit_test[method]:
            check_fit_args(distfn, arg, rvs[:n_fit_samples], method)
        if distname not in skip_fit_fix_test[method]:
            check_fit_args_fix(distfn, arg, rvs[:n_fit_samples], method)

@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
def test_rvs_scalar(distname, arg):
    if False:
        while True:
            i = 10
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'
    assert np.isscalar(distfn.rvs(*arg))
    assert np.isscalar(distfn.rvs(*arg, size=()))
    assert np.isscalar(distfn.rvs(*arg, size=None))

def test_levy_stable_random_state_property():
    if False:
        while True:
            i = 10
    check_random_state_property(stats.levy_stable, (0.5, 0.1))

def cases_test_moments():
    if False:
        for i in range(10):
            print('nop')
    fail_normalization = set()
    fail_higher = {'ncf'}
    fail_moment = {'johnsonsu'}
    for (distname, arg) in distcont[:] + histogram_test_instances:
        if distname == 'levy_stable':
            continue
        if distname in distxslow_test_moments:
            yield pytest.param(distname, arg, True, True, True, True, marks=pytest.mark.xslow(reason='too slow'))
            continue
        cond1 = distname not in fail_normalization
        cond2 = distname not in fail_higher
        cond3 = distname not in fail_moment
        marks = list()
        yield pytest.param(distname, arg, cond1, cond2, cond3, False, marks=marks)
        if not cond1 or not cond2 or (not cond3):
            yield pytest.param(distname, arg, True, True, True, True, marks=[pytest.mark.xfail] + marks)

@pytest.mark.slow
@pytest.mark.parametrize('distname,arg,normalization_ok,higher_ok,moment_ok,is_xfailing', cases_test_moments())
def test_moments(distname, arg, normalization_ok, higher_ok, moment_ok, is_xfailing):
    if False:
        while True:
            i = 10
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'
    with npt.suppress_warnings() as sup:
        sup.filter(IntegrationWarning, 'The integral is probably divergent, or slowly convergent.')
        sup.filter(IntegrationWarning, 'The maximum number of subdivisions.')
        sup.filter(IntegrationWarning, 'The algorithm does not converge.')
        if is_xfailing:
            sup.filter(IntegrationWarning)
        (m, v, s, k) = distfn.stats(*arg, moments='mvsk')
        with np.errstate(all='ignore'):
            if normalization_ok:
                check_normalization(distfn, arg, distname)
            if higher_ok:
                check_mean_expect(distfn, arg, m, distname)
                check_skew_expect(distfn, arg, m, v, s, distname)
                check_var_expect(distfn, arg, m, v, distname)
                check_kurt_expect(distfn, arg, m, v, k, distname)
                check_munp_expect(distfn, arg, distname)
        check_loc_scale(distfn, arg, m, v, distname)
        if moment_ok:
            check_moment(distfn, arg, m, v, distname)

@pytest.mark.parametrize('dist,shape_args', distcont)
def test_rvs_broadcast(dist, shape_args):
    if False:
        print('Hello World!')
    if dist in ['gausshyper', 'studentized_range']:
        pytest.skip('too slow')
    if dist in ['rel_breitwigner'] and _IS_32BIT:
        pytest.skip('fails on Linux 32-bit')
    shape_only = dist in ['argus', 'betaprime', 'dgamma', 'dweibull', 'exponnorm', 'genhyperbolic', 'geninvgauss', 'levy_stable', 'nct', 'norminvgauss', 'rice', 'skewnorm', 'semicircular', 'gennorm', 'loggamma']
    distfunc = getattr(stats, dist)
    loc = np.zeros(2)
    scale = np.ones((3, 1))
    nargs = distfunc.numargs
    allargs = []
    bshape = [3, 2]
    for k in range(nargs):
        shp = (k + 4,) + (1,) * (k + 2)
        allargs.append(shape_args[k] * np.ones(shp))
        bshape.insert(0, k + 4)
    allargs.extend([loc, scale])
    check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, 'd')

@pytest.mark.parametrize('x,n,sf,cdf,pdf,rtol', [(2e-05, 1000000000, 0.44932297307934443, 0.5506770269206556, 35946.13739499628, 5e-15), (2e-09, 1000000000, 0.9999999906111111, 9.388888844813272e-09, 8.666666585296298, 5e-14), (0.0005, 1000000000, 7.122201943309037e-218, 1.0, 1.4244408634752703e-211, 5e-14)])
def test_gh17775_regression(x, n, sf, cdf, pdf, rtol):
    if False:
        while True:
            i = 10
    ks = stats.ksone
    vals = np.array([ks.sf(x, n), ks.cdf(x, n), ks.pdf(x, n)])
    expected = np.array([sf, cdf, pdf])
    npt.assert_allclose(vals, expected, rtol=rtol)
    npt.assert_equal(vals[0] + vals[1], 1.0)
    npt.assert_allclose([ks.isf(sf, n)], [x], rtol=1e-08)

def test_rvs_gh2069_regression():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(123)
    vals = stats.norm.rvs(loc=np.zeros(5), scale=1, random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    vals = stats.norm.rvs(loc=0, scale=np.ones(5), random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    vals = stats.norm.rvs(loc=np.zeros(5), scale=np.ones(5), random_state=rng)
    d = np.diff(vals)
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    vals = stats.norm.rvs(loc=np.array([[0], [0]]), scale=np.ones(5), random_state=rng)
    d = np.diff(vals.ravel())
    npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
    assert_raises(ValueError, stats.norm.rvs, [[0, 0], [0, 0]], [[1, 1], [1, 1]], 1)
    assert_raises(ValueError, stats.gamma.rvs, [2, 3, 4, 5], 0, 1, (2, 2))
    assert_raises(ValueError, stats.gamma.rvs, [1, 1, 1, 1], [0, 0, 0, 0], [[1], [2]], (4,))

def test_nomodify_gh9900_regression():
    if False:
        for i in range(10):
            print('nop')
    tn = stats.truncnorm
    npt.assert_almost_equal(tn.cdf(1, 0, np.inf), 0.6826894921370859)
    npt.assert_almost_equal(tn._cdf([1], [0], [np.inf]), 0.6826894921370859)
    npt.assert_almost_equal(tn.cdf(-1, -np.inf, 0), 0.31731050786291415)
    npt.assert_almost_equal(tn._cdf([-1], [-np.inf], [0]), 0.31731050786291415)
    npt.assert_almost_equal(tn._cdf([1], [0], [np.inf]), 0.6826894921370859)
    npt.assert_almost_equal(tn.cdf(1, 0, np.inf), 0.6826894921370859)
    npt.assert_almost_equal(tn._cdf([-1], [-np.inf], [0]), 0.31731050786291415)
    npt.assert_almost_equal(tn.cdf(1, -np.inf, 0), 1)
    npt.assert_almost_equal(tn.cdf(-1, -np.inf, 0), 0.31731050786291415)

def test_broadcast_gh9990_regression():
    if False:
        for i in range(10):
            print('nop')
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([8, 16, 1, 32, 1, 48])
    ans = [stats.reciprocal.cdf(7, _a, _b) for (_a, _b) in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(7, a, b), ans)
    ans = [stats.reciprocal.cdf(1, _a, _b) for (_a, _b) in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(1, a, b), ans)
    ans = [stats.reciprocal.cdf(_a, _a, _b) for (_a, _b) in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(a, a, b), ans)
    ans = [stats.reciprocal.cdf(_b, _a, _b) for (_a, _b) in zip(a, b)]
    npt.assert_array_almost_equal(stats.reciprocal.cdf(b, a, b), ans)

def test_broadcast_gh7933_regression():
    if False:
        return 10
    stats.truncnorm.logpdf(np.array([3.0, 2.0, 1.0]), a=(1.5 - np.array([6.0, 5.0, 4.0])) / 3.0, b=np.inf, loc=np.array([6.0, 5.0, 4.0]), scale=3.0)

def test_gh2002_regression():
    if False:
        for i in range(10):
            print('nop')
    x = np.r_[-2:2:101j]
    a = np.r_[-np.ones(50), np.ones(51)]
    expected = [stats.truncnorm.pdf(_x, _a, np.inf) for (_x, _a) in zip(x, a)]
    ans = stats.truncnorm.pdf(x, a, np.inf)
    npt.assert_array_almost_equal(ans, expected)

def test_gh1320_regression():
    if False:
        for i in range(10):
            print('nop')
    c = 2.62
    stats.genextreme.ppf(0.5, np.array([[c], [c + 0.5]]))

def test_method_of_moments():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(1234)
    x = [0, 0, 0, 0, 1]
    a = 1 / 5 - 2 * np.sqrt(3) / 5
    b = 1 / 5 + 2 * np.sqrt(3) / 5
    (loc, scale) = super(type(stats.uniform), stats.uniform).fit(x, method='MM')
    npt.assert_almost_equal(loc, a, decimal=4)
    npt.assert_almost_equal(loc + scale, b, decimal=4)

def check_sample_meanvar_(popmean, popvar, sample):
    if False:
        while True:
            i = 10
    if np.isfinite(popmean):
        check_sample_mean(sample, popmean)
    if np.isfinite(popvar):
        check_sample_var(sample, popvar)

def check_sample_mean(sample, popmean):
    if False:
        i = 10
        return i + 15
    prob = stats.ttest_1samp(sample, popmean).pvalue
    assert prob > 0.01

def check_sample_var(sample, popvar):
    if False:
        return 10
    res = stats.bootstrap((sample,), lambda x, axis: x.var(ddof=1, axis=axis), confidence_level=0.995)
    conf = res.confidence_interval
    (low, high) = (conf.low, conf.high)
    assert low <= popvar <= high

def check_cdf_ppf(distfn, arg, msg):
    if False:
        for i in range(10):
            print('nop')
    values = [0.001, 0.5, 0.999]
    npt.assert_almost_equal(distfn.cdf(distfn.ppf(values, *arg), *arg), values, decimal=DECIMAL, err_msg=msg + ' - cdf-ppf roundtrip')

def check_sf_isf(distfn, arg, msg):
    if False:
        i = 10
        return i + 15
    npt.assert_almost_equal(distfn.sf(distfn.isf([0.1, 0.5, 0.9], *arg), *arg), [0.1, 0.5, 0.9], decimal=DECIMAL, err_msg=msg + ' - sf-isf roundtrip')

def check_cdf_sf(distfn, arg, msg):
    if False:
        while True:
            i = 10
    npt.assert_almost_equal(distfn.cdf([0.1, 0.9], *arg), 1.0 - distfn.sf([0.1, 0.9], *arg), decimal=DECIMAL, err_msg=msg + ' - cdf-sf relationship')

def check_ppf_isf(distfn, arg, msg):
    if False:
        print('Hello World!')
    p = np.array([0.1, 0.9])
    npt.assert_almost_equal(distfn.isf(p, *arg), distfn.ppf(1 - p, *arg), decimal=DECIMAL, err_msg=msg + ' - ppf-isf relationship')

def check_pdf(distfn, arg, msg):
    if False:
        while True:
            i = 10
    median = distfn.ppf(0.5, *arg)
    eps = 1e-06
    pdfv = distfn.pdf(median, *arg)
    if pdfv < 0.0001 or pdfv > 10000.0:
        median = median + 0.1
        pdfv = distfn.pdf(median, *arg)
    cdfdiff = (distfn.cdf(median + eps, *arg) - distfn.cdf(median - eps, *arg)) / eps / 2.0
    msg += ' - cdf-pdf relationship'
    npt.assert_almost_equal(pdfv, cdfdiff, decimal=DECIMAL, err_msg=msg)

def check_pdf_logpdf(distfn, args, msg):
    if False:
        print('Hello World!')
    points = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    pdf = distfn.pdf(vals, *args)
    logpdf = distfn.logpdf(vals, *args)
    pdf = pdf[(pdf != 0) & np.isfinite(pdf)]
    logpdf = logpdf[np.isfinite(logpdf)]
    msg += ' - logpdf-log(pdf) relationship'
    npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)

def check_pdf_logpdf_at_endpoints(distfn, args, msg):
    if False:
        for i in range(10):
            print('nop')
    points = np.array([0, 1])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    pdf = distfn.pdf(vals, *args)
    logpdf = distfn.logpdf(vals, *args)
    pdf = pdf[(pdf != 0) & np.isfinite(pdf)]
    logpdf = logpdf[np.isfinite(logpdf)]
    msg += ' - logpdf-log(pdf) relationship'
    npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)

def check_sf_logsf(distfn, args, msg):
    if False:
        i = 10
        return i + 15
    points = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    sf = distfn.sf(vals, *args)
    logsf = distfn.logsf(vals, *args)
    sf = sf[sf != 0]
    logsf = logsf[np.isfinite(logsf)]
    msg += ' - logsf-log(sf) relationship'
    npt.assert_almost_equal(np.log(sf), logsf, decimal=7, err_msg=msg)

def check_cdf_logcdf(distfn, args, msg):
    if False:
        for i in range(10):
            print('nop')
    points = np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    vals = distfn.ppf(points, *args)
    vals = vals[np.isfinite(vals)]
    cdf = distfn.cdf(vals, *args)
    logcdf = distfn.logcdf(vals, *args)
    cdf = cdf[cdf != 0]
    logcdf = logcdf[np.isfinite(logcdf)]
    msg += ' - logcdf-log(cdf) relationship'
    npt.assert_almost_equal(np.log(cdf), logcdf, decimal=7, err_msg=msg)

def check_ppf_broadcast(distfn, arg, msg):
    if False:
        i = 10
        return i + 15
    num_repeats = 5
    args = [] * num_repeats
    if arg:
        args = [np.array([_] * num_repeats) for _ in arg]
    median = distfn.ppf(0.5, *arg)
    medians = distfn.ppf(0.5, *args)
    msg += ' - ppf multiple'
    npt.assert_almost_equal(medians, [median] * num_repeats, decimal=7, err_msg=msg)

def check_distribution_rvs(dist, args, alpha, rvs):
    if False:
        i = 10
        return i + 15
    (D, pval) = stats.kstest(rvs, dist, args=args, N=1000)
    if pval < alpha:
        (D, pval) = stats.kstest(dist, dist, args=args, N=1000)
        npt.assert_(pval > alpha, 'D = ' + str(D) + '; pval = ' + str(pval) + '; alpha = ' + str(alpha) + '\nargs = ' + str(args))

def check_vecentropy(distfn, args):
    if False:
        while True:
            i = 10
    npt.assert_equal(distfn.vecentropy(*args), distfn._entropy(*args))

def check_loc_scale(distfn, arg, m, v, msg):
    if False:
        for i in range(10):
            print('nop')
    (loc, scale) = (np.array([10.0, 20.0]), np.array([10.0, 20.0]))
    (mt, vt) = distfn.stats(*arg, loc=loc, scale=scale)
    npt.assert_allclose(m * scale + loc, mt)
    npt.assert_allclose(v * scale * scale, vt)

def check_ppf_private(distfn, arg, msg):
    if False:
        i = 10
        return i + 15
    ppfs = distfn._ppf(np.array([0.1, 0.5, 0.9]), *arg)
    npt.assert_(not np.any(np.isnan(ppfs)), msg + 'ppf private is nan')

def check_retrieving_support(distfn, args):
    if False:
        while True:
            i = 10
    (loc, scale) = (1, 2)
    supp = distfn.support(*args)
    supp_loc_scale = distfn.support(*args, loc=loc, scale=scale)
    npt.assert_almost_equal(np.array(supp) * scale + loc, np.array(supp_loc_scale))

def check_fit_args(distfn, arg, rvs, method):
    if False:
        print('Hello World!')
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning, message='The shape parameter of the erlang')
        sup.filter(category=RuntimeWarning, message='floating point number truncated')
        vals = distfn.fit(rvs, method=method)
        vals2 = distfn.fit(rvs, optimizer='powell', method=method)
    npt.assert_(len(vals) == 2 + len(arg))
    npt.assert_(len(vals2) == 2 + len(arg))

def check_fit_args_fix(distfn, arg, rvs, method):
    if False:
        print('Hello World!')
    with np.errstate(all='ignore'), npt.suppress_warnings() as sup:
        sup.filter(category=RuntimeWarning, message='The shape parameter of the erlang')
        vals = distfn.fit(rvs, floc=0, method=method)
        vals2 = distfn.fit(rvs, fscale=1, method=method)
        npt.assert_(len(vals) == 2 + len(arg))
        npt.assert_(vals[-2] == 0)
        npt.assert_(vals2[-1] == 1)
        npt.assert_(len(vals2) == 2 + len(arg))
        if len(arg) > 0:
            vals3 = distfn.fit(rvs, f0=arg[0], method=method)
            npt.assert_(len(vals3) == 2 + len(arg))
            npt.assert_(vals3[0] == arg[0])
        if len(arg) > 1:
            vals4 = distfn.fit(rvs, f1=arg[1], method=method)
            npt.assert_(len(vals4) == 2 + len(arg))
            npt.assert_(vals4[1] == arg[1])
        if len(arg) > 2:
            vals5 = distfn.fit(rvs, f2=arg[2], method=method)
            npt.assert_(len(vals5) == 2 + len(arg))
            npt.assert_(vals5[2] == arg[2])

@pytest.mark.parametrize('method', ['pdf', 'logpdf', 'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', distcont)
def test_methods_with_lists(method, distname, args):
    if False:
        return 10
    dist = getattr(stats, distname)
    f = getattr(dist, method)
    if distname == 'invweibull' and method.startswith('log'):
        x = [1.5, 2]
    else:
        x = [0.1, 0.2]
    shape2 = [[a] * 2 for a in args]
    loc = [0, 0.1]
    scale = [1, 1.01]
    result = f(x, *shape2, loc=loc, scale=scale)
    npt.assert_allclose(result, [f(*v) for v in zip(x, *shape2, loc, scale)], rtol=1e-14, atol=5e-14)

def test_burr_fisk_moment_gh13234_regression():
    if False:
        while True:
            i = 10
    vals0 = stats.burr.moment(1, 5, 4)
    assert isinstance(vals0, float)
    vals1 = stats.fisk.moment(1, 8)
    assert isinstance(vals1, float)

def test_moments_with_array_gh12192_regression():
    if False:
        return 10
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=1)
    expected0 = np.array([1.0, 2.0, 3.0])
    npt.assert_equal(vals0, expected0)
    vals1 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=-1)
    expected1 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals1, expected1)
    vals2 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=[-3, 1, 0])
    expected2 = np.array([np.nan, 2.0, np.nan])
    npt.assert_equal(vals2, expected2)
    vals3 = stats.norm.moment(order=2, loc=0, scale=-4)
    expected3 = np.nan
    npt.assert_equal(vals3, expected3)
    assert isinstance(vals3, expected3.__class__)
    vals4 = stats.norm.moment(order=2, loc=[1, 0, 2], scale=[3, -4, -5])
    expected4 = np.array([10.0, np.nan, np.nan])
    npt.assert_equal(vals4, expected4)
    vals5 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[5.0, -2, 100.0])
    expected5 = np.array([25.0, np.nan, 10000.0])
    npt.assert_equal(vals5, expected5)
    vals6 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[-5.0, -2, -100.0])
    expected6 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals6, expected6)
    vals7 = stats.chi.moment(order=2, df=1, loc=0, scale=0)
    expected7 = np.nan
    npt.assert_equal(vals7, expected7)
    assert isinstance(vals7, expected7.__class__)
    vals8 = stats.chi.moment(order=2, df=[1, 2, 3], loc=0, scale=0)
    expected8 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals8, expected8)
    vals9 = stats.chi.moment(order=2, df=[1, 2, 3], loc=[1.0, 0.0, 2.0], scale=[1.0, -3.0, 0.0])
    expected9 = np.array([3.59576912, np.nan, np.nan])
    npt.assert_allclose(vals9, expected9, rtol=1e-08)
    vals10 = stats.norm.moment(5, [1.0, 2.0], [1.0, 2.0])
    expected10 = np.array([26.0, 832.0])
    npt.assert_allclose(vals10, expected10, rtol=1e-13)
    a = [-1.1, 0, 1, 2.2, np.pi]
    b = [-1.1, 0, 1, 2.2, np.pi]
    loc = [-1.1, 0, np.sqrt(2)]
    scale = [-2.1, 0, 1, 2.2, np.pi]
    a = np.array(a).reshape((-1, 1, 1, 1))
    b = np.array(b).reshape((-1, 1, 1))
    loc = np.array(loc).reshape((-1, 1))
    scale = np.array(scale)
    vals11 = stats.beta.moment(order=2, a=a, b=b, loc=loc, scale=scale)
    (a, b, loc, scale) = np.broadcast_arrays(a, b, loc, scale)
    for i in np.ndenumerate(a):
        with np.errstate(invalid='ignore', divide='ignore'):
            i = i[0]
            expected = stats.beta.moment(order=2, a=a[i], b=b[i], loc=loc[i], scale=scale[i])
            np.testing.assert_equal(vals11[i], expected)

def test_broadcasting_in_moments_gh12192_regression():
    if False:
        for i in range(10):
            print('nop')
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=[[1]])
    expected0 = np.array([[1.0, 2.0, 3.0]])
    npt.assert_equal(vals0, expected0)
    assert vals0.shape == expected0.shape
    vals1 = stats.norm.moment(order=1, loc=np.array([[1], [2], [3]]), scale=[1, 2, 3])
    expected1 = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    npt.assert_equal(vals1, expected1)
    assert vals1.shape == expected1.shape
    vals2 = stats.chi.moment(order=1, df=[1.0, 2.0, 3.0], loc=0.0, scale=1.0)
    expected2 = np.array([0.79788456, 1.25331414, 1.59576912])
    npt.assert_allclose(vals2, expected2, rtol=1e-08)
    assert vals2.shape == expected2.shape
    vals3 = stats.chi.moment(order=1, df=[[1.0], [2.0], [3.0]], loc=[0.0, 1.0, 2.0], scale=[-1.0, 0.0, 3.0])
    expected3 = np.array([[np.nan, np.nan, 4.39365368], [np.nan, np.nan, 5.75994241], [np.nan, np.nan, 6.78730736]])
    npt.assert_allclose(vals3, expected3, rtol=1e-08)
    assert vals3.shape == expected3.shape

def test_kappa3_array_gh13582():
    if False:
        while True:
            i = 10
    shapes = [0.5, 1.5, 2.5, 3.5, 4.5]
    moments = 'mvsk'
    res = np.array([[stats.kappa3.stats(shape, moments=moment) for shape in shapes] for moment in moments])
    res2 = np.array(stats.kappa3.stats(shapes, moments=moments))
    npt.assert_allclose(res, res2)

@pytest.mark.xslow
def test_kappa4_array_gh13582():
    if False:
        for i in range(10):
            print('nop')
    h = np.array([-0.5, 2.5, 3.5, 4.5, -3])
    k = np.array([-0.5, 1, -1.5, 0, 3.5])
    moments = 'mvsk'
    res = np.array([[stats.kappa4.stats(h[i], k[i], moments=moment) for i in range(5)] for moment in moments])
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    npt.assert_allclose(res, res2)
    h = np.array([-1, -1 / 4, -1 / 4, 1, -1, 0])
    k = np.array([1, 1, 1 / 2, -1 / 3, -1, 0])
    res = np.array([[stats.kappa4.stats(h[i], k[i], moments=moment) for i in range(6)] for moment in moments])
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    npt.assert_allclose(res, res2)
    h = np.array([-1, -0.5, 1])
    k = np.array([-1, -0.5, 0, 1])[:, None]
    res2 = np.array(stats.kappa4.stats(h, k, moments=moments))
    assert res2.shape == (4, 4, 3)

def test_frozen_attributes():
    if False:
        for i in range(10):
            print('nop')
    message = "'rv_continuous_frozen' object has no attribute"
    with pytest.raises(AttributeError, match=message):
        stats.norm().pmf
    with pytest.raises(AttributeError, match=message):
        stats.norm().logpmf
    stats.norm.pmf = 'herring'
    frozen_norm = stats.norm()
    assert isinstance(frozen_norm, rv_continuous_frozen)
    delattr(stats.norm, 'pmf')

def test_skewnorm_pdf_gh16038():
    if False:
        print('Hello World!')
    rng = np.random.default_rng(0)
    (x, a) = (-np.inf, 0)
    npt.assert_equal(stats.skewnorm.pdf(x, a), stats.norm.pdf(x))
    (x, a) = (rng.random(size=(3, 3)), rng.random(size=(3, 3)))
    mask = rng.random(size=(3, 3)) < 0.5
    a[mask] = 0
    x_norm = x[mask]
    res = stats.skewnorm.pdf(x, a)
    npt.assert_equal(res[mask], stats.norm.pdf(x_norm))
    npt.assert_equal(res[~mask], stats.skewnorm.pdf(x[~mask], a[~mask]))
scalar_out = [['rvs', []], ['pdf', [0]], ['logpdf', [0]], ['cdf', [0]], ['logcdf', [0]], ['sf', [0]], ['logsf', [0]], ['ppf', [0]], ['isf', [0]], ['moment', [1]], ['entropy', []], ['expect', []], ['median', []], ['mean', []], ['std', []], ['var', []]]
scalars_out = [['interval', [0.95]], ['support', []], ['stats', ['mv']]]

@pytest.mark.parametrize('case', scalar_out + scalars_out)
def test_scalar_for_scalar(case):
    if False:
        for i in range(10):
            print('nop')
    (method_name, args) = case
    method = getattr(stats.norm(), method_name)
    res = method(*args)
    if case in scalar_out:
        assert isinstance(res, np.number)
    else:
        assert isinstance(res[0], np.number)
        assert isinstance(res[1], np.number)

def test_scalar_for_scalar2():
    if False:
        i = 10
        return i + 15
    res = stats.norm.fit([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    res = stats.norm.fit_loc_scale([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    res = stats.norm.nnlf((0, 1), [1, 2, 3])
    assert isinstance(res, np.number)