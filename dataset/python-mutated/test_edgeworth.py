import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import _faa_di_bruno_partitions, cumulant_from_moments, ExpandedNormal

class TestFaaDiBruno:

    def test_neg_arg(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, _faa_di_bruno_partitions, -1)
        assert_raises(ValueError, _faa_di_bruno_partitions, 0)

    def test_small_vals(self):
        if False:
            for i in range(10):
                print('nop')
        for n in range(1, 5):
            for ks in _faa_di_bruno_partitions(n):
                lhs = sum((m * k for (m, k) in ks))
                assert_equal(lhs, n)

def _norm_moment(n):
    if False:
        i = 10
        return i + 15
    return (1 - n % 2) * factorial2(n - 1)

def _norm_cumulant(n):
    if False:
        return 10
    try:
        return {1: 0, 2: 1}[n]
    except KeyError:
        return 0

def _chi2_moment(n, df):
    if False:
        print('Hello World!')
    return 2 ** n * gamma(n + df / 2.0) / gamma(df / 2.0)

def _chi2_cumulant(n, df):
    if False:
        while True:
            i = 10
    assert n > 0
    return 2 ** (n - 1) * factorial(n - 1) * df

class TestCumulants:

    def test_badvalues(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 0)
        assert_raises(ValueError, cumulant_from_moments, [1, 2, 3], 4)

    def test_norm(self):
        if False:
            return 10
        N = 4
        momt = [_norm_moment(j + 1) for j in range(N)]
        for n in range(1, N + 1):
            kappa = cumulant_from_moments(momt, n)
            assert_allclose(kappa, _norm_cumulant(n), atol=1e-12)

    def test_chi2(self):
        if False:
            i = 10
            return i + 15
        N = 4
        df = 8
        momt = [_chi2_moment(j + 1, df) for j in range(N)]
        for n in range(1, N + 1):
            kappa = cumulant_from_moments(momt, n)
            assert_allclose(kappa, _chi2_cumulant(n, df))

class TestExpandedNormal:

    def test_too_few_cumulants(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, ExpandedNormal, [1])

    def test_coefficients(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ne3 = ExpandedNormal([0.0, 1.0, 1.0])
            assert_allclose(ne3._coef, [1.0, 0.0, 0.0, 1.0 / 6])
            ne4 = ExpandedNormal([0.0, 1.0, 1.0, 1.0])
            assert_allclose(ne4._coef, [1.0, 0.0, 0.0, 1.0 / 6, 1.0 / 24, 0.0, 1.0 / 72])
            ne5 = ExpandedNormal([0.0, 1.0, 1.0, 1.0, 1.0])
            assert_allclose(ne5._coef, [1.0, 0.0, 0.0, 1.0 / 6, 1.0 / 24, 1.0 / 120, 1.0 / 72, 1.0 / 144, 0.0, 1.0 / 1296])
            ne33 = ExpandedNormal([0.0, 1.0, 1.0, 0.0])
            assert_allclose(ne33._coef, [1.0, 0.0, 0.0, 1.0 / 6, 0.0, 0.0, 1.0 / 72])

    def test_normal(self):
        if False:
            while True:
                i = 10
        ne2 = ExpandedNormal([3, 4])
        x = np.linspace(-2.0, 2.0, 100)
        assert_allclose(ne2.pdf(x), stats.norm.pdf(x, loc=3, scale=2))

    def test_chi2_moments(self):
        if False:
            return 10
        (N, df) = (6, 15)
        cum = [_chi2_cumulant(n + 1, df) for n in range(N)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ne = ExpandedNormal(cum, name='edgw_chi2')
        assert_allclose([_chi2_moment(n, df) for n in range(N)], [ne.moment(n) for n in range(N)])
        check_pdf(ne, arg=(), msg='')
        check_cdf_ppf(ne, arg=(), msg='')
        check_cdf_sf(ne, arg=(), msg='')
        np.random.seed(765456)
        rvs = ne.rvs(size=500)
        check_distribution_rvs(ne, args=(), alpha=0.01, rvs=rvs)

    def test_pdf_no_roots(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            ne = ExpandedNormal([0, 1])
            ne = ExpandedNormal([0, 1, 0.1, 0.1])

    def test_pdf_has_roots(self):
        if False:
            return 10
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            assert_raises(RuntimeWarning, ExpandedNormal, [0, 1, 101])
DECIMAL = 8

def check_pdf(distfn, arg, msg):
    if False:
        i = 10
        return i + 15
    median = distfn.ppf(0.5, *arg)
    eps = 1e-06
    pdfv = distfn.pdf(median, *arg)
    if pdfv < 0.0001 or pdfv > 10000.0:
        median = median + 0.1
        pdfv = distfn.pdf(median, *arg)
    cdfdiff = (distfn.cdf(median + eps, *arg) - distfn.cdf(median - eps, *arg)) / eps / 2.0
    npt.assert_almost_equal(pdfv, cdfdiff, decimal=DECIMAL, err_msg=msg + ' - cdf-pdf relationship')

def check_cdf_ppf(distfn, arg, msg):
    if False:
        for i in range(10):
            print('nop')
    values = [0.001, 0.5, 0.999]
    npt.assert_almost_equal(distfn.cdf(distfn.ppf(values, *arg), *arg), values, decimal=DECIMAL, err_msg=msg + ' - cdf-ppf roundtrip')

def check_cdf_sf(distfn, arg, msg):
    if False:
        i = 10
        return i + 15
    values = [0.001, 0.5, 0.999]
    npt.assert_almost_equal(distfn.cdf(values, *arg), 1.0 - distfn.sf(values, *arg), decimal=DECIMAL, err_msg=msg + ' - sf+cdf == 1')

def check_distribution_rvs(distfn, args, alpha, rvs):
    if False:
        while True:
            i = 10
    (D, pval) = stats.kstest(rvs, distfn.cdf, args=args, N=1000)
    if pval < alpha:
        (D, pval) = stats.kstest(distfn.rvs, distfn.cdf, args=args, N=1000)
        npt.assert_(pval > alpha, 'D = ' + str(D) + '; pval = ' + str(pval) + '; alpha = ' + str(alpha) + '\nargs = ' + str(args))