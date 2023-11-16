"""script to test expect and moments in distributions.stats method

not written as a test, prints results, renamed to prevent test runner from running it


"""
import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.sppatch import expect_v2
from .distparams import distcont
specialcases = {'ncf': {'ub': 1000}}

def mc2mvsk(args):
    if False:
        while True:
            i = 10
    'convert central moments to mean, variance, skew, kurtosis\n    '
    (mc, mc2, mc3, mc4) = args
    skew = np.divide(mc3, mc2 ** 1.5)
    kurt = np.divide(mc4, mc2 ** 2.0) - 3.0
    return (mc, mc2, skew, kurt)

def mnc2mvsk(args):
    if False:
        for i in range(10):
            print('nop')
    'convert central moments to mean, variance, skew, kurtosis\n    '
    (mnc, mnc2, mnc3, mnc4) = args
    mc = mnc
    mc2 = mnc2 - mnc * mnc
    mc3 = mnc3 - (3 * mc * mc2 + mc ** 3)
    mc4 = mnc4 - (4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4)
    return mc2mvsk((mc, mc2, mc3, mc4))

def mom_nc0(x):
    if False:
        print('Hello World!')
    return 1.0

def mom_nc1(x):
    if False:
        return 10
    return x

def mom_nc2(x):
    if False:
        while True:
            i = 10
    return x * x

def mom_nc3(x):
    if False:
        i = 10
        return i + 15
    return x * x * x

def mom_nc4(x):
    if False:
        return 10
    return np.power(x, 4)
res = {}
distex = []
distlow = []
distok = []
distnonfinite = []

def check_cont_basic():
    if False:
        print('Hello World!')
    for (distname, distargs) in distcont[:]:
        distfn = getattr(stats, distname)
        (m, v, s, k) = distfn.stats(*distargs, **dict(moments='mvsk'))
        st = np.array([m, v, s, k])
        mask = np.isfinite(st)
        if mask.sum() < 4:
            distnonfinite.append(distname)
        print(distname)
        expect = distfn.expect
        expect = lambda *args, **kwds: expect_v2(distfn, *args, **kwds)
        special_kwds = specialcases.get(distname, {})
        mnc0 = expect(mom_nc0, args=distargs, **special_kwds)
        mnc1 = expect(args=distargs, **special_kwds)
        mnc2 = expect(mom_nc2, args=distargs, **special_kwds)
        mnc3 = expect(mom_nc3, args=distargs, **special_kwds)
        mnc4 = expect(mom_nc4, args=distargs, **special_kwds)
        mnc1_lc = expect(args=distargs, loc=1, scale=2, **special_kwds)
        try:
            (me, ve, se, ke) = mnc2mvsk((mnc1, mnc2, mnc3, mnc4))
        except:
            print('exception', mnc1, mnc2, mnc3, mnc4, st)
            (me, ve, se, ke) = [np.nan] * 4
            if mask.size > 0:
                distex.append(distname)
        em = np.array([me, ve, se, ke])
        diff = st[mask] - em[mask]
        print(diff, mnc1_lc - (1 + 2 * mnc1))
        if np.size(diff) > 0 and np.max(np.abs(diff)) > 0.001:
            distlow.append(distname)
        else:
            distok.append(distname)
        res[distname] = [mnc0, st, em, diff, mnc1_lc]

def nct_kurt_bug():
    if False:
        i = 10
        return i + 15
    'test for incorrect kurtosis of nct\n\n    D. Hogben, R. S. Pinkham, M. B. Wilk: The Moments of the Non-Central\n    t-DistributionAuthor(s): Biometrika, Vol. 48, No. 3/4 (Dec., 1961),\n    pp. 465-468\n    '
    from numpy.testing import assert_almost_equal
    mvsk_10_1 = (1.08372, 1.325546, 0.39993, 1.2499424941142943)
    assert_almost_equal(stats.nct.stats(10, 1, moments='mvsk'), mvsk_10_1, decimal=6)
    c1 = np.array([1.08372])
    c2 = np.array([0.075546, 1.25])
    c3 = np.array([0.0297802, 0.580566])
    c4 = np.array([0.0425458, 1.17491, 6.25])
    nc = 1
    mc1 = c1.item()
    mc2 = (c2 * nc ** np.array([2, 0])).sum()
    mc3 = (c3 * nc ** np.array([3, 1])).sum()
    mc4 = c4 = np.array([0.0425458, 1.17491, 6.25])
    mvsk_nc = mc2mvsk((mc1, mc2, mc3, mc4))
if __name__ == '__main__':
    check_cont_basic()
    mean_ = [(k, v[1][0], v[2][0]) for (k, v) in res.items() if np.abs(v[1][0] - v[2][0]) > 1e-06 and np.isfinite(v[1][0])]
    var_ = [(k, v[1][1], v[2][1]) for (k, v) in res.items() if np.abs(v[1][1] - v[2][1]) > 0.01 and np.isfinite(v[1][1])]
    skew = [(k, v[1][2], v[2][2]) for (k, v) in res.items() if np.abs(v[1][2] - v[2][2]) > 0.01 and np.isfinite(v[1][1])]
    kurt = [(k, v[1][3], v[2][3]) for (k, v) in res.items() if np.abs(v[1][3] - v[2][3]) > 0.01 and np.isfinite(v[1][1])]
    from statsmodels.iolib import SimpleTable
    if len(mean_) > 0:
        print('\nMean difference at least 1e-6')
        print(SimpleTable(mean_, headers=['distname', 'diststats', 'expect']))
    print('\nVariance difference at least 1e-2')
    print(SimpleTable(var_, headers=['distname', 'diststats', 'expect']))
    print('\nSkew difference at least 1e-2')
    print(SimpleTable(skew, headers=['distname', 'diststats', 'expect']))
    print('\nKurtosis difference at least 1e-2')
    print(SimpleTable(kurt, headers=['distname', 'diststats', 'expect']))