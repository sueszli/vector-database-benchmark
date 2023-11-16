import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS
dense_X = [[-1, 0], [0, 1], [1, 1], [1, 1]]
Y1 = [0, 1, 1, 1]
Y2 = [2, 1, 0, 0]

@pytest.mark.parametrize('X_container', CSR_CONTAINERS + [np.array])
@pytest.mark.parametrize('loss', ['squared_hinge', 'log'])
@pytest.mark.parametrize('Y_label', ['two-classes', 'multi-class'])
@pytest.mark.parametrize('intercept_label', ['no-intercept', 'fit-intercept'])
def test_l1_min_c(X_container, loss, Y_label, intercept_label):
    if False:
        i = 10
        return i + 15
    Ys = {'two-classes': Y1, 'multi-class': Y2}
    intercepts = {'no-intercept': {'fit_intercept': False}, 'fit-intercept': {'fit_intercept': True, 'intercept_scaling': 10}}
    X = X_container(dense_X)
    Y = Ys[Y_label]
    intercept_params = intercepts[intercept_label]
    check_l1_min_c(X, Y, loss, **intercept_params)

def check_l1_min_c(X, y, loss, fit_intercept=True, intercept_scaling=1.0):
    if False:
        i = 10
        return i + 15
    min_c = l1_min_c(X, y, loss=loss, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)
    clf = {'log': LogisticRegression(penalty='l1', solver='liblinear'), 'squared_hinge': LinearSVC(loss='squared_hinge', penalty='l1', dual=False)}[loss]
    clf.fit_intercept = fit_intercept
    clf.intercept_scaling = intercept_scaling
    clf.C = min_c
    clf.fit(X, y)
    assert (np.asarray(clf.coef_) == 0).all()
    assert (np.asarray(clf.intercept_) == 0).all()
    clf.C = min_c * 1.01
    clf.fit(X, y)
    assert (np.asarray(clf.coef_) != 0).any() or (np.asarray(clf.intercept_) != 0).any()

def test_ill_posed_min_c():
    if False:
        for i in range(10):
            print('nop')
    X = [[0, 0], [0, 0]]
    y = [0, 1]
    with pytest.raises(ValueError):
        l1_min_c(X, y)
_MAX_UNSIGNED_INT = 4294967295

def test_newrand_default():
    if False:
        return 10
    'Test that bounded_rand_int_wrap without seeding respects the range\n\n    Note this test should pass either if executed alone, or in conjunctions\n    with other tests that call set_seed explicit in any order: it checks\n    invariants on the RNG instead of specific values.\n    '
    generated = [bounded_rand_int_wrap(100) for _ in range(10)]
    assert all((0 <= x < 100 for x in generated))
    assert not all((x == generated[0] for x in generated))

@pytest.mark.parametrize('seed, expected', [(0, 54), (_MAX_UNSIGNED_INT, 9)])
def test_newrand_set_seed(seed, expected):
    if False:
        for i in range(10):
            print('nop')
    'Test that `set_seed` produces deterministic results'
    set_seed_wrap(seed)
    generated = bounded_rand_int_wrap(100)
    assert generated == expected

@pytest.mark.parametrize('seed', [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_set_seed_overflow(seed):
    if False:
        i = 10
        return i + 15
    'Test that `set_seed_wrap` is defined for unsigned 32bits ints'
    with pytest.raises(OverflowError):
        set_seed_wrap(seed)

@pytest.mark.parametrize('range_, n_pts', [(_MAX_UNSIGNED_INT, 10000), (100, 25)])
def test_newrand_bounded_rand_int(range_, n_pts):
    if False:
        while True:
            i = 10
    'Test that `bounded_rand_int` follows a uniform distribution'
    set_seed_wrap(42)
    n_iter = 100
    ks_pvals = []
    uniform_dist = stats.uniform(loc=0, scale=range_)
    for _ in range(n_iter):
        sample = [bounded_rand_int_wrap(range_) for _ in range(n_pts)]
        res = stats.kstest(sample, uniform_dist.cdf)
        ks_pvals.append(res.pvalue)
    uniform_p_vals_dist = stats.uniform(loc=0, scale=1)
    res_pvals = stats.kstest(ks_pvals, uniform_p_vals_dist.cdf)
    assert res_pvals.pvalue > 0.05, f'Null hypothesis rejected: generated random numbers are not uniform. Details: the (meta) p-value of the test of uniform distribution of p-values is {res_pvals.pvalue} which is not > 0.05'
    min_10pct_pval = np.percentile(ks_pvals, q=10)
    assert min_10pct_pval > 0.05, f'Null hypothesis rejected: generated random numbers are not uniform. Details: lower 10th quantile p-value of {min_10pct_pval} not > 0.05.'

@pytest.mark.parametrize('range_', [-1, _MAX_UNSIGNED_INT + 1])
def test_newrand_bounded_rand_int_limits(range_):
    if False:
        for i in range(10):
            print('nop')
    'Test that `bounded_rand_int_wrap` is defined for unsigned 32bits ints'
    with pytest.raises(OverflowError):
        bounded_rand_int_wrap(range_)