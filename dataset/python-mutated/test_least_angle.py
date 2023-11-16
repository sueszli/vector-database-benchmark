import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lars, LarsCV, LassoLars, LassoLarsCV, LassoLarsIC, lars_path
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils._testing import TempMemmap, assert_allclose, assert_array_almost_equal, ignore_warnings
diabetes = datasets.load_diabetes()
(X, y) = (diabetes.data, diabetes.target)
G = np.dot(X.T, X)
Xy = np.dot(X.T, y)
n_samples = y.size
filterwarnings_normalize = pytest.mark.filterwarnings("ignore:'normalize' was deprecated")

@pytest.mark.parametrize('LeastAngleModel', [Lars, LassoLars, LarsCV, LassoLarsCV, LassoLarsIC])
@pytest.mark.parametrize('normalize, n_warnings', [(True, 1), (False, 1), ('deprecated', 0)])
def test_assure_warning_when_normalize(LeastAngleModel, normalize, n_warnings):
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    y = rng.rand(n_samples)
    model = LeastAngleModel(normalize=normalize)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always', FutureWarning)
        model.fit(X, y)
    assert len([w.message for w in rec]) == n_warnings

def test_simple():
    if False:
        while True:
            i = 10
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        (_, _, coef_path_) = linear_model.lars_path(X, y, method='lar', verbose=10)
        sys.stdout = old_stdout
        for (i, coef_) in enumerate(coef_path_.T):
            res = y - np.dot(X, coef_)
            cov = np.dot(X.T, res)
            C = np.max(abs(cov))
            eps = 0.001
            ocur = len(cov[C - eps < abs(cov)])
            if i < X.shape[1]:
                assert ocur == i + 1
            else:
                assert ocur == X.shape[1]
    finally:
        sys.stdout = old_stdout

def test_simple_precomputed():
    if False:
        print('Hello World!')
    (_, _, coef_path_) = linear_model.lars_path(X, y, Gram=G, method='lar')
    for (i, coef_) in enumerate(coef_path_.T):
        res = y - np.dot(X, coef_)
        cov = np.dot(X.T, res)
        C = np.max(abs(cov))
        eps = 0.001
        ocur = len(cov[C - eps < abs(cov)])
        if i < X.shape[1]:
            assert ocur == i + 1
        else:
            assert ocur == X.shape[1]

def _assert_same_lars_path_result(output1, output2):
    if False:
        return 10
    assert len(output1) == len(output2)
    for (o1, o2) in zip(output1, output2):
        assert_allclose(o1, o2)

@pytest.mark.parametrize('method', ['lar', 'lasso'])
@pytest.mark.parametrize('return_path', [True, False])
def test_lars_path_gram_equivalent(method, return_path):
    if False:
        print('Hello World!')
    _assert_same_lars_path_result(linear_model.lars_path_gram(Xy=Xy, Gram=G, n_samples=n_samples, method=method, return_path=return_path), linear_model.lars_path(X, y, Gram=G, method=method, return_path=return_path))

def test_x_none_gram_none_raises_value_error():
    if False:
        for i in range(10):
            print('nop')
    Xy = np.dot(X.T, y)
    with pytest.raises(ValueError, match='X and Gram cannot both be unspecified'):
        linear_model.lars_path(None, y, Gram=None, Xy=Xy)

def test_all_precomputed():
    if False:
        return 10
    G = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    for method in ('lar', 'lasso'):
        output = linear_model.lars_path(X, y, method=method)
        output_pre = linear_model.lars_path(X, y, Gram=G, Xy=Xy, method=method)
        for (expected, got) in zip(output, output_pre):
            assert_array_almost_equal(expected, got)

@filterwarnings_normalize
@pytest.mark.filterwarnings('ignore: `rcond` parameter will change')
def test_lars_lstsq():
    if False:
        for i in range(10):
            print('nop')
    X1 = 3 * X
    clf = linear_model.LassoLars(alpha=0.0)
    clf.fit(X1, y)
    coef_lstsq = np.linalg.lstsq(X1, y, rcond=None)[0]
    assert_array_almost_equal(clf.coef_, coef_lstsq)

@pytest.mark.filterwarnings('ignore:`rcond` parameter will change')
def test_lasso_gives_lstsq_solution():
    if False:
        return 10
    (_, _, coef_path_) = linear_model.lars_path(X, y, method='lasso')
    coef_lstsq = np.linalg.lstsq(X, y)[0]
    assert_array_almost_equal(coef_lstsq, coef_path_[:, -1])

def test_collinearity():
    if False:
        print('Hello World!')
    X = np.array([[3.0, 3.0, 1.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0]])
    y = np.array([1.0, 0.0, 0])
    rng = np.random.RandomState(0)
    f = ignore_warnings
    (_, _, coef_path_) = f(linear_model.lars_path)(X, y, alpha_min=0.01)
    assert not np.isnan(coef_path_).any()
    residual = np.dot(X, coef_path_[:, -1]) - y
    assert (residual ** 2).sum() < 1.0
    n_samples = 10
    X = rng.rand(n_samples, 5)
    y = np.zeros(n_samples)
    (_, _, coef_path_) = linear_model.lars_path(X, y, Gram='auto', copy_X=False, copy_Gram=False, alpha_min=0.0, method='lasso', verbose=0, max_iter=500)
    assert_array_almost_equal(coef_path_, np.zeros_like(coef_path_))

def test_no_path():
    if False:
        while True:
            i = 10
    (alphas_, _, coef_path_) = linear_model.lars_path(X, y, method='lar')
    (alpha_, _, coef) = linear_model.lars_path(X, y, method='lar', return_path=False)
    assert_array_almost_equal(coef, coef_path_[:, -1])
    assert alpha_ == alphas_[-1]

def test_no_path_precomputed():
    if False:
        print('Hello World!')
    (alphas_, _, coef_path_) = linear_model.lars_path(X, y, method='lar', Gram=G)
    (alpha_, _, coef) = linear_model.lars_path(X, y, method='lar', Gram=G, return_path=False)
    assert_array_almost_equal(coef, coef_path_[:, -1])
    assert alpha_ == alphas_[-1]

def test_no_path_all_precomputed():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = (3 * diabetes.data, diabetes.target)
    G = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    (alphas_, _, coef_path_) = linear_model.lars_path(X, y, method='lasso', Xy=Xy, Gram=G, alpha_min=0.9)
    (alpha_, _, coef) = linear_model.lars_path(X, y, method='lasso', Gram=G, Xy=Xy, alpha_min=0.9, return_path=False)
    assert_array_almost_equal(coef, coef_path_[:, -1])
    assert alpha_ == alphas_[-1]

@filterwarnings_normalize
@pytest.mark.parametrize('classifier', [linear_model.Lars, linear_model.LarsCV, linear_model.LassoLarsIC])
def test_lars_precompute(classifier):
    if False:
        print('Hello World!')
    G = np.dot(X.T, X)
    clf = classifier(precompute=G)
    output_1 = ignore_warnings(clf.fit)(X, y).coef_
    for precompute in [True, False, 'auto', None]:
        clf = classifier(precompute=precompute)
        output_2 = clf.fit(X, y).coef_
        assert_array_almost_equal(output_1, output_2, decimal=8)

def test_singular_matrix():
    if False:
        while True:
            i = 10
    X1 = np.array([[1, 1.0], [1.0, 1.0]])
    y1 = np.array([1, 1])
    (_, _, coef_path) = linear_model.lars_path(X1, y1)
    assert_array_almost_equal(coef_path.T, [[0, 0], [1, 0]])

def test_rank_deficient_design():
    if False:
        print('Hello World!')
    y = [5, 0, 5]
    for X in ([[5, 0], [0, 5], [10, 10]], [[10, 10, 0], [1e-32, 0, 0], [0, 0, 1]]):
        lars = linear_model.LassoLars(0.1)
        coef_lars_ = lars.fit(X, y).coef_
        obj_lars = 1.0 / (2.0 * 3.0) * linalg.norm(y - np.dot(X, coef_lars_)) ** 2 + 0.1 * linalg.norm(coef_lars_, 1)
        coord_descent = linear_model.Lasso(0.1, tol=1e-06)
        coef_cd_ = coord_descent.fit(X, y).coef_
        obj_cd = 1.0 / (2.0 * 3.0) * linalg.norm(y - np.dot(X, coef_cd_)) ** 2 + 0.1 * linalg.norm(coef_cd_, 1)
        assert obj_lars < obj_cd * (1.0 + 1e-08)

def test_lasso_lars_vs_lasso_cd():
    if False:
        print('Hello World!')
    X = 3 * diabetes.data
    (alphas, _, lasso_path) = linear_model.lars_path(X, y, method='lasso')
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08)
    for (c, a) in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01
    for alpha in np.linspace(0.01, 1 - 0.01, 20):
        clf1 = linear_model.LassoLars(alpha=alpha).fit(X, y)
        clf2 = linear_model.Lasso(alpha=alpha, tol=1e-08).fit(X, y)
        err = linalg.norm(clf1.coef_ - clf2.coef_)
        assert err < 0.001
    X = diabetes.data
    X = X - X.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    (alphas, _, lasso_path) = linear_model.lars_path(X, y, method='lasso')
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08)
    for (c, a) in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01

@filterwarnings_normalize
def test_lasso_lars_vs_lasso_cd_early_stopping():
    if False:
        i = 10
        return i + 15
    alphas_min = [10, 0.9, 0.0001]
    X = diabetes.data
    for alpha_min in alphas_min:
        (alphas, _, lasso_path) = linear_model.lars_path(X, y, method='lasso', alpha_min=alpha_min)
        lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08)
        lasso_cd.alpha = alphas[-1]
        lasso_cd.fit(X, y)
        error = linalg.norm(lasso_path[:, -1] - lasso_cd.coef_)
        assert error < 0.01
    X = diabetes.data - diabetes.data.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    for alpha_min in alphas_min:
        (alphas, _, lasso_path) = linear_model.lars_path(X, y, method='lasso', alpha_min=alpha_min)
        lasso_cd = linear_model.Lasso(tol=1e-08)
        lasso_cd.alpha = alphas[-1]
        lasso_cd.fit(X, y)
        error = linalg.norm(lasso_path[:, -1] - lasso_cd.coef_)
        assert error < 0.01

@filterwarnings_normalize
def test_lasso_lars_path_length():
    if False:
        for i in range(10):
            print('nop')
    lasso = linear_model.LassoLars()
    lasso.fit(X, y)
    lasso2 = linear_model.LassoLars(alpha=lasso.alphas_[2])
    lasso2.fit(X, y)
    assert_array_almost_equal(lasso.alphas_[:3], lasso2.alphas_)
    assert np.all(np.diff(lasso.alphas_) < 0)

def test_lasso_lars_vs_lasso_cd_ill_conditioned():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(42)
    (n, m) = (70, 100)
    k = 5
    X = rng.randn(n, m)
    w = np.zeros((m, 1))
    i = np.arange(0, m)
    rng.shuffle(i)
    supp = i[:k]
    w[supp] = np.sign(rng.randn(k, 1)) * (rng.rand(k, 1) + 1)
    y = np.dot(X, w)
    sigma = 0.2
    y += sigma * rng.rand(*y.shape)
    y = y.squeeze()
    (lars_alphas, _, lars_coef) = linear_model.lars_path(X, y, method='lasso')
    (_, lasso_coef2, _) = linear_model.lasso_path(X, y, alphas=lars_alphas, tol=1e-06)
    assert_array_almost_equal(lars_coef, lasso_coef2, decimal=1)

def test_lasso_lars_vs_lasso_cd_ill_conditioned2():
    if False:
        for i in range(10):
            print('nop')
    X = [[1e+20, 1e+20, 0], [-1e-32, 0, 0], [1, 1, 1]]
    y = [10, 10, 1]
    alpha = 0.0001

    def objective_function(coef):
        if False:
            i = 10
            return i + 15
        return 1.0 / (2.0 * len(X)) * linalg.norm(y - np.dot(X, coef)) ** 2 + alpha * linalg.norm(coef, 1)
    lars = linear_model.LassoLars(alpha=alpha)
    warning_message = 'Regressors in active set degenerate.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        lars.fit(X, y)
    lars_coef_ = lars.coef_
    lars_obj = objective_function(lars_coef_)
    coord_descent = linear_model.Lasso(alpha=alpha, tol=0.0001)
    cd_coef_ = coord_descent.fit(X, y).coef_
    cd_obj = objective_function(cd_coef_)
    assert lars_obj < cd_obj * (1.0 + 1e-08)

@filterwarnings_normalize
def test_lars_add_features():
    if False:
        while True:
            i = 10
    n = 5
    H = 1.0 / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
    clf = linear_model.Lars(fit_intercept=False).fit(H, np.arange(n))
    assert np.all(np.isfinite(clf.coef_))

@filterwarnings_normalize
def test_lars_n_nonzero_coefs(verbose=False):
    if False:
        for i in range(10):
            print('nop')
    lars = linear_model.Lars(n_nonzero_coefs=6, verbose=verbose)
    lars.fit(X, y)
    assert len(lars.coef_.nonzero()[0]) == 6
    assert len(lars.alphas_) == 7

@filterwarnings_normalize
@ignore_warnings
def test_multitarget():
    if False:
        for i in range(10):
            print('nop')
    Y = np.vstack([y, y ** 2]).T
    n_targets = Y.shape[1]
    estimators = [linear_model.LassoLars(), linear_model.Lars(), linear_model.LassoLars(fit_intercept=False), linear_model.Lars(fit_intercept=False)]
    for estimator in estimators:
        estimator.fit(X, Y)
        Y_pred = estimator.predict(X)
        (alphas, active, coef, path) = (estimator.alphas_, estimator.active_, estimator.coef_, estimator.coef_path_)
        for k in range(n_targets):
            estimator.fit(X, Y[:, k])
            y_pred = estimator.predict(X)
            assert_array_almost_equal(alphas[k], estimator.alphas_)
            assert_array_almost_equal(active[k], estimator.active_)
            assert_array_almost_equal(coef[k], estimator.coef_)
            assert_array_almost_equal(path[k], estimator.coef_path_)
            assert_array_almost_equal(Y_pred[:, k], y_pred)

@filterwarnings_normalize
def test_lars_cv():
    if False:
        for i in range(10):
            print('nop')
    old_alpha = 0
    lars_cv = linear_model.LassoLarsCV()
    for length in (400, 200, 100):
        X = diabetes.data[:length]
        y = diabetes.target[:length]
        lars_cv.fit(X, y)
        np.testing.assert_array_less(old_alpha, lars_cv.alpha_)
        old_alpha = lars_cv.alpha_
    assert not hasattr(lars_cv, 'n_nonzero_coefs')

def test_lars_cv_max_iter(recwarn):
    if False:
        while True:
            i = 10
    warnings.simplefilter('always')
    with np.errstate(divide='raise', invalid='raise'):
        X = diabetes.data
        y = diabetes.target
        rng = np.random.RandomState(42)
        x = rng.randn(len(y))
        X = diabetes.data
        X = np.c_[X, x, x]
        X = StandardScaler().fit_transform(X)
        lars_cv = linear_model.LassoLarsCV(max_iter=5, cv=5)
        lars_cv.fit(X, y)
    recorded_warnings = [str(w) for w in recwarn]
    assert len(recorded_warnings) == 0

def test_lasso_lars_ic():
    if False:
        i = 10
        return i + 15
    lars_bic = linear_model.LassoLarsIC('bic')
    lars_aic = linear_model.LassoLarsIC('aic')
    rng = np.random.RandomState(42)
    X = diabetes.data
    X = np.c_[X, rng.randn(X.shape[0], 5)]
    X = StandardScaler().fit_transform(X)
    lars_bic.fit(X, y)
    lars_aic.fit(X, y)
    nonzero_bic = np.where(lars_bic.coef_)[0]
    nonzero_aic = np.where(lars_aic.coef_)[0]
    assert lars_bic.alpha_ > lars_aic.alpha_
    assert len(nonzero_bic) < len(nonzero_aic)
    assert np.max(nonzero_bic) < diabetes.data.shape[1]

def test_lars_path_readonly_data():
    if False:
        for i in range(10):
            print('nop')
    splitted_data = train_test_split(X, y, random_state=42)
    with TempMemmap(splitted_data) as (X_train, X_test, y_train, y_test):
        _lars_path_residues(X_train, y_train, X_test, y_test, copy=False)

def test_lars_path_positive_constraint():
    if False:
        print('Hello World!')
    err_msg = "Positive constraint not supported for 'lar' coding method."
    with pytest.raises(ValueError, match=err_msg):
        linear_model.lars_path(diabetes['data'], diabetes['target'], method='lar', positive=True)
    method = 'lasso'
    (_, _, coefs) = linear_model.lars_path(X, y, return_path=True, method=method, positive=False)
    assert coefs.min() < 0
    (_, _, coefs) = linear_model.lars_path(X, y, return_path=True, method=method, positive=True)
    assert coefs.min() >= 0
default_parameter = {'fit_intercept': False}
estimator_parameter_map = {'LassoLars': {'alpha': 0.1}, 'LassoLarsCV': {}, 'LassoLarsIC': {}}

@filterwarnings_normalize
def test_estimatorclasses_positive_constraint():
    if False:
        while True:
            i = 10
    default_parameter = {'fit_intercept': False}
    estimator_parameter_map = {'LassoLars': {'alpha': 0.1}, 'LassoLarsCV': {}, 'LassoLarsIC': {}}
    for estname in estimator_parameter_map:
        params = default_parameter.copy()
        params.update(estimator_parameter_map[estname])
        estimator = getattr(linear_model, estname)(positive=False, **params)
        estimator.fit(X, y)
        assert estimator.coef_.min() < 0
        estimator = getattr(linear_model, estname)(positive=True, **params)
        estimator.fit(X, y)
        assert min(estimator.coef_) >= 0

def test_lasso_lars_vs_lasso_cd_positive():
    if False:
        print('Hello World!')
    X = 3 * diabetes.data
    (alphas, _, lasso_path) = linear_model.lars_path(X, y, method='lasso', positive=True)
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08, positive=True)
    for (c, a) in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01
    for alpha in np.linspace(0.6, 1 - 0.01, 20):
        clf1 = linear_model.LassoLars(fit_intercept=False, alpha=alpha, positive=True).fit(X, y)
        clf2 = linear_model.Lasso(fit_intercept=False, alpha=alpha, tol=1e-08, positive=True).fit(X, y)
        err = linalg.norm(clf1.coef_ - clf2.coef_)
        assert err < 0.001
    X = diabetes.data - diabetes.data.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    (alphas, _, lasso_path) = linear_model.lars_path(X, y, method='lasso', positive=True)
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08, positive=True)
    for (c, a) in zip(lasso_path.T[:-1], alphas[:-1]):
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01

def test_lasso_lars_vs_R_implementation():
    if False:
        while True:
            i = 10
    y = np.array([-6.45006793, -3.51251449, -8.52445396, 6.12277822, -19.42109366])
    x = np.array([[0.47299829, 0, 0, 0, 0], [0.08239882, 0.85784863, 0, 0, 0], [0.30114139, -0.07501577, 0.80895216, 0, 0], [-0.01460346, -0.1015233, 0.0407278, 0.80338378, 0], [-0.69363927, 0.06754067, 0.18064514, -0.0803561, 0.40427291]])
    X = x.T
    r = np.array([[0, 0, 0, 0, 0, -79.81036280949903, -83.52878873278283, -83.77765373919071, -83.78415693288893, -84.03339059175666], [0, 0, 0, 0, -0.476624256777266, 0, 0, 0, 0, 0.025219751009936], [0, -3.577397088285891, -4.702795355871871, -7.016748621359461, -7.614898471899412, -0.336938391359179, 0, 0, 0.001213370600853, 0.048162321585148], [0, 0, 0, 2.231558436628169, 2.723267514525966, 2.811549786389614, 2.813766976061531, 2.817462468949557, 2.817368178703816, 2.816221090636795], [0, 0, -1.218422599914637, -3.457726183014808, -4.02130452206071, -45.827461592423745, -47.776608869312305, -47.9115616107464, -47.914845922736234, -48.03956233426572]])
    model_lasso_lars = linear_model.LassoLars(alpha=0, fit_intercept=False)
    model_lasso_lars.fit(X, y)
    skl_betas = model_lasso_lars.coef_path_
    assert_array_almost_equal(r, skl_betas, decimal=12)

@filterwarnings_normalize
@pytest.mark.parametrize('copy_X', [True, False])
def test_lasso_lars_copyX_behaviour(copy_X):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that user input regarding copy_X is not being overridden (it was until\n    at least version 0.21)\n\n    '
    lasso_lars = LassoLarsIC(copy_X=copy_X, precompute=False)
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    y = X[:, 2]
    lasso_lars.fit(X, y)
    assert copy_X == np.array_equal(X, X_copy)

@filterwarnings_normalize
@pytest.mark.parametrize('copy_X', [True, False])
def test_lasso_lars_fit_copyX_behaviour(copy_X):
    if False:
        return 10
    '\n    Test that user input to .fit for copy_X overrides default __init__ value\n\n    '
    lasso_lars = LassoLarsIC(precompute=False)
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    y = X[:, 2]
    lasso_lars.fit(X, y, copy_X=copy_X)
    assert copy_X == np.array_equal(X, X_copy)

@filterwarnings_normalize
@pytest.mark.parametrize('est', (LassoLars(alpha=0.001), Lars()))
def test_lars_with_jitter(est):
    if False:
        print('Hello World!')
    X = np.array([[0.0, 0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0]])
    y = [-2.5, -2.5]
    expected_coef = [0, 2.5, 0, 2.5, 0]
    est.set_params(fit_intercept=False)
    est_jitter = clone(est).set_params(jitter=1e-07, random_state=0)
    est.fit(X, y)
    est_jitter.fit(X, y)
    assert np.mean((est.coef_ - est_jitter.coef_) ** 2) > 0.1
    np.testing.assert_allclose(est_jitter.coef_, expected_coef, rtol=0.001)

def test_X_none_gram_not_none():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='X cannot be None if Gram is not None'):
        lars_path(X=None, y=np.array([1]), Gram=True)

def test_copy_X_with_auto_gram():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(42)
    X = rng.rand(6, 6)
    y = rng.rand(6)
    X_before = X.copy()
    linear_model.lars_path(X, y, Gram='auto', copy_X=True, method='lasso')
    assert_allclose(X, X_before)

@pytest.mark.parametrize('LARS, has_coef_path, args', ((Lars, True, {}), (LassoLars, True, {}), (LassoLarsIC, False, {}), (LarsCV, True, {}), (LassoLarsCV, True, {'max_iter': 5})))
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@filterwarnings_normalize
def test_lars_dtype_match(LARS, has_coef_path, args, dtype):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    X = rng.rand(20, 6).astype(dtype)
    y = rng.rand(20).astype(dtype)
    model = LARS(**args)
    model.fit(X, y)
    assert model.coef_.dtype == dtype
    if has_coef_path:
        assert model.coef_path_.dtype == dtype
    assert model.intercept_.dtype == dtype

@pytest.mark.parametrize('LARS, has_coef_path, args', ((Lars, True, {}), (LassoLars, True, {}), (LassoLarsIC, False, {}), (LarsCV, True, {}), (LassoLarsCV, True, {'max_iter': 5})))
@filterwarnings_normalize
def test_lars_numeric_consistency(LARS, has_coef_path, args):
    if False:
        return 10
    rtol = 1e-05
    atol = 1e-05
    rng = np.random.RandomState(0)
    X_64 = rng.rand(10, 6)
    y_64 = rng.rand(10)
    model_64 = LARS(**args).fit(X_64, y_64)
    model_32 = LARS(**args).fit(X_64.astype(np.float32), y_64.astype(np.float32))
    assert_allclose(model_64.coef_, model_32.coef_, rtol=rtol, atol=atol)
    if has_coef_path:
        assert_allclose(model_64.coef_path_, model_32.coef_path_, rtol=rtol, atol=atol)
    assert_allclose(model_64.intercept_, model_32.intercept_, rtol=rtol, atol=atol)

@pytest.mark.parametrize('criterion', ['aic', 'bic'])
def test_lassolarsic_alpha_selection(criterion):
    if False:
        i = 10
        return i + 15
    'Check that we properly compute the AIC and BIC score.\n\n    In this test, we reproduce the example of the Fig. 2 of Zou et al.\n    (reference [1] in LassoLarsIC) In this example, only 7 features should be\n    selected.\n    '
    model = make_pipeline(StandardScaler(), LassoLarsIC(criterion=criterion))
    model.fit(X, y)
    best_alpha_selected = np.argmin(model[-1].criterion_)
    assert best_alpha_selected == 7

@pytest.mark.parametrize('fit_intercept', [True, False])
def test_lassolarsic_noise_variance(fit_intercept):
    if False:
        return 10
    'Check the behaviour when `n_samples` < `n_features` and that one needs\n    to provide the noise variance.'
    rng = np.random.RandomState(0)
    (X, y) = datasets.make_regression(n_samples=10, n_features=11 - fit_intercept, random_state=rng)
    model = make_pipeline(StandardScaler(), LassoLarsIC(fit_intercept=fit_intercept))
    err_msg = 'You are using LassoLarsIC in the case where the number of samples is smaller than the number of features'
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y)
    model.set_params(lassolarsic__noise_variance=1.0)
    model.fit(X, y).predict(X)