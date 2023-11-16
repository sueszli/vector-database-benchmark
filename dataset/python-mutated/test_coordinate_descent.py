import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lars, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV, OrthogonalMatchingPursuit, Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, enet_path, lars_path, lasso_path
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import TempMemmap, assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal, ignore_warnings
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS

@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('input_order', ['C', 'F'])
def test_set_order_dense(order, input_order):
    if False:
        print('Hello World!')
    'Check that _set_order returns arrays with promised order.'
    X = np.array([[0], [0], [0]], order=input_order)
    y = np.array([0, 0, 0], order=input_order)
    (X2, y2) = _set_order(X, y, order=order)
    if order == 'C':
        assert X2.flags['C_CONTIGUOUS']
        assert y2.flags['C_CONTIGUOUS']
    elif order == 'F':
        assert X2.flags['F_CONTIGUOUS']
        assert y2.flags['F_CONTIGUOUS']
    if order == input_order:
        assert X is X2
        assert y is y2

@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('input_order', ['C', 'F'])
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_set_order_sparse(order, input_order, coo_container):
    if False:
        for i in range(10):
            print('nop')
    'Check that _set_order returns sparse matrices in promised format.'
    X = coo_container(np.array([[0], [0], [0]]))
    y = coo_container(np.array([0, 0, 0]))
    sparse_format = 'csc' if input_order == 'F' else 'csr'
    X = X.asformat(sparse_format)
    y = X.asformat(sparse_format)
    (X2, y2) = _set_order(X, y, order=order)
    format = 'csc' if order == 'F' else 'csr'
    assert sparse.issparse(X2) and X2.format == format
    assert sparse.issparse(y2) and y2.format == format

def test_lasso_zero():
    if False:
        return 10
    X = [[0], [0], [0]]
    y = [0, 0, 0]
    clf = Lasso(alpha=0.1).fit(X, y)
    pred = clf.predict([[1], [2], [3]])
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)

def test_enet_nonfinite_params():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    n_samples = 10
    fmax = np.finfo(np.float64).max
    X = fmax * rng.uniform(size=(n_samples, 2))
    y = rng.randint(0, 2, size=n_samples)
    clf = ElasticNet(alpha=0.1)
    msg = 'Coordinate descent iterations resulted in non-finite parameter values'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)

def test_lasso_toy():
    if False:
        return 10
    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]
    T = [[2], [3], [4]]
    clf = Lasso(alpha=1e-08)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)
    clf = Lasso(alpha=0.1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.85])
    assert_array_almost_equal(pred, [1.7, 2.55, 3.4])
    assert_almost_equal(clf.dual_gap_, 0)
    clf = Lasso(alpha=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.25])
    assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
    assert_almost_equal(clf.dual_gap_, 0)
    clf = Lasso(alpha=1)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0])
    assert_array_almost_equal(pred, [0, 0, 0])
    assert_almost_equal(clf.dual_gap_, 0)

def test_enet_toy():
    if False:
        while True:
            i = 10
    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]
    T = [[2.0], [3.0], [4.0]]
    clf = ElasticNet(alpha=1e-08, l1_ratio=1.0)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    assert_almost_equal(clf.dual_gap_, 0)
    clf = ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100, precompute=False)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)
    clf.set_params(max_iter=100, precompute=True)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)
    clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
    assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
    assert_almost_equal(clf.dual_gap_, 0)
    clf = ElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.45454], 3)
    assert_array_almost_equal(pred, [0.909, 1.3636, 1.8181], 3)
    assert_almost_equal(clf.dual_gap_, 0)

def test_lasso_dual_gap():
    if False:
        return 10
    '\n    Check that Lasso.dual_gap_ matches its objective formulation, with the\n    datafit normalized by n_samples\n    '
    (X, y, _, _) = build_dataset(n_samples=10, n_features=30)
    n_samples = len(y)
    alpha = 0.01 * np.max(np.abs(X.T @ y)) / n_samples
    clf = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
    w = clf.coef_
    R = y - X @ w
    primal = 0.5 * np.mean(R ** 2) + clf.alpha * np.sum(np.abs(w))
    R /= np.max(np.abs(X.T @ R) / (n_samples * alpha))
    dual = 0.5 * (np.mean(y ** 2) - np.mean((y - R) ** 2))
    assert_allclose(clf.dual_gap_, primal - dual)

def build_dataset(n_samples=50, n_features=200, n_informative_features=10, n_targets=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    build an ill-posed linear regression problem with many noisy features and\n    comparatively few samples\n    '
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return (X, y, X_test, y_test)

def test_lasso_cv():
    if False:
        i = 10
        return i + 15
    (X, y, X_test, y_test) = build_dataset()
    max_iter = 150
    clf = LassoCV(n_alphas=10, eps=0.001, max_iter=max_iter, cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)
    clf = LassoCV(n_alphas=10, eps=0.001, max_iter=max_iter, precompute=True, cv=3)
    clf.fit(X, y)
    assert_almost_equal(clf.alpha_, 0.056, 2)
    lars = LassoLarsCV(max_iter=30, cv=3).fit(X, y)
    assert np.abs(np.searchsorted(clf.alphas_[::-1], lars.alpha_) - np.searchsorted(clf.alphas_[::-1], clf.alpha_)) <= 1
    mse_lars = interpolate.interp1d(lars.cv_alphas_, lars.mse_path_.T)
    np.testing.assert_approx_equal(mse_lars(clf.alphas_[5]).mean(), clf.mse_path_[5].mean(), significant=2)
    assert clf.score(X_test, y_test) > 0.99

def test_lasso_cv_with_some_model_selection():
    if False:
        return 10
    from sklearn import datasets
    from sklearn.model_selection import ShuffleSplit
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    pipe = make_pipeline(StandardScaler(), LassoCV(cv=ShuffleSplit(random_state=0)))
    pipe.fit(X, y)

def test_lasso_cv_positive_constraint():
    if False:
        while True:
            i = 10
    (X, y, X_test, y_test) = build_dataset()
    max_iter = 500
    clf_unconstrained = LassoCV(n_alphas=3, eps=0.1, max_iter=max_iter, cv=2, n_jobs=1)
    clf_unconstrained.fit(X, y)
    assert min(clf_unconstrained.coef_) < 0
    clf_constrained = LassoCV(n_alphas=3, eps=0.1, max_iter=max_iter, positive=True, cv=2, n_jobs=1)
    clf_constrained.fit(X, y)
    assert min(clf_constrained.coef_) >= 0

@pytest.mark.parametrize('alphas, err_type, err_msg', [((1, -1, -100), ValueError, 'alphas\\[1\\] == -1, must be >= 0.0.'), ((-0.1, -1.0, -10.0), ValueError, 'alphas\\[0\\] == -0.1, must be >= 0.0.'), ((1, 1.0, '1'), TypeError, 'alphas\\[2\\] must be an instance of float, not str')])
def test_lassocv_alphas_validation(alphas, err_type, err_msg):
    if False:
        print('Hello World!')
    'Check the `alphas` validation in LassoCV.'
    (n_samples, n_features) = (5, 5)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, n_samples)
    lassocv = LassoCV(alphas=alphas)
    with pytest.raises(err_type, match=err_msg):
        lassocv.fit(X, y)

def _scale_alpha_inplace(estimator, n_samples):
    if False:
        while True:
            i = 10
    'Rescale the parameter alpha from when the estimator is evoked with\n    normalize set to True as if it were evoked in a Pipeline with normalize set\n    to False and with a StandardScaler.\n    '
    if 'alpha' not in estimator.get_params() and 'alphas' not in estimator.get_params():
        return
    if isinstance(estimator, (RidgeCV, RidgeClassifierCV)):
        alphas = np.asarray(estimator.alphas) * n_samples
        return estimator.set_params(alphas=alphas)
    if isinstance(estimator, (Lasso, LassoLars, MultiTaskLasso)):
        alpha = estimator.alpha * np.sqrt(n_samples)
    if isinstance(estimator, (Ridge, RidgeClassifier)):
        alpha = estimator.alpha * n_samples
    if isinstance(estimator, (ElasticNet, MultiTaskElasticNet)):
        if estimator.l1_ratio == 1:
            alpha = estimator.alpha * np.sqrt(n_samples)
        elif estimator.l1_ratio == 0:
            alpha = estimator.alpha * n_samples
        else:
            raise NotImplementedError
    estimator.set_params(alpha=alpha)

@pytest.mark.filterwarnings("ignore:'normalize' was deprecated")
@pytest.mark.parametrize('LinearModel, params', [(LassoLars, {'alpha': 0.1}), (OrthogonalMatchingPursuit, {}), (Lars, {}), (LassoLarsIC, {})])
def test_model_pipeline_same_as_normalize_true(LinearModel, params):
    if False:
        while True:
            i = 10
    model_normalize = LinearModel(normalize=True, fit_intercept=True, **params)
    pipeline = make_pipeline(StandardScaler(), LinearModel(normalize=False, fit_intercept=True, **params))
    is_multitask = model_normalize._get_tags()['multioutput_only']
    (n_samples, n_features) = (100, 2)
    rng = np.random.RandomState(0)
    w = rng.randn(n_features)
    X = rng.randn(n_samples, n_features)
    X += 20
    y = X.dot(w)
    if is_classifier(model_normalize):
        y[y > np.mean(y)] = -1
        y[y > 0] = 1
    if is_multitask:
        y = np.stack((y, y), axis=1)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42)
    _scale_alpha_inplace(pipeline[1], X_train.shape[0])
    model_normalize.fit(X_train, y_train)
    y_pred_normalize = model_normalize.predict(X_test)
    pipeline.fit(X_train, y_train)
    y_pred_standardize = pipeline.predict(X_test)
    assert_allclose(model_normalize.coef_ * pipeline[0].scale_, pipeline[1].coef_)
    assert pipeline[1].intercept_ == pytest.approx(y_train.mean())
    assert model_normalize.intercept_ == pytest.approx(y_train.mean() - model_normalize.coef_.dot(X_train.mean(0)))
    assert_allclose(y_pred_normalize, y_pred_standardize)

@pytest.mark.parametrize('LinearModel, params', [(Lasso, {'tol': 1e-16, 'alpha': 0.1}), (LassoCV, {'tol': 1e-16}), (ElasticNetCV, {}), (RidgeClassifier, {'solver': 'sparse_cg', 'alpha': 0.1}), (ElasticNet, {'tol': 1e-16, 'l1_ratio': 1, 'alpha': 0.01}), (ElasticNet, {'tol': 1e-16, 'l1_ratio': 0, 'alpha': 0.01}), (Ridge, {'solver': 'sparse_cg', 'tol': 1e-12, 'alpha': 0.1}), (LinearRegression, {}), (RidgeCV, {}), (RidgeClassifierCV, {})])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_model_pipeline_same_dense_and_sparse(LinearModel, params, csr_container):
    if False:
        while True:
            i = 10
    model_dense = make_pipeline(StandardScaler(with_mean=False), LinearModel(**params))
    model_sparse = make_pipeline(StandardScaler(with_mean=False), LinearModel(**params))
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 2
    X = rng.randn(n_samples, n_features)
    X[X < 0.1] = 0.0
    X_sparse = csr_container(X)
    y = rng.rand(n_samples)
    if is_classifier(model_dense):
        y = np.sign(y)
    model_dense.fit(X, y)
    model_sparse.fit(X_sparse, y)
    assert_allclose(model_sparse[1].coef_, model_dense[1].coef_)
    y_pred_dense = model_dense.predict(X)
    y_pred_sparse = model_sparse.predict(X_sparse)
    assert_allclose(y_pred_dense, y_pred_sparse)
    assert_allclose(model_dense[1].intercept_, model_sparse[1].intercept_)

def test_lasso_path_return_models_vs_new_return_gives_same_coefficients():
    if False:
        i = 10
        return i + 15
    X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    y = np.array([1, 2, 3.1])
    alphas = [5.0, 1.0, 0.5]
    (alphas_lars, _, coef_path_lars) = lars_path(X, y, method='lasso')
    coef_path_cont_lars = interpolate.interp1d(alphas_lars[::-1], coef_path_lars[:, ::-1])
    (alphas_lasso2, coef_path_lasso2, _) = lasso_path(X, y, alphas=alphas)
    coef_path_cont_lasso = interpolate.interp1d(alphas_lasso2[::-1], coef_path_lasso2[:, ::-1])
    assert_array_almost_equal(coef_path_cont_lasso(alphas), coef_path_cont_lars(alphas), decimal=1)

def test_enet_path():
    if False:
        for i in range(10):
            print('nop')
    (X, y, X_test, y_test) = build_dataset(n_samples=200, n_features=100, n_informative_features=100)
    max_iter = 150
    clf = ElasticNetCV(alphas=[0.01, 0.05, 0.1], eps=0.002, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter)
    ignore_warnings(clf.fit)(X, y)
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    assert clf.l1_ratio_ == min(clf.l1_ratio)
    clf = ElasticNetCV(alphas=[0.01, 0.05, 0.1], eps=0.002, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter, precompute=True)
    ignore_warnings(clf.fit)(X, y)
    assert_almost_equal(clf.alpha_, min(clf.alphas_))
    assert clf.l1_ratio_ == min(clf.l1_ratio)
    assert clf.score(X_test, y_test) > 0.99
    (X, y, X_test, y_test) = build_dataset(n_features=10, n_targets=3)
    clf = MultiTaskElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter)
    ignore_warnings(clf.fit)(X, y)
    assert clf.score(X_test, y_test) > 0.99
    assert clf.coef_.shape == (3, 10)
    (X, y, _, _) = build_dataset(n_features=10)
    clf1 = ElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    clf2 = MultiTaskElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7])
    clf2.fit(X, y[:, np.newaxis])
    assert_almost_equal(clf1.l1_ratio_, clf2.l1_ratio_)
    assert_almost_equal(clf1.alpha_, clf2.alpha_)

def test_path_parameters():
    if False:
        print('Hello World!')
    (X, y, _, _) = build_dataset()
    max_iter = 100
    clf = ElasticNetCV(n_alphas=50, eps=0.001, max_iter=max_iter, l1_ratio=0.5, tol=0.001)
    clf.fit(X, y)
    assert_almost_equal(0.5, clf.l1_ratio)
    assert 50 == clf.n_alphas
    assert 50 == len(clf.alphas_)

def test_warm_start():
    if False:
        i = 10
        return i + 15
    (X, y, _, _) = build_dataset()
    clf = ElasticNet(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, y)
    ignore_warnings(clf.fit)(X, y)
    clf2 = ElasticNet(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, y)
    assert_array_almost_equal(clf2.coef_, clf.coef_)

def test_lasso_alpha_warning():
    if False:
        return 10
    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]
    clf = Lasso(alpha=0)
    warning_message = 'With alpha=0, this algorithm does not converge well. You are advised to use the LinearRegression estimator'
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, Y)

def test_lasso_positive_constraint():
    if False:
        return 10
    X = [[-1], [0], [1]]
    y = [1, 0, -1]
    lasso = Lasso(alpha=0.1, positive=True)
    lasso.fit(X, y)
    assert min(lasso.coef_) >= 0
    lasso = Lasso(alpha=0.1, precompute=True, positive=True)
    lasso.fit(X, y)
    assert min(lasso.coef_) >= 0

def test_enet_positive_constraint():
    if False:
        print('Hello World!')
    X = [[-1], [0], [1]]
    y = [1, 0, -1]
    enet = ElasticNet(alpha=0.1, positive=True)
    enet.fit(X, y)
    assert min(enet.coef_) >= 0

def test_enet_cv_positive_constraint():
    if False:
        print('Hello World!')
    (X, y, X_test, y_test) = build_dataset()
    max_iter = 500
    enetcv_unconstrained = ElasticNetCV(n_alphas=3, eps=0.1, max_iter=max_iter, cv=2, n_jobs=1)
    enetcv_unconstrained.fit(X, y)
    assert min(enetcv_unconstrained.coef_) < 0
    enetcv_constrained = ElasticNetCV(n_alphas=3, eps=0.1, max_iter=max_iter, cv=2, positive=True, n_jobs=1)
    enetcv_constrained.fit(X, y)
    assert min(enetcv_constrained.coef_) >= 0

def test_uniform_targets():
    if False:
        return 10
    enet = ElasticNetCV(n_alphas=3)
    m_enet = MultiTaskElasticNetCV(n_alphas=3)
    lasso = LassoCV(n_alphas=3)
    m_lasso = MultiTaskLassoCV(n_alphas=3)
    models_single_task = (enet, lasso)
    models_multi_task = (m_enet, m_lasso)
    rng = np.random.RandomState(0)
    X_train = rng.random_sample(size=(10, 3))
    X_test = rng.random_sample(size=(10, 3))
    y1 = np.empty(10)
    y2 = np.empty((10, 2))
    for model in models_single_task:
        for y_values in (0, 5):
            y1.fill(y_values)
            assert_array_equal(model.fit(X_train, y1).predict(X_test), y1)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)
    for model in models_multi_task:
        for y_values in (0, 5):
            y2[:, 0].fill(y_values)
            y2[:, 1].fill(2 * y_values)
            assert_array_equal(model.fit(X_train, y2).predict(X_test), y2)
            assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)

def test_multi_task_lasso_and_enet():
    if False:
        for i in range(10):
            print('nop')
    (X, y, X_test, y_test) = build_dataset()
    Y = np.c_[y, y]
    clf = MultiTaskLasso(alpha=1, tol=1e-08).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-05
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])
    clf = MultiTaskElasticNet(alpha=1, tol=1e-08).fit(X, Y)
    assert 0 < clf.dual_gap_ < 1e-05
    assert_array_almost_equal(clf.coef_[0], clf.coef_[1])
    clf = MultiTaskElasticNet(alpha=1.0, tol=1e-08, max_iter=1)
    warning_message = 'Objective did not converge. You might want to increase the number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, Y)

def test_lasso_readonly_data():
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[-1], [0], [1]])
    Y = np.array([-1, 0, 1])
    T = np.array([[2], [3], [4]])
    with TempMemmap((X, Y)) as (X, Y):
        clf = Lasso(alpha=0.5)
        clf.fit(X, Y)
        pred = clf.predict(T)
        assert_array_almost_equal(clf.coef_, [0.25])
        assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
        assert_almost_equal(clf.dual_gap_, 0)

def test_multi_task_lasso_readonly_data():
    if False:
        return 10
    (X, y, X_test, y_test) = build_dataset()
    Y = np.c_[y, y]
    with TempMemmap((X, Y)) as (X, Y):
        Y = np.c_[y, y]
        clf = MultiTaskLasso(alpha=1, tol=1e-08).fit(X, Y)
        assert 0 < clf.dual_gap_ < 1e-05
        assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

def test_enet_multitarget():
    if False:
        for i in range(10):
            print('nop')
    n_targets = 3
    (X, y, _, _) = build_dataset(n_samples=10, n_features=8, n_informative_features=10, n_targets=n_targets)
    estimator = ElasticNet(alpha=0.01)
    estimator.fit(X, y)
    (coef, intercept, dual_gap) = (estimator.coef_, estimator.intercept_, estimator.dual_gap_)
    for k in range(n_targets):
        estimator.fit(X, y[:, k])
        assert_array_almost_equal(coef[k, :], estimator.coef_)
        assert_array_almost_equal(intercept[k], estimator.intercept_)
        assert_array_almost_equal(dual_gap[k], estimator.dual_gap_)

def test_multioutput_enetcv_error():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    y = rng.randn(10, 2)
    clf = ElasticNetCV()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_multitask_enet_and_lasso_cv():
    if False:
        return 10
    (X, y, _, _) = build_dataset(n_features=50, n_targets=3)
    clf = MultiTaskElasticNetCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00556, 3)
    clf = MultiTaskLassoCV(cv=3).fit(X, y)
    assert_almost_equal(clf.alpha_, 0.00278, 3)
    (X, y, _, _) = build_dataset(n_targets=3)
    clf = MultiTaskElasticNetCV(n_alphas=10, eps=0.001, max_iter=100, l1_ratio=[0.3, 0.5], tol=0.001, cv=3)
    clf.fit(X, y)
    assert 0.5 == clf.l1_ratio_
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (2, 10, 3) == clf.mse_path_.shape
    assert (2, 10) == clf.alphas_.shape
    (X, y, _, _) = build_dataset(n_targets=3)
    clf = MultiTaskLassoCV(n_alphas=10, eps=0.001, max_iter=100, tol=0.001, cv=3)
    clf.fit(X, y)
    assert (3, X.shape[1]) == clf.coef_.shape
    assert (3,) == clf.intercept_.shape
    assert (10, 3) == clf.mse_path_.shape
    assert 10 == len(clf.alphas_)

def test_1d_multioutput_enet_and_multitask_enet_cv():
    if False:
        for i in range(10):
            print('nop')
    (X, y, _, _) = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = ElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7])
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskElasticNetCV(n_alphas=5, eps=0.002, l1_ratio=[0.5, 0.7])
    clf1.fit(X, y)
    assert_almost_equal(clf.l1_ratio_, clf1.l1_ratio_)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])

def test_1d_multioutput_lasso_and_multitask_lasso_cv():
    if False:
        i = 10
        return i + 15
    (X, y, _, _) = build_dataset(n_features=10)
    y = y[:, np.newaxis]
    clf = LassoCV(n_alphas=5, eps=0.002)
    clf.fit(X, y[:, 0])
    clf1 = MultiTaskLassoCV(n_alphas=5, eps=0.002)
    clf1.fit(X, y)
    assert_almost_equal(clf.alpha_, clf1.alpha_)
    assert_almost_equal(clf.coef_, clf1.coef_[0])
    assert_almost_equal(clf.intercept_, clf1.intercept_[0])

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_input_dtype_enet_and_lassocv(csr_container):
    if False:
        return 10
    (X, y, _, _) = build_dataset(n_features=10)
    clf = ElasticNetCV(n_alphas=5)
    clf.fit(csr_container(X), y)
    clf1 = ElasticNetCV(n_alphas=5)
    clf1.fit(csr_container(X, dtype=np.float32), y)
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)
    clf = LassoCV(n_alphas=5)
    clf.fit(csr_container(X), y)
    clf1 = LassoCV(n_alphas=5)
    clf1.fit(csr_container(X, dtype=np.float32), y)
    assert_almost_equal(clf.alpha_, clf1.alpha_, decimal=6)
    assert_almost_equal(clf.coef_, clf1.coef_, decimal=6)

def test_elasticnet_precompute_incorrect_gram():
    if False:
        i = 10
        return i + 15
    (X, y, _, _) = build_dataset()
    rng = np.random.RandomState(0)
    X_centered = X - np.average(X, axis=0)
    garbage = rng.standard_normal(X.shape)
    precompute = np.dot(garbage.T, garbage)
    clf = ElasticNet(alpha=0.01, precompute=precompute)
    msg = 'Gram matrix.*did not pass validation.*'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X_centered, y)

def test_elasticnet_precompute_gram_weighted_samples():
    if False:
        return 10
    (X, y, _, _) = build_dataset()
    rng = np.random.RandomState(0)
    sample_weight = rng.lognormal(size=y.shape)
    w_norm = sample_weight * (y.shape / np.sum(sample_weight))
    X_c = X - np.average(X, axis=0, weights=w_norm)
    X_r = X_c * np.sqrt(w_norm)[:, np.newaxis]
    gram = np.dot(X_r.T, X_r)
    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y, sample_weight=sample_weight)
    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y, sample_weight=sample_weight)
    assert_allclose(clf1.coef_, clf2.coef_)

def test_elasticnet_precompute_gram():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(58)
    X = rng.binomial(1, 0.25, (1000, 4)).astype(np.float32)
    y = rng.rand(1000).astype(np.float32)
    X_c = X - np.average(X, axis=0)
    gram = np.dot(X_c.T, X_c)
    clf1 = ElasticNet(alpha=0.01, precompute=gram)
    clf1.fit(X_c, y)
    clf2 = ElasticNet(alpha=0.01, precompute=False)
    clf2.fit(X, y)
    assert_allclose(clf1.coef_, clf2.coef_)

def test_warm_start_convergence():
    if False:
        return 10
    (X, y, _, _) = build_dataset()
    model = ElasticNet(alpha=0.001, tol=0.001).fit(X, y)
    n_iter_reference = model.n_iter_
    assert n_iter_reference > 2
    model.fit(X, y)
    n_iter_cold_start = model.n_iter_
    assert n_iter_cold_start == n_iter_reference
    model.set_params(warm_start=True)
    model.fit(X, y)
    n_iter_warm_start = model.n_iter_
    assert n_iter_warm_start == 1

def test_warm_start_convergence_with_regularizer_decrement():
    if False:
        print('Hello World!')
    (X, y) = load_diabetes(return_X_y=True)
    final_alpha = 1e-05
    low_reg_model = ElasticNet(alpha=final_alpha).fit(X, y)
    high_reg_model = ElasticNet(alpha=final_alpha * 10).fit(X, y)
    assert low_reg_model.n_iter_ > high_reg_model.n_iter_
    warm_low_reg_model = deepcopy(high_reg_model)
    warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
    warm_low_reg_model.fit(X, y)
    assert low_reg_model.n_iter_ > warm_low_reg_model.n_iter_

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_random_descent(csr_container):
    if False:
        while True:
            i = 10
    (X, y, _, _) = build_dataset(n_samples=50, n_features=20)
    clf_cyclic = ElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(X, y)
    clf_random = ElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(X, y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
    clf_cyclic = ElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(X.T, y[:20])
    clf_random = ElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(X.T, y[:20])
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
    clf_cyclic = ElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(csr_container(X), y)
    clf_random = ElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(csr_container(X), y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)
    new_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
    clf_cyclic = MultiTaskElasticNet(selection='cyclic', tol=1e-08)
    clf_cyclic.fit(X, new_y)
    clf_random = MultiTaskElasticNet(selection='random', tol=1e-08, random_state=42)
    clf_random.fit(X, new_y)
    assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
    assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

def test_enet_path_positive():
    if False:
        while True:
            i = 10
    (X, Y, _, _) = build_dataset(n_samples=50, n_features=50, n_targets=2)
    for path in [enet_path, lasso_path]:
        pos_path_coef = path(X, Y[:, 0], positive=True)[1]
        assert np.all(pos_path_coef >= 0)
    for path in [enet_path, lasso_path]:
        with pytest.raises(ValueError):
            path(X, Y, positive=True)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_dense_descent_paths(csr_container):
    if False:
        print('Hello World!')
    (X, y, _, _) = build_dataset(n_samples=50, n_features=20)
    csr = csr_container(X)
    for path in [enet_path, lasso_path]:
        (_, coefs, _) = path(X, y)
        (_, sparse_coefs, _) = path(csr, y)
        assert_array_almost_equal(coefs, sparse_coefs)

@pytest.mark.parametrize('path_func', [enet_path, lasso_path])
def test_path_unknown_parameter(path_func):
    if False:
        i = 10
        return i + 15
    'Check that passing parameter not used by the coordinate descent solver\n    will raise an error.'
    (X, y, _, _) = build_dataset(n_samples=50, n_features=20)
    err_msg = 'Unexpected parameters in params'
    with pytest.raises(ValueError, match=err_msg):
        path_func(X, y, normalize=True, fit_intercept=True)

def test_check_input_false():
    if False:
        return 10
    (X, y, _, _) = build_dataset(n_samples=20, n_features=10)
    X = check_array(X, order='F', dtype='float64')
    y = check_array(X, order='F', dtype='float64')
    clf = ElasticNet(selection='cyclic', tol=1e-08)
    clf.fit(X, y, check_input=False)
    X = check_array(X, order='F', dtype='float32')
    clf.fit(X, y, check_input=False)
    X = check_array(X, order='C', dtype='float64')
    with pytest.raises(ValueError):
        clf.fit(X, y, check_input=False)

@pytest.mark.parametrize('check_input', [True, False])
def test_enet_copy_X_True(check_input):
    if False:
        i = 10
        return i + 15
    (X, y, _, _) = build_dataset()
    X = X.copy(order='F')
    original_X = X.copy()
    enet = ElasticNet(copy_X=True)
    enet.fit(X, y, check_input=check_input)
    assert_array_equal(original_X, X)

def test_enet_copy_X_False_check_input_False():
    if False:
        while True:
            i = 10
    (X, y, _, _) = build_dataset()
    X = X.copy(order='F')
    original_X = X.copy()
    enet = ElasticNet(copy_X=False)
    enet.fit(X, y, check_input=False)
    assert np.any(np.not_equal(original_X, X))

def test_overrided_gram_matrix():
    if False:
        for i in range(10):
            print('nop')
    (X, y, _, _) = build_dataset(n_samples=20, n_features=10)
    Gram = X.T.dot(X)
    clf = ElasticNet(selection='cyclic', tol=1e-08, precompute=Gram)
    warning_message = 'Gram matrix was provided but X was centered to fit intercept, or X was normalized : recomputing Gram matrix.'
    with pytest.warns(UserWarning, match=warning_message):
        clf.fit(X, y)

@pytest.mark.parametrize('model', [ElasticNet, Lasso])
def test_lasso_non_float_y(model):
    if False:
        return 10
    X = [[0, 0], [1, 1], [-1, -1]]
    y = [0, 1, 2]
    y_float = [0.0, 1.0, 2.0]
    clf = model(fit_intercept=False)
    clf.fit(X, y)
    clf_float = model(fit_intercept=False)
    clf_float.fit(X, y_float)
    assert_array_equal(clf.coef_, clf_float.coef_)

def test_enet_float_precision():
    if False:
        return 10
    (X, y, X_test, y_test) = build_dataset(n_samples=20, n_features=10)
    for fit_intercept in [True, False]:
        coef = {}
        intercept = {}
        for dtype in [np.float64, np.float32]:
            clf = ElasticNet(alpha=0.5, max_iter=100, precompute=False, fit_intercept=fit_intercept)
            X = dtype(X)
            y = dtype(y)
            ignore_warnings(clf.fit)(X, y)
            coef['simple', dtype] = clf.coef_
            intercept['simple', dtype] = clf.intercept_
            assert clf.coef_.dtype == dtype
            Gram = X.T.dot(X)
            clf_precompute = ElasticNet(alpha=0.5, max_iter=100, precompute=Gram, fit_intercept=fit_intercept)
            ignore_warnings(clf_precompute.fit)(X, y)
            assert_array_almost_equal(clf.coef_, clf_precompute.coef_)
            assert_array_almost_equal(clf.intercept_, clf_precompute.intercept_)
            multi_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
            clf_multioutput = MultiTaskElasticNet(alpha=0.5, max_iter=100, fit_intercept=fit_intercept)
            clf_multioutput.fit(X, multi_y)
            coef['multi', dtype] = clf_multioutput.coef_
            intercept['multi', dtype] = clf_multioutput.intercept_
            assert clf.coef_.dtype == dtype
        for v in ['simple', 'multi']:
            assert_array_almost_equal(coef[v, np.float32], coef[v, np.float64], decimal=4)
            assert_array_almost_equal(intercept[v, np.float32], intercept[v, np.float64], decimal=4)

def test_enet_l1_ratio():
    if False:
        return 10
    msg = 'Automatic alpha grid generation is not supported for l1_ratio=0. Please supply a grid by providing your estimator with the appropriate `alphas=` argument.'
    X = np.array([[1, 2, 4, 5, 8], [3, 5, 7, 7, 8]]).T
    y = np.array([12, 10, 11, 21, 5])
    with pytest.raises(ValueError, match=msg):
        ElasticNetCV(l1_ratio=0, random_state=42).fit(X, y)
    with pytest.raises(ValueError, match=msg):
        MultiTaskElasticNetCV(l1_ratio=0, random_state=42).fit(X, y[:, None])
    warning_message = 'Coordinate descent without L1 regularization may lead to unexpected results and is discouraged. Set l1_ratio > 0 to add L1 regularization.'
    est = ElasticNetCV(l1_ratio=[0], alphas=[1])
    with pytest.warns(UserWarning, match=warning_message):
        est.fit(X, y)
    alphas = [0.1, 10]
    estkwds = {'alphas': alphas, 'random_state': 42}
    est_desired = ElasticNetCV(l1_ratio=1e-05, **estkwds)
    est = ElasticNetCV(l1_ratio=0, **estkwds)
    with ignore_warnings():
        est_desired.fit(X, y)
        est.fit(X, y)
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)
    est_desired = MultiTaskElasticNetCV(l1_ratio=1e-05, **estkwds)
    est = MultiTaskElasticNetCV(l1_ratio=0, **estkwds)
    with ignore_warnings():
        est.fit(X, y[:, None])
        est_desired.fit(X, y[:, None])
    assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)

def test_coef_shape_not_zero():
    if False:
        i = 10
        return i + 15
    est_no_intercept = Lasso(fit_intercept=False)
    est_no_intercept.fit(np.c_[np.ones(3)], np.ones(3))
    assert est_no_intercept.coef_.shape == (1,)

def test_warm_start_multitask_lasso():
    if False:
        i = 10
        return i + 15
    (X, y, X_test, y_test) = build_dataset()
    Y = np.c_[y, y]
    clf = MultiTaskLasso(alpha=0.1, max_iter=5, warm_start=True)
    ignore_warnings(clf.fit)(X, Y)
    ignore_warnings(clf.fit)(X, Y)
    clf2 = MultiTaskLasso(alpha=0.1, max_iter=10)
    ignore_warnings(clf2.fit)(X, Y)
    assert_array_almost_equal(clf2.coef_, clf.coef_)

@pytest.mark.parametrize('klass, n_classes, kwargs', [(Lasso, 1, dict(precompute=True)), (Lasso, 1, dict(precompute=False)), (MultiTaskLasso, 2, dict()), (MultiTaskLasso, 2, dict())])
def test_enet_coordinate_descent(klass, n_classes, kwargs):
    if False:
        i = 10
        return i + 15
    'Test that a warning is issued if model does not converge'
    clf = klass(max_iter=2, **kwargs)
    n_samples = 5
    n_features = 2
    X = np.ones((n_samples, n_features)) * 1e+50
    y = np.ones((n_samples, n_classes))
    if klass == Lasso:
        y = y.ravel()
    warning_message = 'Objective did not converge. You might want to increase the number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        clf.fit(X, y)

def test_convergence_warnings():
    if False:
        for i in range(10):
            print('nop')
    random_state = np.random.RandomState(0)
    X = random_state.standard_normal((1000, 500))
    y = random_state.standard_normal((1000, 3))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        MultiTaskElasticNet().fit(X, y)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_input_convergence_warning(csr_container):
    if False:
        for i in range(10):
            print('nop')
    (X, y, _, _) = build_dataset(n_samples=1000, n_features=500)
    with pytest.warns(ConvergenceWarning):
        ElasticNet(max_iter=1, tol=0).fit(csr_container(X, dtype=np.float32), y)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        Lasso().fit(csr_container(X, dtype=np.float32), y)

@pytest.mark.parametrize('precompute, inner_precompute', [(True, True), ('auto', False), (False, False)])
def test_lassoCV_does_not_set_precompute(monkeypatch, precompute, inner_precompute):
    if False:
        while True:
            i = 10
    (X, y, _, _) = build_dataset()
    calls = 0

    class LassoMock(Lasso):

        def fit(self, X, y):
            if False:
                while True:
                    i = 10
            super().fit(X, y)
            nonlocal calls
            calls += 1
            assert self.precompute == inner_precompute
    monkeypatch.setattr('sklearn.linear_model._coordinate_descent.Lasso', LassoMock)
    clf = LassoCV(precompute=precompute)
    clf.fit(X, y)
    assert calls > 0

def test_multi_task_lasso_cv_dtype():
    if False:
        while True:
            i = 10
    (n_samples, n_features) = (10, 3)
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(n_samples, n_features))
    X = X.astype(int)
    y = X[:, [0, 0]].copy()
    est = MultiTaskLassoCV(n_alphas=5, fit_intercept=True).fit(X, y)
    assert_array_almost_equal(est.coef_, [[1, 0, 0]] * 2, decimal=3)

@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('alpha', [0.01])
@pytest.mark.parametrize('precompute', [False, True])
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_enet_sample_weight_consistency(fit_intercept, alpha, precompute, sparse_container, global_random_seed):
    if False:
        print('Hello World!')
    'Test that the impact of sample_weight is consistent.\n\n    Note that this test is stricter than the common test\n    check_sample_weights_invariance alone and also tests sparse X.\n    '
    rng = np.random.RandomState(global_random_seed)
    (n_samples, n_features) = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    if sparse_container is not None:
        X = sparse_container(X)
    params = dict(alpha=alpha, fit_intercept=fit_intercept, precompute=precompute, tol=1e-06, l1_ratio=0.5)
    reg = ElasticNet(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = rng.uniform(low=0.01, high=2, size=X.shape[0])
    reg = reg.fit(X, y, sample_weight=sample_weight)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    reg.fit(X, y, sample_weight=np.pi * sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight_0 = sample_weight.copy()
    sample_weight_0[-5:] = 0
    y[-5:] *= 1000
    reg.fit(X, y, sample_weight=sample_weight_0)
    coef_0 = reg.coef_.copy()
    if fit_intercept:
        intercept_0 = reg.intercept_
    reg.fit(X[:-5], y[:-5], sample_weight=sample_weight[:-5])
    assert_allclose(reg.coef_, coef_0, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept_0)
    if sparse_container is not None:
        X2 = sparse.vstack([X, X[:n_samples // 2]], format='csc')
    else:
        X2 = np.concatenate([X, X[:n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[:n_samples // 2]])
    sample_weight_1 = sample_weight.copy()
    sample_weight_1[:n_samples // 2] *= 2
    sample_weight_2 = np.concatenate([sample_weight, sample_weight[:n_samples // 2]], axis=0)
    reg1 = ElasticNet(**params).fit(X, y, sample_weight=sample_weight_1)
    reg2 = ElasticNet(**params).fit(X2, y2, sample_weight=sample_weight_2)
    assert_allclose(reg1.coef_, reg2.coef_, rtol=1e-06)

@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_correctness(fit_intercept, sparse_container):
    if False:
        return 10
    'Test that ElasticNetCV with sample weights gives correct results.'
    rng = np.random.RandomState(42)
    (n_splits, n_samples, n_features) = (3, 10, 5)
    X = rng.rand(n_splits * n_samples, n_features)
    beta = rng.rand(n_features)
    beta[0:2] = 0
    y = X @ beta + rng.rand(n_splits * n_samples)
    sw = np.ones_like(y)
    if sparse_container is not None:
        X = sparse_container(X)
    params = dict(tol=1e-06)
    if fit_intercept:
        alphas = np.linspace(0.001, 0.01, num=91)
    else:
        alphas = np.linspace(0.01, 0.1, num=91)
    sw[:n_samples] = 2
    groups_sw = np.r_[np.full(n_samples, 0), np.full(n_samples, 1), np.full(n_samples, 2)]
    splits_sw = list(LeaveOneGroupOut().split(X, groups=groups_sw))
    reg_sw = ElasticNetCV(alphas=alphas, cv=splits_sw, fit_intercept=fit_intercept, **params)
    reg_sw.fit(X, y, sample_weight=sw)
    if sparse_container is not None:
        X = X.toarray()
    X = np.r_[X[:n_samples], X]
    if sparse_container is not None:
        X = sparse_container(X)
    y = np.r_[y[:n_samples], y]
    groups = np.r_[np.full(2 * n_samples, 0), np.full(n_samples, 1), np.full(n_samples, 2)]
    splits = list(LeaveOneGroupOut().split(X, groups=groups))
    reg = ElasticNetCV(alphas=alphas, cv=splits, fit_intercept=fit_intercept, **params)
    reg.fit(X, y)
    assert alphas[0] < reg.alpha_ < alphas[-1]
    assert reg_sw.alpha_ == reg.alpha_
    assert_allclose(reg_sw.coef_, reg.coef_)
    assert reg_sw.intercept_ == pytest.approx(reg.intercept_)

@pytest.mark.parametrize('sample_weight', [False, True])
def test_enet_cv_grid_search(sample_weight):
    if False:
        while True:
            i = 10
    'Test that ElasticNetCV gives same result as GridSearchCV.'
    (n_samples, n_features) = (200, 10)
    cv = 5
    (X, y) = make_regression(n_samples=n_samples, n_features=n_features, effective_rank=10, n_informative=n_features - 4, noise=10, random_state=0)
    if sample_weight:
        sample_weight = np.linspace(1, 5, num=n_samples)
    else:
        sample_weight = None
    alphas = np.logspace(np.log10(1e-05), np.log10(1), num=10)
    l1_ratios = [0.1, 0.5, 0.9]
    reg = ElasticNetCV(cv=cv, alphas=alphas, l1_ratio=l1_ratios)
    reg.fit(X, y, sample_weight=sample_weight)
    param = {'alpha': alphas, 'l1_ratio': l1_ratios}
    gs = GridSearchCV(estimator=ElasticNet(), param_grid=param, cv=cv, scoring='neg_mean_squared_error').fit(X, y, sample_weight=sample_weight)
    assert reg.l1_ratio_ == pytest.approx(gs.best_params_['l1_ratio'])
    assert reg.alpha_ == pytest.approx(gs.best_params_['alpha'])

@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('l1_ratio', [0, 0.5, 1])
@pytest.mark.parametrize('precompute', [False, True])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS)
def test_enet_cv_sample_weight_consistency(fit_intercept, l1_ratio, precompute, sparse_container):
    if False:
        i = 10
        return i + 15
    'Test that the impact of sample_weight is consistent.'
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = X.sum(axis=1) + rng.rand(n_samples)
    params = dict(l1_ratio=l1_ratio, fit_intercept=fit_intercept, precompute=precompute, tol=1e-06, cv=3)
    if sparse_container is not None:
        X = sparse_container(X)
    if l1_ratio == 0:
        params.pop('l1_ratio', None)
        reg = LassoCV(**params).fit(X, y)
    else:
        reg = ElasticNetCV(**params).fit(X, y)
    coef = reg.coef_.copy()
    if fit_intercept:
        intercept = reg.intercept_
    sample_weight = np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = 123.0
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)
    sample_weight = 2 * np.ones_like(y)
    reg.fit(X, y, sample_weight=sample_weight)
    assert_allclose(reg.coef_, coef, rtol=1e-06)
    if fit_intercept:
        assert_allclose(reg.intercept_, intercept)

@pytest.mark.parametrize('estimator', [ElasticNetCV, LassoCV])
def test_linear_models_cv_fit_with_loky(estimator):
    if False:
        return 10
    (X, y) = make_regression(int(1000000.0) // 8 + 1, 1)
    assert X.nbytes > 1000000.0
    with joblib.parallel_backend('loky'):
        estimator(n_jobs=2, cv=3).fit(X, y)

@pytest.mark.parametrize('check_input', [True, False])
def test_enet_sample_weight_does_not_overwrite_sample_weight(check_input):
    if False:
        i = 10
        return i + 15
    'Check that ElasticNet does not overwrite sample_weights.'
    rng = np.random.RandomState(0)
    (n_samples, n_features) = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    sample_weight_1_25 = 1.25 * np.ones_like(y)
    sample_weight = sample_weight_1_25.copy()
    reg = ElasticNet()
    reg.fit(X, y, sample_weight=sample_weight, check_input=check_input)
    assert_array_equal(sample_weight, sample_weight_1_25)

@pytest.mark.parametrize('ridge_alpha', [0.1, 1.0, 1000000.0])
def test_enet_ridge_consistency(ridge_alpha):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(42)
    n_samples = 300
    (X, y) = make_regression(n_samples=n_samples, n_features=100, effective_rank=10, n_informative=50, random_state=rng)
    sw = rng.uniform(low=0.01, high=10, size=X.shape[0])
    alpha = 1.0
    common_params = dict(tol=1e-12)
    ridge = Ridge(alpha=alpha, **common_params).fit(X, y, sample_weight=sw)
    alpha_enet = alpha / sw.sum()
    enet = ElasticNet(alpha=alpha_enet, l1_ratio=0, **common_params).fit(X, y, sample_weight=sw)
    assert_allclose(ridge.coef_, enet.coef_)
    assert_allclose(ridge.intercept_, enet.intercept_)

@pytest.mark.parametrize('estimator', [Lasso(alpha=1.0), ElasticNet(alpha=1.0, l1_ratio=0.1)])
def test_sample_weight_invariance(estimator):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(42)
    (X, y) = make_regression(n_samples=100, n_features=300, effective_rank=10, n_informative=50, random_state=rng)
    sw = rng.uniform(low=0.01, high=2, size=X.shape[0])
    params = dict(tol=1e-12)
    cutoff = X.shape[0] // 3
    sw_with_null = sw.copy()
    sw_with_null[:cutoff] = 0.0
    (X_trimmed, y_trimmed) = (X[cutoff:, :], y[cutoff:])
    sw_trimmed = sw[cutoff:]
    reg_trimmed = clone(estimator).set_params(**params).fit(X_trimmed, y_trimmed, sample_weight=sw_trimmed)
    reg_null_weighted = clone(estimator).set_params(**params).fit(X, y, sample_weight=sw_with_null)
    assert_allclose(reg_null_weighted.coef_, reg_trimmed.coef_)
    assert_allclose(reg_null_weighted.intercept_, reg_trimmed.intercept_)
    X_dup = np.concatenate([X, X], axis=0)
    y_dup = np.concatenate([y, y], axis=0)
    sw_dup = np.concatenate([sw, sw], axis=0)
    reg_2sw = clone(estimator).set_params(**params).fit(X, y, sample_weight=2 * sw)
    reg_dup = clone(estimator).set_params(**params).fit(X_dup, y_dup, sample_weight=sw_dup)
    assert_allclose(reg_2sw.coef_, reg_dup.coef_)
    assert_allclose(reg_2sw.intercept_, reg_dup.intercept_)

def test_read_only_buffer():
    if False:
        print('Hello World!')
    'Test that sparse coordinate descent works for read-only buffers'
    rng = np.random.RandomState(0)
    clf = ElasticNet(alpha=0.1, copy_X=True, random_state=rng)
    X = np.asfortranarray(rng.uniform(size=(100, 10)))
    X.setflags(write=False)
    y = rng.rand(100)
    clf.fit(X, y)

@pytest.mark.parametrize('EstimatorCV', [ElasticNetCV, LassoCV, MultiTaskElasticNetCV, MultiTaskLassoCV])
def test_cv_estimators_reject_params_with_no_routing_enabled(EstimatorCV):
    if False:
        for i in range(10):
            print('nop')
    'Check that the models inheriting from class:`LinearModelCV` raise an\n    error when any `params` are passed when routing is not enabled.\n    '
    (X, y) = make_regression(random_state=42)
    groups = np.array([0, 1] * (len(y) // 2))
    estimator = EstimatorCV()
    msg = 'is only supported if enable_metadata_routing=True'
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, groups=groups)

@pytest.mark.usefixtures('enable_slep006')
@pytest.mark.parametrize('MultiTaskEstimatorCV', [MultiTaskElasticNetCV, MultiTaskLassoCV])
def test_multitask_cv_estimators_with_sample_weight(MultiTaskEstimatorCV):
    if False:
        while True:
            i = 10
    'Check that for :class:`MultiTaskElasticNetCV` and\n    class:`MultiTaskLassoCV` if `sample_weight` is passed and the\n    CV splitter does not support `sample_weight` an error is raised.\n    On the other hand if the splitter does support `sample_weight`\n    while `sample_weight` is passed there is no error and process\n    completes smoothly as before.\n    '

    class CVSplitter(BaseCrossValidator, GroupsConsumerMixin):

        def get_n_splits(self, X=None, y=None, groups=None, metadata=None):
            if False:
                print('Hello World!')
            pass

    class CVSplitterSampleWeight(CVSplitter):

        def split(self, X, y=None, groups=None, sample_weight=None):
            if False:
                i = 10
                return i + 15
            split_index = len(X) // 2
            train_indices = list(range(0, split_index))
            test_indices = list(range(split_index, len(X)))
            yield (test_indices, train_indices)
            yield (train_indices, test_indices)
    (X, y) = make_regression(random_state=42, n_targets=2)
    sample_weight = np.ones(X.shape[0])
    splitter = CVSplitter().set_split_request(groups=True)
    estimator = MultiTaskEstimatorCV(cv=splitter)
    msg = 'do not support sample weights'
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y, sample_weight=sample_weight)
    splitter = CVSplitterSampleWeight().set_split_request(groups=True, sample_weight=True)
    estimator = MultiTaskEstimatorCV(cv=splitter)
    estimator.fit(X, y, sample_weight=sample_weight)