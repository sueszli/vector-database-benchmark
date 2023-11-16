import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('loss', ['squared_error', 'poisson', pytest.param('gamma', marks=pytest.mark.skip('LightGBM with gamma loss has larger deviation.'))])
@pytest.mark.parametrize('min_samples_leaf', (1, 20))
@pytest.mark.parametrize('n_samples, max_leaf_nodes', [(255, 4096), (1000, 8)])
def test_same_predictions_regression(seed, loss, min_samples_leaf, n_samples, max_leaf_nodes):
    if False:
        return 10
    pytest.importorskip('lightgbm')
    rng = np.random.RandomState(seed=seed)
    max_iter = 1
    max_bins = 255
    (X, y) = make_regression(n_samples=n_samples, n_features=5, n_informative=5, random_state=0)
    if loss in ('gamma', 'poisson'):
        y = np.abs(y) + np.mean(np.abs(y))
    if n_samples > 255:
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=rng)
    est_sklearn = HistGradientBoostingRegressor(loss=loss, max_iter=max_iter, max_bins=max_bins, learning_rate=1, early_stopping=False, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    est_lightgbm = get_equivalent_estimator(est_sklearn, lib='lightgbm')
    est_lightgbm.set_params(min_sum_hessian_in_leaf=0)
    est_lightgbm.fit(X_train, y_train)
    est_sklearn.fit(X_train, y_train)
    (X_train, X_test) = (X_train.astype(np.float32), X_test.astype(np.float32))
    pred_lightgbm = est_lightgbm.predict(X_train)
    pred_sklearn = est_sklearn.predict(X_train)
    if loss in ('gamma', 'poisson'):
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=0.01, atol=0.01)) > 0.65
    else:
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=0.001)) > 1 - 0.01
    if max_leaf_nodes < 10 and n_samples >= 1000 and (loss in ('squared_error',)):
        pred_lightgbm = est_lightgbm.predict(X_test)
        pred_sklearn = est_sklearn.predict(X_test)
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=0.0001)) > 1 - 0.01

@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('min_samples_leaf', (1, 20))
@pytest.mark.parametrize('n_samples, max_leaf_nodes', [(255, 4096), (1000, 8)])
def test_same_predictions_classification(seed, min_samples_leaf, n_samples, max_leaf_nodes):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('lightgbm')
    rng = np.random.RandomState(seed=seed)
    max_iter = 1
    n_classes = 2
    max_bins = 255
    (X, y) = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=5, n_informative=5, n_redundant=0, random_state=0)
    if n_samples > 255:
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=rng)
    est_sklearn = HistGradientBoostingClassifier(loss='log_loss', max_iter=max_iter, max_bins=max_bins, learning_rate=1, early_stopping=False, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    est_lightgbm = get_equivalent_estimator(est_sklearn, lib='lightgbm', n_classes=n_classes)
    est_lightgbm.fit(X_train, y_train)
    est_sklearn.fit(X_train, y_train)
    (X_train, X_test) = (X_train.astype(np.float32), X_test.astype(np.float32))
    pred_lightgbm = est_lightgbm.predict(X_train)
    pred_sklearn = est_sklearn.predict(X_train)
    assert np.mean(pred_sklearn == pred_lightgbm) > 0.89
    acc_lightgbm = accuracy_score(y_train, pred_lightgbm)
    acc_sklearn = accuracy_score(y_train, pred_sklearn)
    np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn)
    if max_leaf_nodes < 10 and n_samples >= 1000:
        pred_lightgbm = est_lightgbm.predict(X_test)
        pred_sklearn = est_sklearn.predict(X_test)
        assert np.mean(pred_sklearn == pred_lightgbm) > 0.89
        acc_lightgbm = accuracy_score(y_test, pred_lightgbm)
        acc_sklearn = accuracy_score(y_test, pred_sklearn)
        np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn, decimal=2)

@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('min_samples_leaf', (1, 20))
@pytest.mark.parametrize('n_samples, max_leaf_nodes', [(255, 4096), (10000, 8)])
def test_same_predictions_multiclass_classification(seed, min_samples_leaf, n_samples, max_leaf_nodes):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('lightgbm')
    rng = np.random.RandomState(seed=seed)
    n_classes = 3
    max_iter = 1
    max_bins = 255
    lr = 1
    (X, y) = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=5, n_informative=5, n_redundant=0, n_clusters_per_class=1, random_state=0)
    if n_samples > 255:
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=rng)
    est_sklearn = HistGradientBoostingClassifier(loss='log_loss', max_iter=max_iter, max_bins=max_bins, learning_rate=lr, early_stopping=False, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    est_lightgbm = get_equivalent_estimator(est_sklearn, lib='lightgbm', n_classes=n_classes)
    est_lightgbm.fit(X_train, y_train)
    est_sklearn.fit(X_train, y_train)
    (X_train, X_test) = (X_train.astype(np.float32), X_test.astype(np.float32))
    pred_lightgbm = est_lightgbm.predict(X_train)
    pred_sklearn = est_sklearn.predict(X_train)
    assert np.mean(pred_sklearn == pred_lightgbm) > 0.89
    proba_lightgbm = est_lightgbm.predict_proba(X_train)
    proba_sklearn = est_sklearn.predict_proba(X_train)
    assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 0.01) > 0.75
    acc_lightgbm = accuracy_score(y_train, pred_lightgbm)
    acc_sklearn = accuracy_score(y_train, pred_sklearn)
    np.testing.assert_allclose(acc_lightgbm, acc_sklearn, rtol=0, atol=0.05)
    if max_leaf_nodes < 10 and n_samples >= 1000:
        pred_lightgbm = est_lightgbm.predict(X_test)
        pred_sklearn = est_sklearn.predict(X_test)
        assert np.mean(pred_sklearn == pred_lightgbm) > 0.89
        proba_lightgbm = est_lightgbm.predict_proba(X_train)
        proba_sklearn = est_sklearn.predict_proba(X_train)
        assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 0.01) > 0.75
        acc_lightgbm = accuracy_score(y_test, pred_lightgbm)
        acc_sklearn = accuracy_score(y_test, pred_sklearn)
        np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn, decimal=2)