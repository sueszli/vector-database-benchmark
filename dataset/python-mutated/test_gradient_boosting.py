import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.loss import AbsoluteError, HalfBinomialLoss, HalfSquaredError, PinballLoss
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_regressor
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification, make_low_rank_matrix, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
n_threads = _openmp_effective_n_threads()
(X_classification, y_classification) = make_classification(random_state=0)
(X_regression, y_regression) = make_regression(random_state=0)
(X_multi_classification, y_multi_classification) = make_classification(n_classes=3, n_informative=3, random_state=0)

def _make_dumb_dataset(n_samples):
    if False:
        for i in range(10):
            print('nop')
    'Make a dumb dataset to test early stopping.'
    rng = np.random.RandomState(42)
    X_dumb = rng.randn(n_samples, 1)
    y_dumb = (X_dumb[:, 0] > 0).astype('int64')
    return (X_dumb, y_dumb)

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
@pytest.mark.parametrize('params, err_msg', [({'interaction_cst': [0, 1]}, 'Interaction constraints must be a sequence of tuples or lists'), ({'interaction_cst': [{0, 9999}]}, 'Interaction constraints must consist of integer indices in \\[0, n_features - 1\\] = \\[.*\\], specifying the position of features,'), ({'interaction_cst': [{-1, 0}]}, 'Interaction constraints must consist of integer indices in \\[0, n_features - 1\\] = \\[.*\\], specifying the position of features,'), ({'interaction_cst': [{0.5}]}, 'Interaction constraints must consist of integer indices in \\[0, n_features - 1\\] = \\[.*\\], specifying the position of features,')])
def test_init_parameters_validation(GradientBoosting, X, y, params, err_msg):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match=err_msg):
        GradientBoosting(**params).fit(X, y)

@pytest.mark.parametrize('scoring, validation_fraction, early_stopping, n_iter_no_change, tol', [('neg_mean_squared_error', 0.1, True, 5, 1e-07), ('neg_mean_squared_error', None, True, 5, 0.1), (None, 0.1, True, 5, 1e-07), (None, None, True, 5, 0.1), ('loss', 0.1, True, 5, 1e-07), ('loss', None, True, 5, 0.1), (None, None, False, 5, 0.0)])
def test_early_stopping_regression(scoring, validation_fraction, early_stopping, n_iter_no_change, tol):
    if False:
        for i in range(10):
            print('nop')
    max_iter = 200
    (X, y) = make_regression(n_samples=50, random_state=0)
    gb = HistGradientBoostingRegressor(verbose=1, min_samples_leaf=5, scoring=scoring, tol=tol, early_stopping=early_stopping, validation_fraction=validation_fraction, max_iter=max_iter, n_iter_no_change=n_iter_no_change, random_state=0)
    gb.fit(X, y)
    if early_stopping:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter

@pytest.mark.parametrize('data', (make_classification(n_samples=30, random_state=0), make_classification(n_samples=30, n_classes=3, n_clusters_per_class=1, random_state=0)))
@pytest.mark.parametrize('scoring, validation_fraction, early_stopping, n_iter_no_change, tol', [('accuracy', 0.1, True, 5, 1e-07), ('accuracy', None, True, 5, 0.1), (None, 0.1, True, 5, 1e-07), (None, None, True, 5, 0.1), ('loss', 0.1, True, 5, 1e-07), ('loss', None, True, 5, 0.1), (None, None, False, 5, 0.0)])
def test_early_stopping_classification(data, scoring, validation_fraction, early_stopping, n_iter_no_change, tol):
    if False:
        i = 10
        return i + 15
    max_iter = 50
    (X, y) = data
    gb = HistGradientBoostingClassifier(verbose=1, min_samples_leaf=5, scoring=scoring, tol=tol, early_stopping=early_stopping, validation_fraction=validation_fraction, max_iter=max_iter, n_iter_no_change=n_iter_no_change, random_state=0)
    gb.fit(X, y)
    if early_stopping is True:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, *_make_dumb_dataset(10000)), (HistGradientBoostingClassifier, *_make_dumb_dataset(10001)), (HistGradientBoostingRegressor, *_make_dumb_dataset(10000)), (HistGradientBoostingRegressor, *_make_dumb_dataset(10001))])
def test_early_stopping_default(GradientBoosting, X, y):
    if False:
        print('Hello World!')
    gb = GradientBoosting(max_iter=10, n_iter_no_change=2, tol=0.1)
    gb.fit(X, y)
    if X.shape[0] > 10000:
        assert gb.n_iter_ < gb.max_iter
    else:
        assert gb.n_iter_ == gb.max_iter

@pytest.mark.parametrize('scores, n_iter_no_change, tol, stopping', [([], 1, 0.001, False), ([1, 1, 1], 5, 0.001, False), ([1, 1, 1, 1, 1], 5, 0.001, False), ([1, 2, 3, 4, 5, 6], 5, 0.001, False), ([1, 2, 3, 4, 5, 6], 5, 0.0, False), ([1, 2, 3, 4, 5, 6], 5, 0.999, False), ([1, 2, 3, 4, 5, 6], 5, 5 - 1e-05, False), ([1] * 6, 5, 0.0, True), ([1] * 6, 5, 0.001, True), ([1] * 6, 5, 5, True)])
def test_should_stop(scores, n_iter_no_change, tol, stopping):
    if False:
        return 10
    gbdt = HistGradientBoostingClassifier(n_iter_no_change=n_iter_no_change, tol=tol)
    assert gbdt._should_stop(scores) == stopping

def test_absolute_error():
    if False:
        return 10
    (X, y) = make_regression(n_samples=500, random_state=0)
    gbdt = HistGradientBoostingRegressor(loss='absolute_error', random_state=0)
    gbdt.fit(X, y)
    assert gbdt.score(X, y) > 0.9

def test_absolute_error_sample_weight():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = rng.uniform(-1, 1, size=n_samples)
    sample_weight = rng.uniform(0, 1, size=n_samples)
    gbdt = HistGradientBoostingRegressor(loss='absolute_error')
    gbdt.fit(X, y, sample_weight=sample_weight)

@pytest.mark.parametrize('y', [[1.0, -2.0, 0.0], [0.0, 1.0, 2.0]])
def test_gamma_y_positive(y):
    if False:
        while True:
            i = 10
    err_msg = "loss='gamma' requires strictly positive y."
    gbdt = HistGradientBoostingRegressor(loss='gamma', random_state=0)
    with pytest.raises(ValueError, match=err_msg):
        gbdt.fit(np.zeros(shape=(len(y), 1)), y)

def test_gamma():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(42)
    (n_train, n_test, n_features) = (500, 100, 20)
    X = make_low_rank_matrix(n_samples=n_train + n_test, n_features=n_features, random_state=rng)
    coef = rng.uniform(low=-10, high=20, size=n_features)
    dispersion = 0.5
    y = rng.gamma(shape=1 / dispersion, scale=dispersion * np.exp(X @ coef))
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=n_test, random_state=rng)
    gbdt_gamma = HistGradientBoostingRegressor(loss='gamma', random_state=123)
    gbdt_mse = HistGradientBoostingRegressor(loss='squared_error', random_state=123)
    dummy = DummyRegressor(strategy='mean')
    for model in (gbdt_gamma, gbdt_mse, dummy):
        model.fit(X_train, y_train)
    for (X, y) in [(X_train, y_train), (X_test, y_test)]:
        loss_gbdt_gamma = mean_gamma_deviance(y, gbdt_gamma.predict(X))
        loss_gbdt_mse = mean_gamma_deviance(y, np.maximum(np.min(y_train), gbdt_mse.predict(X)))
        loss_dummy = mean_gamma_deviance(y, dummy.predict(X))
        assert loss_gbdt_gamma < loss_dummy
        assert loss_gbdt_gamma < loss_gbdt_mse

@pytest.mark.parametrize('quantile', [0.2, 0.5, 0.8])
def test_quantile_asymmetric_error(quantile):
    if False:
        print('Hello World!')
    'Test quantile regression for asymmetric distributed targets.'
    n_samples = 10000
    rng = np.random.RandomState(42)
    X = np.concatenate((np.abs(rng.randn(n_samples)[:, None]), -rng.randint(2, size=(n_samples, 1))), axis=1)
    intercept = 1.23
    coef = np.array([0.5, -2])
    y = rng.exponential(scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples)
    model = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, max_iter=25, random_state=0, max_leaf_nodes=10).fit(X, y)
    assert_allclose(np.mean(model.predict(X) > y), quantile, rtol=0.01)
    pinball_loss = PinballLoss(quantile=quantile)
    loss_true_quantile = pinball_loss(y, X @ coef + intercept)
    loss_pred_quantile = pinball_loss(y, model.predict(X))
    assert loss_pred_quantile <= loss_true_quantile

@pytest.mark.parametrize('y', [[1.0, -2.0, 0.0], [0.0, 0.0, 0.0]])
def test_poisson_y_positive(y):
    if False:
        for i in range(10):
            print('nop')
    err_msg = "loss='poisson' requires non-negative y and sum\\(y\\) > 0."
    gbdt = HistGradientBoostingRegressor(loss='poisson', random_state=0)
    with pytest.raises(ValueError, match=err_msg):
        gbdt.fit(np.zeros(shape=(len(y), 1)), y)

def test_poisson():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    (n_train, n_test, n_features) = (500, 100, 100)
    X = make_low_rank_matrix(n_samples=n_train + n_test, n_features=n_features, random_state=rng)
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=n_test, random_state=rng)
    gbdt_pois = HistGradientBoostingRegressor(loss='poisson', random_state=rng)
    gbdt_ls = HistGradientBoostingRegressor(loss='squared_error', random_state=rng)
    gbdt_pois.fit(X_train, y_train)
    gbdt_ls.fit(X_train, y_train)
    dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)
    for (X, y) in [(X_train, y_train), (X_test, y_test)]:
        metric_pois = mean_poisson_deviance(y, gbdt_pois.predict(X))
        metric_ls = mean_poisson_deviance(y, np.clip(gbdt_ls.predict(X), 1e-15, None))
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        assert metric_pois < metric_ls
        assert metric_pois < metric_dummy

def test_binning_train_validation_are_separated():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    validation_fraction = 0.2
    gb = HistGradientBoostingClassifier(early_stopping=True, validation_fraction=validation_fraction, random_state=rng)
    gb.fit(X_classification, y_classification)
    mapper_training_data = gb._bin_mapper
    mapper_whole_data = _BinMapper(random_state=0)
    mapper_whole_data.fit(X_classification)
    n_samples = X_classification.shape[0]
    assert np.all(mapper_training_data.n_bins_non_missing_ == int((1 - validation_fraction) * n_samples))
    assert np.all(mapper_training_data.n_bins_non_missing_ != mapper_whole_data.n_bins_non_missing_)

def test_missing_values_trivial():
    if False:
        print('Hello World!')
    n_samples = 100
    n_features = 1
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    mask = rng.binomial(1, 0.5, size=X.shape).astype(bool)
    X[mask] = np.nan
    y = mask.ravel()
    gb = HistGradientBoostingClassifier()
    gb.fit(X, y)
    assert gb.score(X, y) == pytest.approx(1)

@pytest.mark.parametrize('problem', ('classification', 'regression'))
@pytest.mark.parametrize('missing_proportion, expected_min_score_classification, expected_min_score_regression', [(0.1, 0.97, 0.89), (0.2, 0.93, 0.81), (0.5, 0.79, 0.52)])
def test_missing_values_resilience(problem, missing_proportion, expected_min_score_classification, expected_min_score_regression):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    if problem == 'regression':
        (X, y) = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, random_state=rng)
        gb = HistGradientBoostingRegressor()
        expected_min_score = expected_min_score_regression
    else:
        (X, y) = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0, random_state=rng)
        gb = HistGradientBoostingClassifier()
        expected_min_score = expected_min_score_classification
    mask = rng.binomial(1, missing_proportion, size=X.shape).astype(bool)
    X[mask] = np.nan
    gb.fit(X, y)
    assert gb.score(X, y) > expected_min_score

@pytest.mark.parametrize('data', [make_classification(random_state=0, n_classes=2), make_classification(random_state=0, n_classes=3, n_informative=3)], ids=['binary_log_loss', 'multiclass_log_loss'])
def test_zero_division_hessians(data):
    if False:
        print('Hello World!')
    (X, y) = data
    gb = HistGradientBoostingClassifier(learning_rate=100, max_iter=10)
    gb.fit(X, y)

def test_small_trainset():
    if False:
        while True:
            i = 10
    n_samples = 20000
    original_distrib = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples).reshape(n_samples, 1)
    y = [[class_] * int(prop * n_samples) for (class_, prop) in original_distrib.items()]
    y = shuffle(np.concatenate(y))
    gb = HistGradientBoostingClassifier()
    (X_small, y_small, _) = gb._get_small_trainset(X, y, seed=42, sample_weight_train=None)
    (unique, counts) = np.unique(y_small, return_counts=True)
    small_distrib = {class_: count / 10000 for (class_, count) in zip(unique, counts)}
    assert X_small.shape[0] == 10000
    assert y_small.shape[0] == 10000
    assert small_distrib == pytest.approx(original_distrib)

def test_missing_values_minmax_imputation():
    if False:
        return 10

    class MinMaxImputer(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            if False:
                print('Hello World!')
            mm = MinMaxScaler().fit(X)
            self.data_min_ = mm.data_min_
            self.data_max_ = mm.data_max_
            return self

        def transform(self, X):
            if False:
                while True:
                    i = 10
            (X_min, X_max) = (X.copy(), X.copy())
            for feature_idx in range(X.shape[1]):
                nan_mask = np.isnan(X[:, feature_idx])
                X_min[nan_mask, feature_idx] = self.data_min_[feature_idx] - 1
                X_max[nan_mask, feature_idx] = self.data_max_[feature_idx] + 1
            return np.concatenate([X_min, X_max], axis=1)

    def make_missing_value_data(n_samples=int(10000.0), seed=0):
        if False:
            return 10
        rng = np.random.RandomState(seed)
        (X, y) = make_regression(n_samples=n_samples, n_features=4, random_state=rng)
        X = KBinsDiscretizer(n_bins=42, encode='ordinal').fit_transform(X)
        rnd_mask = rng.rand(X.shape[0]) > 0.9
        X[rnd_mask, 0] = np.nan
        low_mask = X[:, 1] == 0
        X[low_mask, 1] = np.nan
        high_mask = X[:, 2] == X[:, 2].max()
        X[high_mask, 2] = np.nan
        y_max = np.percentile(y, 70)
        y_max_mask = y >= y_max
        y[y_max_mask] = y_max
        X[y_max_mask, 3] = np.nan
        for feature_idx in range(X.shape[1]):
            assert any(np.isnan(X[:, feature_idx]))
        return train_test_split(X, y, random_state=rng)
    (X_train, X_test, y_train, y_test) = make_missing_value_data(n_samples=int(10000.0), seed=0)
    gbm1 = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=5, random_state=0)
    gbm1.fit(X_train, y_train)
    gbm2 = make_pipeline(MinMaxImputer(), clone(gbm1))
    gbm2.fit(X_train, y_train)
    assert gbm1.score(X_train, y_train) == pytest.approx(gbm2.score(X_train, y_train))
    assert gbm1.score(X_test, y_test) == pytest.approx(gbm2.score(X_test, y_test))
    assert_allclose(gbm1.predict(X_train), gbm2.predict(X_train))
    assert_allclose(gbm1.predict(X_test), gbm2.predict(X_test))

def test_infinite_values():
    if False:
        return 10
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    gbdt = HistGradientBoostingRegressor(min_samples_leaf=1)
    gbdt.fit(X, y)
    np.testing.assert_allclose(gbdt.predict(X), y, atol=0.0001)

def test_consistent_lengths():
    if False:
        print('Hello World!')
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    sample_weight = np.array([0.1, 0.3, 0.1])
    gbdt = HistGradientBoostingRegressor()
    with pytest.raises(ValueError, match='sample_weight.shape == \\(3,\\), expected'):
        gbdt.fit(X, y, sample_weight)
    with pytest.raises(ValueError, match='Found input variables with inconsistent number'):
        gbdt.fit(X, y[1:])

def test_infinite_values_missing_values():
    if False:
        print('Hello World!')
    X = np.asarray([-np.inf, 0, 1, np.inf, np.nan]).reshape(-1, 1)
    y_isnan = np.isnan(X.ravel())
    y_isinf = X.ravel() == np.inf
    stump_clf = HistGradientBoostingClassifier(min_samples_leaf=1, max_iter=1, learning_rate=1, max_depth=2)
    assert stump_clf.fit(X, y_isinf).score(X, y_isinf) == 1
    assert stump_clf.fit(X, y_isnan).score(X, y_isnan) == 1

@pytest.mark.parametrize('scoring', [None, 'loss'])
def test_string_target_early_stopping(scoring):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = np.array(['x'] * 50 + ['y'] * 50, dtype=object)
    gbrt = HistGradientBoostingClassifier(n_iter_no_change=10, scoring=scoring)
    gbrt.fit(X, y)

def test_zero_sample_weights_regression():
    if False:
        for i in range(10):
            print('nop')
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    sample_weight = [0, 0, 1, 1]
    gb = HistGradientBoostingRegressor(min_samples_leaf=1)
    gb.fit(X, y, sample_weight=sample_weight)
    assert gb.predict([[1, 0]])[0] > 0.5

def test_zero_sample_weights_classification():
    if False:
        print('Hello World!')
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    sample_weight = [0, 0, 1, 1]
    gb = HistGradientBoostingClassifier(loss='log_loss', min_samples_leaf=1)
    gb.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(gb.predict([[1, 0]]), [1])
    X = [[1, 0], [1, 0], [1, 0], [0, 1], [1, 1]]
    y = [0, 0, 1, 0, 2]
    sample_weight = [0, 0, 1, 1, 1]
    gb = HistGradientBoostingClassifier(loss='log_loss', min_samples_leaf=1)
    gb.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(gb.predict([[1, 0]]), [1])

@pytest.mark.parametrize('problem', ('regression', 'binary_classification', 'multiclass_classification'))
@pytest.mark.parametrize('duplication', ('half', 'all'))
def test_sample_weight_effect(problem, duplication):
    if False:
        while True:
            i = 10
    n_samples = 255
    n_features = 2
    if problem == 'regression':
        (X, y) = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, random_state=0)
        Klass = HistGradientBoostingRegressor
    else:
        n_classes = 2 if problem == 'binary_classification' else 3
        (X, y) = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, n_classes=n_classes, random_state=0)
        Klass = HistGradientBoostingClassifier
    est = Klass(min_samples_leaf=1)
    if duplication == 'half':
        lim = n_samples // 2
    else:
        lim = n_samples
    X_dup = np.r_[X, X[:lim]]
    y_dup = np.r_[y, y[:lim]]
    sample_weight = np.ones(shape=n_samples)
    sample_weight[:lim] = 2
    est_sw = clone(est).fit(X, y, sample_weight=sample_weight)
    est_dup = clone(est).fit(X_dup, y_dup)
    assert np.allclose(est_sw._raw_predict(X_dup), est_dup._raw_predict(X_dup))

@pytest.mark.parametrize('Loss', (HalfSquaredError, AbsoluteError))
def test_sum_hessians_are_sample_weight(Loss):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    (X, y) = make_regression(n_samples=n_samples, n_features=n_features, random_state=rng)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)
    sample_weight = rng.normal(size=n_samples)
    loss = Loss(sample_weight=sample_weight)
    (gradients, hessians) = loss.init_gradient_and_hessian(n_samples=n_samples, dtype=G_H_DTYPE)
    (gradients, hessians) = (gradients.reshape((-1, 1)), hessians.reshape((-1, 1)))
    raw_predictions = rng.normal(size=(n_samples, 1))
    loss.gradient_hessian(y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight, gradient_out=gradients, hessian_out=hessians, n_threads=n_threads)
    sum_sw = np.zeros(shape=(n_features, bin_mapper.n_bins))
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            sum_sw[feature_idx, X_binned[sample_idx, feature_idx]] += sample_weight[sample_idx]
    grower = TreeGrower(X_binned, gradients[:, 0], hessians[:, 0], n_bins=bin_mapper.n_bins)
    histograms = grower.histogram_builder.compute_histograms_brute(grower.root.sample_indices)
    for feature_idx in range(n_features):
        for bin_idx in range(bin_mapper.n_bins):
            assert histograms[feature_idx, bin_idx]['sum_hessians'] == pytest.approx(sum_sw[feature_idx, bin_idx], rel=1e-05)

def test_max_depth_max_leaf_nodes():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(random_state=0)
    est = HistGradientBoostingClassifier(max_depth=2, max_leaf_nodes=3, max_iter=1).fit(X, y)
    tree = est._predictors[0][0]
    assert tree.get_max_depth() == 2
    assert tree.get_n_leaf_nodes() == 3

def test_early_stopping_on_test_set_with_warm_start():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(random_state=0)
    gb = HistGradientBoostingClassifier(max_iter=1, scoring='loss', warm_start=True, early_stopping=True, n_iter_no_change=1, validation_fraction=None)
    gb.fit(X, y)
    gb.set_params(max_iter=2)
    gb.fit(X, y)

@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
def test_single_node_trees(Est):
    if False:
        print('Hello World!')
    (X, y) = make_classification(random_state=0)
    y[:] = 1
    est = Est(max_iter=20)
    est.fit(X, y)
    assert all((len(predictor[0].nodes) == 1 for predictor in est._predictors))
    assert all((predictor[0].nodes[0]['value'] == 0 for predictor in est._predictors))
    assert_allclose(est.predict(X), y)

@pytest.mark.parametrize('Est, loss, X, y', [(HistGradientBoostingClassifier, HalfBinomialLoss(sample_weight=None), X_classification, y_classification), (HistGradientBoostingRegressor, HalfSquaredError(sample_weight=None), X_regression, y_regression)])
def test_custom_loss(Est, loss, X, y):
    if False:
        return 10
    est = Est(loss=loss, max_iter=20)
    est.fit(X, y)

@pytest.mark.parametrize('HistGradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression), (HistGradientBoostingClassifier, X_multi_classification, y_multi_classification)])
def test_staged_predict(HistGradientBoosting, X, y):
    if False:
        i = 10
        return i + 15
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.5, random_state=0)
    gb = HistGradientBoosting(max_iter=10)
    with pytest.raises(NotFittedError):
        next(gb.staged_predict(X_test))
    gb.fit(X_train, y_train)
    method_names = ['predict'] if is_regressor(gb) else ['predict', 'predict_proba', 'decision_function']
    for method_name in method_names:
        staged_method = getattr(gb, 'staged_' + method_name)
        staged_predictions = list(staged_method(X_test))
        assert len(staged_predictions) == gb.n_iter_
        for (n_iter, staged_predictions) in enumerate(staged_method(X_test), 1):
            aux = HistGradientBoosting(max_iter=n_iter)
            aux.fit(X_train, y_train)
            pred_aux = getattr(aux, method_name)(X_test)
            assert_allclose(staged_predictions, pred_aux)
            assert staged_predictions.shape == pred_aux.shape

@pytest.mark.parametrize('insert_missing', [False, True])
@pytest.mark.parametrize('Est', (HistGradientBoostingRegressor, HistGradientBoostingClassifier))
@pytest.mark.parametrize('bool_categorical_parameter', [True, False])
@pytest.mark.parametrize('missing_value', [np.nan, -1])
def test_unknown_categories_nan(insert_missing, Est, bool_categorical_parameter, missing_value):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    n_samples = 1000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(4, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1
    if bool_categorical_parameter:
        categorical_features = [False, True]
    else:
        categorical_features = [1]
    if insert_missing:
        mask = rng.binomial(1, 0.01, size=X.shape).astype(bool)
        assert mask.sum() > 0
        X[mask] = missing_value
    est = Est(max_iter=20, categorical_features=categorical_features).fit(X, y)
    assert_array_equal(est.is_categorical_, [False, True])
    X_test = np.zeros((10, X.shape[1]), dtype=float)
    X_test[:5, 1] = 30
    X_test[5:, 1] = missing_value
    assert len(np.unique(est.predict(X_test))) == 1

def test_categorical_encoding_strategies():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    n_samples = 10000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(6, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1
    assert 0.49 < y.mean() < 0.51
    native_cat_specs = [[False, True], [1]]
    try:
        import pandas as pd
        X = pd.DataFrame(X, columns=['f_0', 'f_1'])
        native_cat_specs.append(['f_1'])
    except ImportError:
        pass
    for native_cat_spec in native_cat_specs:
        clf_cat = HistGradientBoostingClassifier(max_iter=1, max_depth=1, categorical_features=native_cat_spec)
        assert cross_val_score(clf_cat, X, y).mean() == 1
    expected_left_bitset = [21, 0, 0, 0, 0, 0, 0, 0]
    left_bitset = clf_cat.fit(X, y)._predictors[0][0].raw_left_cat_bitsets[0]
    assert_array_equal(left_bitset, expected_left_bitset)
    clf_no_cat = HistGradientBoostingClassifier(max_iter=1, max_depth=4, categorical_features=None)
    assert cross_val_score(clf_no_cat, X, y).mean() < 0.9
    clf_no_cat.set_params(max_depth=5)
    assert cross_val_score(clf_no_cat, X, y).mean() == 1
    ct = make_column_transformer((OneHotEncoder(sparse_output=False), [1]), remainder='passthrough')
    X_ohe = ct.fit_transform(X)
    clf_no_cat.set_params(max_depth=2)
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() < 0.9
    clf_no_cat.set_params(max_depth=3)
    assert cross_val_score(clf_no_cat, X_ohe, y).mean() == 1

@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
@pytest.mark.parametrize('categorical_features, monotonic_cst, expected_msg', [([b'hello', b'world'], None, re.escape('categorical_features must be an array-like of bool, int or str, got: bytes40.')), (np.array([b'hello', 1.3], dtype=object), None, re.escape('categorical_features must be an array-like of bool, int or str, got: bytes, float.')), ([0, -1], None, re.escape('categorical_features set as integer indices must be in [0, n_features - 1]')), ([True, True, False, False, True], None, re.escape('categorical_features set as a boolean mask must have shape (n_features,)')), ([True, True, False, False], [0, -1, 0, 1], 'Categorical features cannot have monotonic constraints')])
def test_categorical_spec_errors(Est, categorical_features, monotonic_cst, expected_msg):
    if False:
        for i in range(10):
            print('nop')
    n_samples = 100
    (X, y) = make_classification(random_state=0, n_features=4, n_samples=n_samples)
    rng = np.random.RandomState(0)
    X[:, 0] = rng.randint(0, 10, size=n_samples)
    X[:, 1] = rng.randint(0, 10, size=n_samples)
    est = Est(categorical_features=categorical_features, monotonic_cst=monotonic_cst)
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)

@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
def test_categorical_spec_errors_with_feature_names(Est):
    if False:
        i = 10
        return i + 15
    pd = pytest.importorskip('pandas')
    n_samples = 10
    X = pd.DataFrame({'f0': range(n_samples), 'f1': range(n_samples), 'f2': [1.0] * n_samples})
    y = [0, 1] * (n_samples // 2)
    est = Est(categorical_features=['f0', 'f1', 'f3'])
    expected_msg = re.escape("categorical_features has a item value 'f3' which is not a valid feature name of the training data.")
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)
    est = Est(categorical_features=['f0', 'f1'])
    expected_msg = re.escape('categorical_features should be passed as an array of integers or as a boolean mask when the model is fitted on data without feature names.')
    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X.to_numpy(), y)

@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
@pytest.mark.parametrize('categorical_features', ([False, False], []))
@pytest.mark.parametrize('as_array', (True, False))
def test_categorical_spec_no_categories(Est, categorical_features, as_array):
    if False:
        i = 10
        return i + 15
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    if as_array:
        categorical_features = np.asarray(categorical_features)
    est = Est(categorical_features=categorical_features).fit(X, y)
    assert est.is_categorical_ is None

@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
@pytest.mark.parametrize('use_pandas, feature_name', [(False, 'at index 0'), (True, "'f0'")])
def test_categorical_bad_encoding_errors(Est, use_pandas, feature_name):
    if False:
        return 10
    gb = Est(categorical_features=[True], max_bins=2)
    if use_pandas:
        pd = pytest.importorskip('pandas')
        X = pd.DataFrame({'f0': [0, 1, 2]})
    else:
        X = np.array([[0, 1, 2]]).T
    y = np.arange(3)
    msg = f'Categorical feature {feature_name} is expected to have a cardinality <= 2 but actually has a cardinality of 3.'
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)
    if use_pandas:
        X = pd.DataFrame({'f0': [0, 2]})
    else:
        X = np.array([[0, 2]]).T
    y = np.arange(2)
    msg = f'Categorical feature {feature_name} is expected to be encoded with values < 2 but the largest value for the encoded categories is 2.0.'
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)
    X = np.array([[0, 1, np.nan]]).T
    y = np.arange(3)
    gb.fit(X, y)

@pytest.mark.parametrize('Est', (HistGradientBoostingClassifier, HistGradientBoostingRegressor))
def test_uint8_predict(Est):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.randint(0, 100, size=(10, 2)).astype(np.uint8)
    y = rng.randint(0, 2, size=10).astype(np.uint8)
    est = Est()
    est.fit(X, y)
    est.predict(X)

@pytest.mark.parametrize('interaction_cst, n_features, result', [(None, 931, None), ([{0, 1}], 2, [{0, 1}]), ('pairwise', 2, [{0, 1}]), ('pairwise', 4, [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]), ('no_interactions', 2, [{0}, {1}]), ('no_interactions', 4, [{0}, {1}, {2}, {3}]), ([(1, 0), [5, 1]], 6, [{0, 1}, {1, 5}, {2, 3, 4}])])
def test_check_interaction_cst(interaction_cst, n_features, result):
    if False:
        for i in range(10):
            print('nop')
    'Check that _check_interaction_cst returns the expected list of sets'
    est = HistGradientBoostingRegressor()
    est.set_params(interaction_cst=interaction_cst)
    assert est._check_interaction_cst(n_features) == result

def test_interaction_cst_numerically():
    if False:
        return 10
    'Check that interaction constraints have no forbidden interactions.'
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = rng.uniform(size=(n_samples, 2))
    y = np.hstack((X, 5 * X[:, [0]] * X[:, [1]])).sum(axis=1)
    est = HistGradientBoostingRegressor(random_state=42)
    est.fit(X, y)
    est_no_interactions = HistGradientBoostingRegressor(interaction_cst=[{0}, {1}], random_state=42)
    est_no_interactions.fit(X, y)
    delta = 0.25
    X_test = X[(X[:, 0] < 1 - delta) & (X[:, 1] < 1 - delta)]
    X_delta_d_0 = X_test + [delta, 0]
    X_delta_0_d = X_test + [0, delta]
    X_delta_d_d = X_test + [delta, delta]
    assert_allclose(est_no_interactions.predict(X_delta_d_d) + est_no_interactions.predict(X_test) - est_no_interactions.predict(X_delta_d_0) - est_no_interactions.predict(X_delta_0_d), 0, atol=1e-12)
    assert np.all(est.predict(X_delta_d_d) + est.predict(X_test) - est.predict(X_delta_d_0) - est.predict(X_delta_0_d) > 0.01)

def test_no_user_warning_with_scoring():
    if False:
        while True:
            i = 10
    'Check that no UserWarning is raised when scoring is set.\n\n    Non-regression test for #22907.\n    '
    pd = pytest.importorskip('pandas')
    (X, y) = make_regression(n_samples=50, random_state=0)
    X_df = pd.DataFrame(X, columns=[f'col{i}' for i in range(X.shape[1])])
    est = HistGradientBoostingRegressor(random_state=0, scoring='neg_mean_absolute_error', early_stopping=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        est.fit(X_df, y)

def test_class_weights():
    if False:
        for i in range(10):
            print('nop')
    'High level test to check class_weights.'
    n_samples = 255
    n_features = 2
    (X, y) = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=0)
    y_is_1 = y == 1
    clf = HistGradientBoostingClassifier(min_samples_leaf=2, random_state=0, max_depth=2)
    sample_weight = np.ones(shape=n_samples)
    sample_weight[y_is_1] = 3.0
    clf.fit(X, y, sample_weight=sample_weight)
    class_weight = {0: 1.0, 1: 3.0}
    clf_class_weighted = clone(clf).set_params(class_weight=class_weight)
    clf_class_weighted.fit(X, y)
    assert_allclose(clf.decision_function(X), clf_class_weighted.decision_function(X))
    clf.fit(X, y, sample_weight=sample_weight ** 2)
    clf_class_weighted.fit(X, y, sample_weight=sample_weight)
    assert_allclose(clf.decision_function(X), clf_class_weighted.decision_function(X))
    X_imb = np.concatenate((X[~y_is_1], X[y_is_1][:10]))
    y_imb = np.concatenate((y[~y_is_1], y[y_is_1][:10]))
    clf_balanced = clone(clf).set_params(class_weight='balanced')
    clf_balanced.fit(X_imb, y_imb)
    class_weight = y_imb.shape[0] / (2 * np.bincount(y_imb))
    sample_weight = class_weight[y_imb]
    clf_sample_weight = clone(clf).set_params(class_weight=None)
    clf_sample_weight.fit(X_imb, y_imb, sample_weight=sample_weight)
    assert_allclose(clf_balanced.decision_function(X_imb), clf_sample_weight.decision_function(X_imb))

def test_unknown_category_that_are_negative():
    if False:
        return 10
    'Check that unknown categories that are negative does not error.\n\n    Non-regression test for #24274.\n    '
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = np.c_[rng.rand(n_samples), rng.randint(4, size=n_samples)]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1
    hist = HistGradientBoostingRegressor(random_state=0, categorical_features=[False, True], max_iter=10).fit(X, y)
    X_test_neg = np.asarray([[1, -2], [3, -4]])
    X_test_nan = np.asarray([[1, np.nan], [3, np.nan]])
    assert_allclose(hist.predict(X_test_neg), hist.predict(X_test_nan))