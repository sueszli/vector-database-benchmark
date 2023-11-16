import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import check_scoring
(X_classification, y_classification) = make_classification(random_state=0)
(X_regression, y_regression) = make_regression(random_state=0)

def _assert_predictor_equal(gb_1, gb_2, X):
    if False:
        print('Hello World!')
    'Assert that two HistGBM instances are identical.'
    for (pred_ith_1, pred_ith_2) in zip(gb_1._predictors, gb_2._predictors):
        for (predictor_1, predictor_2) in zip(pred_ith_1, pred_ith_2):
            assert_array_equal(predictor_1.nodes, predictor_2.nodes)
    assert_allclose(gb_1.predict(X), gb_2.predict(X))

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_max_iter_with_warm_start_validation(GradientBoosting, X, y):
    if False:
        print('Hello World!')
    estimator = GradientBoosting(max_iter=10, early_stopping=False, warm_start=True)
    estimator.fit(X, y)
    estimator.set_params(max_iter=5)
    err_msg = 'max_iter=5 must be larger than or equal to n_iter_=10 when warm_start==True'
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_yields_identical_results(GradientBoosting, X, y):
    if False:
        print('Hello World!')
    rng = 42
    gb_warm_start = GradientBoosting(n_iter_no_change=100, max_iter=50, random_state=rng, warm_start=True)
    gb_warm_start.fit(X, y).set_params(max_iter=75).fit(X, y)
    gb_no_warm_start = GradientBoosting(n_iter_no_change=100, max_iter=75, random_state=rng, warm_start=False)
    gb_no_warm_start.fit(X, y)
    _assert_predictor_equal(gb_warm_start, gb_no_warm_start, X)

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_max_depth(GradientBoosting, X, y):
    if False:
        for i in range(10):
            print('nop')
    gb = GradientBoosting(max_iter=20, min_samples_leaf=1, warm_start=True, max_depth=2, early_stopping=False)
    gb.fit(X, y)
    gb.set_params(max_iter=30, max_depth=3, n_iter_no_change=110)
    gb.fit(X, y)
    for i in range(20):
        assert gb._predictors[i][0].get_max_depth() == 2
    for i in range(1, 11):
        assert gb._predictors[-i][0].get_max_depth() == 3

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
@pytest.mark.parametrize('scoring', (None, 'loss'))
def test_warm_start_early_stopping(GradientBoosting, X, y, scoring):
    if False:
        i = 10
        return i + 15
    n_iter_no_change = 5
    gb = GradientBoosting(n_iter_no_change=n_iter_no_change, max_iter=10000, early_stopping=True, random_state=42, warm_start=True, tol=0.001, scoring=scoring)
    gb.fit(X, y)
    n_iter_first_fit = gb.n_iter_
    gb.fit(X, y)
    n_iter_second_fit = gb.n_iter_
    assert 0 < n_iter_second_fit - n_iter_first_fit < n_iter_no_change

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_equal_n_estimators(GradientBoosting, X, y):
    if False:
        i = 10
        return i + 15
    gb_1 = GradientBoosting(max_depth=2, early_stopping=False)
    gb_1.fit(X, y)
    gb_2 = clone(gb_1)
    gb_2.set_params(max_iter=gb_1.max_iter, warm_start=True, n_iter_no_change=5)
    gb_2.fit(X, y)
    _assert_predictor_equal(gb_1, gb_2, X)

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
def test_warm_start_clear(GradientBoosting, X, y):
    if False:
        while True:
            i = 10
    gb_1 = GradientBoosting(n_iter_no_change=5, random_state=42)
    gb_1.fit(X, y)
    gb_2 = GradientBoosting(n_iter_no_change=5, random_state=42, warm_start=True)
    gb_2.fit(X, y)
    gb_2.set_params(warm_start=False)
    gb_2.fit(X, y)
    assert_allclose(gb_1.train_score_, gb_2.train_score_)
    assert_allclose(gb_1.validation_score_, gb_2.validation_score_)
    _assert_predictor_equal(gb_1, gb_2, X)

@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
@pytest.mark.parametrize('rng_type', ('none', 'int', 'instance'))
def test_random_seeds_warm_start(GradientBoosting, X, y, rng_type):
    if False:
        return 10

    def _get_rng(rng_type):
        if False:
            i = 10
            return i + 15
        if rng_type == 'none':
            return None
        elif rng_type == 'int':
            return 42
        else:
            return np.random.RandomState(0)
    random_state = _get_rng(rng_type)
    gb_1 = GradientBoosting(early_stopping=True, max_iter=2, random_state=random_state)
    gb_1.set_params(scoring=check_scoring(gb_1))
    gb_1.fit(X, y)
    random_seed_1_1 = gb_1._random_seed
    gb_1.fit(X, y)
    random_seed_1_2 = gb_1._random_seed
    random_state = _get_rng(rng_type)
    gb_2 = GradientBoosting(early_stopping=True, max_iter=2, random_state=random_state, warm_start=True)
    gb_2.set_params(scoring=check_scoring(gb_2))
    gb_2.fit(X, y)
    random_seed_2_1 = gb_2._random_seed
    gb_2.fit(X, y)
    random_seed_2_2 = gb_2._random_seed
    if rng_type == 'none':
        assert random_seed_1_1 != random_seed_1_2 != random_seed_2_1
    elif rng_type == 'int':
        assert random_seed_1_1 == random_seed_1_2 == random_seed_2_1
    else:
        assert random_seed_1_1 == random_seed_2_1 != random_seed_1_2
    assert random_seed_2_1 == random_seed_2_2