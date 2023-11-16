"""
Testing for the bagging ensemble module (sklearn.ensemble.bagging).
"""
from itertools import cycle, product
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes, load_iris, make_hastie_10_2
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
rng = check_random_state(0)
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

def test_classification():
    if False:
        i = 10
        return i + 15
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, random_state=rng)
    grid = ParameterGrid({'max_samples': [0.5, 1.0], 'max_features': [1, 4], 'bootstrap': [True, False], 'bootstrap_features': [True, False]})
    estimators = [None, DummyClassifier(), Perceptron(max_iter=20), DecisionTreeClassifier(max_depth=2), KNeighborsClassifier(), SVC()]
    for (params, estimator) in zip(grid, cycle(estimators)):
        BaggingClassifier(estimator=estimator, random_state=rng, n_estimators=2, **params).fit(X_train, y_train).predict(X_test)

@pytest.mark.parametrize('sparse_container, params, method', product(CSR_CONTAINERS + CSC_CONTAINERS, [{'max_samples': 0.5, 'max_features': 2, 'bootstrap': True, 'bootstrap_features': True}, {'max_samples': 1.0, 'max_features': 4, 'bootstrap': True, 'bootstrap_features': True}, {'max_features': 2, 'bootstrap': False, 'bootstrap_features': True}, {'max_samples': 0.5, 'bootstrap': True, 'bootstrap_features': False}], ['predict', 'predict_proba', 'predict_log_proba', 'decision_function']))
def test_sparse_classification(sparse_container, params, method):
    if False:
        for i in range(10):
            print('nop')

    class CustomSVC(SVC):
        """SVC variant that records the nature of the training set"""

        def fit(self, X, y):
            if False:
                i = 10
                return i + 15
            super().fit(X, y)
            self.data_type_ = type(X)
            return self
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(scale(iris.data), iris.target, random_state=rng)
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    sparse_classifier = BaggingClassifier(estimator=CustomSVC(kernel='linear', decision_function_shape='ovr'), random_state=1, **params).fit(X_train_sparse, y_train)
    sparse_results = getattr(sparse_classifier, method)(X_test_sparse)
    dense_classifier = BaggingClassifier(estimator=CustomSVC(kernel='linear', decision_function_shape='ovr'), random_state=1, **params).fit(X_train, y_train)
    dense_results = getattr(dense_classifier, method)(X_test)
    assert_array_almost_equal(sparse_results, dense_results)
    sparse_type = type(X_train_sparse)
    types = [i.data_type_ for i in sparse_classifier.estimators_]
    assert all([t == sparse_type for t in types])

def test_regression():
    if False:
        while True:
            i = 10
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data[:50], diabetes.target[:50], random_state=rng)
    grid = ParameterGrid({'max_samples': [0.5, 1.0], 'max_features': [0.5, 1.0], 'bootstrap': [True, False], 'bootstrap_features': [True, False]})
    for estimator in [None, DummyRegressor(), DecisionTreeRegressor(), KNeighborsRegressor(), SVR()]:
        for params in grid:
            BaggingRegressor(estimator=estimator, random_state=rng, **params).fit(X_train, y_train).predict(X_test)

@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS)
def test_sparse_regression(sparse_container):
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data[:50], diabetes.target[:50], random_state=rng)

    class CustomSVR(SVR):
        """SVC variant that records the nature of the training set"""

        def fit(self, X, y):
            if False:
                for i in range(10):
                    print('nop')
            super().fit(X, y)
            self.data_type_ = type(X)
            return self
    parameter_sets = [{'max_samples': 0.5, 'max_features': 2, 'bootstrap': True, 'bootstrap_features': True}, {'max_samples': 1.0, 'max_features': 4, 'bootstrap': True, 'bootstrap_features': True}, {'max_features': 2, 'bootstrap': False, 'bootstrap_features': True}, {'max_samples': 0.5, 'bootstrap': True, 'bootstrap_features': False}]
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    for params in parameter_sets:
        sparse_classifier = BaggingRegressor(estimator=CustomSVR(), random_state=1, **params).fit(X_train_sparse, y_train)
        sparse_results = sparse_classifier.predict(X_test_sparse)
        dense_results = BaggingRegressor(estimator=CustomSVR(), random_state=1, **params).fit(X_train, y_train).predict(X_test)
        sparse_type = type(X_train_sparse)
        types = [i.data_type_ for i in sparse_classifier.estimators_]
        assert_array_almost_equal(sparse_results, dense_results)
        assert all([t == sparse_type for t in types])
        assert_array_almost_equal(sparse_results, dense_results)

class DummySizeEstimator(BaseEstimator):

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        self.training_size_ = X.shape[0]
        self.training_hash_ = joblib.hash(X)

    def predict(self, X):
        if False:
            print('Hello World!')
        return np.ones(X.shape[0])

def test_bootstrap_samples():
    if False:
        i = 10
        return i + 15
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data, diabetes.target, random_state=rng)
    estimator = DecisionTreeRegressor().fit(X_train, y_train)
    ensemble = BaggingRegressor(estimator=DecisionTreeRegressor(), max_samples=1.0, bootstrap=False, random_state=rng).fit(X_train, y_train)
    assert estimator.score(X_train, y_train) == ensemble.score(X_train, y_train)
    ensemble = BaggingRegressor(estimator=DecisionTreeRegressor(), max_samples=1.0, bootstrap=True, random_state=rng).fit(X_train, y_train)
    assert estimator.score(X_train, y_train) > ensemble.score(X_train, y_train)
    ensemble = BaggingRegressor(estimator=DummySizeEstimator(), bootstrap=True).fit(X_train, y_train)
    training_hash = []
    for estimator in ensemble.estimators_:
        assert estimator.training_size_ == X_train.shape[0]
        training_hash.append(estimator.training_hash_)
    assert len(set(training_hash)) == len(training_hash)

def test_bootstrap_features():
    if False:
        i = 10
        return i + 15
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data, diabetes.target, random_state=rng)
    ensemble = BaggingRegressor(estimator=DecisionTreeRegressor(), max_features=1.0, bootstrap_features=False, random_state=rng).fit(X_train, y_train)
    for features in ensemble.estimators_features_:
        assert diabetes.data.shape[1] == np.unique(features).shape[0]
    ensemble = BaggingRegressor(estimator=DecisionTreeRegressor(), max_features=1.0, bootstrap_features=True, random_state=rng).fit(X_train, y_train)
    for features in ensemble.estimators_features_:
        assert diabetes.data.shape[1] > np.unique(features).shape[0]

def test_probability():
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, random_state=rng)
    with np.errstate(divide='ignore', invalid='ignore'):
        ensemble = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=rng).fit(X_train, y_train)
        assert_array_almost_equal(np.sum(ensemble.predict_proba(X_test), axis=1), np.ones(len(X_test)))
        assert_array_almost_equal(ensemble.predict_proba(X_test), np.exp(ensemble.predict_log_proba(X_test)))
        ensemble = BaggingClassifier(estimator=LogisticRegression(), random_state=rng, max_samples=5).fit(X_train, y_train)
        assert_array_almost_equal(np.sum(ensemble.predict_proba(X_test), axis=1), np.ones(len(X_test)))
        assert_array_almost_equal(ensemble.predict_proba(X_test), np.exp(ensemble.predict_log_proba(X_test)))

def test_oob_score_classification():
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, random_state=rng)
    for estimator in [DecisionTreeClassifier(), SVC()]:
        clf = BaggingClassifier(estimator=estimator, n_estimators=100, bootstrap=True, oob_score=True, random_state=rng).fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        assert abs(test_score - clf.oob_score_) < 0.1
        warn_msg = 'Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.'
        with pytest.warns(UserWarning, match=warn_msg):
            clf = BaggingClassifier(estimator=estimator, n_estimators=1, bootstrap=True, oob_score=True, random_state=rng)
            clf.fit(X_train, y_train)

def test_oob_score_regression():
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data, diabetes.target, random_state=rng)
    clf = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=50, bootstrap=True, oob_score=True, random_state=rng).fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    assert abs(test_score - clf.oob_score_) < 0.1
    warn_msg = 'Some inputs do not have OOB scores. This probably means too few estimators were used to compute any reliable oob estimates.'
    with pytest.warns(UserWarning, match=warn_msg):
        regr = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=1, bootstrap=True, oob_score=True, random_state=rng)
        regr.fit(X_train, y_train)

def test_single_estimator():
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data, diabetes.target, random_state=rng)
    clf1 = BaggingRegressor(estimator=KNeighborsRegressor(), n_estimators=1, bootstrap=False, bootstrap_features=False, random_state=rng).fit(X_train, y_train)
    clf2 = KNeighborsRegressor().fit(X_train, y_train)
    assert_array_almost_equal(clf1.predict(X_test), clf2.predict(X_test))

def test_error():
    if False:
        print('Hello World!')
    (X, y) = (iris.data, iris.target)
    base = DecisionTreeClassifier()
    assert not hasattr(BaggingClassifier(base).fit(X, y), 'decision_function')

def test_parallel_classification():
    if False:
        print('Hello World!')
    (X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, random_state=0)
    ensemble = BaggingClassifier(DecisionTreeClassifier(), n_jobs=3, random_state=0).fit(X_train, y_train)
    y1 = ensemble.predict_proba(X_test)
    ensemble.set_params(n_jobs=1)
    y2 = ensemble.predict_proba(X_test)
    assert_array_almost_equal(y1, y2)
    ensemble = BaggingClassifier(DecisionTreeClassifier(), n_jobs=1, random_state=0).fit(X_train, y_train)
    y3 = ensemble.predict_proba(X_test)
    assert_array_almost_equal(y1, y3)
    ensemble = BaggingClassifier(SVC(decision_function_shape='ovr'), n_jobs=3, random_state=0).fit(X_train, y_train)
    decisions1 = ensemble.decision_function(X_test)
    ensemble.set_params(n_jobs=1)
    decisions2 = ensemble.decision_function(X_test)
    assert_array_almost_equal(decisions1, decisions2)
    ensemble = BaggingClassifier(SVC(decision_function_shape='ovr'), n_jobs=1, random_state=0).fit(X_train, y_train)
    decisions3 = ensemble.decision_function(X_test)
    assert_array_almost_equal(decisions1, decisions3)

def test_parallel_regression():
    if False:
        while True:
            i = 10
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data, diabetes.target, random_state=rng)
    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=3, random_state=0).fit(X_train, y_train)
    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y2)
    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=1, random_state=0).fit(X_train, y_train)
    y3 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y3)

def test_gridsearch():
    if False:
        i = 10
        return i + 15
    (X, y) = (iris.data, iris.target)
    y[y == 2] = 1
    parameters = {'n_estimators': (1, 2), 'estimator__C': (1, 2)}
    GridSearchCV(BaggingClassifier(SVC()), parameters, scoring='roc_auc').fit(X, y)

def test_estimator():
    if False:
        for i in range(10):
            print('nop')
    rng = check_random_state(0)
    (X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, random_state=rng)
    ensemble = BaggingClassifier(None, n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, DecisionTreeClassifier)
    ensemble = BaggingClassifier(DecisionTreeClassifier(), n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, DecisionTreeClassifier)
    ensemble = BaggingClassifier(Perceptron(), n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, Perceptron)
    (X_train, X_test, y_train, y_test) = train_test_split(diabetes.data, diabetes.target, random_state=rng)
    ensemble = BaggingRegressor(None, n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, DecisionTreeRegressor)
    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, DecisionTreeRegressor)
    ensemble = BaggingRegressor(SVR(), n_jobs=3, random_state=0).fit(X_train, y_train)
    assert isinstance(ensemble.estimator_, SVR)

def test_bagging_with_pipeline():
    if False:
        return 10
    estimator = BaggingClassifier(make_pipeline(SelectKBest(k=1), DecisionTreeClassifier()), max_features=2)
    estimator.fit(iris.data, iris.target)
    assert isinstance(estimator[0].steps[-1][1].random_state, int)

class DummyZeroEstimator(BaseEstimator):

    def fit(self, X, y):
        if False:
            while True:
                i = 10
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if False:
            return 10
        return self.classes_[np.zeros(X.shape[0], dtype=int)]

def test_bagging_sample_weight_unsupported_but_passed():
    if False:
        i = 10
        return i + 15
    estimator = BaggingClassifier(DummyZeroEstimator())
    rng = check_random_state(0)
    estimator.fit(iris.data, iris.target).predict(iris.data)
    with pytest.raises(ValueError):
        estimator.fit(iris.data, iris.target, sample_weight=rng.randint(10, size=iris.data.shape[0]))

def test_warm_start(random_state=42):
    if False:
        print('Hello World!')
    (X, y) = make_hastie_10_2(n_samples=20, random_state=1)
    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = BaggingClassifier(n_estimators=n_estimators, random_state=random_state, warm_start=True)
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators
    clf_no_ws = BaggingClassifier(n_estimators=10, random_state=random_state, warm_start=False)
    clf_no_ws.fit(X, y)
    assert set([tree.random_state for tree in clf_ws]) == set([tree.random_state for tree in clf_no_ws])

def test_warm_start_smaller_n_estimators():
    if False:
        print('Hello World!')
    (X, y) = make_hastie_10_2(n_samples=20, random_state=1)
    clf = BaggingClassifier(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    clf.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_warm_start_equal_n_estimators():
    if False:
        while True:
            i = 10
    (X, y) = make_hastie_10_2(n_samples=20, random_state=1)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=43)
    clf = BaggingClassifier(n_estimators=5, warm_start=True, random_state=83)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    X_train += 1.0
    warn_msg = 'Warm-start fitting without increasing n_estimators does not'
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X_train, y_train)
    assert_array_equal(y_pred, clf.predict(X_test))

def test_warm_start_equivalence():
    if False:
        print('Hello World!')
    (X, y) = make_hastie_10_2(n_samples=20, random_state=1)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=43)
    clf_ws = BaggingClassifier(n_estimators=5, warm_start=True, random_state=3141)
    clf_ws.fit(X_train, y_train)
    clf_ws.set_params(n_estimators=10)
    clf_ws.fit(X_train, y_train)
    y1 = clf_ws.predict(X_test)
    clf = BaggingClassifier(n_estimators=10, warm_start=False, random_state=3141)
    clf.fit(X_train, y_train)
    y2 = clf.predict(X_test)
    assert_array_almost_equal(y1, y2)

def test_warm_start_with_oob_score_fails():
    if False:
        i = 10
        return i + 15
    (X, y) = make_hastie_10_2(n_samples=20, random_state=1)
    clf = BaggingClassifier(n_estimators=5, warm_start=True, oob_score=True)
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_oob_score_removed_on_warm_start():
    if False:
        return 10
    (X, y) = make_hastie_10_2(n_samples=100, random_state=1)
    clf = BaggingClassifier(n_estimators=5, oob_score=True)
    clf.fit(X, y)
    clf.set_params(warm_start=True, oob_score=False, n_estimators=10)
    clf.fit(X, y)
    with pytest.raises(AttributeError):
        getattr(clf, 'oob_score_')

def test_oob_score_consistency():
    if False:
        return 10
    (X, y) = make_hastie_10_2(n_samples=200, random_state=1)
    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, oob_score=True, random_state=1)
    assert bagging.fit(X, y).oob_score_ == bagging.fit(X, y).oob_score_

def test_estimators_samples():
    if False:
        return 10
    (X, y) = make_hastie_10_2(n_samples=200, random_state=1)
    bagging = BaggingClassifier(LogisticRegression(), max_samples=0.5, max_features=0.5, random_state=1, bootstrap=False)
    bagging.fit(X, y)
    estimators_samples = bagging.estimators_samples_
    estimators_features = bagging.estimators_features_
    estimators = bagging.estimators_
    assert len(estimators_samples) == len(estimators)
    assert len(estimators_samples[0]) == len(X) // 2
    assert estimators_samples[0].dtype.kind == 'i'
    estimator_index = 0
    estimator_samples = estimators_samples[estimator_index]
    estimator_features = estimators_features[estimator_index]
    estimator = estimators[estimator_index]
    X_train = X[estimator_samples][:, estimator_features]
    y_train = y[estimator_samples]
    orig_coefs = estimator.coef_
    estimator.fit(X_train, y_train)
    new_coefs = estimator.coef_
    assert_array_almost_equal(orig_coefs, new_coefs)

def test_estimators_samples_deterministic():
    if False:
        i = 10
        return i + 15
    iris = load_iris()
    (X, y) = (iris.data, iris.target)
    base_pipeline = make_pipeline(SparseRandomProjection(n_components=2), LogisticRegression())
    clf = BaggingClassifier(estimator=base_pipeline, max_samples=0.5, random_state=0)
    clf.fit(X, y)
    pipeline_estimator_coef = clf.estimators_[0].steps[-1][1].coef_.copy()
    estimator = clf.estimators_[0]
    estimator_sample = clf.estimators_samples_[0]
    estimator_feature = clf.estimators_features_[0]
    X_train = X[estimator_sample][:, estimator_feature]
    y_train = y[estimator_sample]
    estimator.fit(X_train, y_train)
    assert_array_equal(estimator.steps[-1][1].coef_, pipeline_estimator_coef)

def test_max_samples_consistency():
    if False:
        return 10
    max_samples = 100
    (X, y) = make_hastie_10_2(n_samples=2 * max_samples, random_state=1)
    bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=max_samples, max_features=0.5, random_state=1)
    bagging.fit(X, y)
    assert bagging._max_samples == max_samples

def test_set_oob_score_label_encoding():
    if False:
        for i in range(10):
            print('nop')
    random_state = 5
    X = [[-1], [0], [1]] * 5
    Y1 = ['A', 'B', 'C'] * 5
    Y2 = [-1, 0, 1] * 5
    Y3 = [0, 1, 2] * 5
    x1 = BaggingClassifier(oob_score=True, random_state=random_state).fit(X, Y1).oob_score_
    x2 = BaggingClassifier(oob_score=True, random_state=random_state).fit(X, Y2).oob_score_
    x3 = BaggingClassifier(oob_score=True, random_state=random_state).fit(X, Y3).oob_score_
    assert [x1, x2] == [x3, x3]

def replace(X):
    if False:
        i = 10
        return i + 15
    X = X.astype('float', copy=True)
    X[~np.isfinite(X)] = 0
    return X

def test_bagging_regressor_with_missing_inputs():
    if False:
        return 10
    X = np.array([[1, 3, 5], [2, None, 6], [2, np.nan, 6], [2, np.inf, 6], [2, -np.inf, 6]])
    y_values = [np.array([2, 3, 3, 3, 3]), np.array([[2, 1, 9], [3, 6, 8], [3, 6, 8], [3, 6, 8], [3, 6, 8]])]
    for y in y_values:
        regressor = DecisionTreeRegressor()
        pipeline = make_pipeline(FunctionTransformer(replace), regressor)
        pipeline.fit(X, y).predict(X)
        bagging_regressor = BaggingRegressor(pipeline)
        y_hat = bagging_regressor.fit(X, y).predict(X)
        assert y.shape == y_hat.shape
        regressor = DecisionTreeRegressor()
        pipeline = make_pipeline(regressor)
        with pytest.raises(ValueError):
            pipeline.fit(X, y)
        bagging_regressor = BaggingRegressor(pipeline)
        with pytest.raises(ValueError):
            bagging_regressor.fit(X, y)

def test_bagging_classifier_with_missing_inputs():
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[1, 3, 5], [2, None, 6], [2, np.nan, 6], [2, np.inf, 6], [2, -np.inf, 6]])
    y = np.array([3, 6, 6, 6, 6])
    classifier = DecisionTreeClassifier()
    pipeline = make_pipeline(FunctionTransformer(replace), classifier)
    pipeline.fit(X, y).predict(X)
    bagging_classifier = BaggingClassifier(pipeline)
    bagging_classifier.fit(X, y)
    y_hat = bagging_classifier.predict(X)
    assert y.shape == y_hat.shape
    bagging_classifier.predict_log_proba(X)
    bagging_classifier.predict_proba(X)
    classifier = DecisionTreeClassifier()
    pipeline = make_pipeline(classifier)
    with pytest.raises(ValueError):
        pipeline.fit(X, y)
    bagging_classifier = BaggingClassifier(pipeline)
    with pytest.raises(ValueError):
        bagging_classifier.fit(X, y)

def test_bagging_small_max_features():
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    bagging = BaggingClassifier(LogisticRegression(), max_features=0.3, random_state=1)
    bagging.fit(X, y)

def test_bagging_get_estimators_indices():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = rng.randn(13, 4)
    y = np.arange(13)

    class MyEstimator(DecisionTreeRegressor):
        """An estimator which stores y indices information at fit."""

        def fit(self, X, y):
            if False:
                while True:
                    i = 10
            self._sample_indices = y
    clf = BaggingRegressor(estimator=MyEstimator(), n_estimators=1, random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.estimators_[0]._sample_indices, clf.estimators_samples_[0])

@pytest.mark.parametrize('Bagging, Estimator', [(BaggingClassifier, DecisionTreeClassifier), (BaggingRegressor, DecisionTreeRegressor)])
def test_base_estimator_argument_deprecated(Bagging, Estimator):
    if False:
        return 10
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = Bagging(base_estimator=Estimator(), n_estimators=10)
    warn_msg = '`base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.'
    with pytest.warns(FutureWarning, match=warn_msg):
        model.fit(X, y)

@pytest.mark.parametrize('Bagging', [BaggingClassifier, BaggingClassifier])
def test_base_estimator_property_deprecated(Bagging):
    if False:
        print('Hello World!')
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = Bagging()
    model.fit(X, y)
    warn_msg = 'Attribute `base_estimator_` was deprecated in version 1.2 and will be removed in 1.4. Use `estimator_` instead.'
    with pytest.warns(FutureWarning, match=warn_msg):
        model.base_estimator_

def test_deprecated_base_estimator_has_decision_function():
    if False:
        return 10
    'Check that `BaggingClassifier` delegate to classifier with\n    `decision_function`.'
    iris = load_iris()
    (X, y) = (iris.data, iris.target)
    clf = BaggingClassifier(base_estimator=SVC())
    assert hasattr(clf, 'decision_function')
    warn_msg = '`base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.'
    with pytest.warns(FutureWarning, match=warn_msg):
        y_decision = clf.fit(X, y).decision_function(X)
    assert y_decision.shape == (150, 3)

@pytest.mark.parametrize('bagging, expected_allow_nan', [(BaggingClassifier(HistGradientBoostingClassifier(max_iter=1)), True), (BaggingRegressor(HistGradientBoostingRegressor(max_iter=1)), True), (BaggingClassifier(LogisticRegression()), False), (BaggingRegressor(SVR()), False)])
def test_bagging_allow_nan_tag(bagging, expected_allow_nan):
    if False:
        print('Hello World!')
    'Check that bagging inherits allow_nan tag.'
    assert bagging._get_tags()['allow_nan'] == expected_allow_nan