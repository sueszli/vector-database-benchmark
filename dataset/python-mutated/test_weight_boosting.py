"""Testing for the boost module (sklearn.ensemble.boost)."""
import re
import numpy as np
import pytest
from sklearn import datasets
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.utils._mocking import NoSampleWeightWrapper
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal, assert_array_equal, assert_array_less
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS, DOK_CONTAINERS, LIL_CONTAINERS
rng = np.random.RandomState(0)
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y_class = ['foo', 'foo', 'foo', 1, 1, 1]
y_regr = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
y_t_class = ['foo', 1, 1]
y_t_regr = [-1, 1, 1]
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
(iris.data, iris.target) = shuffle(iris.data, iris.target, random_state=rng)
diabetes = datasets.load_diabetes()
(diabetes.data, diabetes.target) = shuffle(diabetes.data, diabetes.target, random_state=rng)

def test_samme_proba():
    if False:
        i = 10
        return i + 15
    probs = np.array([[1, 1e-06, 0], [0.19, 0.6, 0.2], [-999, 0.51, 0.5], [1e-06, 1, 1e-09]])
    probs /= np.abs(probs.sum(axis=1))[:, np.newaxis]

    class MockEstimator:

        def predict_proba(self, X):
            if False:
                for i in range(10):
                    print('nop')
            assert_array_equal(X.shape, probs.shape)
            return probs
    mock = MockEstimator()
    samme_proba = _samme_proba(mock, 3, np.ones_like(probs))
    assert_array_equal(samme_proba.shape, probs.shape)
    assert np.isfinite(samme_proba).all()
    assert_array_equal(np.argmin(samme_proba, axis=1), [2, 0, 0, 2])
    assert_array_equal(np.argmax(samme_proba, axis=1), [0, 1, 1, 1])

def test_oneclass_adaboost_proba():
    if False:
        while True:
            i = 10
    y_t = np.ones(len(X))
    clf = AdaBoostClassifier(algorithm='SAMME').fit(X, y_t)
    assert_array_almost_equal(clf.predict_proba(X), np.ones((len(X), 1)))

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_classification_toy(algorithm):
    if False:
        while True:
            i = 10
    clf = AdaBoostClassifier(algorithm=algorithm, random_state=0)
    clf.fit(X, y_class)
    assert_array_equal(clf.predict(T), y_t_class)
    assert_array_equal(np.unique(np.asarray(y_t_class)), clf.classes_)
    assert clf.predict_proba(T).shape == (len(T), 2)
    assert clf.decision_function(T).shape == (len(T),)

def test_regression_toy():
    if False:
        i = 10
        return i + 15
    clf = AdaBoostRegressor(random_state=0)
    clf.fit(X, y_regr)
    assert_array_equal(clf.predict(T), y_t_regr)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
def test_iris():
    if False:
        print('Hello World!')
    classes = np.unique(iris.target)
    clf_samme = prob_samme = None
    for alg in ['SAMME', 'SAMME.R']:
        clf = AdaBoostClassifier(algorithm=alg)
        clf.fit(iris.data, iris.target)
        assert_array_equal(classes, clf.classes_)
        proba = clf.predict_proba(iris.data)
        if alg == 'SAMME':
            clf_samme = clf
            prob_samme = proba
        assert proba.shape[1] == len(classes)
        assert clf.decision_function(iris.data).shape[1] == len(classes)
        score = clf.score(iris.data, iris.target)
        assert score > 0.9, 'Failed with algorithm %s and score = %f' % (alg, score)
        assert len(clf.estimators_) > 1
        assert len(set((est.random_state for est in clf.estimators_))) == len(clf.estimators_)
    clf_samme.algorithm = 'SAMME.R'
    assert_array_less(0, np.abs(clf_samme.predict_proba(iris.data) - prob_samme))

@pytest.mark.parametrize('loss', ['linear', 'square', 'exponential'])
def test_diabetes(loss):
    if False:
        i = 10
        return i + 15
    reg = AdaBoostRegressor(loss=loss, random_state=0)
    reg.fit(diabetes.data, diabetes.target)
    score = reg.score(diabetes.data, diabetes.target)
    assert score > 0.55
    assert len(reg.estimators_) > 1
    assert len(set((est.random_state for est in reg.estimators_))) == len(reg.estimators_)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_staged_predict(algorithm):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    iris_weights = rng.randint(10, size=iris.target.shape)
    diabetes_weights = rng.randint(10, size=diabetes.target.shape)
    clf = AdaBoostClassifier(algorithm=algorithm, n_estimators=10)
    clf.fit(iris.data, iris.target, sample_weight=iris_weights)
    predictions = clf.predict(iris.data)
    staged_predictions = [p for p in clf.staged_predict(iris.data)]
    proba = clf.predict_proba(iris.data)
    staged_probas = [p for p in clf.staged_predict_proba(iris.data)]
    score = clf.score(iris.data, iris.target, sample_weight=iris_weights)
    staged_scores = [s for s in clf.staged_score(iris.data, iris.target, sample_weight=iris_weights)]
    assert len(staged_predictions) == 10
    assert_array_almost_equal(predictions, staged_predictions[-1])
    assert len(staged_probas) == 10
    assert_array_almost_equal(proba, staged_probas[-1])
    assert len(staged_scores) == 10
    assert_array_almost_equal(score, staged_scores[-1])
    clf = AdaBoostRegressor(n_estimators=10, random_state=0)
    clf.fit(diabetes.data, diabetes.target, sample_weight=diabetes_weights)
    predictions = clf.predict(diabetes.data)
    staged_predictions = [p for p in clf.staged_predict(diabetes.data)]
    score = clf.score(diabetes.data, diabetes.target, sample_weight=diabetes_weights)
    staged_scores = [s for s in clf.staged_score(diabetes.data, diabetes.target, sample_weight=diabetes_weights)]
    assert len(staged_predictions) == 10
    assert_array_almost_equal(predictions, staged_predictions[-1])
    assert len(staged_scores) == 10
    assert_array_almost_equal(score, staged_scores[-1])

def test_gridsearch():
    if False:
        i = 10
        return i + 15
    boost = AdaBoostClassifier(estimator=DecisionTreeClassifier())
    parameters = {'n_estimators': (1, 2), 'estimator__max_depth': (1, 2), 'algorithm': ('SAMME', 'SAMME.R')}
    clf = GridSearchCV(boost, parameters)
    clf.fit(iris.data, iris.target)
    boost = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=0)
    parameters = {'n_estimators': (1, 2), 'estimator__max_depth': (1, 2)}
    clf = GridSearchCV(boost, parameters)
    clf.fit(diabetes.data, diabetes.target)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
def test_pickle():
    if False:
        print('Hello World!')
    import pickle
    for alg in ['SAMME', 'SAMME.R']:
        obj = AdaBoostClassifier(algorithm=alg)
        obj.fit(iris.data, iris.target)
        score = obj.score(iris.data, iris.target)
        s = pickle.dumps(obj)
        obj2 = pickle.loads(s)
        assert type(obj2) == obj.__class__
        score2 = obj2.score(iris.data, iris.target)
        assert score == score2
    obj = AdaBoostRegressor(random_state=0)
    obj.fit(diabetes.data, diabetes.target)
    score = obj.score(diabetes.data, diabetes.target)
    s = pickle.dumps(obj)
    obj2 = pickle.loads(s)
    assert type(obj2) == obj.__class__
    score2 = obj2.score(diabetes.data, diabetes.target)
    assert score == score2

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
def test_importances():
    if False:
        return 10
    (X, y) = datasets.make_classification(n_samples=2000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, shuffle=False, random_state=1)
    for alg in ['SAMME', 'SAMME.R']:
        clf = AdaBoostClassifier(algorithm=alg)
        clf.fit(X, y)
        importances = clf.feature_importances_
        assert importances.shape[0] == 10
        assert (importances[:3, np.newaxis] >= importances[3:]).all()

def test_adaboost_classifier_sample_weight_error():
    if False:
        while True:
            i = 10
    clf = AdaBoostClassifier()
    msg = re.escape('sample_weight.shape == (1,), expected (6,)')
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y_class, sample_weight=np.asarray([-1]))

def test_estimator():
    if False:
        i = 10
        return i + 15
    from sklearn.ensemble import RandomForestClassifier
    clf = AdaBoostClassifier(RandomForestClassifier(), algorithm='SAMME')
    clf.fit(X, y_regr)
    clf = AdaBoostClassifier(SVC(), algorithm='SAMME')
    clf.fit(X, y_class)
    from sklearn.ensemble import RandomForestRegressor
    clf = AdaBoostRegressor(RandomForestRegressor(), random_state=0)
    clf.fit(X, y_regr)
    clf = AdaBoostRegressor(SVR(), random_state=0)
    clf.fit(X, y_regr)
    X_fail = [[1, 1], [1, 1], [1, 1], [1, 1]]
    y_fail = ['foo', 'bar', 1, 2]
    clf = AdaBoostClassifier(SVC(), algorithm='SAMME')
    with pytest.raises(ValueError, match='worse than random'):
        clf.fit(X_fail, y_fail)

def test_sample_weights_infinite():
    if False:
        for i in range(10):
            print('nop')
    msg = 'Sample weights have reached infinite values'
    clf = AdaBoostClassifier(n_estimators=30, learning_rate=23.0, algorithm='SAMME')
    with pytest.warns(UserWarning, match=msg):
        clf.fit(iris.data, iris.target)

@pytest.mark.parametrize('sparse_container, expected_internal_type', zip([*CSC_CONTAINERS, *CSR_CONTAINERS, *LIL_CONTAINERS, *COO_CONTAINERS, *DOK_CONTAINERS], CSC_CONTAINERS + 4 * CSR_CONTAINERS))
def test_sparse_classification(sparse_container, expected_internal_type):
    if False:
        print('Hello World!')

    class CustomSVC(SVC):
        """SVC variant that records the nature of the training set."""

        def fit(self, X, y, sample_weight=None):
            if False:
                i = 10
                return i + 15
            'Modification on fit caries data type for later verification.'
            super().fit(X, y, sample_weight=sample_weight)
            self.data_type_ = type(X)
            return self
    (X, y) = datasets.make_multilabel_classification(n_classes=1, n_samples=15, n_features=5, random_state=42)
    y = np.ravel(y)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    sparse_classifier = AdaBoostClassifier(estimator=CustomSVC(probability=True), random_state=1, algorithm='SAMME').fit(X_train_sparse, y_train)
    dense_classifier = AdaBoostClassifier(estimator=CustomSVC(probability=True), random_state=1, algorithm='SAMME').fit(X_train, y_train)
    sparse_clf_results = sparse_classifier.predict(X_test_sparse)
    dense_clf_results = dense_classifier.predict(X_test)
    assert_array_equal(sparse_clf_results, dense_clf_results)
    sparse_clf_results = sparse_classifier.decision_function(X_test_sparse)
    dense_clf_results = dense_classifier.decision_function(X_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    sparse_clf_results = sparse_classifier.predict_log_proba(X_test_sparse)
    dense_clf_results = dense_classifier.predict_log_proba(X_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    sparse_clf_results = sparse_classifier.predict_proba(X_test_sparse)
    dense_clf_results = dense_classifier.predict_proba(X_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    sparse_clf_results = sparse_classifier.score(X_test_sparse, y_test)
    dense_clf_results = dense_classifier.score(X_test, y_test)
    assert_array_almost_equal(sparse_clf_results, dense_clf_results)
    sparse_clf_results = sparse_classifier.staged_decision_function(X_test_sparse)
    dense_clf_results = dense_classifier.staged_decision_function(X_test)
    for (sparse_clf_res, dense_clf_res) in zip(sparse_clf_results, dense_clf_results):
        assert_array_almost_equal(sparse_clf_res, dense_clf_res)
    sparse_clf_results = sparse_classifier.staged_predict(X_test_sparse)
    dense_clf_results = dense_classifier.staged_predict(X_test)
    for (sparse_clf_res, dense_clf_res) in zip(sparse_clf_results, dense_clf_results):
        assert_array_equal(sparse_clf_res, dense_clf_res)
    sparse_clf_results = sparse_classifier.staged_predict_proba(X_test_sparse)
    dense_clf_results = dense_classifier.staged_predict_proba(X_test)
    for (sparse_clf_res, dense_clf_res) in zip(sparse_clf_results, dense_clf_results):
        assert_array_almost_equal(sparse_clf_res, dense_clf_res)
    sparse_clf_results = sparse_classifier.staged_score(X_test_sparse, y_test)
    dense_clf_results = dense_classifier.staged_score(X_test, y_test)
    for (sparse_clf_res, dense_clf_res) in zip(sparse_clf_results, dense_clf_results):
        assert_array_equal(sparse_clf_res, dense_clf_res)
    types = [i.data_type_ for i in sparse_classifier.estimators_]
    assert all([t == expected_internal_type for t in types])

@pytest.mark.parametrize('sparse_container, expected_internal_type', zip([*CSC_CONTAINERS, *CSR_CONTAINERS, *LIL_CONTAINERS, *COO_CONTAINERS, *DOK_CONTAINERS], CSC_CONTAINERS + 4 * CSR_CONTAINERS))
def test_sparse_regression(sparse_container, expected_internal_type):
    if False:
        return 10

    class CustomSVR(SVR):
        """SVR variant that records the nature of the training set."""

        def fit(self, X, y, sample_weight=None):
            if False:
                i = 10
                return i + 15
            'Modification on fit caries data type for later verification.'
            super().fit(X, y, sample_weight=sample_weight)
            self.data_type_ = type(X)
            return self
    (X, y) = datasets.make_regression(n_samples=15, n_features=50, n_targets=1, random_state=42)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    sparse_regressor = AdaBoostRegressor(estimator=CustomSVR(), random_state=1).fit(X_train_sparse, y_train)
    dense_regressor = AdaBoostRegressor(estimator=CustomSVR(), random_state=1).fit(X_train, y_train)
    sparse_regr_results = sparse_regressor.predict(X_test_sparse)
    dense_regr_results = dense_regressor.predict(X_test)
    assert_array_almost_equal(sparse_regr_results, dense_regr_results)
    sparse_regr_results = sparse_regressor.staged_predict(X_test_sparse)
    dense_regr_results = dense_regressor.staged_predict(X_test)
    for (sparse_regr_res, dense_regr_res) in zip(sparse_regr_results, dense_regr_results):
        assert_array_almost_equal(sparse_regr_res, dense_regr_res)
    types = [i.data_type_ for i in sparse_regressor.estimators_]
    assert all([t == expected_internal_type for t in types])

def test_sample_weight_adaboost_regressor():
    if False:
        return 10
    '\n    AdaBoostRegressor should work without sample_weights in the base estimator\n    The random weighted sampling is done internally in the _boost method in\n    AdaBoostRegressor.\n    '

    class DummyEstimator(BaseEstimator):

        def fit(self, X, y):
            if False:
                i = 10
                return i + 15
            pass

        def predict(self, X):
            if False:
                return 10
            return np.zeros(X.shape[0])
    boost = AdaBoostRegressor(DummyEstimator(), n_estimators=3)
    boost.fit(X, y_regr)
    assert len(boost.estimator_weights_) == len(boost.estimator_errors_)

def test_multidimensional_X():
    if False:
        while True:
            i = 10
    '\n    Check that the AdaBoost estimators can work with n-dimensional\n    data matrix\n    '
    rng = np.random.RandomState(0)
    X = rng.randn(51, 3, 3)
    yc = rng.choice([0, 1], 51)
    yr = rng.randn(51)
    boost = AdaBoostClassifier(DummyClassifier(strategy='most_frequent'), algorithm='SAMME')
    boost.fit(X, yc)
    boost.predict(X)
    boost.predict_proba(X)
    boost = AdaBoostRegressor(DummyRegressor())
    boost.fit(X, yr)
    boost.predict(X)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_adaboostclassifier_without_sample_weight(algorithm):
    if False:
        i = 10
        return i + 15
    (X, y) = (iris.data, iris.target)
    estimator = NoSampleWeightWrapper(DummyClassifier())
    clf = AdaBoostClassifier(estimator=estimator, algorithm=algorithm)
    err_msg = "{} doesn't support sample_weight".format(estimator.__class__.__name__)
    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, y)

def test_adaboostregressor_sample_weight():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(42)
    X = np.linspace(0, 100, num=1000)
    y = 0.8 * X + 0.2 + rng.rand(X.shape[0]) * 0.0001
    X = X.reshape(-1, 1)
    X[-1] *= 10
    y[-1] = 10000
    regr_no_outlier = AdaBoostRegressor(estimator=LinearRegression(), n_estimators=1, random_state=0)
    regr_with_weight = clone(regr_no_outlier)
    regr_with_outlier = clone(regr_no_outlier)
    regr_with_outlier.fit(X, y)
    regr_no_outlier.fit(X[:-1], y[:-1])
    sample_weight = np.ones_like(y)
    sample_weight[-1] = 0
    regr_with_weight.fit(X, y, sample_weight=sample_weight)
    score_with_outlier = regr_with_outlier.score(X[:-1], y[:-1])
    score_no_outlier = regr_no_outlier.score(X[:-1], y[:-1])
    score_with_weight = regr_with_weight.score(X[:-1], y[:-1])
    assert score_with_outlier < score_no_outlier
    assert score_with_outlier < score_with_weight
    assert score_no_outlier == pytest.approx(score_with_weight)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_adaboost_consistent_predict(algorithm):
    if False:
        print('Hello World!')
    (X_train, X_test, y_train, y_test) = train_test_split(*datasets.load_digits(return_X_y=True), random_state=42)
    model = AdaBoostClassifier(algorithm=algorithm, random_state=42)
    model.fit(X_train, y_train)
    assert_array_equal(np.argmax(model.predict_proba(X_test), axis=1), model.predict(X_test))

@pytest.mark.parametrize('model, X, y', [(AdaBoostClassifier(), iris.data, iris.target), (AdaBoostRegressor(), diabetes.data, diabetes.target)])
def test_adaboost_negative_weight_error(model, X, y):
    if False:
        print('Hello World!')
    sample_weight = np.ones_like(y)
    sample_weight[-1] = -10
    err_msg = 'Negative values in data passed to `sample_weight`'
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y, sample_weight=sample_weight)

def test_adaboost_numerically_stable_feature_importance_with_small_weights():
    if False:
        i = 10
        return i + 15
    "Check that we don't create NaN feature importance with numerically\n    instable inputs.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/20320\n    "
    rng = np.random.RandomState(42)
    X = rng.normal(size=(1000, 10))
    y = rng.choice([0, 1], size=1000)
    sample_weight = np.ones_like(y) * 1e-263
    tree = DecisionTreeClassifier(max_depth=10, random_state=12)
    ada_model = AdaBoostClassifier(estimator=tree, n_estimators=20, algorithm='SAMME', random_state=12)
    ada_model.fit(X, y, sample_weight=sample_weight)
    assert np.isnan(ada_model.feature_importances_).sum() == 0

@pytest.mark.parametrize('AdaBoost, Estimator', [(AdaBoostClassifier, DecisionTreeClassifier), (AdaBoostRegressor, DecisionTreeRegressor)])
def test_base_estimator_argument_deprecated(AdaBoost, Estimator):
    if False:
        i = 10
        return i + 15
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = AdaBoost(base_estimator=Estimator())
    warn_msg = '`base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.'
    with pytest.warns(FutureWarning, match=warn_msg):
        model.fit(X, y)

@pytest.mark.parametrize('AdaBoost', [AdaBoostClassifier, AdaBoostRegressor])
def test_base_estimator_argument_deprecated_none(AdaBoost):
    if False:
        i = 10
        return i + 15
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = AdaBoost(base_estimator=None)
    warn_msg = '`base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.'
    with pytest.warns(FutureWarning, match=warn_msg):
        model.fit(X, y)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('AdaBoost', [AdaBoostClassifier, AdaBoostRegressor])
def test_base_estimator_property_deprecated(AdaBoost):
    if False:
        while True:
            i = 10
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    model = AdaBoost()
    model.fit(X, y)
    warn_msg = 'Attribute `base_estimator_` was deprecated in version 1.2 and will be removed in 1.4. Use `estimator_` instead.'
    with pytest.warns(FutureWarning, match=warn_msg):
        model.base_estimator_

def test_deprecated_base_estimator_parameters_can_be_set():
    if False:
        i = 10
        return i + 15
    'Check that setting base_estimator parameters works.\n\n    During the deprecation cycle setting "base_estimator__*" params should\n    work.\n\n    Non-regression test for https://github.com/scikit-learn/scikit-learn/issues/25470\n    '
    clf = AdaBoostClassifier(DecisionTreeClassifier())
    with pytest.warns(FutureWarning, match="Parameter 'base_estimator' of"):
        clf.set_params(base_estimator__max_depth=2)

@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_adaboost_decision_function(algorithm, global_random_seed):
    if False:
        return 10
    'Check that the decision function respects the symmetric constraint for weak\n    learners.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/26520\n    '
    n_classes = 3
    (X, y) = datasets.make_classification(n_classes=n_classes, n_clusters_per_class=1, random_state=global_random_seed)
    clf = AdaBoostClassifier(n_estimators=1, random_state=global_random_seed, algorithm=algorithm).fit(X, y)
    y_score = clf.decision_function(X)
    assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
    if algorithm == 'SAMME':
        assert set(np.unique(y_score)) == {1, -1 / (n_classes - 1)}
    for y_score in clf.staged_decision_function(X):
        assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
        if algorithm == 'SAMME':
            assert set(np.unique(y_score)) == {1, -1 / (n_classes - 1)}
    clf.set_params(n_estimators=5).fit(X, y)
    y_score = clf.decision_function(X)
    assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)
    for y_score in clf.staged_decision_function(X):
        assert_allclose(y_score.sum(axis=1), 0, atol=1e-08)

def test_deprecated_samme_r_algorithm():
    if False:
        i = 10
        return i + 15
    adaboost_clf = AdaBoostClassifier(n_estimators=1)
    with pytest.warns(FutureWarning, match=re.escape('The SAMME.R algorithm (the default) is deprecated')):
        adaboost_clf.fit(X, y_class)