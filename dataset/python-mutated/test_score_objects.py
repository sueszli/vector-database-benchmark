import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, make_blobs, make_classification, make_multilabel_classification, make_regression
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, check_scoring, f1_score, fbeta_score, get_scorer, get_scorer_names, jaccard_score, log_loss, make_scorer, matthews_corrcoef, precision_score, r2_score, recall_score, roc_auc_score, top_k_accuracy_score
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer, _PassthroughScorer, _Scorer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import assert_request_equal, assert_request_is_empty
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import assert_almost_equal, assert_array_equal, ignore_warnings
from sklearn.utils.metadata_routing import MetadataRouter
REGRESSION_SCORERS = ['explained_variance', 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'neg_root_mean_squared_log_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'median_absolute_error', 'max_error', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance']
CLF_SCORERS = ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'f1', 'f1_weighted', 'f1_macro', 'f1_micro', 'roc_auc', 'average_precision', 'precision', 'precision_weighted', 'precision_macro', 'precision_micro', 'recall', 'recall_weighted', 'recall_macro', 'recall_micro', 'neg_log_loss', 'neg_brier_score', 'jaccard', 'jaccard_weighted', 'jaccard_macro', 'jaccard_micro', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 'matthews_corrcoef', 'positive_likelihood_ratio', 'neg_negative_likelihood_ratio']
CLUSTER_SCORERS = ['adjusted_rand_score', 'rand_score', 'homogeneity_score', 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score']
MULTILABEL_ONLY_SCORERS = ['precision_samples', 'recall_samples', 'f1_samples', 'jaccard_samples']
REQUIRE_POSITIVE_Y_SCORERS = ['neg_mean_poisson_deviance', 'neg_mean_gamma_deviance']

def _require_positive_y(y):
    if False:
        i = 10
        return i + 15
    'Make targets strictly positive'
    offset = abs(y.min()) + 1
    y = y + offset
    return y

def _make_estimators(X_train, y_train, y_ml_train):
    if False:
        for i in range(10):
            print('nop')
    sensible_regr = DecisionTreeRegressor(random_state=0)
    sensible_regr.fit(X_train, _require_positive_y(y_train))
    sensible_clf = DecisionTreeClassifier(random_state=0)
    sensible_clf.fit(X_train, y_train)
    sensible_ml_clf = DecisionTreeClassifier(random_state=0)
    sensible_ml_clf.fit(X_train, y_ml_train)
    return dict([(name, sensible_regr) for name in REGRESSION_SCORERS] + [(name, sensible_clf) for name in CLF_SCORERS] + [(name, sensible_clf) for name in CLUSTER_SCORERS] + [(name, sensible_ml_clf) for name in MULTILABEL_ONLY_SCORERS])
(X_mm, y_mm, y_ml_mm) = (None, None, None)
ESTIMATORS = None
TEMP_FOLDER = None

def setup_module():
    if False:
        return 10
    global X_mm, y_mm, y_ml_mm, TEMP_FOLDER, ESTIMATORS
    TEMP_FOLDER = tempfile.mkdtemp(prefix='sklearn_test_score_objects_')
    (X, y) = make_classification(n_samples=30, n_features=5, random_state=0)
    (_, y_ml) = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    filename = os.path.join(TEMP_FOLDER, 'test_data.pkl')
    joblib.dump((X, y, y_ml), filename)
    (X_mm, y_mm, y_ml_mm) = joblib.load(filename, mmap_mode='r')
    ESTIMATORS = _make_estimators(X_mm, y_mm, y_ml_mm)

def teardown_module():
    if False:
        print('Hello World!')
    global X_mm, y_mm, y_ml_mm, TEMP_FOLDER, ESTIMATORS
    (X_mm, y_mm, y_ml_mm, ESTIMATORS) = (None, None, None, None)
    shutil.rmtree(TEMP_FOLDER)

class EstimatorWithFit(BaseEstimator):
    """Dummy estimator to test scoring validators"""

    def fit(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        return self

class EstimatorWithFitAndScore:
    """Dummy estimator to test scoring validators"""

    def fit(self, X, y):
        if False:
            return 10
        return self

    def score(self, X, y):
        if False:
            while True:
                i = 10
        return 1.0

class EstimatorWithFitAndPredict:
    """Dummy estimator to test scoring validators"""

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        self.y = y
        return self

    def predict(self, X):
        if False:
            print('Hello World!')
        return self.y

class DummyScorer:
    """Dummy scorer that always returns 1."""

    def __call__(self, est, X, y):
        if False:
            return 10
        return 1

def test_all_scorers_repr():
    if False:
        print('Hello World!')
    for name in get_scorer_names():
        repr(get_scorer(name))

def check_scoring_validator_for_single_metric_usecases(scoring_validator):
    if False:
        i = 10
        return i + 15
    estimator = EstimatorWithFitAndScore()
    estimator.fit([[1]], [1])
    scorer = scoring_validator(estimator)
    assert isinstance(scorer, _PassthroughScorer)
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])
    pattern = "If no scoring is specified, the estimator passed should have a 'score' method\\. The estimator .* does not\\."
    with pytest.raises(TypeError, match=pattern):
        scoring_validator(estimator)
    scorer = scoring_validator(estimator, scoring='accuracy')
    assert_almost_equal(scorer(estimator, [[1]], [1]), 1.0)
    estimator = EstimatorWithFit()
    scorer = scoring_validator(estimator, scoring='accuracy')
    assert isinstance(scorer, _Scorer)
    assert scorer._response_method == 'predict'
    if scoring_validator is check_scoring:
        estimator = EstimatorWithFit()
        scorer = scoring_validator(estimator, allow_none=True)
        assert scorer is None

@pytest.mark.parametrize('scoring', (('accuracy',), ['precision'], {'acc': 'accuracy', 'precision': 'precision'}, ('accuracy', 'precision'), ['precision', 'accuracy'], {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}), ids=['single_tuple', 'single_list', 'dict_str', 'multi_tuple', 'multi_list', 'dict_callable'])
def test_check_scoring_and_check_multimetric_scoring(scoring):
    if False:
        while True:
            i = 10
    check_scoring_validator_for_single_metric_usecases(check_scoring)
    estimator = LinearSVC(dual='auto', random_state=0)
    estimator.fit([[1], [2], [3]], [1, 1, 0])
    scorers = _check_multimetric_scoring(estimator, scoring)
    assert isinstance(scorers, dict)
    assert sorted(scorers.keys()) == sorted(list(scoring))
    assert all([isinstance(scorer, _Scorer) for scorer in list(scorers.values())])
    assert all((scorer._response_method == 'predict' for scorer in scorers.values()))
    if 'acc' in scoring:
        assert_almost_equal(scorers['acc'](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0)
    if 'accuracy' in scoring:
        assert_almost_equal(scorers['accuracy'](estimator, [[1], [2], [3]], [1, 0, 0]), 2.0 / 3.0)
    if 'precision' in scoring:
        assert_almost_equal(scorers['precision'](estimator, [[1], [2], [3]], [1, 0, 0]), 0.5)

@pytest.mark.parametrize('scoring, msg', [((make_scorer(precision_score), make_scorer(accuracy_score)), 'One or more of the elements were callables'), ([5], 'Non-string types were found'), ((make_scorer(precision_score),), 'One or more of the elements were callables'), ((), 'Empty list was given'), (('f1', 'f1'), 'Duplicate elements were found'), ({4: 'accuracy'}, 'Non-string types were found in the keys'), ({}, 'An empty dict was passed')], ids=['tuple of callables', 'list of int', 'tuple of one callable', 'empty tuple', 'non-unique str', 'non-string key dict', 'empty dict'])
def test_check_scoring_and_check_multimetric_scoring_errors(scoring, msg):
    if False:
        print('Hello World!')
    estimator = EstimatorWithFitAndPredict()
    estimator.fit([[1]], [1])
    with pytest.raises(ValueError, match=msg):
        _check_multimetric_scoring(estimator, scoring=scoring)

def test_check_scoring_gridsearchcv():
    if False:
        print('Hello World!')
    grid = GridSearchCV(LinearSVC(dual='auto'), param_grid={'C': [0.1, 1]}, cv=3)
    scorer = check_scoring(grid, scoring='f1')
    assert isinstance(scorer, _Scorer)
    assert scorer._response_method == 'predict'
    pipe = make_pipeline(LinearSVC(dual='auto'))
    scorer = check_scoring(pipe, scoring='f1')
    assert isinstance(scorer, _Scorer)
    assert scorer._response_method == 'predict'
    scores = cross_val_score(EstimatorWithFit(), [[1], [2], [3]], [1, 0, 1], scoring=DummyScorer(), cv=3)
    assert_array_equal(scores, 1)

@pytest.mark.parametrize('scorer_name, metric', [('f1', f1_score), ('f1_weighted', partial(f1_score, average='weighted')), ('f1_macro', partial(f1_score, average='macro')), ('f1_micro', partial(f1_score, average='micro')), ('precision', precision_score), ('precision_weighted', partial(precision_score, average='weighted')), ('precision_macro', partial(precision_score, average='macro')), ('precision_micro', partial(precision_score, average='micro')), ('recall', recall_score), ('recall_weighted', partial(recall_score, average='weighted')), ('recall_macro', partial(recall_score, average='macro')), ('recall_micro', partial(recall_score, average='micro')), ('jaccard', jaccard_score), ('jaccard_weighted', partial(jaccard_score, average='weighted')), ('jaccard_macro', partial(jaccard_score, average='macro')), ('jaccard_micro', partial(jaccard_score, average='micro')), ('top_k_accuracy', top_k_accuracy_score), ('matthews_corrcoef', matthews_corrcoef)])
def test_classification_binary_scores(scorer_name, metric):
    if False:
        return 10
    (X, y) = make_blobs(random_state=0, centers=2)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf = LinearSVC(dual='auto', random_state=0)
    clf.fit(X_train, y_train)
    score = get_scorer(scorer_name)(clf, X_test, y_test)
    expected_score = metric(y_test, clf.predict(X_test))
    assert_almost_equal(score, expected_score)

@pytest.mark.parametrize('scorer_name, metric', [('accuracy', accuracy_score), ('balanced_accuracy', balanced_accuracy_score), ('f1_weighted', partial(f1_score, average='weighted')), ('f1_macro', partial(f1_score, average='macro')), ('f1_micro', partial(f1_score, average='micro')), ('precision_weighted', partial(precision_score, average='weighted')), ('precision_macro', partial(precision_score, average='macro')), ('precision_micro', partial(precision_score, average='micro')), ('recall_weighted', partial(recall_score, average='weighted')), ('recall_macro', partial(recall_score, average='macro')), ('recall_micro', partial(recall_score, average='micro')), ('jaccard_weighted', partial(jaccard_score, average='weighted')), ('jaccard_macro', partial(jaccard_score, average='macro')), ('jaccard_micro', partial(jaccard_score, average='micro'))])
def test_classification_multiclass_scores(scorer_name, metric):
    if False:
        return 10
    (X, y) = make_classification(n_classes=3, n_informative=3, n_samples=30, random_state=0)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0, stratify=y)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    score = get_scorer(scorer_name)(clf, X_test, y_test)
    expected_score = metric(y_test, clf.predict(X_test))
    assert score == pytest.approx(expected_score)

def test_custom_scorer_pickling():
    if False:
        return 10
    (X, y) = make_blobs(random_state=0, centers=2)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf = LinearSVC(dual='auto', random_state=0)
    clf.fit(X_train, y_train)
    scorer = make_scorer(fbeta_score, beta=2)
    score1 = scorer(clf, X_test, y_test)
    unpickled_scorer = pickle.loads(pickle.dumps(scorer))
    score2 = unpickled_scorer(clf, X_test, y_test)
    assert score1 == pytest.approx(score2)
    repr(fbeta_score)

def test_regression_scorers():
    if False:
        return 10
    diabetes = load_diabetes()
    (X, y) = (diabetes.data, diabetes.target)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf = Ridge()
    clf.fit(X_train, y_train)
    score1 = get_scorer('r2')(clf, X_test, y_test)
    score2 = r2_score(y_test, clf.predict(X_test))
    assert_almost_equal(score1, score2)

def test_thresholded_scorers():
    if False:
        return 10
    (X, y) = make_blobs(random_state=0, centers=2)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    score3 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    assert_almost_equal(score1, score2)
    assert_almost_equal(score1, score3)
    logscore = get_scorer('neg_log_loss')(clf, X_test, y_test)
    logloss = log_loss(y_test, clf.predict_proba(X_test))
    assert_almost_equal(-logscore, logloss)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    assert_almost_equal(score1, score2)
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    err_msg = 'DecisionTreeRegressor has none of the following attributes'
    with pytest.raises(AttributeError, match=err_msg):
        get_scorer('roc_auc')(reg, X_test, y_test)
    (X, y) = make_blobs(random_state=0, centers=3)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)
    with pytest.raises(ValueError, match="multi_class must be in \\('ovo', 'ovr'\\)"):
        get_scorer('roc_auc')(clf, X_test, y_test)
    (X, y) = make_blobs(random_state=0, centers=2)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, np.zeros_like(y_train))
    with pytest.raises(ValueError, match='need classifier with two classes'):
        get_scorer('roc_auc')(clf, X_test, y_test)
    with pytest.raises(ValueError, match='need classifier with two classes'):
        get_scorer('neg_log_loss')(clf, X_test, y_test)

def test_thresholded_scorers_multilabel_indicator_data():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_multilabel_classification(allow_unlabeled=False, random_state=0)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, np.vstack([p[:, -1] for p in y_proba]).T)
    assert_almost_equal(score1, score2)
    clf = OneVsRestClassifier(DecisionTreeClassifier())
    clf.fit(X_train, y_train)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.predict_proba(X_test))
    assert_almost_equal(score1, score2)
    clf = OneVsRestClassifier(LinearSVC(dual='auto', random_state=0))
    clf.fit(X_train, y_train)
    score1 = get_scorer('roc_auc')(clf, X_test, y_test)
    score2 = roc_auc_score(y_test, clf.decision_function(X_test))
    assert_almost_equal(score1, score2)

def test_supervised_cluster_scorers():
    if False:
        while True:
            i = 10
    (X, y) = make_blobs(random_state=0, centers=2)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    km = KMeans(n_clusters=3, n_init='auto')
    km.fit(X_train)
    for name in CLUSTER_SCORERS:
        score1 = get_scorer(name)(km, X_test, y_test)
        score2 = getattr(cluster_module, name)(y_test, km.predict(X_test))
        assert_almost_equal(score1, score2)

@ignore_warnings
def test_raises_on_score_list():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_blobs(random_state=0)
    f1_scorer_no_average = make_scorer(f1_score, average=None)
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        cross_val_score(clf, X, y, scoring=f1_scorer_no_average)
    grid_search = GridSearchCV(clf, scoring=f1_scorer_no_average, param_grid={'max_depth': [1, 2]})
    with pytest.raises(ValueError):
        grid_search.fit(X, y)

@ignore_warnings
def test_classification_scorer_sample_weight():
    if False:
        print('Hello World!')
    (X, y) = make_classification(random_state=0)
    (_, y_ml) = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    split = train_test_split(X, y, y_ml, random_state=0)
    (X_train, X_test, y_train, y_test, y_ml_train, y_ml_test) = split
    sample_weight = np.ones_like(y_test)
    sample_weight[:10] = 0
    estimator = _make_estimators(X_train, y_train, y_ml_train)
    for name in get_scorer_names():
        scorer = get_scorer(name)
        if name in REGRESSION_SCORERS:
            continue
        if name == 'top_k_accuracy':
            scorer._kwargs = {'k': 1}
        if name in MULTILABEL_ONLY_SCORERS:
            target = y_ml_test
        else:
            target = y_test
        try:
            weighted = scorer(estimator[name], X_test, target, sample_weight=sample_weight)
            ignored = scorer(estimator[name], X_test[10:], target[10:])
            unweighted = scorer(estimator[name], X_test, target)
            _ = scorer(estimator[name], X_test[:10], target[:10], sample_weight=None)
            assert weighted != unweighted, f'scorer {name} behaves identically when called with sample weights: {weighted} vs {unweighted}'
            assert_almost_equal(weighted, ignored, err_msg=f'scorer {name} behaves differently when ignoring samples and setting sample_weight to 0: {weighted} vs {ignored}')
        except TypeError as e:
            assert 'sample_weight' in str(e), f'scorer {name} raises unhelpful exception when called with sample weights: {str(e)}'

@ignore_warnings
def test_regression_scorer_sample_weight():
    if False:
        while True:
            i = 10
    (X, y) = make_regression(n_samples=101, n_features=20, random_state=0)
    y = _require_positive_y(y)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    sample_weight = np.ones_like(y_test)
    sample_weight[:11] = 0
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, y_train)
    for name in get_scorer_names():
        scorer = get_scorer(name)
        if name not in REGRESSION_SCORERS:
            continue
        try:
            weighted = scorer(reg, X_test, y_test, sample_weight=sample_weight)
            ignored = scorer(reg, X_test[11:], y_test[11:])
            unweighted = scorer(reg, X_test, y_test)
            assert weighted != unweighted, f'scorer {name} behaves identically when called with sample weights: {weighted} vs {unweighted}'
            assert_almost_equal(weighted, ignored, err_msg=f'scorer {name} behaves differently when ignoring samples and setting sample_weight to 0: {weighted} vs {ignored}')
        except TypeError as e:
            assert 'sample_weight' in str(e), f'scorer {name} raises unhelpful exception when called with sample weights: {str(e)}'

@pytest.mark.parametrize('name', get_scorer_names())
def test_scorer_memmap_input(name):
    if False:
        for i in range(10):
            print('nop')
    if name in REQUIRE_POSITIVE_Y_SCORERS:
        y_mm_1 = _require_positive_y(y_mm)
        y_ml_mm_1 = _require_positive_y(y_ml_mm)
    else:
        (y_mm_1, y_ml_mm_1) = (y_mm, y_ml_mm)
    with ignore_warnings():
        (scorer, estimator) = (get_scorer(name), ESTIMATORS[name])
        if name in MULTILABEL_ONLY_SCORERS:
            score = scorer(estimator, X_mm, y_ml_mm_1)
        else:
            score = scorer(estimator, X_mm, y_mm_1)
        assert isinstance(score, numbers.Number), name

def test_scoring_is_not_metric():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(LogisticRegression(), scoring=f1_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(LogisticRegression(), scoring=roc_auc_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(Ridge(), scoring=r2_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(KMeans(), scoring=cluster_module.adjusted_rand_score)
    with pytest.raises(ValueError, match='make_scorer'):
        check_scoring(KMeans(), scoring=cluster_module.rand_score)

@pytest.mark.parametrize('scorers,expected_predict_count,expected_predict_proba_count,expected_decision_func_count', [({'a1': 'accuracy', 'a2': 'accuracy', 'll1': 'neg_log_loss', 'll2': 'neg_log_loss', 'ra1': 'roc_auc', 'ra2': 'roc_auc'}, 1, 1, 1), (['roc_auc', 'accuracy'], 1, 0, 1), (['neg_log_loss', 'accuracy'], 1, 1, 0)])
def test_multimetric_scorer_calls_method_once(scorers, expected_predict_count, expected_predict_proba_count, expected_decision_func_count):
    if False:
        while True:
            i = 10
    (X, y) = (np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0]))
    mock_est = Mock()
    mock_est._estimator_type = 'classifier'
    fit_func = Mock(return_value=mock_est, name='fit')
    fit_func.__name__ = 'fit'
    predict_func = Mock(return_value=y, name='predict')
    predict_func.__name__ = 'predict'
    pos_proba = np.random.rand(X.shape[0])
    proba = np.c_[1 - pos_proba, pos_proba]
    predict_proba_func = Mock(return_value=proba, name='predict_proba')
    predict_proba_func.__name__ = 'predict_proba'
    decision_function_func = Mock(return_value=pos_proba, name='decision_function')
    decision_function_func.__name__ = 'decision_function'
    mock_est.fit = fit_func
    mock_est.predict = predict_func
    mock_est.predict_proba = predict_proba_func
    mock_est.decision_function = decision_function_func
    mock_est.classes_ = np.array([0, 1])
    scorer_dict = _check_multimetric_scoring(LogisticRegression(), scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    results = multi_scorer(mock_est, X, y)
    assert set(scorers) == set(results)
    assert predict_func.call_count == expected_predict_count
    assert predict_proba_func.call_count == expected_predict_proba_count
    assert decision_function_func.call_count == expected_decision_func_count

@pytest.mark.parametrize('scorers', [['roc_auc', 'neg_log_loss'], {'roc_auc': make_scorer(roc_auc_score, response_method=['predict_proba', 'decision_function']), 'neg_log_loss': make_scorer(log_loss, response_method='predict_proba')}])
def test_multimetric_scorer_calls_method_once_classifier_no_decision(scorers):
    if False:
        return 10
    predict_proba_call_cnt = 0

    class MockKNeighborsClassifier(KNeighborsClassifier):

        def predict_proba(self, X):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal predict_proba_call_cnt
            predict_proba_call_cnt += 1
            return super().predict_proba(X)
    (X, y) = (np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0]))
    clf = MockKNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    scorer = _MultimetricScorer(scorers=scorer_dict)
    scorer(clf, X, y)
    assert predict_proba_call_cnt == 1

def test_multimetric_scorer_calls_method_once_regressor_threshold():
    if False:
        return 10
    predict_called_cnt = 0

    class MockDecisionTreeRegressor(DecisionTreeRegressor):

        def predict(self, X):
            if False:
                i = 10
                return i + 15
            nonlocal predict_called_cnt
            predict_called_cnt += 1
            return super().predict(X)
    (X, y) = (np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0]))
    clf = MockDecisionTreeRegressor()
    clf.fit(X, y)
    scorers = {'neg_mse': 'neg_mean_squared_error', 'r2': 'r2'}
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    scorer = _MultimetricScorer(scorers=scorer_dict)
    scorer(clf, X, y)
    assert predict_called_cnt == 1

def test_multimetric_scorer_sanity_check():
    if False:
        i = 10
        return i + 15
    scorers = {'a1': 'accuracy', 'a2': 'accuracy', 'll1': 'neg_log_loss', 'll2': 'neg_log_loss', 'ra1': 'roc_auc', 'ra2': 'roc_auc'}
    (X, y) = make_classification(random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    result = multi_scorer(clf, X, y)
    separate_scores = {name: get_scorer(name)(clf, X, y) for name in ['accuracy', 'neg_log_loss', 'roc_auc']}
    for (key, value) in result.items():
        score_name = scorers[key]
        assert_allclose(value, separate_scores[score_name])

@pytest.mark.parametrize('raise_exc', [True, False])
def test_multimetric_scorer_exception_handling(raise_exc):
    if False:
        for i in range(10):
            print('nop')
    'Check that the calling of the `_MultimetricScorer` returns\n    exception messages in the result dict for the failing scorers\n    in case of `raise_exc` is `False` and if `raise_exc` is `True`,\n    then the proper exception is raised.\n    '
    scorers = {'failing_1': 'neg_mean_squared_log_error', 'non_failing': 'neg_median_absolute_error', 'failing_2': 'neg_mean_squared_log_error'}
    (X, y) = make_classification(n_samples=50, n_features=2, n_redundant=0, random_state=0)
    y *= -1
    clf = DecisionTreeClassifier().fit(X, y)
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict, raise_exc=raise_exc)
    error_msg = 'Mean Squared Logarithmic Error cannot be used when targets contain negative values.'
    if raise_exc:
        with pytest.raises(ValueError, match=error_msg):
            multi_scorer(clf, X, y)
    else:
        result = multi_scorer(clf, X, y)
        exception_message_1 = result['failing_1']
        score = result['non_failing']
        exception_message_2 = result['failing_2']
        assert isinstance(exception_message_1, str) and error_msg in exception_message_1
        assert isinstance(score, float)
        assert isinstance(exception_message_2, str) and error_msg in exception_message_2

@pytest.mark.parametrize('scorer_name, metric', [('roc_auc_ovr', partial(roc_auc_score, multi_class='ovr')), ('roc_auc_ovo', partial(roc_auc_score, multi_class='ovo')), ('roc_auc_ovr_weighted', partial(roc_auc_score, multi_class='ovr', average='weighted')), ('roc_auc_ovo_weighted', partial(roc_auc_score, multi_class='ovo', average='weighted'))])
def test_multiclass_roc_proba_scorer(scorer_name, metric):
    if False:
        i = 10
        return i + 15
    scorer = get_scorer(scorer_name)
    (X, y) = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression(multi_class='multinomial').fit(X, y)
    y_proba = lr.predict_proba(X)
    expected_score = metric(y, y_proba)
    assert scorer(lr, X, y) == pytest.approx(expected_score)

def test_multiclass_roc_proba_scorer_label():
    if False:
        while True:
            i = 10
    scorer = make_scorer(roc_auc_score, multi_class='ovo', labels=[0, 1, 2], response_method='predict_proba')
    (X, y) = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression(multi_class='multinomial').fit(X, y)
    y_proba = lr.predict_proba(X)
    y_binary = y == 0
    expected_score = roc_auc_score(y_binary, y_proba, multi_class='ovo', labels=[0, 1, 2])
    assert scorer(lr, X, y_binary) == pytest.approx(expected_score)

@pytest.mark.parametrize('scorer_name', ['roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'])
def test_multiclass_roc_no_proba_scorer_errors(scorer_name):
    if False:
        while True:
            i = 10
    scorer = get_scorer(scorer_name)
    (X, y) = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = Perceptron().fit(X, y)
    msg = 'Perceptron has none of the following attributes: predict_proba.'
    with pytest.raises(AttributeError, match=msg):
        scorer(lr, X, y)

@pytest.fixture
def string_labeled_classification_problem():
    if False:
        while True:
            i = 10
    'Train a classifier on binary problem with string target.\n\n    The classifier is trained on a binary classification problem where the\n    minority class of interest has a string label that is intentionally not the\n    greatest class label using the lexicographic order. In this case, "cancer"\n    is the positive label, and `classifier.classes_` is\n    `["cancer", "not cancer"]`.\n\n    In addition, the dataset is imbalanced to better identify problems when\n    using non-symmetric performance metrics such as f1-score, average precision\n    and so on.\n\n    Returns\n    -------\n    classifier : estimator object\n        Trained classifier on the binary problem.\n    X_test : ndarray of shape (n_samples, n_features)\n        Data to be used as testing set in tests.\n    y_test : ndarray of shape (n_samples,), dtype=object\n        Binary target where labels are strings.\n    y_pred : ndarray of shape (n_samples,), dtype=object\n        Prediction of `classifier` when predicting for `X_test`.\n    y_pred_proba : ndarray of shape (n_samples, 2), dtype=np.float64\n        Probabilities of `classifier` when predicting for `X_test`.\n    y_pred_decision : ndarray of shape (n_samples,), dtype=np.float64\n        Decision function values of `classifier` when predicting on `X_test`.\n    '
    from sklearn.datasets import load_breast_cancer
    from sklearn.utils import shuffle
    (X, y) = load_breast_cancer(return_X_y=True)
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    (X, y) = (X[idx_selected], y[idx_selected])
    (X, y) = shuffle(X, y, random_state=42)
    X = X[:, :2]
    y = np.array(['cancer' if c == 1 else 'not cancer' for c in y], dtype=object)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y, random_state=0)
    classifier = LogisticRegression().fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_decision = classifier.decision_function(X_test)
    return (classifier, X_test, y_test, y_pred, y_pred_proba, y_pred_decision)

def test_average_precision_pos_label(string_labeled_classification_problem):
    if False:
        while True:
            i = 10
    (clf, X_test, y_test, _, y_pred_proba, y_pred_decision) = string_labeled_classification_problem
    pos_label = 'cancer'
    y_pred_proba = y_pred_proba[:, 0]
    y_pred_decision = y_pred_decision * -1
    assert clf.classes_[0] == pos_label
    ap_proba = average_precision_score(y_test, y_pred_proba, pos_label=pos_label)
    ap_decision_function = average_precision_score(y_test, y_pred_decision, pos_label=pos_label)
    assert ap_proba == pytest.approx(ap_decision_function)
    average_precision_scorer = make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'))
    err_msg = 'pos_label=1 is not a valid label. It should be one of '
    with pytest.raises(ValueError, match=err_msg):
        average_precision_scorer(clf, X_test, y_test)
    average_precision_scorer = make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'), pos_label=pos_label)
    ap_scorer = average_precision_scorer(clf, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)
    clf_without_predict_proba = deepcopy(clf)

    def _predict_proba(self, X):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError
    clf_without_predict_proba.predict_proba = partial(_predict_proba, clf_without_predict_proba)
    with pytest.raises(NotImplementedError):
        clf_without_predict_proba.predict_proba(X_test)
    ap_scorer = average_precision_scorer(clf_without_predict_proba, X_test, y_test)
    assert ap_scorer == pytest.approx(ap_proba)

def test_brier_score_loss_pos_label(string_labeled_classification_problem):
    if False:
        return 10
    (clf, X_test, y_test, _, y_pred_proba, _) = string_labeled_classification_problem
    pos_label = 'cancer'
    assert clf.classes_[0] == pos_label
    brier_pos_cancer = brier_score_loss(y_test, y_pred_proba[:, 0], pos_label='cancer')
    brier_pos_not_cancer = brier_score_loss(y_test, y_pred_proba[:, 1], pos_label='not cancer')
    assert brier_pos_cancer == pytest.approx(brier_pos_not_cancer)
    brier_scorer = make_scorer(brier_score_loss, response_method='predict_proba', pos_label=pos_label)
    assert brier_scorer(clf, X_test, y_test) == pytest.approx(brier_pos_cancer)

@pytest.mark.parametrize('score_func', [f1_score, precision_score, recall_score, jaccard_score])
def test_non_symmetric_metric_pos_label(score_func, string_labeled_classification_problem):
    if False:
        for i in range(10):
            print('nop')
    (clf, X_test, y_test, y_pred, _, _) = string_labeled_classification_problem
    pos_label = 'cancer'
    assert clf.classes_[0] == pos_label
    score_pos_cancer = score_func(y_test, y_pred, pos_label='cancer')
    score_pos_not_cancer = score_func(y_test, y_pred, pos_label='not cancer')
    assert score_pos_cancer != pytest.approx(score_pos_not_cancer)
    scorer = make_scorer(score_func, pos_label=pos_label)
    assert scorer(clf, X_test, y_test) == pytest.approx(score_pos_cancer)

@pytest.mark.parametrize('scorer', [make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'), pos_label='xxx'), make_scorer(brier_score_loss, response_method='predict_proba', pos_label='xxx'), make_scorer(f1_score, pos_label='xxx')], ids=['non-thresholded scorer', 'probability scorer', 'thresholded scorer'])
def test_scorer_select_proba_error(scorer):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(n_classes=2, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression().fit(X, y)
    assert scorer._kwargs['pos_label'] not in np.unique(y).tolist()
    err_msg = 'is not a valid label'
    with pytest.raises(ValueError, match=err_msg):
        scorer(lr, X, y)

def test_get_scorer_return_copy():
    if False:
        while True:
            i = 10
    assert get_scorer('roc_auc') is not get_scorer('roc_auc')

def test_scorer_no_op_multiclass_select_proba():
    if False:
        print('Hello World!')
    (X, y) = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression().fit(X, y)
    mask_last_class = y == lr.classes_[-1]
    (X_test, y_test) = (X[~mask_last_class], y[~mask_last_class])
    assert_array_equal(np.unique(y_test), lr.classes_[:-1])
    scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovo', labels=lr.classes_)
    scorer(lr, X_test, y_test)

@pytest.mark.parametrize('name', get_scorer_names())
def test_scorer_set_score_request_raises(name):
    if False:
        i = 10
        return i + 15
    'Test that set_score_request is only available when feature flag is on.'
    scorer = get_scorer(name)
    with pytest.raises(RuntimeError, match='This method is only available'):
        scorer.set_score_request()

@pytest.mark.usefixtures('enable_slep006')
@pytest.mark.parametrize('name', get_scorer_names(), ids=get_scorer_names())
def test_scorer_metadata_request(name):
    if False:
        print('Hello World!')
    'Testing metadata requests for scorers.\n\n    This test checks many small things in a large test, to reduce the\n    boilerplate required for each section.\n    '
    scorer = get_scorer(name)
    assert hasattr(scorer, 'set_score_request')
    assert hasattr(scorer, 'get_metadata_routing')
    assert_request_is_empty(scorer.get_metadata_routing())
    weighted_scorer = scorer.set_score_request(sample_weight=True)
    assert weighted_scorer is scorer
    assert_request_is_empty(weighted_scorer.get_metadata_routing(), exclude='score')
    assert weighted_scorer.get_metadata_routing().score.requests['sample_weight'] is True
    router = MetadataRouter(owner='test').add(method_mapping='score', scorer=get_scorer(name))
    with pytest.raises(TypeError, match='got unexpected argument'):
        router.validate_metadata(params={'sample_weight': 1}, method='score')
    routed_params = router.route_params(params={'sample_weight': 1}, caller='score')
    assert not routed_params.scorer.score
    router = MetadataRouter(owner='test').add(scorer=weighted_scorer, method_mapping='score')
    router.validate_metadata(params={'sample_weight': 1}, method='score')
    routed_params = router.route_params(params={'sample_weight': 1}, caller='score')
    assert list(routed_params.scorer.score.keys()) == ['sample_weight']

@pytest.mark.usefixtures('enable_slep006')
def test_metadata_kwarg_conflict():
    if False:
        i = 10
        return i + 15
    'This test makes sure the right warning is raised if the user passes\n    some metadata both as a constructor to make_scorer, and during __call__.\n    '
    (X, y) = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression().fit(X, y)
    scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovo', labels=lr.classes_)
    with pytest.warns(UserWarning, match='already set as kwargs'):
        scorer.set_score_request(labels=True)
    with pytest.warns(UserWarning, match='There is an overlap'):
        scorer(lr, X, y, labels=lr.classes_)

@pytest.mark.usefixtures('enable_slep006')
def test_PassthroughScorer_metadata_request():
    if False:
        for i in range(10):
            print('nop')
    'Test that _PassthroughScorer properly routes metadata.\n\n    _PassthroughScorer should behave like a consumer, mirroring whatever is the\n    underlying score method.\n    '
    scorer = _PassthroughScorer(estimator=LinearSVC().set_score_request(sample_weight='alias').set_fit_request(sample_weight=True))
    assert_request_equal(scorer.get_metadata_routing(), {'fit': {'sample_weight': True}, 'score': {'sample_weight': 'alias'}})

@pytest.mark.usefixtures('enable_slep006')
def test_multimetric_scoring_metadata_routing():
    if False:
        return 10

    def score1(y_true, y_pred):
        if False:
            i = 10
            return i + 15
        return 1

    def score2(y_true, y_pred, sample_weight='test'):
        if False:
            print('Hello World!')
        assert sample_weight == 'test'
        return 1

    def score3(y_true, y_pred, sample_weight=None):
        if False:
            print('Hello World!')
        assert sample_weight is not None
        return 1
    scorers = {'score1': make_scorer(score1), 'score2': make_scorer(score2).set_score_request(sample_weight=False), 'score3': make_scorer(score3).set_score_request(sample_weight=True)}
    (X, y) = make_classification(n_samples=50, n_features=2, n_redundant=0, random_state=0)
    clf = DecisionTreeClassifier().fit(X, y)
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    with config_context(enable_metadata_routing=False):
        with pytest.raises(TypeError, match='got an unexpected keyword argument'):
            multi_scorer(clf, X, y, sample_weight=1)
    multi_scorer(clf, X, y, sample_weight=1)

def test_kwargs_without_metadata_routing_error():
    if False:
        while True:
            i = 10

    def score(y_true, y_pred, param=None):
        if False:
            for i in range(10):
                print('nop')
        return 1
    (X, y) = make_classification(n_samples=50, n_features=2, n_redundant=0, random_state=0)
    clf = DecisionTreeClassifier().fit(X, y)
    scorer = make_scorer(score)
    with config_context(enable_metadata_routing=False):
        with pytest.raises(ValueError, match='is only supported if enable_metadata_routing=True'):
            scorer(clf, X, y, param='blah')

def test_get_scorer_multilabel_indicator():
    if False:
        return 10
    'Check that our scorer deal with multi-label indicator matrices.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/26817\n    '
    (X, Y) = make_multilabel_classification(n_samples=72, n_classes=3, random_state=0)
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, random_state=0)
    estimator = KNeighborsClassifier().fit(X_train, Y_train)
    score = get_scorer('average_precision')(estimator, X_test, Y_test)
    assert score > 0.8

@pytest.mark.parametrize('scorer, expected_repr', [(get_scorer('accuracy'), "make_scorer(accuracy_score, response_method='predict')"), (get_scorer('neg_log_loss'), "make_scorer(log_loss, greater_is_better=False, response_method='predict_proba')"), (get_scorer('roc_auc'), "make_scorer(roc_auc_score, response_method=('decision_function', 'predict_proba'))"), (make_scorer(fbeta_score, beta=2), "make_scorer(fbeta_score, response_method='predict', beta=2)")])
def test_make_scorer_repr(scorer, expected_repr):
    if False:
        return 10
    'Check the representation of the scorer.'
    assert repr(scorer) == expected_repr

@pytest.mark.filterwarnings('ignore:.*needs_proba.*:FutureWarning')
@pytest.mark.parametrize('params, err_type, err_msg', [({'response_method': 'predict_proba', 'needs_proba': True}, ValueError, 'You cannot set both `response_method`'), ({'response_method': 'predict_proba', 'needs_threshold': True}, ValueError, 'You cannot set both `response_method`'), ({'needs_proba': True, 'needs_threshold': True}, ValueError, 'You cannot set both `needs_proba` and `needs_threshold`')])
def test_make_scorer_error(params, err_type, err_msg):
    if False:
        return 10
    'Check that `make_scorer` raises errors if the parameter used.'
    with pytest.raises(err_type, match=err_msg):
        make_scorer(lambda y_true, y_pred: 1, **params)

@pytest.mark.parametrize('deprecated_params, new_params, warn_msg', [({'needs_proba': True}, {'response_method': 'predict_proba'}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_proba': True, 'needs_threshold': False}, {'response_method': 'predict_proba'}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_threshold': True}, {'response_method': ('decision_function', 'predict_proba')}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_threshold': True, 'needs_proba': False}, {'response_method': ('decision_function', 'predict_proba')}, 'The `needs_threshold` and `needs_proba` parameter are deprecated'), ({'needs_threshold': False, 'needs_proba': False}, {'response_method': 'predict'}, 'The `needs_threshold` and `needs_proba` parameter are deprecated')])
def test_make_scorer_deprecation(deprecated_params, new_params, warn_msg):
    if False:
        i = 10
        return i + 15
    'Check that we raise a deprecation warning when using `needs_proba` or\n    `needs_threshold`.'
    (X, y) = make_classification(n_samples=150, n_features=10, random_state=0)
    classifier = LogisticRegression().fit(X, y)
    with pytest.warns(FutureWarning, match=warn_msg):
        deprecated_roc_auc_scorer = make_scorer(roc_auc_score, **deprecated_params)
    roc_auc_scorer = make_scorer(roc_auc_score, **new_params)
    assert deprecated_roc_auc_scorer(classifier, X, y) == pytest.approx(roc_auc_scorer(classifier, X, y))