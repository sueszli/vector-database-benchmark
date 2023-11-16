import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import LogisticRegression as LogisticRegressionDefault
from sklearn.linear_model._logistic import LogisticRegressionCV as LogisticRegressionCVDefault
from sklearn.linear_model._logistic import _log_reg_scoring_path, _logistic_regression_path
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
pytestmark = pytest.mark.filterwarnings('error::sklearn.exceptions.ConvergenceWarning:sklearn.*')
LogisticRegression = partial(LogisticRegressionDefault, random_state=0)
LogisticRegressionCV = partial(LogisticRegressionCVDefault, random_state=0)
SOLVERS = ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')
X = [[-1, 0], [0, 1], [1, 1]]
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]
iris = load_iris()

def check_predictions(clf, X, y):
    if False:
        i = 10
        return i + 15
    'Check that the model is able to fit the classification data'
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]
    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)
    assert predicted.shape == (n_samples,)
    assert_array_equal(predicted, y)
    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_samples, n_classes)
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_predict_2_classes(csr_container):
    if False:
        print('Hello World!')
    check_predictions(LogisticRegression(random_state=0), X, Y1)
    check_predictions(LogisticRegression(random_state=0), csr_container(X), Y1)
    check_predictions(LogisticRegression(C=100, random_state=0), X, Y1)
    check_predictions(LogisticRegression(C=100, random_state=0), csr_container(X), Y1)
    check_predictions(LogisticRegression(fit_intercept=False, random_state=0), X, Y1)
    check_predictions(LogisticRegression(fit_intercept=False, random_state=0), csr_container(X), Y1)

def test_logistic_cv_mock_scorer():
    if False:
        for i in range(10):
            print('nop')

    class MockScorer:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.calls = 0
            self.scores = [0.1, 0.4, 0.8, 0.5]

        def __call__(self, model, X, y, sample_weight=None):
            if False:
                i = 10
                return i + 15
            score = self.scores[self.calls % len(self.scores)]
            self.calls += 1
            return score
    mock_scorer = MockScorer()
    Cs = [1, 2, 3, 4]
    cv = 2
    lr = LogisticRegressionCV(Cs=Cs, scoring=mock_scorer, cv=cv)
    (X, y) = make_classification(random_state=0)
    lr.fit(X, y)
    assert lr.C_[0] == Cs[2]
    assert mock_scorer.calls == cv * len(Cs)
    mock_scorer.calls = 0
    custom_score = lr.score(X, lr.predict(X))
    assert custom_score == mock_scorer.scores[0]
    assert mock_scorer.calls == 1

@skip_if_no_parallel
def test_lr_liblinear_warning():
    if False:
        return 10
    (n_samples, n_features) = iris.data.shape
    target = iris.target_names[iris.target]
    lr = LogisticRegression(solver='liblinear', n_jobs=2)
    warning_message = "'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = 2."
    with pytest.warns(UserWarning, match=warning_message):
        lr.fit(iris.data, target)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_predict_3_classes(csr_container):
    if False:
        for i in range(10):
            print('nop')
    check_predictions(LogisticRegression(C=10), X, Y2)
    check_predictions(LogisticRegression(C=10), csr_container(X), Y2)

@pytest.mark.parametrize('clf', [LogisticRegression(C=len(iris.data), solver='liblinear', multi_class='ovr'), LogisticRegression(C=len(iris.data), solver='lbfgs', multi_class='multinomial'), LogisticRegression(C=len(iris.data), solver='newton-cg', multi_class='multinomial'), LogisticRegression(C=len(iris.data), solver='sag', tol=0.01, multi_class='ovr', random_state=42), LogisticRegression(C=len(iris.data), solver='saga', tol=0.01, multi_class='ovr', random_state=42), LogisticRegression(C=len(iris.data), solver='newton-cholesky', multi_class='ovr')])
def test_predict_iris(clf):
    if False:
        print('Hello World!')
    'Test logistic regression with the iris dataset.\n\n    Test that both multinomial and OvR solvers handle multiclass data correctly and\n    give good accuracy score (>0.95) for the training data.\n    '
    (n_samples, n_features) = iris.data.shape
    target = iris.target_names[iris.target]
    if clf.solver == 'lbfgs':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            clf.fit(iris.data, target)
    else:
        clf.fit(iris.data, target)
    assert_array_equal(np.unique(target), clf.classes_)
    pred = clf.predict(iris.data)
    assert np.mean(pred == target) > 0.95
    probabilities = clf.predict_proba(iris.data)
    assert_allclose(probabilities.sum(axis=1), np.ones(n_samples))
    pred = iris.target_names[probabilities.argmax(axis=1)]
    assert np.mean(pred == target) > 0.95

@pytest.mark.parametrize('LR', [LogisticRegression, LogisticRegressionCV])
def test_check_solver_option(LR):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = (iris.data, iris.target)
    for solver in ['liblinear', 'newton-cholesky']:
        msg = f'Solver {solver} does not support a multinomial backend.'
        lr = LR(solver=solver, multi_class='multinomial')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']:
        msg = "Solver %s supports only 'l2' or 'none' penalties," % solver
        lr = LR(solver=solver, penalty='l1', multi_class='ovr')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
        msg = 'Solver %s supports only dual=False, got dual=True' % solver
        lr = LR(solver=solver, dual=True, multi_class='ovr')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ['liblinear']:
        msg = "Only 'saga' solver supports elasticnet penalty, got solver={}.".format(solver)
        lr = LR(solver=solver, penalty='elasticnet')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    if LR is LogisticRegression:
        msg = "penalty='none' is not supported for the liblinear solver"
        lr = LR(penalty='none', solver='liblinear')
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

@pytest.mark.parametrize('LR', [LogisticRegression, LogisticRegressionCV])
def test_elasticnet_l1_ratio_err_helpful(LR):
    if False:
        i = 10
        return i + 15
    model = LR(penalty='elasticnet', solver='saga')
    with pytest.raises(ValueError, match='.*l1_ratio.*'):
        model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

@pytest.mark.parametrize('solver', ['lbfgs', 'newton-cg', 'sag', 'saga'])
def test_multinomial_binary(solver):
    if False:
        print('Hello World!')
    target = (iris.target > 0).astype(np.intp)
    target = np.array(['setosa', 'not-setosa'])[target]
    clf = LogisticRegression(solver=solver, multi_class='multinomial', random_state=42, max_iter=2000)
    clf.fit(iris.data, target)
    assert clf.coef_.shape == (1, iris.data.shape[1])
    assert clf.intercept_.shape == (1,)
    assert_array_equal(clf.predict(iris.data), target)
    mlr = LogisticRegression(solver=solver, multi_class='multinomial', random_state=42, fit_intercept=False)
    mlr.fit(iris.data, target)
    pred = clf.classes_[np.argmax(clf.predict_log_proba(iris.data), axis=1)]
    assert np.mean(pred == target) > 0.9

def test_multinomial_binary_probabilities(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(random_state=global_random_seed)
    clf = LogisticRegression(multi_class='multinomial', solver='saga', tol=0.001, random_state=global_random_seed)
    clf.fit(X, y)
    decision = clf.decision_function(X)
    proba = clf.predict_proba(X)
    expected_proba_class_1 = np.exp(decision) / (np.exp(decision) + np.exp(-decision))
    expected_proba = np.c_[1 - expected_proba_class_1, expected_proba_class_1]
    assert_almost_equal(proba, expected_proba)

@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_sparsify(coo_container):
    if False:
        for i in range(10):
            print('nop')
    (n_samples, n_features) = iris.data.shape
    target = iris.target_names[iris.target]
    X = scale(iris.data)
    clf = LogisticRegression(random_state=0).fit(X, target)
    pred_d_d = clf.decision_function(X)
    clf.sparsify()
    assert sparse.issparse(clf.coef_)
    pred_s_d = clf.decision_function(X)
    sp_data = coo_container(X)
    pred_s_s = clf.decision_function(sp_data)
    clf.densify()
    pred_d_s = clf.decision_function(sp_data)
    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)

def test_inconsistent_input():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X_ = rng.random_sample((5, 10))
    y_ = np.ones(X_.shape[0])
    y_[0] = 0
    clf = LogisticRegression(random_state=0)
    y_wrong = y_[:-1]
    with pytest.raises(ValueError):
        clf.fit(X, y_wrong)
    with pytest.raises(ValueError):
        clf.fit(X_, y_).predict(rng.random_sample((3, 12)))

def test_write_parameters():
    if False:
        return 10
    clf = LogisticRegression(random_state=0)
    clf.fit(X, Y1)
    clf.coef_[:] = 0
    clf.intercept_[:] = 0
    assert_array_almost_equal(clf.decision_function(X), 0)

def test_nan():
    if False:
        while True:
            i = 10
    Xnan = np.array(X, dtype=np.float64)
    Xnan[0, 1] = np.nan
    logistic = LogisticRegression(random_state=0)
    with pytest.raises(ValueError):
        logistic.fit(Xnan, Y1)

def test_consistency_path():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = np.logspace(0, 4, 10)
    f = ignore_warnings
    for solver in ['sag', 'saga']:
        (coefs, Cs, _) = f(_logistic_regression_path)(X, y, Cs=Cs, fit_intercept=False, tol=1e-05, solver=solver, max_iter=1000, multi_class='ovr', random_state=0)
        for (i, C) in enumerate(Cs):
            lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-05, solver=solver, multi_class='ovr', random_state=0, max_iter=1000)
            lr.fit(X, y)
            lr_coef = lr.coef_.ravel()
            assert_array_almost_equal(lr_coef, coefs[i], decimal=4, err_msg='with solver = %s' % solver)
    for solver in ('lbfgs', 'newton-cg', 'newton-cholesky', 'liblinear', 'sag', 'saga'):
        Cs = [1000.0]
        (coefs, Cs, _) = f(_logistic_regression_path)(X, y, Cs=Cs, tol=1e-06, solver=solver, intercept_scaling=10000.0, random_state=0, multi_class='ovr')
        lr = LogisticRegression(C=Cs[0], tol=1e-06, intercept_scaling=10000.0, random_state=0, multi_class='ovr', solver=solver)
        lr.fit(X, y)
        lr_coef = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        assert_array_almost_equal(lr_coef, coefs[0], decimal=4, err_msg='with solver = %s' % solver)

def test_logistic_regression_path_convergence_fail():
    if False:
        return 10
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = [1000.0]
    with pytest.warns(ConvergenceWarning) as record:
        _logistic_regression_path(X, y, Cs=Cs, tol=0.0, max_iter=1, random_state=0, verbose=0)
    assert len(record) == 1
    warn_msg = record[0].message.args[0]
    assert 'lbfgs failed to converge' in warn_msg
    assert 'Increase the number of iterations' in warn_msg
    assert 'scale the data' in warn_msg
    assert 'linear_model.html#logistic-regression' in warn_msg

def test_liblinear_dual_random_state():
    if False:
        i = 10
        return i + 15
    (X, y) = make_classification(n_samples=20, random_state=0)
    lr1 = LogisticRegression(random_state=0, dual=True, tol=0.001, solver='liblinear', multi_class='ovr')
    lr1.fit(X, y)
    lr2 = LogisticRegression(random_state=0, dual=True, tol=0.001, solver='liblinear', multi_class='ovr')
    lr2.fit(X, y)
    lr3 = LogisticRegression(random_state=8, dual=True, tol=0.001, solver='liblinear', multi_class='ovr')
    lr3.fit(X, y)
    assert_array_almost_equal(lr1.coef_, lr2.coef_)
    msg = 'Arrays are not almost equal to 6 decimals'
    with pytest.raises(AssertionError, match=msg):
        assert_array_almost_equal(lr1.coef_, lr3.coef_)

def test_logistic_cv():
    if False:
        i = 10
        return i + 15
    (n_samples, n_features) = (50, 5)
    rng = np.random.RandomState(0)
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    lr_cv = LogisticRegressionCV(Cs=[1.0], fit_intercept=False, solver='liblinear', multi_class='ovr', cv=3)
    lr_cv.fit(X_ref, y)
    lr = LogisticRegression(C=1.0, fit_intercept=False, solver='liblinear', multi_class='ovr')
    lr.fit(X_ref, y)
    assert_array_almost_equal(lr.coef_, lr_cv.coef_)
    assert_array_equal(lr_cv.coef_.shape, (1, n_features))
    assert_array_equal(lr_cv.classes_, [-1, 1])
    assert len(lr_cv.classes_) == 2
    coefs_paths = np.asarray(list(lr_cv.coefs_paths_.values()))
    assert_array_equal(coefs_paths.shape, (1, 3, 1, n_features))
    assert_array_equal(lr_cv.Cs_.shape, (1,))
    scores = np.asarray(list(lr_cv.scores_.values()))
    assert_array_equal(scores.shape, (1, 3, 1))

@pytest.mark.parametrize('scoring, multiclass_agg_list', [('accuracy', ['']), ('precision', ['_macro', '_weighted']), ('f1', ['_macro', '_weighted']), ('neg_log_loss', ['']), ('recall', ['_macro', '_weighted'])])
def test_logistic_cv_multinomial_score(scoring, multiclass_agg_list):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(n_samples=100, random_state=0, n_classes=3, n_informative=6)
    (train, test) = (np.arange(80), np.arange(80, 100))
    lr = LogisticRegression(C=1.0, multi_class='multinomial')
    params = lr.get_params()
    for key in ['C', 'n_jobs', 'warm_start']:
        del params[key]
    lr.fit(X[train], y[train])
    for averaging in multiclass_agg_list:
        scorer = get_scorer(scoring + averaging)
        assert_array_almost_equal(_log_reg_scoring_path(X, y, train, test, Cs=[1.0], scoring=scorer, pos_class=None, max_squared_sum=None, sample_weight=None, score_params=None, **params)[2][0], scorer(lr, X[test], y[test]))

def test_multinomial_logistic_regression_string_inputs():
    if False:
        for i in range(10):
            print('nop')
    (n_samples, n_features, n_classes) = (50, 5, 3)
    (X_ref, y) = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=3, random_state=0)
    y_str = LabelEncoder().fit(['bar', 'baz', 'foo']).inverse_transform(y)
    y = np.array(y) - 1
    lr = LogisticRegression(multi_class='multinomial')
    lr_cv = LogisticRegressionCV(multi_class='multinomial', Cs=3)
    lr_str = LogisticRegression(multi_class='multinomial')
    lr_cv_str = LogisticRegressionCV(multi_class='multinomial', Cs=3)
    lr.fit(X_ref, y)
    lr_cv.fit(X_ref, y)
    lr_str.fit(X_ref, y_str)
    lr_cv_str.fit(X_ref, y_str)
    assert_array_almost_equal(lr.coef_, lr_str.coef_)
    assert sorted(lr_str.classes_) == ['bar', 'baz', 'foo']
    assert_array_almost_equal(lr_cv.coef_, lr_cv_str.coef_)
    assert sorted(lr_str.classes_) == ['bar', 'baz', 'foo']
    assert sorted(lr_cv_str.classes_) == ['bar', 'baz', 'foo']
    assert sorted(np.unique(lr_str.predict(X_ref))) == ['bar', 'baz', 'foo']
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ['bar', 'baz', 'foo']
    lr_cv_str = LogisticRegression(class_weight={'bar': 1, 'baz': 2, 'foo': 0}, multi_class='multinomial').fit(X_ref, y_str)
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ['bar', 'baz']

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_logistic_cv_sparse(csr_container):
    if False:
        return 10
    (X, y) = make_classification(n_samples=50, n_features=5, random_state=0)
    X[X < 1.0] = 0.0
    csr = csr_container(X)
    clf = LogisticRegressionCV()
    clf.fit(X, y)
    clfs = LogisticRegressionCV()
    clfs.fit(csr, y)
    assert_array_almost_equal(clfs.coef_, clf.coef_)
    assert_array_almost_equal(clfs.intercept_, clf.intercept_)
    assert clfs.C_ == clf.C_

def test_ovr_multinomial_iris():
    if False:
        print('Hello World!')
    (train, target) = (iris.data, iris.target)
    (n_samples, n_features) = train.shape
    n_cv = 2
    cv = StratifiedKFold(n_cv)
    precomputed_folds = list(cv.split(train, target))
    clf = LogisticRegressionCV(cv=precomputed_folds, multi_class='ovr')
    clf.fit(train, target)
    clf1 = LogisticRegressionCV(cv=precomputed_folds, multi_class='ovr')
    target_copy = target.copy()
    target_copy[target_copy == 0] = 1
    clf1.fit(train, target_copy)
    assert_allclose(clf.scores_[2], clf1.scores_[2])
    assert_allclose(clf.intercept_[2:], clf1.intercept_)
    assert_allclose(clf.coef_[2][np.newaxis, :], clf1.coef_)
    assert clf.coef_.shape == (3, n_features)
    assert_array_equal(clf.classes_, [0, 1, 2])
    coefs_paths = np.asarray(list(clf.coefs_paths_.values()))
    assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
    assert clf.Cs_.shape == (10,)
    scores = np.asarray(list(clf.scores_.values()))
    assert scores.shape == (3, n_cv, 10)
    for solver in ['lbfgs', 'newton-cg', 'sag', 'saga']:
        max_iter = 500 if solver in ['sag', 'saga'] else 30
        clf_multi = LogisticRegressionCV(solver=solver, multi_class='multinomial', max_iter=max_iter, random_state=42, tol=0.001 if solver in ['sag', 'saga'] else 0.01, cv=2)
        if solver == 'lbfgs':
            train = scale(train)
        clf_multi.fit(train, target)
        multi_score = clf_multi.score(train, target)
        ovr_score = clf.score(train, target)
        assert multi_score > ovr_score
        assert clf.coef_.shape == clf_multi.coef_.shape
        assert_array_equal(clf_multi.classes_, [0, 1, 2])
        coefs_paths = np.asarray(list(clf_multi.coefs_paths_.values()))
        assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
        assert clf_multi.Cs_.shape == (10,)
        scores = np.asarray(list(clf_multi.scores_.values()))
        assert scores.shape == (3, n_cv, 10)

def test_logistic_regression_solvers():
    if False:
        i = 10
        return i + 15
    'Test solvers converge to the same result.'
    (X, y) = make_classification(n_features=10, n_informative=5, random_state=0)
    params = dict(fit_intercept=False, random_state=42, multi_class='ovr')
    regressors = {solver: LogisticRegression(solver=solver, **params).fit(X, y) for solver in SOLVERS}
    for (solver_1, solver_2) in itertools.combinations(regressors, r=2):
        assert_array_almost_equal(regressors[solver_1].coef_, regressors[solver_2].coef_, decimal=3)

def test_logistic_regression_solvers_multiclass():
    if False:
        print('Hello World!')
    'Test solvers converge to the same result for multiclass problems.'
    (X, y) = make_classification(n_samples=20, n_features=20, n_informative=10, n_classes=3, random_state=0)
    tol = 1e-07
    params = dict(fit_intercept=False, tol=tol, random_state=42, multi_class='ovr')
    solver_max_iter = {'sag': 1000, 'saga': 10000}
    regressors = {solver: LogisticRegression(solver=solver, max_iter=solver_max_iter.get(solver, 100), **params).fit(X, y) for solver in SOLVERS}
    for (solver_1, solver_2) in itertools.combinations(regressors, r=2):
        assert_allclose(regressors[solver_1].coef_, regressors[solver_2].coef_, rtol=0.005 if solver_2 == 'saga' else 0.001, err_msg=f'{solver_1} vs {solver_2}')

@pytest.mark.parametrize('weight', [{0: 0.1, 1: 0.2}, {0: 0.1, 1: 0.2, 2: 0.5}])
@pytest.mark.parametrize('class_weight', ['weight', 'balanced'])
def test_logistic_regressioncv_class_weights(weight, class_weight, global_random_seed):
    if False:
        return 10
    'Test class_weight for LogisticRegressionCV.'
    n_classes = len(weight)
    if class_weight == 'weight':
        class_weight = weight
    (X, y) = make_classification(n_samples=30, n_features=3, n_repeated=0, n_informative=3, n_redundant=0, n_classes=n_classes, random_state=global_random_seed)
    params = dict(Cs=1, fit_intercept=False, multi_class='ovr', class_weight=class_weight, tol=1e-08)
    clf_lbfgs = LogisticRegressionCV(solver='lbfgs', **params)
    clf_lbfgs.fit(X, y)
    for solver in set(SOLVERS) - set(['lbfgs']):
        clf = LogisticRegressionCV(solver=solver, **params)
        if solver in ('sag', 'saga'):
            clf.set_params(tol=1e-18, max_iter=10000, random_state=global_random_seed + 1)
        clf.fit(X, y)
        assert_allclose(clf.coef_, clf_lbfgs.coef_, rtol=0.001, err_msg=f'{solver} vs lbfgs')

def test_logistic_regression_sample_weights():
    if False:
        return 10
    (X, y) = make_classification(n_samples=20, n_features=5, n_informative=3, n_classes=2, random_state=0)
    sample_weight = y + 1
    for LR in [LogisticRegression, LogisticRegressionCV]:
        kw = {'random_state': 42, 'fit_intercept': False, 'multi_class': 'ovr'}
        if LR is LogisticRegressionCV:
            kw.update({'Cs': 3, 'cv': 3})
        for solver in ['lbfgs', 'liblinear']:
            clf_sw_none = LR(solver=solver, **kw)
            clf_sw_ones = LR(solver=solver, **kw)
            clf_sw_none.fit(X, y)
            clf_sw_ones.fit(X, y, sample_weight=np.ones(y.shape[0]))
            assert_allclose(clf_sw_none.coef_, clf_sw_ones.coef_, rtol=0.0001)
        clf_sw_lbfgs = LR(**kw, tol=1e-05)
        clf_sw_lbfgs.fit(X, y, sample_weight=sample_weight)
        for solver in set(SOLVERS) - set(('lbfgs', 'saga')):
            clf_sw = LR(solver=solver, tol=1e-10 if solver == 'sag' else 1e-05, **kw)
            with ignore_warnings():
                clf_sw.fit(X, y, sample_weight=sample_weight)
            assert_allclose(clf_sw_lbfgs.coef_, clf_sw.coef_, rtol=0.0001)
        for solver in ['lbfgs', 'liblinear']:
            clf_cw_12 = LR(solver=solver, class_weight={0: 1, 1: 2}, **kw)
            clf_cw_12.fit(X, y)
            clf_sw_12 = LR(solver=solver, **kw)
            clf_sw_12.fit(X, y, sample_weight=sample_weight)
            assert_allclose(clf_cw_12.coef_, clf_sw_12.coef_, rtol=0.0001)
    clf_cw = LogisticRegression(solver='liblinear', fit_intercept=False, class_weight={0: 1, 1: 2}, penalty='l1', tol=1e-05, random_state=42, multi_class='ovr')
    clf_cw.fit(X, y)
    clf_sw = LogisticRegression(solver='liblinear', fit_intercept=False, penalty='l1', tol=1e-05, random_state=42, multi_class='ovr')
    clf_sw.fit(X, y, sample_weight)
    assert_array_almost_equal(clf_cw.coef_, clf_sw.coef_, decimal=4)
    clf_cw = LogisticRegression(solver='liblinear', fit_intercept=False, class_weight={0: 1, 1: 2}, penalty='l2', dual=True, random_state=42, multi_class='ovr')
    clf_cw.fit(X, y)
    clf_sw = LogisticRegression(solver='liblinear', fit_intercept=False, penalty='l2', dual=True, random_state=42, multi_class='ovr')
    clf_sw.fit(X, y, sample_weight)
    assert_array_almost_equal(clf_cw.coef_, clf_sw.coef_, decimal=4)

def _compute_class_weight_dictionary(y):
    if False:
        return 10
    classes = np.unique(y)
    class_weight = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weight))
    return class_weight_dict

def test_logistic_regression_class_weights():
    if False:
        while True:
            i = 10
    X_iris = scale(iris.data)
    X = X_iris[45:, :]
    y = iris.target[45:]
    solvers = ('lbfgs', 'newton-cg')
    class_weight_dict = _compute_class_weight_dictionary(y)
    for solver in solvers:
        clf1 = LogisticRegression(solver=solver, multi_class='multinomial', class_weight='balanced')
        clf2 = LogisticRegression(solver=solver, multi_class='multinomial', class_weight=class_weight_dict)
        clf1.fit(X, y)
        clf2.fit(X, y)
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=4)
    X = X_iris[45:100, :]
    y = iris.target[45:100]
    class_weight_dict = _compute_class_weight_dictionary(y)
    for solver in set(SOLVERS) - set(('sag', 'saga')):
        clf1 = LogisticRegression(solver=solver, multi_class='ovr', class_weight='balanced')
        clf2 = LogisticRegression(solver=solver, multi_class='ovr', class_weight=class_weight_dict)
        clf1.fit(X, y)
        clf2.fit(X, y)
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=6)

def test_logistic_regression_multinomial():
    if False:
        print('Hello World!')
    (n_samples, n_features, n_classes) = (50, 20, 3)
    (X, y) = make_classification(n_samples=n_samples, n_features=n_features, n_informative=10, n_classes=n_classes, random_state=0)
    X = StandardScaler(with_mean=False).fit_transform(X)
    solver = 'lbfgs'
    ref_i = LogisticRegression(solver=solver, multi_class='multinomial', tol=1e-06)
    ref_w = LogisticRegression(solver=solver, multi_class='multinomial', fit_intercept=False, tol=1e-06)
    ref_i.fit(X, y)
    ref_w.fit(X, y)
    assert ref_i.coef_.shape == (n_classes, n_features)
    assert ref_w.coef_.shape == (n_classes, n_features)
    for solver in ['sag', 'saga', 'newton-cg']:
        clf_i = LogisticRegression(solver=solver, multi_class='multinomial', random_state=42, max_iter=2000, tol=1e-07)
        clf_w = LogisticRegression(solver=solver, multi_class='multinomial', random_state=42, max_iter=2000, tol=1e-07, fit_intercept=False)
        clf_i.fit(X, y)
        clf_w.fit(X, y)
        assert clf_i.coef_.shape == (n_classes, n_features)
        assert clf_w.coef_.shape == (n_classes, n_features)
        assert_allclose(ref_i.coef_, clf_i.coef_, rtol=0.001)
        assert_allclose(ref_w.coef_, clf_w.coef_, rtol=0.01)
        assert_allclose(ref_i.intercept_, clf_i.intercept_, rtol=0.001)
    for solver in ['lbfgs', 'newton-cg', 'sag', 'saga']:
        clf_path = LogisticRegressionCV(solver=solver, max_iter=2000, tol=1e-06, multi_class='multinomial', Cs=[1.0])
        clf_path.fit(X, y)
        assert_allclose(clf_path.coef_, ref_i.coef_, rtol=0.01)
        assert_allclose(clf_path.intercept_, ref_i.intercept_, rtol=0.01)

def test_liblinear_decision_function_zero():
    if False:
        print('Hello World!')
    (X, y) = make_classification(n_samples=5, n_features=5, random_state=0)
    clf = LogisticRegression(fit_intercept=False, solver='liblinear', multi_class='ovr')
    clf.fit(X, y)
    X = np.zeros((5, 5))
    assert_array_equal(clf.predict(X), np.zeros(5))

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_liblinear_logregcv_sparse(csr_container):
    if False:
        print('Hello World!')
    (X, y) = make_classification(n_samples=10, n_features=5, random_state=0)
    clf = LogisticRegressionCV(solver='liblinear', multi_class='ovr')
    clf.fit(csr_container(X), y)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_saga_sparse(csr_container):
    if False:
        return 10
    (X, y) = make_classification(n_samples=10, n_features=5, random_state=0)
    clf = LogisticRegressionCV(solver='saga', tol=0.01)
    clf.fit(csr_container(X), y)

def test_logreg_intercept_scaling_zero():
    if False:
        for i in range(10):
            print('nop')
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X, Y1)
    assert clf.intercept_ == 0.0

def test_logreg_l1():
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(42)
    n_samples = 50
    (X, y) = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(size=(n_samples, 3))
    X_constant = np.ones(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    lr_liblinear = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', fit_intercept=False, multi_class='ovr', tol=1e-10)
    lr_liblinear.fit(X, y)
    lr_saga = LogisticRegression(penalty='l1', C=1.0, solver='saga', fit_intercept=False, multi_class='ovr', max_iter=1000, tol=1e-10)
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_logreg_l1_sparse_data(csr_container):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(42)
    n_samples = 50
    (X, y) = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(scale=0.1, size=(n_samples, 3))
    X_constant = np.zeros(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    X[X < 1] = 0
    X = csr_container(X)
    lr_liblinear = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', fit_intercept=False, multi_class='ovr', tol=1e-10)
    lr_liblinear.fit(X, y)
    lr_saga = LogisticRegression(penalty='l1', C=1.0, solver='saga', fit_intercept=False, multi_class='ovr', max_iter=1000, tol=1e-10)
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))
    lr_saga_dense = LogisticRegression(penalty='l1', C=1.0, solver='saga', fit_intercept=False, multi_class='ovr', max_iter=1000, tol=1e-10)
    lr_saga_dense.fit(X.toarray(), y)
    assert_array_almost_equal(lr_saga.coef_, lr_saga_dense.coef_)

@pytest.mark.parametrize('random_seed', [42])
@pytest.mark.parametrize('penalty', ['l1', 'l2'])
def test_logistic_regression_cv_refit(random_seed, penalty):
    if False:
        i = 10
        return i + 15
    (X, y) = make_classification(n_samples=100, n_features=20, random_state=random_seed)
    common_params = dict(solver='saga', penalty=penalty, random_state=random_seed, max_iter=1000, tol=1e-12)
    lr_cv = LogisticRegressionCV(Cs=[1.0], refit=True, **common_params)
    lr_cv.fit(X, y)
    lr = LogisticRegression(C=1.0, **common_params)
    lr.fit(X, y)
    assert_array_almost_equal(lr_cv.coef_, lr.coef_)

def test_logreg_predict_proba_multinomial():
    if False:
        return 10
    (X, y) = make_classification(n_samples=10, n_features=20, random_state=0, n_classes=3, n_informative=10)
    clf_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf_multi.fit(X, y)
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs')
    clf_ovr.fit(X, y)
    clf_ovr_loss = log_loss(y, clf_ovr.predict_proba(X))
    assert clf_ovr_loss > clf_multi_loss
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_wrong_loss = log_loss(y, clf_multi._predict_proba_lr(X))
    assert clf_wrong_loss > clf_multi_loss

@pytest.mark.parametrize('max_iter', np.arange(1, 5))
@pytest.mark.parametrize('multi_class', ['ovr', 'multinomial'])
@pytest.mark.parametrize('solver, message', [('newton-cg', 'newton-cg failed to converge. Increase the number of iterations.'), ('liblinear', 'Liblinear failed to converge, increase the number of iterations.'), ('sag', 'The max_iter was reached which means the coef_ did not converge'), ('saga', 'The max_iter was reached which means the coef_ did not converge'), ('lbfgs', 'lbfgs failed to converge'), ('newton-cholesky', 'Newton solver did not converge after [0-9]* iterations')])
def test_max_iter(max_iter, multi_class, solver, message):
    if False:
        i = 10
        return i + 15
    (X, y_bin) = (iris.data, iris.target.copy())
    y_bin[y_bin == 2] = 0
    if solver in ('liblinear', 'newton-cholesky') and multi_class == 'multinomial':
        pytest.skip("'multinomial' is not supported by liblinear and newton-cholesky")
    if solver == 'newton-cholesky' and max_iter > 1:
        pytest.skip('solver newton-cholesky might converge very fast')
    lr = LogisticRegression(max_iter=max_iter, tol=1e-15, multi_class=multi_class, random_state=0, solver=solver)
    with pytest.warns(ConvergenceWarning, match=message):
        lr.fit(X, y_bin)
    assert lr.n_iter_[0] == max_iter

@pytest.mark.parametrize('solver', SOLVERS)
def test_n_iter(solver):
    if False:
        return 10
    (X, y) = (iris.data, iris.target)
    if solver == 'lbfgs':
        X = scale(X)
    n_classes = np.unique(y).shape[0]
    assert n_classes == 3
    y_bin = y.copy()
    y_bin[y_bin == 2] = 0
    n_Cs = 4
    n_cv_fold = 2
    clf = LogisticRegression(tol=0.01, C=1.0, solver=solver, random_state=42)
    clf.fit(X, y_bin)
    assert clf.n_iter_.shape == (1,)
    clf_cv = LogisticRegressionCV(tol=0.01, solver=solver, Cs=n_Cs, cv=n_cv_fold, random_state=42)
    clf_cv.fit(X, y_bin)
    assert clf_cv.n_iter_.shape == (1, n_cv_fold, n_Cs)
    clf.set_params(multi_class='ovr').fit(X, y)
    assert clf.n_iter_.shape == (n_classes,)
    clf_cv.set_params(multi_class='ovr').fit(X, y)
    assert clf_cv.n_iter_.shape == (n_classes, n_cv_fold, n_Cs)
    if solver in ('liblinear', 'newton-cholesky'):
        return
    clf.set_params(multi_class='multinomial').fit(X, y)
    assert clf.n_iter_.shape == (1,)
    clf_cv.set_params(multi_class='multinomial').fit(X, y)
    assert clf_cv.n_iter_.shape == (1, n_cv_fold, n_Cs)

@pytest.mark.parametrize('solver', sorted(set(SOLVERS) - set(['liblinear'])))
@pytest.mark.parametrize('warm_start', (True, False))
@pytest.mark.parametrize('fit_intercept', (True, False))
@pytest.mark.parametrize('multi_class', ['ovr', 'multinomial'])
def test_warm_start(solver, warm_start, fit_intercept, multi_class):
    if False:
        i = 10
        return i + 15
    (X, y) = (iris.data, iris.target)
    if solver == 'newton-cholesky' and multi_class == 'multinomial':
        return
    clf = LogisticRegression(tol=0.0001, multi_class=multi_class, warm_start=warm_start, solver=solver, random_state=42, fit_intercept=fit_intercept)
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
        coef_1 = clf.coef_
        clf.max_iter = 1
        clf.fit(X, y)
    cum_diff = np.sum(np.abs(coef_1 - clf.coef_))
    msg = 'Warm starting issue with %s solver in %s mode with fit_intercept=%s and warm_start=%s' % (solver, multi_class, str(fit_intercept), str(warm_start))
    if warm_start:
        assert 2.0 > cum_diff, msg
    else:
        assert cum_diff > 2.0, msg

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_saga_vs_liblinear(csr_container):
    if False:
        for i in range(10):
            print('nop')
    iris = load_iris()
    (X, y) = (iris.data, iris.target)
    X = np.concatenate([X] * 3)
    y = np.concatenate([y] * 3)
    X_bin = X[y <= 1]
    y_bin = y[y <= 1] * 2 - 1
    (X_sparse, y_sparse) = make_classification(n_samples=50, n_features=20, random_state=0)
    X_sparse = csr_container(X_sparse)
    for (X, y) in ((X_bin, y_bin), (X_sparse, y_sparse)):
        for penalty in ['l1', 'l2']:
            n_samples = X.shape[0]
            for alpha in np.logspace(-1, 1, 3):
                saga = LogisticRegression(C=1.0 / (n_samples * alpha), solver='saga', multi_class='ovr', max_iter=200, fit_intercept=False, penalty=penalty, random_state=0, tol=1e-06)
                liblinear = LogisticRegression(C=1.0 / (n_samples * alpha), solver='liblinear', multi_class='ovr', max_iter=200, fit_intercept=False, penalty=penalty, random_state=0, tol=1e-06)
                saga.fit(X, y)
                liblinear.fit(X, y)
                assert_array_almost_equal(saga.coef_, liblinear.coef_, 3)

@pytest.mark.parametrize('multi_class', ['ovr', 'multinomial'])
@pytest.mark.parametrize('solver', ['liblinear', 'newton-cg', 'newton-cholesky', 'saga'])
@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dtype_match(solver, multi_class, fit_intercept, csr_container):
    if False:
        for i in range(10):
            print('nop')
    if solver in ('liblinear', 'newton-cholesky') and multi_class == 'multinomial':
        pytest.skip(f'Solver={solver} does not support multinomial logistic.')
    out32_type = np.float64 if solver == 'liblinear' else np.float32
    X_32 = np.array(X).astype(np.float32)
    y_32 = np.array(Y1).astype(np.float32)
    X_64 = np.array(X).astype(np.float64)
    y_64 = np.array(Y1).astype(np.float64)
    X_sparse_32 = csr_container(X, dtype=np.float32)
    X_sparse_64 = csr_container(X, dtype=np.float64)
    solver_tol = 0.0005
    lr_templ = LogisticRegression(solver=solver, multi_class=multi_class, random_state=42, tol=solver_tol, fit_intercept=fit_intercept)
    lr_32 = clone(lr_templ)
    lr_32.fit(X_32, y_32)
    assert lr_32.coef_.dtype == out32_type
    lr_32_sparse = clone(lr_templ)
    lr_32_sparse.fit(X_sparse_32, y_32)
    assert lr_32_sparse.coef_.dtype == out32_type
    lr_64 = clone(lr_templ)
    lr_64.fit(X_64, y_64)
    assert lr_64.coef_.dtype == np.float64
    lr_64_sparse = clone(lr_templ)
    lr_64_sparse.fit(X_sparse_64, y_64)
    assert lr_64_sparse.coef_.dtype == np.float64
    atol = 2 * 1.72 * solver_tol
    if os.name == 'nt' and _IS_32BIT:
        atol = 0.01
    assert_allclose(lr_32.coef_, lr_64.coef_.astype(np.float32), atol=atol)
    if solver == 'saga' and fit_intercept:
        atol = 0.1
    assert_allclose(lr_32.coef_, lr_32_sparse.coef_, atol=atol)
    assert_allclose(lr_64.coef_, lr_64_sparse.coef_, atol=atol)

def test_warm_start_converge_LR():
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = np.array([1] * 100 + [-1] * 100)
    lr_no_ws = LogisticRegression(multi_class='multinomial', solver='sag', warm_start=False, random_state=0)
    lr_ws = LogisticRegression(multi_class='multinomial', solver='sag', warm_start=True, random_state=0)
    lr_no_ws_loss = log_loss(y, lr_no_ws.fit(X, y).predict_proba(X))
    for i in range(5):
        lr_ws.fit(X, y)
    lr_ws_loss = log_loss(y, lr_ws.predict_proba(X))
    assert_allclose(lr_no_ws_loss, lr_ws_loss, rtol=1e-05)

def test_elastic_net_coeffs():
    if False:
        return 10
    (X, y) = make_classification(random_state=0)
    C = 2.0
    l1_ratio = 0.5
    coeffs = list()
    for (penalty, ratio) in (('elasticnet', l1_ratio), ('l1', None), ('l2', None)):
        lr = LogisticRegression(penalty=penalty, C=C, solver='saga', random_state=0, l1_ratio=ratio, tol=0.001, max_iter=200)
        lr.fit(X, y)
        coeffs.append(lr.coef_)
    (elastic_net_coeffs, l1_coeffs, l2_coeffs) = coeffs
    assert not np.allclose(elastic_net_coeffs, l1_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(elastic_net_coeffs, l2_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(l2_coeffs, l1_coeffs, rtol=0, atol=0.1)

@pytest.mark.parametrize('C', [0.001, 0.1, 1, 10, 100, 1000, 1000000.0])
@pytest.mark.parametrize('penalty, l1_ratio', [('l1', 1), ('l2', 0)])
def test_elastic_net_l1_l2_equivalence(C, penalty, l1_ratio):
    if False:
        print('Hello World!')
    (X, y) = make_classification(random_state=0)
    lr_enet = LogisticRegression(penalty='elasticnet', C=C, l1_ratio=l1_ratio, solver='saga', random_state=0, tol=0.01)
    lr_expected = LogisticRegression(penalty=penalty, C=C, solver='saga', random_state=0, tol=0.01)
    lr_enet.fit(X, y)
    lr_expected.fit(X, y)
    assert_array_almost_equal(lr_enet.coef_, lr_expected.coef_)

@pytest.mark.parametrize('C', [0.001, 1, 100, 1000000.0])
def test_elastic_net_vs_l1_l2(C):
    if False:
        print('Hello World!')
    (X, y) = make_classification(500, random_state=0)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    param_grid = {'l1_ratio': np.linspace(0, 1, 5)}
    enet_clf = LogisticRegression(penalty='elasticnet', C=C, solver='saga', random_state=0, tol=0.01)
    gs = GridSearchCV(enet_clf, param_grid, refit=True)
    l1_clf = LogisticRegression(penalty='l1', C=C, solver='saga', random_state=0, tol=0.01)
    l2_clf = LogisticRegression(penalty='l2', C=C, solver='saga', random_state=0, tol=0.01)
    for clf in (gs, l1_clf, l2_clf):
        clf.fit(X_train, y_train)
    assert gs.score(X_test, y_test) >= l1_clf.score(X_test, y_test)
    assert gs.score(X_test, y_test) >= l2_clf.score(X_test, y_test)

@pytest.mark.parametrize('C', np.logspace(-3, 2, 4))
@pytest.mark.parametrize('l1_ratio', [0.1, 0.5, 0.9])
def test_LogisticRegression_elastic_net_objective(C, l1_ratio):
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_classification(n_samples=1000, n_classes=2, n_features=20, n_informative=10, n_redundant=0, n_repeated=0, random_state=0)
    X = scale(X)
    lr_enet = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, C=C, l1_ratio=l1_ratio, fit_intercept=False)
    lr_l2 = LogisticRegression(penalty='l2', solver='saga', random_state=0, C=C, fit_intercept=False)
    lr_enet.fit(X, y)
    lr_l2.fit(X, y)

    def enet_objective(lr):
        if False:
            return 10
        coef = lr.coef_.ravel()
        obj = C * log_loss(y, lr.predict_proba(X))
        obj += l1_ratio * np.sum(np.abs(coef))
        obj += (1.0 - l1_ratio) * 0.5 * np.dot(coef, coef)
        return obj
    assert enet_objective(lr_enet) < enet_objective(lr_l2)

@pytest.mark.parametrize('multi_class', ('ovr', 'multinomial'))
def test_LogisticRegressionCV_GridSearchCV_elastic_net(multi_class):
    if False:
        print('Hello World!')
    if multi_class == 'ovr':
        (X, y) = make_classification(random_state=0)
    else:
        (X, y) = make_classification(n_samples=100, n_classes=3, n_informative=3, random_state=0)
    cv = StratifiedKFold(5)
    l1_ratios = np.linspace(0, 1, 3)
    Cs = np.logspace(-4, 4, 3)
    lrcv = LogisticRegressionCV(penalty='elasticnet', Cs=Cs, solver='saga', cv=cv, l1_ratios=l1_ratios, random_state=0, multi_class=multi_class, tol=0.01)
    lrcv.fit(X, y)
    param_grid = {'C': Cs, 'l1_ratio': l1_ratios}
    lr = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, multi_class=multi_class, tol=0.01)
    gs = GridSearchCV(lr, param_grid, cv=cv)
    gs.fit(X, y)
    assert gs.best_params_['l1_ratio'] == lrcv.l1_ratio_[0]
    assert gs.best_params_['C'] == lrcv.C_[0]

def test_LogisticRegressionCV_GridSearchCV_elastic_net_ovr():
    if False:
        i = 10
        return i + 15
    (X, y) = make_classification(n_samples=100, n_classes=3, n_informative=3, random_state=0)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0)
    cv = StratifiedKFold(5)
    l1_ratios = np.linspace(0, 1, 3)
    Cs = np.logspace(-4, 4, 3)
    lrcv = LogisticRegressionCV(penalty='elasticnet', Cs=Cs, solver='saga', cv=cv, l1_ratios=l1_ratios, random_state=0, multi_class='ovr', tol=0.01)
    lrcv.fit(X_train, y_train)
    param_grid = {'C': Cs, 'l1_ratio': l1_ratios}
    lr = LogisticRegression(penalty='elasticnet', solver='saga', random_state=0, multi_class='ovr', tol=0.01)
    gs = GridSearchCV(lr, param_grid, cv=cv)
    gs.fit(X_train, y_train)
    assert (lrcv.predict(X_train) == gs.predict(X_train)).mean() >= 0.8
    assert (lrcv.predict(X_test) == gs.predict(X_test)).mean() >= 0.8

@pytest.mark.parametrize('penalty', ('l2', 'elasticnet'))
@pytest.mark.parametrize('multi_class', ('ovr', 'multinomial', 'auto'))
def test_LogisticRegressionCV_no_refit(penalty, multi_class):
    if False:
        for i in range(10):
            print('nop')
    n_classes = 3
    n_features = 20
    (X, y) = make_classification(n_samples=200, n_classes=n_classes, n_informative=n_classes, n_features=n_features, random_state=0)
    Cs = np.logspace(-4, 4, 3)
    if penalty == 'elasticnet':
        l1_ratios = np.linspace(0, 1, 2)
    else:
        l1_ratios = None
    lrcv = LogisticRegressionCV(penalty=penalty, Cs=Cs, solver='saga', l1_ratios=l1_ratios, random_state=0, multi_class=multi_class, tol=0.01, refit=False)
    lrcv.fit(X, y)
    assert lrcv.C_.shape == (n_classes,)
    assert lrcv.l1_ratio_.shape == (n_classes,)
    assert lrcv.coef_.shape == (n_classes, n_features)

def test_LogisticRegressionCV_elasticnet_attribute_shapes():
    if False:
        for i in range(10):
            print('nop')
    n_classes = 3
    n_features = 20
    (X, y) = make_classification(n_samples=200, n_classes=n_classes, n_informative=n_classes, n_features=n_features, random_state=0)
    Cs = np.logspace(-4, 4, 3)
    l1_ratios = np.linspace(0, 1, 2)
    n_folds = 2
    lrcv = LogisticRegressionCV(penalty='elasticnet', Cs=Cs, solver='saga', cv=n_folds, l1_ratios=l1_ratios, multi_class='ovr', random_state=0, tol=0.01)
    lrcv.fit(X, y)
    coefs_paths = np.asarray(list(lrcv.coefs_paths_.values()))
    assert coefs_paths.shape == (n_classes, n_folds, Cs.size, l1_ratios.size, n_features + 1)
    scores = np.asarray(list(lrcv.scores_.values()))
    assert scores.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)
    assert lrcv.n_iter_.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)

def test_l1_ratio_non_elasticnet():
    if False:
        print('Hello World!')
    msg = "l1_ratio parameter is only used when penalty is 'elasticnet'\\. Got \\(penalty=l1\\)"
    with pytest.warns(UserWarning, match=msg):
        LogisticRegression(penalty='l1', solver='saga', l1_ratio=0.5).fit(X, Y1)

@pytest.mark.parametrize('C', np.logspace(-3, 2, 4))
@pytest.mark.parametrize('l1_ratio', [0.1, 0.5, 0.9])
def test_elastic_net_versus_sgd(C, l1_ratio):
    if False:
        for i in range(10):
            print('nop')
    n_samples = 500
    (X, y) = make_classification(n_samples=n_samples, n_classes=2, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, random_state=1)
    X = scale(X)
    sgd = SGDClassifier(penalty='elasticnet', random_state=1, fit_intercept=False, tol=None, max_iter=2000, l1_ratio=l1_ratio, alpha=1.0 / C / n_samples, loss='log_loss')
    log = LogisticRegression(penalty='elasticnet', random_state=1, fit_intercept=False, tol=1e-05, max_iter=1000, l1_ratio=l1_ratio, C=C, solver='saga')
    sgd.fit(X, y)
    log.fit(X, y)
    assert_array_almost_equal(sgd.coef_, log.coef_, decimal=1)

def test_logistic_regression_path_coefs_multinomial():
    if False:
        i = 10
        return i + 15
    (X, y) = make_classification(n_samples=200, n_classes=3, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=0, n_features=2)
    Cs = [1e-05, 1, 10000]
    (coefs, _, _) = _logistic_regression_path(X, y, penalty='l1', Cs=Cs, solver='saga', random_state=0, multi_class='multinomial')
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[0], coefs[1], decimal=1)
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[0], coefs[2], decimal=1)
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[1], coefs[2], decimal=1)

@pytest.mark.parametrize('est', [LogisticRegression(random_state=0, max_iter=500), LogisticRegressionCV(random_state=0, cv=3, Cs=3, tol=0.001, max_iter=500)], ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize('solver', SOLVERS)
def test_logistic_regression_multi_class_auto(est, solver):
    if False:
        return 10

    def fit(X, y, **kw):
        if False:
            return 10
        return clone(est).set_params(**kw).fit(X, y)
    scaled_data = scale(iris.data)
    X = scaled_data[::10]
    X2 = scaled_data[1::10]
    y_multi = iris.target[::10]
    y_bin = y_multi == 0
    est_auto_bin = fit(X, y_bin, multi_class='auto', solver=solver)
    est_ovr_bin = fit(X, y_bin, multi_class='ovr', solver=solver)
    assert_allclose(est_auto_bin.coef_, est_ovr_bin.coef_)
    assert_allclose(est_auto_bin.predict_proba(X2), est_ovr_bin.predict_proba(X2))
    est_auto_multi = fit(X, y_multi, multi_class='auto', solver=solver)
    if solver in ('liblinear', 'newton-cholesky'):
        est_ovr_multi = fit(X, y_multi, multi_class='ovr', solver=solver)
        assert_allclose(est_auto_multi.coef_, est_ovr_multi.coef_)
        assert_allclose(est_auto_multi.predict_proba(X2), est_ovr_multi.predict_proba(X2))
    else:
        est_multi_multi = fit(X, y_multi, multi_class='multinomial', solver=solver)
        assert_allclose(est_auto_multi.coef_, est_multi_multi.coef_)
        assert_allclose(est_auto_multi.predict_proba(X2), est_multi_multi.predict_proba(X2))
        assert not np.allclose(est_auto_bin.coef_, fit(X, y_bin, multi_class='multinomial', solver=solver).coef_)
        assert not np.allclose(est_auto_bin.coef_, fit(X, y_multi, multi_class='multinomial', solver=solver).coef_)

@pytest.mark.parametrize('solver', sorted(set(SOLVERS) - set(['liblinear'])))
def test_penalty_none(solver):
    if False:
        print('Hello World!')
    (X, y) = make_classification(n_samples=1000, n_redundant=0, random_state=0)
    msg = 'Setting penalty=None will ignore the C'
    lr = LogisticRegression(penalty=None, solver=solver, C=4)
    with pytest.warns(UserWarning, match=msg):
        lr.fit(X, y)
    lr_none = LogisticRegression(penalty=None, solver=solver, random_state=0)
    lr_l2_C_inf = LogisticRegression(penalty='l2', C=np.inf, solver=solver, random_state=0)
    pred_none = lr_none.fit(X, y).predict(X)
    pred_l2_C_inf = lr_l2_C_inf.fit(X, y).predict(X)
    assert_array_equal(pred_none, pred_l2_C_inf)

@pytest.mark.parametrize('params', [{'penalty': 'l1', 'dual': False, 'tol': 1e-06, 'max_iter': 1000}, {'penalty': 'l2', 'dual': True, 'tol': 1e-12, 'max_iter': 1000}, {'penalty': 'l2', 'dual': False, 'tol': 1e-12, 'max_iter': 1000}])
def test_logisticregression_liblinear_sample_weight(params):
    if False:
        i = 10
        return i + 15
    X = np.array([[1, 3], [1, 3], [1, 3], [1, 3], [2, 1], [2, 1], [2, 1], [2, 1], [3, 3], [3, 3], [3, 3], [3, 3], [4, 1], [4, 1], [4, 1], [4, 1]], dtype=np.dtype('float'))
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype('int'))
    X2 = np.vstack([X, X])
    y2 = np.hstack([y, 3 - y])
    sample_weight = np.ones(shape=len(y) * 2)
    sample_weight[len(y):] = 0
    (X2, y2, sample_weight) = shuffle(X2, y2, sample_weight, random_state=0)
    base_clf = LogisticRegression(solver='liblinear', random_state=42)
    base_clf.set_params(**params)
    clf_no_weight = clone(base_clf).fit(X, y)
    clf_with_weight = clone(base_clf).fit(X2, y2, sample_weight=sample_weight)
    for method in ('predict', 'predict_proba', 'decision_function'):
        X_clf_no_weight = getattr(clf_no_weight, method)(X)
        X_clf_with_weight = getattr(clf_with_weight, method)(X)
        assert_allclose(X_clf_no_weight, X_clf_with_weight)

def test_scores_attribute_layout_elasticnet():
    if False:
        print('Hello World!')
    (X, y) = make_classification(n_samples=1000, random_state=0)
    cv = StratifiedKFold(n_splits=5)
    l1_ratios = [0.1, 0.9]
    Cs = [0.1, 1, 10]
    lrcv = LogisticRegressionCV(penalty='elasticnet', solver='saga', l1_ratios=l1_ratios, Cs=Cs, cv=cv, random_state=0, max_iter=250, tol=0.001)
    lrcv.fit(X, y)
    avg_scores_lrcv = lrcv.scores_[1].mean(axis=0)
    for (i, C) in enumerate(Cs):
        for (j, l1_ratio) in enumerate(l1_ratios):
            lr = LogisticRegression(penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio, random_state=0, max_iter=250, tol=0.001)
            avg_score_lr = cross_val_score(lr, X, y, cv=cv).mean()
            assert avg_scores_lrcv[i, j] == pytest.approx(avg_score_lr)

@pytest.mark.parametrize('fit_intercept', [False, True])
def test_multinomial_identifiability_on_iris(fit_intercept):
    if False:
        for i in range(10):
            print('nop')
    'Test that the multinomial classification is identifiable.\n\n    A multinomial with c classes can be modeled with\n    probability_k = exp(X@coef_k) / sum(exp(X@coef_l), l=1..c) for k=1..c.\n    This is not identifiable, unless one chooses a further constraint.\n    According to [1], the maximum of the L2 penalized likelihood automatically\n    satisfies the symmetric constraint:\n    sum(coef_k, k=1..c) = 0\n\n    Further details can be found in [2].\n\n    Reference\n    ---------\n    .. [1] :doi:`Zhu, Ji and Trevor J. Hastie. "Classification of gene microarrays by\n           penalized logistic regression". Biostatistics 5 3 (2004): 427-43.\n           <10.1093/biostatistics/kxg046>`\n\n    .. [2] :arxiv:`Noah Simon and Jerome Friedman and Trevor Hastie. (2013)\n           "A Blockwise Descent Algorithm for Group-penalized Multiresponse and\n           Multinomial Regression". <1311.6529>`\n    '
    (n_samples, n_features) = iris.data.shape
    target = iris.target_names[iris.target]
    clf = LogisticRegression(C=len(iris.data), solver='lbfgs', multi_class='multinomial', fit_intercept=fit_intercept)
    X_scaled = scale(iris.data)
    clf.fit(X_scaled, target)
    assert_allclose(clf.coef_.sum(axis=0), 0, atol=1e-10)
    if fit_intercept:
        clf.intercept_.sum(axis=0) == pytest.approx(0, abs=1e-15)

@pytest.mark.parametrize('multi_class', ['ovr', 'multinomial', 'auto'])
@pytest.mark.parametrize('class_weight', [{0: 1.0, 1: 10.0, 2: 1.0}, 'balanced'])
def test_sample_weight_not_modified(multi_class, class_weight):
    if False:
        i = 10
        return i + 15
    (X, y) = load_iris(return_X_y=True)
    n_features = len(X)
    W = np.ones(n_features)
    W[:n_features // 2] = 2
    expected = W.copy()
    clf = LogisticRegression(random_state=0, class_weight=class_weight, max_iter=200, multi_class=multi_class)
    clf.fit(X, y, sample_weight=W)
    assert_allclose(expected, W)

@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_large_sparse_matrix(solver, global_random_seed, csr_container):
    if False:
        while True:
            i = 10
    X = csr_container(sparse.rand(20, 10, random_state=global_random_seed))
    for attr in ['indices', 'indptr']:
        setattr(X, attr, getattr(X, attr).astype('int64'))
    rng = np.random.RandomState(global_random_seed)
    y = rng.randint(2, size=X.shape[0])
    if solver in ['liblinear', 'sag', 'saga']:
        msg = 'Only sparse matrices with 32-bit integer indices'
        with pytest.raises(ValueError, match=msg):
            LogisticRegression(solver=solver).fit(X, y)
    else:
        LogisticRegression(solver=solver).fit(X, y)

def test_single_feature_newton_cg():
    if False:
        print('Hello World!')
    X = np.array([[0.5, 0.65, 1.1, 1.25, 0.8, 0.54, 0.95, 0.7]]).T
    y = np.array([1, 1, 0, 0, 1, 1, 0, 1])
    assert X.shape[1] == 1
    LogisticRegression(solver='newton-cg', fit_intercept=True).fit(X, y)

def test_warning_on_penalty_string_none():
    if False:
        i = 10
        return i + 15
    target = iris.target_names[iris.target]
    lr = LogisticRegression(penalty='none')
    warning_message = "`penalty='none'`has been deprecated in 1.2 and will be removed in 1.4. To keep the past behaviour, set `penalty=None`."
    with pytest.warns(FutureWarning, match=warning_message):
        lr.fit(iris.data, target)

def test_liblinear_not_stuck():
    if False:
        for i in range(10):
            print('nop')
    X = iris.data.copy()
    y = iris.target.copy()
    X = X[y != 2]
    y = y[y != 2]
    X_prep = StandardScaler().fit_transform(X)
    C = l1_min_c(X, y, loss='log') * 10 ** (10 / 29)
    clf = LogisticRegression(penalty='l1', solver='liblinear', tol=1e-06, max_iter=100, intercept_scaling=10000.0, random_state=0, C=C)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        clf.fit(X_prep, y)

@pytest.mark.usefixtures('enable_slep006')
def test_lr_cv_scores_differ_when_sample_weight_is_requested():
    if False:
        print('Hello World!')
    'Test that `sample_weight` is correctly passed to the scorer in\n    `LogisticRegressionCV.fit` and `LogisticRegressionCV.score` by\n    checking the difference in scores with the case when `sample_weight`\n    is not requested.\n    '
    rng = np.random.RandomState(10)
    (X, y) = make_classification(n_samples=10, random_state=rng)
    (X_t, y_t) = make_classification(n_samples=10, random_state=rng)
    sample_weight = np.ones(len(y))
    sample_weight[:len(y) // 2] = 2
    kwargs = {'sample_weight': sample_weight}
    scorer1 = get_scorer('accuracy')
    lr_cv1 = LogisticRegressionCV(scoring=scorer1)
    lr_cv1.fit(X, y, **kwargs)
    scorer2 = get_scorer('accuracy')
    scorer2.set_score_request(sample_weight=True)
    lr_cv2 = LogisticRegressionCV(scoring=scorer2)
    lr_cv2.fit(X, y, **kwargs)
    assert not np.allclose(lr_cv1.scores_[1], lr_cv2.scores_[1])
    score_1 = lr_cv1.score(X_t, y_t, **kwargs)
    score_2 = lr_cv2.score(X_t, y_t, **kwargs)
    assert not np.allclose(score_1, score_2)

def test_lr_cv_scores_without_enabling_metadata_routing():
    if False:
        print('Hello World!')
    'Test that `sample_weight` is passed correctly to the scorer in\n    `LogisticRegressionCV.fit` and `LogisticRegressionCV.score` even\n    when `enable_metadata_routing=False`\n    '
    rng = np.random.RandomState(10)
    (X, y) = make_classification(n_samples=10, random_state=rng)
    (X_t, y_t) = make_classification(n_samples=10, random_state=rng)
    sample_weight = np.ones(len(y))
    sample_weight[:len(y) // 2] = 2
    kwargs = {'sample_weight': sample_weight}
    with config_context(enable_metadata_routing=False):
        scorer1 = get_scorer('accuracy')
        lr_cv1 = LogisticRegressionCV(scoring=scorer1)
        lr_cv1.fit(X, y, **kwargs)
        score_1 = lr_cv1.score(X_t, y_t, **kwargs)
    with config_context(enable_metadata_routing=True):
        scorer2 = get_scorer('accuracy')
        scorer2.set_score_request(sample_weight=True)
        lr_cv2 = LogisticRegressionCV(scoring=scorer2)
        lr_cv2.fit(X, y, **kwargs)
        score_2 = lr_cv2.score(X_t, y_t, **kwargs)
    assert_allclose(lr_cv1.scores_[1], lr_cv2.scores_[1])
    assert_allclose(score_1, score_2)

@pytest.mark.parametrize('solver', SOLVERS)
def test_zero_max_iter(solver):
    if False:
        i = 10
        return i + 15
    (X, y) = load_iris(return_X_y=True)
    y = y == 2
    with ignore_warnings(category=ConvergenceWarning):
        clf = LogisticRegression(solver=solver, max_iter=0).fit(X, y)
    if solver not in ['saga', 'sag']:
        assert clf.n_iter_ == 0
    if solver != 'lbfgs':
        assert_allclose(clf.coef_, np.zeros_like(clf.coef_))
        assert_allclose(clf.decision_function(X), np.full(shape=X.shape[0], fill_value=clf.intercept_))
        assert_allclose(clf.predict_proba(X), np.full(shape=(X.shape[0], 2), fill_value=0.5))
    assert clf.score(X, y) < 0.7

def test_passing_params_without_enabling_metadata_routing():
    if False:
        print('Hello World!')
    'Test that the right error message is raised when metadata params\n    are passed while not supported when `enable_metadata_routing=False`.'
    (X, y) = make_classification(n_samples=10, random_state=0)
    lr_cv = LogisticRegressionCV()
    msg = 'is only supported if enable_metadata_routing=True'
    with config_context(enable_metadata_routing=False):
        params = {'extra_param': 1.0}
        with pytest.raises(ValueError, match=msg):
            lr_cv.fit(X, y, **params)
        with pytest.raises(ValueError, match=msg):
            lr_cv.score(X, y, **params)