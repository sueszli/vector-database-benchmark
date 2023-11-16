import pickle
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import datasets, linear_model, metrics
from sklearn.base import clone, is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import _sgd_fast as sgd_fast
from sklearn.linear_model import _stochastic_gradient
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, scale
from sklearn.svm import OneClassSVM
from sklearn.utils._testing import assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal, ignore_warnings

def _update_kwargs(kwargs):
    if False:
        for i in range(10):
            print('nop')
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 42
    if 'tol' not in kwargs:
        kwargs['tol'] = None
    if 'max_iter' not in kwargs:
        kwargs['max_iter'] = 5

class _SparseSGDClassifier(linear_model.SGDClassifier):

    def fit(self, X, y, *args, **kw):
        if False:
            i = 10
            return i + 15
        X = sp.csr_matrix(X)
        return super().fit(X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        X = sp.csr_matrix(X)
        return super().partial_fit(X, y, *args, **kw)

    def decision_function(self, X):
        if False:
            print('Hello World!')
        X = sp.csr_matrix(X)
        return super().decision_function(X)

    def predict_proba(self, X):
        if False:
            i = 10
            return i + 15
        X = sp.csr_matrix(X)
        return super().predict_proba(X)

class _SparseSGDRegressor(linear_model.SGDRegressor):

    def fit(self, X, y, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.fit(self, X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        if False:
            print('Hello World!')
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.partial_fit(self, X, y, *args, **kw)

    def decision_function(self, X, *args, **kw):
        if False:
            while True:
                i = 10
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.decision_function(self, X, *args, **kw)

class _SparseSGDOneClassSVM(linear_model.SGDOneClassSVM):

    def fit(self, X, *args, **kw):
        if False:
            return 10
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.fit(self, X, *args, **kw)

    def partial_fit(self, X, *args, **kw):
        if False:
            return 10
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.partial_fit(self, X, *args, **kw)

    def decision_function(self, X, *args, **kw):
        if False:
            while True:
                i = 10
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.decision_function(self, X, *args, **kw)

def SGDClassifier(**kwargs):
    if False:
        return 10
    _update_kwargs(kwargs)
    return linear_model.SGDClassifier(**kwargs)

def SGDRegressor(**kwargs):
    if False:
        return 10
    _update_kwargs(kwargs)
    return linear_model.SGDRegressor(**kwargs)

def SGDOneClassSVM(**kwargs):
    if False:
        return 10
    _update_kwargs(kwargs)
    return linear_model.SGDOneClassSVM(**kwargs)

def SparseSGDClassifier(**kwargs):
    if False:
        return 10
    _update_kwargs(kwargs)
    return _SparseSGDClassifier(**kwargs)

def SparseSGDRegressor(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    _update_kwargs(kwargs)
    return _SparseSGDRegressor(**kwargs)

def SparseSGDOneClassSVM(**kwargs):
    if False:
        while True:
            i = 10
    _update_kwargs(kwargs)
    return _SparseSGDOneClassSVM(**kwargs)
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y = [1, 1, 1, 2, 2, 2]
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = [1, 2, 2]
X2 = np.array([[-1, 1], [-0.75, 0.5], [-1.5, 1.5], [1, 1], [0.75, 0.5], [1.5, 1.5], [-1, -1], [0, -0.5], [1, -1]])
Y2 = ['one'] * 3 + ['two'] * 3 + ['three'] * 3
T2 = np.array([[-1.5, 0.5], [1, 2], [0, -2]])
true_result2 = ['one', 'two', 'three']
X3 = np.array([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]])
Y3 = np.array([1, 1, 1, 1, 2, 2, 2, 2])
X4 = np.array([[1, 0.9, 0.8, 0, 0, 0], [1, 0.84, 0.98, 0, 0, 0], [1, 0.96, 0.88, 0, 0, 0], [1, 0.91, 0.99, 0, 0, 0], [0, 0, 0, 0.89, 0.91, 1], [0, 0, 0, 0.79, 0.84, 1], [0, 0, 0, 0.91, 0.95, 1], [0, 0, 0, 0.93, 1, 1]])
Y4 = np.array([1, 1, 1, 1, 2, 2, 2, 2])
iris = datasets.load_iris()
X5 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y5 = [1, 1, 1, 2, 2, 2]
true_result5 = [0, 1, 1]

def asgd(klass, X, y, eta, alpha, weight_init=None, intercept_init=0.0):
    if False:
        return 10
    if weight_init is None:
        weights = np.zeros(X.shape[1])
    else:
        weights = weight_init
    average_weights = np.zeros(X.shape[1])
    intercept = intercept_init
    average_intercept = 0.0
    decay = 1.0
    if klass in (SparseSGDClassifier, SparseSGDRegressor):
        decay = 0.01
    for (i, entry) in enumerate(X):
        p = np.dot(entry, weights)
        p += intercept
        gradient = p - y[i]
        weights *= 1.0 - eta * alpha
        weights += -(eta * gradient * entry)
        intercept += -(eta * gradient) * decay
        average_weights *= i
        average_weights += weights
        average_weights /= i + 1.0
        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0
    return (average_weights, average_intercept)

def _test_warm_start(klass, X, Y, lr):
    if False:
        print('Hello World!')
    clf = klass(alpha=0.01, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X, Y)
    clf2 = klass(alpha=0.001, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, Y, coef_init=clf.coef_.copy(), intercept_init=clf.intercept_.copy())
    clf3 = klass(alpha=0.01, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X, Y)
    assert clf3.t_ == clf.t_
    assert_array_almost_equal(clf3.coef_, clf.coef_)
    clf3.set_params(alpha=0.001)
    clf3.fit(X, Y)
    assert clf3.t_ == clf2.t_
    assert_array_almost_equal(clf3.coef_, clf2.coef_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_warm_start(klass, lr):
    if False:
        i = 10
        return i + 15
    _test_warm_start(klass, X, Y, lr)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_input_format(klass):
    if False:
        while True:
            i = 10
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    Y_ = np.array(Y)[:, np.newaxis]
    Y_ = np.c_[Y_, Y_]
    with pytest.raises(ValueError):
        clf.fit(X, Y_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_clone(klass):
    if False:
        while True:
            i = 10
    clf = klass(alpha=0.01, penalty='l1')
    clf = clone(clf)
    clf.set_params(penalty='l2')
    clf.fit(X, Y)
    clf2 = klass(alpha=0.01, penalty='l2')
    clf2.fit(X, Y)
    assert_array_equal(clf.coef_, clf2.coef_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_plain_has_no_average_attr(klass):
    if False:
        print('Hello World!')
    clf = klass(average=True, eta0=0.01)
    clf.fit(X, Y)
    assert hasattr(clf, '_average_coef')
    assert hasattr(clf, '_average_intercept')
    assert hasattr(clf, '_standard_intercept')
    assert hasattr(clf, '_standard_coef')
    clf = klass()
    clf.fit(X, Y)
    assert not hasattr(clf, '_average_coef')
    assert not hasattr(clf, '_average_intercept')
    assert not hasattr(clf, '_standard_intercept')
    assert not hasattr(clf, '_standard_coef')

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_late_onset_averaging_not_reached(klass):
    if False:
        i = 10
        return i + 15
    clf1 = klass(average=600)
    clf2 = klass()
    for _ in range(100):
        if is_classifier(clf1):
            clf1.partial_fit(X, Y, classes=np.unique(Y))
            clf2.partial_fit(X, Y, classes=np.unique(Y))
        else:
            clf1.partial_fit(X, Y)
            clf2.partial_fit(X, Y)
    assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=16)
    if klass in [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]:
        assert_almost_equal(clf1.intercept_, clf2.intercept_, decimal=16)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        assert_allclose(clf1.offset_, clf2.offset_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_late_onset_averaging_reached(klass):
    if False:
        i = 10
        return i + 15
    eta0 = 0.001
    alpha = 0.0001
    Y_encode = np.array(Y)
    Y_encode[Y_encode == 1] = -1.0
    Y_encode[Y_encode == 2] = 1.0
    clf1 = klass(average=7, learning_rate='constant', loss='squared_error', eta0=eta0, alpha=alpha, max_iter=2, shuffle=False)
    clf2 = klass(average=0, learning_rate='constant', loss='squared_error', eta0=eta0, alpha=alpha, max_iter=1, shuffle=False)
    clf1.fit(X, Y_encode)
    clf2.fit(X, Y_encode)
    (average_weights, average_intercept) = asgd(klass, X, Y_encode, eta0, alpha, weight_init=clf2.coef_.ravel(), intercept_init=clf2.intercept_)
    assert_array_almost_equal(clf1.coef_.ravel(), average_weights.ravel(), decimal=16)
    assert_almost_equal(clf1.intercept_, average_intercept, decimal=16)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_early_stopping(klass):
    if False:
        for i in range(10):
            print('nop')
    X = iris.data[iris.target > 0]
    Y = iris.target[iris.target > 0]
    for early_stopping in [True, False]:
        max_iter = 1000
        clf = klass(early_stopping=early_stopping, tol=0.001, max_iter=max_iter).fit(X, Y)
        assert clf.n_iter_ < max_iter

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_adaptive_longer_than_constant(klass):
    if False:
        return 10
    clf1 = klass(learning_rate='adaptive', eta0=0.01, tol=0.001, max_iter=100)
    clf1.fit(iris.data, iris.target)
    clf2 = klass(learning_rate='constant', eta0=0.01, tol=0.001, max_iter=100)
    clf2.fit(iris.data, iris.target)
    assert clf1.n_iter_ > clf2.n_iter_

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_validation_set_not_used_for_training(klass):
    if False:
        print('Hello World!')
    (X, Y) = (iris.data, iris.target)
    validation_fraction = 0.4
    seed = 42
    shuffle = False
    max_iter = 10
    clf1 = klass(early_stopping=True, random_state=np.random.RandomState(seed), validation_fraction=validation_fraction, learning_rate='constant', eta0=0.01, tol=None, max_iter=max_iter, shuffle=shuffle)
    clf1.fit(X, Y)
    assert clf1.n_iter_ == max_iter
    clf2 = klass(early_stopping=False, random_state=np.random.RandomState(seed), learning_rate='constant', eta0=0.01, tol=None, max_iter=max_iter, shuffle=shuffle)
    if is_classifier(clf2):
        cv = StratifiedShuffleSplit(test_size=validation_fraction, random_state=seed)
    else:
        cv = ShuffleSplit(test_size=validation_fraction, random_state=seed)
    (idx_train, idx_val) = next(cv.split(X, Y))
    idx_train = np.sort(idx_train)
    clf2.fit(X[idx_train], Y[idx_train])
    assert clf2.n_iter_ == max_iter
    assert_array_equal(clf1.coef_, clf2.coef_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_n_iter_no_change(klass):
    if False:
        for i in range(10):
            print('nop')
    (X, Y) = (iris.data, iris.target)
    for early_stopping in [True, False]:
        n_iter_list = [klass(early_stopping=early_stopping, n_iter_no_change=n_iter_no_change, tol=0.0001, max_iter=1000).fit(X, Y).n_iter_ for n_iter_no_change in [2, 3, 10]]
        assert_array_equal(n_iter_list, sorted(n_iter_list))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_not_enough_sample_for_early_stopping(klass):
    if False:
        print('Hello World!')
    clf = klass(early_stopping=True, validation_fraction=0.99)
    with pytest.raises(ValueError):
        clf.fit(X3, Y3)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_clf(klass):
    if False:
        for i in range(10):
            print('nop')
    for loss in ('hinge', 'squared_hinge', 'log_loss', 'modified_huber'):
        clf = klass(penalty='l2', alpha=0.01, fit_intercept=True, loss=loss, max_iter=10, shuffle=True)
        clf.fit(X, Y)
        assert_array_equal(clf.predict(T), true_result)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_provide_coef(klass):
    if False:
        while True:
            i = 10
    'Check that the shape of `coef_init` is validated.'
    with pytest.raises(ValueError, match='Provided coef_init does not match dataset'):
        klass().fit(X, Y, coef_init=np.zeros((3,)))

@pytest.mark.parametrize('klass, fit_params', [(SGDClassifier, {'intercept_init': np.zeros((3,))}), (SparseSGDClassifier, {'intercept_init': np.zeros((3,))}), (SGDOneClassSVM, {'offset_init': np.zeros((3,))}), (SparseSGDOneClassSVM, {'offset_init': np.zeros((3,))})])
def test_set_intercept_offset(klass, fit_params):
    if False:
        return 10
    'Check that `intercept_init` or `offset_init` is validated.'
    sgd_estimator = klass()
    with pytest.raises(ValueError, match='does not match dataset'):
        sgd_estimator.fit(X, Y, **fit_params)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor])
def test_sgd_early_stopping_with_partial_fit(klass):
    if False:
        print('Hello World!')
    'Check that we raise an error for `early_stopping` used with\n    `partial_fit`.\n    '
    err_msg = 'early_stopping should be False with partial_fit'
    with pytest.raises(ValueError, match=err_msg):
        klass(early_stopping=True).partial_fit(X, Y)

@pytest.mark.parametrize('klass, fit_params', [(SGDClassifier, {'intercept_init': 0}), (SparseSGDClassifier, {'intercept_init': 0}), (SGDOneClassSVM, {'offset_init': 0}), (SparseSGDOneClassSVM, {'offset_init': 0})])
def test_set_intercept_offset_binary(klass, fit_params):
    if False:
        print('Hello World!')
    'Check that we can pass a scaler with binary classification to\n    `intercept_init` or `offset_init`.'
    klass().fit(X5, Y5, **fit_params)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_average_binary_computed_correctly(klass):
    if False:
        return 10
    eta = 0.1
    alpha = 2.0
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    y = np.dot(X, w)
    y = np.sign(y)
    clf.fit(X, y)
    (average_weights, average_intercept) = asgd(klass, X, y, eta, alpha)
    average_weights = average_weights.reshape(1, -1)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=14)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=14)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_set_intercept_to_intercept(klass):
    if False:
        while True:
            i = 10
    clf = klass().fit(X5, Y5)
    klass().fit(X5, Y5, intercept_init=clf.intercept_)
    clf = klass().fit(X, Y)
    klass().fit(X, Y, intercept_init=clf.intercept_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_at_least_two_labels(klass):
    if False:
        return 10
    clf = klass(alpha=0.01, max_iter=20)
    with pytest.raises(ValueError):
        clf.fit(X2, np.ones(9))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_weight_class_balanced(klass):
    if False:
        for i in range(10):
            print('nop')
    regex = "class_weight 'balanced' is not supported for partial_fit\\. In order to use 'balanced' weights, use compute_class_weight\\('balanced', classes=classes, y=y\\). In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions\\. Pass the resulting weights as the class_weight parameter\\."
    with pytest.raises(ValueError, match=regex):
        klass(class_weight='balanced').partial_fit(X, Y, classes=np.unique(Y))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass(klass):
    if False:
        print('Hello World!')
    clf = klass(alpha=0.01, max_iter=20).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_average(klass):
    if False:
        i = 10
        return i + 15
    eta = 0.001
    alpha = 0.01
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    np_Y2 = np.array(Y2)
    clf.fit(X2, np_Y2)
    classes = np.unique(np_Y2)
    for (i, cl) in enumerate(classes):
        y_i = np.ones(np_Y2.shape[0])
        y_i[np_Y2 != cl] = -1
        (average_coef, average_intercept) = asgd(klass, X2, y_i, eta, alpha)
        assert_array_almost_equal(average_coef, clf.coef_[i], decimal=16)
        assert_almost_equal(average_intercept, clf.intercept_[i], decimal=16)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_with_init_coef(klass):
    if False:
        i = 10
        return i + 15
    clf = klass(alpha=0.01, max_iter=20)
    clf.fit(X2, Y2, coef_init=np.zeros((3, 2)), intercept_init=np.zeros(3))
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape, (3,)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_njobs(klass):
    if False:
        i = 10
        return i + 15
    clf = klass(alpha=0.01, max_iter=20, n_jobs=2).fit(X2, Y2)
    assert clf.coef_.shape == (3, 2)
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_set_coef_multiclass(klass):
    if False:
        for i in range(10):
            print('nop')
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, coef_init=np.zeros((2, 2)))
    clf = klass().fit(X2, Y2, coef_init=np.zeros((3, 2)))
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, intercept_init=np.zeros((1,)))
    clf = klass().fit(X2, Y2, intercept_init=np.zeros((3,)))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_predict_proba_method_access(klass):
    if False:
        while True:
            i = 10
    for loss in linear_model.SGDClassifier.loss_functions:
        clf = SGDClassifier(loss=loss)
        if loss in ('log_loss', 'modified_huber'):
            assert hasattr(clf, 'predict_proba')
            assert hasattr(clf, 'predict_log_proba')
        else:
            message = 'probability estimates are not available for loss={!r}'.format(loss)
            assert not hasattr(clf, 'predict_proba')
            assert not hasattr(clf, 'predict_log_proba')
            with pytest.raises(AttributeError, match=message):
                clf.predict_proba
            with pytest.raises(AttributeError, match=message):
                clf.predict_log_proba

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_proba(klass):
    if False:
        i = 10
        return i + 15
    clf = SGDClassifier(loss='hinge', alpha=0.01, max_iter=10, tol=None).fit(X, Y)
    assert not hasattr(clf, 'predict_proba')
    assert not hasattr(clf, 'predict_log_proba')
    for loss in ['log_loss', 'modified_huber']:
        clf = klass(loss=loss, alpha=0.01, max_iter=10)
        clf.fit(X, Y)
        p = clf.predict_proba([[3, 2]])
        assert p[0, 1] > 0.5
        p = clf.predict_proba([[-1, -1]])
        assert p[0, 1] < 0.5
        p = clf.predict_log_proba([[3, 2]])
        assert p[0, 1] > p[0, 0]
        p = clf.predict_log_proba([[-1, -1]])
        assert p[0, 1] < p[0, 0]
    clf = klass(loss='log_loss', alpha=0.01, max_iter=10).fit(X2, Y2)
    d = clf.decision_function([[0.1, -0.1], [0.3, 0.2]])
    p = clf.predict_proba([[0.1, -0.1], [0.3, 0.2]])
    assert_array_equal(np.argmax(p, axis=1), np.argmax(d, axis=1))
    assert_almost_equal(p[0].sum(), 1)
    assert np.all(p[0] >= 0)
    p = clf.predict_proba([[-1, -1]])
    d = clf.decision_function([[-1, -1]])
    assert_array_equal(np.argsort(p[0]), np.argsort(d[0]))
    lp = clf.predict_log_proba([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    assert_array_almost_equal(np.log(p), lp)
    lp = clf.predict_log_proba([[-1, -1]])
    p = clf.predict_proba([[-1, -1]])
    assert_array_almost_equal(np.log(p), lp)
    clf = klass(loss='modified_huber', alpha=0.01, max_iter=10)
    clf.fit(X2, Y2)
    d = clf.decision_function([[3, 2]])
    p = clf.predict_proba([[3, 2]])
    if klass != SparseSGDClassifier:
        assert np.argmax(d, axis=1) == np.argmax(p, axis=1)
    else:
        assert np.argmin(d, axis=1) == np.argmin(p, axis=1)
    x = X.mean(axis=0)
    d = clf.decision_function([x])
    if np.all(d < -1):
        p = clf.predict_proba([x])
        assert_array_almost_equal(p[0], [1 / 3.0] * 3)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sgd_l1(klass):
    if False:
        return 10
    n = len(X4)
    rng = np.random.RandomState(13)
    idx = np.arange(n)
    rng.shuffle(idx)
    X = X4[idx, :]
    Y = Y4[idx]
    clf = klass(penalty='l1', alpha=0.2, fit_intercept=False, max_iter=2000, tol=None, shuffle=False)
    clf.fit(X, Y)
    assert_array_equal(clf.coef_[0, 1:-1], np.zeros((4,)))
    pred = clf.predict(X)
    assert_array_equal(pred, Y)
    clf.sparsify()
    assert sp.issparse(clf.coef_)
    pred = clf.predict(X)
    assert_array_equal(pred, Y)
    clf = pickle.loads(pickle.dumps(clf))
    assert sp.issparse(clf.coef_)
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_class_weights(klass):
    if False:
        print('Hello World!')
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight=None)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight={1: 0.001})
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_equal_class_weight(klass):
    if False:
        return 10
    X = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y = [0, 0, 1, 1]
    clf = klass(alpha=0.1, max_iter=1000, class_weight=None)
    clf.fit(X, y)
    X = [[1, 0], [0, 1]]
    y = [0, 1]
    clf_weighted = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5, 1: 0.5})
    clf_weighted.fit(X, y)
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_wrong_class_weight_label(klass):
    if False:
        while True:
            i = 10
    clf = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5})
    with pytest.raises(ValueError):
        clf.fit(X, Y)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_weights_multiplied(klass):
    if False:
        return 10
    class_weights = {1: 0.6, 2: 0.3}
    rng = np.random.RandomState(0)
    sample_weights = rng.random_sample(Y4.shape[0])
    multiplied_together = np.copy(sample_weights)
    multiplied_together[Y4 == 1] *= class_weights[1]
    multiplied_together[Y4 == 2] *= class_weights[2]
    clf1 = klass(alpha=0.1, max_iter=20, class_weight=class_weights)
    clf2 = klass(alpha=0.1, max_iter=20)
    clf1.fit(X4, Y4, sample_weight=sample_weights)
    clf2.fit(X4, Y4, sample_weight=multiplied_together)
    assert_almost_equal(clf1.coef_, clf2.coef_)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_balanced_weight(klass):
    if False:
        return 10
    (X, y) = (iris.data, iris.target)
    X = scale(X)
    idx = np.arange(X.shape[0])
    rng = np.random.RandomState(6)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    clf = klass(alpha=0.0001, max_iter=1000, class_weight=None, shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf.predict(X), average='weighted')
    assert_almost_equal(f1, 0.96, decimal=1)
    clf_balanced = klass(alpha=0.0001, max_iter=1000, class_weight='balanced', shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf_balanced.predict(X), average='weighted')
    assert_almost_equal(f1, 0.96, decimal=1)
    assert_array_almost_equal(clf.coef_, clf_balanced.coef_, 6)
    X_0 = X[y == 0, :]
    y_0 = y[y == 0]
    X_imbalanced = np.vstack([X] + [X_0] * 10)
    y_imbalanced = np.concatenate([y] + [y_0] * 10)
    clf = klass(max_iter=1000, class_weight=None, shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average='weighted') < 0.96
    clf = klass(max_iter=1000, class_weight='balanced', shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average='weighted') > 0.96

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_sample_weights(klass):
    if False:
        return 10
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))
    clf.fit(X, y, sample_weight=[0.001] * 3 + [1] * 2)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_wrong_sample_weights(klass):
    if False:
        while True:
            i = 10
    if klass in [SGDClassifier, SparseSGDClassifier]:
        clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        clf = klass(nu=0.1, max_iter=1000, fit_intercept=False)
    with pytest.raises(ValueError):
        clf.fit(X, Y, sample_weight=np.arange(7))

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_exception(klass):
    if False:
        return 10
    clf = klass(alpha=0.01)
    with pytest.raises(ValueError):
        clf.partial_fit(X3, Y3)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_binary(klass):
    if False:
        i = 10
        return i + 15
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)
    classes = np.unique(Y)
    clf.partial_fit(X[:third], Y[:third], classes=classes)
    assert clf.coef_.shape == (1, X.shape[1])
    assert clf.intercept_.shape == (1,)
    assert clf.decision_function([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)
    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    assert id1, id2
    y_pred = clf.predict(T)
    assert_array_equal(y_pred, true_result)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_multiclass(klass):
    if False:
        while True:
            i = 10
    third = X2.shape[0] // 3
    clf = klass(alpha=0.01)
    classes = np.unique(Y2)
    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    id1 = id(clf.coef_.data)
    clf.partial_fit(X2[third:], Y2[third:])
    id2 = id(clf.coef_.data)
    assert id1, id2

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_multiclass_average(klass):
    if False:
        i = 10
        return i + 15
    third = X2.shape[0] // 3
    clf = klass(alpha=0.01, average=X2.shape[0])
    classes = np.unique(Y2)
    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)
    clf.partial_fit(X2[third:], Y2[third:])
    assert clf.coef_.shape == (3, X2.shape[1])
    assert clf.intercept_.shape == (3,)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_fit_then_partial_fit(klass):
    if False:
        print('Hello World!')
    clf = klass()
    clf.fit(X2, Y2)
    clf.partial_fit(X2, Y2)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_partial_fit_equal_fit_classif(klass, lr):
    if False:
        return 10
    for (X_, Y_, T_) in ((X, Y, T), (X2, Y2, T2)):
        clf = klass(alpha=0.01, eta0=0.01, max_iter=2, learning_rate=lr, shuffle=False)
        clf.fit(X_, Y_)
        y_pred = clf.decision_function(T_)
        t = clf.t_
        classes = np.unique(Y_)
        clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
        for i in range(2):
            clf.partial_fit(X_, Y_, classes=classes)
        y_pred2 = clf.decision_function(T_)
        assert clf.t_ == t
        assert_array_almost_equal(y_pred, y_pred2, decimal=2)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_regression_losses(klass):
    if False:
        return 10
    random_state = np.random.RandomState(1)
    clf = klass(alpha=0.01, learning_rate='constant', eta0=0.1, loss='epsilon_insensitive', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)
    clf = klass(alpha=0.01, learning_rate='constant', eta0=0.1, loss='squared_epsilon_insensitive', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)
    clf = klass(alpha=0.01, loss='huber', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)
    clf = klass(alpha=0.01, learning_rate='constant', eta0=0.01, loss='squared_error', random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_warm_start_multiclass(klass):
    if False:
        print('Hello World!')
    _test_warm_start(klass, X2, Y2, 'optimal')

@pytest.mark.parametrize('klass', [SGDClassifier, SparseSGDClassifier])
def test_multiple_fit(klass):
    if False:
        print('Hello World!')
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    assert hasattr(clf, 'coef_')
    y = [['ham', 'spam'][i] for i in LabelEncoder().fit_transform(Y)]
    clf.fit(X[:, :-1], y)

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_reg(klass):
    if False:
        i = 10
        return i + 15
    clf = klass(alpha=0.1, max_iter=2, fit_intercept=False)
    clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    assert clf.coef_[0] == clf.coef_[1]

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_computed_correctly(klass):
    if False:
        print('Hello World!')
    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w)
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    clf.fit(X, y)
    (average_weights, average_intercept) = asgd(klass, X, y, eta, alpha)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_partial_fit(klass):
    if False:
        for i in range(10):
            print('nop')
    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    y = np.dot(X, w)
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    clf.partial_fit(X[:int(n_samples / 2)][:], y[:int(n_samples / 2)])
    clf.partial_fit(X[int(n_samples / 2):][:], y[int(n_samples / 2):])
    (average_weights, average_intercept) = asgd(klass, X, y, eta, alpha)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_[0], average_intercept, decimal=16)

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_average_sparse(klass):
    if False:
        while True:
            i = 10
    eta = 0.001
    alpha = 0.01
    clf = klass(loss='squared_error', learning_rate='constant', eta0=eta, alpha=alpha, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    n_samples = Y3.shape[0]
    clf.partial_fit(X3[:int(n_samples / 2)][:], Y3[:int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2):][:], Y3[int(n_samples / 2):])
    (average_weights, average_intercept) = asgd(klass, X3, Y3, eta, alpha)
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_least_squares_fit(klass):
    if False:
        while True:
            i = 10
    (xmin, xmax) = (-5, 5)
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)
    y = 0.5 * X.ravel()
    clf = klass(loss='squared_error', alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()
    clf = klass(loss='squared_error', alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_epsilon_insensitive(klass):
    if False:
        i = 10
        return i + 15
    (xmin, xmax) = (-5, 5)
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)
    y = 0.5 * X.ravel()
    clf = klass(loss='epsilon_insensitive', epsilon=0.01, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()
    clf = klass(loss='epsilon_insensitive', epsilon=0.01, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_sgd_huber_fit(klass):
    if False:
        return 10
    (xmin, xmax) = (-5, 5)
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)
    y = 0.5 * X.ravel()
    clf = klass(loss='huber', epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.99
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()
    clf = klass(loss='huber', epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    clf.fit(X, y)
    score = clf.score(X, y)
    assert score > 0.5

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_elasticnet_convergence(klass):
    if False:
        for i in range(10):
            print('nop')
    (n_samples, n_features) = (1000, 5)
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    ground_truth_coef = rng.randn(n_features)
    y = np.dot(X, ground_truth_coef)
    for alpha in [0.01, 0.001]:
        for l1_ratio in [0.5, 0.8, 1.0]:
            cd = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
            cd.fit(X, y)
            sgd = klass(penalty='elasticnet', max_iter=50, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
            sgd.fit(X, y)
            err_msg = 'cd and sgd did not converge to comparable results for alpha=%f and l1_ratio=%f' % (alpha, l1_ratio)
            assert_almost_equal(cd.coef_, sgd.coef_, decimal=2, err_msg=err_msg)

@ignore_warnings
@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_partial_fit(klass):
    if False:
        print('Hello World!')
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)
    clf.partial_fit(X[:third], Y[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.intercept_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)
    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    assert id1, id2

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_partial_fit_equal_fit(klass, lr):
    if False:
        while True:
            i = 10
    clf = klass(alpha=0.01, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X, Y)
    y_pred = clf.predict(T)
    t = clf.t_
    clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
    for i in range(2):
        clf.partial_fit(X, Y)
    y_pred2 = clf.predict(T)
    assert clf.t_ == t
    assert_array_almost_equal(y_pred, y_pred2, decimal=2)

@pytest.mark.parametrize('klass', [SGDRegressor, SparseSGDRegressor])
def test_loss_function_epsilon(klass):
    if False:
        while True:
            i = 10
    clf = klass(epsilon=0.9)
    clf.set_params(epsilon=0.1)
    assert clf.loss_functions['huber'][1] == 0.1

def asgd_oneclass(klass, X, eta, nu, coef_init=None, offset_init=0.0):
    if False:
        print('Hello World!')
    if coef_init is None:
        coef = np.zeros(X.shape[1])
    else:
        coef = coef_init
    average_coef = np.zeros(X.shape[1])
    offset = offset_init
    intercept = 1 - offset
    average_intercept = 0.0
    decay = 1.0
    if klass == SparseSGDOneClassSVM:
        decay = 0.01
    for (i, entry) in enumerate(X):
        p = np.dot(entry, coef)
        p += intercept
        if p <= 1.0:
            gradient = -1
        else:
            gradient = 0
        coef *= max(0, 1.0 - eta * nu / 2)
        coef += -(eta * gradient * entry)
        intercept += -(eta * (nu + gradient)) * decay
        average_coef *= i
        average_coef += coef
        average_coef /= i + 1.0
        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0
    return (average_coef, 1 - average_intercept)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def _test_warm_start_oneclass(klass, X, lr):
    if False:
        i = 10
        return i + 15
    clf = klass(nu=0.5, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X)
    clf2 = klass(nu=0.1, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, coef_init=clf.coef_.copy(), offset_init=clf.offset_.copy())
    clf3 = klass(nu=0.5, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X)
    assert clf3.t_ == clf.t_
    assert_allclose(clf3.coef_, clf.coef_)
    clf3.set_params(nu=0.1)
    clf3.fit(X)
    assert clf3.t_ == clf2.t_
    assert_allclose(clf3.coef_, clf2.coef_)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_warm_start_oneclass(klass, lr):
    if False:
        for i in range(10):
            print('nop')
    _test_warm_start_oneclass(klass, X, lr)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_clone_oneclass(klass):
    if False:
        while True:
            i = 10
    clf = klass(nu=0.5)
    clf = clone(clf)
    clf.set_params(nu=0.1)
    clf.fit(X)
    clf2 = klass(nu=0.1)
    clf2.fit(X)
    assert_array_equal(clf.coef_, clf2.coef_)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_partial_fit_oneclass(klass):
    if False:
        for i in range(10):
            print('nop')
    third = X.shape[0] // 3
    clf = klass(nu=0.1)
    clf.partial_fit(X[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.offset_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    previous_coefs = clf.coef_
    clf.partial_fit(X[third:])
    assert clf.coef_ is previous_coefs
    with pytest.raises(ValueError):
        clf.partial_fit(X[:, 1])

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize('lr', ['constant', 'optimal', 'invscaling', 'adaptive'])
def test_partial_fit_equal_fit_oneclass(klass, lr):
    if False:
        print('Hello World!')
    clf = klass(nu=0.05, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X)
    y_scores = clf.decision_function(T)
    t = clf.t_
    coef = clf.coef_
    offset = clf.offset_
    clf = klass(nu=0.05, eta0=0.01, max_iter=1, learning_rate=lr, shuffle=False)
    for _ in range(2):
        clf.partial_fit(X)
    y_scores2 = clf.decision_function(T)
    assert clf.t_ == t
    assert_allclose(y_scores, y_scores2)
    assert_allclose(clf.coef_, coef)
    assert_allclose(clf.offset_, offset)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_late_onset_averaging_reached_oneclass(klass):
    if False:
        while True:
            i = 10
    eta0 = 0.001
    nu = 0.05
    clf1 = klass(average=7, learning_rate='constant', eta0=eta0, nu=nu, max_iter=2, shuffle=False)
    clf2 = klass(average=0, learning_rate='constant', eta0=eta0, nu=nu, max_iter=1, shuffle=False)
    clf1.fit(X)
    clf2.fit(X)
    (average_coef, average_offset) = asgd_oneclass(klass, X, eta0, nu, coef_init=clf2.coef_.ravel(), offset_init=clf2.offset_)
    assert_allclose(clf1.coef_.ravel(), average_coef.ravel())
    assert_allclose(clf1.offset_, average_offset)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_computed_correctly_oneclass(klass):
    if False:
        print('Hello World!')
    eta = 0.001
    nu = 0.05
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    clf = klass(learning_rate='constant', eta0=eta, nu=nu, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    clf.fit(X)
    (average_coef, average_offset) = asgd_oneclass(klass, X, eta, nu)
    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_partial_fit_oneclass(klass):
    if False:
        return 10
    eta = 0.001
    nu = 0.05
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    clf = klass(learning_rate='constant', eta0=eta, nu=nu, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    clf.partial_fit(X[:int(n_samples / 2)][:])
    clf.partial_fit(X[int(n_samples / 2):][:])
    (average_coef, average_offset) = asgd_oneclass(klass, X, eta, nu)
    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)

@pytest.mark.parametrize('klass', [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_average_sparse_oneclass(klass):
    if False:
        print('Hello World!')
    eta = 0.001
    nu = 0.01
    clf = klass(learning_rate='constant', eta0=eta, nu=nu, fit_intercept=True, max_iter=1, average=True, shuffle=False)
    n_samples = X3.shape[0]
    clf.partial_fit(X3[:int(n_samples / 2)])
    clf.partial_fit(X3[int(n_samples / 2):])
    (average_coef, average_offset) = asgd_oneclass(klass, X3, eta, nu)
    assert_allclose(clf.coef_, average_coef)
    assert_allclose(clf.offset_, average_offset)

def test_sgd_oneclass():
    if False:
        i = 10
        return i + 15
    X_train = np.array([[-2, -1], [-1, -1], [1, 1]])
    X_test = np.array([[0.5, -2], [2, 2]])
    clf = SGDOneClassSVM(nu=0.5, eta0=1, learning_rate='constant', shuffle=False, max_iter=1)
    clf.fit(X_train)
    assert_allclose(clf.coef_, np.array([-0.125, 0.4375]))
    assert clf.offset_[0] == -0.5
    scores = clf.score_samples(X_test)
    assert_allclose(scores, np.array([-0.9375, 0.625]))
    dec = clf.score_samples(X_test) - clf.offset_
    assert_allclose(clf.decision_function(X_test), dec)
    pred = clf.predict(X_test)
    assert_array_equal(pred, np.array([-1, 1]))

def test_ocsvm_vs_sgdocsvm():
    if False:
        while True:
            i = 10
    nu = 0.05
    gamma = 2.0
    random_state = 42
    rng = np.random.RandomState(random_state)
    X = 0.3 * rng.randn(500, 2)
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * rng.randn(100, 2)
    X_test = np.r_[X + 2, X - 2]
    clf = OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
    clf.fit(X_train)
    y_pred_ocsvm = clf.predict(X_test)
    dec_ocsvm = clf.decision_function(X_test).reshape(1, -1)
    max_iter = 15
    transform = Nystroem(gamma=gamma, random_state=random_state)
    clf_sgd = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, max_iter=max_iter, random_state=random_state, tol=None)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    pipe_sgd.fit(X_train)
    y_pred_sgdocsvm = pipe_sgd.predict(X_test)
    dec_sgdocsvm = pipe_sgd.decision_function(X_test).reshape(1, -1)
    assert np.mean(y_pred_sgdocsvm == y_pred_ocsvm) >= 0.99
    corrcoef = np.corrcoef(np.concatenate((dec_ocsvm, dec_sgdocsvm)))[0, 1]
    assert corrcoef >= 0.9

def test_l1_ratio():
    if False:
        print('Hello World!')
    (X, y) = datasets.make_classification(n_samples=1000, n_features=100, n_informative=20, random_state=1234)
    est_en = SGDClassifier(alpha=0.001, penalty='elasticnet', tol=None, max_iter=6, l1_ratio=0.9999999999, random_state=42).fit(X, y)
    est_l1 = SGDClassifier(alpha=0.001, penalty='l1', max_iter=6, random_state=42, tol=None).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l1.coef_)
    est_en = SGDClassifier(alpha=0.001, penalty='elasticnet', tol=None, max_iter=6, l1_ratio=1e-10, random_state=42).fit(X, y)
    est_l2 = SGDClassifier(alpha=0.001, penalty='l2', max_iter=6, random_state=42, tol=None).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l2.coef_)

def test_underflow_or_overlow():
    if False:
        i = 10
        return i + 15
    with np.errstate(all='raise'):
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 10
        X = rng.normal(size=(n_samples, n_features))
        X[:, :2] *= 1e+300
        assert np.isfinite(X).all()
        X_scaled = MinMaxScaler().fit_transform(X)
        assert np.isfinite(X_scaled).all()
        ground_truth = rng.normal(size=n_features)
        y = (np.dot(X_scaled, ground_truth) > 0.0).astype(np.int32)
        assert_array_equal(np.unique(y), [0, 1])
        model = SGDClassifier(alpha=0.1, loss='squared_hinge', max_iter=500)
        model.fit(X_scaled, y)
        assert np.isfinite(model.coef_).all()
        msg_regxp = 'Floating-point under-/overflow occurred at epoch #.* Scaling input data with StandardScaler or MinMaxScaler might help.'
        with pytest.raises(ValueError, match=msg_regxp):
            model.fit(X, y)

def test_numerical_stability_large_gradient():
    if False:
        for i in range(10):
            print('nop')
    model = SGDClassifier(loss='squared_hinge', max_iter=10, shuffle=True, penalty='elasticnet', l1_ratio=0.3, alpha=0.01, eta0=0.001, random_state=0, tol=None)
    with np.errstate(all='raise'):
        model.fit(iris.data, iris.target)
    assert np.isfinite(model.coef_).all()

@pytest.mark.parametrize('penalty', ['l2', 'l1', 'elasticnet'])
def test_large_regularization(penalty):
    if False:
        i = 10
        return i + 15
    model = SGDClassifier(alpha=100000.0, learning_rate='constant', eta0=0.1, penalty=penalty, shuffle=False, tol=None, max_iter=6)
    with np.errstate(all='raise'):
        model.fit(iris.data, iris.target)
    assert_array_almost_equal(model.coef_, np.zeros_like(model.coef_))

def test_tol_parameter():
    if False:
        while True:
            i = 10
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target == 1
    max_iter = 42
    model_0 = SGDClassifier(tol=None, random_state=0, max_iter=max_iter)
    model_0.fit(X, y)
    assert max_iter == model_0.n_iter_
    max_iter = 2000
    model_1 = SGDClassifier(tol=0, random_state=0, max_iter=max_iter)
    model_1.fit(X, y)
    assert max_iter > model_1.n_iter_
    assert model_1.n_iter_ > 5
    model_2 = SGDClassifier(tol=0.1, random_state=0, max_iter=max_iter)
    model_2.fit(X, y)
    assert model_1.n_iter_ > model_2.n_iter_
    assert model_2.n_iter_ > 3
    model_3 = SGDClassifier(max_iter=3, tol=0.001, random_state=0)
    warning_message = 'Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        model_3.fit(X, y)
    assert model_3.n_iter_ == 3

def _test_loss_common(loss_function, cases):
    if False:
        while True:
            i = 10
    for (p, y, expected_loss, expected_dloss) in cases:
        assert_almost_equal(loss_function.py_loss(p, y), expected_loss)
        assert_almost_equal(loss_function.py_dloss(p, y), expected_dloss)

def test_loss_hinge():
    if False:
        return 10
    loss = sgd_fast.Hinge(1.0)
    cases = [(1.1, 1.0, 0.0, 0.0), (-2.0, -1.0, 0.0, 0.0), (1.0, 1.0, 0.0, -1.0), (-1.0, -1.0, 0.0, 1.0), (0.5, 1.0, 0.5, -1.0), (2.0, -1.0, 3.0, 1.0), (-0.5, -1.0, 0.5, 1.0), (0.0, 1.0, 1, -1.0)]
    _test_loss_common(loss, cases)
    loss = sgd_fast.Hinge(0.0)
    cases = [(1.0, 1.0, 0.0, 0.0), (-0.1, -1.0, 0.0, 0.0), (0.0, 1.0, 0.0, -1.0), (0.0, -1.0, 0.0, 1.0), (0.5, -1.0, 0.5, 1.0), (2.0, -1.0, 2.0, 1.0), (-0.5, 1.0, 0.5, -1.0), (-1.0, 1.0, 1.0, -1.0)]
    _test_loss_common(loss, cases)

def test_gradient_squared_hinge():
    if False:
        while True:
            i = 10
    loss = sgd_fast.SquaredHinge(1.0)
    cases = [(1.0, 1.0, 0.0, 0.0), (-2.0, -1.0, 0.0, 0.0), (1.0, -1.0, 4.0, 4.0), (-1.0, 1.0, 4.0, -4.0), (0.5, 1.0, 0.25, -1.0), (0.5, -1.0, 2.25, 3.0)]
    _test_loss_common(loss, cases)

def test_loss_log():
    if False:
        i = 10
        return i + 15
    loss = sgd_fast.Log()
    cases = [(1.0, 1.0, np.log(1.0 + np.exp(-1.0)), -1.0 / (np.exp(1.0) + 1.0)), (1.0, -1.0, np.log(1.0 + np.exp(1.0)), 1.0 / (np.exp(-1.0) + 1.0)), (-1.0, -1.0, np.log(1.0 + np.exp(-1.0)), 1.0 / (np.exp(1.0) + 1.0)), (-1.0, 1.0, np.log(1.0 + np.exp(1.0)), -1.0 / (np.exp(-1.0) + 1.0)), (0.0, 1.0, np.log(2), -0.5), (0.0, -1.0, np.log(2), 0.5), (17.9, -1.0, 17.9, 1.0), (-17.9, 1.0, 17.9, -1.0)]
    _test_loss_common(loss, cases)
    assert_almost_equal(loss.py_dloss(18.1, 1.0), np.exp(-18.1) * -1.0, 16)
    assert_almost_equal(loss.py_loss(18.1, 1.0), np.exp(-18.1), 16)
    assert_almost_equal(loss.py_dloss(-18.1, -1.0), np.exp(-18.1) * 1.0, 16)
    assert_almost_equal(loss.py_loss(-18.1, 1.0), 18.1, 16)

def test_loss_squared_loss():
    if False:
        print('Hello World!')
    loss = sgd_fast.SquaredLoss()
    cases = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.5, 1.0), (0.5, -1.0, 1.125, 1.5), (-2.5, 2.0, 10.125, -4.5)]
    _test_loss_common(loss, cases)

def test_loss_huber():
    if False:
        return 10
    loss = sgd_fast.Huber(0.1)
    cases = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.005, 0.1), (0.0, 0.1, 0.005, -0.1), (3.95, 4.0, 0.00125, -0.05), (5.0, 2.0, 0.295, 0.1), (-1.0, 5.0, 0.595, -0.1)]
    _test_loss_common(loss, cases)

def test_loss_modified_huber():
    if False:
        print('Hello World!')
    loss = sgd_fast.ModifiedHuber()
    cases = [(1.0, 1.0, 0.0, 0.0), (-1.0, -1.0, 0.0, 0.0), (2.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, -2.0), (-1.0, 1.0, 4.0, -4.0), (0.5, -1.0, 2.25, 3.0), (-2.0, 1.0, 8, -4.0), (-3.0, 1.0, 12, -4.0)]
    _test_loss_common(loss, cases)

def test_loss_epsilon_insensitive():
    if False:
        for i in range(10):
            print('nop')
    loss = sgd_fast.EpsilonInsensitive(0.1)
    cases = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0), (-2.05, -2.0, 0.0, 0.0), (3.05, 3.0, 0.0, 0.0), (2.2, 2.0, 0.1, 1.0), (2.0, -1.0, 2.9, 1.0), (2.0, 2.2, 0.1, -1.0), (-2.0, 1.0, 2.9, -1.0)]
    _test_loss_common(loss, cases)

def test_loss_squared_epsilon_insensitive():
    if False:
        for i in range(10):
            print('nop')
    loss = sgd_fast.SquaredEpsilonInsensitive(0.1)
    cases = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0), (-2.05, -2.0, 0.0, 0.0), (3.05, 3.0, 0.0, 0.0), (2.2, 2.0, 0.01, 0.2), (2.0, -1.0, 8.41, 5.8), (2.0, 2.2, 0.01, -0.2), (-2.0, 1.0, 8.41, -5.8)]
    _test_loss_common(loss, cases)

def test_multi_thread_multi_class_and_early_stopping():
    if False:
        while True:
            i = 10
    clf = SGDClassifier(alpha=0.001, tol=0.001, max_iter=1000, early_stopping=True, n_iter_no_change=100, random_state=0, n_jobs=2)
    clf.fit(iris.data, iris.target)
    assert clf.n_iter_ > clf.n_iter_no_change
    assert clf.n_iter_ < clf.n_iter_no_change + 20
    assert clf.score(iris.data, iris.target) > 0.8

def test_multi_core_gridsearch_and_early_stopping():
    if False:
        for i in range(10):
            print('nop')
    param_grid = {'alpha': np.logspace(-4, 4, 9), 'n_iter_no_change': [5, 10, 50]}
    clf = SGDClassifier(tol=0.01, max_iter=1000, early_stopping=True, random_state=0)
    search = RandomizedSearchCV(clf, param_grid, n_iter=5, n_jobs=2, random_state=0)
    search.fit(iris.data, iris.target)
    assert search.best_score_ > 0.8

@pytest.mark.parametrize('backend', ['loky', 'multiprocessing', 'threading'])
def test_SGDClassifier_fit_for_all_backends(backend):
    if False:
        while True:
            i = 10
    random_state = np.random.RandomState(42)
    X = sp.random(500, 2000, density=0.02, format='csr', random_state=random_state)
    y = random_state.choice(20, 500)
    clf_sequential = SGDClassifier(max_iter=1000, n_jobs=1, random_state=42)
    clf_sequential.fit(X, y)
    clf_parallel = SGDClassifier(max_iter=1000, n_jobs=4, random_state=42)
    with joblib.parallel_backend(backend=backend):
        clf_parallel.fit(X, y)
    assert_array_almost_equal(clf_sequential.coef_, clf_parallel.coef_)

@pytest.mark.parametrize('Estimator', [linear_model.SGDClassifier, linear_model.SGDRegressor])
def test_sgd_random_state(Estimator, global_random_seed):
    if False:
        return 10
    if Estimator == linear_model.SGDRegressor:
        (X, y) = datasets.make_regression(random_state=global_random_seed)
    else:
        (X, y) = datasets.make_classification(random_state=global_random_seed)
    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_a = est.fit(X, y).coef_
        assert est.n_iter_ == 1
    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_b = est.fit(X, y).coef_
        assert est.n_iter_ == 1
    assert_allclose(coef_same_seed_a, coef_same_seed_b)
    est = Estimator(random_state=global_random_seed + 1, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_other_seed = est.fit(X, y).coef_
        assert est.n_iter_ == 1
    assert np.abs(coef_same_seed_a - coef_other_seed).max() > 1.0

def test_validation_mask_correctly_subsets(monkeypatch):
    if False:
        return 10
    'Test that data passed to validation callback correctly subsets.\n\n    Non-regression test for #23255.\n    '
    (X, Y) = (iris.data, iris.target)
    n_samples = X.shape[0]
    validation_fraction = 0.2
    clf = linear_model.SGDClassifier(early_stopping=True, tol=0.001, max_iter=1000, validation_fraction=validation_fraction)
    mock = Mock(side_effect=_stochastic_gradient._ValidationScoreCallback)
    monkeypatch.setattr(_stochastic_gradient, '_ValidationScoreCallback', mock)
    clf.fit(X, Y)
    (X_val, y_val) = mock.call_args[0][1:3]
    assert X_val.shape[0] == int(n_samples * validation_fraction)
    assert y_val.shape[0] == int(n_samples * validation_fraction)

def test_sgd_error_on_zero_validation_weight():
    if False:
        return 10
    (X, Y) = (iris.data, iris.target)
    sample_weight = np.zeros_like(Y)
    validation_fraction = 0.4
    clf = linear_model.SGDClassifier(early_stopping=True, validation_fraction=validation_fraction, random_state=0)
    error_message = 'The sample weights for validation set are all zero, consider using a different random state.'
    with pytest.raises(ValueError, match=error_message):
        clf.fit(X, Y, sample_weight=sample_weight)

@pytest.mark.parametrize('Estimator', [SGDClassifier, SGDRegressor])
def test_sgd_verbose(Estimator):
    if False:
        print('Hello World!')
    'non-regression test for gh #25249'
    Estimator(verbose=1).fit(X, Y)

@pytest.mark.parametrize('SGDEstimator', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize('data_type', (np.float32, np.float64))
def test_sgd_dtype_match(SGDEstimator, data_type):
    if False:
        return 10
    _X = X.astype(data_type)
    _Y = np.array(Y, dtype=data_type)
    sgd_model = SGDEstimator()
    sgd_model.fit(_X, _Y)
    assert sgd_model.coef_.dtype == data_type

@pytest.mark.parametrize('SGDEstimator', [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor, SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_numerical_consistency(SGDEstimator):
    if False:
        for i in range(10):
            print('nop')
    X_64 = X.astype(dtype=np.float64)
    Y_64 = np.array(Y, dtype=np.float64)
    X_32 = X.astype(dtype=np.float32)
    Y_32 = np.array(Y, dtype=np.float32)
    sgd_64 = SGDEstimator(max_iter=20)
    sgd_64.fit(X_64, Y_64)
    sgd_32 = SGDEstimator(max_iter=20)
    sgd_32.fit(X_32, Y_32)
    assert_allclose(sgd_64.coef_, sgd_32.coef_)