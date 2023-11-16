import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils._metadata_requests import RequestMethod
from .metaestimators import available_if
from .validation import _check_sample_weight, _num_samples, check_array, check_is_fitted

class ArraySlicingWrapper:
    """
    Parameters
    ----------
    array
    """

    def __init__(self, array):
        if False:
            for i in range(10):
                print('nop')
        self.array = array

    def __getitem__(self, aslice):
        if False:
            print('Hello World!')
        return MockDataFrame(self.array[aslice])

class MockDataFrame:
    """
    Parameters
    ----------
    array
    """

    def __init__(self, array):
        if False:
            i = 10
            return i + 15
        self.array = array
        self.values = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.iloc = ArraySlicingWrapper(array)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.array)

    def __array__(self, dtype=None):
        if False:
            return 10
        return self.array

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return MockDataFrame(self.array == other.array)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

    def take(self, indices, axis=0):
        if False:
            print('Hello World!')
        return MockDataFrame(self.array.take(indices, axis=axis))

class CheckingClassifier(ClassifierMixin, BaseEstimator):
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of `X` and `y`in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.

    Can also be used to check if `fit_params` are passed correctly, and
    to force a certain score to be returned.

    Parameters
    ----------
    check_y, check_X : callable, default=None
        The callable used to validate `X` and `y`. These callable should return
        a bool where `False` will trigger an `AssertionError`. If `None`, the
        data is not validated. Default is `None`.

    check_y_params, check_X_params : dict, default=None
        The optional parameters to pass to `check_X` and `check_y`. If `None`,
        then no parameters are passed in.

    methods_to_check : "all" or list of str, default="all"
        The methods in which the checks should be applied. By default,
        all checks will be done on all methods (`fit`, `predict`,
        `predict_proba`, `decision_function` and `score`).

    foo_param : int, default=0
        A `foo` param. When `foo > 1`, the output of :meth:`score` will be 1
        otherwise it is 0.

    expected_sample_weight : bool, default=False
        Whether to check if a valid `sample_weight` was passed to `fit`.

    expected_fit_params : list of str, default=None
        A list of the expected parameters given when calling `fit`.

    Attributes
    ----------
    classes_ : int
        The classes seen during `fit`.

    n_features_in_ : int
        The number of features seen during `fit`.

    Examples
    --------
    >>> from sklearn.utils._mocking import CheckingClassifier

    This helper allow to assert to specificities regarding `X` or `y`. In this
    case we expect `check_X` or `check_y` to return a boolean.

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = CheckingClassifier(check_X=lambda x: x.shape == (150, 4))
    >>> clf.fit(X, y)
    CheckingClassifier(...)

    We can also provide a check which might raise an error. In this case, we
    expect `check_X` to return `X` and `check_y` to return `y`.

    >>> from sklearn.utils import check_array
    >>> clf = CheckingClassifier(check_X=check_array)
    >>> clf.fit(X, y)
    CheckingClassifier(...)
    """

    def __init__(self, *, check_y=None, check_y_params=None, check_X=None, check_X_params=None, methods_to_check='all', foo_param=0, expected_sample_weight=None, expected_fit_params=None):
        if False:
            i = 10
            return i + 15
        self.check_y = check_y
        self.check_y_params = check_y_params
        self.check_X = check_X
        self.check_X_params = check_X_params
        self.methods_to_check = methods_to_check
        self.foo_param = foo_param
        self.expected_sample_weight = expected_sample_weight
        self.expected_fit_params = expected_fit_params

    def _check_X_y(self, X, y=None, should_be_fitted=True):
        if False:
            return 10
        'Validate X and y and make extra check.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data set.\n            `X` is checked only if `check_X` is not `None` (default is None).\n        y : array-like of shape (n_samples), default=None\n            The corresponding target, by default `None`.\n            `y` is checked only if `check_y` is not `None` (default is None).\n        should_be_fitted : bool, default=True\n            Whether or not the classifier should be already fitted.\n            By default True.\n\n        Returns\n        -------\n        X, y\n        '
        if should_be_fitted:
            check_is_fitted(self)
        if self.check_X is not None:
            params = {} if self.check_X_params is None else self.check_X_params
            checked_X = self.check_X(X, **params)
            if isinstance(checked_X, (bool, np.bool_)):
                assert checked_X
            else:
                X = checked_X
        if y is not None and self.check_y is not None:
            params = {} if self.check_y_params is None else self.check_y_params
            checked_y = self.check_y(y, **params)
            if isinstance(checked_y, (bool, np.bool_)):
                assert checked_y
            else:
                y = checked_y
        return (X, y)

    def fit(self, X, y, sample_weight=None, **fit_params):
        if False:
            i = 10
            return i + 15
        'Fit classifier.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Training vector, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples, n_outputs) or (n_samples,),                 default=None\n            Target relative to X for classification or regression;\n            None for unsupervised learning.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights. If None, then samples are equally weighted.\n\n        **fit_params : dict of string -> object\n            Parameters passed to the ``fit`` method of the estimator\n\n        Returns\n        -------\n        self\n        '
        assert _num_samples(X) == _num_samples(y)
        if self.methods_to_check == 'all' or 'fit' in self.methods_to_check:
            (X, y) = self._check_X_y(X, y, should_be_fitted=False)
        self.n_features_in_ = np.shape(X)[1]
        self.classes_ = np.unique(check_array(y, ensure_2d=False, allow_nd=True))
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            if missing:
                raise AssertionError(f'Expected fit parameter(s) {list(missing)} not seen.')
            for (key, value) in fit_params.items():
                if _num_samples(value) != _num_samples(X):
                    raise AssertionError(f'Fit parameter {key} has length {_num_samples(value)}; expected {_num_samples(X)}.')
        if self.expected_sample_weight:
            if sample_weight is None:
                raise AssertionError('Expected sample_weight to be passed')
            _check_sample_weight(sample_weight, X)
        return self

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict the first class seen in `classes_`.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input data.\n\n        Returns\n        -------\n        preds : ndarray of shape (n_samples,)\n            Predictions of the first class seens in `classes_`.\n        '
        if self.methods_to_check == 'all' or 'predict' in self.methods_to_check:
            (X, y) = self._check_X_y(X)
        return self.classes_[np.zeros(_num_samples(X), dtype=int)]

    def predict_proba(self, X):
        if False:
            i = 10
            return i + 15
        'Predict probabilities for each class.\n\n        Here, the dummy classifier will provide a probability of 1 for the\n        first class of `classes_` and 0 otherwise.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input data.\n\n        Returns\n        -------\n        proba : ndarray of shape (n_samples, n_classes)\n            The probabilities for each sample and class.\n        '
        if self.methods_to_check == 'all' or 'predict_proba' in self.methods_to_check:
            (X, y) = self._check_X_y(X)
        proba = np.zeros((_num_samples(X), len(self.classes_)))
        proba[:, 0] = 1
        return proba

    def decision_function(self, X):
        if False:
            print('Hello World!')
        'Confidence score.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The input data.\n\n        Returns\n        -------\n        decision : ndarray of shape (n_samples,) if n_classes == 2                else (n_samples, n_classes)\n            Confidence score.\n        '
        if self.methods_to_check == 'all' or 'decision_function' in self.methods_to_check:
            (X, y) = self._check_X_y(X)
        if len(self.classes_) == 2:
            return np.zeros(_num_samples(X))
        else:
            decision = np.zeros((_num_samples(X), len(self.classes_)))
            decision[:, 0] = 1
            return decision

    def score(self, X=None, Y=None):
        if False:
            i = 10
            return i + 15
        'Fake score.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Input data, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        Y : array-like of shape (n_samples, n_output) or (n_samples,)\n            Target relative to X for classification or regression;\n            None for unsupervised learning.\n\n        Returns\n        -------\n        score : float\n            Either 0 or 1 depending of `foo_param` (i.e. `foo_param > 1 =>\n            score=1` otherwise `score=0`).\n        '
        if self.methods_to_check == 'all' or 'score' in self.methods_to_check:
            self._check_X_y(X, Y)
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def _more_tags(self):
        if False:
            i = 10
            return i + 15
        return {'_skip_test': True, 'X_types': ['1dlabel']}
CheckingClassifier.set_fit_request = RequestMethod(name='fit', keys=[], validate_keys=False)

class NoSampleWeightWrapper(BaseEstimator):
    """Wrap estimator which will not expose `sample_weight`.

    Parameters
    ----------
    est : estimator, default=None
        The estimator to wrap.
    """

    def __init__(self, est=None):
        if False:
            while True:
                i = 10
        self.est = est

    def fit(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        return self.est.fit(X, y)

    def predict(self, X):
        if False:
            while True:
                i = 10
        return self.est.predict(X)

    def predict_proba(self, X):
        if False:
            while True:
                i = 10
        return self.est.predict_proba(X)

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'_skip_test': True}

def _check_response(method):
    if False:
        for i in range(10):
            print('nop')

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        return self.response_methods is not None and method in self.response_methods
    return check

class _MockEstimatorOnOffPrediction(BaseEstimator):
    """Estimator for which we can turn on/off the prediction methods.

    Parameters
    ----------
    response_methods: list of             {"predict", "predict_proba", "decision_function"}, default=None
        List containing the response implemented by the estimator. When, the
        response is in the list, it will return the name of the response method
        when called. Otherwise, an `AttributeError` is raised. It allows to
        use `getattr` as any conventional estimator. By default, no response
        methods are mocked.
    """

    def __init__(self, response_methods=None):
        if False:
            for i in range(10):
                print('nop')
        self.response_methods = response_methods

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        self.classes_ = np.unique(y)
        return self

    @available_if(_check_response('predict'))
    def predict(self, X):
        if False:
            return 10
        return 'predict'

    @available_if(_check_response('predict_proba'))
    def predict_proba(self, X):
        if False:
            print('Hello World!')
        return 'predict_proba'

    @available_if(_check_response('decision_function'))
    def decision_function(self, X):
        if False:
            for i in range(10):
                print('nop')
        return 'decision_function'