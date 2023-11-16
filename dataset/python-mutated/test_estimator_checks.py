import importlib
import sys
import unittest
import warnings
from numbers import Integral, Real
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskElasticNet, SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._testing import MinimalClassifier, MinimalRegressor, MinimalTransformer, SkipTest, ignore_warnings, raises
from sklearn.utils.estimator_checks import _NotAnArray, _set_checking_parameters, _yield_all_checks, check_array_api_input, check_class_weight_balanced_linear_classifier, check_classifier_data_not_an_array, check_classifiers_multilabel_output_format_decision_function, check_classifiers_multilabel_output_format_predict, check_classifiers_multilabel_output_format_predict_proba, check_dataframe_column_names_consistency, check_decision_proba_consistency, check_estimator, check_estimator_get_tags_default_keys, check_estimators_unfitted, check_fit_check_is_fitted, check_fit_score_takes_y, check_methods_sample_order_invariance, check_methods_subset_invariance, check_no_attributes_set_in_init, check_outlier_contamination, check_outlier_corruption, check_regressor_data_not_an_array, check_requires_y_none, set_random_state
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class CorrectNotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting.

    Like NotFittedError, it inherits from ValueError, but not from
    AttributeError. Used for testing only.
    """

class BaseBadClassifier(ClassifierMixin, BaseEstimator):

    def fit(self, X, y):
        if False:
            while True:
                i = 10
        return self

    def predict(self, X):
        if False:
            return 10
        return np.ones(X.shape[0])

class ChangesDict(BaseEstimator):

    def __init__(self, key=0):
        if False:
            for i in range(10):
                print('nop')
        self.key = key

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        (X, y) = self._validate_data(X, y)
        return self

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        X = check_array(X)
        self.key = 1000
        return np.ones(X.shape[0])

class SetsWrongAttribute(BaseEstimator):

    def __init__(self, acceptable_key=0):
        if False:
            while True:
                i = 10
        self.acceptable_key = acceptable_key

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        self.wrong_attribute = 0
        (X, y) = self._validate_data(X, y)
        return self

class ChangesWrongAttribute(BaseEstimator):

    def __init__(self, wrong_attribute=0):
        if False:
            for i in range(10):
                print('nop')
        self.wrong_attribute = wrong_attribute

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        self.wrong_attribute = 1
        (X, y) = self._validate_data(X, y)
        return self

class ChangesUnderscoreAttribute(BaseEstimator):

    def fit(self, X, y=None):
        if False:
            return 10
        self._good_attribute = 1
        (X, y) = self._validate_data(X, y)
        return self

class RaisesErrorInSetParams(BaseEstimator):

    def __init__(self, p=0):
        if False:
            i = 10
            return i + 15
        self.p = p

    def set_params(self, **kwargs):
        if False:
            while True:
                i = 10
        if 'p' in kwargs:
            p = kwargs.pop('p')
            if p < 0:
                raise ValueError("p can't be less than 0")
            self.p = p
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        (X, y) = self._validate_data(X, y)
        return self

class HasMutableParameters(BaseEstimator):

    def __init__(self, p=object()):
        if False:
            print('Hello World!')
        self.p = p

    def fit(self, X, y=None):
        if False:
            return 10
        (X, y) = self._validate_data(X, y)
        return self

class HasImmutableParameters(BaseEstimator):

    def __init__(self, p=42, q=np.int32(42), r=object):
        if False:
            for i in range(10):
                print('nop')
        self.p = p
        self.q = q
        self.r = r

    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        (X, y) = self._validate_data(X, y)
        return self

class ModifiesValueInsteadOfRaisingError(BaseEstimator):

    def __init__(self, p=0):
        if False:
            i = 10
            return i + 15
        self.p = p

    def set_params(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'p' in kwargs:
            p = kwargs.pop('p')
            if p < 0:
                p = 0
            self.p = p
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        (X, y) = self._validate_data(X, y)
        return self

class ModifiesAnotherValue(BaseEstimator):

    def __init__(self, a=0, b='method1'):
        if False:
            print('Hello World!')
        self.a = a
        self.b = b

    def set_params(self, **kwargs):
        if False:
            return 10
        if 'a' in kwargs:
            a = kwargs.pop('a')
            self.a = a
            if a is None:
                kwargs.pop('b')
                self.b = 'method2'
        return super().set_params(**kwargs)

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        (X, y) = self._validate_data(X, y)
        return self

class NoCheckinPredict(BaseBadClassifier):

    def fit(self, X, y):
        if False:
            print('Hello World!')
        (X, y) = self._validate_data(X, y)
        return self

class NoSparseClassifier(BaseBadClassifier):

    def fit(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        (X, y) = self._validate_data(X, y, accept_sparse=['csr', 'csc'])
        if sp.issparse(X):
            raise ValueError('Nonsensical Error')
        return self

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        X = check_array(X)
        return np.ones(X.shape[0])

class CorrectNotFittedErrorClassifier(BaseBadClassifier):

    def fit(self, X, y):
        if False:
            print('Hello World!')
        (X, y) = self._validate_data(X, y)
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        if False:
            while True:
                i = 10
        check_is_fitted(self)
        X = check_array(X)
        return np.ones(X.shape[0])

class NoSampleWeightPandasSeriesType(BaseEstimator):

    def fit(self, X, y, sample_weight=None):
        if False:
            i = 10
            return i + 15
        (X, y) = self._validate_data(X, y, accept_sparse=('csr', 'csc'), multi_output=True, y_numeric=True)
        from pandas import Series
        if isinstance(sample_weight, Series):
            raise ValueError("Estimator does not accept 'sample_weight'of type pandas.Series")
        return self

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        X = check_array(X)
        return np.ones(X.shape[0])

class BadBalancedWeightsClassifier(BaseBadClassifier):

    def __init__(self, class_weight=None):
        if False:
            i = 10
            return i + 15
        self.class_weight = class_weight

    def fit(self, X, y):
        if False:
            return 10
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import compute_class_weight
        label_encoder = LabelEncoder().fit(y)
        classes = label_encoder.classes_
        class_weight = compute_class_weight(self.class_weight, classes=classes, y=y)
        if self.class_weight == 'balanced':
            class_weight += 1.0
        self.coef_ = class_weight
        return self

class BadTransformerWithoutMixin(BaseEstimator):

    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        X = self._validate_data(X)
        return self

    def transform(self, X):
        if False:
            return 10
        X = check_array(X)
        return X

class NotInvariantPredict(BaseEstimator):

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        (X, y) = self._validate_data(X, y, accept_sparse=('csr', 'csc'), multi_output=True, y_numeric=True)
        return self

    def predict(self, X):
        if False:
            return 10
        X = check_array(X)
        if X.shape[0] > 1:
            return np.ones(X.shape[0])
        return np.zeros(X.shape[0])

class NotInvariantSampleOrder(BaseEstimator):

    def fit(self, X, y):
        if False:
            return 10
        (X, y) = self._validate_data(X, y, accept_sparse=('csr', 'csc'), multi_output=True, y_numeric=True)
        self._X = X
        return self

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        X = check_array(X)
        if np.array_equiv(np.sort(X, axis=0), np.sort(self._X, axis=0)) and (X != self._X).any():
            return np.zeros(X.shape[0])
        return X[:, 0]

class OneClassSampleErrorClassifier(BaseBadClassifier):
    """Classifier allowing to trigger different behaviors when `sample_weight` reduces
    the number of classes to 1."""

    def __init__(self, raise_when_single_class=False):
        if False:
            print('Hello World!')
        self.raise_when_single_class = raise_when_single_class

    def fit(self, X, y, sample_weight=None):
        if False:
            i = 10
            return i + 15
        (X, y) = check_X_y(X, y, accept_sparse=('csr', 'csc'), multi_output=True, y_numeric=True)
        self.has_single_class_ = False
        (self.classes_, y) = np.unique(y, return_inverse=True)
        n_classes_ = self.classes_.shape[0]
        if n_classes_ < 2 and self.raise_when_single_class:
            self.has_single_class_ = True
            raise ValueError('normal class error')
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray) and len(sample_weight) > 0:
                n_classes_ = np.count_nonzero(np.bincount(y, sample_weight))
            if n_classes_ < 2:
                self.has_single_class_ = True
                raise ValueError('Nonsensical Error')
        return self

    def predict(self, X):
        if False:
            print('Hello World!')
        check_is_fitted(self)
        X = check_array(X)
        if self.has_single_class_:
            return np.zeros(X.shape[0])
        return np.ones(X.shape[0])

class LargeSparseNotSupportedClassifier(BaseEstimator):

    def fit(self, X, y):
        if False:
            return 10
        (X, y) = self._validate_data(X, y, accept_sparse=('csr', 'csc', 'coo'), accept_large_sparse=True, multi_output=True, y_numeric=True)
        if sp.issparse(X):
            if X.getformat() == 'coo':
                if X.row.dtype == 'int64' or X.col.dtype == 'int64':
                    raise ValueError("Estimator doesn't support 64-bit indices")
            elif X.getformat() in ['csc', 'csr']:
                assert 'int64' not in (X.indices.dtype, X.indptr.dtype), "Estimator doesn't support 64-bit indices"
        return self

class SparseTransformer(BaseEstimator):

    def __init__(self, sparse_container=None):
        if False:
            return 10
        self.sparse_container = sparse_container

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        self.X_shape_ = self._validate_data(X).shape
        return self

    def fit_transform(self, X, y=None):
        if False:
            print('Hello World!')
        return self.fit(X, y).transform(X)

    def transform(self, X):
        if False:
            print('Hello World!')
        X = check_array(X)
        if X.shape[1] != self.X_shape_[1]:
            raise ValueError('Bad number of features')
        return self.sparse_container(X)

class EstimatorInconsistentForPandas(BaseEstimator):

    def fit(self, X, y):
        if False:
            return 10
        try:
            from pandas import DataFrame
            if isinstance(X, DataFrame):
                self.value_ = X.iloc[0, 0]
            else:
                X = check_array(X)
                self.value_ = X[1, 0]
            return self
        except ImportError:
            X = check_array(X)
            self.value_ = X[1, 0]
            return self

    def predict(self, X):
        if False:
            while True:
                i = 10
        X = check_array(X)
        return np.array([self.value_] * X.shape[0])

class UntaggedBinaryClassifier(SGDClassifier):

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        if False:
            print('Hello World!')
        super().fit(X, y, coef_init, intercept_init, sample_weight)
        if len(self.classes_) > 2:
            raise ValueError('Only 2 classes are supported')
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if False:
            while True:
                i = 10
        super().partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
        if len(self.classes_) > 2:
            raise ValueError('Only 2 classes are supported')
        return self

class TaggedBinaryClassifier(UntaggedBinaryClassifier):

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'binary_only': True}

class EstimatorMissingDefaultTags(BaseEstimator):

    def _get_tags(self):
        if False:
            print('Hello World!')
        tags = super()._get_tags().copy()
        del tags['allow_nan']
        return tags

class RequiresPositiveXRegressor(LinearRegression):

    def fit(self, X, y):
        if False:
            print('Hello World!')
        (X, y) = self._validate_data(X, y, multi_output=True)
        if (X < 0).any():
            raise ValueError('negative X values not supported!')
        return super().fit(X, y)

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'requires_positive_X': True}

class RequiresPositiveYRegressor(LinearRegression):

    def fit(self, X, y):
        if False:
            return 10
        (X, y) = self._validate_data(X, y, multi_output=True)
        if (y <= 0).any():
            raise ValueError('negative y values not supported!')
        return super().fit(X, y)

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'requires_positive_y': True}

class PoorScoreLogisticRegression(LogisticRegression):

    def decision_function(self, X):
        if False:
            print('Hello World!')
        return super().decision_function(X) + 1

    def _more_tags(self):
        if False:
            return 10
        return {'poor_score': True}

class PartialFitChecksName(BaseEstimator):

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        self._validate_data(X, y)
        return self

    def partial_fit(self, X, y):
        if False:
            while True:
                i = 10
        reset = not hasattr(self, '_fitted')
        self._validate_data(X, y, reset=reset)
        self._fitted = True
        return self

class BrokenArrayAPI(BaseEstimator):
    """Make different predictions when using Numpy and the Array API"""

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        return self

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        enabled = get_config()['array_api_dispatch']
        (xp, _) = _array_api.get_namespace(X)
        if enabled:
            return xp.asarray([1, 2, 3])
        else:
            return np.array([3, 2, 1])

def test_check_array_api_input():
    if False:
        while True:
            i = 10
    try:
        importlib.import_module('array_api_compat')
    except ModuleNotFoundError:
        raise SkipTest('array_api_compat is required to run this test')
    try:
        importlib.import_module('numpy.array_api')
    except ModuleNotFoundError:
        raise SkipTest('numpy.array_api is required to run this test')
    with raises(AssertionError, match='Not equal to tolerance'):
        check_array_api_input('BrokenArrayAPI', BrokenArrayAPI(), array_namespace='numpy.array_api', check_values=True)

def test_not_an_array_array_function():
    if False:
        print('Hello World!')
    not_array = _NotAnArray(np.ones(10))
    msg = "Don't want to call array_function sum!"
    with raises(TypeError, match=msg):
        np.sum(not_array)
    assert np.may_share_memory(not_array, None)

def test_check_fit_score_takes_y_works_on_deprecated_fit():
    if False:
        while True:
            i = 10

    class TestEstimatorWithDeprecatedFitMethod(BaseEstimator):

        @deprecated('Deprecated for the purpose of testing check_fit_score_takes_y')
        def fit(self, X, y):
            if False:
                for i in range(10):
                    print('nop')
            return self
    check_fit_score_takes_y('test', TestEstimatorWithDeprecatedFitMethod())

def test_check_estimator():
    if False:
        i = 10
        return i + 15
    msg = 'Passing a class was deprecated'
    with raises(TypeError, match=msg):
        check_estimator(object)
    msg = "Parameter 'p' of estimator 'HasMutableParameters' is of type object which is not allowed"
    check_estimator(HasImmutableParameters())
    with raises(AssertionError, match=msg):
        check_estimator(HasMutableParameters())
    msg = 'get_params result does not match what was passed to set_params'
    with raises(AssertionError, match=msg):
        check_estimator(ModifiesValueInsteadOfRaisingError())
    with warnings.catch_warnings(record=True) as records:
        check_estimator(RaisesErrorInSetParams())
    assert UserWarning in [rec.category for rec in records]
    with raises(AssertionError, match=msg):
        check_estimator(ModifiesAnotherValue())
    msg = "object has no attribute 'fit'"
    with raises(AttributeError, match=msg):
        check_estimator(BaseEstimator())
    msg = 'Did not raise'
    with raises(AssertionError, match=msg):
        check_estimator(BaseBadClassifier())
    try:
        from pandas import Series
        msg = "Estimator NoSampleWeightPandasSeriesType raises error if 'sample_weight' parameter is of type pandas.Series"
        with raises(ValueError, match=msg):
            check_estimator(NoSampleWeightPandasSeriesType())
    except ImportError:
        pass
    msg = "Estimator NoCheckinPredict doesn't check for NaN and inf in predict"
    with raises(AssertionError, match=msg):
        check_estimator(NoCheckinPredict())
    msg = 'Estimator changes __dict__ during predict'
    with raises(AssertionError, match=msg):
        check_estimator(ChangesDict())
    msg = 'Estimator ChangesWrongAttribute should not change or mutate  the parameter wrong_attribute from 0 to 1 during fit.'
    with raises(AssertionError, match=msg):
        check_estimator(ChangesWrongAttribute())
    check_estimator(ChangesUnderscoreAttribute())
    msg = 'Estimator adds public attribute\\(s\\) during the fit method. Estimators are only allowed to add private attributes either started with _ or ended with _ but wrong_attribute added'
    with raises(AssertionError, match=msg):
        check_estimator(SetsWrongAttribute())
    name = NotInvariantSampleOrder.__name__
    method = 'predict'
    msg = '{method} of {name} is not invariant when applied to a datasetwith different sample order.'.format(method=method, name=name)
    with raises(AssertionError, match=msg):
        check_estimator(NotInvariantSampleOrder())
    name = NotInvariantPredict.__name__
    method = 'predict'
    msg = '{method} of {name} is not invariant when applied to a subset.'.format(method=method, name=name)
    with raises(AssertionError, match=msg):
        check_estimator(NotInvariantPredict())
    name = NoSparseClassifier.__name__
    msg = "Estimator %s doesn't seem to fail gracefully on sparse data" % name
    with raises(AssertionError, match=msg):
        check_estimator(NoSparseClassifier())
    name = OneClassSampleErrorClassifier.__name__
    msg = f"{name} failed when fitted on one label after sample_weight trimming. Error message is not explicit, it should have 'class'."
    with raises(AssertionError, match=msg):
        check_estimator(OneClassSampleErrorClassifier())
    msg = "Estimator LargeSparseNotSupportedClassifier doesn't seem to support \\S{3}_64 matrix, and is not failing gracefully.*"
    with raises(AssertionError, match=msg):
        check_estimator(LargeSparseNotSupportedClassifier())
    msg = 'Only 2 classes are supported'
    with raises(ValueError, match=msg):
        check_estimator(UntaggedBinaryClassifier())
    for csr_container in CSR_CONTAINERS:
        check_estimator(SparseTransformer(sparse_container=csr_container))
    check_estimator(LogisticRegression())
    check_estimator(LogisticRegression(C=0.01))
    check_estimator(MultiTaskElasticNet())
    check_estimator(TaggedBinaryClassifier())
    check_estimator(RequiresPositiveXRegressor())
    msg = 'negative y values not supported!'
    with raises(ValueError, match=msg):
        check_estimator(RequiresPositiveYRegressor())
    check_estimator(PoorScoreLogisticRegression())

def test_check_outlier_corruption():
    if False:
        return 10
    decision = np.array([0.0, 1.0, 1.5, 2.0])
    with raises(AssertionError):
        check_outlier_corruption(1, 2, decision)
    decision = np.array([0.0, 1.0, 1.0, 2.0])
    check_outlier_corruption(1, 2, decision)

def test_check_estimator_transformer_no_mixin():
    if False:
        for i in range(10):
            print('nop')
    with raises(AttributeError, '.*fit_transform.*'):
        check_estimator(BadTransformerWithoutMixin())

def test_check_estimator_clones():
    if False:
        i = 10
        return i + 15
    from sklearn.datasets import load_iris
    iris = load_iris()
    for Estimator in [GaussianMixture, LinearRegression, SGDClassifier, PCA, ExtraTreesClassifier, MiniBatchKMeans]:
        with ignore_warnings(category=ConvergenceWarning):
            est = Estimator()
            _set_checking_parameters(est)
            set_random_state(est)
            old_hash = joblib.hash(est)
            check_estimator(est)
        assert old_hash == joblib.hash(est)
        with ignore_warnings(category=ConvergenceWarning):
            est = Estimator()
            _set_checking_parameters(est)
            set_random_state(est)
            est.fit(iris.data + 10, iris.target)
            old_hash = joblib.hash(est)
            check_estimator(est)
        assert old_hash == joblib.hash(est)

def test_check_estimators_unfitted():
    if False:
        return 10
    msg = 'Did not raise'
    with raises(AssertionError, match=msg):
        check_estimators_unfitted('estimator', NoSparseClassifier())
    check_estimators_unfitted('estimator', CorrectNotFittedErrorClassifier())

def test_check_no_attributes_set_in_init():
    if False:
        for i in range(10):
            print('nop')

    class NonConformantEstimatorPrivateSet(BaseEstimator):

        def __init__(self):
            if False:
                return 10
            self.you_should_not_set_this_ = None

    class NonConformantEstimatorNoParamSet(BaseEstimator):

        def __init__(self, you_should_set_this_=None):
            if False:
                while True:
                    i = 10
            pass

    class ConformantEstimatorClassAttribute(BaseEstimator):
        __metadata_request__fit = {'foo': True}
    msg = "Estimator estimator_name should not set any attribute apart from parameters during init. Found attributes \\['you_should_not_set_this_'\\]."
    with raises(AssertionError, match=msg):
        check_no_attributes_set_in_init('estimator_name', NonConformantEstimatorPrivateSet())
    msg = 'Estimator estimator_name should store all parameters as an attribute during init'
    with raises(AttributeError, match=msg):
        check_no_attributes_set_in_init('estimator_name', NonConformantEstimatorNoParamSet())
    check_no_attributes_set_in_init('estimator_name', ConformantEstimatorClassAttribute())
    with config_context(enable_metadata_routing=True):
        check_no_attributes_set_in_init('estimator_name', ConformantEstimatorClassAttribute().set_fit_request(foo=True))

def test_check_estimator_pairwise():
    if False:
        for i in range(10):
            print('nop')
    est = SVC(kernel='precomputed')
    check_estimator(est)
    est = KNeighborsRegressor(metric='precomputed')
    check_estimator(est)

def test_check_classifier_data_not_an_array():
    if False:
        print('Hello World!')
    with raises(AssertionError, match='Not equal to tolerance'):
        check_classifier_data_not_an_array('estimator_name', EstimatorInconsistentForPandas())

def test_check_regressor_data_not_an_array():
    if False:
        while True:
            i = 10
    with raises(AssertionError, match='Not equal to tolerance'):
        check_regressor_data_not_an_array('estimator_name', EstimatorInconsistentForPandas())

def test_check_estimator_get_tags_default_keys():
    if False:
        while True:
            i = 10
    estimator = EstimatorMissingDefaultTags()
    err_msg = "EstimatorMissingDefaultTags._get_tags\\(\\) is missing entries for the following default tags: {'allow_nan'}"
    with raises(AssertionError, match=err_msg):
        check_estimator_get_tags_default_keys(estimator.__class__.__name__, estimator)
    estimator = MinimalTransformer()
    check_estimator_get_tags_default_keys(estimator.__class__.__name__, estimator)

def test_check_dataframe_column_names_consistency():
    if False:
        print('Hello World!')
    err_msg = 'Estimator does not have a feature_names_in_'
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency('estimator_name', BaseBadClassifier())
    check_dataframe_column_names_consistency('estimator_name', PartialFitChecksName())
    lr = LogisticRegression()
    check_dataframe_column_names_consistency(lr.__class__.__name__, lr)
    lr.__doc__ = "Docstring that does not document the estimator's attributes"
    err_msg = 'Estimator LogisticRegression does not document its feature_names_in_ attribute'
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency(lr.__class__.__name__, lr)

class _BaseMultiLabelClassifierMock(ClassifierMixin, BaseEstimator):

    def __init__(self, response_output):
        if False:
            for i in range(10):
                print('nop')
        self.response_output = response_output

    def fit(self, X, y):
        if False:
            print('Hello World!')
        return self

    def _more_tags(self):
        if False:
            i = 10
            return i + 15
        return {'multilabel': True}

def test_check_classifiers_multilabel_output_format_predict():
    if False:
        print('Hello World!')
    (n_samples, test_size, n_outputs) = (100, 25, 5)
    (_, y) = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    y_test = y[-test_size:]

    class MultiLabelClassifierPredict(_BaseMultiLabelClassifierMock):

        def predict(self, X):
            if False:
                print('Hello World!')
            return self.response_output
    clf = MultiLabelClassifierPredict(response_output=y_test.tolist())
    err_msg = "MultiLabelClassifierPredict.predict is expected to output a NumPy array. Got <class 'list'> instead."
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredict(response_output=y_test[:, :-1])
    err_msg = 'MultiLabelClassifierPredict.predict outputs a NumPy array of shape \\(25, 4\\) instead of \\(25, 5\\).'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredict(response_output=y_test.astype(np.float64))
    err_msg = 'MultiLabelClassifierPredict.predict does not output the same dtype than the targets.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict(clf.__class__.__name__, clf)

def test_check_classifiers_multilabel_output_format_predict_proba():
    if False:
        for i in range(10):
            print('nop')
    (n_samples, test_size, n_outputs) = (100, 25, 5)
    (_, y) = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    y_test = y[-test_size:]

    class MultiLabelClassifierPredictProba(_BaseMultiLabelClassifierMock):

        def predict_proba(self, X):
            if False:
                for i in range(10):
                    print('nop')
            return self.response_output
    for csr_container in CSR_CONTAINERS:
        clf = MultiLabelClassifierPredictProba(response_output=csr_container(y_test))
        err_msg = f'Unknown returned type .*{csr_container.__name__}.* by MultiLabelClassifierPredictProba.predict_proba. A list or a Numpy array is expected.'
        with raises(ValueError, match=err_msg):
            check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredictProba(response_output=y_test.tolist())
    err_msg = f'When MultiLabelClassifierPredictProba.predict_proba returns a list, the list should be of length n_outputs and contain NumPy arrays. Got length of {test_size} instead of {n_outputs}.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = [np.ones_like(y_test) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a list, this list should contain NumPy arrays of shape \\(n_samples, 2\\). Got NumPy arrays of shape \\(25, 5\\) instead of \\(25, 2\\).'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = [np.ones(shape=(y_test.shape[0], 2), dtype=np.int64) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a list, it should contain NumPy arrays with floating dtype.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = [np.ones(shape=(y_test.shape[0], 2), dtype=np.float64) for _ in range(n_outputs)]
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a list, each NumPy array should contain probabilities for each class and thus each row should sum to 1'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredictProba(response_output=y_test[:, :-1])
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a NumPy array, the expected shape is \\(n_samples, n_outputs\\). Got \\(25, 4\\) instead of \\(25, 5\\).'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    response_output = np.zeros_like(y_test, dtype=np.int64)
    clf = MultiLabelClassifierPredictProba(response_output=response_output)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a NumPy array, the expected data type is floating.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierPredictProba(response_output=y_test * 2.0)
    err_msg = 'When MultiLabelClassifierPredictProba.predict_proba returns a NumPy array, this array is expected to provide probabilities of the positive class and should therefore contain values between 0 and 1.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_predict_proba(clf.__class__.__name__, clf)

def test_check_classifiers_multilabel_output_format_decision_function():
    if False:
        print('Hello World!')
    (n_samples, test_size, n_outputs) = (100, 25, 5)
    (_, y) = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    y_test = y[-test_size:]

    class MultiLabelClassifierDecisionFunction(_BaseMultiLabelClassifierMock):

        def decision_function(self, X):
            if False:
                while True:
                    i = 10
            return self.response_output
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test.tolist())
    err_msg = "MultiLabelClassifierDecisionFunction.decision_function is expected to output a NumPy array. Got <class 'list'> instead."
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test[:, :-1])
    err_msg = 'MultiLabelClassifierDecisionFunction.decision_function is expected to provide a NumPy array of shape \\(n_samples, n_outputs\\). Got \\(25, 4\\) instead of \\(25, 5\\)'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(clf.__class__.__name__, clf)
    clf = MultiLabelClassifierDecisionFunction(response_output=y_test)
    err_msg = 'MultiLabelClassifierDecisionFunction.decision_function is expected to output a floating dtype.'
    with raises(AssertionError, match=err_msg):
        check_classifiers_multilabel_output_format_decision_function(clf.__class__.__name__, clf)

def run_tests_without_pytest():
    if False:
        for i in range(10):
            print('nop')
    'Runs the tests in this file without using pytest.'
    main_module = sys.modules['__main__']
    test_functions = [getattr(main_module, name) for name in dir(main_module) if name.startswith('test_')]
    test_cases = [unittest.FunctionTestCase(fn) for fn in test_functions]
    suite = unittest.TestSuite()
    suite.addTests(test_cases)
    runner = unittest.TextTestRunner()
    runner.run(suite)

def test_check_class_weight_balanced_linear_classifier():
    if False:
        while True:
            i = 10
    msg = 'Classifier estimator_name is not computing class_weight=balanced properly'
    with raises(AssertionError, match=msg):
        check_class_weight_balanced_linear_classifier('estimator_name', BadBalancedWeightsClassifier)

def test_all_estimators_all_public():
    if False:
        while True:
            i = 10
    with warnings.catch_warnings(record=True) as record:
        estimators = all_estimators()
    assert not record
    for est in estimators:
        assert not est.__class__.__name__.startswith('_')
if __name__ == '__main__':
    run_tests_without_pytest()

def test_xfail_ignored_in_check_estimator():
    if False:
        i = 10
        return i + 15
    with warnings.catch_warnings(record=True) as records:
        check_estimator(NuSVC())
    assert SkipTestWarning in [rec.category for rec in records]

def test_minimal_class_implementation_checks():
    if False:
        print('Hello World!')
    raise SkipTest
    minimal_estimators = [MinimalTransformer(), MinimalRegressor(), MinimalClassifier()]
    for estimator in minimal_estimators:
        check_estimator(estimator)

def test_check_fit_check_is_fitted():
    if False:
        print('Hello World!')

    class Estimator(BaseEstimator):

        def __init__(self, behavior='attribute'):
            if False:
                print('Hello World!')
            self.behavior = behavior

        def fit(self, X, y, **kwargs):
            if False:
                i = 10
                return i + 15
            if self.behavior == 'attribute':
                self.is_fitted_ = True
            elif self.behavior == 'method':
                self._is_fitted = True
            return self

        @available_if(lambda self: self.behavior in {'method', 'always-true'})
        def __sklearn_is_fitted__(self):
            if False:
                i = 10
                return i + 15
            if self.behavior == 'always-true':
                return True
            return hasattr(self, '_is_fitted')
    with raises(Exception, match='passes check_is_fitted before being fit'):
        check_fit_check_is_fitted('estimator', Estimator(behavior='always-true'))
    check_fit_check_is_fitted('estimator', Estimator(behavior='method'))
    check_fit_check_is_fitted('estimator', Estimator(behavior='attribute'))

def test_check_requires_y_none():
    if False:
        print('Hello World!')

    class Estimator(BaseEstimator):

        def fit(self, X, y):
            if False:
                return 10
            (X, y) = check_X_y(X, y)
    with warnings.catch_warnings(record=True) as record:
        check_requires_y_none('estimator', Estimator())
    assert not [r.message for r in record]

def test_non_deterministic_estimator_skip_tests():
    if False:
        for i in range(10):
            print('nop')
    for est in [MinimalTransformer, MinimalRegressor, MinimalClassifier]:
        all_tests = list(_yield_all_checks(est()))
        assert check_methods_sample_order_invariance in all_tests
        assert check_methods_subset_invariance in all_tests

        class Estimator(est):

            def _more_tags(self):
                if False:
                    print('Hello World!')
                return {'non_deterministic': True}
        all_tests = list(_yield_all_checks(Estimator()))
        assert check_methods_sample_order_invariance not in all_tests
        assert check_methods_subset_invariance not in all_tests

def test_check_outlier_contamination():
    if False:
        while True:
            i = 10
    'Check the test for the contamination parameter in the outlier detectors.'

    class OutlierDetectorWithoutConstraint(OutlierMixin, BaseEstimator):
        """Outlier detector without parameter validation."""

        def __init__(self, contamination=0.1):
            if False:
                for i in range(10):
                    print('nop')
            self.contamination = contamination

        def fit(self, X, y=None, sample_weight=None):
            if False:
                i = 10
                return i + 15
            return self

        def predict(self, X, y=None):
            if False:
                for i in range(10):
                    print('nop')
            return np.ones(X.shape[0])
    detector = OutlierDetectorWithoutConstraint()
    assert check_outlier_contamination(detector.__class__.__name__, detector) is None

    class OutlierDetectorWithConstraint(OutlierDetectorWithoutConstraint):
        _parameter_constraints = {'contamination': [StrOptions({'auto'})]}
    detector = OutlierDetectorWithConstraint()
    err_msg = 'contamination constraints should contain a Real Interval constraint.'
    with raises(AssertionError, match=err_msg):
        check_outlier_contamination(detector.__class__.__name__, detector)
    OutlierDetectorWithConstraint._parameter_constraints['contamination'] = [Interval(Real, 0, 0.5, closed='right')]
    detector = OutlierDetectorWithConstraint()
    check_outlier_contamination(detector.__class__.__name__, detector)
    incorrect_intervals = [Interval(Integral, 0, 1, closed='right'), Interval(Real, -1, 1, closed='right'), Interval(Real, 0, 2, closed='right'), Interval(Real, 0, 0.5, closed='left')]
    err_msg = 'contamination constraint should be an interval in \\(0, 0.5\\]'
    for interval in incorrect_intervals:
        OutlierDetectorWithConstraint._parameter_constraints['contamination'] = [interval]
        detector = OutlierDetectorWithConstraint()
        with raises(AssertionError, match=err_msg):
            check_outlier_contamination(detector.__class__.__name__, detector)

def test_decision_proba_tie_ranking():
    if False:
        return 10
    'Check that in case with some probabilities ties, we relax the\n    ranking comparison with the decision function.\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/24025\n    '
    estimator = SGDClassifier(loss='log_loss')
    check_decision_proba_consistency('SGDClassifier', estimator)