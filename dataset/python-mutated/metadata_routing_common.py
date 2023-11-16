from functools import partial
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, RegressorMixin, TransformerMixin, clone
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import SIMPLE_METHODS
from sklearn.utils.metadata_routing import MetadataRouter, process_routing
from sklearn.utils.multiclass import _check_partial_fit_first_call

def record_metadata(obj, method, record_default=True, **kwargs):
    if False:
        while True:
            i = 10
    'Utility function to store passed metadata to a method.\n\n    If record_default is False, kwargs whose values are "default" are skipped.\n    This is so that checks on keyword arguments whose default was not changed\n    are skipped.\n\n    '
    if not hasattr(obj, '_records'):
        obj._records = {}
    if not record_default:
        kwargs = {key: val for (key, val) in kwargs.items() if not isinstance(val, str) or val != 'default'}
    obj._records[method] = kwargs

def check_recorded_metadata(obj, method, split_params=tuple(), **kwargs):
    if False:
        print('Hello World!')
    "Check whether the expected metadata is passed to the object's method.\n\n    Parameters\n    ----------\n    obj : estimator object\n        sub-estimator to check routed params for\n    method : str\n        sub-estimator's method where metadata is routed to\n    split_params : tuple, default=empty\n        specifies any parameters which are to be checked as being a subset\n        of the original values.\n    "
    records = getattr(obj, '_records', dict()).get(method, dict())
    assert set(kwargs.keys()) == set(records.keys())
    for (key, value) in kwargs.items():
        recorded_value = records[key]
        if key in split_params and recorded_value is not None:
            assert np.isin(recorded_value, value).all()
        else:
            assert recorded_value is value
record_metadata_not_default = partial(record_metadata, record_default=False)

def assert_request_is_empty(metadata_request, exclude=None):
    if False:
        return 10
    'Check if a metadata request dict is empty.\n\n    One can exclude a method or a list of methods from the check using the\n    ``exclude`` parameter. If metadata_request is a MetadataRouter, then\n    ``exclude`` can be of the form ``{"object" : [method, ...]}``.\n    '
    if isinstance(metadata_request, MetadataRouter):
        for (name, route_mapping) in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return
    exclude = [] if exclude is None else exclude
    for method in SIMPLE_METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        props = [prop for (prop, alias) in mmr.requests.items() if isinstance(alias, str) or alias is not None]
        assert not props

def assert_request_equal(request, dictionary):
    if False:
        while True:
            i = 10
    for (method, requests) in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests
    empty_methods = [method for method in SIMPLE_METHODS if method not in dictionary]
    for method in empty_methods:
        assert not len(getattr(request, method).requests)

class _Registry(list):

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        return self

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        return self

class ConsumingRegressor(RegressorMixin, BaseEstimator):
    """A regressor consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        if False:
            return 10
        self.registry = registry

    def partial_fit(self, X, y, sample_weight='default', metadata='default'):
        if False:
            while True:
                i = 10
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'partial_fit', sample_weight=sample_weight, metadata=metadata)
        return self

    def fit(self, X, y, sample_weight='default', metadata='default'):
        if False:
            i = 10
            return i + 15
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'fit', sample_weight=sample_weight, metadata=metadata)
        return self

    def predict(self, X, sample_weight='default', metadata='default'):
        if False:
            i = 10
            return i + 15
        pass

class NonConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier which accepts no metadata on any method."""

    def __init__(self, registry=None):
        if False:
            i = 10
            return i + 15
        self.registry = registry

    def fit(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        if self.registry is not None:
            self.registry.append(self)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if False:
            return 10
        return np.ones(len(X))

class ConsumingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier consuming metadata.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.

    alpha : float, default=0
        This parameter is only used to test the ``*SearchCV`` objects, and
        doesn't do anything.
    """

    def __init__(self, registry=None, alpha=0.0):
        if False:
            i = 10
            return i + 15
        self.alpha = alpha
        self.registry = registry

    def partial_fit(self, X, y, classes=None, sample_weight='default', metadata='default'):
        if False:
            i = 10
            return i + 15
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'partial_fit', sample_weight=sample_weight, metadata=metadata)
        _check_partial_fit_first_call(self, classes)
        return self

    def fit(self, X, y, sample_weight='default', metadata='default'):
        if False:
            print('Hello World!')
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'fit', sample_weight=sample_weight, metadata=metadata)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, sample_weight='default', metadata='default'):
        if False:
            print('Hello World!')
        record_metadata_not_default(self, 'predict', sample_weight=sample_weight, metadata=metadata)
        return np.zeros(shape=(len(X),))

    def predict_proba(self, X, sample_weight='default', metadata='default'):
        if False:
            print('Hello World!')
        pass

    def predict_log_proba(self, X, sample_weight='default', metadata='default'):
        if False:
            i = 10
            return i + 15
        pass

    def decision_function(self, X, sample_weight='default', metadata='default'):
        if False:
            i = 10
            return i + 15
        record_metadata_not_default(self, 'predict_proba', sample_weight=sample_weight, metadata=metadata)
        return np.zeros(shape=(len(X),))

class ConsumingTransformer(TransformerMixin, BaseEstimator):
    """A transformer which accepts metadata on fit and transform.

    Parameters
    ----------
    registry : list, default=None
        If a list, the estimator will append itself to the list in order to have
        a reference to the estimator later on. Since that reference is not
        required in all tests, registration can be skipped by leaving this value
        as None.
    """

    def __init__(self, registry=None):
        if False:
            print('Hello World!')
        self.registry = registry

    def fit(self, X, y=None, sample_weight=None, metadata=None):
        if False:
            while True:
                i = 10
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'fit', sample_weight=sample_weight, metadata=metadata)
        return self

    def transform(self, X, sample_weight=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        record_metadata(self, 'transform', sample_weight=sample_weight, metadata=metadata)
        return X

    def fit_transform(self, X, y, sample_weight=None, metadata=None):
        if False:
            return 10
        record_metadata(self, 'fit_transform', sample_weight=sample_weight, metadata=metadata)
        return self.fit(X, y, sample_weight=sample_weight, metadata=metadata).transform(X, sample_weight=sample_weight, metadata=metadata)

    def inverse_transform(self, X, sample_weight=None, metadata=None):
        if False:
            while True:
                i = 10
        record_metadata(self, 'inverse_transform', sample_weight=sample_weight, metadata=metadata)
        return X

class ConsumingScorer(_Scorer):

    def __init__(self, registry=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(score_func=mean_squared_error, sign=1, kwargs={}, response_method='predict')
        self.registry = registry

    def _score(self, method_caller, clf, X, y, **kwargs):
        if False:
            print('Hello World!')
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'score', **kwargs)
        sample_weight = kwargs.get('sample_weight', None)
        return super()._score(method_caller, clf, X, y, sample_weight=sample_weight)

class ConsumingSplitter(BaseCrossValidator, GroupsConsumerMixin):

    def __init__(self, registry=None):
        if False:
            return 10
        self.registry = registry

    def split(self, X, y=None, groups='default', metadata='default'):
        if False:
            i = 10
            return i + 15
        if self.registry is not None:
            self.registry.append(self)
        record_metadata_not_default(self, 'split', groups=groups, metadata=metadata)
        split_index = len(X) // 2
        train_indices = list(range(0, split_index))
        test_indices = list(range(split_index, len(X)))
        yield (test_indices, train_indices)
        yield (train_indices, test_indices)

    def get_n_splits(self, X=None, y=None, groups=None, metadata=None):
        if False:
            while True:
                i = 10
        return 2

    def _iter_test_indices(self, X=None, y=None, groups=None):
        if False:
            return 10
        split_index = len(X) // 2
        train_indices = list(range(0, split_index))
        test_indices = list(range(split_index, len(X)))
        yield test_indices
        yield train_indices

class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is only a router."""

    def __init__(self, estimator):
        if False:
            i = 10
            return i + 15
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        if False:
            while True:
                i = 10
        params = process_routing(self, 'fit', **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    def get_metadata_routing(self):
        if False:
            i = 10
            return i + 15
        router = MetadataRouter(owner=self.__class__.__name__).add(estimator=self.estimator, method_mapping='one-to-one')
        return router

class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """A meta-regressor which is also a consumer."""

    def __init__(self, estimator, registry=None):
        if False:
            while True:
                i = 10
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **fit_params):
        if False:
            print('Hello World!')
        if self.registry is not None:
            self.registry.append(self)
        record_metadata(self, 'fit', sample_weight=sample_weight)
        params = process_routing(self, 'fit', sample_weight=sample_weight, **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def predict(self, X, **predict_params):
        if False:
            for i in range(10):
                print('nop')
        params = process_routing(self, 'predict', **predict_params)
        return self.estimator_.predict(X, **params.estimator.predict)

    def get_metadata_routing(self):
        if False:
            return 10
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(estimator=self.estimator, method_mapping='one-to-one')
        return router

class WeightedMetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """A meta-estimator which also consumes sample_weight itself in ``fit``."""

    def __init__(self, estimator, registry=None):
        if False:
            i = 10
            return i + 15
        self.estimator = estimator
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **kwargs):
        if False:
            while True:
                i = 10
        if self.registry is not None:
            self.registry.append(self)
        record_metadata(self, 'fit', sample_weight=sample_weight)
        params = process_routing(self, 'fit', sample_weight=sample_weight, **kwargs)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        return self

    def get_metadata_routing(self):
        if False:
            print('Hello World!')
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(estimator=self.estimator, method_mapping='fit')
        return router

class MetaTransformer(MetaEstimatorMixin, TransformerMixin, BaseEstimator):
    """A simple meta-transformer."""

    def __init__(self, transformer):
        if False:
            return 10
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        if False:
            return 10
        params = process_routing(self, 'fit', **fit_params)
        self.transformer_ = clone(self.transformer).fit(X, y, **params.transformer.fit)
        return self

    def transform(self, X, y=None, **transform_params):
        if False:
            print('Hello World!')
        params = process_routing(self, 'transform', **transform_params)
        return self.transformer_.transform(X, **params.transformer.transform)

    def get_metadata_routing(self):
        if False:
            while True:
                i = 10
        return MetadataRouter(owner=self.__class__.__name__).add(transformer=self.transformer, method_mapping='one-to-one')