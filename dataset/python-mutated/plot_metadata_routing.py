"""
================
Metadata Routing
================

.. currentmodule:: sklearn

This document shows how you can use the :ref:`metadata routing mechanism
<metadata_routing>` in scikit-learn to route metadata through meta-estimators
to the estimators consuming them. To better understand the rest of the
document, we need to introduce two concepts: routers and consumers. A router is
an object, in most cases a meta-estimator, which forwards given data and
metadata to other objects and estimators. A consumer, on the other hand, is an
object which accepts and uses a certain given metadata. For instance, an
estimator taking into account ``sample_weight`` in its :term:`fit` method is a
consumer of ``sample_weight``. It is possible for an object to be both a router
and a consumer. For instance, a meta-estimator may take into account
``sample_weight`` in certain calculations, but it may also route it to the
underlying estimator.

First a few imports and some random data for the rest of the script.
"""
import warnings
from pprint import pprint
import numpy as np
from sklearn import set_config
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, RegressorMixin, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils import metadata_routing
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping, get_routing_for_object, process_routing
from sklearn.utils.validation import check_is_fitted
(n_samples, n_features) = (100, 4)
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)
y = rng.randint(0, 2, size=n_samples)
my_groups = rng.randint(0, 10, size=n_samples)
my_weights = rng.rand(n_samples)
my_other_weights = rng.rand(n_samples)
set_config(enable_metadata_routing=True)

def check_metadata(obj, **kwargs):
    if False:
        while True:
            i = 10
    for (key, value) in kwargs.items():
        if value is not None:
            print(f'Received {key} of length = {len(value)} in {obj.__class__.__name__}.')
        else:
            print(f'{key} is None in {obj.__class__.__name__}.')

def print_routing(obj):
    if False:
        for i in range(10):
            print('nop')
    pprint(obj.get_metadata_routing()._serialize())

class ExampleClassifier(ClassifierMixin, BaseEstimator):

    def fit(self, X, y, sample_weight=None):
        if False:
            print('Hello World!')
        check_metadata(self, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X, groups=None):
        if False:
            return 10
        check_metadata(self, groups=groups)
        return np.ones(len(X))
print_routing(ExampleClassifier())
est = ExampleClassifier().set_fit_request(sample_weight=False).set_predict_request(groups=True).set_score_request(sample_weight=False)
print_routing(est)
est = ExampleClassifier()
est.fit(X, y, sample_weight=my_weights)
est.predict(X[:3, :], groups=my_groups)

class MetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):

    def __init__(self, estimator):
        if False:
            for i in range(10):
                print('nop')
        self.estimator = estimator

    def get_metadata_routing(self):
        if False:
            while True:
                i = 10
        router = MetadataRouter(owner=self.__class__.__name__).add(estimator=self.estimator, method_mapping='one-to-one')
        return router

    def fit(self, X, y, **fit_params):
        if False:
            return 10
        request_router = get_routing_for_object(self)
        request_router.validate_metadata(params=fit_params, method='fit')
        routed_params = request_router.route_params(params=fit_params, caller='fit')
        self.estimator_ = clone(self.estimator).fit(X, y, **routed_params.estimator.fit)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X, **predict_params):
        if False:
            i = 10
            return i + 15
        check_is_fitted(self)
        request_router = get_routing_for_object(self)
        request_router.validate_metadata(params=predict_params, method='predict')
        routed_params = request_router.route_params(params=predict_params, caller='predict')
        return self.estimator_.predict(X, **routed_params.estimator.predict)
est = MetaClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight=True))
est.fit(X, y, sample_weight=my_weights)
est.fit(X, y)
try:
    est.fit(X, y, test=my_weights)
except TypeError as e:
    print(e)
try:
    est.fit(X, y, sample_weight=my_weights).predict(X, groups=my_groups)
except ValueError as e:
    print(e)
est = MetaClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight=True).set_predict_request(groups=False))
try:
    est.fit(X, y, sample_weight=my_weights).predict(X[:3, :], groups=my_groups)
except TypeError as e:
    print(e)
est = MetaClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight='aliased_sample_weight'))
est.fit(X, y, aliased_sample_weight=my_weights)
try:
    est.fit(X, y, sample_weight=my_weights)
except TypeError as e:
    print(e)
print_routing(est)
meta_est = MetaClassifier(estimator=est).fit(X, y, aliased_sample_weight=my_weights)

class RouterConsumerClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):

    def __init__(self, estimator):
        if False:
            print('Hello World!')
        self.estimator = estimator

    def get_metadata_routing(self):
        if False:
            i = 10
            return i + 15
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(estimator=self.estimator, method_mapping='one-to-one')
        return router

    def fit(self, X, y, sample_weight, **fit_params):
        if False:
            i = 10
            return i + 15
        if self.estimator is None:
            raise ValueError('estimator cannot be None!')
        check_metadata(self, sample_weight=sample_weight)
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        request_router = get_routing_for_object(self)
        request_router.validate_metadata(params=fit_params, method='fit')
        params = request_router.route_params(params=fit_params, caller='fit')
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X, **predict_params):
        if False:
            print('Hello World!')
        check_is_fitted(self)
        request_router = get_routing_for_object(self)
        request_router.validate_metadata(params=predict_params, method='predict')
        params = request_router.route_params(params=predict_params, caller='predict')
        return self.estimator_.predict(X, **params.estimator.predict)
est = RouterConsumerClassifier(estimator=ExampleClassifier())
print_routing(est)
est = RouterConsumerClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight=True))
print_routing(est)
est = RouterConsumerClassifier(estimator=ExampleClassifier()).set_fit_request(sample_weight=True)
print_routing(est)
est = RouterConsumerClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight='clf_sample_weight')).set_fit_request(sample_weight='meta_clf_sample_weight')
print_routing(est)
est.fit(X, y, sample_weight=my_weights, clf_sample_weight=my_other_weights)
est = RouterConsumerClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight='aliased_sample_weight')).set_fit_request(sample_weight=True)
print_routing(est)

class SimplePipeline(ClassifierMixin, BaseEstimator):
    _required_parameters = ['estimator']

    def __init__(self, transformer, classifier):
        if False:
            return 10
        self.transformer = transformer
        self.classifier = classifier

    def get_metadata_routing(self):
        if False:
            for i in range(10):
                print('nop')
        router = MetadataRouter(owner=self.__class__.__name__).add(transformer=self.transformer, method_mapping=MethodMapping().add(callee='fit', caller='fit').add(callee='transform', caller='fit').add(callee='transform', caller='predict')).add(classifier=self.classifier, method_mapping='one-to-one')
        return router

    def fit(self, X, y, **fit_params):
        if False:
            for i in range(10):
                print('nop')
        params = process_routing(self, 'fit', **fit_params)
        self.transformer_ = clone(self.transformer).fit(X, y, **params.transformer.fit)
        X_transformed = self.transformer_.transform(X, **params.transformer.transform)
        self.classifier_ = clone(self.classifier).fit(X_transformed, y, **params.classifier.fit)
        return self

    def predict(self, X, **predict_params):
        if False:
            return 10
        params = process_routing(self, 'predict', **predict_params)
        X_transformed = self.transformer_.transform(X, **params.transformer.transform)
        return self.classifier_.predict(X_transformed, **params.classifier.predict)

class ExampleTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y, sample_weight=None):
        if False:
            print('Hello World!')
        check_metadata(self, sample_weight=sample_weight)
        return self

    def transform(self, X, groups=None):
        if False:
            print('Hello World!')
        check_metadata(self, groups=groups)
        return X

    def fit_transform(self, X, y, sample_weight=None, groups=None):
        if False:
            return 10
        return self.fit(X, y, sample_weight).transform(X, groups)
est = SimplePipeline(transformer=ExampleTransformer().set_fit_request(sample_weight=True).set_transform_request(groups=True), classifier=RouterConsumerClassifier(estimator=ExampleClassifier().set_fit_request(sample_weight=True).set_predict_request(groups=False)).set_fit_request(sample_weight=True))
est.fit(X, y, sample_weight=my_weights, groups=my_groups).predict(X[:3], groups=my_groups)

class MetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):

    def __init__(self, estimator):
        if False:
            i = 10
            return i + 15
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        if False:
            i = 10
            return i + 15
        params = process_routing(self, 'fit', **fit_params)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    def get_metadata_routing(self):
        if False:
            i = 10
            return i + 15
        router = MetadataRouter(owner=self.__class__.__name__).add(estimator=self.estimator, method_mapping='one-to-one')
        return router
reg = MetaRegressor(estimator=LinearRegression().set_fit_request(sample_weight=True))
reg.fit(X, y, sample_weight=my_weights)

class WeightedMetaRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    __metadata_request__fit = {'sample_weight': metadata_routing.WARN}

    def __init__(self, estimator):
        if False:
            i = 10
            return i + 15
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **fit_params):
        if False:
            i = 10
            return i + 15
        params = process_routing(self, 'fit', sample_weight=sample_weight, **fit_params)
        check_metadata(self, sample_weight=sample_weight)
        self.estimator_ = clone(self.estimator).fit(X, y, **params.estimator.fit)

    def get_metadata_routing(self):
        if False:
            for i in range(10):
                print('nop')
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(estimator=self.estimator, method_mapping='one-to-one')
        return router
with warnings.catch_warnings(record=True) as record:
    WeightedMetaRegressor(estimator=LinearRegression().set_fit_request(sample_weight=False)).fit(X, y, sample_weight=my_weights)
for w in record:
    print(w.message)

class ExampleRegressor(RegressorMixin, BaseEstimator):
    __metadata_request__fit = {'sample_weight': metadata_routing.WARN}

    def fit(self, X, y, sample_weight=None):
        if False:
            for i in range(10):
                print('nop')
        check_metadata(self, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        return np.zeros(shape=len(X))
with warnings.catch_warnings(record=True) as record:
    MetaRegressor(estimator=ExampleRegressor()).fit(X, y, sample_weight=my_weights)
for w in record:
    print(w.message)