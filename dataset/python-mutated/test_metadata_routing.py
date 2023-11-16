"""
Metadata Routing Utility Tests
"""
import re
import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.tests.metadata_routing_common import ConsumingClassifier, ConsumingRegressor, ConsumingTransformer, MetaRegressor, MetaTransformer, NonConsumingClassifier, WeightedMetaClassifier, WeightedMetaRegressor, _Registry, assert_request_equal, assert_request_is_empty, check_recorded_metadata
from sklearn.utils import metadata_routing
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS, SIMPLE_METHODS, MethodMetadataRequest, MethodPair, _MetadataRequester, request_is_alias, request_is_valid
from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter, MethodMapping, get_routing_for_object, process_routing
from sklearn.utils.validation import check_is_fitted
rng = np.random.RandomState(42)
(N, M) = (100, 4)
X = rng.rand(N, M)
y = rng.randint(0, 2, size=N)
my_groups = rng.randint(0, 10, size=N)
my_weights = rng.rand(N)
my_other_weights = rng.rand(N)

@pytest.fixture(autouse=True)
def enable_slep006():
    if False:
        i = 10
        return i + 15
    'Enable SLEP006 for all tests.'
    with config_context(enable_metadata_routing=True):
        yield

class SimplePipeline(BaseEstimator):
    """A very simple pipeline, assuming the last step is always a predictor."""

    def __init__(self, steps):
        if False:
            return 10
        self.steps = steps

    def fit(self, X, y, **fit_params):
        if False:
            i = 10
            return i + 15
        self.steps_ = []
        params = process_routing(self, 'fit', **fit_params)
        X_transformed = X
        for (i, step) in enumerate(self.steps[:-1]):
            transformer = clone(step).fit(X_transformed, y, **params.get(f'step_{i}').fit)
            self.steps_.append(transformer)
            X_transformed = transformer.transform(X_transformed, **params.get(f'step_{i}').transform)
        self.steps_.append(clone(self.steps[-1]).fit(X_transformed, y, **params.predictor.fit))
        return self

    def predict(self, X, **predict_params):
        if False:
            return 10
        check_is_fitted(self)
        X_transformed = X
        params = process_routing(self, 'predict', **predict_params)
        for (i, step) in enumerate(self.steps_[:-1]):
            X_transformed = step.transform(X, **params.get(f'step_{i}').transform)
        return self.steps_[-1].predict(X_transformed, **params.predictor.predict)

    def get_metadata_routing(self):
        if False:
            while True:
                i = 10
        router = MetadataRouter(owner=self.__class__.__name__)
        for (i, step) in enumerate(self.steps[:-1]):
            router.add(**{f'step_{i}': step}, method_mapping=MethodMapping().add(callee='fit', caller='fit').add(callee='transform', caller='fit').add(callee='transform', caller='predict'))
        router.add(predictor=self.steps[-1], method_mapping='one-to-one')
        return router

def test_assert_request_is_empty():
    if False:
        for i in range(10):
            print('nop')
    requests = MetadataRequest(owner='test')
    assert_request_is_empty(requests)
    requests.fit.add_request(param='foo', alias=None)
    assert_request_is_empty(requests)
    requests.fit.add_request(param='bar', alias='value')
    with pytest.raises(AssertionError):
        assert_request_is_empty(requests)
    assert_request_is_empty(requests, exclude='fit')
    requests.score.add_request(param='carrot', alias=True)
    with pytest.raises(AssertionError):
        assert_request_is_empty(requests, exclude='fit')
    assert_request_is_empty(requests, exclude=['fit', 'score'])
    assert_request_is_empty(MetadataRouter(owner='test').add_self_request(WeightedMetaRegressor(estimator=None)).add(method_mapping='fit', estimator=ConsumingRegressor()))

@pytest.mark.parametrize('estimator', [ConsumingClassifier(registry=_Registry()), ConsumingRegressor(registry=_Registry()), ConsumingTransformer(registry=_Registry()), NonConsumingClassifier(registry=_Registry()), WeightedMetaClassifier(estimator=ConsumingClassifier(), registry=_Registry()), WeightedMetaRegressor(estimator=ConsumingRegressor(), registry=_Registry())])
def test_estimator_puts_self_in_registry(estimator):
    if False:
        return 10
    'Check that an estimator puts itself in the registry upon fit.'
    estimator.fit(X, y)
    assert estimator in estimator.registry

@pytest.mark.parametrize('val, res', [(False, False), (True, False), (None, False), ('$UNUSED$', False), ('$WARN$', False), ('invalid-input', False), ('valid_arg', True)])
def test_request_type_is_alias(val, res):
    if False:
        print('Hello World!')
    assert request_is_alias(val) == res

@pytest.mark.parametrize('val, res', [(False, True), (True, True), (None, True), ('$UNUSED$', True), ('$WARN$', True), ('invalid-input', False), ('alias_arg', False)])
def test_request_type_is_valid(val, res):
    if False:
        i = 10
        return i + 15
    assert request_is_valid(val) == res

def test_default_requests():
    if False:
        while True:
            i = 10

    class OddEstimator(BaseEstimator):
        __metadata_request__fit = {'sample_weight': True}
    odd_request = get_routing_for_object(OddEstimator())
    assert odd_request.fit.requests == {'sample_weight': True}
    assert not len(get_routing_for_object(NonConsumingClassifier()).fit.requests)
    assert_request_is_empty(NonConsumingClassifier().get_metadata_routing())
    trs_request = get_routing_for_object(ConsumingTransformer())
    assert trs_request.fit.requests == {'sample_weight': None, 'metadata': None}
    assert trs_request.transform.requests == {'metadata': None, 'sample_weight': None}
    assert_request_is_empty(trs_request)
    est_request = get_routing_for_object(ConsumingClassifier())
    assert est_request.fit.requests == {'sample_weight': None, 'metadata': None}
    assert_request_is_empty(est_request)

def test_process_routing_invalid_method():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError, match='Can only route and process input'):
        process_routing(ConsumingClassifier(), 'invalid_method', **{})

def test_process_routing_invalid_object():
    if False:
        print('Hello World!')

    class InvalidObject:
        pass
    with pytest.raises(AttributeError, match='either implement the routing method'):
        process_routing(InvalidObject(), 'fit', **{})

def test_simple_metadata_routing():
    if False:
        while True:
            i = 10
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    clf.fit(X, y)
    clf = WeightedMetaClassifier(estimator=NonConsumingClassifier())
    clf.fit(X, y, sample_weight=my_weights)
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier())
    err_message = '[sample_weight] are passed but are not explicitly set as requested or not for ConsumingClassifier.fit'
    with pytest.raises(ValueError, match=re.escape(err_message)):
        clf.fit(X, y, sample_weight=my_weights)
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier().set_fit_request(sample_weight=False))
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, 'fit')
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier().set_fit_request(sample_weight=True))
    clf.fit(X, y, sample_weight=my_weights)
    check_recorded_metadata(clf.estimator_, 'fit', sample_weight=my_weights)
    clf = WeightedMetaClassifier(estimator=ConsumingClassifier().set_fit_request(sample_weight='alternative_weight'))
    clf.fit(X, y, alternative_weight=my_weights)
    check_recorded_metadata(clf.estimator_, 'fit', sample_weight=my_weights)

def test_nested_routing():
    if False:
        print('Hello World!')
    pipeline = SimplePipeline([MetaTransformer(transformer=ConsumingTransformer().set_fit_request(metadata=True, sample_weight=False).set_transform_request(sample_weight=True, metadata=False)), WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight='inner_weights', metadata=False).set_predict_request(sample_weight=False)).set_fit_request(sample_weight='outer_weights')])
    (w1, w2, w3) = ([1], [2], [3])
    pipeline.fit(X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2, inner_weights=w3)
    check_recorded_metadata(pipeline.steps_[0].transformer_, 'fit', metadata=my_groups, sample_weight=None)
    check_recorded_metadata(pipeline.steps_[0].transformer_, 'transform', sample_weight=w1, metadata=None)
    check_recorded_metadata(pipeline.steps_[1], 'fit', sample_weight=w2)
    check_recorded_metadata(pipeline.steps_[1].estimator_, 'fit', sample_weight=w3)
    pipeline.predict(X, sample_weight=w3)
    check_recorded_metadata(pipeline.steps_[0].transformer_, 'transform', sample_weight=w3, metadata=None)

def test_nested_routing_conflict():
    if False:
        return 10
    pipeline = SimplePipeline([MetaTransformer(transformer=ConsumingTransformer().set_fit_request(metadata=True, sample_weight=False).set_transform_request(sample_weight=True)), WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight=True)).set_fit_request(sample_weight='outer_weights')])
    (w1, w2) = ([1], [2])
    with pytest.raises(ValueError, match=re.escape('In WeightedMetaRegressor, there is a conflict on sample_weight between what is requested for this estimator and what is requested by its children. You can resolve this conflict by using an alias for the child estimator(s) requested metadata.')):
        pipeline.fit(X, y, metadata=my_groups, sample_weight=w1, outer_weights=w2)

def test_invalid_metadata():
    if False:
        print('Hello World!')
    trs = MetaTransformer(transformer=ConsumingTransformer().set_transform_request(sample_weight=True))
    with pytest.raises(TypeError, match=re.escape("transform got unexpected argument(s) {'other_param'}")):
        trs.fit(X, y).transform(X, other_param=my_weights)
    trs = MetaTransformer(transformer=ConsumingTransformer().set_transform_request(sample_weight=False))
    with pytest.raises(TypeError, match=re.escape("transform got unexpected argument(s) {'sample_weight'}")):
        trs.fit(X, y).transform(X, sample_weight=my_weights)

def test_get_metadata_routing():
    if False:
        while True:
            i = 10

    class TestDefaultsBadMethodName(_MetadataRequester):
        __metadata_request__fit = {'sample_weight': None, 'my_param': None}
        __metadata_request__score = {'sample_weight': None, 'my_param': True, 'my_other_param': None}
        __metadata_request__other_method = {'my_param': True}

    class TestDefaults(_MetadataRequester):
        __metadata_request__fit = {'sample_weight': None, 'my_other_param': None}
        __metadata_request__score = {'sample_weight': None, 'my_param': True, 'my_other_param': None}
        __metadata_request__predict = {'my_param': True}
    with pytest.raises(AttributeError, match="'MetadataRequest' object has no attribute 'other_method'"):
        TestDefaultsBadMethodName().get_metadata_routing()
    expected = {'score': {'my_param': True, 'my_other_param': None, 'sample_weight': None}, 'fit': {'my_other_param': None, 'sample_weight': None}, 'predict': {'my_param': True}}
    assert_request_equal(TestDefaults().get_metadata_routing(), expected)
    est = TestDefaults().set_score_request(my_param='other_param')
    expected = {'score': {'my_param': 'other_param', 'my_other_param': None, 'sample_weight': None}, 'fit': {'my_other_param': None, 'sample_weight': None}, 'predict': {'my_param': True}}
    assert_request_equal(est.get_metadata_routing(), expected)
    est = TestDefaults().set_fit_request(sample_weight=True)
    expected = {'score': {'my_param': True, 'my_other_param': None, 'sample_weight': None}, 'fit': {'my_other_param': None, 'sample_weight': True}, 'predict': {'my_param': True}}
    assert_request_equal(est.get_metadata_routing(), expected)

def test_setting_default_requests():
    if False:
        for i in range(10):
            print('nop')
    test_cases = dict()

    class ExplicitRequest(BaseEstimator):
        __metadata_request__fit = {'prop': None}

        def fit(self, X, y, **kwargs):
            if False:
                print('Hello World!')
            return self
    test_cases[ExplicitRequest] = {'prop': None}

    class ExplicitRequestOverwrite(BaseEstimator):
        __metadata_request__fit = {'prop': True}

        def fit(self, X, y, prop=None, **kwargs):
            if False:
                while True:
                    i = 10
            return self
    test_cases[ExplicitRequestOverwrite] = {'prop': True}

    class ImplicitRequest(BaseEstimator):

        def fit(self, X, y, prop=None, **kwargs):
            if False:
                return 10
            return self
    test_cases[ImplicitRequest] = {'prop': None}

    class ImplicitRequestRemoval(BaseEstimator):
        __metadata_request__fit = {'prop': metadata_routing.UNUSED}

        def fit(self, X, y, prop=None, **kwargs):
            if False:
                return 10
            return self
    test_cases[ImplicitRequestRemoval] = {}
    for (Klass, requests) in test_cases.items():
        assert get_routing_for_object(Klass()).fit.requests == requests
        assert_request_is_empty(Klass().get_metadata_routing(), exclude='fit')
        Klass().fit(None, None)

def test_removing_non_existing_param_raises():
    if False:
        return 10
    "Test that removing a metadata using UNUSED which doesn't exist raises."

    class InvalidRequestRemoval(BaseEstimator):
        __metadata_request__fit = {'prop': metadata_routing.UNUSED}

        def fit(self, X, y, **kwargs):
            if False:
                i = 10
                return i + 15
            return self
    with pytest.raises(ValueError, match='Trying to remove parameter'):
        InvalidRequestRemoval().get_metadata_routing()

def test_method_metadata_request():
    if False:
        return 10
    mmr = MethodMetadataRequest(owner='test', method='fit')
    with pytest.raises(ValueError, match="The alias you're setting for"):
        mmr.add_request(param='foo', alias=1.4)
    mmr.add_request(param='foo', alias=None)
    assert mmr.requests == {'foo': None}
    mmr.add_request(param='foo', alias=False)
    assert mmr.requests == {'foo': False}
    mmr.add_request(param='foo', alias=True)
    assert mmr.requests == {'foo': True}
    mmr.add_request(param='foo', alias='foo')
    assert mmr.requests == {'foo': True}
    mmr.add_request(param='foo', alias='bar')
    assert mmr.requests == {'foo': 'bar'}
    assert mmr._get_param_names(return_alias=False) == {'foo'}
    assert mmr._get_param_names(return_alias=True) == {'bar'}

def test_get_routing_for_object():
    if False:
        i = 10
        return i + 15

    class Consumer(BaseEstimator):
        __metadata_request__fit = {'prop': None}
    assert_request_is_empty(get_routing_for_object(None))
    assert_request_is_empty(get_routing_for_object(object()))
    mr = MetadataRequest(owner='test')
    mr.fit.add_request(param='foo', alias='bar')
    mr_factory = get_routing_for_object(mr)
    assert_request_is_empty(mr_factory, exclude='fit')
    assert mr_factory.fit.requests == {'foo': 'bar'}
    mr = get_routing_for_object(Consumer())
    assert_request_is_empty(mr, exclude='fit')
    assert mr.fit.requests == {'prop': None}

def test_metadata_request_consumes_method():
    if False:
        for i in range(10):
            print('nop')
    'Test that MetadataRequest().consumes() method works as expected.'
    request = MetadataRouter(owner='test')
    assert request.consumes(method='fit', params={'foo'}) == set()
    request = MetadataRequest(owner='test')
    request.fit.add_request(param='foo', alias=True)
    assert request.consumes(method='fit', params={'foo'}) == {'foo'}
    request = MetadataRequest(owner='test')
    request.fit.add_request(param='foo', alias='bar')
    assert request.consumes(method='fit', params={'bar', 'foo'}) == {'bar'}

def test_metadata_router_consumes_method():
    if False:
        while True:
            i = 10
    'Test that MetadataRouter().consumes method works as expected.'
    cases = [(WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight=True)), {'sample_weight'}, {'sample_weight'}), (WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight='my_weights')), {'my_weights', 'sample_weight'}, {'my_weights'})]
    for (obj, input, output) in cases:
        assert obj.get_metadata_routing().consumes(method='fit', params=input) == output

def test_metaestimator_warnings():
    if False:
        while True:
            i = 10

    class WeightedMetaRegressorWarn(WeightedMetaRegressor):
        __metadata_request__fit = {'sample_weight': metadata_routing.WARN}
    with pytest.warns(UserWarning, match='Support for .* has recently been added to this class'):
        WeightedMetaRegressorWarn(estimator=LinearRegression().set_fit_request(sample_weight=False)).fit(X, y, sample_weight=my_weights)

def test_estimator_warnings():
    if False:
        return 10

    class ConsumingRegressorWarn(ConsumingRegressor):
        __metadata_request__fit = {'sample_weight': metadata_routing.WARN}
    with pytest.warns(UserWarning, match='Support for .* has recently been added to this class'):
        MetaRegressor(estimator=ConsumingRegressorWarn()).fit(X, y, sample_weight=my_weights)

@pytest.mark.parametrize('obj, string', [(MethodMetadataRequest(owner='test', method='fit').add_request(param='foo', alias='bar'), "{'foo': 'bar'}"), (MetadataRequest(owner='test'), '{}'), (MethodMapping.from_str('score'), "[{'callee': 'score', 'caller': 'score'}]"), (MetadataRouter(owner='test').add(method_mapping='predict', estimator=ConsumingRegressor()), "{'estimator': {'mapping': [{'callee': 'predict', 'caller': 'predict'}], 'router': {'fit': {'sample_weight': None, 'metadata': None}, 'partial_fit': {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight': None, 'metadata': None}, 'score': {'sample_weight': None}}}}")])
def test_string_representations(obj, string):
    if False:
        i = 10
        return i + 15
    assert str(obj) == string

@pytest.mark.parametrize('obj, method, inputs, err_cls, err_msg', [(MethodMapping(), 'add', {'callee': 'invalid', 'caller': 'fit'}, ValueError, 'Given callee'), (MethodMapping(), 'add', {'callee': 'fit', 'caller': 'invalid'}, ValueError, 'Given caller'), (MethodMapping, 'from_str', {'route': 'invalid'}, ValueError, "route should be 'one-to-one' or a single method!"), (MetadataRouter(owner='test'), 'add_self_request', {'obj': MetadataRouter(owner='test')}, ValueError, 'Given `obj` is neither a `MetadataRequest` nor does it implement'), (ConsumingClassifier(), 'set_fit_request', {'invalid': True}, TypeError, 'Unexpected args')])
def test_validations(obj, method, inputs, err_cls, err_msg):
    if False:
        i = 10
        return i + 15
    with pytest.raises(err_cls, match=err_msg):
        getattr(obj, method)(**inputs)

def test_methodmapping():
    if False:
        while True:
            i = 10
    mm = MethodMapping().add(caller='fit', callee='transform').add(caller='fit', callee='fit')
    mm_list = list(mm)
    assert mm_list[0] == ('transform', 'fit')
    assert mm_list[1] == ('fit', 'fit')
    mm = MethodMapping.from_str('one-to-one')
    for method in METHODS:
        assert MethodPair(method, method) in mm._routes
    assert len(mm._routes) == len(METHODS)
    mm = MethodMapping.from_str('score')
    assert repr(mm) == "[{'callee': 'score', 'caller': 'score'}]"

def test_metadatarouter_add_self_request():
    if False:
        i = 10
        return i + 15
    request = MetadataRequest(owner='nested')
    request.fit.add_request(param='param', alias=True)
    router = MetadataRouter(owner='test').add_self_request(request)
    assert str(router._self_request) == str(request)
    assert router._self_request is not request
    est = ConsumingRegressor().set_fit_request(sample_weight='my_weights')
    router = MetadataRouter(owner='test').add_self_request(obj=est)
    assert str(router._self_request) == str(est.get_metadata_routing())
    assert router._self_request is not est.get_metadata_routing()
    est = WeightedMetaRegressor(estimator=ConsumingRegressor().set_fit_request(sample_weight='nested_weights'))
    router = MetadataRouter(owner='test').add_self_request(obj=est)
    assert str(router._self_request) == str(est._get_metadata_request())
    assert str(router._self_request) != str(est.get_metadata_routing())
    assert router._self_request is not est._get_metadata_request()

def test_metadata_routing_add():
    if False:
        print('Hello World!')
    router = MetadataRouter(owner='test').add(method_mapping='fit', est=ConsumingRegressor().set_fit_request(sample_weight='weights'))
    assert str(router) == "{'est': {'mapping': [{'callee': 'fit', 'caller': 'fit'}], 'router': {'fit': {'sample_weight': 'weights', 'metadata': None}, 'partial_fit': {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight': None, 'metadata': None}, 'score': {'sample_weight': None}}}}"
    router = MetadataRouter(owner='test').add(method_mapping=MethodMapping().add(callee='score', caller='fit'), est=ConsumingRegressor().set_score_request(sample_weight=True))
    assert str(router) == "{'est': {'mapping': [{'callee': 'score', 'caller': 'fit'}], 'router': {'fit': {'sample_weight': None, 'metadata': None}, 'partial_fit': {'sample_weight': None, 'metadata': None}, 'predict': {'sample_weight': None, 'metadata': None}, 'score': {'sample_weight': True}}}}"

def test_metadata_routing_get_param_names():
    if False:
        print('Hello World!')
    router = MetadataRouter(owner='test').add_self_request(WeightedMetaRegressor(estimator=ConsumingRegressor()).set_fit_request(sample_weight='self_weights')).add(method_mapping='fit', trs=ConsumingTransformer().set_fit_request(sample_weight='transform_weights'))
    assert str(router) == "{'$self_request': {'fit': {'sample_weight': 'self_weights'}, 'score': {'sample_weight': None}}, 'trs': {'mapping': [{'callee': 'fit', 'caller': 'fit'}], 'router': {'fit': {'sample_weight': 'transform_weights', 'metadata': None}, 'transform': {'sample_weight': None, 'metadata': None}, 'inverse_transform': {'sample_weight': None, 'metadata': None}}}}"
    assert router._get_param_names(method='fit', return_alias=True, ignore_self_request=False) == {'transform_weights', 'metadata', 'self_weights'}
    assert router._get_param_names(method='fit', return_alias=False, ignore_self_request=False) == {'sample_weight', 'metadata', 'transform_weights'}
    assert router._get_param_names(method='fit', return_alias=False, ignore_self_request=True) == {'metadata', 'transform_weights'}
    assert router._get_param_names(method='fit', return_alias=True, ignore_self_request=True) == router._get_param_names(method='fit', return_alias=False, ignore_self_request=True)

def test_method_generation():
    if False:
        i = 10
        return i + 15

    class SimpleEstimator(BaseEstimator):

        def fit(self, X, y):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def fit_transform(self, X, y):
            if False:
                return 10
            pass

        def fit_predict(self, X, y):
            if False:
                return 10
            pass

        def partial_fit(self, X, y):
            if False:
                i = 10
                return i + 15
            pass

        def predict(self, X):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def predict_proba(self, X):
            if False:
                while True:
                    i = 10
            pass

        def predict_log_proba(self, X):
            if False:
                return 10
            pass

        def decision_function(self, X):
            if False:
                i = 10
                return i + 15
            pass

        def score(self, X, y):
            if False:
                print('Hello World!')
            pass

        def split(self, X, y=None):
            if False:
                i = 10
                return i + 15
            pass

        def transform(self, X):
            if False:
                while True:
                    i = 10
            pass

        def inverse_transform(self, X):
            if False:
                return 10
            pass
    for method in METHODS:
        assert not hasattr(SimpleEstimator(), f'set_{method}_request')

    class SimpleEstimator(BaseEstimator):

        def fit(self, X, y, sample_weight=None):
            if False:
                return 10
            pass

        def fit_transform(self, X, y, sample_weight=None):
            if False:
                i = 10
                return i + 15
            pass

        def fit_predict(self, X, y, sample_weight=None):
            if False:
                while True:
                    i = 10
            pass

        def partial_fit(self, X, y, sample_weight=None):
            if False:
                while True:
                    i = 10
            pass

        def predict(self, X, sample_weight=None):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def predict_proba(self, X, sample_weight=None):
            if False:
                return 10
            pass

        def predict_log_proba(self, X, sample_weight=None):
            if False:
                print('Hello World!')
            pass

        def decision_function(self, X, sample_weight=None):
            if False:
                i = 10
                return i + 15
            pass

        def score(self, X, y, sample_weight=None):
            if False:
                print('Hello World!')
            pass

        def split(self, X, y=None, sample_weight=None):
            if False:
                i = 10
                return i + 15
            pass

        def transform(self, X, sample_weight=None):
            if False:
                print('Hello World!')
            pass

        def inverse_transform(self, X, sample_weight=None):
            if False:
                i = 10
                return i + 15
            pass
    for method in COMPOSITE_METHODS:
        assert not hasattr(SimpleEstimator(), f'set_{method}_request')
    for method in SIMPLE_METHODS:
        assert hasattr(SimpleEstimator(), f'set_{method}_request')

def test_composite_methods():
    if False:
        return 10

    class SimpleEstimator(BaseEstimator):

        def fit(self, X, y, foo=None, bar=None):
            if False:
                i = 10
                return i + 15
            pass

        def predict(self, X, foo=None, bar=None):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def transform(self, X, other_param=None):
            if False:
                i = 10
                return i + 15
            pass
    est = SimpleEstimator()
    assert est.get_metadata_routing().fit_transform.requests == {'bar': None, 'foo': None, 'other_param': None}
    assert est.get_metadata_routing().fit_predict.requests == {'bar': None, 'foo': None}
    est.set_fit_request(foo=True, bar='test')
    with pytest.raises(ValueError, match='Conflicting metadata requests for'):
        est.get_metadata_routing().fit_predict
    est.set_predict_request(bar=True)
    with pytest.raises(ValueError, match='Conflicting metadata requests for'):
        est.get_metadata_routing().fit_predict
    est.set_predict_request(foo=True, bar='test')
    est.get_metadata_routing().fit_predict
    est.set_transform_request(other_param=True)
    assert est.get_metadata_routing().fit_transform.requests == {'bar': 'test', 'foo': True, 'other_param': True}

def test_no_feature_flag_raises_error():
    if False:
        print('Hello World!')
    'Test that when feature flag disabled, set_{method}_requests raises.'
    with config_context(enable_metadata_routing=False):
        with pytest.raises(RuntimeError, match='This method is only available'):
            ConsumingClassifier().set_fit_request(sample_weight=True)

def test_none_metadata_passed():
    if False:
        return 10
    "Test that passing None as metadata when not requested doesn't raise"
    MetaRegressor(estimator=ConsumingRegressor()).fit(X, y, sample_weight=None)