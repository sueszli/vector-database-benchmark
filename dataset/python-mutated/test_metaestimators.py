"""Common tests for metaestimators"""
import functools
from inspect import signature
import numpy as np
import pytest
from sklearn.base import BaseEstimator, is_regressor
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import all_estimators
from sklearn.utils._testing import set_random_state
from sklearn.utils.estimator_checks import _enforce_estimator_tags_X, _enforce_estimator_tags_y
from sklearn.utils.validation import check_is_fitted

class DelegatorData:

    def __init__(self, name, construct, skip_methods=(), fit_args=make_classification(random_state=0)):
        if False:
            while True:
                i = 10
        self.name = name
        self.construct = construct
        self.fit_args = fit_args
        self.skip_methods = skip_methods
DELEGATING_METAESTIMATORS = [DelegatorData('Pipeline', lambda est: Pipeline([('est', est)])), DelegatorData('GridSearchCV', lambda est: GridSearchCV(est, param_grid={'param': [5]}, cv=2), skip_methods=['score']), DelegatorData('RandomizedSearchCV', lambda est: RandomizedSearchCV(est, param_distributions={'param': [5]}, cv=2, n_iter=1), skip_methods=['score']), DelegatorData('RFE', RFE, skip_methods=['transform', 'inverse_transform']), DelegatorData('RFECV', RFECV, skip_methods=['transform', 'inverse_transform']), DelegatorData('BaggingClassifier', BaggingClassifier, skip_methods=['transform', 'inverse_transform', 'score', 'predict_proba', 'predict_log_proba', 'predict']), DelegatorData('SelfTrainingClassifier', lambda est: SelfTrainingClassifier(est), skip_methods=['transform', 'inverse_transform', 'predict_proba'])]

def test_metaestimator_delegation():
    if False:
        for i in range(10):
            print('nop')

    def hides(method):
        if False:
            print('Hello World!')

        @property
        def wrapper(obj):
            if False:
                for i in range(10):
                    print('nop')
            if obj.hidden_method == method.__name__:
                raise AttributeError('%r is hidden' % obj.hidden_method)
            return functools.partial(method, obj)
        return wrapper

    class SubEstimator(BaseEstimator):

        def __init__(self, param=1, hidden_method=None):
            if False:
                for i in range(10):
                    print('nop')
            self.param = param
            self.hidden_method = hidden_method

        def fit(self, X, y=None, *args, **kwargs):
            if False:
                print('Hello World!')
            self.coef_ = np.arange(X.shape[1])
            self.classes_ = []
            return True

        def _check_fit(self):
            if False:
                return 10
            check_is_fitted(self)

        @hides
        def inverse_transform(self, X, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            self._check_fit()
            return X

        @hides
        def transform(self, X, *args, **kwargs):
            if False:
                return 10
            self._check_fit()
            return X

        @hides
        def predict(self, X, *args, **kwargs):
            if False:
                return 10
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_proba(self, X, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def predict_log_proba(self, X, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def decision_function(self, X, *args, **kwargs):
            if False:
                while True:
                    i = 10
            self._check_fit()
            return np.ones(X.shape[0])

        @hides
        def score(self, X, y, *args, **kwargs):
            if False:
                while True:
                    i = 10
            self._check_fit()
            return 1.0
    methods = [k for k in SubEstimator.__dict__.keys() if not k.startswith('_') and (not k.startswith('fit'))]
    methods.sort()
    for delegator_data in DELEGATING_METAESTIMATORS:
        delegate = SubEstimator()
        delegator = delegator_data.construct(delegate)
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            assert hasattr(delegate, method)
            assert hasattr(delegator, method), '%s does not have method %r when its delegate does' % (delegator_data.name, method)
            if method == 'score':
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(delegator_data.fit_args[0], delegator_data.fit_args[1])
            else:
                with pytest.raises(NotFittedError):
                    getattr(delegator, method)(delegator_data.fit_args[0])
        delegator.fit(*delegator_data.fit_args)
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            if method == 'score':
                getattr(delegator, method)(delegator_data.fit_args[0], delegator_data.fit_args[1])
            else:
                getattr(delegator, method)(delegator_data.fit_args[0])
        for method in methods:
            if method in delegator_data.skip_methods:
                continue
            delegate = SubEstimator(hidden_method=method)
            delegator = delegator_data.construct(delegate)
            assert not hasattr(delegate, method)
            assert not hasattr(delegator, method), '%s has method %r when its delegate does not' % (delegator_data.name, method)

def _generate_meta_estimator_instances_with_pipeline():
    if False:
        for i in range(10):
            print('nop')
    'Generate instances of meta-estimators fed with a pipeline\n\n    Are considered meta-estimators all estimators accepting one of "estimator",\n    "base_estimator" or "estimators".\n    '
    for (_, Estimator) in sorted(all_estimators()):
        sig = set(signature(Estimator).parameters)
        if 'estimator' in sig or 'base_estimator' in sig or 'regressor' in sig:
            if is_regressor(Estimator):
                estimator = make_pipeline(TfidfVectorizer(), Ridge())
                param_grid = {'ridge__alpha': [0.1, 1.0]}
            else:
                estimator = make_pipeline(TfidfVectorizer(), LogisticRegression())
                param_grid = {'logisticregression__C': [0.1, 1.0]}
            if 'param_grid' in sig or 'param_distributions' in sig:
                extra_params = {'n_iter': 2} if 'n_iter' in sig else {}
                yield Estimator(estimator, param_grid, **extra_params)
            else:
                yield Estimator(estimator)
        elif 'transformer_list' in sig:
            transformer_list = [('trans1', make_pipeline(TfidfVectorizer(), MaxAbsScaler())), ('trans2', make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False)))]
            yield Estimator(transformer_list)
        elif 'estimators' in sig:
            if is_regressor(Estimator):
                estimator = [('est1', make_pipeline(TfidfVectorizer(), Ridge(alpha=0.1))), ('est2', make_pipeline(TfidfVectorizer(), Ridge(alpha=1)))]
            else:
                estimator = [('est1', make_pipeline(TfidfVectorizer(), LogisticRegression(C=0.1))), ('est2', make_pipeline(TfidfVectorizer(), LogisticRegression(C=1)))]
            yield Estimator(estimator)
        else:
            continue
DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE = ['AdaBoostClassifier', 'AdaBoostRegressor', 'BaggingClassifier', 'BaggingRegressor', 'ClassifierChain', 'IterativeImputer', 'OneVsOneClassifier', 'RANSACRegressor', 'RFE', 'RFECV', 'RegressorChain', 'SelfTrainingClassifier', 'SequentialFeatureSelector']
DATA_VALIDATION_META_ESTIMATORS = [est for est in _generate_meta_estimator_instances_with_pipeline() if est.__class__.__name__ not in DATA_VALIDATION_META_ESTIMATORS_TO_IGNORE]

def _get_meta_estimator_id(estimator):
    if False:
        while True:
            i = 10
    return estimator.__class__.__name__

@pytest.mark.parametrize('estimator', DATA_VALIDATION_META_ESTIMATORS, ids=_get_meta_estimator_id)
def test_meta_estimators_delegate_data_validation(estimator):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(0)
    set_random_state(estimator)
    n_samples = 30
    X = rng.choice(np.array(['aa', 'bb', 'cc'], dtype=object), size=n_samples)
    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(3, size=n_samples)
    X = _enforce_estimator_tags_X(estimator, X).tolist()
    y = _enforce_estimator_tags_y(estimator, y).tolist()
    estimator.fit(X, y)
    assert not hasattr(estimator, 'n_features_in_')