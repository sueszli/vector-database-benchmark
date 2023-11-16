from typing import Callable
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from Orange.base import SklLearner, SklModel
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.data.util import get_unique_names, SharedComputeValue
from Orange.preprocess import AdaptiveNormalize
from Orange.util import dummy_callback
__all__ = ['LocalOutlierFactorLearner', 'IsolationForestLearner', 'EllipticEnvelopeLearner', 'OneClassSVMLearner']

class _CachedTransform:

    def __init__(self, model):
        if False:
            print('Hello World!')
        self.model = model

    def __call__(self, data):
        if False:
            print('Hello World!')
        return self.model.data_to_model_domain(data)

class _OutlierModel(SklModel):

    def __init__(self, skl_model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(skl_model)
        self.outlier_var = None
        self.cached_transform = _CachedTransform(self)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        pred = self.skl_model.predict(X)
        pred[pred == -1] = 0
        return pred[:, None]

    def new_domain(self, data: Table) -> Domain:
        if False:
            print('Hello World!')
        assert self.outlier_var is not None
        return Domain(data.domain.attributes, data.domain.class_vars, data.domain.metas + (self.outlier_var,))

    def __call__(self, data: Table, progress_callback: Callable=None) -> Table:
        if False:
            while True:
                i = 10
        assert isinstance(data, Table)
        domain = self.new_domain(data)
        if progress_callback is None:
            progress_callback = dummy_callback
        progress_callback(0, 'Predicting...')
        new_table = data.transform(domain)
        progress_callback(1)
        return new_table

class _OutlierLearner(SklLearner):
    __returns__ = _OutlierModel
    supports_multiclass = True

    def _fit_model(self, data: Table) -> _OutlierModel:
        if False:
            return 10
        domain = data.domain
        model = super()._fit_model(data.transform(Domain(domain.attributes)))
        transformer = _Transformer(model)
        names = [v.name for v in domain.variables + domain.metas]
        variable = DiscreteVariable(get_unique_names(names, 'Outlier'), values=('Yes', 'No'), compute_value=transformer)
        model.outlier_var = variable
        return model

class _Transformer(SharedComputeValue):

    def __init__(self, model: _OutlierModel):
        if False:
            i = 10
            return i + 15
        super().__init__(model.cached_transform)
        self._model = model

    def compute(self, data: Table, shared_data: Table) -> np.ndarray:
        if False:
            return 10
        return self._model.predict(shared_data.X)[:, 0]

class OneClassSVMLearner(_OutlierLearner):
    name = 'One class SVM'
    __wraps__ = OneClassSVM
    preprocessors = SklLearner.preprocessors + [AdaptiveNormalize()]
    supports_weights = True

    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, max_iter=-1, preprocessors=None):
        if False:
            print('Hello World!')
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

class LocalOutlierFactorLearner(_OutlierLearner):
    __wraps__ = LocalOutlierFactor
    name = 'Local Outlier Factor'
    supports_weights = False

    def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=True, n_jobs=None, preprocessors=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

class IsolationForestLearner(_OutlierLearner):
    __wraps__ = IsolationForest
    name = 'Isolation Forest'
    supports_weights = True

    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, behaviour='deprecated', random_state=None, verbose=0, warm_start=False, preprocessors=None):
        if False:
            print('Hello World!')
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

class EllipticEnvelopeClassifier(_OutlierModel):

    def __init__(self, skl_model):
        if False:
            while True:
                i = 10
        super().__init__(skl_model)
        self.mahal_var = None

    def mahalanobis(self, observations: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        'Computes squared Mahalanobis distances of given observations.\n\n        Parameters\n        ----------\n        observations : ndarray (n_samples, n_features)\n\n        Returns\n        -------\n        distances : ndarray (n_samples, 1)\n            Squared Mahalanobis distances given observations.\n        '
        return self.skl_model.mahalanobis(observations)[:, None]

    def new_domain(self, data: Table) -> Domain:
        if False:
            for i in range(10):
                print('nop')
        assert self.mahal_var is not None
        domain = super().new_domain(data)
        return Domain(domain.attributes, domain.class_vars, domain.metas + (self.mahal_var,))

class _TransformerMahalanobis(_Transformer):

    def compute(self, data: Table, shared_data: Table) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        return self._model.mahalanobis(shared_data.X)[:, 0]

class EllipticEnvelopeLearner(_OutlierLearner):
    __wraps__ = EllipticEnvelope
    __returns__ = EllipticEnvelopeClassifier
    name = 'Covariance Estimator'
    supports_weights = False

    def __init__(self, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None, preprocessors=None):
        if False:
            while True:
                i = 10
        super().__init__(preprocessors=preprocessors)
        self.params = vars()

    def _fit_model(self, data: Table) -> EllipticEnvelopeClassifier:
        if False:
            print('Hello World!')
        domain = data.domain
        model = super()._fit_model(data.transform(Domain(domain.attributes)))
        transformer = _TransformerMahalanobis(model)
        names = [v.name for v in domain.variables + domain.metas]
        variable = ContinuousVariable(get_unique_names(names, 'Mahalanobis'), compute_value=transformer)
        model.mahal_var = variable
        return model