import sklearn.compose
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted

class PowerTransformedTargetRegressor(sklearn.compose.TransformedTargetRegressor):

    def __init__(self, regressor=None, *, power_transformer_method='box-cox', power_transformer_standardize=True, **kwargs):
        if False:
            while True:
                i = 10
        self.regressor = regressor
        self.power_transformer_method = power_transformer_method
        self.power_transformer_standardize = power_transformer_standardize
        self.transformer = PowerTransformer(method=self.power_transformer_method, standardize=self.power_transformer_standardize)
        self.func = None
        self.inverse_func = None
        self.check_inverse = False
        self._fit_vars = set()
        self.set_params(**kwargs)

    def __getattr__(self, name: str):
        if False:
            print('Hello World!')
        if name not in ('regressor', 'regressor_'):
            if hasattr(self, 'regressor_'):
                return getattr(self.regressor_, name)
            return getattr(self.regressor, name)

    def fit(self, X, y, **fit_params):
        if False:
            for i in range(10):
                print('nop')
        'Fit the model according to the given training data.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vector, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        **fit_params : dict\n            Parameters passed to the ``fit`` method of the underlying\n            regressor.\n\n\n        Returns\n        -------\n        self : object\n        '
        y = y.astype('float64')
        super().fit(X, y, **fit_params)
        return self

    def set_params(self, **params):
        if False:
            while True:
                i = 10
        "\n        Set the parameters of this estimator.\n\n        The method works on simple estimators as well as on nested objects\n        (such as pipelines). The latter have parameters of the form\n        ``<component>__<parameter>`` so that it's possible to update each\n        component of a nested object.\n\n        Parameters\n        ----------\n        **params : dict\n            Estimator parameters.\n\n        Returns\n        -------\n        self : object\n            Estimator instance.\n        "
        if 'regressor' in params:
            self.regressor = params.pop('regressor')
        if 'power_transformer_method' in params:
            self.power_transformer_method = params.pop('power_transformer_method')
            self.transformer.set_params(**{'method': self.power_transformer_method})
        if 'power_transformer_standardize' in params:
            self.power_transformer_standardize = params.pop('power_transformer_standardize')
            self.transformer.set_params(**{'standardize': self.power_transformer_standardize})
        return self.regressor.set_params(**params)

    def get_params(self, deep=True):
        if False:
            i = 10
            return i + 15
        '\n        Get parameters for this estimator.\n\n        Parameters\n        ----------\n        deep : bool, default=True\n            If True, will return the parameters for this estimator and\n            contained subobjects that are estimators.\n\n        Returns\n        -------\n        params : mapping of string to any\n            Parameter names mapped to their values.\n        '
        r = self.regressor.get_params(deep=deep)
        r['power_transformer_method'] = self.power_transformer_method
        r['power_transformer_standardize'] = self.power_transformer_standardize
        r['regressor'] = self.regressor
        return r

class CustomProbabilityThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Meta-estimator to set a custom probability threshold."""

    def __init__(self, classifier=None, *, probability_threshold=0.5, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.classifier = classifier
        self.probability_threshold = probability_threshold
        self.set_params(**kwargs)

    def fit(self, X, y, **fit_params):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.probability_threshold, (int, float)) or self.probability_threshold > 1 or self.probability_threshold < 0:
            raise TypeError('probability_threshold parameter only accepts value between 0 to 1.')
        if self.classifier is None:
            from sklearn.linear_model import LogisticRegression
            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        if False:
            print('Hello World!')
        check_is_fitted(self)
        if not hasattr(self.classifier_, 'predict_proba'):
            return self.classifier_.predict(X, **predict_params)
        pred = self.classifier_.predict_proba(X, **predict_params)
        if pred.shape[1] > 2:
            raise ValueError(f'{self.__class__.__name__} can only be used for binary classification.')
        return (pred[:, 1] >= self.probability_threshold).astype('int')

    def __getattr__(self, name: str):
        if False:
            while True:
                i = 10
        if name not in ('classifier', 'classifier_'):
            if hasattr(self, 'classifier_'):
                return getattr(self.classifier_, name)
            return getattr(self.classifier, name)

    def set_params(self, **params):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the parameters of this estimator.\n\n        The method works on simple estimators as well as on nested objects\n        (such as pipelines). The latter have parameters of the form\n        ``<component>__<parameter>`` so that it's possible to update each\n        component of a nested object.\n\n        Parameters\n        ----------\n        **params : dict\n            Estimator parameters.\n\n        Returns\n        -------\n        self : object\n            Estimator instance.\n        "
        if 'classifier' in params:
            self.classifier = params.pop('classifier')
        if 'probability_threshold' in params:
            self.probability_threshold = params.pop('probability_threshold')
        return self.classifier.set_params(**params)

    def get_params(self, deep=True):
        if False:
            while True:
                i = 10
        '\n        Get parameters for this estimator.\n\n        Parameters\n        ----------\n        deep : bool, default=True\n            If True, will return the parameters for this estimator and\n            contained subobjects that are estimators.\n\n        Returns\n        -------\n        params : mapping of string to any\n            Parameter names mapped to their values.\n        '
        r = self.classifier.get_params(deep=deep)
        r['classifier'] = self.classifier
        r['probability_threshold'] = self.probability_threshold
        return r

def get_estimator_from_meta_estimator(estimator):
    if False:
        i = 10
        return i + 15
    '\n    If ``estimator`` is a meta estimator, get estimator inside.\n    Otherwise return ``estimator``. Will try to return the fitted\n    estimator first.\n    '
    if not isinstance(estimator, (TransformedTargetRegressor, CustomProbabilityThresholdClassifier)):
        return estimator
    if hasattr(estimator, 'regressor_'):
        return get_estimator_from_meta_estimator(estimator.regressor_)
    if hasattr(estimator, 'classifier_'):
        return get_estimator_from_meta_estimator(estimator.classifier_)
    if hasattr(estimator, 'regressor'):
        return get_estimator_from_meta_estimator(estimator.regressor)
    if hasattr(estimator, 'classifier'):
        return get_estimator_from_meta_estimator(estimator.classifier)
    return estimator