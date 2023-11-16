from __future__ import annotations
import copy
import typing
import numpy as np
try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
from sklearn import base as sklearn_base
from sklearn import pipeline, preprocessing, utils
from river import base, compose, stream
__all__ = ['convert_river_to_sklearn', 'River2SKLRegressor', 'River2SKLClassifier', 'River2SKLClusterer', 'River2SKLTransformer']
STREAM_METHODS: dict[type, typing.Callable] = {np.ndarray: stream.iter_array}
if PANDAS_INSTALLED:
    STREAM_METHODS[pd.DataFrame] = stream.iter_pandas
SKLEARN_INPUT_X_PARAMS = {'accept_sparse': False, 'accept_large_sparse': True, 'dtype': 'numeric', 'order': None, 'copy': False, 'force_all_finite': True, 'ensure_2d': True, 'allow_nd': False, 'ensure_min_samples': 1, 'ensure_min_features': 1}
SKLEARN_INPUT_Y_PARAMS = {'multi_output': False, 'y_numeric': False}

def convert_river_to_sklearn(estimator: base.Estimator):
    if False:
        print('Hello World!')
    'Wraps a river estimator to make it compatible with scikit-learn.\n\n    Parameters\n    ----------\n    estimator\n\n    '
    if isinstance(estimator, compose.Pipeline):
        return pipeline.Pipeline([(name, convert_river_to_sklearn(step)) for (name, step) in estimator.steps.items()])
    wrappers = [(base.Classifier, River2SKLClassifier), (base.Clusterer, River2SKLClusterer), (base.Regressor, River2SKLRegressor), (base.Transformer, River2SKLTransformer)]
    for (base_type, wrapper) in wrappers:
        if isinstance(estimator, base_type):
            obj = wrapper(estimator)
            obj.instance_ = copy.deepcopy(estimator)
            return obj
    raise ValueError("Couldn't find an appropriate wrapper")

class River2SKLBase(base.Wrapper, sklearn_base.BaseEstimator):
    """This class is just here for house-keeping."""

    @property
    def _wrapped_model(self):
        if False:
            print('Hello World!')
        return self.river_estimator
    _required_parameters = ['river_estimator']

class River2SKLRegressor(River2SKLBase, sklearn_base.RegressorMixin):
    """Compatibility layer from River to scikit-learn for regression.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Regressor):
        if False:
            return 10
        if not isinstance(river_estimator, base.Regressor):
            raise ValueError('river_estimator is not a Regressor')
        self.river_estimator = river_estimator

    def _partial_fit(self, X, y):
        if False:
            i = 10
            return i + 15
        (X, y) = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        self.n_features_in_ = X.shape[1]
        if not hasattr(self, 'instance_'):
            self.instance_ = copy.deepcopy(self.river_estimator)
        for (x, yi) in STREAM_METHODS[type(X)](X, y):
            self.instance_.learn_one(x, yi)
        return self

    def fit(self, X, y):
        if False:
            while True:
                i = 10
        'Fits to an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        for attr in ('instance_', 'n_features_in_'):
            self.__dict__.pop(attr, None)
        return self._partial_fit(X, y)

    def partial_fit(self, X, y):
        if False:
            i = 10
            return i + 15
        'Fits incrementally on a portion of a dataset.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        return self._partial_fit(X, y)

    def predict(self, X) -> np.ndarray:
        if False:
            return 10
        'Predicts the target of an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n\n        Returns\n        -------\n        Predicted target values for each row of `X`.\n\n        '
        utils.validation.check_is_fitted(self, attributes='instance_')
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        y_pred = np.empty(shape=len(X))
        for (i, (x, _)) in enumerate(stream.iter_array(X)):
            y_pred[i] = self.instance_.predict_one(x)
        return y_pred

class River2SKLClassifier(River2SKLBase, sklearn_base.ClassifierMixin):
    """Compatibility layer from River to scikit-learn for classification.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Classifier):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(river_estimator, base.Classifier):
            raise ValueError('estimator is not a Classifier')
        self.river_estimator = river_estimator

    def _more_tags(self):
        if False:
            print('Hello World!')
        return {'binary_only': not self.river_estimator._multiclass}

    def _partial_fit(self, X, y, classes):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'classes_'):
            self.classes_ = classes
        (X, y) = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)
        if len(self.classes_) > 2 and (not self.river_estimator._multiclass):
            import warnings
            warnings.warn(f'more than 2 classes were given but {self.river_estimator} is a binary classifier')
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        self.n_features_in_ = X.shape[1]
        utils.multiclass.check_classification_targets(y)
        if set(y) - set(self.classes_):
            raise ValueError('classes should include all valid labels that can be in y')
        if not hasattr(self, 'instance_'):
            self.instance_ = copy.deepcopy(self.river_estimator)
        if not self.river_estimator._multiclass:
            if not hasattr(self, 'label_encoder_'):
                self.label_encoder_ = preprocessing.LabelEncoder().fit(self.classes_)
            y = self.label_encoder_.transform(y)
        for (x, yi) in STREAM_METHODS[type(X)](X, y):
            self.instance_.learn_one(x, yi)
        return self

    def fit(self, X, y):
        if False:
            print('Hello World!')
        'Fits to an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        for attr in ('classes_', 'instance_', 'label_encoder_', 'n_features_in_'):
            self.__dict__.pop(attr, None)
        classes = utils.multiclass.unique_labels(y)
        return self._partial_fit(X, y, classes)

    def partial_fit(self, X, y, classes=None):
        if False:
            return 10
        "Fits incrementally on a portion of a dataset.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n        classes\n            Classes across all calls to partial_fit. This argument is required for the first call\n            to partial_fit and can be omitted in the subsequent calls. Note that y doesn't need to\n            contain all labels in `classes`.\n\n        Returns\n        -------\n        self\n\n        "
        return self._partial_fit(X, y, classes)

    def predict_proba(self, X):
        if False:
            print('Hello World!')
        'Predicts the target probability of an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n\n        Returns\n        -------\n        Predicted target values for each row of `X`.\n\n        '
        utils.validation.check_is_fitted(self, attributes='instance_')
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')

        def reshape_probas(y_pred):
            if False:
                print('Hello World!')
            return [y_pred.get(c, 0) for c in self.classes_]
        y_pred = np.empty(shape=(len(X), len(self.classes_)))
        for (i, (x, _)) in enumerate(stream.iter_array(X)):
            y_pred[i] = reshape_probas(self.instance_.predict_proba_one(x))
        return y_pred

    def predict(self, X):
        if False:
            return 10
        'Predicts the target of an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n\n        Returns\n        -------\n        Predicted target values for each row of `X`.\n\n        '
        utils.validation.check_is_fitted(self, attributes='instance_')
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        y_pred = [None] * len(X)
        for (i, (x, _)) in enumerate(stream.iter_array(X)):
            y_pred[i] = self.instance_.predict_one(x)
        y_pred = np.asarray(y_pred)
        if hasattr(self, 'label_encoder_'):
            y_pred = self.label_encoder_.inverse_transform(y_pred.astype(int))
        return y_pred

class River2SKLTransformer(River2SKLBase, sklearn_base.TransformerMixin):
    """Compatibility layer from River to scikit-learn for transformation.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Transformer):
        if False:
            i = 10
            return i + 15
        if not isinstance(river_estimator, base.Transformer):
            raise ValueError('estimator is not a Transformer')
        self.river_estimator = river_estimator

    def _partial_fit(self, X, y):
        if False:
            i = 10
            return i + 15
        if y is None:
            X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        else:
            (X, y) = utils.check_X_y(X, y, **SKLEARN_INPUT_X_PARAMS, **SKLEARN_INPUT_Y_PARAMS)
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        self.n_features_in_ = X.shape[1]
        if not hasattr(self, 'instance_'):
            self.instance_ = copy.deepcopy(self.river_estimator)
        if isinstance(self.instance_, base.SupervisedTransformer):
            for (x, yi) in STREAM_METHODS[type(X)](X, y):
                self.instance_.learn_one(x, yi)
        else:
            for (x, _) in STREAM_METHODS[type(X)](X):
                self.instance_.learn_one(x)
        return self

    def fit(self, X, y=None):
        if False:
            return 10
        'Fits to an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        for attr in ('instance_', 'n_features_in_'):
            self.__dict__.pop(attr, None)
        return self._partial_fit(X, y)

    def partial_fit(self, X, y=None):
        if False:
            while True:
                i = 10
        'Fits incrementally on a portion of a dataset.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        return self._partial_fit(X, y)

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predicts the target of an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features)\n\n        Returns\n        -------\n        Transformed output.\n\n        '
        utils.validation.check_is_fitted(self, attributes='instance_')
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        X_trans = [None] * len(X)
        for (i, (x, _)) in enumerate(STREAM_METHODS[type(X)](X)):
            X_trans[i] = list(self.instance_.transform_one(x).values())
        return np.asarray(X_trans)

class River2SKLClusterer(River2SKLBase, sklearn_base.ClusterMixin):
    """Compatibility layer from River to scikit-learn for clustering.

    Parameters
    ----------
    river_estimator

    """

    def __init__(self, river_estimator: base.Clusterer):
        if False:
            return 10
        if not isinstance(river_estimator, base.Clusterer):
            raise ValueError('estimator is not a Clusterer')
        self.river_estimator = river_estimator

    def _partial_fit(self, X, y):
        if False:
            while True:
                i = 10
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        self.n_features_in_ = X.shape[1]
        if not hasattr(self, 'instance_'):
            self.instance_ = copy.deepcopy(self.river_estimator)
        self.labels_ = np.empty(len(X), dtype=np.int32)
        for (i, (x, _)) in enumerate(STREAM_METHODS[type(X)](X)):
            label = self.instance_.learn_one(x).predict_one(x)
            self.labels_[i] = label
        return self

    def fit(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Fits to an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        for attr in ('instance_', 'n_features_in_'):
            self.__dict__.pop(attr, None)
        return self._partial_fit(X, y)

    def partial_fit(self, X, y):
        if False:
            return 10
        'Fits incrementally on a portion of a dataset.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n        y\n            array-like of shape n_samples.\n\n        Returns\n        -------\n        self\n\n        '
        return self._partial_fit(X, y)

    def predict(self, X):
        if False:
            print('Hello World!')
        'Predicts the target of an entire dataset contained in memory.\n\n        Parameters\n        ----------\n        X\n            array-like of shape (n_samples, n_features).\n\n        Returns\n        -------\n        Transformed output.\n\n        '
        utils.validation.check_is_fitted(self, attributes='instance_')
        X = utils.check_array(X, **SKLEARN_INPUT_X_PARAMS)
        y_pred = np.empty(len(X), dtype=np.int32)
        for (i, (x, _)) in enumerate(STREAM_METHODS[type(X)](X)):
            y_pred[i] = self.instance_.predict_one(x)
        return y_pred