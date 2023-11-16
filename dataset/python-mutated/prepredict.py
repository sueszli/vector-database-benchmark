"""
PrePredict estimator allows Yellowbrick to work with results produced by an estimator
prior to the visual diagnostic workflow, particularly for inferences that require
extensive time or compute resources.
"""
import pathlib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from yellowbrick.contrib.wrapper import CLASSIFIER, CLUSTERER, REGRESSOR

class PrePredict(BaseEstimator):
    """
    The Passthrough estimator allows users to specify pre-predicted results to
    Yellowbrick without the need to input the original estimator. Note that Yellowbrick
    often uses the learned attributes of the estimator to produce rich visual
    diagnostics, so this estimator may not work for all Yellowbrick visualizers.

    The passthrough estimator can accept data either in memory as a numpy array or it
    can accept a string, which it interprets as a path on disk to load the data from.

    Currently passthrough does not support predict_proba or decision_function methods,
    which it could if it was passed predicted data as 2D array instead of a 1D array.

    Parameters
    ----------
    data : array-like, func, or file-like object, string, or pathlib.Path
        The predicted values wrapped by the estimator and returned on predict() and
        used by the score function. The default expectation is that data is a 1D numpy
        array of y_hat or y_pred values produced by some other estimator. Data can also
        be a func, which is called and returned, or a file-like object, string, or
        pathlib.Path at which point the data is loaded from disk using ``np.load``.

    estimator_type : str, optional
        One of "classifier", "regressor", "clusterer", "DensityEstimator", or
        "outlier_detector" that allows the contrib estimator to pass the scikit-learn
        ``is_classifier``, etc. functions. If not specified, the Yellowbrick visualizer
        you're trying to use may error.
    """

    def __init__(self, data, estimator_type=None):
        if False:
            while True:
                i = 10
        self.data = data
        self._estimator_type = estimator_type

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        '\n        Fit is a no-op, simply returning self per the scikit-learn API.\n        '
        return self

    def predict(self, X):
        if False:
            print('Hello World!')
        '\n        Predict returns the embedded data but does not perform any checks on the\n        validity of X (e.g. that it has the same shape as the internal data).\n        '
        return self._load()

    def score(self, X, y=None):
        if False:
            return 10
        '\n        Score uses an appropriate metric for the estimator type and compares the input\n        y values with the pre-predicted values.\n        '
        if self._estimator_type == CLASSIFIER:
            return accuracy_score(y, self._load())
        if self._estimator_type == REGRESSOR:
            return r2_score(y, self._load())
        if self._estimator_type == CLUSTERER:
            labels = y if y is not None else self._load()
            return silhouette_score(X, labels)
        return np.nan

    def _load(self):
        if False:
            while True:
                i = 10
        '\n        Loads the data by performing type checking to determine if data is a callable\n        whose result needs to be returned, or an argument that supports from disk\n        loading. If neither of these things, then assumes the data is array-like and\n        returns it directly.\n        '
        if callable(self.data):
            return self.data()
        if hasattr(self.data, 'read') or isinstance(self.data, (str, pathlib.Path)):
            return np.load(self.data)
        return self.data