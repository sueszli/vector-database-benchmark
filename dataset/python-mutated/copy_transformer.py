import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator

class CopyTransformer(BaseEstimator):
    """Transformer that returns a copy of the input array

    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/preprocessing/CopyTransformer/

    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def transform(self, X, y=None):
        if False:
            while True:
                i = 10
        'Return a copy of the input array.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples] (default: None)\n\n        Returns\n        ---------\n        X_copy : copy of the input X array.\n\n        '
        if isinstance(X, list):
            return np.asarray(X)
        elif isinstance(X, np.ndarray) or issparse(X):
            return X.copy()
        else:
            raise ValueError('X must be a list or NumPy array or SciPy sparse array. Found %s' % type(X))

    def fit_transform(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the input array.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples] (default: None)\n\n        Returns\n        ---------\n        X_copy : copy of the input X array.\n\n        '
        return self.transform(X)

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        'Mock method. Does nothing.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples] (default: None)\n\n        Returns\n        ---------\n        self\n\n        '
        return self