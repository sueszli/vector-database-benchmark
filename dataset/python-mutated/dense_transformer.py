from scipy.sparse import issparse
from sklearn.base import BaseEstimator

class DenseTransformer(BaseEstimator):
    """
    Convert a sparse array into a dense array.

    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/preprocessing/DenseTransformer/

    """

    def __init__(self, return_copy=True):
        if False:
            for i in range(10):
                print('nop')
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        if False:
            print('Hello World!')
        'Return a dense version of the input array.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples] (default: None)\n\n        Returns\n        ---------\n        X_dense : dense version of the input X array.\n\n        '
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        'Mock method. Does nothing.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples] (default: None)\n\n        Returns\n        ---------\n        self\n\n        '
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a dense version of the input array.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples] (default: None)\n\n        Returns\n        ---------\n        X_dense : dense version of the input X array.\n\n        '
        return self.transform(X=X, y=y)