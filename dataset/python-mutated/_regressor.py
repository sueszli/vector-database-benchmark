from time import time
import numpy as np

class _Regressor(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def _check_target_array(self, y, allowed=None):
        if False:
            i = 10
            return i + 15
        if not isinstance(y[0], (float, np.float_)):
            raise AttributeError('y must be a float array.\nFound %s' % y.dtype)

    def fit(self, X, y, init_params=True):
        if False:
            return 10
        'Learn model from training data.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n        y : array-like, shape = [n_samples]\n            Target values.\n        init_params : bool (default: True)\n            Re-initializes model parameters prior to fitting.\n            Set False to continue training with weights from\n            a previous model fitting.\n\n        Returns\n        -------\n        self : object\n\n        '
        self._is_fitted = False
        self._check_arrays(X=X, y=y)
        self._check_target_array(y)
        if hasattr(self, 'self.random_seed') and self.random_seed:
            self._rgen = np.random.RandomState(self.random_seed)
        self._init_time = time()
        self._fit(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self

    def predict(self, X):
        if False:
            print('Hello World!')
        'Predict targets from X.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Training vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        ----------\n        target_values : array-like, shape = [n_samples]\n          Predicted target values.\n\n        '
        self._check_arrays(X=X)
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)