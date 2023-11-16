import numpy as np

class MeanCenterer(object):
    """Column centering of vectors and matrices.

    Attributes
    -----------
    col_means : numpy.ndarray [n_columns]
        NumPy array storing the mean values for centering after fitting
        the MeanCenterer object.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/preprocessing/MeanCenterer/

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Centers a NumPy array.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Array of data vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        --------\n        X_tr : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            A copy of the input array with the columns centered.\n\n        '
        if not hasattr(self, 'col_means'):
            raise AttributeError('MeanCenterer has not been fitted, yet.')
        X_tr = np.copy(self._get_array(X))
        X_tr = np.apply_along_axis(func1d=lambda x: x - self.col_means, axis=1, arr=X_tr)
        return X_tr

    def fit(self, X):
        if False:
            print('Hello World!')
        'Gets the column means for mean centering.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Array of data vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        --------\n        self\n        '
        self.col_means = self._get_array(X).mean(axis=0)
        return self

    def fit_transform(self, X):
        if False:
            i = 10
            return i + 15
        'Fits and transforms an arry.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            Array of data vectors, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        Returns\n        --------\n        X_tr : {array-like, sparse matrix}, shape = [n_samples, n_features]\n            A copy of the input array with the columns centered.\n        '
        self.fit(X)
        return self.transform(X)

    def _get_array(self, X):
        if False:
            i = 10
            return i + 15
        if isinstance(X, list):
            X_fl = np.asarray(X, dtype='float')[:, None]
        else:
            X_fl = X.astype('float')
        return X_fl