"""
Feature agglomeration. Base classes and functions for performing feature
agglomeration.
"""
import warnings
import numpy as np
from scipy.sparse import issparse
from ..base import TransformerMixin
from ..utils import metadata_routing
from ..utils.validation import check_is_fitted

class AgglomerationTransform(TransformerMixin):
    """
    A class for feature agglomeration via the transform interface.
    """
    __metadata_request__inverse_transform = {'Xred': metadata_routing.UNUSED}

    def transform(self, X):
        if False:
            return 10
        '\n        Transform a new matrix using the built clustering.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features) or                 (n_samples, n_samples)\n            A M by N array of M observations in N dimensions or a length\n            M array of M one-dimensional observations.\n\n        Returns\n        -------\n        Y : ndarray of shape (n_samples, n_clusters) or (n_clusters,)\n            The pooled values for each feature cluster.\n        '
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        if self.pooling_func == np.mean and (not issparse(X)):
            size = np.bincount(self.labels_)
            n_samples = X.shape[0]
            nX = np.array([np.bincount(self.labels_, X[i, :]) / size for i in range(n_samples)])
        else:
            nX = [self.pooling_func(X[:, self.labels_ == l], axis=1) for l in np.unique(self.labels_)]
            nX = np.array(nX).T
        return nX

    def inverse_transform(self, Xt=None, Xred=None):
        if False:
            i = 10
            return i + 15
        '\n        Inverse the transformation and return a vector of size `n_features`.\n\n        Parameters\n        ----------\n        Xt : array-like of shape (n_samples, n_clusters) or (n_clusters,)\n            The values to be assigned to each cluster of samples.\n\n        Xred : deprecated\n            Use `Xt` instead.\n\n            .. deprecated:: 1.3\n\n        Returns\n        -------\n        X : ndarray of shape (n_samples, n_features) or (n_features,)\n            A vector of size `n_samples` with the values of `Xred` assigned to\n            each of the cluster of samples.\n        '
        if Xt is None and Xred is None:
            raise TypeError('Missing required positional argument: Xt')
        if Xred is not None and Xt is not None:
            raise ValueError('Please provide only `Xt`, and not `Xred`.')
        if Xred is not None:
            warnings.warn('Input argument `Xred` was renamed to `Xt` in v1.3 and will be removed in v1.5.', FutureWarning)
            Xt = Xred
        check_is_fitted(self)
        (unil, inverse) = np.unique(self.labels_, return_inverse=True)
        return Xt[..., inverse]