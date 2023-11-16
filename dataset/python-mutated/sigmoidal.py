import numpy as np

class SigmoidalFeature(object):
    """
    Sigmoidal features

    1 / (1 + exp((m - x) @ c)
    """

    def __init__(self, mean, coef=1):
        if False:
            print('Hello World!')
        '\n        construct sigmoidal features\n\n        Parameters\n        ----------\n        mean : (n_features, ndim) or (n_features,) ndarray\n            center of sigmoid function\n        coef : (ndim,) ndarray or int or float\n            coefficient to be multplied with the distance\n        '
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        if isinstance(coef, int) or isinstance(coef, float):
            if np.size(mean, 1) == 1:
                coef = np.array([coef])
            else:
                raise ValueError('mismatch of dimension')
        else:
            assert coef.ndim == 1
            assert np.size(mean, 1) == len(coef)
        self.mean = mean
        self.coef = coef

    def _sigmoid(self, x, mean):
        if False:
            print('Hello World!')
        return np.tanh((x - mean) @ self.coef * 0.5) * 0.5 + 0.5

    def transform(self, x):
        if False:
            return 10
        '\n        transform input array with sigmoidal features\n\n        Parameters\n        ----------\n        x : (sample_size, ndim) or (sample_size,) ndarray\n            input array\n\n        Returns\n        -------\n        output : (sample_size, n_features) ndarray\n            sigmoidal features\n        '
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        basis = [np.ones(len(x))]
        for m in self.mean:
            basis.append(self._sigmoid(x, m))
        return np.asarray(basis).transpose()