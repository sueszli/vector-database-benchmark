import numpy as np
from prml.kernel.kernel import Kernel

class RBF(Kernel):

    def __init__(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        construct Radial basis kernel function\n\n        Parameters\n        ----------\n        params : (ndim + 1,) ndarray\n            parameters of radial basis function\n\n        Attributes\n        ----------\n        ndim : int\n            dimension of expected input data\n        '
        assert params.ndim == 1
        self.params = params
        self.ndim = len(params) - 1

    def __call__(self, x, y, pairwise=True):
        if False:
            print('Hello World!')
        '\n        calculate radial basis function\n        k(x, y) = c0 * exp(-0.5 * c1 * (x1 - y1) ** 2 ...)\n\n        Parameters\n        ----------\n        x : ndarray [..., ndim]\n            input of this kernel function\n        y : ndarray [..., ndim]\n            another input\n\n        Returns\n        -------\n        output : ndarray\n            output of this radial basis function\n        '
        assert x.shape[-1] == self.ndim
        assert y.shape[-1] == self.ndim
        if pairwise:
            (x, y) = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        return self.params[0] * np.exp(-0.5 * np.sum(d, axis=-1))

    def derivatives(self, x, y, pairwise=True):
        if False:
            for i in range(10):
                print('nop')
        if pairwise:
            (x, y) = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        delta = np.exp(-0.5 * np.sum(d, axis=-1))
        deltas = -0.5 * (x - y) ** 2 * (delta * self.params[0])[:, :, None]
        return np.concatenate((np.expand_dims(delta, 0), deltas.T))

    def update_parameters(self, updates):
        if False:
            print('Hello World!')
        self.params += updates