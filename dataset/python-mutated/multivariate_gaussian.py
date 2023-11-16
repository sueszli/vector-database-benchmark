import numpy as np
from prml.rv.rv import RandomVariable

class MultivariateGaussian(RandomVariable):
    """
    The multivariate Gaussian distribution
    p(x|mu, cov)
    = exp{-0.5 * (x - mu)^T @ cov^-1 @ (x - mu)}
      / (2pi)^(D/2) / |cov|^0.5
    """

    def __init__(self, mu=None, cov=None, tau=None):
        if False:
            return 10
        super().__init__()
        self.mu = mu
        if cov is not None:
            self.cov = cov
        elif tau is not None:
            self.tau = tau
        else:
            self.cov = None
            self.tau = None

    @property
    def mu(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parameter['mu']

    @mu.setter
    def mu(self, mu):
        if False:
            i = 10
            return i + 15
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 1
            self.parameter['mu'] = mu
        else:
            assert mu is None
            self.parameter['mu'] = None

    @property
    def cov(self):
        if False:
            i = 10
            return i + 15
        return self.parameter['cov']

    @cov.setter
    def cov(self, cov):
        if False:
            while True:
                i = 10
        if isinstance(cov, np.ndarray):
            assert cov.ndim == 2
            self.tau_ = np.linalg.inv(cov)
            self.parameter['cov'] = cov
        else:
            assert cov is None
            self.tau_ = None
            self.parameter['cov'] = None

    @property
    def tau(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tau_

    @tau.setter
    def tau(self, tau):
        if False:
            return 10
        if isinstance(tau, np.ndarray):
            assert tau.ndim == 2
            self.parameter['cov'] = np.linalg.inv(tau)
            self.tau_ = tau
        else:
            assert tau is None
            self.tau_ = None
            self.parameter['cov'] = None

    @property
    def ndim(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self.mu, 'ndim'):
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self.mu, 'size'):
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self.mu, 'shape'):
            return self.mu.shape
        else:
            return None

    def _fit(self, X):
        if False:
            return 10
        self.mu = np.mean(X, axis=0)
        self.cov = np.atleast_2d(np.cov(X.T, bias=True))

    def _pdf(self, X):
        if False:
            i = 10
            return i + 15
        d = X - self.mu
        return np.exp(-0.5 * np.sum(d @ self.tau * d, axis=-1)) * np.sqrt(np.linalg.det(self.tau)) / np.power(2 * np.pi, 0.5 * self.size)

    def _draw(self, sample_size=1):
        if False:
            while True:
                i = 10
        return np.random.multivariate_normal(self.mu, self.cov, sample_size)