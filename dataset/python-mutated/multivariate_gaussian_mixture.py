import numpy as np
from prml.clustering import KMeans
from prml.rv.rv import RandomVariable

class MultivariateGaussianMixture(RandomVariable):
    """
    p(x|mu, L, pi(coef))
    = sum_k pi_k N(x|mu_k, L_k^-1)
    """

    def __init__(self, n_components, mu=None, cov=None, tau=None, coef=None):
        if False:
            i = 10
            return i + 15
        '\n        construct mixture of Gaussians\n\n        Parameters\n        ----------\n        n_components : int\n            number of gaussian component\n        mu : (n_components, ndim) np.ndarray\n            mean parameter of each gaussian component\n        cov : (n_components, ndim, ndim) np.ndarray\n            variance parameter of each gaussian component\n        tau : (n_components, ndim, ndim) np.ndarray\n            precision parameter of each gaussian component\n        coef : (n_components,) np.ndarray\n            mixing coefficients\n        '
        super().__init__()
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.mu = mu
        if cov is not None and tau is not None:
            raise ValueError('Cannot assign both cov and tau at a time')
        elif cov is not None:
            self.cov = cov
        elif tau is not None:
            self.tau = tau
        else:
            self.cov = None
            self.tau = None
        self.coef = coef

    @property
    def mu(self):
        if False:
            i = 10
            return i + 15
        return self.parameter['mu']

    @mu.setter
    def mu(self, mu):
        if False:
            while True:
                i = 10
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 2
            assert np.size(mu, 0) == self.n_components
            self.ndim = np.size(mu, 1)
            self.parameter['mu'] = mu
        elif mu is None:
            self.parameter['mu'] = None
        else:
            raise TypeError('mu must be either np.ndarray or None')

    @property
    def cov(self):
        if False:
            while True:
                i = 10
        return self.parameter['cov']

    @cov.setter
    def cov(self, cov):
        if False:
            i = 10
            return i + 15
        if isinstance(cov, np.ndarray):
            assert cov.shape == (self.n_components, self.ndim, self.ndim)
            self._tau = np.linalg.inv(cov)
            self.parameter['cov'] = cov
        elif cov is None:
            self.parameter['cov'] = None
            self._tau = None
        else:
            raise TypeError('cov must be either np.ndarray or None')

    @property
    def tau(self):
        if False:
            print('Hello World!')
        return self._tau

    @tau.setter
    def tau(self, tau):
        if False:
            while True:
                i = 10
        if isinstance(tau, np.ndarray):
            assert tau.shape == (self.n_components, self.ndim, self.ndim)
            self.parameter['cov'] = np.linalg.inv(tau)
            self._tau = tau
        elif tau is None:
            self.parameter['cov'] = None
            self._tau = None
        else:
            raise TypeError('tau must be either np.ndarray or None')

    @property
    def coef(self):
        if False:
            return 10
        return self.parameter['coef']

    @coef.setter
    def coef(self, coef):
        if False:
            i = 10
            return i + 15
        if isinstance(coef, np.ndarray):
            assert coef.ndim == 1
            if np.isnan(coef).any():
                self.parameter['coef'] = np.ones(self.n_components) / self.n_components
            elif not np.allclose(coef.sum(), 1):
                raise ValueError(f'sum of coef must be equal to 1 {coef}')
            self.parameter['coef'] = coef
        elif coef is None:
            self.parameter['coef'] = None
        else:
            raise TypeError('coef must be either np.ndarray or None')

    @property
    def shape(self):
        if False:
            return 10
        if hasattr(self.mu, 'shape'):
            return self.mu.shape[1:]
        else:
            return None

    def _gauss(self, X):
        if False:
            while True:
                i = 10
        d = X[:, None, :] - self.mu
        D_sq = np.sum(np.einsum('nki,kij->nkj', d, self.cov) * d, -1)
        return np.exp(-0.5 * D_sq) / np.sqrt(np.linalg.det(self.cov) * (2 * np.pi) ** self.ndim)

    def _fit(self, X):
        if False:
            i = 10
            return i + 15
        cov = np.cov(X.T)
        kmeans = KMeans(self.n_components)
        kmeans.fit(X)
        self.mu = kmeans.centers
        self.cov = np.array([cov for _ in range(self.n_components)])
        self.coef = np.ones(self.n_components) / self.n_components
        params = np.hstack((self.mu.ravel(), self.cov.ravel(), self.coef.ravel()))
        while True:
            stats = self._expectation(X)
            self._maximization(X, stats)
            new_params = np.hstack((self.mu.ravel(), self.cov.ravel(), self.coef.ravel()))
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        if False:
            while True:
                i = 10
        resps = self.coef * self._gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps

    def _maximization(self, X, resps):
        if False:
            return 10
        Nk = np.sum(resps, axis=0)
        self.coef = Nk / len(X)
        self.mu = (X.T @ resps / Nk).T
        d = X[:, None, :] - self.mu
        self.cov = np.einsum('nki,nkj->kij', d, d * resps[:, :, None]) / Nk[:, None, None]

    def joint_proba(self, X):
        if False:
            print('Hello World!')
        '\n        calculate joint probability p(X, Z)\n\n        Parameters\n        ----------\n        X : (sample_size, n_features) ndarray\n            input data\n\n        Returns\n        -------\n        joint_prob : (sample_size, n_components) ndarray\n            joint probability of input and component\n        '
        return self.coef * self._gauss(X)

    def _pdf(self, X):
        if False:
            for i in range(10):
                print('nop')
        joint_prob = self.coef * self._gauss(X)
        return np.sum(joint_prob, axis=-1)

    def classify(self, X):
        if False:
            while True:
                i = 10
        '\n        classify input\n        max_z p(z|x, theta)\n\n        Parameters\n        ----------\n        X : (sample_size, ndim) ndarray\n            input\n\n        Returns\n        -------\n        output : (sample_size,) ndarray\n            corresponding cluster index\n        '
        return np.argmax(self.classify_proba(X), axis=1)

    def classify_proba(self, X):
        if False:
            return 10
        '\n        posterior probability of cluster\n        p(z|x,theta)\n\n        Parameters\n        ----------\n        X : (sample_size, ndim) ndarray\n            input\n\n        Returns\n        -------\n        output : (sample_size, n_components) ndarray\n            posterior probability of cluster\n        '
        return self._expectation(X)