import numpy as np
from scipy.special import logsumexp
from prml.rv.rv import RandomVariable

class BernoulliMixture(RandomVariable):
    """
    p(x|pi,mu)
    = sum_k pi_k mu_k^x (1 - mu_k)^(1 - x)
    """

    def __init__(self, n_components=3, mu=None, coef=None):
        if False:
            return 10
        '\n        construct mixture of Bernoulli\n\n        Parameters\n        ----------\n        n_components : int\n            number of bernoulli component\n        mu : (n_components, ndim) np.ndarray\n            probability of value 1 for each component\n        coef : (n_components,) np.ndarray\n            mixing coefficients\n        '
        super().__init__()
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.mu = mu
        self.coef = coef

    @property
    def mu(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parameter['mu']

    @mu.setter
    def mu(self, mu):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 2
            assert np.size(mu, 0) == self.n_components
            assert (mu >= 0.0).all() and (mu <= 1.0).all()
            self.ndim = np.size(mu, 1)
            self.parameter['mu'] = mu
        else:
            assert mu is None
            self.parameter['mu'] = None

    @property
    def coef(self):
        if False:
            return 10
        return self.parameter['coef']

    @coef.setter
    def coef(self, coef):
        if False:
            return 10
        if isinstance(coef, np.ndarray):
            assert coef.ndim == 1
            assert np.allclose(coef.sum(), 1)
            self.parameter['coef'] = coef
        else:
            assert coef is None
            self.parameter['coef'] = np.ones(self.n_components) / self.n_components

    def _log_bernoulli(self, X):
        if False:
            while True:
                i = 10
        np.clip(self.mu, 1e-10, 1 - 1e-10, out=self.mu)
        return (X[:, None, :] * np.log(self.mu) + (1 - X[:, None, :]) * np.log(1 - self.mu)).sum(axis=-1)

    def _fit(self, X):
        if False:
            while True:
                i = 10
        self.mu = np.random.uniform(0.25, 0.75, size=(self.n_components, np.size(X, 1)))
        params = np.hstack((self.mu.ravel(), self.coef.ravel()))
        while True:
            resp = self._expectation(X)
            self._maximization(X, resp)
            new_params = np.hstack((self.mu.ravel(), self.coef.ravel()))
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        if False:
            while True:
                i = 10
        log_resps = np.log(self.coef) + self._log_bernoulli(X)
        log_resps -= logsumexp(log_resps, axis=-1)[:, None]
        resps = np.exp(log_resps)
        return resps

    def _maximization(self, X, resp):
        if False:
            return 10
        Nk = np.sum(resp, axis=0)
        self.coef = Nk / len(X)
        self.mu = (X.T @ resp / Nk).T

    def classify(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        classify input\n        max_z p(z|x, theta)\n\n        Parameters\n        ----------\n        X : (sample_size, ndim) ndarray\n            input\n\n        Returns\n        -------\n        output : (sample_size,) ndarray\n            corresponding cluster index\n        '
        return np.argmax(self.classify_proba(X), axis=1)

    def classfiy_proba(self, X):
        if False:
            i = 10
            return i + 15
        '\n        posterior probability of cluster\n        p(z|x,theta)\n\n        Parameters\n        ----------\n        X : (sample_size, ndim) ndarray\n            input\n\n        Returns\n        -------\n        output : (sample_size, n_components) ndarray\n            posterior probability of cluster\n        '
        return self._expectation(X)