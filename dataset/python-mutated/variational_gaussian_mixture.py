import numpy as np
from scipy.special import digamma, gamma, logsumexp
from prml.rv.rv import RandomVariable

class VariationalGaussianMixture(RandomVariable):

    def __init__(self, n_components=1, alpha0=None, m0=None, W0=1.0, dof0=None, beta0=1.0):
        if False:
            while True:
                i = 10
        '\n        construct variational gaussian mixture model\n        Parameters\n        ----------\n        n_components : int\n            maximum numnber of gaussian components\n        alpha0 : float\n            parameter of prior dirichlet distribution\n        m0 : float\n            mean parameter of prior gaussian distribution\n        W0 : float\n            mean of the prior Wishart distribution\n        dof0 : float\n            number of degrees of freedom of the prior Wishart distribution\n        beta0 : float\n            prior on the precision distribution\n        '
        super().__init__()
        self.n_components = n_components
        if alpha0 is None:
            self.alpha0 = 1 / n_components
        else:
            self.alpha0 = alpha0
        self.m0 = m0
        self.W0 = W0
        self.dof0 = dof0
        self.beta0 = beta0

    def _init_params(self, X):
        if False:
            return 10
        (sample_size, self.ndim) = X.shape
        self.alpha0 = np.ones(self.n_components) * self.alpha0
        if self.m0 is None:
            self.m0 = np.mean(X, axis=0)
        else:
            self.m0 = np.zeros(self.ndim) + self.m0
        self.W0 = np.eye(self.ndim) * self.W0
        if self.dof0 is None:
            self.dof0 = self.ndim
        self.component_size = sample_size / self.n_components + np.zeros(self.n_components)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(sample_size, self.n_components, replace=False)
        self.mu = X[indices]
        self.W = np.tile(self.W0, (self.n_components, 1, 1))
        self.dof = self.dof0 + self.component_size

    @property
    def alpha(self):
        if False:
            i = 10
            return i + 15
        return self.parameter['alpha']

    @alpha.setter
    def alpha(self, alpha):
        if False:
            for i in range(10):
                print('nop')
        self.parameter['alpha'] = alpha

    @property
    def beta(self):
        if False:
            while True:
                i = 10
        return self.parameter['beta']

    @beta.setter
    def beta(self, beta):
        if False:
            for i in range(10):
                print('nop')
        self.parameter['beta'] = beta

    @property
    def mu(self):
        if False:
            i = 10
            return i + 15
        return self.parameter['mu']

    @mu.setter
    def mu(self, mu):
        if False:
            print('Hello World!')
        self.parameter['mu'] = mu

    @property
    def W(self):
        if False:
            while True:
                i = 10
        return self.parameter['W']

    @W.setter
    def W(self, W):
        if False:
            print('Hello World!')
        self.parameter['W'] = W

    @property
    def dof(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parameter['dof']

    @dof.setter
    def dof(self, dof):
        if False:
            while True:
                i = 10
        self.parameter['dof'] = dof

    def get_params(self):
        if False:
            return 10
        return (self.alpha, self.beta, self.mu, self.W, self.dof)

    def _fit(self, X, iter_max=100):
        if False:
            i = 10
            return i + 15
        self._init_params(X)
        for _ in range(iter_max):
            params = np.hstack([p.flatten() for p in self.get_params()])
            r = self._variational_expectation(X)
            self._variational_maximization(X, r)
            if np.allclose(params, np.hstack([p.flatten() for p in self.get_params()])):
                break

    def _variational_expectation(self, X):
        if False:
            i = 10
            return i + 15
        d = X[:, None, :] - self.mu
        maha_sq = -0.5 * (self.ndim / self.beta + self.dof * np.sum(np.einsum('kij,nkj->nki', self.W, d) * d, axis=-1))
        ln_pi = digamma(self.alpha) - digamma(self.alpha.sum())
        ln_Lambda = digamma(0.5 * (self.dof - np.arange(self.ndim)[:, None])).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W)[1]
        ln_r = ln_pi + 0.5 * ln_Lambda + maha_sq
        ln_r -= logsumexp(ln_r, axis=-1)[:, None]
        r = np.exp(ln_r)
        return r

    def _variational_maximization(self, X, r):
        if False:
            for i in range(10):
                print('nop')
        self.component_size = r.sum(axis=0)
        Xm = (X.T.dot(r) / self.component_size).T
        d = X[:, None, :] - Xm
        S = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / self.component_size[:, None, None]
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.mu = (self.beta0 * self.m0 + self.component_size[:, None] * Xm) / self.beta[:, None]
        d = Xm - self.m0
        self.W = np.linalg.inv(np.linalg.inv(self.W0) + (self.component_size * S.T).T + (self.beta0 * self.component_size * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.component_size)).T)
        self.dof = self.dof0 + self.component_size

    def classify(self, X):
        if False:
            i = 10
            return i + 15
        '\n        index of highest posterior of the latent variable\n        Parameters\n        ----------\n        X : (sample_size, ndim) ndarray\n            input\n        Returns\n        -------\n        output : (sample_size, n_components) ndarray\n            index of maximum posterior of the latent variable\n        '
        return np.argmax(self._variational_expectation(X), 1)

    def classify_proba(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        compute posterior of the latent variable\n        Parameters\n        ----------\n        X : (sample_size, ndim) ndarray\n            input\n        Returns\n        -------\n        output : (sample_size, n_components) ndarray\n            posterior of the latent variable\n        '
        return self._variational_expectation(X)

    def student_t(self, X):
        if False:
            for i in range(10):
                print('nop')
        nu = self.dof + 1 - self.ndim
        L = (nu * self.beta * self.W.T / (1 + self.beta)).T
        d = X[:, None, :] - self.mu
        maha_sq = np.sum(np.einsum('nki,kij->nkj', d, L) * d, axis=-1)
        return gamma(0.5 * (nu + self.ndim)) * np.sqrt(np.linalg.det(L)) * (1 + maha_sq / nu) ** (-0.5 * (nu + self.ndim)) / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * self.ndim))

    def _pdf(self, X):
        if False:
            return 10
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()