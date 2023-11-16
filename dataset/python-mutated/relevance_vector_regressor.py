import numpy as np

class RelevanceVectorRegressor(object):

    def __init__(self, kernel, alpha=1.0, beta=1.0):
        if False:
            return 10
        '\n        construct relevance vector regressor\n\n        Parameters\n        ----------\n        kernel : Kernel\n            kernel function to compute components of feature vectors\n        alpha : float\n            initial precision of prior weight distribution\n        beta : float\n            precision of observation\n        '
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, t, iter_max=1000):
        if False:
            return 10
        '\n        maximize evidence with respect to hyperparameter\n\n        Parameters\n        ----------\n        X : (sample_size, n_features) ndarray\n            input\n        t : (sample_size,) ndarray\n            corresponding target\n        iter_max : int\n            maximum number of iterations\n\n        Attributes\n        -------\n        X : (N, n_features) ndarray\n            relevance vector\n        t : (N,) ndarray\n            corresponding target\n        alpha : (N,) ndarray\n            hyperparameter for each weight or training sample\n        cov : (N, N) ndarray\n            covariance matrix of weight\n        mean : (N,) ndarray\n            mean of each weight\n        '
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        assert t.ndim == 1
        N = len(t)
        Phi = self.kernel(X, X)
        self.alpha = np.zeros(N) + self.alpha
        for _ in range(iter_max):
            params = np.hstack([self.alpha, self.beta])
            precision = np.diag(self.alpha) + self.beta * Phi.T @ Phi
            covariance = np.linalg.inv(precision)
            mean = self.beta * covariance @ Phi.T @ t
            gamma = 1 - self.alpha * np.diag(covariance)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 10000000000.0, out=self.alpha)
            self.beta = (N - np.sum(gamma)) / np.sum((t - Phi.dot(mean)) ** 2)
            if np.allclose(params, np.hstack([self.alpha, self.beta])):
                break
        mask = self.alpha < 1000000000.0
        self.X = X[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel(self.X, self.X)
        precision = np.diag(self.alpha) + self.beta * Phi.T @ Phi
        self.covariance = np.linalg.inv(precision)
        self.mean = self.beta * self.covariance @ Phi.T @ self.t

    def predict(self, X, with_error=True):
        if False:
            print('Hello World!')
        '\n        predict output with this model\n\n        Parameters\n        ----------\n        X : (sample_size, n_features)\n            input\n        with_error : bool\n            if True, predict with standard deviation of the outputs\n\n        Returns\n        -------\n        mean : (sample_size,) ndarray\n            mean of predictive distribution\n        std : (sample_size,) ndarray\n            standard deviation of predictive distribution\n        '
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        mean = phi @ self.mean
        if with_error:
            var = 1 / self.beta + np.sum(phi @ self.covariance * phi, axis=1)
            return (mean, np.sqrt(var))
        return mean