import numpy as np

class RelevanceVectorClassifier(object):

    def __init__(self, kernel, alpha=1.0):
        if False:
            print('Hello World!')
        '\n        construct relevance vector classifier\n\n        Parameters\n        ----------\n        kernel : Kernel\n            kernel function to compute components of feature vectors\n        alpha : float\n            initial precision of prior weight distribution\n        '
        self.kernel = kernel
        self.alpha = alpha

    def _sigmoid(self, a):
        if False:
            i = 10
            return i + 15
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _map_estimate(self, X, t, w, n_iter=10):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(n_iter):
            y = self._sigmoid(X @ w)
            g = X.T @ (y - t) + self.alpha * w
            H = X.T * y * (1 - y) @ X + np.diag(self.alpha)
            w -= np.linalg.solve(H, g)
        return (w, np.linalg.inv(H))

    def fit(self, X, t, iter_max=100):
        if False:
            while True:
                i = 10
        '\n        maximize evidence with respect ot hyperparameter\n\n        Parameters\n        ----------\n        X : (sample_size, n_features) ndarray\n            input\n        t : (sample_size,) ndarray\n            corresponding target\n        iter_max : int\n            maximum number of iterations\n\n        Attributes\n        ----------\n        X : (N, n_features) ndarray\n            relevance vector\n        t : (N,) ndarray\n            corresponding target\n        alpha : (N,) ndarray\n            hyperparameter for each weight or training sample\n        cov : (N, N) ndarray\n            covariance matrix of weight\n        mean : (N,) ndarray\n            mean of each weight\n        '
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        assert t.ndim == 1
        Phi = self.kernel(X, X)
        N = len(t)
        self.alpha = np.zeros(N) + self.alpha
        mean = np.zeros(N)
        for _ in range(iter_max):
            param = np.copy(self.alpha)
            (mean, cov) = self._map_estimate(Phi, t, mean, 10)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 10000000000.0, out=self.alpha)
            if np.allclose(param, self.alpha):
                break
        mask = self.alpha < 100000000.0
        self.X = X[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel(self.X, self.X)
        mean = mean[mask]
        (self.mean, self.covariance) = self._map_estimate(Phi, self.t, mean, 100)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        predict class label\n\n        Parameters\n        ----------\n        X : (sample_size, n_features)\n            input\n\n        Returns\n        -------\n        label : (sample_size,) ndarray\n            predicted label\n        '
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        label = (phi @ self.mean > 0).astype(np.int)
        return label

    def predict_proba(self, X):
        if False:
            i = 10
            return i + 15
        '\n        probability of input belonging class one\n\n        Parameters\n        ----------\n        X : (sample_size, n_features) ndarray\n            input\n\n        Returns\n        -------\n        proba : (sample_size,) ndarray\n            probability of predictive distribution p(C1|x)\n        '
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        mu_a = phi @ self.mean
        var_a = np.sum(phi @ self.covariance * phi, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))