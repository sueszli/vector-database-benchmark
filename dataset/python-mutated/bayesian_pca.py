import numpy as np
from prml.dimreduction.pca import PCA

class BayesianPCA(PCA):

    def fit(self, X, iter_max=100, initial='random'):
        if False:
            return 10
        '\n        empirical bayes estimation of pca parameters\n\n        Parameters\n        ----------\n        X : (sample_size, n_features) ndarray\n            input data\n        iter_max : int\n            maximum number of em steps\n\n        Returns\n        -------\n        mean : (n_features,) ndarray\n            sample mean fo the input data\n        W : (n_features, n_components) ndarray\n            projection matrix\n        var : float\n            variance of observation noise\n        '
        initial_list = ['random', 'eigen']
        self.mean = np.mean(X, axis=0)
        self.I = np.eye(self.n_components)
        if initial not in initial_list:
            print('availabel initializations are {}'.format(initial_list))
        if initial == 'random':
            self.W = np.eye(np.size(X, 1), self.n_components)
            self.var = 1.0
        elif initial == 'eigen':
            self.eigen(X)
        self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)
        for i in range(iter_max):
            W = np.copy(self.W)
            stats = self._expectation(X - self.mean)
            self._maximization(X - self.mean, *stats)
            self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)
            if np.allclose(W, self.W):
                break
        self.n_iter = i + 1

    def _maximization(self, X, Ez, Ezz):
        if False:
            while True:
                i = 10
        self.W = X.T @ Ez @ np.linalg.inv(np.sum(Ezz, axis=0) + self.var * np.diag(self.alpha))
        self.var = np.mean(np.mean(X ** 2, axis=-1) - 2 * np.mean(Ez @ self.W.T * X, axis=-1) + np.trace((Ezz @ self.W.T @ self.W).T) / len(self.mean))

    def maximize(self, D, Ez, Ezz):
        if False:
            return 10
        self.W = D.T.dot(Ez).dot(np.linalg.inv(np.sum(Ezz, axis=0) + self.var * np.diag(self.alpha)))
        self.var = np.mean(np.mean(D ** 2, axis=-1) - 2 * np.mean(Ez.dot(self.W.T) * D, axis=-1) + np.trace(Ezz.dot(self.W.T).dot(self.W).T) / self.ndim)