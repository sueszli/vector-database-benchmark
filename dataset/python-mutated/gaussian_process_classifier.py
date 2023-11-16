import numpy as np

class GaussianProcessClassifier(object):

    def __init__(self, kernel, noise_level=0.0001):
        if False:
            for i in range(10):
                print('nop')
        '\n        construct gaussian process classifier\n\n        Parameters\n        ----------\n        kernel\n            kernel function to be used to compute Gram matrix\n        noise_level : float\n            parameter to ensure the matrix to be positive\n        '
        self.kernel = kernel
        self.noise_level = noise_level

    def _sigmoid(self, a):
        if False:
            while True:
                i = 10
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t):
        if False:
            i = 10
            return i + 15
        if X.ndim == 1:
            X = X[:, None]
        self.X = X
        self.t = t
        Gram = self.kernel(X, X)
        self.covariance = Gram + np.eye(len(Gram)) * self.noise_level
        self.precision = np.linalg.inv(self.covariance)

    def predict(self, X):
        if False:
            print('Hello World!')
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(X, self.X)
        a_mean = K @ self.precision @ self.t
        return self._sigmoid(a_mean)