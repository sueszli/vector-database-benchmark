import numpy as np

class GaussianProcessRegressor(object):

    def __init__(self, kernel, beta=1.0):
        if False:
            while True:
                i = 10
        '\n        construct gaussian process regressor\n\n        Parameters\n        ----------\n        kernel\n            kernel function\n        beta : float\n            precision parameter of observation noise\n        '
        self.kernel = kernel
        self.beta = beta

    def fit(self, X, t, iter_max=0, learning_rate=0.1):
        if False:
            i = 10
            return i + 15
        '\n        maximum likelihood estimation of parameters in kernel function\n\n        Parameters\n        ----------\n        X : ndarray (sample_size, n_features)\n            input\n        t : ndarray (sample_size,)\n            corresponding target\n        iter_max : int\n            maximum number of iterations updating hyperparameters\n        learning_rate : float\n            updation coefficient\n\n        Attributes\n        ----------\n        covariance : ndarray (sample_size, sample_size)\n            variance covariance matrix of gaussian process\n        precision : ndarray (sample_size, sample_size)\n            precision matrix of gaussian process\n\n        Returns\n        -------\n        log_likelihood_list : list\n            list of log likelihood value at each iteration\n        '
        if X.ndim == 1:
            X = X[:, None]
        log_likelihood_list = [-np.Inf]
        self.X = X
        self.t = t
        I = np.eye(len(X))
        Gram = self.kernel(X, X)
        self.covariance = Gram + I / self.beta
        self.precision = np.linalg.inv(self.covariance)
        for i in range(iter_max):
            gradients = self.kernel.derivatives(X, X)
            updates = np.array([-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
            for j in range(iter_max):
                self.kernel.update_parameters(learning_rate * updates)
                Gram = self.kernel(X, X)
                self.covariance = Gram + I / self.beta
                self.precision = np.linalg.inv(self.covariance)
                log_like = self.log_likelihood()
                if log_like > log_likelihood_list[-1]:
                    log_likelihood_list.append(log_like)
                    break
                else:
                    self.kernel.update_parameters(-learning_rate * updates)
                    learning_rate *= 0.9
        log_likelihood_list.pop(0)
        return log_likelihood_list

    def log_likelihood(self):
        if False:
            while True:
                i = 10
        return -0.5 * (np.linalg.slogdet(self.covariance)[1] + self.t @ self.precision @ self.t + len(self.t) * np.log(2 * np.pi))

    def predict(self, X, with_error=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        mean of the gaussian process\n\n        Parameters\n        ----------\n        X : ndarray (sample_size, n_features)\n            input\n\n        Returns\n        -------\n        mean : ndarray (sample_size,)\n            predictions of corresponding inputs\n        '
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(X, self.X)
        mean = K @ self.precision @ self.t
        if with_error:
            var = self.kernel(X, X, False) + 1 / self.beta - np.sum(K @ self.precision * K, axis=1)
            return (mean.ravel(), np.sqrt(var.ravel()))
        return mean