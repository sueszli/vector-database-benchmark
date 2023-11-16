import numpy as np
from prml.linear._bayesian_regression import BayesianRegression

class EmpiricalBayesRegression(BayesianRegression):
    """Empirical Bayes Regression model.

    a.k.a.
    type 2 maximum likelihood,
    generalized maximum likelihood,
    evidence approximation

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    evidence function p(t|X,alpha,beta) = S p(t|w;X,beta)p(w|0;alpha) dw
    """

    def __init__(self, alpha: float=1.0, beta: float=1.0):
        if False:
            i = 10
            return i + 15
        'Initialize empirical bayesian linear regression model.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            Precision parameter of the prior, by default 1.\n        beta : float, optional\n            Precision parameter of the likelihood, by default 1.\n        '
        super().__init__(alpha, beta)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int=100):
        if False:
            i = 10
            return i + 15
        'Maximize of evidence function with respect to the hyperparameters.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training independent variable (N, D)\n        y_train : np.ndarray\n            training dependent variable (N,)\n        max_iter : int\n            maximum number of iteration\n        '
        xtx = x_train.T @ x_train
        eigenvalues = np.linalg.eigvalsh(xtx)
        eye = np.eye(np.size(x_train, 1))
        n = len(y_train)
        for _ in range(max_iter):
            params = [self.alpha, self.beta]
            w_precision = self.alpha * eye + self.beta * xtx
            w_mean = self.beta * np.linalg.solve(w_precision, x_train.T @ y_train)
            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            self.beta = float((n - gamma) / np.sum(np.square(y_train - x_train @ w_mean)))
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)

    def _log_prior(self, w):
        if False:
            for i in range(10):
                print('nop')
        return -0.5 * self.alpha * np.sum(w ** 2)

    def _log_likelihood(self, x, y, w):
        if False:
            return 10
        return -0.5 * self.beta * np.square(y - x @ w).sum()

    def _log_posterior(self, x, y, w):
        if False:
            print('Hello World!')
        return self._log_likelihood(x, y, w) + self._log_prior(w)

    def log_evidence(self, x: np.ndarray, y: np.ndarray):
        if False:
            while True:
                i = 10
        'Return logarithm or the evidence function.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            indenpendent variable (N, D)\n        y : np.ndarray\n            dependent variable (N,)\n        Returns\n        -------\n        float\n            log evidence\n        '
        n = len(y)
        d = np.size(x, 1)
        return 0.5 * (d * np.log(self.alpha) + n * np.log(self.beta) - np.linalg.slogdet(self.w_precision)[1] - d * np.log(2 * np.pi)) + self._log_posterior(x, y, self.w_mean)