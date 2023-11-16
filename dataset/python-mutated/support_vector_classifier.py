import numpy as np

class SupportVectorClassifier(object):

    def __init__(self, kernel, C=np.Inf):
        if False:
            print('Hello World!')
        '\n        construct support vector classifier\n\n        Parameters\n        ----------\n        kernel : Kernel\n            kernel function to compute inner products\n        C : float\n            penalty of misclassification\n        '
        self.kernel = kernel
        self.C = C

    def fit(self, X: np.ndarray, t: np.ndarray, tol: float=1e-08):
        if False:
            while True:
                i = 10
        '\n        estimate support vectors and their parameters\n\n        Parameters\n        ----------\n        X : (N, D) np.ndarray\n            training independent variable\n        t : (N,) np.ndarray\n            training dependent variable\n            binary -1 or 1\n        tol : float, optional\n            numerical tolerance (the default is 1e-8)\n        '
        N = len(t)
        coef = np.zeros(N)
        grad = np.ones(N)
        Gram = self.kernel(X, X)
        while True:
            tg = t * grad
            mask_up = (t == 1) & (coef < self.C - tol)
            mask_up |= (t == -1) & (coef > tol)
            mask_down = (t == -1) & (coef < self.C - tol)
            mask_down |= (t == 1) & (coef > tol)
            i = np.where(mask_up)[0][np.argmax(tg[mask_up])]
            j = np.where(mask_down)[0][np.argmin(tg[mask_down])]
            if tg[i] < tg[j] + tol:
                self.b = 0.5 * (tg[i] + tg[j])
                break
            else:
                A = self.C - coef[i] if t[i] == 1 else coef[i]
                B = coef[j] if t[j] == 1 else self.C - coef[j]
                direction = (tg[i] - tg[j]) / (Gram[i, i] - 2 * Gram[i, j] + Gram[j, j])
                direction = min(A, B, direction)
                coef[i] += direction * t[i]
                coef[j] -= direction * t[j]
                grad -= direction * t * (Gram[i] - Gram[j])
        support_mask = coef > tol
        self.a = coef[support_mask]
        self.X = X[support_mask]
        self.t = t[support_mask]

    def lagrangian_function(self):
        if False:
            for i in range(10):
                print('nop')
        return np.sum(self.a) - self.a @ (self.t * self.t[:, None] * self.kernel(self.X, self.X)) @ self.a

    def predict(self, x):
        if False:
            print('Hello World!')
        '\n        predict labels of the input\n\n        Parameters\n        ----------\n        x : (sample_size, n_features) ndarray\n            input\n\n        Returns\n        -------\n        label : (sample_size,) ndarray\n            predicted labels\n        '
        y = self.distance(x)
        label = np.sign(y)
        return label

    def distance(self, x):
        if False:
            print('Hello World!')
        '\n        calculate distance from the decision boundary\n\n        Parameters\n        ----------\n        x : (sample_size, n_features) ndarray\n            input\n\n        Returns\n        -------\n        distance : (sample_size,) ndarray\n            distance from the boundary\n        '
        distance = np.sum(self.a * self.t * self.kernel(x, self.X), axis=-1) + self.b
        return distance