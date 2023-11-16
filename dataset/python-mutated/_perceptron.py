import numpy as np
from prml.linear._classifier import Classifier

class Perceptron(Classifier):
    """Perceptron model."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_epoch: int=100):
        if False:
            print('Hello World!')
        'Fit perceptron model on given input pair.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training independent variable (N, D)\n        y_train : np.ndarray\n            training dependent variable (N,)\n            binary -1 or 1\n        max_epoch : int, optional\n            maximum number of epoch (the default is 100)\n        '
        self.w = np.zeros(np.size(x_train, 1))
        for _ in range(max_epoch):
            prediction = self.classify(x_train)
            error_indices = prediction != y_train
            x_error = x_train[error_indices]
            y_error = y_train[error_indices]
            idx = np.random.choice(len(x_error))
            self.w += x_error[idx] * y_error[idx]
            if (x_train @ self.w * y_train > 0).all():
                break

    def classify(self, x: np.ndarray):
        if False:
            i = 10
            return i + 15
        'Classify input data.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            independent variable to be classified (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            binary class (-1 or 1) for each input (N,)\n        '
        return np.sign(x @ self.w).astype(np.int)