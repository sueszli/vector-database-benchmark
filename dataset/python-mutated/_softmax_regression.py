import numpy as np
from prml.linear._classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer

class SoftmaxRegression(Classifier):
    """Softmax regression model.

    aka
    multinomial logistic regression,
    multiclass logistic regression,
    maximum entropy classifier.

    y = softmax(X @ W)
    t ~ Categorical(t|y)
    """

    @staticmethod
    def _softmax(a):
        if False:
            while True:
                i = 10
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, max_iter: int=100, learning_rate: float=0.1):
        if False:
            return 10
        'Maximum likelihood estimation of the parameter.\n\n        Parameters\n        ----------\n        X : (N, D) np.ndarray\n            training independent variable\n        t : (N,) or (N, K) np.ndarray\n            training dependent variable\n            in class index or one-of-k encoding\n        max_iter : int, optional\n            maximum number of iteration (the default is 100)\n        learning_rate : float, optional\n            learning rate of gradient descent (the default is 0.1)\n        '
        if y_train.ndim == 1:
            y_train = LabelTransformer().encode(y_train)
        self.n_classes = np.size(y_train, 1)
        w = np.zeros((np.size(x_train, 1), self.n_classes))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._softmax(x_train @ w)
            grad = x_train.T @ (y - y_train)
            w -= learning_rate * grad
            if np.allclose(w, w_prev):
                break
        self.w = w

    def proba(self, x: np.ndarray):
        if False:
            return 10
        'Return probability of input belonging each class.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            Input independent variable (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            probability of each class (N, K)\n        '
        return self._softmax(x @ self.w)

    def classify(self, x: np.ndarray):
        if False:
            return 10
        'Classify input data.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            independent variable to be classified (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            class index for each input (N,)\n        '
        return np.argmax(self.proba(x), axis=-1)