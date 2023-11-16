import numpy as np
from prml.linear._classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer

class LeastSquaresClassifier(Classifier):
    """Least squares classifier model.

    X : (N, D)
    W : (D, K)
    y = argmax_k X @ W
    """

    def __init__(self, w: np.ndarray=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize least squares classifier model.\n\n        Parameters\n        ----------\n        w : np.ndarray, optional\n            Initial parameter, by default None\n        '
        self.w = w

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        if False:
            print('Hello World!')
        'Least squares fitting for classification.\n\n        Parameters\n        ----------\n        x_train : np.ndarray\n            training independent variable (N, D)\n        y_train : np.ndarray\n            training dependent variable\n            in class index (N,) or one-of-k coding (N,K)\n        '
        if y_train.ndim == 1:
            y_train = LabelTransformer().encode(y_train)
        self.w = np.linalg.pinv(x_train) @ y_train

    def classify(self, x: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        'Classify input data.\n\n        Parameters\n        ----------\n        x : np.ndarray\n            independent variable to be classified (N, D)\n\n        Returns\n        -------\n        np.ndarray\n            class index for each input (N,)\n        '
        return np.argmax(x @ self.w, axis=-1)