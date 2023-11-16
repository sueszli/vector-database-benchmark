"""Contains simple models used in checks."""
import numpy as np
__all__ = ['PerfectModel', 'RandomModel', 'ClassificationUniformModel', 'RegressionUniformModel']

def create_proba_result(predictions, classes):
    if False:
        while True:
            i = 10

    def prediction_to_proba(y_pred):
        if False:
            while True:
                i = 10
        proba = np.zeros(len(classes))
        proba[classes.index(y_pred)] = 1
        return proba
    return np.apply_along_axis(prediction_to_proba, axis=1, arr=predictions.reshape(-1, 1))

class PerfectModel:
    """Model used to perfectly predict from given series of labels."""

    def __init__(self):
        if False:
            return 10
        self.labels = None

    def fit(self, X, y):
        if False:
            return 10
        'Fit model.'
        self.labels = y

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Predict on given X.'
        return self.labels.to_numpy()

    def predict_proba(self, X):
        if False:
            return 10
        'Predict proba for given X.'
        classes = sorted(self.labels.unique().tolist())
        predictions = self.predict(X)
        return create_proba_result(predictions, classes)

class RandomModel:
    """Model used to randomly predict from given series of labels."""

    def __init__(self, random_state: int=42):
        if False:
            print('Hello World!')
        self.labels = None
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        'Fit model.'
        self.labels = y

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        'Predict on given X.'
        return np.random.choice(self.labels, X.shape[0])

    def predict_proba(self, X):
        if False:
            i = 10
            return i + 15
        'Predict proba for given X.'
        classes = sorted(self.labels.unique().tolist())
        predictions = self.predict(X)
        return create_proba_result(predictions, classes)

class ClassificationUniformModel:
    """Model that draws predictions uniformly at random from the list of classes in y."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.unique_labels = None

    def fit(self, X, y):
        if False:
            return 10
        'Fit model.'
        self.unique_labels = np.unique(y)

    def predict(self, X):
        if False:
            return 10
        'Predict on given X.'
        return np.random.choice(self.unique_labels, X.shape[0])

    def predict_proba(self, X):
        if False:
            return 10
        'Predict proba for given X.'
        classes = sorted(self.unique_labels.tolist())
        predictions = self.predict(X)
        return create_proba_result(predictions, classes)

class RegressionUniformModel:
    """Model that draws predictions uniformly at random from the range of values in y."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.min_value = None
        self.max_value = None

    def fit(self, X, y):
        if False:
            print('Hello World!')
        'Fit model.'
        self.min_value = y.min()
        self.max_value = y.max()

    def predict(self, X):
        if False:
            print('Hello World!')
        'Predict on given X.'
        return np.random.uniform(self.min_value, self.max_value, X.shape[0])