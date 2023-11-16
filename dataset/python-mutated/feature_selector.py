import logging
_logger = logging.getLogger(__name__)

class FeatureSelector:

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.selected_features_ = None
        self.X = None
        self.y = None

    def fit(self, X, y, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit the training data to FeatureSelector\n\n        Paramters\n        ---------\n        X : array-like numpy matrix\n            The training input samples, which shape is [n_samples, n_features].\n        y: array-like numpy matrix\n            The target values (class labels in classification, real numbers in\n            regression). Which shape is [n_samples].\n        '
        self.X = X
        self.y = y

    def get_selected_features(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit the training data to FeatureSelector\n\n        Returns\n        -------\n        list :\n                Return the index of imprtant feature.\n        '
        return self.selected_features_