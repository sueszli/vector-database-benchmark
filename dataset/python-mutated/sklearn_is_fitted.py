"""
========================================
`__sklearn_is_fitted__` as Developer API
========================================

The `__sklearn_is_fitted__` method is a convention used in scikit-learn for
checking whether an estimator object has been fitted or not. This method is
typically implemented in custom estimator classes that are built on top of
scikit-learn's base classes like `BaseEstimator` or its subclasses.

Developers should use :func:`~sklearn.utils.validation.check_is_fitted`
at the beginning of all methods except `fit`. If they need to customize or
speed-up the check, they can implement the `__sklearn_is_fitted__` method as
shown below.

In this example the custom estimator showcases the usage of the
`__sklearn_is_fitted__` method and the `check_is_fitted` utility function
as developer APIs. The `__sklearn_is_fitted__` method checks fitted status
by verifying the presence of the `_is_fitted` attribute.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class CustomEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, parameter=1):
        if False:
            for i in range(10):
                print('nop')
        self.parameter = parameter

    def fit(self, X, y):
        if False:
            return 10
        '\n        Fit the estimator to the training data.\n        '
        self.classes_ = sorted(set(y))
        self._is_fitted = True
        return self

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        '\n        Perform Predictions\n\n        If the estimator is not fitted, then raise NotFittedError\n        '
        check_is_fitted(self)
        predictions = [self.classes_[0]] * len(X)
        return predictions

    def score(self, X, y):
        if False:
            i = 10
            return i + 15
        '\n        Calculate Score\n\n        If the estimator is not fitted, then raise NotFittedError\n        '
        check_is_fitted(self)
        return 0.5

    def __sklearn_is_fitted__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check fitted status and return a Boolean value.\n        '
        return hasattr(self, '_is_fitted') and self._is_fitted