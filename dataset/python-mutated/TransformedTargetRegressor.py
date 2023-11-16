from sklearn.compose import TransformedTargetRegressor as _TransformedTargetRegressor
from .utils import TargetTransformerMixin

class TransformedTargetRegressor(TargetTransformerMixin, _TransformedTargetRegressor):

    def fit(self, X, y, **fit_params):
        if False:
            print('Hello World!')
        'Fit the model according to the given training data.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            Training vector, where n_samples is the number of samples and\n            n_features is the number of features.\n\n        y : array-like of shape (n_samples,)\n            Target values.\n\n        **fit_params : dict\n            Parameters passed to the ``fit`` method of the underlying\n            regressor.\n\n\n        Returns\n        -------\n        self : object\n        '
        y = y.astype('float64')
        r = super().fit(X, y, **fit_params)
        self._carry_over_estimator_fit_vars(self.regressor_, ignore=['transformer_', 'regressor_'])
        return r