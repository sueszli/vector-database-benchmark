from sklearn.base import is_classifier
from sklearn.multioutput import MultiOutputRegressor as sk_MultiOutputRegressor
from sklearn.multioutput import _fit_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_fit_params, has_fit_parameter
try:
    from sklearn.utils.parallel import Parallel, delayed
except ImportError:
    from joblib import Parallel
    from sklearn.utils.fixes import delayed

class MultiOutputRegressor(sk_MultiOutputRegressor):
    """
    :class:`sklearn.utils.multioutput.MultiOutputRegressor` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        if False:
            return 10
        'Fit the model to data, separately for each output variable.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape (n_samples, n_features)\n            The input data.\n\n        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)\n            Multi-output targets. An indicator matrix turns on multilabel\n            estimation.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights. If `None`, then samples are equally weighted.\n            Only supported if the underlying regressor supports sample\n            weights.\n\n        **fit_params : dict of string -> object\n            Parameters passed to the ``estimator.fit`` method of each step.\n\n            .. versionadded:: 0.23\n\n        Returns\n        -------\n        self : object\n            Returns a fitted instance.\n        '
        if not hasattr(self.estimator, 'fit'):
            raise ValueError('The base estimator should implement a fit method')
        y = self._validate_data(X='no_validation', y=y, multi_output=True)
        if is_classifier(self):
            check_classification_targets(y)
        if y.ndim == 1:
            raise ValueError('y must have at least two dimensions for multi-output regression but has only one.')
        if sample_weight is not None and (not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError('Underlying estimator does not support sample weights.')
        fit_params_validated = _check_fit_params(X, fit_params)
        if 'eval_set' in fit_params_validated.keys():
            eval_set = fit_params_validated.pop('eval_set')
            self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_fit_estimator)(self.estimator, X, y[:, i], sample_weight, eval_set=[(eval_set[0][0], eval_set[0][1][:, i])] if isinstance(eval_set, list) else (eval_set[0], eval_set[1][:, i]), **fit_params_validated) for i in range(y.shape[1])))
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_fit_estimator)(self.estimator, X, y[:, i], sample_weight, **fit_params_validated) for i in range(y.shape[1])))
        if hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], 'feature_names_in_'):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_
        return self