"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
import pandas as pd

class PredictionResults:
    """
    Results class for predictions.

    Parameters
    ----------
    predicted_mean : ndarray
        The array containing the prediction means.
    var_pred_mean : ndarray
        The array of the variance of the prediction means.
    var_resid : ndarray
        The array of residual variances.
    df : int
        The degree of freedom used if dist is 't'.
    dist : {'norm', 't', object}
        Either a string for the normal or t distribution or another object
        that exposes a `ppf` method.
    row_labels : list[str]
        Row labels used in summary frame.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid, df=None, dist=None, row_labels=None):
        if False:
            for i in range(10):
                print('nop')
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se_obs(self):
        if False:
            i = 10
            return i + 15
        return np.sqrt(self.var_pred_mean + self.var_resid)

    @property
    def se_mean(self):
        if False:
            return 10
        return self.se

    @property
    def predicted_mean(self):
        if False:
            while True:
                i = 10
        return self.predicted

    @property
    def var_pred_mean(self):
        if False:
            print('Hello World!')
        return self.var_pred

    @property
    def se(self):
        if False:
            for i in range(10):
                print('nop')
        return np.sqrt(self.var_pred_mean)

    def conf_int(self, obs=False, alpha=0.05):
        if False:
            while True:
                i = 10
        '\n        Returns the confidence interval of the value, `effect` of the\n        constraint.\n\n        This is currently only available for t and z tests.\n\n        Parameters\n        ----------\n        alpha : float, optional\n            The significance level for the confidence interval.\n            ie., The default `alpha` = .05 returns a 95% confidence interval.\n\n        Returns\n        -------\n        ci : ndarray, (k_constraints, 2)\n            The array has the lower and the upper limit of the confidence\n            interval in the columns.\n        '
        se = self.se_obs if obs else self.se_mean
        q = self.dist.ppf(1 - alpha / 2.0, *self.dist_args)
        lower = self.predicted_mean - q * se
        upper = self.predicted_mean + q * se
        return np.column_stack((lower, upper))

    def summary_frame(self, alpha=0.05):
        if False:
            print('Hello World!')
        ci_obs = self.conf_int(alpha=alpha, obs=True)
        ci_mean = self.conf_int(alpha=alpha, obs=False)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]
        to_include['obs_ci_lower'] = ci_obs[:, 0]
        to_include['obs_ci_upper'] = ci_obs[:, 1]
        self.table = to_include
        res = pd.DataFrame(to_include, index=self.row_labels, columns=to_include.keys())
        return res

def get_prediction(self, exog=None, transform=True, weights=None, row_labels=None, pred_kwds=None):
    if False:
        i = 10
        return i + 15
    "\n    Compute prediction results.\n\n    Parameters\n    ----------\n    exog : array_like, optional\n        The values for which you want to predict.\n    transform : bool, optional\n        If the model was fit via a formula, do you want to pass\n        exog through the formula. Default is True. E.g., if you fit\n        a model y ~ log(x1) + log(x2), and transform is True, then\n        you can pass a data structure that contains x1 and x2 in\n        their original form. Otherwise, you'd need to log the data\n        first.\n    weights : array_like, optional\n        Weights interpreted as in WLS, used for the variance of the predicted\n        residual.\n    row_labels : list\n        A list of row labels to use.  If not provided, read `exog` is\n        available.\n    **kwargs\n        Some models can take additional keyword arguments, see the predict\n        method of the model for the details.\n\n    Returns\n    -------\n    linear_model.PredictionResults\n        The prediction results instance contains prediction and prediction\n        variance and can on demand calculate confidence intervals and summary\n        tables for the prediction of the mean and of new observations.\n    "
    if transform and hasattr(self.model, 'formula') and (exog is not None):
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)
    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None
        exog = np.asarray(exog)
        if exog.ndim == 1:
            if self.params.shape[0] > 1:
                exog = exog[None, :]
            else:
                exog = exog[:, None]
        exog = np.atleast_2d(exog)
    else:
        exog = self.model.exog
        if weights is None:
            weights = getattr(self.model, 'weights', None)
        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.size > 1 and (weights.ndim != 1 or weights.shape[0] == exog.shape[1]):
            raise ValueError('weights has wrong shape')
    if pred_kwds is None:
        pred_kwds = {}
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)
    covb = self.cov_params()
    var_pred_mean = (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']
    if weights is not None:
        var_resid /= weights
    dist = ['norm', 't'][self.use_t]
    return PredictionResults(predicted_mean, var_pred_mean, var_resid, df=self.df_resid, dist=dist, row_labels=row_labels)