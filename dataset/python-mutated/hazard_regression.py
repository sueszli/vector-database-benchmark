"""
Implementation of proportional hazards regression models for duration
data that may be censored ("Cox models").

References
----------
T Therneau (1996).  Extending the Cox model.  Technical report.
http://www.mayo.edu/research/documents/biostat-58pdf/DOC-10027288

G Rodriguez (2005).  Non-parametric estimation in survival models.
http://data.princeton.edu/pop509/NonParametricSurvival.pdf

B Gillespie (2006).  Checking the assumptions in the Cox proportional
hazards model.
http://www.mwsug.org/proceedings/2006/stats/MWSUG-2006-SD08.pdf
"""
import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
_predict_docstring = "\n    Returns predicted values from the proportional hazards\n    regression model.\n\n    Parameters\n    ----------%(params_doc)s\n    exog : array_like\n        Data to use as `exog` in forming predictions.  If not\n        provided, the `exog` values from the model used to fit the\n        data are used.%(cov_params_doc)s\n    endog : array_like\n        Duration (time) values at which the predictions are made.\n        Only used if pred_type is either 'cumhaz' or 'surv'.  If\n        using model `exog`, defaults to model `endog` (time), but\n        may be provided explicitly to make predictions at\n        alternative times.\n    strata : array_like\n        A vector of stratum values used to form the predictions.\n        Not used (may be 'None') if pred_type is 'lhr' or 'hr'.\n        If `exog` is None, the model stratum values are used.  If\n        `exog` is not None and pred_type is 'surv' or 'cumhaz',\n        stratum values must be provided (unless there is only one\n        stratum).\n    offset : array_like\n        Offset values used to create the predicted values.\n    pred_type : str\n        If 'lhr', returns log hazard ratios, if 'hr' returns\n        hazard ratios, if 'surv' returns the survival function, if\n        'cumhaz' returns the cumulative hazard function.\n    pred_only : bool\n        If True, returns only an array of predicted values.  Otherwise\n        returns a bunch containing the predicted values and standard\n        errors.\n\n    Returns\n    -------\n    A bunch containing two fields: `predicted_values` and\n    `standard_errors`.\n\n    Notes\n    -----\n    Standard errors are only returned when predicting the log\n    hazard ratio (pred_type is 'lhr').\n\n    Types `surv` and `cumhaz` require estimation of the cumulative\n    hazard function.\n"
_predict_params_doc = '\n    params : array_like\n        The proportional hazards model parameters.'
_predict_cov_params_docstring = "\n    cov_params : array_like\n        The covariance matrix of the estimated `params` vector,\n        used to obtain prediction errors if pred_type='lhr',\n        otherwise optional."

class PHSurvivalTime:

    def __init__(self, time, status, exog, strata=None, entry=None, offset=None):
        if False:
            i = 10
            return i + 15
        '\n        Represent a collection of survival times with possible\n        stratification and left truncation.\n\n        Parameters\n        ----------\n        time : array_like\n            The times at which either the event (failure) occurs or\n            the observation is censored.\n        status : array_like\n            Indicates whether the event (failure) occurs at `time`\n            (`status` is 1), or if `time` is a censoring time (`status`\n            is 0).\n        exog : array_like\n            The exogeneous (covariate) data matrix, cases are rows and\n            variables are columns.\n        strata : array_like\n            Grouping variable defining the strata.  If None, all\n            observations are in a single stratum.\n        entry : array_like\n            Entry (left truncation) times.  The observation is not\n            part of the risk set for times before the entry time.  If\n            None, the entry time is treated as being zero, which\n            gives no left truncation.  The entry time must be less\n            than or equal to `time`.\n        offset : array_like\n            An optional array of offsets\n        '
        if strata is None:
            strata = np.zeros(len(time), dtype=np.int32)
        if entry is None:
            entry = np.zeros(len(time))
        self._check(time, status, strata, entry)
        stu = np.unique(strata)
        sth = {x: [] for x in stu}
        for (i, k) in enumerate(strata):
            sth[k].append(i)
        stratum_rows = [np.asarray(sth[k], dtype=np.int32) for k in stu]
        stratum_names = stu
        ix = [i for (i, ix) in enumerate(stratum_rows) if status[ix].sum() > 0]
        self.nstrat_orig = len(stratum_rows)
        stratum_rows = [stratum_rows[i] for i in ix]
        stratum_names = [stratum_names[i] for i in ix]
        nstrat = len(stratum_rows)
        self.nstrat = nstrat
        for (stx, ix) in enumerate(stratum_rows):
            last_failure = max(time[ix][status[ix] == 1])
            ii = [i for (i, t) in enumerate(entry[ix]) if t <= last_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]
        for (stx, ix) in enumerate(stratum_rows):
            first_failure = min(time[ix][status[ix] == 1])
            ii = [i for (i, t) in enumerate(time[ix]) if t >= first_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]
        for (stx, ix) in enumerate(stratum_rows):
            ii = np.argsort(time[ix])
            stratum_rows[stx] = stratum_rows[stx][ii]
        if offset is not None:
            self.offset_s = []
            for stx in range(nstrat):
                self.offset_s.append(offset[stratum_rows[stx]])
        else:
            self.offset_s = None
        self.n_obs = sum([len(ix) for ix in stratum_rows])
        self.stratum_rows = stratum_rows
        self.stratum_names = stratum_names
        self.time_s = self._split(time)
        self.exog_s = self._split(exog)
        self.status_s = self._split(status)
        self.entry_s = self._split(entry)
        (self.ufailt_ix, self.risk_enter, self.risk_exit, self.ufailt) = ([], [], [], [])
        for stx in range(self.nstrat):
            ift = np.flatnonzero(self.status_s[stx] == 1)
            ft = self.time_s[stx][ift]
            uft = np.unique(ft)
            nuft = len(uft)
            uft_map = dict([(x, i) for (i, x) in enumerate(uft)])
            uft_ix = [[] for k in range(nuft)]
            for (ix, ti) in zip(ift, ft):
                uft_ix[uft_map[ti]].append(ix)
            risk_enter1 = [[] for k in range(nuft)]
            for (i, t) in enumerate(self.time_s[stx]):
                ix = np.searchsorted(uft, t, 'right') - 1
                if ix >= 0:
                    risk_enter1[ix].append(i)
            risk_exit1 = [[] for k in range(nuft)]
            for (i, t) in enumerate(self.entry_s[stx]):
                ix = np.searchsorted(uft, t)
                risk_exit1[ix].append(i)
            self.ufailt.append(uft)
            self.ufailt_ix.append([np.asarray(x, dtype=np.int32) for x in uft_ix])
            self.risk_enter.append([np.asarray(x, dtype=np.int32) for x in risk_enter1])
            self.risk_exit.append([np.asarray(x, dtype=np.int32) for x in risk_exit1])

    def _split(self, x):
        if False:
            print('Hello World!')
        v = []
        if x.ndim == 1:
            for ix in self.stratum_rows:
                v.append(x[ix])
        else:
            for ix in self.stratum_rows:
                v.append(x[ix, :])
        return v

    def _check(self, time, status, strata, entry):
        if False:
            for i in range(10):
                print('nop')
        (n1, n2, n3, n4) = (len(time), len(status), len(strata), len(entry))
        nv = [n1, n2, n3, n4]
        if max(nv) != min(nv):
            raise ValueError('endog, status, strata, and ' + 'entry must all have the same length')
        if min(time) < 0:
            raise ValueError('endog must be non-negative')
        if min(entry) < 0:
            raise ValueError('entry time must be non-negative')
        if np.any(entry > time):
            raise ValueError('entry times may not occur ' + 'after event or censoring times')

class PHReg(model.LikelihoodModel):
    """
    Cox Proportional Hazards Regression Model

    The Cox PH Model is for right censored data.

    Parameters
    ----------
    endog : array_like
        The observed times (event or censoring)
    exog : 2D array_like
        The covariates or exogeneous variables
    status : array_like
        The censoring status values; status=1 indicates that an
        event occurred (e.g. failure or death), status=0 indicates
        that the observation was right censored. If None, defaults
        to status=1 for all cases.
    entry : array_like
        The entry times, if left truncation occurs
    strata : array_like
        Stratum labels.  If None, all observations are taken to be
        in a single stratum.
    ties : str
        The method used to handle tied times, must be either 'breslow'
        or 'efron'.
    offset : array_like
        Array of offset values
    missing : str
        The method used to handle missing data

    Notes
    -----
    Proportional hazards regression models should not include an
    explicit or implicit intercept.  The effect of an intercept is
    not identified using the partial likelihood approach.

    `endog`, `event`, `strata`, `entry`, and the first dimension
    of `exog` all must have the same length
    """

    def __init__(self, endog, exog, status=None, entry=None, strata=None, offset=None, ties='breslow', missing='drop', **kwargs):
        if False:
            i = 10
            return i + 15
        if status is None:
            status = np.ones(len(endog))
        super(PHReg, self).__init__(endog, exog, status=status, entry=entry, strata=strata, offset=offset, missing=missing, **kwargs)
        if self.status is not None:
            self.status = np.asarray(self.status)
        if self.entry is not None:
            self.entry = np.asarray(self.entry)
        if self.strata is not None:
            self.strata = np.asarray(self.strata)
        if self.offset is not None:
            self.offset = np.asarray(self.offset)
        self.surv = PHSurvivalTime(self.endog, self.status, self.exog, self.strata, self.entry, self.offset)
        self.nobs = len(self.endog)
        self.groups = None
        self.missing = missing
        self.df_resid = float(self.exog.shape[0] - np.linalg.matrix_rank(self.exog))
        self.df_model = float(np.linalg.matrix_rank(self.exog))
        ties = ties.lower()
        if ties not in ('efron', 'breslow'):
            raise ValueError('`ties` must be either `efron` or ' + '`breslow`')
        self.ties = ties

    @classmethod
    def from_formula(cls, formula, data, status=None, entry=None, strata=None, offset=None, subset=None, ties='breslow', missing='drop', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a proportional hazards regression model from a formula\n        and dataframe.\n\n        Parameters\n        ----------\n        formula : str or generic Formula object\n            The formula specifying the model\n        data : array_like\n            The data for the model. See Notes.\n        status : array_like\n            The censoring status values; status=1 indicates that an\n            event occurred (e.g. failure or death), status=0 indicates\n            that the observation was right censored. If None, defaults\n            to status=1 for all cases.\n        entry : array_like\n            The entry times, if left truncation occurs\n        strata : array_like\n            Stratum labels.  If None, all observations are taken to be\n            in a single stratum.\n        offset : array_like\n            Array of offset values\n        subset : array_like\n            An array-like object of booleans, integers, or index\n            values that indicate the subset of df to use in the\n            model. Assumes df is a `pandas.DataFrame`\n        ties : str\n            The method used to handle tied times, must be either \'breslow\'\n            or \'efron\'.\n        missing : str\n            The method used to handle missing data\n        args : extra arguments\n            These are passed to the model\n        kwargs : extra keyword arguments\n            These are passed to the model with one exception. The\n            ``eval_env`` keyword is passed to patsy. It can be either a\n            :class:`patsy:patsy.EvalEnvironment` object or an integer\n            indicating the depth of the namespace to use. For example, the\n            default ``eval_env=0`` uses the calling namespace. If you wish\n            to use a "clean" environment set ``eval_env=-1``.\n\n        Returns\n        -------\n        model : PHReg model instance\n        '
        if isinstance(status, str):
            status = data[status]
        if isinstance(entry, str):
            entry = data[entry]
        if isinstance(strata, str):
            strata = data[strata]
        if isinstance(offset, str):
            offset = data[offset]
        import re
        terms = re.split('[+\\-~]', formula)
        for term in terms:
            term = term.strip()
            if term in ('0', '1'):
                import warnings
                warnings.warn("PHReg formulas should not include any '0' or '1' terms")
        mod = super(PHReg, cls).from_formula(formula, data, *args, status=status, entry=entry, strata=strata, offset=offset, subset=subset, ties=ties, missing=missing, drop_cols=['Intercept'], **kwargs)
        return mod

    def fit(self, groups=None, **args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit a proportional hazards regression model.\n\n        Parameters\n        ----------\n        groups : array_like\n            Labels indicating groups of observations that may be\n            dependent.  If present, the standard errors account for\n            this dependence. Does not affect fitted values.\n\n        Returns\n        -------\n        PHRegResults\n            Returns a results instance.\n        '
        if groups is not None:
            if len(groups) != len(self.endog):
                msg = 'len(groups) = %d and len(endog) = %d differ' % (len(groups), len(self.endog))
                raise ValueError(msg)
            self.groups = np.asarray(groups)
        else:
            self.groups = None
        if 'disp' not in args:
            args['disp'] = False
        fit_rslts = super(PHReg, self).fit(**args)
        if self.groups is None:
            cov_params = fit_rslts.cov_params()
        else:
            cov_params = self.robust_covariance(fit_rslts.params)
        results = PHRegResults(self, fit_rslts.params, cov_params)
        return results

    def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, **kwargs):
        if False:
            return 10
        "\n        Return a regularized fit to a linear regression model.\n\n        Parameters\n        ----------\n        method : {'elastic_net'}\n            Only the `elastic_net` approach is currently implemented.\n        alpha : scalar or array_like\n            The penalty weight.  If a scalar, the same penalty weight\n            applies to all variables in the model.  If a vector, it\n            must have the same length as `params`, and contains a\n            penalty weight for each coefficient.\n        start_params : array_like\n            Starting values for `params`.\n        refit : bool\n            If True, the model is refit using only the variables that\n            have non-zero coefficients in the regularized fit.  The\n            refitted model is not regularized.\n        **kwargs\n            Additional keyword arguments used to fit the model.\n\n        Returns\n        -------\n        PHRegResults\n            Returns a results instance.\n\n        Notes\n        -----\n        The penalty is the ``elastic net`` penalty, which is a\n        combination of L1 and L2 penalties.\n\n        The function that is minimized is:\n\n        .. math::\n\n            -loglike/n + alpha*((1-L1\\_wt)*|params|_2^2/2 + L1\\_wt*|params|_1)\n\n        where :math:`|*|_1` and :math:`|*|_2` are the L1 and L2 norms.\n\n        Post-estimation results are based on the same data used to\n        select variables, hence may be subject to overfitting biases.\n\n        The elastic_net method uses the following keyword arguments:\n\n        maxiter : int\n            Maximum number of iterations\n        L1_wt  : float\n            Must be in [0, 1].  The L1 penalty has weight L1_wt and the\n            L2 penalty has weight 1 - L1_wt.\n        cnvrg_tol : float\n            Convergence threshold for line searches\n        zero_tol : float\n            Coefficients below this threshold are treated as zero.\n        "
        from statsmodels.base.elastic_net import fit_elasticnet
        if method != 'elastic_net':
            raise ValueError('method for fit_regularized must be elastic_net')
        defaults = {'maxiter': 50, 'L1_wt': 1, 'cnvrg_tol': 1e-10, 'zero_tol': 1e-10}
        defaults.update(kwargs)
        return fit_elasticnet(self, method=method, alpha=alpha, start_params=start_params, refit=refit, **defaults)

    def loglike(self, params):
        if False:
            while True:
                i = 10
        '\n        Returns the log partial likelihood function evaluated at\n        `params`.\n        '
        if self.ties == 'breslow':
            return self.breslow_loglike(params)
        elif self.ties == 'efron':
            return self.efron_loglike(params)

    def score(self, params):
        if False:
            print('Hello World!')
        '\n        Returns the score function evaluated at `params`.\n        '
        if self.ties == 'breslow':
            return self.breslow_gradient(params)
        elif self.ties == 'efron':
            return self.efron_gradient(params)

    def hessian(self, params):
        if False:
            print('Hello World!')
        '\n        Returns the Hessian matrix of the log partial likelihood\n        function evaluated at `params`.\n        '
        if self.ties == 'breslow':
            return self.breslow_hessian(params)
        else:
            return self.efron_hessian(params)

    def breslow_loglike(self, params):
        if False:
            print('Hello World!')
        '\n        Returns the value of the log partial likelihood function\n        evaluated at `params`, using the Breslow method to handle tied\n        times.\n        '
        surv = self.surv
        like = 0.0
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0 = 0.0
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                ix = uft_ix[i]
                like += (linpred[ix] - np.log(xp0)).sum()
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
        return like

    def efron_loglike(self, params):
        if False:
            while True:
                i = 10
        '\n        Returns the value of the log partial likelihood function\n        evaluated at `params`, using the Efron method to handle tied\n        times.\n        '
        surv = self.surv
        like = 0.0
        for stx in range(surv.nstrat):
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            xp0 = 0.0
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp0f = e_linpred[uft_ix[i]].sum()
                ix = uft_ix[i]
                like += linpred[ix].sum()
                m = len(ix)
                J = np.arange(m, dtype=np.float64) / m
                like -= np.log(xp0 - J * xp0f).sum()
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
        return like

    def breslow_gradient(self, params):
        if False:
            return 10
        '\n        Returns the gradient of the log partial likelihood, using the\n        Breslow method to handle tied times.\n        '
        surv = self.surv
        grad = 0.0
        for stx in range(surv.nstrat):
            strat_ix = surv.stratum_rows[stx]
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            (xp0, xp1) = (0.0, 0.0)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                ix = uft_ix[i]
                grad += (exog_s[ix, :] - xp1 / xp0).sum(0)
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
        return grad

    def efron_gradient(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the gradient of the log partial likelihood evaluated\n        at `params`, using the Efron method to handle tied times.\n        '
        surv = self.surv
        grad = 0.0
        for stx in range(surv.nstrat):
            strat_ix = surv.stratum_rows[stx]
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            (xp0, xp1) = (0.0, 0.0)
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 += e_linpred[ix].sum()
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf, :]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:, None] * v).sum(0)
                    grad += v.sum(0)
                    m = len(ixf)
                    J = np.arange(m, dtype=np.float64) / m
                    numer = xp1 - np.outer(J, xp1f)
                    denom = xp0 - np.outer(J, xp0f)
                    ratio = numer / denom
                    rsum = ratio.sum(0)
                    grad -= rsum
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    v = exog_s[ix, :]
                    xp0 -= e_linpred[ix].sum()
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
        return grad

    def breslow_hessian(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Returns the Hessian of the log partial likelihood evaluated at\n        `params`, using the Breslow method to handle tied times.\n        '
        surv = self.surv
        hess = 0.0
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            (xp0, xp1, xp2) = (0.0, 0.0, 0.0)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 += np.einsum('ij,ik,i->jk', v, v, elx)
                m = len(uft_ix[i])
                hess += m * (xp2 / xp0 - np.outer(xp1, xp1) / xp0 ** 2)
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 -= np.einsum('ij,ik,i->jk', v, v, elx)
        return -hess

    def efron_hessian(self, params):
        if False:
            print('Hello World!')
        '\n        Returns the Hessian matrix of the partial log-likelihood\n        evaluated at `params`, using the Efron method to handle tied\n        times.\n        '
        surv = self.surv
        hess = 0.0
        for stx in range(surv.nstrat):
            exog_s = surv.exog_s[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            (xp0, xp1, xp2) = (0.0, 0.0, 0.0)
            uft_ix = surv.ufailt_ix[stx]
            nuft = len(uft_ix)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                if len(ix) > 0:
                    xp0 += e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 += (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 += np.einsum('ij,ik,i->jk', v, v, elx)
                ixf = uft_ix[i]
                if len(ixf) > 0:
                    v = exog_s[ixf, :]
                    xp0f = e_linpred[ixf].sum()
                    xp1f = (e_linpred[ixf][:, None] * v).sum(0)
                    elx = e_linpred[ixf]
                    xp2f = np.einsum('ij,ik,i->jk', v, v, elx)
                m = len(uft_ix[i])
                J = np.arange(m, dtype=np.float64) / m
                c0 = xp0 - J * xp0f
                hess += xp2 * np.sum(1 / c0)
                hess -= xp2f * np.sum(J / c0)
                mat = (xp1[None, :] - np.outer(J, xp1f)) / c0[:, None]
                hess -= np.einsum('ij,ik->jk', mat, mat)
                ix = surv.risk_exit[stx][i]
                if len(ix) > 0:
                    xp0 -= e_linpred[ix].sum()
                    v = exog_s[ix, :]
                    xp1 -= (e_linpred[ix][:, None] * v).sum(0)
                    elx = e_linpred[ix]
                    xp2 -= np.einsum('ij,ik,i->jk', v, v, elx)
        return -hess

    def robust_covariance(self, params):
        if False:
            print('Hello World!')
        '\n        Returns a covariance matrix for the proportional hazards model\n        regresion coefficient estimates that is robust to certain\n        forms of model misspecification.\n\n        Parameters\n        ----------\n        params : ndarray\n            The parameter vector at which the covariance matrix is\n            calculated.\n\n        Returns\n        -------\n        The robust covariance matrix as a square ndarray.\n\n        Notes\n        -----\n        This function uses the `groups` argument to determine groups\n        within which observations may be dependent.  The covariance\n        matrix is calculated using the Huber-White "sandwich" approach.\n        '
        if self.groups is None:
            raise ValueError('`groups` must be specified to calculate the robust covariance matrix')
        hess = self.hessian(params)
        score_obs = self.score_residuals(params)
        grads = {}
        for (i, g) in enumerate(self.groups):
            if g not in grads:
                grads[g] = 0.0
            grads[g] += score_obs[i, :]
        grads = np.asarray(list(grads.values()))
        mat = grads[None, :, :]
        mat = mat.T * mat
        mat = mat.sum(1)
        hess_inv = np.linalg.inv(hess)
        cmat = np.dot(hess_inv, np.dot(mat, hess_inv))
        return cmat

    def score_residuals(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the score residuals calculated at a given vector of\n        parameters.\n\n        Parameters\n        ----------\n        params : ndarray\n            The parameter vector at which the score residuals are\n            calculated.\n\n        Returns\n        -------\n        The score residuals, returned as a ndarray having the same\n        shape as `exog`.\n\n        Notes\n        -----\n        Observations in a stratum with no observed events have undefined\n        score residuals, and contain NaN in the returned matrix.\n        '
        surv = self.surv
        score_resid = np.zeros(self.exog.shape, dtype=np.float64)
        mask = np.zeros(self.exog.shape[0], dtype=np.int32)
        w_avg = self.weighted_covariate_averages(params)
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            strat_ix = surv.stratum_rows[stx]
            xp0 = 0.0
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            at_risk_ix = set()
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                at_risk_ix |= set(ix)
                xp0 += e_linpred[ix].sum()
                atr_ix = list(at_risk_ix)
                leverage = exog_s[atr_ix, :] - w_avg[stx][i, :]
                d = np.zeros(exog_s.shape[0])
                d[uft_ix[i]] = 1
                dchaz = len(uft_ix[i]) / xp0
                mrp = d[atr_ix] - e_linpred[atr_ix] * dchaz
                ii = strat_ix[atr_ix]
                score_resid[ii, :] += leverage * mrp[:, None]
                mask[ii] = 1
                ix = surv.risk_exit[stx][i]
                at_risk_ix -= set(ix)
                xp0 -= e_linpred[ix].sum()
        jj = np.flatnonzero(mask == 0)
        if len(jj) > 0:
            score_resid[jj, :] = np.nan
        return score_resid

    def weighted_covariate_averages(self, params):
        if False:
            i = 10
            return i + 15
        '\n        Returns the hazard-weighted average of covariate values for\n        subjects who are at-risk at a particular time.\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter vector\n\n        Returns\n        -------\n        averages : list of ndarrays\n            averages[stx][i,:] is a row vector containing the weighted\n            average values (for all the covariates) of at-risk\n            subjects a the i^th largest observed failure time in\n            stratum `stx`, using the hazard multipliers as weights.\n\n        Notes\n        -----\n        Used to calculate leverages and score residuals.\n        '
        surv = self.surv
        averages = []
        (xp0, xp1) = (0.0, 0.0)
        for stx in range(surv.nstrat):
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            average_s = np.zeros((len(uft_ix), exog_s.shape[1]), dtype=np.float64)
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            linpred -= linpred.max()
            e_linpred = np.exp(linpred)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                xp1 += np.dot(e_linpred[ix], exog_s[ix, :])
                average_s[i, :] = xp1 / xp0
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
                xp1 -= np.dot(e_linpred[ix], exog_s[ix, :])
            averages.append(average_s)
        return averages

    def baseline_cumulative_hazard(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Estimate the baseline cumulative hazard and survival\n        functions.\n\n        Parameters\n        ----------\n        params : ndarray\n            The model parameters.\n\n        Returns\n        -------\n        A list of triples (time, hazard, survival) containing the time\n        values and corresponding cumulative hazard and survival\n        function values for each stratum.\n\n        Notes\n        -----\n        Uses the Nelson-Aalen estimator.\n        '
        surv = self.surv
        rslt = []
        for stx in range(surv.nstrat):
            uft = surv.ufailt[stx]
            uft_ix = surv.ufailt_ix[stx]
            exog_s = surv.exog_s[stx]
            nuft = len(uft_ix)
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)
            xp0 = 0.0
            h0 = np.zeros(nuft, dtype=np.float64)
            for i in range(nuft)[::-1]:
                ix = surv.risk_enter[stx][i]
                xp0 += e_linpred[ix].sum()
                ix = uft_ix[i]
                h0[i] = len(ix) / xp0
                ix = surv.risk_exit[stx][i]
                xp0 -= e_linpred[ix].sum()
            cumhaz = np.cumsum(h0) - h0
            current_strata_surv = np.exp(-cumhaz)
            rslt.append([uft, cumhaz, current_strata_surv])
        return rslt

    def baseline_cumulative_hazard_function(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a function that calculates the baseline cumulative\n        hazard function for each stratum.\n\n        Parameters\n        ----------\n        params : ndarray\n            The model parameters.\n\n        Returns\n        -------\n        A dict mapping stratum names to the estimated baseline\n        cumulative hazard function.\n        '
        from scipy.interpolate import interp1d
        surv = self.surv
        base = self.baseline_cumulative_hazard(params)
        cumhaz_f = {}
        for stx in range(surv.nstrat):
            time_h = base[stx][0]
            cumhaz = base[stx][1]
            time_h = np.r_[-np.inf, time_h, np.inf]
            cumhaz = np.r_[cumhaz[0], cumhaz, cumhaz[-1]]
            func = interp1d(time_h, cumhaz, kind='zero')
            cumhaz_f[self.surv.stratum_names[stx]] = func
        return cumhaz_f

    @Appender(_predict_docstring % {'params_doc': _predict_params_doc, 'cov_params_doc': _predict_cov_params_docstring})
    def predict(self, params, exog=None, cov_params=None, endog=None, strata=None, offset=None, pred_type='lhr', pred_only=False):
        if False:
            return 10
        pred_type = pred_type.lower()
        if pred_type not in ['lhr', 'hr', 'surv', 'cumhaz']:
            msg = 'Type %s not allowed for prediction' % pred_type
            raise ValueError(msg)

        class bunch:
            predicted_values = None
            standard_errors = None
        ret_val = bunch()
        exog_provided = True
        if exog is None:
            exog = self.exog
            exog_provided = False
        lhr = np.dot(exog, params)
        if offset is not None:
            lhr += offset
        elif self.offset is not None and (not exog_provided):
            lhr += self.offset
        if pred_type == 'lhr':
            ret_val.predicted_values = lhr
            if cov_params is not None:
                mat = np.dot(exog, cov_params)
                va = (mat * exog).sum(1)
                ret_val.standard_errors = np.sqrt(va)
            if pred_only:
                return ret_val.predicted_values
            return ret_val
        hr = np.exp(lhr)
        if pred_type == 'hr':
            ret_val.predicted_values = hr
            if pred_only:
                return ret_val.predicted_values
            return ret_val
        if endog is None and exog_provided:
            msg = 'If `exog` is provided `endog` must be provided.'
            raise ValueError(msg)
        elif endog is None and (not exog_provided):
            endog = self.endog
        if strata is None:
            if exog_provided and self.surv.nstrat > 1:
                raise ValueError('`strata` must be provided')
            if self.strata is None:
                strata = [self.surv.stratum_names[0]] * len(endog)
            else:
                strata = self.strata
        cumhaz = np.nan * np.ones(len(endog), dtype=np.float64)
        stv = np.unique(strata)
        bhaz = self.baseline_cumulative_hazard_function(params)
        for stx in stv:
            ix = np.flatnonzero(strata == stx)
            func = bhaz[stx]
            cumhaz[ix] = func(endog[ix]) * hr[ix]
        if pred_type == 'cumhaz':
            ret_val.predicted_values = cumhaz
        elif pred_type == 'surv':
            ret_val.predicted_values = np.exp(-cumhaz)
        if pred_only:
            return ret_val.predicted_values
        return ret_val

    def get_distribution(self, params, scale=1.0, exog=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns a scipy distribution object corresponding to the\n        distribution of uncensored endog (duration) values for each\n        case.\n\n        Parameters\n        ----------\n        params : array_like\n            The proportional hazards model parameters.\n        scale : float\n            Present for compatibility, not used.\n        exog : array_like\n            A design matrix, defaults to model.exog.\n\n        Returns\n        -------\n        A list of objects of type scipy.stats.distributions.rv_discrete\n\n        Notes\n        -----\n        The distributions are obtained from a simple discrete estimate\n        of the survivor function that puts all mass on the observed\n        failure times within a stratum.\n        '
        surv = self.surv
        bhaz = self.baseline_cumulative_hazard(params)
        (pk, xk) = ([], [])
        if exog is None:
            exog_split = surv.exog_s
        else:
            exog_split = self.surv._split(exog)
        for stx in range(self.surv.nstrat):
            exog_s = exog_split[stx]
            linpred = np.dot(exog_s, params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)
            pts = bhaz[stx][0]
            ichaz = np.outer(e_linpred, bhaz[stx][1])
            usurv = np.exp(-ichaz)
            z = np.zeros((usurv.shape[0], 1))
            usurv = np.concatenate((usurv, z), axis=1)
            probs = -np.diff(usurv, 1)
            pk.append(probs)
            xk.append(np.outer(np.ones(probs.shape[0]), pts))
        mxc = max([x.shape[1] for x in xk])
        for k in range(self.surv.nstrat):
            if xk[k].shape[1] < mxc:
                xk1 = np.zeros((xk[k].shape[0], mxc))
                pk1 = np.zeros((pk[k].shape[0], mxc))
                xk1[:, 0:xk[k].shape[1]] = xk[k]
                pk1[:, 0:pk[k].shape[1]] = pk[k]
                (xk[k], pk[k]) = (xk1, pk1)
        xka = np.nan * np.ones((len(self.endog), mxc))
        pka = np.ones((len(self.endog), mxc), dtype=np.float64) / mxc
        for stx in range(self.surv.nstrat):
            ix = self.surv.stratum_rows[stx]
            xka[ix, :] = xk[stx]
            pka[ix, :] = pk[stx]
        dist = rv_discrete_float(xka, pka)
        return dist

class PHRegResults(base.LikelihoodModelResults):
    """
    Class to contain results of fitting a Cox proportional hazards
    survival model.

    PHregResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Attributes
    ----------
    model : class instance
        PHreg model instance that called fit.
    normalized_cov_params : ndarray
        The sampling covariance matrix of the estimates
    params : ndarray
        The coefficients of the fitted model.  Each coefficient is the
        log hazard ratio corresponding to a 1 unit difference in a
        single covariate while holding the other covariates fixed.
    bse : ndarray
        The standard errors of the fitted parameters.

    See Also
    --------
    statsmodels.LikelihoodModelResults
    """

    def __init__(self, model, params, cov_params, scale=1.0, covariance_type='naive'):
        if False:
            for i in range(10):
                print('nop')
        self.covariance_type = covariance_type
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        super(PHRegResults, self).__init__(model, params, scale=1.0, normalized_cov_params=cov_params)

    @cache_readonly
    def standard_errors(self):
        if False:
            print('Hello World!')
        '\n        Returns the standard errors of the parameter estimates.\n        '
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def bse(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the standard errors of the parameter estimates.\n        '
        return self.standard_errors

    def get_distribution(self):
        if False:
            while True:
                i = 10
        '\n        Returns a scipy distribution object corresponding to the\n        distribution of uncensored endog (duration) values for each\n        case.\n\n        Returns\n        -------\n        A list of objects of type scipy.stats.distributions.rv_discrete\n\n        Notes\n        -----\n        The distributions are obtained from a simple discrete estimate\n        of the survivor function that puts all mass on the observed\n        failure times within a stratum.\n        '
        return self.model.get_distribution(self.params)

    @Appender(_predict_docstring % {'params_doc': '', 'cov_params_doc': ''})
    def predict(self, endog=None, exog=None, strata=None, offset=None, transform=True, pred_type='lhr'):
        if False:
            print('Hello World!')
        return super(PHRegResults, self).predict(exog=exog, transform=transform, cov_params=self.cov_params(), endog=endog, strata=strata, offset=offset, pred_type=pred_type)

    def _group_stats(self, groups):
        if False:
            print('Hello World!')
        '\n        Descriptive statistics of the groups.\n        '
        gsizes = np.unique(groups, return_counts=True)
        gsizes = gsizes[1]
        return (gsizes.min(), gsizes.max(), gsizes.mean(), len(gsizes))

    @cache_readonly
    def weighted_covariate_averages(self):
        if False:
            while True:
                i = 10
        '\n        The average covariate values within the at-risk set at each\n        event time point, weighted by hazard.\n        '
        return self.model.weighted_covariate_averages(self.params)

    @cache_readonly
    def score_residuals(self):
        if False:
            while True:
                i = 10
        '\n        A matrix containing the score residuals.\n        '
        return self.model.score_residuals(self.params)

    @cache_readonly
    def baseline_cumulative_hazard(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A list (corresponding to the strata) containing the baseline\n        cumulative hazard function evaluated at the event points.\n        '
        return self.model.baseline_cumulative_hazard(self.params)

    @cache_readonly
    def baseline_cumulative_hazard_function(self):
        if False:
            i = 10
            return i + 15
        '\n        A list (corresponding to the strata) containing function\n        objects that calculate the cumulative hazard function.\n        '
        return self.model.baseline_cumulative_hazard_function(self.params)

    @cache_readonly
    def schoenfeld_residuals(self):
        if False:
            return 10
        '\n        A matrix containing the Schoenfeld residuals.\n\n        Notes\n        -----\n        Schoenfeld residuals for censored observations are set to zero.\n        '
        surv = self.model.surv
        w_avg = self.weighted_covariate_averages
        sch_resid = np.nan * np.ones(self.model.exog.shape, dtype=np.float64)
        for stx in range(surv.nstrat):
            uft = surv.ufailt[stx]
            exog_s = surv.exog_s[stx]
            time_s = surv.time_s[stx]
            strat_ix = surv.stratum_rows[stx]
            ii = np.searchsorted(uft, time_s)
            jj = np.flatnonzero(ii < len(uft))
            sch_resid[strat_ix[jj], :] = exog_s[jj, :] - w_avg[stx][ii[jj], :]
        jj = np.flatnonzero(self.model.status == 0)
        sch_resid[jj, :] = np.nan
        return sch_resid

    @cache_readonly
    def martingale_residuals(self):
        if False:
            i = 10
            return i + 15
        '\n        The martingale residuals.\n        '
        surv = self.model.surv
        mart_resid = np.nan * np.ones(len(self.model.endog), dtype=np.float64)
        cumhaz_f_list = self.baseline_cumulative_hazard_function
        for stx in range(surv.nstrat):
            cumhaz_f = cumhaz_f_list[stx]
            exog_s = surv.exog_s[stx]
            time_s = surv.time_s[stx]
            linpred = np.dot(exog_s, self.params)
            if surv.offset_s is not None:
                linpred += surv.offset_s[stx]
            e_linpred = np.exp(linpred)
            ii = surv.stratum_rows[stx]
            chaz = cumhaz_f(time_s)
            mart_resid[ii] = self.model.status[ii] - e_linpred * chaz
        return mart_resid

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        if False:
            print('Hello World!')
        '\n        Summarize the proportional hazards regression results.\n\n        Parameters\n        ----------\n        yname : str, optional\n            Default is `y`\n        xname : list[str], optional\n            Names for the exogenous variables, default is `x#` for ## in p the\n            number of regressors. Must match the number of parameters in\n            the model\n        title : str, optional\n            Title for the top table. If not None, then this replaces\n            the default title\n        alpha : float\n            significance level for the confidence intervals\n\n        Returns\n        -------\n        smry : Summary instance\n            this holds the summary tables and text, which can be\n            printed or converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary2.Summary : class to hold summary results\n        '
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        float_format = '%8.3f'
        info = {}
        info['Model:'] = 'PH Reg'
        if yname is None:
            yname = self.model.endog_names
        info['Dependent variable:'] = yname
        info['Ties:'] = self.model.ties.capitalize()
        info['Sample size:'] = str(self.model.surv.n_obs)
        info['Num. events:'] = str(int(sum(self.model.status)))
        if self.model.groups is not None:
            (mn, mx, avg, num) = self._group_stats(self.model.groups)
            info['Num groups:'] = '%.0f' % num
            info['Min group size:'] = '%.0f' % mn
            info['Max group size:'] = '%.0f' % mx
            info['Avg group size:'] = '%.1f' % avg
        if self.model.strata is not None:
            (mn, mx, avg, num) = self._group_stats(self.model.strata)
            info['Num strata:'] = '%.0f' % num
            info['Min stratum size:'] = '%.0f' % mn
            info['Max stratum size:'] = '%.0f' % mx
            info['Avg stratum size:'] = '%.1f' % avg
        smry.add_dict(info, align='l', float_format=float_format)
        param = summary2.summary_params(self, alpha=alpha)
        param = param.rename(columns={'Coef.': 'log HR', 'Std.Err.': 'log HR SE'})
        param.insert(2, 'HR', np.exp(param['log HR']))
        a = '[%.3f' % (alpha / 2)
        param.loc[:, a] = np.exp(param.loc[:, a])
        a = '%.3f]' % (1 - alpha / 2)
        param.loc[:, a] = np.exp(param.loc[:, a])
        if xname is not None:
            param.index = xname
        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)
        smry.add_text('Confidence intervals are for the hazard ratios')
        dstrat = self.model.surv.nstrat_orig - self.model.surv.nstrat
        if dstrat > 0:
            if dstrat == 1:
                smry.add_text('1 stratum dropped for having no events')
            else:
                smry.add_text('%d strata dropped for having no events' % dstrat)
        if self.model.entry is not None:
            n_entry = sum(self.model.entry != 0)
            if n_entry == 1:
                smry.add_text('1 observation has a positive entry time')
            else:
                smry.add_text('%d observations have positive entry times' % n_entry)
        if self.model.groups is not None:
            smry.add_text('Standard errors account for dependence within groups')
        if hasattr(self, 'regularized'):
            smry.add_text('Standard errors do not account for the regularization')
        return smry

class rv_discrete_float:
    """
    A class representing a collection of discrete distributions.

    Parameters
    ----------
    xk : 2d array_like
        The support points, should be non-decreasing within each
        row.
    pk : 2d array_like
        The probabilities, should sum to one within each row.

    Notes
    -----
    Each row of `xk`, and the corresponding row of `pk` describe a
    discrete distribution.

    `xk` and `pk` should both be two-dimensional ndarrays.  Each row
    of `pk` should sum to 1.

    This class is used as a substitute for scipy.distributions.
    rv_discrete, since that class does not allow non-integer support
    points, or vectorized operations.

    Only a limited number of methods are implemented here compared to
    the other scipy distribution classes.
    """

    def __init__(self, xk, pk):
        if False:
            print('Hello World!')
        self.xk = xk
        self.pk = pk
        self.cpk = np.cumsum(self.pk, axis=1)

    def rvs(self, n=None):
        if False:
            while True:
                i = 10
        '\n        Returns a random sample from the discrete distribution.\n\n        A vector is returned containing a single draw from each row of\n        `xk`, using the probabilities of the corresponding row of `pk`\n\n        Parameters\n        ----------\n        n : not used\n            Present for signature compatibility\n        '
        n = self.xk.shape[0]
        u = np.random.uniform(size=n)
        ix = (self.cpk < u[:, None]).sum(1)
        ii = np.arange(n, dtype=np.int32)
        return self.xk[ii, ix]

    def mean(self):
        if False:
            return 10
        '\n        Returns a vector containing the mean values of the discrete\n        distributions.\n\n        A vector is returned containing the mean value of each row of\n        `xk`, using the probabilities in the corresponding row of\n        `pk`.\n        '
        return (self.xk * self.pk).sum(1)

    def var(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a vector containing the variances of the discrete\n        distributions.\n\n        A vector is returned containing the variance for each row of\n        `xk`, using the probabilities in the corresponding row of\n        `pk`.\n        '
        mn = self.mean()
        xkc = self.xk - mn[:, None]
        return (self.pk * (self.xk - xkc) ** 2).sum(1)

    def std(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a vector containing the standard deviations of the\n        discrete distributions.\n\n        A vector is returned containing the standard deviation for\n        each row of `xk`, using the probabilities in the corresponding\n        row of `pk`.\n        '
        return np.sqrt(self.var())