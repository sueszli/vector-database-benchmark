"""
Created on Sat May 19 15:53:21 2018

Author: Josef Perktold
License: BSD-3
"""
from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed

class ScreeningResults:
    """Results for Variable Screening

    Note: Indices except for exog_idx and in the iterated case also
    idx_nonzero_batches are based on the combined [exog_keep, exog] array.

    Attributes
    ----------
    results_final : instance
        Results instance returned by the final fit of the penalized model, i.e.
        after trimming exog with params below trimming threshold.
    results_pen : results instance
        Results instance of the penalized model before trimming. This includes
        variables from the last forward selection
    idx_nonzero
        index of exog columns in the final selection including exog_keep
    idx_exog
        index of exog columns in the final selection for exog candidates, i.e.
        without exog_keep
    idx_excl
        idx of excluded exog based on combined [exog_keep, exog] array. This is
        the complement of idx_nonzero
    converged : bool
        True if the iteration has converged and stopped before maxiter has been
        reached. False if maxiter has been reached.
    iterations : int
        number of iterations in the screening process. Each iteration consists
        of a forward selection step and a trimming step.
    history : dict of lists
        results collected for each iteration during the screening process
        'idx_nonzero' 'params_keep'].append(start_params)
            history['idx_added'].append(idx)

    The ScreeningResults returned by `screen_exog_iterator` has additional
    attributes:

    idx_nonzero_batches : ndarray 2-D
        Two-dimensional array with batch index in the first column and variable
        index withing batch in the second column. They can be used jointly as
        index for the data in the exog_iterator.
    exog_final_names : list[str]
        'var<bidx>_<idx>' where `bidx` is the batch index and `idx` is the
        index of the selected column withing batch `bidx`.
    history_batches : dict of lists
        This provides information about the selected variables within each
        batch during the first round screening
        'idx_nonzero' is based ond the array that includes exog_keep, while
        'idx_exog' is the index based on the exog of the batch.
    """

    def __init__(self, screener, **kwds):
        if False:
            while True:
                i = 10
        self.screener = screener
        self.__dict__.update(**kwds)

class VariableScreening:
    """Ultra-high, conditional sure independence screening

    This is an adjusted version of Fan's sure independence screening.

    Parameters
    ----------
    model : instance of penalizing model
        examples: GLMPenalized, PoissonPenalized and LogitPenalized.
        The attributes of the model instance `pen_weight` and `penal` will be
        ignored.
    pen_weight : None or float
        penalization weight use in SCAD penalized MLE
    k_add : int
        number of exog to add during expansion or forward selection
        see Notes section for tie handling
    k_max_add : int
        maximum number of variables to include during variable addition, i.e.
        forward selection. default is 30
    threshold_trim : float
        threshold for trimming parameters to zero, default is 1e-4
    k_max_included : int
        maximum total number of variables to include in model.
    ranking_attr : str
        This determines the result attribute or model method that is used for
        the ranking of exog to include. The availability of attributes depends
        on the model.
        Default is 'resid_pearson', 'model.score_factor' can be used in GLM.
    ranking_project : bool
        If ranking_project is True, then the exog candidates for inclusion are
        first projected on the already included exog before the computation
        of the ranking measure. This brings the ranking measure closer to
        the statistic of a score test for variable addition.

    Notes
    -----
    Status: experimental, tested only on a limited set of models and
    with a limited set of model options.

    Tie handling: If there are ties at the decision threshold, then all those
    tied exog columns are treated in the same way. During forward selection
    all exog columns with the same boundary value are included. During
    elimination, the tied columns are not dropped. Consequently, if ties are
    present, then the number of included exog can be larger than specified
    by k_add, k_max_add and k_max_included.

    The screening algorithm works similar to step wise regression. Each
    iteration of the screening algorithm includes a forward selection step
    where variables are added to the model, and a backwards selection step
    where variables are removed. In contrast to step wise regression, we add
    a fixed number of variables at each forward selection step. The
    backwards selection step is based on SCAD penalized estimation and
    trimming of variables with estimated coefficients below a threshold.
    The tuning parameters can be used to adjust the number of variables to add
    and to include depending on the size of the dataset.

    There is currently no automatic tuning parameter selection. Candidate
    explanatory variables should be standardized or should be on a similar
    scale because penalization and trimming are based on the absolute values
    of the parameters.


    TODOs and current limitations:

    freq_weights are not supported in this. Candidate ranking uses
    moment condition with resid_pearson or others without freq_weights.
    pearson_resid: GLM resid_pearson does not include freq_weights.

    variable names: do we keep track of those? currently made-up names

    currently only supports numpy arrays, no exog type check or conversion

    currently only single columns are selected, no terms (multi column exog)
    """

    def __init__(self, model, pen_weight=None, use_weights=True, k_add=30, k_max_add=30, threshold_trim=0.0001, k_max_included=20, ranking_attr='resid_pearson', ranking_project=True):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.model_class = model.__class__
        self.init_kwds = model._get_init_kwds()
        self.init_kwds.pop('pen_weight', None)
        self.init_kwds.pop('penal', None)
        self.endog = model.endog
        self.exog_keep = model.exog
        self.k_keep = model.exog.shape[1]
        self.nobs = len(self.endog)
        self.penal = self._get_penal()
        if pen_weight is not None:
            self.pen_weight = pen_weight
        else:
            self.pen_weight = self.nobs * 10
        self.use_weights = use_weights
        self.k_add = k_add
        self.k_max_add = k_max_add
        self.threshold_trim = threshold_trim
        self.k_max_included = k_max_included
        self.ranking_attr = ranking_attr
        self.ranking_project = ranking_project

    def _get_penal(self, weights=None):
        if False:
            print('Hello World!')
        'create new Penalty instance\n        '
        return SCADSmoothed(0.1, c0=0.0001, weights=weights)

    def ranking_measure(self, res_pen, exog, keep=None):
        if False:
            return 10
        'compute measure for ranking exog candidates for inclusion\n        '
        endog = self.endog
        if self.ranking_project:
            assert res_pen.model.exog.shape[1] == len(keep)
            ex_incl = res_pen.model.exog[:, keep]
            exog = exog - ex_incl.dot(np.linalg.pinv(ex_incl).dot(exog))
        if self.ranking_attr == 'predicted_poisson':
            p = res_pen.params.copy()
            if keep is not None:
                p[~keep] = 0
            predicted = res_pen.model.predict(p)
            resid_factor = (endog - predicted) / np.sqrt(predicted)
        elif self.ranking_attr[:6] == 'model.':
            attr = self.ranking_attr.split('.')[1]
            resid_factor = getattr(res_pen.model, attr)(res_pen.params)
            if resid_factor.ndim == 2:
                resid_factor = resid_factor[:, 0]
            mom_cond = np.abs(resid_factor.dot(exog)) ** 2
        else:
            resid_factor = getattr(res_pen, self.ranking_attr)
            mom_cond = np.abs(resid_factor.dot(exog)) ** 2
        return mom_cond

    def screen_exog(self, exog, endog=None, maxiter=100, method='bfgs', disp=False, fit_kwds=None):
        if False:
            for i in range(10):
                print('nop')
        'screen and select variables (columns) in exog\n\n        Parameters\n        ----------\n        exog : ndarray\n            candidate explanatory variables that are screened for inclusion in\n            the model\n        endog : ndarray (optional)\n            use a new endog in the screening model.\n            This is not tested yet, and might not work correctly\n        maxiter : int\n            number of screening iterations\n        method : str\n            optimization method to use in fit, needs to be only of the gradient\n            optimizers\n        disp : bool\n            display option for fit during optimization\n\n        Returns\n        -------\n        res_screen : instance of ScreeningResults\n            The attribute `results_final` contains is the results instance\n            with the final model selection.\n            `idx_nonzero` contains the index of the selected exog in the full\n            exog, combined exog that are always kept plust exog_candidates.\n            see ScreeningResults for a full description\n        '
        model_class = self.model_class
        if endog is None:
            endog = self.endog
        x0 = self.exog_keep
        k_keep = self.k_keep
        x1 = exog
        k_current = x0.shape[1]
        x = np.column_stack((x0, x1))
        (nobs, k_vars) = x.shape
        fkwds = fit_kwds if fit_kwds is not None else {}
        fit_kwds = {'maxiter': 200, 'disp': False}
        fit_kwds.update(fkwds)
        history = defaultdict(list)
        idx_nonzero = np.arange(k_keep, dtype=int)
        keep = np.ones(k_keep, np.bool_)
        idx_excl = np.arange(k_keep, k_vars)
        mod_pen = model_class(endog, x0, **self.init_kwds)
        mod_pen.pen_weight = 0
        res_pen = mod_pen.fit(**fit_kwds)
        start_params = res_pen.params
        converged = False
        idx_old = []
        for it in range(maxiter):
            x1 = x[:, idx_excl]
            mom_cond = self.ranking_measure(res_pen, x1, keep=keep)
            assert len(mom_cond) == len(idx_excl)
            mcs = np.sort(mom_cond)[::-1]
            idx_thr = min((self.k_max_add, k_current + self.k_add, len(mcs)))
            threshold = mcs[idx_thr]
            idx = np.concatenate((idx_nonzero, idx_excl[mom_cond > threshold]))
            start_params2 = np.zeros(len(idx))
            start_params2[:len(start_params)] = start_params
            if self.use_weights:
                weights = np.ones(len(idx))
                weights[:k_keep] = 0
                self.penal.weights = weights
            mod_pen = model_class(endog, x[:, idx], penal=self.penal, pen_weight=self.pen_weight, **self.init_kwds)
            res_pen = mod_pen.fit(method=method, start_params=start_params2, warn_convergence=False, skip_hessian=True, **fit_kwds)
            keep = np.abs(res_pen.params) > self.threshold_trim
            if keep.sum() > self.k_max_included:
                thresh_params = np.sort(np.abs(res_pen.params))[-self.k_max_included]
                keep2 = np.abs(res_pen.params) > thresh_params
                keep = np.logical_and(keep, keep2)
            keep[:k_keep] = True
            idx_nonzero = idx[keep]
            if disp:
                print(keep)
                print(idx_nonzero)
            k_current = len(idx_nonzero)
            start_params = res_pen.params[keep]
            mask_excl = np.ones(k_vars, dtype=bool)
            mask_excl[idx_nonzero] = False
            idx_excl = np.nonzero(mask_excl)[0]
            history['idx_nonzero'].append(idx_nonzero)
            history['keep'].append(keep)
            history['params_keep'].append(start_params)
            history['idx_added'].append(idx)
            if len(idx_nonzero) == len(idx_old) and (idx_nonzero == idx_old).all():
                converged = True
                break
            idx_old = idx_nonzero
        assert np.all(idx_nonzero[:k_keep] == np.arange(k_keep))
        if self.use_weights:
            weights = np.ones(len(idx_nonzero))
            weights[:k_keep] = 0
            penal = self._get_penal(weights=weights)
        else:
            penal = self.penal
        mod_final = model_class(endog, x[:, idx_nonzero], penal=penal, pen_weight=self.pen_weight, **self.init_kwds)
        res_final = mod_final.fit(method=method, start_params=start_params, warn_convergence=False, **fit_kwds)
        xnames = ['var%4d' % ii for ii in idx_nonzero]
        res_final.model.exog_names[k_keep:] = xnames[k_keep:]
        res = ScreeningResults(self, results_pen=res_pen, results_final=res_final, idx_nonzero=idx_nonzero, idx_exog=idx_nonzero[k_keep:] - k_keep, idx_excl=idx_excl, history=history, converged=converged, iterations=it + 1)
        return res

    def screen_exog_iterator(self, exog_iterator):
        if False:
            i = 10
            return i + 15
        '\n        batched version of screen exog\n\n        This screens variables in a two step process:\n\n        In the first step screen_exog is used on each element of the\n        exog_iterator, and the batch winners are collected.\n\n        In the second step all batch winners are combined into a new array\n        of exog candidates and `screen_exog` is used to select a final\n        model.\n\n        Parameters\n        ----------\n        exog_iterator : iterator over ndarrays\n\n        Returns\n        -------\n        res_screen_final : instance of ScreeningResults\n            This is the instance returned by the second round call to\n            `screen_exog`. Additional attributes are added to provide\n            more information about the batched selection process.\n            The index of final nonzero variables is\n            `idx_nonzero_batches` which is a 2-dimensional array with batch\n            index in the first column and variable index within batch in the\n            second column. They can be used jointly as index for the data\n            in the exog_iterator.\n            see ScreeningResults for a full description\n        '
        k_keep = self.k_keep
        res_idx = []
        exog_winner = []
        exog_idx = []
        for ex in exog_iterator:
            res_screen = self.screen_exog(ex, maxiter=20)
            res_idx.append(res_screen.idx_nonzero)
            exog_winner.append(ex[:, res_screen.idx_nonzero[k_keep:] - k_keep])
            exog_idx.append(res_screen.idx_nonzero[k_keep:] - k_keep)
        exog_winner = np.column_stack(exog_winner)
        res_screen_final = self.screen_exog(exog_winner, maxiter=20)
        exog_winner_names = ['var%d_%d' % (bidx, idx) for (bidx, batch) in enumerate(exog_idx) for idx in batch]
        idx_full = [(bidx, idx) for (bidx, batch) in enumerate(exog_idx) for idx in batch]
        ex_final_idx = res_screen_final.idx_nonzero[k_keep:] - k_keep
        final_names = np.array(exog_winner_names)[ex_final_idx]
        res_screen_final.idx_nonzero_batches = np.array(idx_full)[ex_final_idx]
        res_screen_final.exog_final_names = final_names
        history = {'idx_nonzero': res_idx, 'idx_exog': exog_idx}
        res_screen_final.history_batches = history
        return res_screen_final