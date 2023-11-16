"""
Sandbox Panel Estimators

References
-----------

Baltagi, Badi H. `Econometric Analysis of Panel Data.` 4th ed. Wiley, 2008.
"""
from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
__all__ = ['PanelModel']
from pandas import Panel

def group(X):
    if False:
        return 10
    "\n    Returns unique numeric values for groups without sorting.\n\n    Examples\n    --------\n    >>> X = np.array(['a','a','b','c','b','c'])\n    >>> group(X)\n    >>> g\n    array([ 0.,  0.,  1.,  2.,  1.,  2.])\n    "
    uniq_dict = {}
    group = np.zeros(len(X))
    for i in range(len(X)):
        if not X[i] in uniq_dict:
            uniq_dict.update({X[i]: len(uniq_dict)})
        group[i] = uniq_dict[X[i]]
    return group

def repanel_cov(groups, sigmas):
    if False:
        print('Hello World!')
    'calculate error covariance matrix for random effects model\n\n    Parameters\n    ----------\n    groups : ndarray, (nobs, nre) or (nobs,)\n        array of group/category observations\n    sigma : ndarray, (nre+1,)\n        array of standard deviations of random effects,\n        last element is the standard deviation of the\n        idiosyncratic error\n\n    Returns\n    -------\n    omega : ndarray, (nobs, nobs)\n        covariance matrix of error\n    omegainv : ndarray, (nobs, nobs)\n        inverse covariance matrix of error\n    omegainvsqrt : ndarray, (nobs, nobs)\n        squareroot inverse covariance matrix of error\n        such that omega = omegainvsqrt * omegainvsqrt.T\n\n    Notes\n    -----\n    This does not use sparse matrices and constructs nobs by nobs\n    matrices. Also, omegainvsqrt is not sparse, i.e. elements are non-zero\n    '
    if groups.ndim == 1:
        groups = groups[:, None]
    (nobs, nre) = groups.shape
    omega = sigmas[-1] * np.eye(nobs)
    for igr in range(nre):
        group = groups[:, igr:igr + 1]
        groupuniq = np.unique(group)
        dummygr = sigmas[igr] * (group == groupuniq).astype(float)
        omega += np.dot(dummygr, dummygr.T)
    (ev, evec) = np.linalg.eigh(omega)
    omegainv = np.dot(evec, (1 / ev * evec).T)
    omegainvhalf = evec / np.sqrt(ev)
    return (omega, omegainv, omegainvhalf)

class PanelData(Panel):
    pass

class PanelModel:
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ----------
    endog : array_like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
#    exog
#    panel_arr
#    time_arr
    panel_data : pandas.Panel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """

    def __init__(self, endog=None, exog=None, panel=None, time=None, xtnames=None, equation=None, panel_data=None):
        if False:
            while True:
                i = 10
        if panel_data is None:
            self.initialize(endog, exog, panel, time, xtnames, equation)

    def initialize(self, endog, exog, panel, time, xtnames, equation):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize plain array model.\n\n        See PanelModel\n        '
        names = equation.split(' ')
        self.endog_name = names[0]
        exog_names = names[1:]
        self.panel_name = xtnames[0]
        self.time_name = xtnames[1]
        novar = exog.var(0) == 0
        if True in novar:
            cons_index = np.where(novar == 1)[0][0]
            exog_names.insert(cons_index, 'cons')
        self._cons_index = novar
        self.exog_names = exog_names
        self.endog = np.squeeze(np.asarray(endog))
        exog = np.asarray(exog)
        self.exog = exog
        self.panel = np.asarray(panel)
        self.time = np.asarray(time)
        self.paneluniq = np.unique(panel)
        self.timeuniq = np.unique(time)

    def initialize_pandas(self, panel_data, endog_name, exog_name):
        if False:
            print('Hello World!')
        self.panel_data = panel_data
        endog = panel_data[endog_name].values
        self.endog = np.squeeze(endog)
        if exog_name is None:
            exog_name = panel_data.columns.tolist()
            exog_name.remove(endog_name)
        self.exog = panel_data.filterItems(exog_name).values
        self._exog_name = exog_name
        self._endog_name = endog_name
        self._timeseries = panel_data.major_axis
        self._panelseries = panel_data.minor_axis

    def _group_mean(self, X, index='oneway', counts=False, dummies=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get group means of X by time or by panel.\n\n        index default is panel\n        '
        if index == 'oneway':
            Y = self.panel
            uniq = self.paneluniq
        elif index == 'time':
            Y = self.time
            uniq = self.timeuniq
        else:
            raise ValueError('index %s not understood' % index)
        print(Y, uniq, uniq[:, None], len(Y), len(uniq), len(uniq[:, None]), index)
        dummy = (Y == uniq[:, None]).astype(float)
        if X.ndim > 1:
            mean = np.dot(dummy, X) / dummy.sum(1)[:, None]
        else:
            mean = np.dot(dummy, X) / dummy.sum(1)
        if counts is False and dummies is False:
            return mean
        elif counts is True and dummies is False:
            return (mean, dummy.sum(1))
        elif counts is True and dummies is True:
            return (mean, dummy.sum(1), dummy)
        elif counts is False and dummies is True:
            return (mean, dummy)

    def fit(self, model=None, method=None, effects='oneway'):
        if False:
            print('Hello World!')
        '\n        method : LSDV, demeaned, MLE, GLS, BE, FE, optional\n        model :\n                between\n                fixed\n                random\n                pooled\n                [gmm]\n        effects :\n                oneway\n                time\n                twoway\n        femethod : demeaned (only one implemented)\n                   WLS\n        remethod :\n                swar -\n                amemiya\n                nerlove\n                walhus\n\n\n        Notes\n        -----\n        This is unfinished.  None of the method arguments work yet.\n        Only oneway effects should work.\n        '
        if method:
            method = method.lower()
        model = model.lower()
        if method and method not in ['lsdv', 'demeaned', 'mle', 'gls', 'be', 'fe']:
            raise ValueError('%s not a valid method' % method)
        if model == 'pooled':
            return GLS(self.endog, self.exog).fit()
        if model == 'between':
            return self._fit_btwn(method, effects)
        if model == 'fixed':
            return self._fit_fixed(method, effects)

    def _fit_btwn(self, method, effects):
        if False:
            i = 10
            return i + 15
        if effects != 'twoway':
            endog = self._group_mean(self.endog, index=effects)
            exog = self._group_mean(self.exog, index=effects)
        else:
            raise ValueError('%s effects is not valid for the between estimator' % effects)
        befit = GLS(endog, exog).fit()
        return befit

    def _fit_fixed(self, method, effects):
        if False:
            i = 10
            return i + 15
        endog = self.endog
        exog = self.exog
        demeantwice = False
        if effects in ['oneway', 'twoways']:
            if effects == 'twoways':
                demeantwice = True
                effects = 'oneway'
            (endog_mean, counts) = self._group_mean(endog, index=effects, counts=True)
            exog_mean = self._group_mean(exog, index=effects)
            counts = counts.astype(int)
            endog = endog - np.repeat(endog_mean, counts)
            exog = exog - np.repeat(exog_mean, counts, axis=0)
        if demeantwice or effects == 'time':
            (endog_mean, dummies) = self._group_mean(endog, index='time', dummies=True)
            exog_mean = self._group_mean(exog, index='time')
            endog = endog - np.dot(endog_mean, dummies)
            exog = exog - np.dot(dummies.T, exog_mean)
        fefit = GLS(endog, exog[:, -self._cons_index]).fit()
        return fefit

class SURPanel(PanelModel):
    pass

class SEMPanel(PanelModel):
    pass

class DynamicPanel(PanelModel):
    pass
if __name__ == '__main__':
    import numpy.lib.recfunctions as nprf
    import pandas
    from pandas import Panel
    import statsmodels.api as sm
    data = sm.datasets.grunfeld.load()
    endog = data.endog[:-20]
    fullexog = data.exog[:-20]
    panel_arr = nprf.append_fields(fullexog, 'investment', endog, float, usemask=False)
    panel_df = pandas.DataFrame(panel_arr)
    panel_panda = panel_df.set_index(['year', 'firm']).to_panel()
    exog = fullexog[['value', 'capital']].view(float).reshape(-1, 2)
    exog = sm.add_constant(exog, prepend=False)
    panel = group(fullexog['firm'])
    year = fullexog['year']
    panel_mod = PanelModel(endog, exog, panel, year, xtnames=['firm', 'year'], equation='invest value capital')
    panel_ols = panel_mod.fit(model='pooled')
    panel_be = panel_mod.fit(model='between', effects='oneway')
    panel_fe = panel_mod.fit(model='fixed', effects='oneway')
    panel_bet = panel_mod.fit(model='between', effects='time')
    panel_fet = panel_mod.fit(model='fixed', effects='time')
    panel_fe2 = panel_mod.fit(model='fixed', effects='twoways')
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    nobs = groups.shape[0]
    groupuniq = np.unique(groups)
    periods = np.array([0, 1, 2, 1, 2, 0, 1, 2])
    perioduniq = np.unique(periods)
    dummygr = (groups[:, None] == groupuniq).astype(float)
    dummype = (periods[:, None] == perioduniq).astype(float)
    sigma = 1.0
    sigmagr = np.sqrt(2.0)
    sigmape = np.sqrt(3.0)
    dummyall = np.c_[sigmagr * dummygr, sigmape * dummype]
    omega = np.dot(dummyall, dummyall.T) + sigma * np.eye(nobs)
    print(omega)
    print(np.linalg.cholesky(omega))
    (ev, evec) = np.linalg.eigh(omega)
    omegainv = np.dot(evec, (1 / ev * evec).T)
    omegainv2 = np.linalg.inv(omega)
    omegacomp = np.dot(evec, (ev * evec).T)
    print(np.max(np.abs(omegacomp - omega)))
    print(np.max(np.abs(np.dot(omegainv, omega) - np.eye(nobs))))
    omegainvhalf = evec / np.sqrt(ev)
    print(np.max(np.abs(np.dot(omegainvhalf, omegainvhalf.T) - omegainv)))
    sigmas2 = np.array([sigmagr, sigmape, sigma])
    groups2 = np.column_stack((groups, periods))
    (omega_, omegainv_, omegainvhalf_) = repanel_cov(groups2, sigmas2)
    print(np.max(np.abs(omega_ - omega)))
    print(np.max(np.abs(omegainv_ - omegainv)))
    print(np.max(np.abs(omegainvhalf_ - omegainvhalf)))
    Pgr = reduce(np.dot, [dummygr, np.linalg.inv(np.dot(dummygr.T, dummygr)), dummygr.T])
    Qgr = np.eye(nobs) - Pgr
    print(np.max(np.abs(np.dot(Qgr, groups))))