import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import statsmodels.api as sm
import theano
import theano.tensor as tt
from pandas.plotting import register_matplotlib_converters
from pandas_datareader.data import DataReader
plt.style.use('seaborn')
register_matplotlib_converters()
cpi = DataReader('CPIAUCNS', 'fred', start='1971-01', end='2018-12')
cpi.index = pd.DatetimeIndex(cpi.index, freq='MS')
inf = np.log(cpi).resample('QS').mean().diff()[1:] * 400
inf = inf.dropna()
print(inf.head())
(fig, ax) = plt.subplots(figsize=(9, 4), dpi=300)
ax.plot(inf.index, inf, label='$\\Delta \\log CPI$', lw=2)
ax.legend(loc='lower left')
plt.show()
mod = sm.tsa.statespace.SARIMAX(inf, order=(1, 0, 1))
res_mle = mod.fit(disp=False)
print(res_mle.summary())
predict_mle = res_mle.get_prediction()
predict_mle_ci = predict_mle.conf_int()
lower = predict_mle_ci['lower CPIAUCNS']
upper = predict_mle_ci['upper CPIAUCNS']
(fig, ax) = plt.subplots(figsize=(9, 4), dpi=300)
inf.plot(ax=ax, style='-', label='Observed')
predict_mle.predicted_mean.plot(ax=ax, style='r.', label='One-step-ahead forecast')
ax.fill_between(predict_mle_ci.index, lower, upper, color='r', alpha=0.1)
ax.legend(loc='lower left')
plt.show()

class Loglike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, model):
        if False:
            return 10
        self.model = model
        self.score = Score(self.model)

    def perform(self, node, inputs, outputs):
        if False:
            return 10
        (theta,) = inputs
        llf = self.model.loglike(theta)
        outputs[0][0] = np.array(llf)

    def grad(self, inputs, g):
        if False:
            i = 10
            return i + 15
        (theta,) = inputs
        out = [g[0] * self.score(theta)]
        return out

class Score(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, model):
        if False:
            i = 10
            return i + 15
        self.model = model

    def perform(self, node, inputs, outputs):
        if False:
            print('Hello World!')
        (theta,) = inputs
        outputs[0][0] = self.model.score(theta)
ndraws = 3000
nburn = 600
loglike = Loglike(mod)
with pm.Model() as m:
    arL1 = pm.Uniform('ar.L1', -0.99, 0.99)
    maL1 = pm.Uniform('ma.L1', -0.99, 0.99)
    sigma2 = pm.InverseGamma('sigma2', 2, 4)
    theta = tt.as_tensor_variable([arL1, maL1, sigma2])
    pm.DensityDist('likelihood', loglike, observed=theta)
    trace = pm.sample(ndraws, tune=nburn, return_inferencedata=True, cores=1, compute_convergence_checks=False)
plt.tight_layout()
_ = pm.plot_trace(trace, lines=[(k, {}, [v]) for (k, v) in dict(res_mle.params).items()], combined=True, figsize=(12, 12))
pm.summary(trace)
params = pm.summary(trace)['mean'].values
res_bayes = mod.smooth(params)
predict_bayes = res_bayes.get_prediction()
predict_bayes_ci = predict_bayes.conf_int()
lower = predict_bayes_ci['lower CPIAUCNS']
upper = predict_bayes_ci['upper CPIAUCNS']
(fig, ax) = plt.subplots(figsize=(9, 4), dpi=300)
inf.plot(ax=ax, style='-', label='Observed')
predict_bayes.predicted_mean.plot(ax=ax, style='r.', label='One-step-ahead forecast')
ax.fill_between(predict_bayes_ci.index, lower, upper, color='r', alpha=0.1)
ax.legend(loc='lower left')
plt.show()
mod_uc = sm.tsa.UnobservedComponents(inf, 'rwalk', autoregressive=1)
res_uc_mle = mod_uc.fit()
print(res_uc_mle.summary())
ndraws = 3000
nburn = 600
loglike_uc = Loglike(mod_uc)
with pm.Model():
    sigma2level = pm.InverseGamma('sigma2.level', 1, 1)
    sigma2ar = pm.InverseGamma('sigma2.ar', 1, 1)
    arL1 = pm.Uniform('ar.L1', -0.99, 0.99)
    theta_uc = tt.as_tensor_variable([sigma2level, sigma2ar, arL1])
    pm.DensityDist('likelihood', loglike_uc, observed=theta_uc)
    trace_uc = pm.sample(ndraws, tune=nburn, return_inferencedata=True, cores=1, compute_convergence_checks=False)
plt.tight_layout()
_ = pm.plot_trace(trace_uc, lines=[(k, {}, [v]) for (k, v) in dict(res_uc_mle.params).items()], combined=True, figsize=(12, 12))
pm.summary(trace_uc)
params = pm.summary(trace_uc)['mean'].values
res_uc_bayes = mod_uc.smooth(params)
(fig, ax) = plt.subplots(figsize=(9, 4), dpi=300)
inf['CPIAUCNS'].plot(ax=ax, style='-', label='Observed data')
res_uc_mle.states.smoothed['level'].plot(ax=ax, label='Smoothed level (MLE)')
res_uc_bayes.states.smoothed['level'].plot(ax=ax, label='Smoothed level (Bayesian)')
ax.legend(loc='lower left')