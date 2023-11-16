"""Examples using Pandas

"""
from statsmodels.compat.pandas import frequencies
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.arima_process import arma_generate_sample
data = sm.datasets.stackloss.load()
X = DataFrame(data.exog, columns=data.exog_name)
X['intercept'] = 1.0
Y = Series(data.endog)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
print(results.params)
print(results.cov_params())
infl = results.get_influence()
print(infl.summary_table())
huber_t = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
print(hub_results.params)
print(hub_results.bcov_scaled)
print(hub_results.summary())

def plot_acf_multiple(ys, lags=20):
    if False:
        for i in range(10):
            print('nop')
    '\n    '
    from statsmodels.tsa.stattools import acf
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8
    plt.figure(figsize=(10, 10))
    xs = np.arange(lags + 1)
    acorr = np.apply_along_axis(lambda x: acf(x, nlags=lags), 0, ys)
    k = acorr.shape[1]
    for i in range(k):
        ax = plt.subplot(k, 1, i + 1)
        ax.vlines(xs, [0], acorr[:, i])
        ax.axhline(0, color='k')
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, xs[-1] + 1])
    mpl.rcParams['font.size'] = old_size
data = sm.datasets.macrodata.load()
mdata = data.data
df = DataFrame.from_records(mdata)
quarter_end = frequencies.BQuarterEnd()
df.index = [quarter_end.rollforward(datetime(int(y), int(q) * 3, 1)) for (y, q) in zip(df.pop('year'), df.pop('quarter'))]
logged = np.log(df.loc[:, ['m1', 'realgdp', 'cpi']])
logged.plot(subplots=True)
log_difference = logged.diff().dropna()
plot_acf_multiple(log_difference.values)
model = tsa.VAR(log_difference, freq='BQ')
print(model.select_order())
res = model.fit(2)
print(res.summary())
print(res.is_stable())
irf = res.irf(20)
irf.plot()
fevd = res.fevd()
fevd.plot()
print(res.test_whiteness())
print(res.test_causality('m1', 'realgdp'))
print(res.test_normality())
arparams = np.array([0.75, -0.25])
maparams = np.array([0.65, 0.35])
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]
nobs = 250
y = arma_generate_sample(arparams, maparams, nobs)
plt.figure()
plt.plot(y)
dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2, 2), freq='M')
arma_res = arma_mod.fit(trend='nc', disp=-1)
print(arma_res.params)
plt.show()