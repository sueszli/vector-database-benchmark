import numpy as np
import pandas as pd
rng = np.random.default_rng(20210819)
eta = rng.standard_normal(5200)
rho = 0.8
beta = 10
epsilon = eta.copy()
for i in range(1, eta.shape[0]):
    epsilon[i] = rho * epsilon[i - 1] + eta[i]
y = beta + epsilon
y = y[200:]
from statsmodels.tsa.api import SARIMAX, AutoReg
from statsmodels.tsa.arima.model import ARIMA
ar0_res = SARIMAX(y, order=(0, 0, 0), trend='c').fit()
sarimax_res = SARIMAX(y, order=(1, 0, 0), trend='c').fit()
arima_res = ARIMA(y, order=(1, 0, 0), trend='c').fit()
autoreg_res = AutoReg(y, 1, trend='c').fit()
intercept = [ar0_res.params[0], sarimax_res.params[0], arima_res.params[0], autoreg_res.params[0]]
rho_hat = [0] + [r.params[1] for r in (sarimax_res, arima_res, autoreg_res)]
long_run = [ar0_res.params[0], sarimax_res.params[0] / (1 - sarimax_res.params[1]), arima_res.params[0], autoreg_res.params[0] / (1 - autoreg_res.params[1])]
cols = ['AR(0)', 'SARIMAX', 'ARIMA', 'AutoReg']
pd.DataFrame([intercept, rho_hat, long_run], columns=cols, index=['delta-or-phi', 'rho', 'long-run mean'])
sarimax_exog_res = SARIMAX(y, exog=np.ones_like(y), order=(1, 0, 0), trend='n').fit()
print(sarimax_exog_res.summary())
full_x = rng.standard_normal(eta.shape)
x = full_x[200:]
y += 3 * x
sarimax_exog_res = SARIMAX(y, exog=x, order=(1, 0, 0), trend='c').fit()
arima_exog_res = ARIMA(y, exog=x, order=(1, 0, 0), trend='c').fit()

def print_params(s):
    if False:
        for i in range(10):
            print('nop')
    from io import StringIO
    return pd.read_csv(StringIO(s.tables[1].as_csv()), index_col=0)
print_params(sarimax_exog_res.summary())
print_params(arima_exog_res.summary())
autoreg_exog_res = AutoReg(y, 1, exog=x, trend='c').fit()
print_params(autoreg_exog_res.summary())
y = beta + eta
epsilon = eta.copy()
for i in range(1, eta.shape[0]):
    y[i] = beta * (1 - rho) + rho * y[i - 1] + 3 * full_x[i] + eta[i]
y = y[200:]
autoreg_alt_exog_res = AutoReg(y, 1, exog=x, trend='c').fit()
print_params(autoreg_alt_exog_res.summary())
arima_res = ARIMA(y, order=(1, 0, 0), trend='c').fit()
print_params(arima_res.summary())
arima_res.predict(0, 2)
(delta_hat, rho_hat) = arima_res.params[:2]
delta_hat + rho_hat * (y[0] - delta_hat)
sarima_res = SARIMAX(y, order=(1, 0, 0), trend='c').fit()
print_params(sarima_res.summary())
sarima_res.predict(0, 2)
(delta_hat, rho_hat) = sarima_res.params[:2]
delta_hat + rho_hat * y[0]
rho = 0.8
beta = 10
epsilon = eta.copy()
for i in range(1, eta.shape[0]):
    epsilon[i] = rho * eta[i - 1] + eta[i]
y = beta + epsilon
y = y[200:]
ma_res = ARIMA(y, order=(0, 0, 1), trend='c').fit()
print_params(ma_res.summary())
ma_res.predict(1, 5)
ma_res.resid[:5]
(delta_hat, rho_hat) = ma_res.params[:2]
direct = delta_hat + rho_hat * ma_res.resid[:5]
direct
ma_res.predict(1, 5) - direct
t = y.shape[0]
ma_res.predict(t - 3, t - 1)
ma_res.resid[-4:-1]
direct = delta_hat + rho_hat * ma_res.resid[-4:-1]
direct
ma_res.predict(t - 3, t - 1) - direct
rho = 0.8
beta = 2
delta0 = 10
delta1 = 0.5
epsilon = eta.copy()
for i in range(1, eta.shape[0]):
    epsilon[i] = rho * epsilon[i - 1] + eta[i]
t = np.arange(epsilon.shape[0])
y = delta0 + delta1 * t + beta * full_x + epsilon
y = y[200:]
start = np.array([110, delta1, beta, rho, 1])
arx_res = ARIMA(y, exog=x, order=(1, 0, 0), trend='ct').fit()
mod = SARIMAX(y, exog=x, order=(1, 0, 0), trend='ct')
start[:2] *= 1 - rho
sarimax_res = mod.fit(start_params=start, method='bfgs')
print(arx_res.summary())
print(sarimax_res.summary())
import numpy as np
import pandas as pd
rho = 0.8
psi = -0.6
beta = 20
epsilon = eta.copy()
for i in range(13, eta.shape[0]):
    epsilon[i] = rho * epsilon[i - 1] + psi * epsilon[i - 12] - rho * psi * epsilon[i - 13] + eta[i]
y = beta + epsilon
y = y[200:]
res = ARIMA(y, order=(1, 0, 0), trend='c', seasonal_order=(1, 0, 0, 12)).fit()
print(res.summary())
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 10))
plt.rc('font', size=14)
_ = plt.scatter(res.resid[:13], eta[200:200 + 13])
_ = plt.scatter(res.resid[13:37], eta[200 + 13:200 + 37])
rng = np.random.default_rng(20210819)
eta = rng.standard_normal(5200)
rho = 0.8
beta = 20
epsilon = eta.copy()
for i in range(2, eta.shape[0]):
    epsilon[i] = (1 + rho) * epsilon[i - 1] - rho * epsilon[i - 2] + eta[i]
t = np.arange(epsilon.shape[0])
y = beta + 2 * t + epsilon
y = y[200:]
res = ARIMA(y, order=(1, 1, 0), trend='t').fit()
print(res.summary())
res.resid[:5]
res.predict(0, 5)
(res.loglikelihood_burn, res.nobs_diffuse)