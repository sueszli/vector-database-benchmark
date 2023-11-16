import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
macrodata = sm.datasets.macrodata.load_pandas().data
macrodata.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
endog = macrodata['infl']
endog.plot(figsize=(15, 5))
mod = sm.tsa.SARIMAX(endog, order=(1, 0, 0), trend='c')
res = mod.fit()
print(res.summary())
print(res.forecast())
fcast_res1 = res.get_forecast()
print(fcast_res1.summary_frame(alpha=0.1))
print(res.forecast(steps=2))
fcast_res2 = res.get_forecast(steps=2)
print(fcast_res2.summary_frame())
print(res.forecast('2010Q2'))
fcast_res3 = res.get_forecast('2010Q2')
print(fcast_res3.summary_frame())
(fig, ax) = plt.subplots(figsize=(15, 5))
endog.loc['1999':].plot(ax=ax)
fcast = res.get_forecast('2011Q4').summary_frame()
fcast['mean'].plot(ax=ax, style='k--')
ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)
training_obs = int(len(endog) * 0.8)
training_endog = endog[:training_obs]
training_mod = sm.tsa.SARIMAX(training_endog, order=(1, 0, 0), trend='c')
training_res = training_mod.fit()
print(training_res.params)
fcast = training_res.forecast()
true = endog.reindex(fcast.index)
error = true - fcast
print(pd.concat([true.rename('true'), fcast.rename('forecast'), error.rename('error')], axis=1))
append_res = training_res.append(endog[training_obs:training_obs + 1], refit=True)
print(append_res.params)
fcast = append_res.forecast()
true = endog.reindex(fcast.index)
error = true - fcast
print(pd.concat([true.rename('true'), fcast.rename('forecast'), error.rename('error')], axis=1))
nforecasts = 3
forecasts = {}
nobs = len(endog)
n_init_training = int(nobs * 0.8)
init_training_endog = endog.iloc[:n_init_training]
mod = sm.tsa.SARIMAX(training_endog, order=(1, 0, 0), trend='c')
res = mod.fit()
forecasts[training_endog.index[-1]] = res.forecast(steps=nforecasts)
for t in range(n_init_training, nobs):
    updated_endog = endog.iloc[t:t + 1]
    res = res.append(updated_endog, refit=False)
    forecasts[updated_endog.index[0]] = res.forecast(steps=nforecasts)
forecasts = pd.concat(forecasts, axis=1)
print(forecasts.iloc[:5, :5])
forecast_errors = forecasts.apply(lambda column: endog - column).reindex(forecasts.index)
print(forecast_errors.iloc[:5, :5])

def flatten(column):
    if False:
        i = 10
        return i + 15
    return column.dropna().reset_index(drop=True)
flattened = forecast_errors.apply(flatten)
flattened.index = (flattened.index + 1).rename('horizon')
print(flattened.iloc[:3, :5])
rmse = (flattened ** 2).mean(axis=1) ** 0.5
print(rmse)
nforecasts = 3
forecasts = {}
nobs = len(endog)
n_init_training = int(nobs * 0.8)
init_training_endog = endog.iloc[:n_init_training]
mod = sm.tsa.SARIMAX(training_endog, order=(1, 0, 0), trend='c')
res = mod.fit()
forecasts[training_endog.index[-1]] = res.forecast(steps=nforecasts)
for t in range(n_init_training, nobs):
    updated_endog = endog.iloc[t:t + 1]
    res = res.extend(updated_endog)
    forecasts[updated_endog.index[0]] = res.forecast(steps=nforecasts)
forecasts = pd.concat(forecasts, axis=1)
print(forecasts.iloc[:5, :5])
forecast_errors = forecasts.apply(lambda column: endog - column).reindex(forecasts.index)
print(forecast_errors.iloc[:5, :5])

def flatten(column):
    if False:
        return 10
    return column.dropna().reset_index(drop=True)
flattened = forecast_errors.apply(flatten)
flattened.index = (flattened.index + 1).rename('horizon')
print(flattened.iloc[:3, :5])
rmse = (flattened ** 2).mean(axis=1) ** 0.5
print(rmse)
print(endog.index)
index = pd.period_range(start='2000', periods=4, freq='A')
endog1 = pd.Series([1, 2, 3, 4], index=index)
print(endog1.index)
index = pd.date_range(start='2000', periods=4, freq='QS')
endog2 = pd.Series([1, 2, 3, 4], index=index)
print(endog2.index)
index = pd.date_range(start='2000', periods=4, freq='M')
endog3 = pd.Series([1, 2, 3, 4], index=index)
print(endog3.index)
index = pd.DatetimeIndex(['2000-01-01 10:08am', '2000-01-01 11:32am', '2000-01-01 5:32pm', '2000-01-02 6:15am'])
endog4 = pd.Series([0.2, 0.5, -0.1, 0.1], index=index)
print(endog4.index)
mod = sm.tsa.SARIMAX(endog4)
res = mod.fit()
res.forecast(1)
try:
    res.forecast('2000-01-03')
except KeyError as e:
    print(e)