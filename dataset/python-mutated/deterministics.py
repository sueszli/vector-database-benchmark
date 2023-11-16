import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rc('figure', figsize=(16, 9))
plt.rc('font', size=16)
from statsmodels.tsa.deterministic import DeterministicProcess
index = pd.RangeIndex(0, 100)
det_proc = DeterministicProcess(index, constant=True, order=1, seasonal=True, period=5)
det_proc.in_sample()
det_proc.out_of_sample(15)
det_proc.range(190, 210)
index = pd.period_range('2020-03-01', freq='M', periods=60)
det_proc = DeterministicProcess(index, constant=True, fourier=2)
det_proc.in_sample().head(12)
det_proc.out_of_sample(12)
det_proc.range('2025-01', '2026-01')
det_proc.range(58, 70)
from statsmodels.tsa.deterministic import Fourier, Seasonality, TimeTrend
index = pd.period_range('2020-03-01', freq='D', periods=2 * 365)
tt = TimeTrend(constant=True)
four = Fourier(period=365.25, order=2)
seas = Seasonality(period=7)
det_proc = DeterministicProcess(index, additional_terms=[tt, seas, four])
det_proc.in_sample().head(28)
from statsmodels.tsa.deterministic import DeterministicTerm

class BrokenTimeTrend(DeterministicTerm):

    def __init__(self, break_period: int):
        if False:
            return 10
        self._break_period = break_period

    def __str__(self):
        if False:
            print('Hello World!')
        return 'Broken Time Trend'

    def _eq_attr(self):
        if False:
            while True:
                i = 10
        return (self._break_period,)

    def in_sample(self, index: pd.Index):
        if False:
            print('Hello World!')
        nobs = index.shape[0]
        terms = np.zeros((nobs, 2))
        terms[self._break_period:, 0] = 1
        terms[self._break_period:, 1] = np.arange(self._break_period + 1, nobs + 1)
        return pd.DataFrame(terms, columns=['const_break', 'trend_break'], index=index)

    def out_of_sample(self, steps: int, index: pd.Index, forecast_index: pd.Index=None):
        if False:
            print('Hello World!')
        fcast_index = self._extend_index(index, steps, forecast_index)
        nobs = index.shape[0]
        terms = np.zeros((steps, 2))
        terms[:, 0] = 1
        terms[:, 1] = np.arange(nobs + 1, nobs + steps + 1)
        return pd.DataFrame(terms, columns=['const_break', 'trend_break'], index=fcast_index)
btt = BrokenTimeTrend(60)
tt = TimeTrend(constant=True, order=1)
index = pd.RangeIndex(100)
det_proc = DeterministicProcess(index, additional_terms=[tt, btt])
det_proc.range(55, 65)

class ExogenousProcess(DeterministicTerm):

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self._data = data

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Custom Exog Process'

    def _eq_attr(self):
        if False:
            print('Hello World!')
        return (id(self._data),)

    def in_sample(self, index: pd.Index):
        if False:
            for i in range(10):
                print('nop')
        return self._data.loc[index]

    def out_of_sample(self, steps: int, index: pd.Index, forecast_index: pd.Index=None):
        if False:
            return 10
        forecast_index = self._extend_index(index, steps, forecast_index)
        return self._data.loc[forecast_index]
import numpy as np
gen = np.random.default_rng(98765432101234567890)
exog = pd.DataFrame(gen.integers(100, size=(300, 2)), columns=['exog1', 'exog2'])
exog.head()
ep = ExogenousProcess(exog)
tt = TimeTrend(constant=True, order=1)
idx = exog.index[:200]
det_proc = DeterministicProcess(idx, additional_terms=[tt, ep])
det_proc.in_sample().head()
det_proc.out_of_sample(10)
gen = np.random.default_rng(98765432101234567890)
idx = pd.RangeIndex(200)
det_proc = DeterministicProcess(idx, constant=True, period=52, fourier=2)
det_terms = det_proc.in_sample().to_numpy()
params = np.array([1.0, 3, -1, 4, -2])
exog = det_terms @ params
y = np.empty(200)
y[0] = det_terms[0] @ params + gen.standard_normal()
for i in range(1, 200):
    y[i] = 0.9 * y[i - 1] + det_terms[i] @ params + gen.standard_normal()
y = pd.Series(y, index=idx)
ax = y.plot()
from statsmodels.tsa.api import AutoReg
mod = AutoReg(y, 1, trend='n', deterministic=det_proc)
res = mod.fit()
print(res.summary())
fig = res.plot_predict(200, 200 + 2 * 52, True)
auto_reg_forecast = res.predict(200, 211)
auto_reg_forecast
from statsmodels.tsa.api import SARIMAX
det_proc = DeterministicProcess(idx, period=52, fourier=2)
det_terms = det_proc.in_sample()
mod = SARIMAX(y, order=(1, 0, 0), trend='c', exog=det_terms)
res = mod.fit(disp=False)
print(res.summary())
sarimax_forecast = res.forecast(12, exog=det_proc.out_of_sample(12))
df = pd.concat([auto_reg_forecast, sarimax_forecast], axis=1)
df.columns = columns = ['AutoReg', 'SARIMAX']
df