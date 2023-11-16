import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
sunspots = sm.datasets.sunspots.load_pandas().data
sunspots.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del sunspots['YEAR']
sunspots.plot(figsize=(12, 8))
from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    if False:
        print('Hello World!')
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for (key, value) in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
from statsmodels.tsa.stattools import kpss

def kpss_test(timeseries):
    if False:
        for i in range(10):
            print('nop')
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags='auto')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for (key, value) in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
adf_test(sunspots['SUNACTIVITY'])
kpss_test(sunspots['SUNACTIVITY'])
sunspots['SUNACTIVITY_diff'] = sunspots['SUNACTIVITY'] - sunspots['SUNACTIVITY'].shift(1)
sunspots['SUNACTIVITY_diff'].dropna().plot(figsize=(12, 8))
adf_test(sunspots['SUNACTIVITY_diff'].dropna())
kpss_test(sunspots['SUNACTIVITY_diff'].dropna())