import numpy as np
import pandas
from statsmodels.api import datasets as ds
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
mdata = ds.macrodata.load_pandas().data
dates = mdata[['year', 'quarter']].astype(int)
quarterly = [str(yr) + 'Q' + str(mo) for (yr, mo) in zip(dates['year'], dates['quarter'])]
quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp', 'realcons', 'realinv']]
mdata.index = pandas.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()
model = VAR(data)
est = model.fit(maxlags=2)

def plot_input():
    if False:
        for i in range(10):
            print('nop')
    est.plot()

def plot_acorr():
    if False:
        return 10
    est.plot_acorr()

def plot_irf():
    if False:
        while True:
            i = 10
    est.irf().plot()

def plot_irf_cum():
    if False:
        return 10
    irf = est.irf()
    irf.plot_cum_effects()

def plot_forecast():
    if False:
        return 10
    est.plot_forecast(10)

def plot_fevd():
    if False:
        return 10
    est.fevd(20).plot()