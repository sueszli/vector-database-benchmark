"""
Labeling Raw Returns.

Most basic form of labeling based on raw return of each observation relative to its previous value.
"""
import numpy as np

def raw_return(prices, binary=False, logarithmic=False, resample_by=None, lag=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Raw returns labeling method.\n\n    This is the most basic and ubiquitous labeling method used as a precursor to almost any kind of financial data\n    analysis or machine learning. User can specify simple or logarithmic returns, numerical or binary labels, a\n    resample period, and whether returns are lagged to be forward looking.\n\n    :param prices: (pd.Series or pd.DataFrame) Time-indexed price data on stocks with which to calculate return.\n    :param binary: (bool) If False, will return numerical returns. If True, will return the sign of the raw return.\n    :param logarithmic: (bool) If False, will calculate simple returns. If True, will calculate logarithmic returns.\n    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per\n                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.\n                        For full details see `here.\n                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_\n    :param lag: (bool) If True, returns will be lagged to make them forward-looking.\n    :return:  (pd.Series or pd.DataFrame) Raw returns on market data. User can specify whether returns will be based on\n                simple or logarithmic return, and whether the output will be numerical or categorical.\n    "
    pass