"""
Return in excess of a given benchmark.

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).

Work "Evaluating multiple classifiers for stock price direction prediction" by Ballings et al. (2015) uses this method
to label yearly returns over a predetermined value to compare the performance of several machine learning algorithms.
"""
import numpy as np
import pandas as pd

def return_over_benchmark(prices, benchmark=0, binary=False, resample_by=None, lag=True):
    if False:
        return 10
    "\n    Return over benchmark labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,\n    by Coqueret, G. and Guida, T. (2020).\n\n    Returns a Series or DataFrame of numerical or categorical returns over a given benchmark. The time index of the\n    benchmark must match those of the price observations.\n\n    :param prices: (pd.Series or pd.DataFrame) Time indexed prices to compare returns against a benchmark.\n    :param benchmark: (pd.Series or float) Benchmark of returns to compare the returns from prices against for labeling.\n                    Can be a constant value, or a Series matching the index of prices. If no benchmark is given, then it\n                    is assumed to have a constant value of 0.\n    :param binary: (bool) If False, labels are given by their numerical value of return over benchmark. If True,\n                labels are given according to the sign of their excess return.\n    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per\n                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.\n                        For full details see `here.\n                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_\n    :param lag: (bool) If True, returns will be lagged to make them forward-looking.\n    :return: (pd.Series or pd.DataFrame) Excess returns over benchmark. If binary, the labels are -1 if the\n            return is below the benchmark, 1 if above, and 0 if it exactly matches the benchmark.\n    "
    pass