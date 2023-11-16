"""
Chapter 3.2 Fixed-Time Horizon Method, in Advances in Financial Machine Learning, by M. L. de Prado.

Work "Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
labeling data this way can be used in training deep neural networks to predict price movements.
"""
import warnings
import pandas as pd

def fixed_time_horizon(prices, threshold=0, resample_by=None, lag=True, standardized=False, window=None):
    if False:
        while True:
            i = 10
    "\n    Fixed-Time Horizon Labeling Method.\n\n    Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.\n\n    Returns 1 if return is greater than the threshold, -1 if less, and 0 if in between. If no threshold is\n    provided then it will simply take the sign of the return.\n\n    :param prices: (pd.Series or pd.DataFrame) Time-indexed stock prices used to calculate returns.\n    :param threshold: (float or pd.Series) When the absolute value of return exceeds the threshold, the observation is\n                    labeled with 1 or -1, depending on the sign of the return. If return is less, it's labeled as 0.\n                    Can be dynamic if threshold is inputted as a pd.Series, and threshold.index must match prices.index.\n                    If resampling is used, the index of threshold must match the index of prices after resampling.\n                    If threshold is negative, then the directionality of the labels will be reversed. If no threshold\n                    is provided, it is assumed to be 0 and the sign of the return is returned.\n    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per\n                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.\n                        For full details see `here.\n                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_\n    :param lag: (bool) If True, returns will be lagged to make them forward-looking.\n    :param standardized: (bool) Whether returns are scaled by mean and standard deviation.\n    :param window: (int) If standardized is True, the rolling window period for calculating the mean and standard\n                    deviation of returns.\n    :return: (pd.Series or pd.DataFrame) -1, 0, or 1 denoting whether the return for each observation is\n                    less/between/greater than the threshold at each corresponding time index. First or last row will be\n                    NaN, depending on lag.\n    "
    pass