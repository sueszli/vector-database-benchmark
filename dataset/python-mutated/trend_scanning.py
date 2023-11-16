"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""
import pandas as pd
import numpy as np
from mlfinlab.structural_breaks.sadf import get_betas

def trend_scanning_labels(price_series: pd.Series, t_events: list=None, observation_window: int=20, look_forward: bool=True, min_sample_length: int=5, step: int=1) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    "\n    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and\n    regression labeling technique.\n\n    That can be used in the following ways:\n\n    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the\n       trends as either downward or upward.\n    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,\n       upward.\n    3. The t-values can be used as sample weights in classification problems.\n    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.\n\n    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for\n    the trend, and bin.\n\n    This function allows using both forward-looking and backward-looking window (use the look_forward parameter).\n\n    :param price_series: (pd.Series) Close prices used to label the data set\n    :param t_events: (list) Filtered events, array of pd.Timestamps\n    :param observation_window: (int) Maximum look forward window used to get the trend value\n    :param look_forward: (bool) True if using a forward-looking window, False if using a backward-looking one\n    :param min_sample_length: (int) Minimum sample length used to fit regression\n    :param step: (int) Optimal t-value index is searched every 'step' indices\n    :return: (pd.DataFrame) Consists of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,\n        ret - price change %, bin - label value based on price change sign\n    "
    pass