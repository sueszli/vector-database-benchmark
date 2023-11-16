"""
Implementation of Chu-Stinchcombe-White test
"""
import pandas as pd
import numpy as np
from mlfinlab.util import mp_pandas_obj

def _get_values_diff(test_type, series, index, ind):
    if False:
        while True:
            i = 10
    "\n    Gets the difference between two values given a test type.\n    :param test_type: (str) Type of the test ['one_sided', 'two_sided']\n    :param series: (pd.Series) Series of values\n    :param index: (pd.Index) primary index\n    :param ind: (pd.Index) secondary index\n    :return: (float) Difference between 2 values\n    "
    pass

def _get_s_n_for_t(series: pd.Series, test_type: str, molecule: list) -> pd.Series:
    if False:
        return 10
    '\n    Get maximum S_n_t value for each value from molecule for Chu-Stinchcombe-White test\n\n    :param series: (pd.Series) Series to get statistics for\n    :param test_type: (str): Two-sided or one-sided test\n    :param molecule: (list) Indices to get test statistics for\n    :return: (pd.Series) Statistics\n    '
    pass

def get_chu_stinchcombe_white_statistics(series: pd.Series, test_type: str='one_sided', num_threads: int=8, verbose: bool=True) -> pd.Series:
    if False:
        print('Hello World!')
    '\n    Multithread Chu-Stinchcombe-White test implementation, p.251\n\n    :param series: (pd.Series) Series to get statistics for\n    :param test_type: (str): Two-sided or one-sided test\n    :param num_threads: (int) Number of cores\n    :param verbose: (bool) Flag to report progress on asynch jobs\n    :return: (pd.Series) Statistics\n    '
    pass