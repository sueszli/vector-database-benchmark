"""
Explosiveness tests: Chow-Type Dickey-Fuller Test
"""
import pandas as pd
from mlfinlab.structural_breaks.sadf import get_betas
from mlfinlab.util import mp_pandas_obj

def _get_dfc_for_t(series: pd.Series, molecule: list) -> pd.Series:
    if False:
        return 10
    '\n    Get Chow-Type Dickey-Fuller Test statistics for each index in molecule\n\n    :param series: (pd.Series) Series to test\n    :param molecule: (list) Dates to test\n    :return: (pd.Series) Statistics for each index from molecule\n    '
    pass

def get_chow_type_stat(series: pd.Series, min_length: int=20, num_threads: int=8, verbose: bool=True) -> pd.Series:
    if False:
        while True:
            i = 10
    '\n    Multithread implementation of Chow-Type Dickey-Fuller Test, p.251-252\n\n    :param series: (pd.Series) Series to test\n    :param min_length: (int) Minimum sample length used to estimate statistics\n    :param num_threads: (int): Number of cores to use\n    :param verbose: (bool) Flag to report progress on asynch jobs\n    :return: (pd.Series) Chow-Type Dickey-Fuller Test statistics\n    '
    pass