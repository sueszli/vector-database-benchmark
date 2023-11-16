"""
Second generation models features: Kyle lambda, Amihud Lambda, Hasbrouck lambda (bar and trade based)
"""
from typing import List
import numpy as np
import pandas as pd
from mlfinlab.structural_breaks.sadf import get_betas

def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int=20) -> pd.Series:
    if False:
        for i in range(10):
            print('nop')
    '\n    Advances in Financial Machine Learning, p. 286-288.\n\n    Get Kyle lambda from bars data\n\n    :param close: (pd.Series) Close prices\n    :param volume: (pd.Series) Bar volume\n    :param window: (int) Rolling window used for estimation\n    :return: (pd.Series) Kyle lambdas\n    '
    pass

def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int=20) -> pd.Series:
    if False:
        for i in range(10):
            print('nop')
    '\n    Advances in Financial Machine Learning, p.288-289.\n\n    Get Amihud lambda from bars data\n\n    :param close: (pd.Series) Close prices\n    :param dollar_volume: (pd.Series) Dollar volumes\n    :param window: (int) rolling window used for estimation\n    :return: (pd.Series) of Amihud lambda\n    '
    pass

def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int=20) -> pd.Series:
    if False:
        print('Hello World!')
    '\n    Advances in Financial Machine Learning, p.289-290.\n\n    Get Hasbrouck lambda from bars data\n\n    :param close: (pd.Series) Close prices\n    :param dollar_volume: (pd.Series) Dollar volumes\n    :param window: (int) Rolling window used for estimation\n    :return: (pd.Series) Hasbrouck lambda\n    '
    pass

def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> List[float]:
    if False:
        i = 10
        return i + 15
    '\n    Advances in Financial Machine Learning, p.286-288.\n\n    Get Kyle lambda from trades data\n\n    :param price_diff: (list) Price diffs\n    :param volume: (list) Trades sizes\n    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)\n    :return: (list) Kyle lambda for a bar and t-value\n    '
    pass

def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> List[float]:
    if False:
        print('Hello World!')
    '\n    Advances in Financial Machine Learning, p.288-289.\n\n    Get Amihud lambda from trades data\n\n    :param log_ret: (list) Log returns\n    :param dollar_volume: (list) Dollar volumes (price * size)\n    :return: (float) Amihud lambda for a bar\n    '
    pass

def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> List[float]:
    if False:
        while True:
            i = 10
    '\n    Advances in Financial Machine Learning, p.289-290.\n\n    Get Hasbrouck lambda from trades data\n\n    :param log_ret: (list) Log returns\n    :param dollar_volume: (list) Dollar volumes (price * size)\n    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)\n    :return: (list) Hasbrouck lambda for a bar and t value\n    '
    pass