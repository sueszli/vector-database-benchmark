"""
Volume classification methods (BVC and tick rule)
"""
from scipy.stats import norm
import pandas as pd

def get_bvc_buy_volume(close: pd.Series, volume: pd.Series, window: int=20) -> pd.Series:
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the BVC buy volume\n\n    :param close: (pd.Series): Close prices\n    :param volume: (pd.Series): Bar volumes\n    :param window: (int): Window for std estimation uses in BVC calculation\n    :return: (pd.Series) BVC buy volume\n    '
    pass