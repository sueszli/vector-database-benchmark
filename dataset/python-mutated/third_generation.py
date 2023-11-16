"""
Third generation models implementation (VPIN)
"""
import pandas as pd

def get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int=1) -> pd.Series:
    if False:
        i = 10
        return i + 15
    '\n    Advances in Financial Machine Learning, p. 292-293.\n\n    Get Volume-Synchronized Probability of Informed Trading (VPIN) from bars\n\n    :param volume: (pd.Series) Bar volume\n    :param buy_volume: (pd.Series) Bar volume classified as buy (either tick rule, BVC or aggressor side methods applied)\n    :param window: (int) Estimation window\n    :return: (pd.Series) VPIN series\n    '
    pass