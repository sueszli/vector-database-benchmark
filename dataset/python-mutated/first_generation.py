"""
First generation features (Roll Measure/Impact, Corwin-Schultz spread estimator)
"""
import numpy as np
import pandas as pd

def get_roll_measure(close_prices: pd.Series, window: int=20) -> pd.Series:
    if False:
        print('Hello World!')
    '\n    Advances in Financial Machine Learning, page 282.\n\n    Get Roll Measure\n\n    Roll Measure gives the estimate of effective bid-ask spread\n    without using quote-data.\n\n    :param close_prices: (pd.Series) Close prices\n    :param window: (int) Estimation window\n    :return: (pd.Series) Roll measure\n    '
    pass

def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int=20) -> pd.Series:
    if False:
        i = 10
        return i + 15
    '\n    Get Roll Impact.\n\n    Derivate from Roll Measure which takes into account dollar volume traded.\n\n    :param close_prices: (pd.Series) Close prices\n    :param dollar_volume: (pd.Series) Dollar volume series\n    :param window: (int) Estimation window\n    :return: (pd.Series) Roll impact\n    '
    pass

def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    if False:
        while True:
            i = 10
    '\n    Advances in Financial Machine Learning, Snippet 19.1, page 285.\n\n    Get beta estimate from Corwin-Schultz algorithm\n\n    :param high: (pd.Series) High prices\n    :param low: (pd.Series) Low prices\n    :param window: (int) Estimation window\n    :return: (pd.Series) Beta estimates\n    '
    pass

def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    if False:
        return 10
    '\n    Advances in Financial Machine Learning, Snippet 19.1, page 285.\n\n    Get gamma estimate from Corwin-Schultz algorithm.\n\n    :param high: (pd.Series) High prices\n    :param low: (pd.Series) Low prices\n    :return: (pd.Series) Gamma estimates\n    '
    pass

def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    if False:
        while True:
            i = 10
    '\n    Advances in Financial Machine Learning, Snippet 19.1, page 285.\n\n    Get alpha from Corwin-Schultz algorithm.\n\n    :param beta: (pd.Series) Beta estimates\n    :param gamma: (pd.Series) Gamma estimates\n    :return: (pd.Series) Alphas\n    '
    pass

def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int=20) -> pd.Series:
    if False:
        print('Hello World!')
    '\n    Advances in Financial Machine Learning, Snippet 19.1, page 285.\n\n    Get Corwin-Schultz spread estimator using high-low prices\n\n    :param high: (pd.Series) High prices\n    :param low: (pd.Series) Low prices\n    :param window: (int) Estimation window\n    :return: (pd.Series) Corwin-Schultz spread estimators\n    '
    pass

def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int=20) -> pd.Series:
    if False:
        while True:
            i = 10
    '\n    Advances in Financial Machine Learning, Snippet 19.2, page 286.\n\n    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.\n\n    :param high: (pd.Series) High prices\n    :param low: (pd.Series) Low prices\n    :param window: (int) Estimation window\n    :return: (pd.Series) Bekker-Parkinson volatility estimates\n    '
    pass