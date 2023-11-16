import os
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.cla import CLA
from pypfopt.efficient_frontier import EfficientCDaR, EfficientCVaR, EfficientFrontier, EfficientSemivariance

def resource(name):
    if False:
        i = 10
        return i + 15
    return os.path.join(os.path.dirname(__file__), 'resources', name)

def get_data():
    if False:
        i = 10
        return i + 15
    return pd.read_csv(resource('stock_prices.csv'), parse_dates=True, index_col='date')

def get_benchmark_data():
    if False:
        print('Hello World!')
    return pd.read_csv(resource('spy_prices.csv'), parse_dates=True, index_col='date')

def get_market_caps():
    if False:
        print('Hello World!')
    mcaps = {'GOOG': 927000000000.0, 'AAPL': 1190000000000.0, 'FB': 574000000000.0, 'BABA': 533000000000.0, 'AMZN': 867000000000.0, 'GE': 96000000000.0, 'AMD': 43000000000.0, 'WMT': 339000000000.0, 'BAC': 301000000000.0, 'GM': 51000000000.0, 'T': 61000000000.0, 'UAA': 78000000000.0, 'SHLD': 0, 'XOM': 295000000000.0, 'RRC': 1000000000.0, 'BBY': 22000000000.0, 'MA': 288000000000.0, 'PFE': 212000000000.0, 'JPM': 422000000000.0, 'SBUX': 102000000000.0}
    return mcaps

def setup_efficient_frontier(data_only=False, *args, **kwargs):
    if False:
        while True:
            i = 10
    df = get_data()
    mean_return = expected_returns.mean_historical_return(df)
    sample_cov_matrix = risk_models.sample_cov(df)
    if data_only:
        return (mean_return, sample_cov_matrix)
    return EfficientFrontier(mean_return, sample_cov_matrix, *args, verbose=True, **kwargs)

def setup_efficient_semivariance(data_only=False, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    df = get_data().dropna(axis=0, how='any')
    mean_return = expected_returns.mean_historical_return(df)
    historic_returns = expected_returns.returns_from_prices(df)
    if data_only:
        return (mean_return, historic_returns)
    return EfficientSemivariance(mean_return, historic_returns, *args, verbose=True, **kwargs)

def setup_efficient_cvar(data_only=False, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    df = get_data().dropna(axis=0, how='any')
    mean_return = expected_returns.mean_historical_return(df)
    historic_returns = expected_returns.returns_from_prices(df)
    if data_only:
        return (mean_return, historic_returns)
    return EfficientCVaR(mean_return, historic_returns, *args, verbose=True, **kwargs)

def setup_efficient_cdar(data_only=False, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    df = get_data().dropna(axis=0, how='any')
    mean_return = expected_returns.mean_historical_return(df)
    historic_returns = expected_returns.returns_from_prices(df)
    if data_only:
        return (mean_return, historic_returns)
    return EfficientCDaR(mean_return, historic_returns, *args, verbose=True, **kwargs)

def setup_cla(data_only=False, *args, **kwargs):
    if False:
        while True:
            i = 10
    df = get_data()
    mean_return = expected_returns.mean_historical_return(df)
    sample_cov_matrix = risk_models.sample_cov(df)
    if data_only:
        return (mean_return, sample_cov_matrix)
    return CLA(mean_return, sample_cov_matrix, *args, **kwargs)

def simple_ef_weights(expected_returns, cov_matrix, target_return, weights_sum):
    if False:
        while True:
            i = 10
    '\n    Calculate weights to achieve target_return on the efficient frontier.\n    The only constraint is the sum of the weights.\n    Note: This is just a simple test utility, it does not support the generalised\n    constraints that EfficientFrontier does and is used to check the results\n    of EfficientFrontier in simple cases.  In particular it is not capable of\n    preventing negative weights (shorting).\n    :param expected_returns: expected returns for each asset.\n    :type expected_returns: np.ndarray\n    :param cov_matrix: covariance of returns for each asset.\n    :type cov_matrix: np.ndarray\n    :param target_return: the target return for the portfolio to achieve.\n    :type target_return: float\n    :param weights_sum: the sum of the returned weights, optimization constraint.\n    :type weights_sum: float\n    :return: weight for each asset, which sum to 1.0\n    :rtype: np.ndarray\n    '
    r = expected_returns.reshape((-1, 1))
    m = np.block([[cov_matrix, r, np.ones(r.shape)], [r.transpose(), 0, 0], [np.ones(r.shape).transpose(), 0, 0]])
    y = np.block([[np.zeros(r.shape)], [target_return], [weights_sum]])
    x = np.linalg.inv(m) @ y
    w = x.flatten()[:-2]
    return w