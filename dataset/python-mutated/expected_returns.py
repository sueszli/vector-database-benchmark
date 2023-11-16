"""
The ``expected_returns`` module provides functions for estimating the expected returns of
the assets, which is a required input in mean-variance optimization.

By convention, the output of these methods is expected *annual* returns. It is assumed that
*daily* prices are provided, though in reality the functions are agnostic
to the time period (just change the ``frequency`` parameter). Asset prices must be given as
a pandas dataframe, as per the format described in the :ref:`user-guide`.

All of the functions process the price data into percentage returns data, before
calculating their respective estimates of expected returns.

Currently implemented:

    - general return model function, allowing you to run any return model from one function.
    - mean historical return
    - exponentially weighted mean historical return
    - CAPM estimate of returns

Additionally, we provide utility functions to convert from returns to prices and vice-versa.
"""
import warnings
import numpy as np
import pandas as pd

def _check_returns(returns):
    if False:
        i = 10
        return i + 15
    if np.any(np.isnan(returns.mask(returns.ffill().isnull(), 0))):
        warnings.warn('Some returns are NaN. Please check your price data.', UserWarning)
    if np.any(np.isinf(returns)):
        warnings.warn('Some returns are infinite. Please check your price data.', UserWarning)

def returns_from_prices(prices, log_returns=False):
    if False:
        while True:
            i = 10
    '\n    Calculate the returns given prices.\n\n    :param prices: adjusted (daily) closing prices of the asset, each row is a\n                   date and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: (daily) returns\n    :rtype: pd.DataFrame\n    '
    if log_returns:
        returns = np.log(1 + prices.pct_change()).dropna(how='all')
    else:
        returns = prices.pct_change().dropna(how='all')
    return returns

def prices_from_returns(returns, log_returns=False):
    if False:
        i = 10
        return i + 15
    '\n    Calculate the pseudo-prices given returns. These are not true prices because\n    the initial prices are all set to 1, but it behaves as intended when passed\n    to any PyPortfolioOpt method.\n\n    :param returns: (daily) percentage returns of the assets\n    :type returns: pd.DataFrame\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: (daily) pseudo-prices.\n    :rtype: pd.DataFrame\n    '
    if log_returns:
        ret = np.exp(returns)
    else:
        ret = 1 + returns
    ret.iloc[0] = 1
    return ret.cumprod()

def return_model(prices, method='mean_historical_return', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Compute an estimate of future returns, using the return model specified in ``method``.\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n    :type returns_data: bool, defaults to False.\n    :param method: the return model to use. Should be one of:\n\n        - ``mean_historical_return``\n        - ``ema_historical_return``\n        - ``capm_return``\n\n    :type method: str, optional\n    :raises NotImplementedError: if the supplied method is not recognised\n    :return: annualised sample covariance matrix\n    :rtype: pd.DataFrame\n    '
    if method == 'mean_historical_return':
        return mean_historical_return(prices, **kwargs)
    elif method == 'ema_historical_return':
        return ema_historical_return(prices, **kwargs)
    elif method == 'capm_return':
        return capm_return(prices, **kwargs)
    else:
        raise NotImplementedError('Return model {} not implemented'.format(method))

def mean_historical_return(prices, returns_data=False, compounding=True, frequency=252, log_returns=False):
    if False:
        print('Hello World!')
    '\n    Calculate annualised mean (daily) historical return from input (daily) asset prices.\n    Use ``compounding`` to toggle between the default geometric mean (CAGR) and the\n    arithmetic mean.\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n                         These **should not** be log returns.\n    :type returns_data: bool, defaults to False.\n    :param compounding: computes geometric mean returns if True,\n                        arithmetic otherwise, optional.\n    :type compounding: bool, defaults to True\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                      of trading days in a year)\n    :type frequency: int, optional\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: annualised mean (daily) return for each asset\n    :rtype: pd.Series\n    '
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('prices are not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    _check_returns(returns)
    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency

def ema_historical_return(prices, returns_data=False, compounding=True, span=500, frequency=252, log_returns=False):
    if False:
        i = 10
        return i + 15
    '\n    Calculate the exponentially-weighted mean of (daily) historical returns, giving\n    higher weight to more recent data.\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n                         These **should not** be log returns.\n    :type returns_data: bool, defaults to False.\n    :param compounding: computes geometric mean returns if True,\n                        arithmetic otherwise, optional.\n    :type compounding: bool, defaults to True\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                      of trading days in a year)\n    :type frequency: int, optional\n    :param span: the time-span for the EMA, defaults to 500-day EMA.\n    :type span: int, optional\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: annualised exponentially-weighted mean (daily) return of each asset\n    :rtype: pd.Series\n    '
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('prices are not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    _check_returns(returns)
    if compounding:
        return (1 + returns.ewm(span=span).mean().iloc[-1]) ** frequency - 1
    else:
        return returns.ewm(span=span).mean().iloc[-1] * frequency

def capm_return(prices, market_prices=None, returns_data=False, risk_free_rate=0.02, compounding=True, frequency=252, log_returns=False):
    if False:
        while True:
            i = 10
    '\n    Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM,\n    asset returns are equal to market returns plus a :math:`\x08eta` term encoding\n    the relative risk of the asset.\n\n    .. math::\n\n        R_i = R_f + \\beta_i (E(R_m) - R_f)\n\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                    and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param market_prices: adjusted closing prices of the benchmark, defaults to None\n    :type market_prices: pd.DataFrame, optional\n    :param returns_data: if true, the first arguments are returns instead of prices.\n    :type returns_data: bool, defaults to False.\n    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                           You should use the appropriate time period, corresponding\n                           to the frequency parameter.\n    :type risk_free_rate: float, optional\n    :param compounding: computes geometric mean returns if True,\n                        arithmetic otherwise, optional.\n    :type compounding: bool, defaults to True\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                        of trading days in a year)\n    :type frequency: int, optional\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: annualised return estimate\n    :rtype: pd.Series\n    '
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('prices are not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    market_returns = None
    if returns_data:
        returns = prices.copy()
        if market_prices is not None:
            market_returns = market_prices
    else:
        returns = returns_from_prices(prices, log_returns)
        if market_prices is not None:
            if not isinstance(market_prices, pd.DataFrame):
                warnings.warn('market prices are not in a dataframe', RuntimeWarning)
                market_prices = pd.DataFrame(market_prices)
            market_returns = returns_from_prices(market_prices, log_returns)
    if market_returns is None:
        returns['mkt'] = returns.mean(axis=1)
    else:
        market_returns.columns = ['mkt']
        returns = returns.join(market_returns, how='left')
    _check_returns(returns)
    cov = returns.cov()
    betas = cov['mkt'] / cov.loc['mkt', 'mkt']
    betas = betas.drop('mkt')
    if compounding:
        mkt_mean_ret = (1 + returns['mkt']).prod() ** (frequency / returns['mkt'].count()) - 1
    else:
        mkt_mean_ret = returns['mkt'].mean() * frequency
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)