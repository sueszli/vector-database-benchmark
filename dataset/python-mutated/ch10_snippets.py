"""
This module contains the code snippets found in Chapter 10 of "Advances in Financial Machine Learning" by
Marcos López de Prado. The code has been amended for readability, to conform to PEP8 rules, to keep the snippets as
manageable single-units of functionality, as well as to account for deprecation of functions originally used, but is
otherwise unaltered.
"""
import warnings
import pandas as pd
import numpy as np
from scipy.stats import norm
from mlfinlab.util.multiprocess import mp_pandas_obj

def get_signal(prob, num_classes, pred=None):
    if False:
        while True:
            i = 10
    '\n    SNIPPET 10.1 - FROM PROBABILITIES TO BET SIZE\n    Calculates the given size of the bet given the side and the probability (i.e. confidence) of the prediction. In this\n    representation, the probability will always be between 1/num_classes and 1.0.\n\n    :param prob: (pd.Series) The probability of the predicted bet side.\n    :param num_classes: (int) The number of predicted bet sides.\n    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size\n     (i.e. without multiplying by the side).\n    :return: (pd.Series) The bet size.\n    '
    pass

def avg_active_signals(signals, num_threads=1):
    if False:
        i = 10
        return i + 15
    "\n    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE\n    Function averages the bet sizes of all concurrently active bets. This function makes use of multiprocessing.\n\n    :param signals: (pandas.DataFrame) Contains at least the following columns:\n     'signal' - the bet size\n     't1' - the closing time of the bet\n     And the index must be datetime format.\n    :param num_threads: (int) Number of threads to use in multiprocessing, default value is 1.\n    :return: (pandas.Series) The averaged bet sizes.\n    "
    pass

def mp_avg_active_signals(signals, molecule):
    if False:
        return 10
    "\n    Part of SNIPPET 10.2\n    A function to be passed to the 'mp_pandas_obj' function to allow the bet sizes to be averaged using multiprocessing.\n\n    At time loc, average signal among those still active.\n    Signal is active if (a) it is issued before or at loc, and (b) loc is before the signal's end time,\n    or end time is still unknown (NaT).\n\n    :param signals: (pandas.DataFrame) Contains at least the following columns: 'signal' (the bet size) and 't1' (the closing time of the bet).\n    :param molecule: (list) Indivisible tasks to be passed to 'mp_pandas_obj', in this case a list of datetimes.\n    :return: (pandas.Series) The averaged bet size sub-series.\n    "
    pass

def discrete_signal(signal0, step_size):
    if False:
        print('Hello World!')
    '\n    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING\n    Discretizes the bet size signal based on the step size given.\n\n    :param signal0: (pandas.Series) The signal to discretize.\n    :param step_size: (float) Step size.\n    :return: (pandas.Series) The discretized signal.\n    '
    pass

def bet_size_sigmoid(w_param, price_div):
    if False:
        print('Hello World!')
    '\n    Part of SNIPPET 10.4\n    Calculates the bet size from the price divergence and a regulating coefficient.\n    Based on a sigmoid function for a bet size algorithm.\n\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param price_div: (float) Price divergence, forecast price - market price.\n    :return: (float) The bet size.\n    '
    pass

def get_target_pos_sigmoid(w_param, forecast_price, market_price, max_pos):
    if False:
        for i in range(10):
            print('nop')
    '\n    Part of SNIPPET 10.4\n    Calculates the target position given the forecast price, market price, maximum position size, and a regulating\n    coefficient. Based on a sigmoid function for a bet size algorithm.\n\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param forecast_price: (float) Forecast price.\n    :param market_price: (float) Market price.\n    :param max_pos: (int) Maximum absolute position size.\n    :return: (int) Target position.\n    '
    pass

def inv_price_sigmoid(forecast_price, w_param, m_bet_size):
    if False:
        while True:
            i = 10
    '\n    Part of SNIPPET 10.4\n    Calculates the inverse of the bet size with respect to the market price.\n    Based on a sigmoid function for a bet size algorithm.\n\n    :param forecast_price: (float) Forecast price.\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param m_bet_size: (float) Bet size.\n    :return: (float) Inverse of bet size with respect to market price.\n    '
    pass

def limit_price_sigmoid(target_pos, pos, forecast_price, w_param, max_pos):
    if False:
        for i in range(10):
            print('nop')
    '\n    Part of SNIPPET 10.4\n    Calculates the limit price.\n    Based on a sigmoid function for a bet size algorithm.\n\n    :param target_pos: (int) Target position.\n    :param pos: (int) Current position.\n    :param forecast_price: (float) Forecast price.\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param max_pos: (int) Maximum absolute position size.\n    :return: (float) Limit price.\n    '
    pass

def get_w_sigmoid(price_div, m_bet_size):
    if False:
        for i in range(10):
            print('nop')
    "\n    Part of SNIPPET 10.4\n    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.\n    Based on a sigmoid function for a bet size algorithm.\n\n    :param price_div: (float) Price divergence, forecast price - market price.\n    :param m_bet_size: (float) Bet size.\n    :return: (float) Inverse of bet size with respect to the\n        regulating coefficient.\n    "
    pass

def bet_size_power(w_param, price_div):
    if False:
        print('Hello World!')
    '\n    Derived from SNIPPET 10.4\n    Calculates the bet size from the price divergence and a regulating coefficient.\n    Based on a power function for a bet size algorithm.\n\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param price_div: (float) Price divergence, f - market_price, must be between -1 and 1, inclusive.\n    :return: (float) The bet size.\n    '
    pass

def get_target_pos_power(w_param, forecast_price, market_price, max_pos):
    if False:
        for i in range(10):
            print('nop')
    '\n    Derived from SNIPPET 10.4\n    Calculates the target position given the forecast price, market price, maximum position size, and a regulating\n    coefficient. Based on a power function for a bet size algorithm.\n\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param forecast_price: (float) Forecast price.\n    :param market_price: (float) Market price.\n    :param max_pos: (float) Maximum absolute position size.\n    :return: (float) Target position.\n    '
    pass

def inv_price_power(forecast_price, w_param, m_bet_size):
    if False:
        return 10
    '\n    Derived from SNIPPET 10.4\n    Calculates the inverse of the bet size with respect to the market price.\n    Based on a power function for a bet size algorithm.\n\n    :param forecast_price: (float) Forecast price.\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param m_bet_size: (float) Bet size.\n    :return: (float) Inverse of bet size with respect to market price.\n    '
    pass

def limit_price_power(target_pos, pos, forecast_price, w_param, max_pos):
    if False:
        while True:
            i = 10
    '\n    Derived from SNIPPET 10.4\n    Calculates the limit price. Based on a power function for a bet size algorithm.\n\n    :param target_pos: (float) Target position.\n    :param pos: (float) Current position.\n    :param forecast_price: (float) Forecast price.\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param max_pos: (float) Maximum absolute position size.\n    :return: (float) Limit price.\n    '
    pass

def get_w_power(price_div, m_bet_size):
    if False:
        while True:
            i = 10
    "\n    Derived from SNIPPET 10.4\n    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.\n    The 'w' coefficient must be greater than or equal to zero.\n    Based on a power function for a bet size algorithm.\n\n    :param price_div: (float) Price divergence, forecast price - market price.\n    :param m_bet_size: (float) Bet size.\n    :return: (float) Inverse of bet size with respect to the regulating coefficient.\n    "
    pass

def bet_size(w_param, price_div, func):
    if False:
        i = 10
        return i + 15
    "\n    Derived from SNIPPET 10.4\n    Calculates the bet size from the price divergence and a regulating coefficient.\n    The 'func' argument allows the user to choose between bet sizing functions.\n\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param price_div: (float) Price divergence, f - market_price\n    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.\n    :return: (float) The bet size.\n    "
    pass

def get_target_pos(w_param, forecast_price, market_price, max_pos, func):
    if False:
        print('Hello World!')
    "\n    Derived from SNIPPET 10.4\n    Calculates the target position given the forecast price, market price, maximum position size, and a regulating\n    coefficient. The 'func' argument allows the user to choose between bet sizing functions.\n\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param forecast_price: (float) Forecast price.\n    :param market_price: (float) Market price.\n    :param max_pos: (int) Maximum absolute position size.\n    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.\n    :return: (int) Target position.\n    "
    pass

def inv_price(forecast_price, w_param, m_bet_size, func):
    if False:
        return 10
    "\n    Derived from SNIPPET 10.4\n    Calculates the inverse of the bet size with respect to the market price.\n    The 'func' argument allows the user to choose between bet sizing functions.\n\n    :param forecast_price: (float) Forecast price.\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param m_bet_size: (float) Bet size.\n    :return: (float) Inverse of bet size with respect to market price.\n    "
    pass

def limit_price(target_pos, pos, forecast_price, w_param, max_pos, func):
    if False:
        print('Hello World!')
    "\n    Derived from SNIPPET 10.4\n    Calculates the limit price. The 'func' argument allows the user to choose between bet sizing functions.\n\n    :param target_pos: (int) Target position.\n    :param pos: (int) Current position.\n    :param forecast_price: (float) Forecast price.\n    :param w_param: (float) Coefficient regulating the width of the bet size function.\n    :param max_pos: (int) Maximum absolute position size.\n    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.\n    :return: (float) Limit price.\n    "
    pass

def get_w(price_div, m_bet_size, func):
    if False:
        i = 10
        return i + 15
    "\n    Derived from SNIPPET 10.4\n    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.\n    The 'func' argument allows the user to choose between bet sizing functions.\n\n    :param price_div: (float) Price divergence, forecast price - market price.\n    :param m_bet_size: (float) Bet size.\n    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.\n    :return: (float) Inverse of bet size with respect to the regulating coefficient.\n    "
    pass