"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures: Imbalance Bars

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of tick, volume, and dollar imbalance bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 29) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time
interval sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high
frequency paradigm, Lopez de Prado, et al. These ideas are then extended in another paper: Flow toxicity and liquidity
in a high-frequency world.

We have introduced two types of imbalance bars: with expected number of tick defined through EMA (book implementation) and
constant number of ticks.

A good blog post to read, which helped us a lot in the implementation here is writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""
from typing import Union, Iterable, List, Optional
import numpy as np
import pandas as pd
from mlfinlab.data_structures.base_bars import BaseImbalanceBars
from mlfinlab.util.fast_ewma import ewma

class EMAImbalanceBars(BaseImbalanceBars):
    """
    Contains all of the logic to construct the imbalance bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_ema_dollar_imbalance_bars which will create an instance of this
    class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, num_prev_bars: int, expected_imbalance_window: int, exp_num_ticks_init: int, exp_num_ticks_constraints: List, batch_size: int, analyse_thresholds: bool):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor\n\n        :param metric: (str) Type of imbalance bar to create. Example: "dollar_imbalance"\n        :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)\n        :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n        :param exp_num_ticks_init: (int) Initial number of expected ticks\n        :param exp_num_ticks_constraints (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence\n        :param batch_size: (int) Number of rows to read in from the csv, per batch\n        :param analyse_thresholds: (bool) flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a\n                                          form of Pandas DataFrame\n        '
        pass

    def _get_exp_num_ticks(self):
        if False:
            while True:
                i = 10
        pass

class ConstImbalanceBars(BaseImbalanceBars):
    """
    Contains all of the logic to construct the imbalance bars with fixed expected number of ticks. This class shouldn't
    be used directly. We have added functions to the package such as get_const_dollar_imbalance_bars which will create
    an instance of this class and then construct the imbalance bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, expected_imbalance_window: int, exp_num_ticks_init: int, batch_size: int, analyse_thresholds: bool):
        if False:
            while True:
                i = 10
        '\n        Constructor\n\n        :param metric: (str) Type of imbalance bar to create. Example: "dollar_imbalance"\n        :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n        :param exp_num_ticks_init: (int) Initial number of expected ticks\n        :param batch_size: (int) Number of rows to read in from the csv, per batch\n        :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n        '
        pass

    def _get_exp_num_ticks(self):
        if False:
            return 10
        pass

def get_ema_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int=3, expected_imbalance_window: int=10000, exp_num_ticks_init: int=20000, exp_num_ticks_constraints: List[float]=None, batch_size: int=20000000.0, analyse_thresholds: bool=False, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        print('Hello World!')
    '\n    Creates the EMA dollar imbalance bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data\n                            in the format[date_time, price, volume]\n    :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)\n    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar\n    :param exp_num_ticks_constraints: (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (bool) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) DataFrame of dollar imbalance bars and DataFrame of thresholds, if to_csv=True returns None\n    '
    pass

def get_ema_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int=3, expected_imbalance_window: int=10000, exp_num_ticks_init: int=20000, exp_num_ticks_constraints: List[float]=None, batch_size: int=20000000.0, analyse_thresholds: bool=False, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        return 10
    '\n    Creates the EMA volume imbalance bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data\n                            in the format[date_time, price, volume]\n    :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)\n    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar\n    :param exp_num_ticks_constraints: (list) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (bool) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) DataFrame of volume imbalance bars and DataFrame of thresholds, if to_csv=True returns None\n    '
    pass

def get_ema_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], num_prev_bars: int=3, expected_imbalance_window: int=10000, exp_num_ticks_init: int=20000, exp_num_ticks_constraints: List[float]=None, batch_size: int=20000000.0, analyse_thresholds: bool=False, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates the EMA tick imbalance bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data\n                             in the format[date_time, price, volume]\n    :param num_prev_bars: (int) Window size for E[T]s (number of previous bars to use for expected number of ticks estimation)\n    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar\n    :param exp_num_ticks_constraints: (array) Minimum and maximum possible number of expected ticks. Used to control bars sampling convergence\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (bool) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) DataFrame of tick imbalance bars and DataFrame of thresholds, if to_csv=True returns None\n    '
    pass

def get_const_dollar_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], expected_imbalance_window: int=10000, exp_num_ticks_init: int=20000, batch_size: int=20000000.0, analyse_thresholds: bool=False, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        while True:
            i = 10
    '\n    Creates the Const dollar imbalance bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data\n                            in the format[date_time, price, volume]\n    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (bool) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) DataFrame of dollar imbalance bars and DataFrame of thresholds, if to_csv=True returns None\n    '
    pass

def get_const_volume_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], expected_imbalance_window: int=10000, exp_num_ticks_init: int=20000, batch_size: int=20000000.0, analyse_thresholds: bool=False, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        print('Hello World!')
    '\n    Creates the Const volume imbalance bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data\n                            in the format[date_time, price, volume]\n    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (bool) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) DataFrame of volume imbalance bars and DataFrame of thresholds, if to_csv=True returns None\n    '
    pass

def get_const_tick_imbalance_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], expected_imbalance_window: int=10000, exp_num_ticks_init: int=20000, batch_size: int=20000000.0, analyse_thresholds: bool=False, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    '\n    Creates the Const tick imbalance bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str or pd.DataFrame) Path to the csv file or Pandas Data Frame containing raw tick data in the format[date_time, price, volume]\n    :param expected_imbalance_window: (int) EMA window used to estimate expected imbalance\n    :param exp_num_ticks_init: (int) Initial expected number of ticks per bar\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (bool) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param analyse_thresholds: (bool) Flag to save  and return thresholds used to sample imbalance bars\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) DataFrame of tick imbalance bars and DataFrame of thresholds, if to_csv=True returns None\n    '
    pass