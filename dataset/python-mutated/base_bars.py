"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional
import numpy as np
import pandas as pd
from mlfinlab.util.fast_ewma import ewma

def _crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> list:
    if False:
        while True:
            i = 10
    '\n    Splits df into chunks of chunksize\n\n    :param df: (pd.DataFrame) Dataframe to split\n    :param chunksize: (int) Number of rows in chunk\n    :return: (list) Chunks (pd.DataFrames)\n    '
    pass

class BaseBars(ABC):
    """
    Abstract base class which contains the structure which is shared between the various standard and information
    driven bars. There are some methods contained in here that would only be applicable to information bars but
    they are included here so as to avoid a complicated nested class structure.
    """

    def __init__(self, metric: str, batch_size: int=20000000.0):
        if False:
            print('Hello World!')
        '\n        Constructor\n\n        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.\n        :param batch_size: (int) Number of rows to read in from the csv, per batch.\n        '
        pass

    def batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None) -> Union[pd.DataFrame, None]:
        if False:
            i = 10
            return i + 15
        '\n        Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in the form of a DataFrame.\n        The csv file or DataFrame must have only 3 columns: date_time, price, & volume.\n\n        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing\n                                raw tick data  in the format[date_time, price, volume]\n        :param verbose: (bool) Flag whether to print message on each processed batch or not\n        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame\n        :param output_path: (bool) Path to results file, if to_csv = True\n\n        :return: (pd.DataFrame or None) Financial data structure\n        '
        pass

    def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
        if False:
            i = 10
            return i + 15
        '\n        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame\n                                containing raw tick data in the format[date_time, price, volume]\n        '
        pass

    def _read_first_row(self, file_path: str):
        if False:
            while True:
                i = 10
        '\n        :param file_path: (str) Path to the csv file containing raw tick data in the format[date_time, price, volume]\n        '
        pass

    def run(self, data: Union[list, tuple, pd.DataFrame]) -> list:
        if False:
            i = 10
            return i + 15
        '\n        Reads a List, Tuple, or Dataframe and then constructs the financial data structure in the form of a list.\n        The List, Tuple, or DataFrame must have only 3 attrs: date_time, price, & volume.\n\n        :param data: (list, tuple, or pd.DataFrame) Dict or ndarray containing raw tick data in the format[date_time, price, volume]\n\n        :return: (list) Financial data structure\n        '
        pass

    @abstractmethod
    def _extract_bars(self, data: pd.DataFrame) -> list:
        if False:
            i = 10
            return i + 15
        '\n        This method is required by all the bar types and is used to create the desired bars.\n\n        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.\n        :return: (list) Bars built using the current batch.\n        '

    @abstractmethod
    def _reset_cache(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method is required by all the bar types. It describes how cache should be reset\n        when new bar is sampled.\n        '

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        if False:
            while True:
                i = 10
        '\n        Tests that the csv file read has the format: date_time, price, and volume.\n        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.\n\n        :param test_batch: (pd.DataFrame) The first row of the dataset.\n        '
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'
        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            raise ValueError('csv file, column 0, not a date time format:', test_batch.iloc[0, 0])

    def _update_high_low(self, price: float) -> Union[float, float]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the high and low prices using the current price.\n\n        :param price: (float) Current price\n        :return: (tuple) Updated high and low prices\n        '
        pass

    def _create_bars(self, date_time: str, price: float, high_price: float, low_price: float, list_bars: list) -> None:
        if False:
            return 10
        '\n        Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close, volume,\n        cum_buy_volume, cum_ticks, cum_dollar_value.\n        These bars are appended to list_bars, which is later used to construct the final bars DataFrame.\n\n        :param date_time: (str) Timestamp of the bar\n        :param price: (float) The current price\n        :param high_price: (float) Highest price in the period\n        :param low_price: (float) Lowest price in the period\n        :param list_bars: (list) List to which we append the bars\n        '
        pass

    def _apply_tick_rule(self, price: float) -> int:
        if False:
            print('Hello World!')
        '\n        Applies the tick rule as defined on page 29 of Advances in Financial Machine Learning.\n\n        :param price: (float) Price at time t\n        :return: (int) The signed tick\n        '
        pass

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        if False:
            print('Hello World!')
        '\n        Advances in Financial Machine Learning, page 29.\n\n        Get the imbalance at a point in time, denoted as Theta_t\n\n        :param price: (float) Price at t\n        :param signed_tick: (int) signed tick, using the tick rule\n        :param volume: (float) Volume traded at t\n        :return: (float) Imbalance at time t\n        '
        pass

class BaseImbalanceBars(BaseBars):
    """
    Base class for Imbalance Bars (EMA and Const) which implements imbalance bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int, expected_imbalance_window: int, exp_num_ticks_init: int, analyse_thresholds: bool):
        if False:
            i = 10
            return i + 15
        '\n        Constructor\n\n        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.\n        :param batch_size: (int) Number of rows to read in from the csv, per batch.\n        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades\n        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.\n                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init\n        :param analyse_thresholds: (bool) Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a\n                                          form of Pandas DataFrame\n        '
        pass

    def _reset_cache(self):
        if False:
            i = 10
            return i + 15
        '\n        Implementation of abstract method _reset_cache for imbalance bars\n        '
        pass

    def _extract_bars(self, data: Tuple[dict, pd.DataFrame]) -> list:
        if False:
            i = 10
            return i + 15
        '\n        For loop which compiles the various imbalance bars: dollar, volume, or tick.\n\n        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.\n        :return: (list) Bars built using the current batch.\n        '
        pass

    def _get_expected_imbalance(self, window: int):
        if False:
            print('Hello World!')
        '\n        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29\n        :param window: (int) EWMA window for calculation\n        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA\n        '
        pass

    @abstractmethod
    def _get_exp_num_ticks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract method which updates expected number of ticks when new run bar is formed\n        '

class BaseRunBars(BaseBars):
    """
    Base class for Run Bars (EMA and Const) which implements run bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int, num_prev_bars: int, expected_imbalance_window: int, exp_num_ticks_init: int, analyse_thresholds: bool):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor\n\n        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.\n        :param batch_size: (int) Number of rows to read in from the csv, per batch.\n        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades\n        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.\n                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init\n        :param analyse_thresholds: (bool) Flag to return thresholds values (thetas, exp_num_ticks, exp_runs) in Pandas DataFrame\n        '
        pass

    def _reset_cache(self):
        if False:
            while True:
                i = 10
        '\n        Implementation of abstract method _reset_cache for imbalance bars\n        '
        pass

    def _extract_bars(self, data: Tuple[list, np.ndarray]) -> list:
        if False:
            for i in range(10):
                print('nop')
        '\n        For loop which compiles the various run bars: dollar, volume, or tick.\n\n        :param data: (list or np.ndarray) Contains 3 columns - date_time, price, and volume.\n        :return: (list) of bars built using the current batch.\n        '
        pass

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool=False):
        if False:
            return 10
        '\n        Advances in Financial Machine Learning, page 29.\n\n        Calculates the expected imbalance: 2P[b_t=1]-1, using a EWMA.\n\n        :param array: (list) of imbalances\n        :param window: (int) EWMA window for calculation\n        :parawm warm_up: (bool) flag of whether warm up period passed\n        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA\n        '
        pass

    @abstractmethod
    def _get_exp_num_ticks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract method which updates expected number of ticks when new imbalance bar is formed\n        '