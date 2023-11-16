"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

Time bars generation logic
"""
from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd
from mlfinlab.data_structures.base_bars import BaseBars

class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the time bars. This class shouldn't be used directly.
    Use get_time_bars instead
    """

    def __init__(self, resolution: str, num_units: int, batch_size: int=20000000):
        if False:
            print('Hello World!')
        "\n        Constructor\n\n        :param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S']\n        :param num_units: (int) Number of days, minutes, etc.\n        :param batch_size: (int) Number of rows to read in from the csv, per batch\n        "
        pass

    def _reset_cache(self):
        if False:
            i = 10
            return i + 15
        '\n        Implementation of abstract method _reset_cache for time bars\n        '
        pass

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        if False:
            print('Hello World!')
        '\n        For loop which compiles time bars.\n        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.\n\n        :param data: (tuple) Contains 3 columns - date_time, price, and volume.\n        :return: (list) Extracted bars\n        '
        pass

def get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str='D', num_units: int=1, batch_size: int=20000000, verbose: bool=True, to_csv: bool=False, output_path: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    "\n    Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.\n\n    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data\n                            in the format[date_time, price, volume]\n    :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')\n    :param num_units: (int) Number of resolution units (3 days for example, 2 hours)\n    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.\n    :param verbose: (int) Print out batch numbers (True or False)\n    :param to_csv: (bool) Save bars to csv after every batch run (True or False)\n    :param output_path: (str) Path to csv file, if to_csv is True\n    :return: (pd.DataFrame) Dataframe of time bars, if to_csv=True return None\n    "
    pass