"""
This module contains class for ETF trick generation and futures roll function, described in Marcos Lopez de Prado's
book 'Advances in Financial Machine Learning' ETF trick class can generate ETF trick series either from .csv files
or from in memory pandas DataFrames
"""
import warnings
import pandas as pd
import numpy as np

class ETFTrick:
    """
    Contains logic of vectorised ETF trick implementation. Can used for both memory data frames (pd.DataFrame) and
    csv files. All data frames, files should be processed in a specific format, described in examples
    """

    def __init__(self, open_df, close_df, alloc_df, costs_df, rates_df=None, index_col=0):
        if False:
            i = 10
            return i + 15
        '\n        Constructor\n\n        Creates class object, for csv based files reads the first data chunk.\n\n        :param open_df: (pd.DataFrame or string): open prices data frame or path to csv file,\n         corresponds to o(t) from the book\n        :param close_df: (pd.DataFrame or string): close prices data frame or path to csv file, corresponds to p(t)\n        :param alloc_df: (pd.DataFrame or string): asset allocations data frame or path to csv file (in # of contracts),\n         corresponds to w(t)\n        :param costs_df: (pd.DataFrame or string): rebalance, carry and dividend costs of holding/rebalancing the\n         position, corresponds to d(t)\n        :param rates_df: (pd.DataFrame or string): dollar value of one point move of contract includes exchange rate,\n         futures contracts multiplies). Corresponds to phi(t)\n         For example, 1$ in VIX index, equals 1000$ in VIX futures contract value.\n         If None then trivial (all values equal 1.0) is generated\n        :param index_col: (int): positional index of index column. Used for to determine index column in csv files\n        '
        pass

    def _append_previous_rows(self, cache):
        if False:
            while True:
                i = 10
        '\n        Uses latest two rows from cache to append into current data. Used for csv based ETF trick, when the next\n        batch is loaded and we need to recalculate K value which corresponds to previous batch.\n\n        :param cache: (dict): dictionary which pd.DataFrames with latest 2 rows of open, close, alloc, costs, rates\n        :return: (pd.DataFrame): data frame with close price differences (updates self.data_dict)\n        '
        pass

    def generate_trick_components(self, cache=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Calculates all etf trick operations which can be vectorised. Outputs multilevel pandas data frame.\n\n        Generated components:\n        'w': alloc_df\n        'h_t': h_t/K value from ETF trick algorithm from the book. Which K to use is based on previous values and\n            cannot be vectorised.\n        'close_open': close_df - open_df\n        'price_diff': close price differences\n        'costs': costs_df\n        'rate': rates_df\n\n        :param cache: (dict of pd.DataFrames): dictionary which contains latest 2 rows of open, close, rates, alloc,\n            costs, rates data\n        :return: (pd.DataFrame): pandas data frame with columns in a format: component_1/asset_name_1,\n            component_1/asset_name_2, ..., component_6/asset_name_n\n        "
        pass

    def _update_cache(self):
        if False:
            while True:
                i = 10
        '\n        Updates cache (two previous rows) when new data batch is read into the memory. Cache is used to\n        recalculate ETF trick value which corresponds to previous batch last row. That is why we need 2 previous rows\n        for close price difference calculation\n\n        :return: (dict): dictionary with open, close, alloc, costs and rates last 2 rows\n        '
        pass

    def _chunk_loop(self, data_df):
        if False:
            for i in range(10):
                print('nop')
        '\n        Single ETF trick iteration for currently stored(with needed components) data set in memory (data_df).\n        For in-memory data set would yield complete ETF trick series, for csv based\n        would generate ETF trick series for current batch.\n\n        :param data_df: The data set on which to apply the ETF trick.\n        :return: (pd.Series): pandas Series with ETF trick values\n        '
        pass

    def _index_check(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal check for all price, rates and allocations data frames have the same index\n        '
        pass

    def _get_batch_from_csv(self, batch_size):
        if False:
            i = 10
            return i + 15
        '\n        Reads the next batch of data sets from csv files and puts them in class variable data_dict\n\n        :param batch_size: number of rows to read\n        '
        pass

    def _rewind_etf_trick(self, alloc_df, etf_series):
        if False:
            return 10
        '\n        ETF trick uses next open price information, when we process csv file in batches the last row in batch will have\n        next open price value as nan, that is why when new batch comes, we need to rewind ETF trick values one step\n        back, recalculate ETF trick value for the last row from previous batch using open price from latest batch\n        received. This function rewinds values needed for ETF trick calculation recalculate\n\n        :param alloc_df: (pd.DataFrame): data frame with allocations vectors\n        :param etf_series (pd.Series): current computed ETF trick series\n        '
        pass

    def _csv_file_etf_series(self, batch_size):
        if False:
            i = 10
            return i + 15
        '\n        Csv based ETF trick series generation\n\n        :param: batch_size: (int): Size of the batch that you would like to make use of\n        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0\n        '
        pass

    def _in_memory_etf_series(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        In-memory based ETF trick series generation.\n\n        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0\n        '
        pass

    def get_etf_series(self, batch_size=100000.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        External method which defines which etf trick method to use.\n\n        :param: batch_size: Size of the batch that you would like to make use of\n        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0\n        '
        pass

    def reset(self):
        if False:
            print('Hello World!')
        '\n        Re-inits class object. This methods can be used to reset file iterators for multiple get_etf_trick() calls.\n        '
        pass

def get_futures_roll_series(data_df, open_col, close_col, sec_col, current_sec_col, roll_backward=False, method='absolute'):
    if False:
        print('Hello World!')
    "\n    Function for generating rolling futures series from data frame of multiple futures.\n\n    :param data_df: (pd.DataFrame): pandas DataFrame containing price info, security name and current active futures\n     column\n    :param open_col: (string): open prices column name\n    :param close_col: (string): close prices column name\n    :param sec_col: (string): security name column name\n    :param current_sec_col: (string): current active security column name. When value in this column changes it means\n     rolling\n    :param roll_backward: (boolean): True for subtracting final gap value from all values\n    :param method: (string): what returns user wants to preserve, 'absolute' or 'relative'\n    :return (pd.Series): futures roll close price series\n    "
    pass