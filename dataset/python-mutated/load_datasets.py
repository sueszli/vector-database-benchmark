"""
The module implementing various functions loading tick, dollar, stock data sets which can be used as
sandbox data.
"""
import os
import numpy as np
import pandas as pd
from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.util.volatility import get_daily_vol
from mlfinlab.filters.filters import cusum_filter

def load_stock_prices() -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    '\n    Loads stock prices data sets consisting of\n    EEM, EWG, TIP, EWJ, EFA, IEF, EWQ, EWU, XLB, XLE, XLF, LQD, XLK, XLU, EPP, FXI, VGK, VPL, SPY, TLT, BND, CSJ,\n    DIA starting from 2008 till 2016.\n\n    :return: (pd.DataFrame) The stock_prices data frame.\n    '
    pass

def load_tick_sample() -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    '\n    Loads E-Mini S&P 500 futures tick data sample.\n\n    :return: (pd.DataFrame) Frame with tick data sample.\n    '
    pass

def load_dollar_bar_sample() -> pd.DataFrame:
    if False:
        while True:
            i = 10
    '\n    Loads E-Mini S&P 500 futures dollar bars data sample.\n\n    :return: (pd.DataFrame) Frame with dollar bar data sample.\n    '
    pass

def generate_multi_asset_data_set(start_date: pd.Timestamp=pd.Timestamp(2008, 1, 1), end_date: pd.Timestamp=pd.Timestamp(2020, 1, 1)) -> tuple:
    if False:
        print('Hello World!')
    '\n    Generates multi-asset dataset from stock prices labelled by triple-barrier method.\n\n    :param start_date: (pd.Timestamp) Dataset start date.\n    :param end_date: (pd.Timestamp) Dataset end date.\n    :return: (tuple) Tuple of dictionaries (asset: data) for X, y, cont contract used to label the dataset.\n    '
    pass