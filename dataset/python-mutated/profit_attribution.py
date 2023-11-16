"""
This module is not well maintained.
"""
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import C
from ..data import D
from .position import Position

def get_benchmark_weight(bench, start_date=None, end_date=None, path=None, freq='day'):
    if False:
        i = 10
        return i + 15
    'get_benchmark_weight\n\n    get the stock weight distribution of the benchmark\n\n    :param bench:\n    :param start_date:\n    :param end_date:\n    :param path:\n    :param freq:\n\n    :return: The weight distribution of the the benchmark described by a pandas dataframe\n             Every row corresponds to a trading day.\n             Every column corresponds to a stock.\n             Every cell represents the strategy.\n\n    '
    if not path:
        path = Path(C.dpm.get_data_uri(freq)).expanduser() / 'raw' / 'AIndexMembers' / 'weights.csv'
    bench_weight_df = pd.read_csv(path, usecols=['code', 'date', 'index', 'weight'])
    bench_weight_df = bench_weight_df[bench_weight_df['index'] == bench]
    bench_weight_df['date'] = pd.to_datetime(bench_weight_df['date'])
    if start_date is not None:
        bench_weight_df = bench_weight_df[bench_weight_df.date >= start_date]
    if end_date is not None:
        bench_weight_df = bench_weight_df[bench_weight_df.date <= end_date]
    bench_stock_weight = bench_weight_df.pivot_table(index='date', columns='code', values='weight') / 100.0
    return bench_stock_weight

def get_stock_weight_df(positions):
    if False:
        while True:
            i = 10
    'get_stock_weight_df\n    :param positions: Given a positions from backtest result.\n    :return:          A weight distribution for the position\n    '
    stock_weight = []
    index = []
    for date in sorted(positions.keys()):
        pos = positions[date]
        if isinstance(pos, dict):
            pos = Position(position_dict=pos)
        index.append(date)
        stock_weight.append(pos.get_stock_weight_dict(only_stock=True))
    return pd.DataFrame(stock_weight, index=index)

def decompose_portofolio_weight(stock_weight_df, stock_group_df):
    if False:
        i = 10
        return i + 15
    "decompose_portofolio_weight\n\n    '''\n    :param stock_weight_df: a pandas dataframe to describe the portofolio by weight.\n                    every row corresponds to a  day\n                    every column corresponds to a stock.\n                    Here is an example below.\n                    code        SH600004  SH600006  SH600017  SH600022  SH600026  SH600037                      date\n                    2016-01-05  0.001543  0.001570  0.002732  0.001320  0.003000       NaN\n                    2016-01-06  0.001538  0.001569  0.002770  0.001417  0.002945       NaN\n                    ....\n    :param stock_group_df: a pandas dataframe to describe  the stock group.\n                    every row corresponds to a  day\n                    every column corresponds to a stock.\n                    the value in the cell repreponds the group id.\n                    Here is a example by for stock_group_df for industry. The value is the industry code\n                    instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008                      datetime\n                    2016-01-05  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    2016-01-06  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    ...\n    :return:        Two dict will be returned.  The group_weight and the stock_weight_in_group.\n                    The key is the group. The value is a Series or Dataframe to describe the weight of group or weight of stock\n    "
    all_group = np.unique(stock_group_df.values.flatten())
    all_group = all_group[~np.isnan(all_group)]
    group_weight = {}
    stock_weight_in_group = {}
    for group_key in all_group:
        group_mask = stock_group_df == group_key
        group_weight[group_key] = stock_weight_df[group_mask].sum(axis=1)
        stock_weight_in_group[group_key] = stock_weight_df[group_mask].divide(group_weight[group_key], axis=0)
    return (group_weight, stock_weight_in_group)

def decompose_portofolio(stock_weight_df, stock_group_df, stock_ret_df):
    if False:
        print('Hello World!')
    '\n    :param stock_weight_df: a pandas dataframe to describe the portofolio by weight.\n                    every row corresponds to a  day\n                    every column corresponds to a stock.\n                    Here is an example below.\n                    code        SH600004  SH600006  SH600017  SH600022  SH600026  SH600037                      date\n                    2016-01-05  0.001543  0.001570  0.002732  0.001320  0.003000       NaN\n                    2016-01-06  0.001538  0.001569  0.002770  0.001417  0.002945       NaN\n                    2016-01-07  0.001555  0.001546  0.002772  0.001393  0.002904       NaN\n                    2016-01-08  0.001564  0.001527  0.002791  0.001506  0.002948       NaN\n                    2016-01-11  0.001597  0.001476  0.002738  0.001493  0.003043       NaN\n                    ....\n\n    :param stock_group_df: a pandas dataframe to describe  the stock group.\n                    every row corresponds to a  day\n                    every column corresponds to a stock.\n                    the value in the cell repreponds the group id.\n                    Here is a example by for stock_group_df for industry. The value is the industry code\n                    instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008                      datetime\n                    2016-01-05  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    2016-01-06  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    2016-01-07  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    2016-01-08  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    2016-01-11  801780.0  801170.0  801040.0  801880.0  801180.0  801160.0\n                    ...\n\n    :param stock_ret_df:   a pandas dataframe to describe the stock return.\n                    every row corresponds to a day\n                    every column corresponds to a stock.\n                    the value in the cell repreponds the return of the group.\n                    Here is a example by for stock_ret_df.\n                    instrument  SH600000  SH600004  SH600005  SH600006  SH600007  SH600008                      datetime\n                    2016-01-05  0.007795  0.022070  0.099099  0.024707  0.009473  0.016216\n                    2016-01-06 -0.032597 -0.075205 -0.098361 -0.098985 -0.099707 -0.098936\n                    2016-01-07 -0.001142  0.022544  0.100000  0.004225  0.000651  0.047226\n                    2016-01-08 -0.025157 -0.047244 -0.038567 -0.098177 -0.099609 -0.074408\n                    2016-01-11  0.023460  0.004959 -0.034384  0.018663  0.014461  0.010962\n                    ...\n\n    :return: It will decompose the portofolio to the group weight and group return.\n    '
    all_group = np.unique(stock_group_df.values.flatten())
    all_group = all_group[~np.isnan(all_group)]
    (group_weight, stock_weight_in_group) = decompose_portofolio_weight(stock_weight_df, stock_group_df)
    group_ret = {}
    for (group_key, val) in stock_weight_in_group.items():
        stock_weight_in_group_start_date = min(val.index)
        stock_weight_in_group_end_date = max(val.index)
        temp_stock_ret_df = stock_ret_df[(stock_ret_df.index >= stock_weight_in_group_start_date) & (stock_ret_df.index <= stock_weight_in_group_end_date)]
        group_ret[group_key] = (temp_stock_ret_df * val).sum(axis=1)
        group_ret[group_key][group_weight[group_key] == 0.0] = np.nan
    group_weight_df = pd.DataFrame(group_weight)
    group_ret_df = pd.DataFrame(group_ret)
    return (group_weight_df, group_ret_df)

def get_daily_bin_group(bench_values, stock_values, group_n):
    if False:
        while True:
            i = 10
    'get_daily_bin_group\n    Group the values of the stocks of benchmark into several bins in a day.\n    Put the stocks into these bins.\n\n    :param bench_values: A series contains the value of stocks in benchmark.\n                         The index is the stock code.\n    :param stock_values: A series contains the value of stocks of your portofolio\n                         The index is the stock code.\n    :param group_n:      Bins will be produced\n\n    :return:             A series with the same size and index as the stock_value.\n                         The value in the series is the group id of the bins.\n                         The No.1 bin contains the biggest values.\n    '
    stock_group = stock_values.copy()
    split_points = np.percentile(bench_values[~bench_values.isna()], np.linspace(0, 100, group_n + 1))
    (split_points[0], split_points[-1]) = (-np.inf, np.inf)
    for (i, (lb, up)) in enumerate(zip(split_points, split_points[1:])):
        stock_group.loc[stock_values[(stock_values >= lb) & (stock_values < up)].index] = group_n - i
    return stock_group

def get_stock_group(stock_group_field_df, bench_stock_weight_df, group_method, group_n=None):
    if False:
        print('Hello World!')
    if group_method == 'category':
        return stock_group_field_df
    elif group_method == 'bins':
        assert group_n is not None
        new_stock_group_df = stock_group_field_df.copy().loc[bench_stock_weight_df.index.min():bench_stock_weight_df.index.max()]
        for (idx, row) in (~bench_stock_weight_df.isna()).iterrows():
            bench_values = stock_group_field_df.loc[idx, row[row].index]
            new_stock_group_df.loc[idx] = get_daily_bin_group(bench_values, stock_group_field_df.loc[idx], group_n=group_n)
        return new_stock_group_df

def brinson_pa(positions, bench='SH000905', group_field='industry', group_method='category', group_n=None, deal_price='vwap', freq='day'):
    if False:
        i = 10
        return i + 15
    "brinson profit attribution\n\n    :param positions: The position produced by the backtest class\n    :param bench: The benchmark for comparing. TODO: if no benchmark is set, the equal-weighted is used.\n    :param group_field: The field used to set the group for assets allocation.\n                        `industry` and `market_value` is often used.\n    :param group_method: 'category' or 'bins'. The method used to set the group for asstes allocation\n                         `bin` will split the value into `group_n` bins and each bins represents a group\n    :param group_n: . Only used when group_method == 'bins'.\n\n    :return:\n        A dataframe with three columns: RAA(excess Return of Assets Allocation),  RSS(excess Return of Stock Selectino),  RTotal(Total excess Return)\n                                        Every row corresponds to a trading day, the value corresponds to the next return for this trading day\n        The middle info of brinson profit attribution\n    "
    dates = sorted(positions.keys())
    (start_date, end_date) = (min(dates), max(dates))
    bench_stock_weight = get_benchmark_weight(bench, start_date, end_date, freq)
    if not group_field.startswith('$'):
        group_field = '$' + group_field
    if not deal_price.startswith('$'):
        deal_price = '$' + deal_price
    shift_start_date = start_date - datetime.timedelta(days=250)
    instruments = D.list_instruments(D.instruments(market='all'), start_time=shift_start_date, end_time=end_date, as_list=True, freq=freq)
    stock_df = D.features(instruments, [group_field, deal_price], start_time=shift_start_date, end_time=end_date, freq=freq)
    stock_df.columns = [group_field, 'deal_price']
    stock_group_field = stock_df[group_field].unstack().T
    stock_group_field = stock_group_field.fillna(method='ffill')
    stock_group_field = stock_group_field.loc[start_date:end_date]
    stock_group = get_stock_group(stock_group_field, bench_stock_weight, group_method, group_n)
    deal_price_df = stock_df['deal_price'].unstack().T
    deal_price_df = deal_price_df.fillna(method='ffill')
    stock_ret = (deal_price_df - deal_price_df.shift(1)) / deal_price_df.shift(1)
    stock_ret = stock_ret.shift(-1).loc[start_date:end_date]
    port_stock_weight_df = get_stock_weight_df(positions)
    (port_group_weight_df, port_group_ret_df) = decompose_portofolio(port_stock_weight_df, stock_group, stock_ret)
    (bench_group_weight_df, bench_group_ret_df) = decompose_portofolio(bench_stock_weight, stock_group, stock_ret)
    mod_port_group_ret_df = port_group_ret_df.copy()
    mod_port_group_ret_df[mod_port_group_ret_df.isna()] = bench_group_ret_df
    Q1 = (bench_group_weight_df * bench_group_ret_df).sum(axis=1)
    Q2 = (port_group_weight_df * bench_group_ret_df).sum(axis=1)
    Q3 = (bench_group_weight_df * mod_port_group_ret_df).sum(axis=1)
    Q4 = (port_group_weight_df * mod_port_group_ret_df).sum(axis=1)
    return (pd.DataFrame({'RAA': Q2 - Q1, 'RSS': Q3 - Q1, 'RIN': Q4 - Q3 - Q2 + Q1, 'RTotal': Q4 - Q1}), {'port_group_ret': port_group_ret_df, 'port_group_weight': port_group_weight_df, 'bench_group_ret': bench_group_ret_df, 'bench_group_weight': bench_group_weight_df, 'stock_group': stock_group, 'bench_stock_weight': bench_stock_weight, 'port_stock_weight': port_stock_weight_df, 'stock_ret': stock_ret})