"""
Utility functions for multi-index dataframes. Useful for creating bi-temporal timeseries.
"""
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import to_datetime as dt
from .date import mktz
PD_VER = pd.__version__
logger = logging.getLogger(__name__)

def fancy_group_by(df, grouping_level=0, aggregate_level=1, method='last', max_=None, min_=None, within=None):
    if False:
        while True:
            i = 10
    ' Dataframe group-by operation that supports aggregating by different methods on the index.\n\n    Parameters\n    ----------\n    df: ``DataFrame``\n        Pandas dataframe with a MultiIndex\n    grouping_level: ``int`` or ``str`` or ``list`` of ``str``\n        Index level to group by. Defaults to 0.\n    aggregate_level: ``int`` or ``str``\n        Index level to aggregate by. Defaults to 1.\n    method: ``str``\n        Aggregation method. One of\n            last: Use the last (lexicographically) value from each group\n            first: Use the first value from each group\n    max_: <any>\n        If set, will limit results to those having aggregate level values <= this value\n    min_: <any>\n        If set, will limit results to those having aggregate level values >= this value\n    within: Any type supported by the index, or ``DateOffset``/timedelta-like for ``DatetimeIndex``.\n        If set, will limit results to those having aggregate level values within this range of the group value.\n        Note that this is currently unsupported for Multi-index of depth > 2\n    '
    if method not in ('first', 'last'):
        raise ValueError('Invalid method')
    if isinstance(aggregate_level, str):
        aggregate_level = df.index.names.index(aggregate_level)
    if max_ is not None or min_ is not None or within is not None:
        agg_idx = df.index.get_level_values(aggregate_level)
        mask = np.full(len(agg_idx), True, dtype='b1')
        if max_ is not None:
            mask &= agg_idx <= max_
        if min_ is not None:
            mask &= agg_idx >= min_
        if within is not None:
            group_idx = df.index.get_level_values(grouping_level)
            if isinstance(agg_idx, pd.DatetimeIndex):
                mask &= group_idx >= agg_idx.shift(-1, freq=within)
            else:
                mask &= group_idx >= agg_idx - within
        df = df.loc[mask]
    if df.index.lexsort_depth < aggregate_level + 1:
        df = df.sort_index(level=grouping_level)
    gb = df.groupby(level=grouping_level)
    if method == 'last':
        return gb.last()
    return gb.first()

def groupby_asof(df, as_of=None, dt_col='sample_dt', asof_col='observed_dt'):
    if False:
        return 10
    ' Common use case for selecting the latest rows from a bitemporal dataframe as-of a certain date.\n\n    Parameters\n    ----------\n    df: ``pd.DataFrame``\n        Dataframe with a MultiIndex index\n    as_of: ``datetime``\n        Return a timeseries with values observed <= this as-of date. By default, the latest observed\n        values will be returned.\n    dt_col: ``str`` or ``int``\n        Name or index of the column in the MultiIndex that is the sample date\n    asof_col: ``str`` or ``int``\n        Name or index of the column in the MultiIndex that is the observed date\n    '
    if as_of:
        if as_of.tzinfo is None and df.index.get_level_values(asof_col).tz is not None:
            as_of = as_of.replace(tzinfo=mktz())
    return fancy_group_by(df, grouping_level=dt_col, aggregate_level=asof_col, method='last', max_=as_of)

def multi_index_insert_row(df, index_row, values_row):
    if False:
        for i in range(10):
            print('nop')
    ' Return a new dataframe with a row inserted for a multi-index dataframe.\n        This will sort the rows according to the ordered multi-index levels.\n    '
    if PD_VER < '0.24.0':
        row_index = pd.MultiIndex(levels=[[i] for i in index_row], labels=[[0] for i in index_row])
    else:
        row_index = pd.MultiIndex(levels=[[i] for i in index_row], codes=[[0] for i in index_row])
    row = pd.DataFrame(values_row, index=row_index, columns=df.columns)
    df = pd.concat((df, row))
    if df.index.lexsort_depth == len(index_row) and df.index[-2] < df.index[-1]:
        return df
    return df.sort_index()

def insert_at(df, sample_date, values):
    if False:
        i = 10
        return i + 15
    ' Insert some values into a bi-temporal dataframe.\n        This is like what would happen when we get a price correction.\n    '
    observed_dt = dt(datetime.now())
    return multi_index_insert_row(df, [sample_date, observed_dt], values)