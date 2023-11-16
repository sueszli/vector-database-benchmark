"""
Utilities for working with pandas objects.
"""
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import operator as op
import warnings
import numpy as np
import pandas as pd
from distutils.version import StrictVersion
from trading_calendars.utils.pandas_utils import days_at_time
pandas_version = StrictVersion(pd.__version__)
new_pandas = pandas_version >= StrictVersion('0.19')
skip_pipeline_new_pandas = 'Pipeline categoricals are not yet compatible with pandas >=0.19'
if pandas_version >= StrictVersion('0.20'):

    def normalize_date(dt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Normalize datetime.datetime value to midnight. Returns datetime.date as\n        a datetime.datetime at midnight\n\n        Returns\n        -------\n        normalized : datetime.datetime or Timestamp\n        '
        return dt.normalize()
else:
    from pandas.tseries.tools import normalize_date

def july_5th_holiday_observance(datetime_index):
    if False:
        i = 10
        return i + 15
    return datetime_index[datetime_index.year != 2013]

def explode(df):
    if False:
        i = 10
        return i + 15
    '\n    Take a DataFrame and return a triple of\n\n    (df.index, df.columns, df.values)\n    '
    return (df.index, df.columns, df.values)

def _time_to_micros(time):
    if False:
        while True:
            i = 10
    'Convert a time into microseconds since midnight.\n    Parameters\n    ----------\n    time : datetime.time\n        The time to convert.\n    Returns\n    -------\n    us : int\n        The number of microseconds since midnight.\n    Notes\n    -----\n    This does not account for leap seconds or daylight savings.\n    '
    seconds = time.hour * 60 * 60 + time.minute * 60 + time.second
    return 1000000 * seconds + time.microsecond
_opmap = dict(zip(product((True, False), repeat=3), product((op.le, op.lt), (op.le, op.lt), (op.and_, op.or_))))

def mask_between_time(dts, start, end, include_start=True, include_end=True):
    if False:
        return 10
    'Return a mask of all of the datetimes in ``dts`` that are between\n    ``start`` and ``end``.\n    Parameters\n    ----------\n    dts : pd.DatetimeIndex\n        The index to mask.\n    start : time\n        Mask away times less than the start.\n    end : time\n        Mask away times greater than the end.\n    include_start : bool, optional\n        Inclusive on ``start``.\n    include_end : bool, optional\n        Inclusive on ``end``.\n    Returns\n    -------\n    mask : np.ndarray[bool]\n        A bool array masking ``dts``.\n    See Also\n    --------\n    :meth:`pandas.DatetimeIndex.indexer_between_time`\n    '
    time_micros = dts._get_time_micros()
    start_micros = _time_to_micros(start)
    end_micros = _time_to_micros(end)
    (left_op, right_op, join_op) = _opmap[bool(include_start), bool(include_end), start_micros <= end_micros]
    return join_op(left_op(start_micros, time_micros), right_op(time_micros, end_micros))

def find_in_sorted_index(dts, dt):
    if False:
        print('Hello World!')
    "\n    Find the index of ``dt`` in ``dts``.\n\n    This function should be used instead of `dts.get_loc(dt)` if the index is\n    large enough that we don't want to initialize a hash table in ``dts``. In\n    particular, this should always be used on minutely trading calendars.\n\n    Parameters\n    ----------\n    dts : pd.DatetimeIndex\n        Index in which to look up ``dt``. **Must be sorted**.\n    dt : pd.Timestamp\n        ``dt`` to be looked up.\n\n    Returns\n    -------\n    ix : int\n        Integer index such that dts[ix] == dt.\n\n    Raises\n    ------\n    KeyError\n        If dt is not in ``dts``.\n    "
    ix = dts.searchsorted(dt)
    if ix == len(dts) or dts[ix] != dt:
        raise LookupError('{dt} is not in {dts}'.format(dt=dt, dts=dts))
    return ix

def nearest_unequal_elements(dts, dt):
    if False:
        while True:
            i = 10
    '\n    Find values in ``dts`` closest but not equal to ``dt``.\n\n    Returns a pair of (last_before, first_after).\n\n    When ``dt`` is less than any element in ``dts``, ``last_before`` is None.\n    When ``dt`` is greater any element in ``dts``, ``first_after`` is None.\n\n    ``dts`` must be unique and sorted in increasing order.\n\n    Parameters\n    ----------\n    dts : pd.DatetimeIndex\n        Dates in which to search.\n    dt : pd.Timestamp\n        Date for which to find bounds.\n    '
    if not dts.is_unique:
        raise ValueError('dts must be unique')
    if not dts.is_monotonic_increasing:
        raise ValueError('dts must be sorted in increasing order')
    if not len(dts):
        return (None, None)
    sortpos = dts.searchsorted(dt, side='left')
    try:
        sortval = dts[sortpos]
    except IndexError:
        return (dts[-1], None)
    if dt < sortval:
        lower_ix = sortpos - 1
        upper_ix = sortpos
    elif dt == sortval:
        lower_ix = sortpos - 1
        upper_ix = sortpos + 1
    else:
        lower_ix = sortpos
        upper_ix = sortpos + 1
    lower_value = dts[lower_ix] if lower_ix >= 0 else None
    upper_value = dts[upper_ix] if upper_ix < len(dts) else None
    return (lower_value, upper_value)

def timedelta_to_integral_seconds(delta):
    if False:
        i = 10
        return i + 15
    '\n    Convert a pd.Timedelta to a number of seconds as an int.\n    '
    return int(delta.total_seconds())

def timedelta_to_integral_minutes(delta):
    if False:
        while True:
            i = 10
    '\n    Convert a pd.Timedelta to a number of minutes as an int.\n    '
    return timedelta_to_integral_seconds(delta) // 60

@contextmanager
def ignore_pandas_nan_categorical_warning():
    if False:
        for i in range(10):
            print('nop')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        yield
_INDEXER_NAMES = ['_' + name for (name, _) in pd.core.indexing.get_indexers_list()]

def clear_dataframe_indexer_caches(df):
    if False:
        for i in range(10):
            print('nop')
    '\n    Clear cached attributes from a pandas DataFrame.\n\n    By default pandas memoizes indexers (`iloc`, `loc`, `ix`, etc.) objects on\n    DataFrames, resulting in refcycles that can lead to unexpectedly long-lived\n    DataFrames. This function attempts to clear those cycles by deleting the\n    cached indexers from the frame.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n    '
    for attr in _INDEXER_NAMES:
        try:
            delattr(df, attr)
        except AttributeError:
            pass

def categorical_df_concat(df_list, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Prepare list of pandas DataFrames to be used as input to pd.concat.\n    Ensure any columns of type 'category' have the same categories across each\n    dataframe.\n\n    Parameters\n    ----------\n    df_list : list\n        List of dataframes with same columns.\n    inplace : bool\n        True if input list can be modified. Default is False.\n\n    Returns\n    -------\n    concatenated : df\n        Dataframe of concatenated list.\n    "
    if not inplace:
        df_list = deepcopy(df_list)
    df = df_list[0]
    if not all([df.dtypes.equals(df_i.dtypes) for df_i in df_list[1:]]):
        raise ValueError('Input DataFrames must have the same columns/dtypes.')
    categorical_columns = df.columns[df.dtypes == 'category']
    for col in categorical_columns:
        new_categories = _sort_set_none_first(_union_all((frame[col].cat.categories for frame in df_list)))
        with ignore_pandas_nan_categorical_warning():
            for df in df_list:
                df[col].cat.set_categories(new_categories, inplace=True)
    return pd.concat(df_list)

def _union_all(iterables):
    if False:
        return 10
    'Union entries in ``iterables`` into a set.\n    '
    return set().union(*iterables)

def _sort_set_none_first(set_):
    if False:
        for i in range(10):
            print('nop')
    'Sort a set, sorting ``None`` before other elements, if present.\n    '
    if None in set_:
        set_.remove(None)
        out = [None]
        out.extend(sorted(set_))
        set_.add(None)
        return out
    else:
        return sorted(set_)

def empty_dataframe(*columns):
    if False:
        print('Hello World!')
    "Create an empty dataframe with columns of particular types.\n\n    Parameters\n    ----------\n    *columns\n        The (column_name, column_dtype) pairs.\n\n    Returns\n    -------\n    typed_dataframe : pd.DataFrame\n        The empty typed dataframe.\n\n    Examples\n    --------\n    >>> df = empty_dataframe(\n    ...     ('a', 'int64'),\n    ...     ('b', 'float64'),\n    ...     ('c', 'datetime64[ns]'),\n    ... )\n\n    >>> df\n    Empty DataFrame\n    Columns: [a, b, c]\n    Index: []\n\n    df.dtypes\n    a             int64\n    b           float64\n    c    datetime64[ns]\n    dtype: object\n    "
    return pd.DataFrame(np.array([], dtype=list(columns)))

def check_indexes_all_same(indexes, message='Indexes are not equal.'):
    if False:
        for i in range(10):
            print('nop')
    'Check that a list of Index objects are all equal.\n\n    Parameters\n    ----------\n    indexes : iterable[pd.Index]\n        Iterable of indexes to check.\n\n    Raises\n    ------\n    ValueError\n        If the indexes are not all the same.\n    '
    iterator = iter(indexes)
    first = next(iterator)
    for other in iterator:
        same = first == other
        if not same.all():
            bad_loc = np.flatnonzero(~same)[0]
            raise ValueError('{}\nFirst difference is at index {}: {} != {}'.format(message, bad_loc, first[bad_loc], other[bad_loc]))