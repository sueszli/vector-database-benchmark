import datetime
import logging
from typing import List, Optional, Union
import dateutil
import numpy as np
from pandas import DataFrame, MultiIndex, Series, DatetimeIndex, Index
import pandas as pd
from .._config import FAST_CHECK_DF_SERIALIZABLE
from .._util import NP_OBJECT_DTYPE
from ..exceptions import ArcticException
try:
    from pandas._libs.tslibs.timezones import is_utc
except ImportError:

    def is_utc(tz):
        if False:
            while True:
                i = 10
        raise AssertionError('Should never be called for Pandas versions where this is not an exported function')
try:
    from pandas._libs.tslib import Timestamp
    from pandas._libs.tslibs.timezones import get_timezone
except ImportError:
    try:
        from pandas._libs.tslib import Timestamp, get_timezone
    except ImportError:
        from pandas.tslib import Timestamp, get_timezone
log = logging.getLogger(__name__)
PD_VER = pd.__version__
DTN64_DTYPE = 'datetime64[ns]'

def set_fast_check_df_serializable(config):
    if False:
        i = 10
        return i + 15
    global FAST_CHECK_DF_SERIALIZABLE
    FAST_CHECK_DF_SERIALIZABLE = bool(config)

def _to_primitive(arr, string_max_len=None, forced_dtype=None):
    if False:
        print('Hello World!')
    if arr.dtype.hasobject:
        if len(arr) > 0 and isinstance(arr[0], Timestamp):
            return np.array([t.value for t in arr], dtype=DTN64_DTYPE)
        if forced_dtype is not None:
            casted_arr = arr.astype(dtype=forced_dtype, copy=False)
        elif string_max_len is not None:
            casted_arr = np.array(arr.astype('U{:d}'.format(string_max_len)))
        else:
            casted_arr = np.array(list(arr))
        if np.array_equal(arr, casted_arr):
            return casted_arr
    return arr

def treat_tz_as_dateutil(tz) -> bool:
    if False:
        while True:
            i = 10
    '\n    Return whether the given tz object is from `dateutil`\n\n    Vendored from Pandas:\n    https://github.com/pandas-dev/pandas/blob/v1.3.5/pandas/_libs/tslibs/timezones.pyx#L66-L67\n    '
    return hasattr(tz, '_trans_list') and hasattr(tz, '_trans_idx')

def consistent_get_timezone_str(tz: Union[datetime.tzinfo, str]) -> str:
    if False:
        print('Hello World!')
    '\n    Convert a tzinfo object to a serializable string\n\n    Unlike the Pandas `get_timezone` function, this function should always return a string.\n    '
    if isinstance(tz, str):
        return tz
    if PD_VER < '0.24.0':
        return str(get_timezone(tz))
    if isinstance(tz, dateutil.tz.tzutc):
        return 'UTC'
    if PD_VER < '1.3.0':
        return str(get_timezone(tz))
    if is_utc(tz) and treat_tz_as_dateutil(tz):
        return 'dateutil/' + tz._filename
    return str(get_timezone(tz))

def _multi_index_to_records(index, empty_index):
    if False:
        while True:
            i = 10
    if not empty_index:
        ix_vals = list(map(np.array, [index.get_level_values(i) for i in range(index.nlevels)]))
    else:
        ix_vals = [np.array([]) for n in index.names]
    index_names = list(index.names)
    count = 0
    for (i, n) in enumerate(index_names):
        if n is None:
            index_names[i] = 'level_%d' % count
            count += 1
            log.info('Level in MultiIndex has no name, defaulting to %s' % index_names[i])
    index_tz = []
    for i in index.levels:
        if isinstance(i, DatetimeIndex) and i.tz is not None:
            index_tz.append(consistent_get_timezone_str(i.tz))
        else:
            index_tz.append(None)
    return (ix_vals, index_names, index_tz)

class PandasSerializer(object):

    def _index_to_records(self, df):
        if False:
            return 10
        metadata = {}
        index = df.index
        index_tz: Union[Optional[str], List[Optional[str]]]
        if isinstance(index, MultiIndex):
            (ix_vals, index_names, index_tz) = _multi_index_to_records(index, len(df) == 0)
        else:
            ix_vals = [index.values]
            index_names = list(index.names)
            if index_names[0] is None:
                index_names = ['index']
                log.info("Index has no name, defaulting to 'index'")
            if isinstance(index, DatetimeIndex) and index.tz is not None:
                index_tz = consistent_get_timezone_str(index.tz)
            else:
                index_tz = None
        if index_tz is not None:
            metadata['index_tz'] = index_tz
        metadata['index'] = index_names
        return (index_names, ix_vals, metadata)

    def _index_from_records(self, recarr):
        if False:
            while True:
                i = 10
        index = recarr.dtype.metadata['index']
        if len(index) == 1:
            rtn = Index(np.copy(recarr[str(index[0])]), name=index[0])
            if isinstance(rtn, DatetimeIndex) and 'index_tz' in recarr.dtype.metadata:
                if PD_VER >= '1.0.4':
                    if isinstance(recarr.dtype.metadata['index_tz'], list):
                        rtn = rtn.tz_localize('UTC').tz_convert(recarr.dtype.metadata['index_tz'][0])
                    else:
                        rtn = rtn.tz_localize('UTC').tz_convert(recarr.dtype.metadata['index_tz'])
                else:
                    rtn = rtn.tz_localize('UTC').tz_convert(recarr.dtype.metadata['index_tz'])
        else:
            level_arrays = []
            index_tz = recarr.dtype.metadata.get('index_tz', [])
            for (level_no, index_name) in enumerate(index):
                level = Index(np.copy(recarr[str(index_name)]))
                if level_no < len(index_tz):
                    tz = index_tz[level_no]
                    if tz is not None:
                        if not isinstance(level, DatetimeIndex) and len(level) == 0:
                            level = DatetimeIndex([], tz=tz)
                        else:
                            level = level.tz_localize('UTC').tz_convert(tz)
                level_arrays.append(level)
            rtn = MultiIndex.from_arrays(level_arrays, names=index)
        return rtn

    def _to_records(self, df, string_max_len=None, forced_dtype=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Similar to DataFrame.to_records()\n        Differences:\n            Attempt type conversion for pandas columns stored as objects (e.g. strings),\n            as we can only store primitives in the ndarray.\n            Use dtype metadata to store column and index names.\n\n        string_max_len: integer - enforces a string size on the dtype, if any\n                                  strings exist in the record\n        '
        (index_names, ix_vals, metadata) = self._index_to_records(df)
        (columns, column_vals, multi_column) = self._column_data(df)
        if '' in columns:
            raise ArcticException('Cannot use empty string as a column name.')
        if multi_column is not None:
            metadata['multi_column'] = multi_column
        metadata['columns'] = columns
        names = index_names + columns
        arrays = []
        for (arr, name) in zip(ix_vals + column_vals, index_names + columns):
            arrays.append(_to_primitive(arr, string_max_len, forced_dtype=None if forced_dtype is None else forced_dtype[name]))
        if forced_dtype is None:
            dtype = np.dtype([(str(x), v.dtype) if len(v.shape) == 1 else (str(x), v.dtype, v.shape[1]) for (x, v) in zip(names, arrays)], metadata=metadata)
        else:
            dtype = forced_dtype
        rtn = np.rec.fromarrays(arrays, dtype=dtype, names=names)
        return (rtn, dtype)

    def fast_check_serializable(self, df):
        if False:
            print('Hello World!')
        "\n        Convert efficiently the frame's object-columns/object-index/multi-index/multi-column to\n        records, by creating a recarray only for the object fields instead for the whole dataframe.\n        If we have no object dtypes, we can safely convert only the first row to recarray to test if serializable.\n        Previously we'd serialize twice the full dataframe when it included object fields or multi-index/columns.\n\n        Parameters\n        ----------\n        df: `pandas.DataFrame` or `pandas.Series`\n\n        Returns\n        -------\n        `tuple[numpy.core.records.recarray, dict[str, numpy.dtype]`\n            If any object dtypes are detected in columns or index will return a dict with field-name -> dtype\n             mappings, and empty dict otherwise.\n        "
        (i_dtype, f_dtypes) = (df.index.dtype, df.dtypes)
        index_has_object = df.index.dtype is NP_OBJECT_DTYPE
        fields_with_object = [f for f in df.columns if f_dtypes[f] is NP_OBJECT_DTYPE]
        if df.empty or (not index_has_object and (not fields_with_object)):
            (arr, _) = self._to_records(df.iloc[:10])
            return (arr, {})
        df_objects_only = df[fields_with_object if fields_with_object else df.columns[:2]]
        (arr, dtype) = self._to_records(df_objects_only)
        return (arr, {f: dtype[f] for f in dtype.names})

    def can_convert_to_records_without_objects(self, df, symbol):
        if False:
            for i in range(10):
                print('nop')
        try:
            if FAST_CHECK_DF_SERIALIZABLE:
                (arr, _) = self.fast_check_serializable(df)
            else:
                (arr, _) = self._to_records(df)
        except Exception as e:
            log.warning('Pandas dataframe %s caused exception "%s" when attempting to convert to records. Saving as Blob.' % (symbol, repr(e)))
            return False
        else:
            if arr.dtype.hasobject:
                log.warning('Pandas dataframe %s contains Objects, saving as Blob' % symbol)
                return False
            elif any([len(x[0].shape) for x in arr.dtype.fields.values()]):
                log.warning('Pandas dataframe %s contains >1 dimensional arrays, saving as Blob' % symbol)
                return False
            else:
                return True

    def serialize(self, item, string_max_len=None, forced_dtype=None):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def deserialize(self, item, force_bytes_to_unicode=False):
        if False:
            while True:
                i = 10
        raise NotImplementedError

class SeriesSerializer(PandasSerializer):
    TYPE = 'series'

    def _column_data(self, s):
        if False:
            i = 10
            return i + 15
        if s.name is None:
            log.info("Series has no name, defaulting to 'values'")
        columns = [s.name if s.name else 'values']
        column_vals = [s.values]
        return (columns, column_vals, None)

    def deserialize(self, item, force_bytes_to_unicode=False):
        if False:
            for i in range(10):
                print('nop')
        index = self._index_from_records(item)
        name = item.dtype.names[-1]
        data = item[name]
        if force_bytes_to_unicode:
            if len(data) and isinstance(data[0], bytes):
                data = data.astype('unicode')
            if isinstance(index, MultiIndex):
                unicode_indexes = []
                for level in range(len(index.levels)):
                    _index = index.get_level_values(level)
                    if isinstance(_index[0], bytes):
                        _index = _index.astype('unicode')
                    unicode_indexes.append(_index)
                index = unicode_indexes
            elif len(index) and type(index[0]) == bytes:
                index = index.astype('unicode')
        if PD_VER < '0.23.0':
            return Series.from_array(data, index=index, name=name)
        else:
            return Series(data, index=index, name=name)

    def serialize(self, item, string_max_len=None, forced_dtype=None):
        if False:
            for i in range(10):
                print('nop')
        return self._to_records(item, string_max_len, forced_dtype)

class DataFrameSerializer(PandasSerializer):
    TYPE = 'df'

    def _column_data(self, df):
        if False:
            return 10
        columns = list(map(str, df.columns))
        if columns != list(df.columns):
            log.info('Dataframe column names converted to strings')
        column_vals = [df[c].values for c in df.columns]
        if isinstance(df.columns, MultiIndex):
            (ix_vals, ix_names, _) = _multi_index_to_records(df.columns, False)
            vals = [list(val) for val in ix_vals]
            str_vals = [list(map(str, val)) for val in ix_vals]
            if vals != str_vals:
                log.info('Dataframe column names converted to strings')
            return (columns, column_vals, {'names': list(ix_names), 'values': str_vals})
        else:
            return (columns, column_vals, None)

    def deserialize(self, item, force_bytes_to_unicode=False):
        if False:
            while True:
                i = 10
        index = self._index_from_records(item)
        column_fields = [x for x in item.dtype.names if x not in item.dtype.metadata['index']]
        multi_column = item.dtype.metadata.get('multi_column')
        if len(item) == 0:
            rdata = item[column_fields] if len(column_fields) > 0 else None
            if multi_column is not None:
                columns = MultiIndex.from_arrays(multi_column['values'], names=multi_column['names'])
                return DataFrame(rdata, index=index, columns=columns)
            else:
                return DataFrame(rdata, index=index)
        columns = item.dtype.metadata['columns']
        df = DataFrame(data=item[column_fields], index=index, columns=columns)
        if multi_column is not None:
            df.columns = MultiIndex.from_arrays(multi_column['values'], names=multi_column['names'])
        if force_bytes_to_unicode:
            for c in df.select_dtypes(object):
                if type(df[c].iloc[0]) == bytes:
                    df[c] = df[c].str.decode('utf-8')
            if isinstance(df.index, MultiIndex):
                unicode_indexes = []
                for level in range(len(df.index.levels)):
                    _index = df.index.get_level_values(level)
                    if isinstance(_index[0], bytes):
                        _index = _index.astype('unicode')
                    unicode_indexes.append(_index)
                df.index = unicode_indexes
            elif type(df.index[0]) == bytes:
                df.index = df.index.astype('unicode')
            if not df.columns.empty and type(df.columns[0]) == bytes:
                df.columns = df.columns.astype('unicode')
        return df

    def serialize(self, item, string_max_len=None, forced_dtype=None):
        if False:
            print('Hello World!')
        return self._to_records(item, string_max_len, forced_dtype)