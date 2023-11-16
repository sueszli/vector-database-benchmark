from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import Buffer, Column, ColumnNullType, DataFrame as DataFrameXchg, DtypeKind
from pandas.core.interchange.utils import ArrowCTypes, Endianness
_NP_DTYPES: dict[DtypeKind, dict[int, Any]] = {DtypeKind.INT: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}, DtypeKind.UINT: {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}, DtypeKind.FLOAT: {32: np.float32, 64: np.float64}, DtypeKind.BOOL: {1: bool, 8: bool}}

def from_dataframe(df, allow_copy: bool=True) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    "\n    Build a ``pd.DataFrame`` from any DataFrame supporting the interchange protocol.\n\n    Parameters\n    ----------\n    df : DataFrameXchg\n        Object supporting the interchange protocol, i.e. `__dataframe__` method.\n    allow_copy : bool, default: True\n        Whether to allow copying the memory to perform the conversion\n        (if false then zero-copy approach is requested).\n\n    Returns\n    -------\n    pd.DataFrame\n\n    Examples\n    --------\n    >>> df_not_necessarily_pandas = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})\n    >>> interchange_object = df_not_necessarily_pandas.__dataframe__()\n    >>> interchange_object.column_names()\n    Index(['A', 'B'], dtype='object')\n    >>> df_pandas = (pd.api.interchange.from_dataframe\n    ...              (interchange_object.select_columns_by_name(['A'])))\n    >>> df_pandas\n         A\n    0    1\n    1    2\n\n    These methods (``column_names``, ``select_columns_by_name``) should work\n    for any dataframe library which implements the interchange protocol.\n    "
    if isinstance(df, pd.DataFrame):
        return df
    if not hasattr(df, '__dataframe__'):
        raise ValueError('`df` does not support __dataframe__')
    return _from_dataframe(df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy)

def _from_dataframe(df: DataFrameXchg, allow_copy: bool=True):
    if False:
        while True:
            i = 10
    '\n    Build a ``pd.DataFrame`` from the DataFrame interchange object.\n\n    Parameters\n    ----------\n    df : DataFrameXchg\n        Object supporting the interchange protocol, i.e. `__dataframe__` method.\n    allow_copy : bool, default: True\n        Whether to allow copying the memory to perform the conversion\n        (if false then zero-copy approach is requested).\n\n    Returns\n    -------\n    pd.DataFrame\n    '
    pandas_dfs = []
    for chunk in df.get_chunks():
        pandas_df = protocol_df_chunk_to_pandas(chunk)
        pandas_dfs.append(pandas_df)
    if not allow_copy and len(pandas_dfs) > 1:
        raise RuntimeError('To join chunks a copy is required which is forbidden by allow_copy=False')
    if not pandas_dfs:
        pandas_df = protocol_df_chunk_to_pandas(df)
    elif len(pandas_dfs) == 1:
        pandas_df = pandas_dfs[0]
    else:
        pandas_df = pd.concat(pandas_dfs, axis=0, ignore_index=True, copy=False)
    index_obj = df.metadata.get('pandas.index', None)
    if index_obj is not None:
        pandas_df.index = index_obj
    return pandas_df

def protocol_df_chunk_to_pandas(df: DataFrameXchg) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    '\n    Convert interchange protocol chunk to ``pd.DataFrame``.\n\n    Parameters\n    ----------\n    df : DataFrameXchg\n\n    Returns\n    -------\n    pd.DataFrame\n    '
    columns: dict[str, Any] = {}
    buffers = []
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f'Column {name} is not a string')
        if name in columns:
            raise ValueError(f'Column {name} is not unique')
        col = df.get_column_by_name(name)
        dtype = col.dtype[0]
        if dtype in (DtypeKind.INT, DtypeKind.UINT, DtypeKind.FLOAT, DtypeKind.BOOL):
            (columns[name], buf) = primitive_column_to_ndarray(col)
        elif dtype == DtypeKind.CATEGORICAL:
            (columns[name], buf) = categorical_column_to_series(col)
        elif dtype == DtypeKind.STRING:
            (columns[name], buf) = string_column_to_ndarray(col)
        elif dtype == DtypeKind.DATETIME:
            (columns[name], buf) = datetime_column_to_ndarray(col)
        else:
            raise NotImplementedError(f'Data type {dtype} not handled yet')
        buffers.append(buf)
    pandas_df = pd.DataFrame(columns)
    pandas_df.attrs['_INTERCHANGE_PROTOCOL_BUFFERS'] = buffers
    return pandas_df

def primitive_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    if False:
        while True:
            i = 10
    '\n    Convert a column holding one of the primitive dtypes to a NumPy array.\n\n    A primitive type is one of: int, uint, float, bool.\n\n    Parameters\n    ----------\n    col : Column\n\n    Returns\n    -------\n    tuple\n        Tuple of np.ndarray holding the data and the memory owner object\n        that keeps the memory alive.\n    '
    buffers = col.get_buffers()
    (data_buff, data_dtype) = buffers['data']
    data = buffer_to_ndarray(data_buff, data_dtype, offset=col.offset, length=col.size())
    data = set_nulls(data, col, buffers['validity'])
    return (data, buffers)

def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
    if False:
        print('Hello World!')
    '\n    Convert a column holding categorical data to a pandas Series.\n\n    Parameters\n    ----------\n    col : Column\n\n    Returns\n    -------\n    tuple\n        Tuple of pd.Series holding the data and the memory owner object\n        that keeps the memory alive.\n    '
    categorical = col.describe_categorical
    if not categorical['is_dictionary']:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')
    cat_column = categorical['categories']
    if hasattr(cat_column, '_col'):
        categories = np.array(cat_column._col)
    else:
        raise NotImplementedError("Interchanging categorical columns isn't supported yet, and our fallback of using the `col._col` attribute (a ndarray) failed.")
    buffers = col.get_buffers()
    (codes_buff, codes_dtype) = buffers['data']
    codes = buffer_to_ndarray(codes_buff, codes_dtype, offset=col.offset, length=col.size())
    if len(categories) > 0:
        values = categories[codes % len(categories)]
    else:
        values = codes
    cat = pd.Categorical(values, categories=categories, ordered=categorical['is_ordered'])
    data = pd.Series(cat)
    data = set_nulls(data, col, buffers['validity'])
    return (data, buffers)

def string_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a column holding string data to a NumPy array.\n\n    Parameters\n    ----------\n    col : Column\n\n    Returns\n    -------\n    tuple\n        Tuple of np.ndarray holding the data and the memory owner object\n        that keeps the memory alive.\n    '
    (null_kind, sentinel_val) = col.describe_null
    if null_kind not in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        raise NotImplementedError(f'{null_kind} null kind is not yet supported for string columns.')
    buffers = col.get_buffers()
    assert buffers['offsets'], 'String buffers must contain offsets'
    (data_buff, _) = buffers['data']
    assert col.dtype[2] in (ArrowCTypes.STRING, ArrowCTypes.LARGE_STRING)
    data_dtype = (DtypeKind.UINT, 8, ArrowCTypes.UINT8, Endianness.NATIVE)
    data = buffer_to_ndarray(data_buff, data_dtype, offset=0, length=data_buff.bufsize)
    (offset_buff, offset_dtype) = buffers['offsets']
    offsets = buffer_to_ndarray(offset_buff, offset_dtype, offset=col.offset, length=col.size() + 1)
    null_pos = None
    if null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        assert buffers['validity'], 'Validity buffers cannot be empty for masks'
        (valid_buff, valid_dtype) = buffers['validity']
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, offset=col.offset, length=col.size())
        if sentinel_val == 0:
            null_pos = ~null_pos
    str_list: list[None | float | str] = [None] * col.size()
    for i in range(col.size()):
        if null_pos is not None and null_pos[i]:
            str_list[i] = np.nan
            continue
        units = data[offsets[i]:offsets[i + 1]]
        str_bytes = bytes(units)
        string = str_bytes.decode(encoding='utf-8')
        str_list[i] = string
    return (np.asarray(str_list, dtype='object'), buffers)

def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
    if False:
        print('Hello World!')
    'Parse datetime `format_str` to interpret the `data`.'
    timestamp_meta = re.match('ts([smun]):(.*)', format_str)
    if timestamp_meta:
        (unit, tz) = (timestamp_meta.group(1), timestamp_meta.group(2))
        if unit != 's':
            unit += 's'
        data = data.astype(f'datetime64[{unit}]')
        if tz != '':
            data = pd.Series(data).dt.tz_localize('UTC').dt.tz_convert(tz)
        return data
    date_meta = re.match('td([Dm])', format_str)
    if date_meta:
        unit = date_meta.group(1)
        if unit == 'D':
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype('datetime64[s]')
        elif unit == 'm':
            data = data.astype('datetime64[ms]')
        else:
            raise NotImplementedError(f'Date unit is not supported: {unit}')
        return data
    raise NotImplementedError(f'DateTime kind is not supported: {format_str}')

def datetime_column_to_ndarray(col: Column) -> tuple[np.ndarray | pd.Series, Any]:
    if False:
        print('Hello World!')
    '\n    Convert a column holding DateTime data to a NumPy array.\n\n    Parameters\n    ----------\n    col : Column\n\n    Returns\n    -------\n    tuple\n        Tuple of np.ndarray holding the data and the memory owner object\n        that keeps the memory alive.\n    '
    buffers = col.get_buffers()
    (_, col_bit_width, format_str, _) = col.dtype
    (dbuf, _) = buffers['data']
    data = buffer_to_ndarray(dbuf, (DtypeKind.INT, col_bit_width, getattr(ArrowCTypes, f'INT{col_bit_width}'), Endianness.NATIVE), offset=col.offset, length=col.size())
    data = parse_datetime_format_str(format_str, data)
    data = set_nulls(data, col, buffers['validity'])
    return (data, buffers)

def buffer_to_ndarray(buffer: Buffer, dtype: tuple[DtypeKind, int, str, str], *, length: int, offset: int=0) -> np.ndarray:
    if False:
        print('Hello World!')
    "\n    Build a NumPy array from the passed buffer.\n\n    Parameters\n    ----------\n    buffer : Buffer\n        Buffer to build a NumPy array from.\n    dtype : tuple\n        Data type of the buffer conforming protocol dtypes format.\n    offset : int, default: 0\n        Number of elements to offset from the start of the buffer.\n    length : int, optional\n        If the buffer is a bit-mask, specifies a number of bits to read\n        from the buffer. Has no effect otherwise.\n\n    Returns\n    -------\n    np.ndarray\n\n    Notes\n    -----\n    The returned array doesn't own the memory. The caller of this function is\n    responsible for keeping the memory owner object alive as long as\n    the returned NumPy array is being used.\n    "
    (kind, bit_width, _, _) = dtype
    column_dtype = _NP_DTYPES.get(kind, {}).get(bit_width, None)
    if column_dtype is None:
        raise NotImplementedError(f'Conversion for {dtype} is not yet supported.')
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    if bit_width == 1:
        assert length is not None, '`length` must be specified for a bit-mask buffer.'
        pa = import_optional_dependency('pyarrow')
        arr = pa.BooleanArray.from_buffers(pa.bool_(), length, [None, pa.foreign_buffer(buffer.ptr, length)], offset=offset)
        return np.asarray(arr)
    else:
        data_pointer = ctypes.cast(buffer.ptr + offset * bit_width // 8, ctypes.POINTER(ctypes_type))
        if length > 0:
            return np.ctypeslib.as_array(data_pointer, shape=(length,))
        return np.array([], dtype=ctypes_type)

def set_nulls(data: np.ndarray | pd.Series, col: Column, validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None, allow_modify_inplace: bool=True):
    if False:
        i = 10
        return i + 15
    '\n    Set null values for the data according to the column null kind.\n\n    Parameters\n    ----------\n    data : np.ndarray or pd.Series\n        Data to set nulls in.\n    col : Column\n        Column object that describes the `data`.\n    validity : tuple(Buffer, dtype) or None\n        The return value of ``col.buffers()``. We do not access the ``col.buffers()``\n        here to not take the ownership of the memory of buffer objects.\n    allow_modify_inplace : bool, default: True\n        Whether to modify the `data` inplace when zero-copy is possible (True) or always\n        modify a copy of the `data` (False).\n\n    Returns\n    -------\n    np.ndarray or pd.Series\n        Data with the nulls being set.\n    '
    (null_kind, sentinel_val) = col.describe_null
    null_pos = None
    if null_kind == ColumnNullType.USE_SENTINEL:
        null_pos = pd.Series(data) == sentinel_val
    elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        assert validity, 'Expected to have a validity buffer for the mask'
        (valid_buff, valid_dtype) = validity
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, offset=col.offset, length=col.size())
        if sentinel_val == 0:
            null_pos = ~null_pos
    elif null_kind in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_NAN):
        pass
    else:
        raise NotImplementedError(f'Null kind {null_kind} is not yet supported.')
    if null_pos is not None and np.any(null_pos):
        if not allow_modify_inplace:
            data = data.copy()
        try:
            data[null_pos] = None
        except TypeError:
            data = data.astype(float)
            data[null_pos] = None
        except SettingWithCopyError:
            data = data.copy()
            data[null_pos] = None
    return data