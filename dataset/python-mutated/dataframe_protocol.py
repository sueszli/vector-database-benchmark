"""
Implementation of the dataframe exchange protocol.

Public API
----------

from_dataframe : construct a vaex.dataframe.DataFrame from an input data frame which
                 implements the exchange protocol

For a background and spec, see:
   * https://data-apis.org/blog/dataframe_protocol_rfc/
   * https://data-apis.org/dataframe-protocol/latest/index.html

Notes
-----
- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""
import enum
import collections.abc
import ctypes
from typing import Any, Optional, Tuple, Dict, Iterable, Sequence
import vaex
import pyarrow
import pyarrow as pa
import numpy as np
import pandas as pd
DataFrameObject = Any
ColumnObject = Any

def from_dataframe_to_vaex(df: DataFrameObject, allow_copy: bool=True) -> vaex.dataframe.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    Construct a vaex DataFrame from ``df`` if it supports ``__dataframe__``\n    '
    if isinstance(df, vaex.dataframe.DataFrame):
        return df
    if not hasattr(df, '__dataframe__'):
        raise ValueError('`df` does not support __dataframe__')
    return _from_dataframe_to_vaex(df.__dataframe__(allow_copy=allow_copy))

def _from_dataframe_to_vaex(df: DataFrameObject) -> vaex.dataframe.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    Note: we need to implement/test support for bit/byte masks, chunk handling, etc.\n    '
    dataframe = []
    _buffers = []
    for chunk in df.get_chunks():
        columns = dict()
        _k = _DtypeKind
        _buffers_chunks = []
        for name in chunk.column_names():
            if not isinstance(name, str):
                raise ValueError(f'Column {name} is not a string')
            if name in columns:
                raise ValueError(f'Column {name} is not unique')
            col = chunk.get_column_by_name(name)
            if col.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
                (columns[name], _buf) = convert_column_to_ndarray(col)
            elif col.dtype[0] == _k.CATEGORICAL:
                (columns[name], _buf) = convert_categorical_column(col)
            elif col.dtype[0] == _k.STRING:
                (columns[name], _buf) = convert_string_column(col)
            else:
                raise NotImplementedError(f'Data type {col.dtype[0]} not handled yet')
            _buffers_chunks.append(_buf)
        dataframe.append(vaex.from_dict(columns))
        _buffers.append(_buffers_chunks)
    if df.num_chunks() == 1:
        _buffers = _buffers[0]
    df_new = vaex.concat(dataframe)
    df_new._buffers = _buffers
    return df_new

class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21
    DATETIME = 22
    CATEGORICAL = 23

def convert_column_to_ndarray(col: ColumnObject) -> pa.Array:
    if False:
        return 10
    '\n    Convert an int, uint, float or bool column to an arrow array\n    '
    if col.offset != 0:
        raise NotImplementedError('column.offset > 0 not handled yet')
    if col.describe_null[0] not in (0, 1, 3, 4):
        raise NotImplementedError('Null values represented assentinel values not handled yet')
    (_buffer, _dtype) = col.get_buffers()['data']
    x = buffer_to_ndarray(_buffer, _dtype)
    if col.describe_null[0] in (3, 4) and col.null_count > 0:
        (mask_buffer, mask_dtype) = col._get_validity_buffer()
        mask = buffer_to_ndarray(mask_buffer, mask_dtype)
        x = pa.array(x, mask=mask)
    else:
        x = pa.array(x)
    return (x, _buffer)

def buffer_to_ndarray(_buffer, _dtype) -> np.ndarray:
    if False:
        return 10
    kind = _dtype[0]
    bitwidth = _dtype[1]
    _k = _DtypeKind
    if _dtype[0] not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
        raise RuntimeError('Not a boolean, integer or floating-point dtype')
    _ints = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    _uints = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _floats = {32: np.float32, 64: np.float64}
    _np_dtypes = {0: _ints, 1: _uints, 2: _floats, 20: {8: bool}}
    column_dtype = _np_dtypes[kind][bitwidth]
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(_buffer.ptr, ctypes.POINTER(ctypes_type))
    x = np.ctypeslib.as_array(data_pointer, shape=(_buffer.bufsize // (bitwidth // 8),))
    return x

def convert_categorical_column(col: ColumnObject) -> Tuple[pa.DictionaryArray, Any]:
    if False:
        while True:
            i = 10
    '\n    Convert a categorical column to an arrow dictionary\n    '
    catinfo = col.describe_categorical
    if not catinfo['is_dictionary']:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')
    assert catinfo['categories'] is not None
    if not col.describe_null[0] in (0, 2, 3, 4):
        raise NotImplementedError('Only categorical columns with sentinel value and masks supported at the moment')
    (codes_buffer, codes_dtype) = col.get_buffers()['data']
    codes = buffer_to_ndarray(codes_buffer, codes_dtype)
    if col.describe_null[0] == 2:
        sentinel = [col.describe_null[1]] * col.size
        mask = codes == sentinel
        indices = pa.array(codes, mask=mask)
    elif col.describe_null[0] in (3, 4) and col.null_count > 0:
        (mask_buffer, mask_dtype) = col._get_validity_buffer()
        mask = buffer_to_ndarray(mask_buffer, mask_dtype)
        indices = pa.array(codes, mask=mask)
    else:
        indices = pa.array(codes)
    (labels_buffer, labels_dtype) = catinfo['categories'].get_buffers()['data']
    if labels_dtype[0] == _DtypeKind.STRING:
        (labels, _) = convert_string_column(catinfo['categories'])
    else:
        labels = buffer_to_ndarray(labels_buffer, labels_dtype)
    values = pa.DictionaryArray.from_arrays(indices, labels)
    return (values, codes_buffer)

def convert_string_column(col: ColumnObject) -> Tuple[pa.Array, list]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a string column to a Arrow array.\n    '
    if col.null_count > 0:
        if col.describe_null != (3, 0):
            raise TypeError('Only support arrow style mask data')
    buffers = col.get_buffers()
    (dbuffer, bdtype) = buffers['data']
    (obuffer, odtype) = buffers['offsets']
    (mbuffer, mdtype) = buffers['validity']
    dt = (_DtypeKind.UINT, 8, None, None)
    dbuf = buffer_to_ndarray(dbuffer, dt)
    obuf = buffer_to_ndarray(obuffer, odtype)
    mbuf = buffer_to_ndarray(mbuffer, mdtype)
    if obuffer._x.dtype == 'int64':
        arrow_type = pa.large_utf8()
    elif obuffer._x.dtype == 'int32':
        arrow_type = pa.utf8()
    else:
        raise TypeError(f'oops')
    length = obuf.size - 1
    buffers = [None, pa.py_buffer(obuf), pa.py_buffer(dbuf)]
    arrow_array = pa.Array.from_buffers(arrow_type, length, buffers)
    if col.null_count > 0:
        arrow_array = pa.array(arrow_array.tolist(), mask=mbuf)
    return (arrow_array, buffers)

class _VaexBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle only regular columns (= numpy arrays) for now.\n        '
        if not x.strides == (x.dtype.itemsize,):
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError('Exports cannot be zero-copy in the case of a non-contiguous buffer')
        self._x = x

    @property
    def bufsize(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Buffer size in bytes\n        '
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pointer to start of the buffer as an integer\n        '
        return self._x.__array_interface__['data'][0]

    def __dlpack__(self):
        if False:
            while True:
                i = 10
        '\n        DLPack not implemented in Vaex, so leave it out here\n        '
        raise NotImplementedError('__dlpack__')

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Device type and device ID for where the data in the buffer resides.\n        '

        class Device(enum.IntEnum):
            CPU = 1
        return (Device.CPU, None)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'VaexBuffer(' + str({'bufsize': self.bufsize, 'ptr': self.ptr, 'device': self.__dlpack_device__()[0].name}) + ')'

class _VaexColumn:
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.

    """

    def __init__(self, column: vaex.expression.Expression, allow_copy: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Note: assuming column is an expression.\n        The values of an expression can be NumPy or Arrow.\n        '
        if not isinstance(column, vaex.expression.Expression):
            raise NotImplementedError('Columns of type {} not handled yet'.format(type(column)))
        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Size of the column, in elements.\n\n        Corresponds to DataFrame.num_rows() if column is a single chunk;\n        equal to size of this current chunk otherwise.\n\n        Is a method rather than a property because it may cause a (potentially\n        expensive) computation for some dataframe implementations.\n        '
        return int(len(self._col.df))

    @property
    def offset(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Offset of first element. Always zero.\n        '
        return 0

    @property
    def dtype(self) -> Tuple[enum.IntEnum, int, str, str]:
        if False:
            return 10
        "\n        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``\n\n        Kind :\n\n            - INT = 0\n            - UINT = 1\n            - FLOAT = 2\n            - BOOL = 20\n            - STRING = 21   # UTF-8\n            - DATETIME = 22\n            - CATEGORICAL = 23\n\n        Bit-width : the number of bits as an integer\n        Format string : data type description format string in Apache Arrow C\n                        Data Interface format.\n        Endianness : current only native endianness (``=``) is supported\n\n        Notes:\n\n            - Kind specifiers are aligned with DLPack where possible (hence the\n              jump to 20, leave enough room for future extension)\n            - Masks must be specified as boolean with either bit width 1 (for bit\n              masks) or 8 (for byte masks).\n            - Dtype width in bits was preferred over bytes\n            - Endianness isn't too useful, but included now in case in the future\n              we need to support non-native endianness\n            - Went with Apache Arrow format strings over NumPy format strings\n              because they're more complete from a dataframe perspective\n            - Format strings are mostly useful for datetime specification, and\n              for categoricals.\n            - For categoricals, the format string describes the type of the\n              categorical in the data buffer. In case of a separate encoding of\n              the categorical (e.g. an integer to string mapping), this can\n              be derived from ``self.describe_categorical``.\n            - Data types not included: complex, Arrow-style null, binary, decimal,\n              and nested (list, struct, map, union) dtypes.\n        "
        dtype = self._col.dtype
        if self._col.df.is_category(self._col):
            return (_DtypeKind.CATEGORICAL, 64, 'u', '=')
        return self._dtype_from_vaexdtype(dtype)

    def _dtype_from_vaexdtype(self, dtype) -> Tuple[enum.IntEnum, int, str, str]:
        if False:
            print('Hello World!')
        '\n        See `self.dtype` for details\n        '
        _k = _DtypeKind
        _np_kinds = {'i': _k.INT, 'u': _k.UINT, 'f': _k.FLOAT, 'b': _k.BOOL, 'U': _k.STRING, 'M': _k.DATETIME, 'm': _k.DATETIME}
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            if not isinstance(self._col.values, np.ndarray) and isinstance(self._col.values.type, pa.DictionaryType):
                kind = 23
            elif dtype == 'string':
                kind = 21
            else:
                raise ValueError(f'Data type {dtype} not supported by exchange protocol')
        if kind not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.CATEGORICAL, _k.STRING):
            raise NotImplementedError(f'Data type {dtype} not handled yet')
        bitwidth = dtype.numpy.itemsize * 8
        if not isinstance(self._col.values, np.ndarray) and isinstance(self._col.values.type, pa.DictionaryType):
            format_str = self._col.index_values().dtype.numpy.str
        else:
            format_str = dtype.numpy.str
        endianness = dtype.byteorder if not kind == _k.CATEGORICAL else '='
        return (kind, bitwidth, format_str, endianness)

    @property
    def describe_categorical(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        If the dtype is categorical, there are two options:\n        - There are only values in the data buffer.\n        - There is a separate non-categorical Column encoding categorical values.\n\n        Raises TypeError if the dtype is not categorical\n\n        Returns the dictionary with description on how to interpret the data buffer:\n            - "is_ordered" : bool, whether the ordering of dictionary indices is\n                             semantically meaningful.\n            - "is_dictionary" : bool, whether a mapping of\n                                categorical values to other objects exists\n            - "categories" : Column representing the (implicit) mapping of indices to\n                             category values (e.g. an array of cat1, cat2, ...).\n                             None if not a dictionary-style categorical.\n\n        TBD: are there any other in-memory representations that are needed?\n        '
        if not self.dtype[0] == _DtypeKind.CATEGORICAL:
            raise TypeError('describe_categorical only works on a column with categorical dtype!')
        df = vaex.from_dict({'labels': self._col.df.category_labels(self._col)})
        labels = df['labels']
        categories = _VaexColumn(labels)
        return {'is_ordered': False, 'is_dictionary': True, 'categories': categories}

    @property
    def describe_null(self) -> Tuple[int, Any]:
        if False:
            return 10
        '\n        Return the missing value (or "null") representation the column dtype\n        uses, as a tuple ``(kind, value)``.\n\n        Kind:\n\n            - 0 : non-nullable\n            - 1 : NaN/NaT\n            - 2 : sentinel value\n            - 3 : bit mask\n            - 4 : byte mask\n\n        Value : if kind is "sentinel value", the actual value.  If kind is a bit\n        mask or a byte mask, the value (0 or 1) indicating a missing value. None\n        otherwise.\n        '
        _k = _DtypeKind
        kind = self.dtype[0]
        value = None
        if kind in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.STRING):
            if self._col.dtype.is_arrow:
                null = 3
                value = 0
            elif self._col.is_masked:
                null = 4
                value = 1
            else:
                null = 0
                value = None
        elif kind == _k.CATEGORICAL:
            if self._col.dtype.is_arrow:
                null = 3
                value = 0
            else:
                null = 0
                value = None
        else:
            raise NotImplementedError(f'Data type {self.dtype} not yet supported')
        return (null, value)

    @property
    def null_count(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Number of null elements. Should always be known.\n        '
        return self._col.countna()

    @property
    def metadata(self) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Store specific metadata of the column.\n        '
        return {}

    def num_chunks(self) -> int:
        if False:
            print('Hello World!')
        '\n        Return the number of chunks the column consists of.\n        '
        if isinstance(self._col.values, pa.ChunkedArray):
            return self._col.values.num_chunks
        else:
            return 1

    def get_chunks(self, n_chunks: Optional[int]=None) -> Iterable['_VaexColumn']:
        if False:
            i = 10
            return i + 15
        '\n        Return an iterator yielding the chunks.\n\n        See `DataFrame.get_chunks` for details on ``n_chunks``.\n        '
        if n_chunks == None:
            size = self.size()
            n_chunks = self.num_chunks()
            i = self._col.df.evaluate_iterator(self._col, chunk_size=size // n_chunks)
            iterator = []
            for (i1, i2, chunk) in i:
                iterator.append(_VaexColumn(self._col[i1:i2]))
            return iterator
        elif self.num_chunks == 1:
            size = self.size()
            i = self._col.df.evaluate_iterator(self._col, chunk_size=size // n_chunks)
            iterator = []
            for (i1, i2, chunk) in i:
                iterator.append(_VaexColumn(self._col[i1:i2]))
            return iterator
        else:
            raise ValueError(f'Column {self._col.expression} is already chunked.')

    def get_buffers(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a dictionary containing the underlying buffers.\n\n        The returned dictionary has the following contents:\n\n            - "data": a two-element tuple whose first element is a buffer\n                      containing the data and whose second element is the data\n                      buffer\'s associated dtype.\n            - "validity": a two-element tuple whose first element is a buffer\n                          containing mask values indicating missing data and\n                          whose second element is the mask value buffer\'s\n                          associated dtype. None if the null representation is\n                          not a bit or byte mask.\n            - "offsets": a two-element tuple whose first element is a buffer\n                         containing the offset values for variable-size binary\n                         data (e.g., variable-length strings) and whose second\n                         element is the offsets buffer\'s associated dtype. None\n                         if the data buffer does not have an associated offsets\n                         buffer.\n        '
        buffers = {}
        buffers['data'] = self._get_data_buffer()
        try:
            buffers['validity'] = self._get_validity_buffer()
        except:
            buffers['validity'] = None
        try:
            buffers['offsets'] = self._get_offsets_buffer()
        except:
            buffers['offsets'] = None
        return buffers

    def _get_data_buffer(self) -> Tuple[_VaexBuffer, Any]:
        if False:
            print('Hello World!')
        "\n        Return the buffer containing the data and the buffer's associated dtype.\n        "
        _k = _DtypeKind
        if self.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            if self.dtype[0] == _k.BOOL and isinstance(self._col.values, (pa.Array, pa.ChunkedArray)):
                buffer = _VaexBuffer(np.array(self._col.tolist(), dtype=bool))
            else:
                buffer = _VaexBuffer(self._col.to_numpy())
            dtype = self.dtype
        elif self.dtype[0] == _k.CATEGORICAL:
            if isinstance(self._col.values, pa.DictionaryArray):
                buffer = _VaexBuffer(self._col.index_values().to_numpy())
                dtype = self._dtype_from_vaexdtype(self._col.index_values().dtype)
            else:
                codes = self._col.values
                indices = self._col
                offset = self._col.df.category_offset(self._col)
                if offset:
                    indices -= offset
                buffer = _VaexBuffer(indices.to_numpy())
                dtype = self._dtype_from_vaexdtype(self._col.dtype)
        elif self.dtype[0] == _k.STRING:
            (bitmap_buffer, offsets, string_bytes) = self._col.evaluate().buffers()
            if string_bytes is None:
                string_bytes = np.array([], dtype='uint8')
            else:
                string_bytes = np.frombuffer(string_bytes, 'uint8', len(string_bytes))
            buffer = _VaexBuffer(string_bytes)
            dtype = (_k.STRING, 8, 'u', '=')
        else:
            raise NotImplementedError(f'Data type {self._col.dtype} not handled yet')
        return (buffer, dtype)

    def _get_validity_buffer(self) -> Tuple[_VaexBuffer, Any]:
        if False:
            i = 10
            return i + 15
        "\n        Return the buffer containing the mask values indicating missing data and\n        the buffer's associated dtype.\n\n        Raises RuntimeError if null representation is not a bit or byte mask.\n        "
        (null, invalid) = self.describe_null
        _k = _DtypeKind
        if null == 3 or null == 4:
            mask = self._col.ismissing()
            data = np.array(mask.tolist()) if null == 3 else mask.to_numpy()
            buffer = _VaexBuffer(data)
            dtype = self._dtype_from_vaexdtype(mask.dtype)
            return (buffer, dtype)
        elif null == 0:
            msg = 'This column is non-nullable so does not have a mask'
        elif null == 1:
            msg = 'This column uses NaN as null so does not have a separate mask'
        else:
            raise NotImplementedError('See self.describe_null')
        raise RuntimeError(msg)

    def _get_offsets_buffer(self) -> Tuple[_VaexBuffer, Any]:
        if False:
            return 10
        "\n        Return the buffer containing the offset values for variable-size binary\n        data (e.g., variable-length strings) and the buffer's associated dtype.\n\n        Raises RuntimeError if the data buffer does not have an associated\n        offsets buffer.\n        "
        _k = _DtypeKind
        if self.dtype[0] == _k.STRING:
            (bitmap_buffer, offsets, string_bytes) = self._col.evaluate().buffers()
            if self._col.evaluate().type == pyarrow.string():
                offsets = np.frombuffer(offsets, np.int32, len(offsets) // 4)
                dtype = (_k.INT, 32, 'i', '=')
            else:
                offsets = np.frombuffer(offsets, np.int64, len(offsets) // 8)
                dtype = (_k.INT, 64, 'l', '=')
            buffer = _VaexBuffer(offsets)
        else:
            raise RuntimeError('This column has a fixed-length dtype so does not have an offsets buffer')
        return (buffer, dtype)

class _VaexDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    Instances of this (private) class are returned from
    ``vaex.dataframe.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """

    def __init__(self, df: vaex.dataframe.DataFrame, nan_as_null: bool=False, allow_copy: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructor - an instance of this (private) class is returned from\n        `vaex.dataframe.DataFrame.__dataframe__`.\n        '
        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(self, nan_as_null: bool=False, allow_copy: bool=True) -> '_VaexDataFrame':
        if False:
            return 10
        return _VaexDataFrame(self._df, nan_as_null=nan_as_null, allow_copy=allow_copy)

    @property
    def metadata(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return {}

    def num_columns(self) -> int:
        if False:
            return 10
        return len(self._df.columns)

    def num_rows(self) -> int:
        if False:
            return 10
        return len(self._df)

    def num_chunks(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.get_column(0)._col.values, pa.ChunkedArray):
            return self.get_column(0)._col.values.num_chunks
        else:
            return 1

    def column_names(self) -> Iterable[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._df.get_column_names()

    def get_column(self, i: int) -> _VaexColumn:
        if False:
            i = 10
            return i + 15
        return _VaexColumn(self._df[:, i], allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _VaexColumn:
        if False:
            print('Hello World!')
        return _VaexColumn(self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_VaexColumn]:
        if False:
            i = 10
            return i + 15
        return [_VaexColumn(self._df[name], allow_copy=self._allow_copy) for name in self._df.columns]

    def select_columns(self, indices: Sequence[int]) -> '_VaexDataFrame':
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(indices, collections.abc.Sequence):
            raise ValueError('`indices` is not a sequence')
        names = []
        for i in indices:
            names.append(self._df[:, i].expression)
        return self.select_columns_by_name(names)

    def select_columns_by_name(self, names: Sequence[str]) -> '_VaexDataFrame':
        if False:
            while True:
                i = 10
        if not isinstance(names, collections.abc.Sequence):
            raise ValueError('`names` is not a sequence')
        return _VaexDataFrame(self._df[names], allow_copy=self._allow_copy)

    def get_chunks(self, n_chunks: Optional[int]=None) -> Iterable['_VaexDataFrame']:
        if False:
            while True:
                i = 10
        '\n        Return an iterator yielding the chunks.\n        TODO: details on ``n_chunks``\n        '
        n_chunks = n_chunks if n_chunks is not None else self.num_chunks()
        size = self.num_rows()
        chunk_size = (size + n_chunks - 1) // n_chunks
        column_names = self.column_names()
        i = self._df._future().evaluate_iterator(column_names, chunk_size=chunk_size)
        for (i1, i2, chunk) in i:
            yield _VaexDataFrame(vaex.from_items(*zip(column_names, chunk)))