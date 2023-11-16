import enum
from collections import abc
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, cast
import cupy as cp
import numpy as np
from numba.cuda import as_cuda_array
import rmm
import cudf
from cudf.core.buffer import Buffer, as_buffer
from cudf.core.column import as_column, build_categorical_column, build_column

class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21
    DATETIME = 22
    CATEGORICAL = 23

class _Device(enum.IntEnum):
    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10

class _MaskKind(enum.IntEnum):
    NON_NULLABLE = 0
    NAN = 1
    SENTINEL = 2
    BITMASK = 3
    BYTEMASK = 4
_SUPPORTED_KINDS = {_DtypeKind.INT, _DtypeKind.UINT, _DtypeKind.FLOAT, _DtypeKind.CATEGORICAL, _DtypeKind.BOOL, _DtypeKind.STRING}
ProtoDtype = Tuple[_DtypeKind, int, str, str]

class _CuDFBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, buf: Buffer, dtype: np.dtype, allow_copy: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Use Buffer object.\n        '
        self._buf = buf
        self._dtype = dtype
        self._allow_copy = allow_copy

    @property
    def bufsize(self) -> int:
        if False:
            print('Hello World!')
        '\n        The Buffer size in bytes.\n        '
        return self._buf.size

    @property
    def ptr(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Pointer to start of the buffer as an integer.\n        '
        return self._buf.get_ptr(mode='write')

    def __dlpack__(self):
        if False:
            return 10
        try:
            cuda_array = as_cuda_array(self._buf).view(self._dtype)
            return cp.asarray(cuda_array).toDlpack()
        except ValueError:
            raise TypeError(f'dtype {self._dtype} unsupported by `dlpack`')

    def __dlpack_device__(self) -> Tuple[_Device, int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        _Device type and _Device ID for where the data in the buffer resides.\n        '
        return (_Device.CUDA, cp.asarray(self._buf).device.id)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(' + str({'bufsize': self.bufsize, 'ptr': self.ptr, 'device': self.__dlpack_device__()[0].name})
        +')'

class _CuDFColumn:
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

    def __init__(self, column: cudf.core.column.ColumnBase, nan_as_null: bool=True, allow_copy: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Note: doesn't deal with extension arrays yet, just assume a regular\n        Series/ndarray for now.\n        "
        if not isinstance(column, cudf.core.column.ColumnBase):
            raise TypeError(f'column must be a subtype of df.core.column.ColumnBase,got {type(column)}')
        self._col = column
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def size(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Size of the column, in elements.\n        '
        return self._col.size

    @property
    def offset(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Offset of first element. Always zero.\n        '
        return 0

    @property
    def dtype(self) -> ProtoDtype:
        if False:
            print('Hello World!')
        "\n        Dtype description as a tuple\n        ``(kind, bit-width, format string, endianness)``\n\n        Kind :\n\n            - INT = 0\n            - UINT = 1\n            - FLOAT = 2\n            - BOOL = 20\n            - STRING = 21   # UTF-8\n            - DATETIME = 22\n            - CATEGORICAL = 23\n\n        Bit-width : the number of bits as an integer\n        Format string : data type description format string in Apache Arrow C\n                        Data Interface format.\n        Endianness : current only native endianness (``=``) is supported\n\n        Notes\n        -----\n        - Kind specifiers are aligned with DLPack where possible\n         (hence the jump to 20, leave enough room for future extension)\n        - Masks must be specified as boolean with either bit width 1\n         (for bit masks) or 8 (for byte masks).\n        - Dtype width in bits was preferred over bytes\n        - Endianness isn't too useful, but included now in case\n          in the future we need to support non-native endianness\n        - Went with Apache Arrow format strings over NumPy format strings\n          because they're more complete from a dataframe perspective\n        - Format strings are mostly useful for datetime specification,\n          and for categoricals.\n        - For categoricals, the format string describes the type of the\n          categorical in the data buffer. In case of a separate encoding\n          of the categorical (e.g. an integer to string mapping),\n          this can be derived from ``self.describe_categorical``.\n        - Data types not included: complex, Arrow-style null,\n          binary, decimal, and nested (list, struct, map, union) dtypes.\n        "
        dtype = self._col.dtype
        if not isinstance(dtype, cudf.CategoricalDtype) and dtype.kind == 'O':
            return (_DtypeKind.STRING, 8, 'u', '=')
        return self._dtype_from_cudfdtype(dtype)

    def _dtype_from_cudfdtype(self, dtype) -> ProtoDtype:
        if False:
            while True:
                i = 10
        '\n        See `self.dtype` for details.\n        '
        _np_kinds = {'i': _DtypeKind.INT, 'u': _DtypeKind.UINT, 'f': _DtypeKind.FLOAT, 'b': _DtypeKind.BOOL, 'U': _DtypeKind.STRING, 'M': _DtypeKind.DATETIME, 'm': _DtypeKind.DATETIME}
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            if isinstance(dtype, cudf.CategoricalDtype):
                kind = _DtypeKind.CATEGORICAL
                codes = cast(cudf.core.column.CategoricalColumn, self._col).codes
                dtype = codes.dtype
            else:
                raise ValueError(f'Data type {dtype} not supported by exchange protocol')
        if kind not in _SUPPORTED_KINDS:
            raise NotImplementedError(f'Data type {dtype} not handled yet')
        bitwidth = dtype.itemsize * 8
        format_str = dtype.str
        endianness = dtype.byteorder if kind != _DtypeKind.CATEGORICAL else '='
        return (kind, bitwidth, format_str, endianness)

    @property
    def describe_categorical(self) -> Tuple[bool, bool, Dict[int, Any]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        If the dtype is categorical, there are two options:\n\n        - There are only values in the data buffer.\n        - There is a separate dictionary-style encoding for categorical values.\n\n        Raises TypeError if the dtype is not categorical\n\n        Content of returned dict:\n\n            - "is_ordered" : bool, whether the ordering of dictionary\n                             indices is semantically meaningful.\n            - "is_dictionary" : bool, whether a dictionary-style mapping of\n                                categorical values to other objects exists\n            - "mapping" : dict, Python-level only (e.g. ``{int: str}``).\n                          None if not a dictionary-style categorical.\n        '
        if not self.dtype[0] == _DtypeKind.CATEGORICAL:
            raise TypeError('`describe_categorical only works on a column with categorical dtype!')
        categ_col = cast(cudf.core.column.CategoricalColumn, self._col)
        ordered = bool(categ_col.dtype.ordered)
        is_dictionary = True
        categories = categ_col.categories
        mapping = {ix: val for (ix, val) in enumerate(categories.values_host)}
        return (ordered, is_dictionary, mapping)

    @property
    def describe_null(self) -> Tuple[int, Any]:
        if False:
            return 10
        '\n        Return the missing value (or "null") representation the column dtype\n        uses, as a tuple ``(kind, value)``.\n\n        Kind:\n\n            - 0 : non-nullable\n            - 1 : NaN/NaT\n            - 2 : sentinel value\n            - 3 : bit mask\n            - 4 : byte mask\n\n        Value : if kind is "sentinel value", the actual value.\n        If kind is a bit mask or a byte mask, the value (0 or 1)\n        indicating a missing value.\n        None otherwise.\n        '
        kind = self.dtype[0]
        if self.null_count == 0:
            return (_MaskKind.NON_NULLABLE, None)
        elif kind in _SUPPORTED_KINDS:
            return (_MaskKind.BITMASK, 0)
        else:
            raise NotImplementedError(f'Data type {self.dtype} not yet supported')

    @property
    def null_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Number of null elements. Should always be known.\n        '
        return self._col.null_count

    @property
    def metadata(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Store specific metadata of the column.\n        '
        return {}

    def num_chunks(self) -> int:
        if False:
            print('Hello World!')
        '\n        Return the number of chunks the column consists of.\n        '
        return 1

    def get_chunks(self, n_chunks: Optional[int]=None) -> Iterable['_CuDFColumn']:
        if False:
            return 10
        '\n        Return an iterable yielding the chunks.\n\n        See `DataFrame.get_chunks` for details on ``n_chunks``.\n        '
        return (self,)

    def get_buffers(self) -> Mapping[str, Optional[Tuple[_CuDFBuffer, ProtoDtype]]]:
        if False:
            print('Hello World!')
        '\n        Return a dictionary containing the underlying buffers.\n\n        The returned dictionary has the following contents:\n\n            - "data": a two-element tuple whose first element is a buffer\n                      containing the data and whose second element is the data\n                      buffer\'s associated dtype.\n            - "validity": a two-element tuple whose first element is a buffer\n                          containing mask values indicating missing data and\n                          whose second element is the mask value buffer\'s\n                          associated dtype. None if the null representation is\n                          not a bit or byte mask.\n            - "offsets": a two-element tuple whose first element is a buffer\n                         containing the offset values for variable-size binary\n                         data (e.g., variable-length strings) and whose second\n                         element is the offsets buffer\'s associated dtype. None\n                         if the data buffer does not have an associated offsets\n                         buffer.\n        '
        buffers = {}
        try:
            buffers['validity'] = self._get_validity_buffer()
        except RuntimeError:
            buffers['validity'] = None
        try:
            buffers['offsets'] = self._get_offsets_buffer()
        except RuntimeError:
            buffers['offsets'] = None
        buffers['data'] = self._get_data_buffer()
        return buffers

    def _get_validity_buffer(self) -> Optional[Tuple[_CuDFBuffer, ProtoDtype]]:
        if False:
            print('Hello World!')
        "\n        Return the buffer containing the mask values\n        indicating missing data and the buffer's associated dtype.\n\n        Raises RuntimeError if null representation is not a bit or byte mask.\n        "
        (null, invalid) = self.describe_null
        if null == _MaskKind.BITMASK:
            assert self._col.mask is not None
            buffer = _CuDFBuffer(self._col.mask, cp.uint8, allow_copy=self._allow_copy)
            dtype = (_DtypeKind.UINT, 8, 'C', '=')
            return (buffer, dtype)
        elif null == _MaskKind.NAN:
            raise RuntimeError('This column uses NaN as null so does not have a separate mask')
        elif null == _MaskKind.NON_NULLABLE:
            raise RuntimeError('This column is non-nullable so does not have a mask')
        else:
            raise NotImplementedError(f'See {self.__class__.__name__}.describe_null method.')

    def _get_offsets_buffer(self) -> Optional[Tuple[_CuDFBuffer, ProtoDtype]]:
        if False:
            return 10
        "\n        Return the buffer containing the offset values for\n        variable-size binary data (e.g., variable-length strings)\n        and the buffer's associated dtype.\n\n        Raises RuntimeError if the data buffer does not have an associated\n        offsets buffer.\n        "
        if self.dtype[0] == _DtypeKind.STRING:
            offsets = self._col.children[0]
            assert offsets is not None and offsets.data is not None, ' '
            'offsets(.data) should not be None for string column'
            buffer = _CuDFBuffer(offsets.data, offsets.dtype, allow_copy=self._allow_copy)
            dtype = self._dtype_from_cudfdtype(offsets.dtype)
        else:
            raise RuntimeError('This column has a fixed-length dtype so does not have an offsets buffer')
        return (buffer, dtype)

    def _get_data_buffer(self) -> Tuple[_CuDFBuffer, ProtoDtype]:
        if False:
            while True:
                i = 10
        "\n        Return the buffer containing the data and\n               the buffer's associated dtype.\n        "
        if self.dtype[0] in (_DtypeKind.INT, _DtypeKind.UINT, _DtypeKind.FLOAT, _DtypeKind.BOOL):
            col_data = self._col
            dtype = self.dtype
        elif self.dtype[0] == _DtypeKind.CATEGORICAL:
            col_data = cast(cudf.core.column.CategoricalColumn, self._col).codes
            dtype = self._dtype_from_cudfdtype(col_data.dtype)
        elif self.dtype[0] == _DtypeKind.STRING:
            col_data = self._col.children[1]
            dtype = self._dtype_from_cudfdtype(col_data.dtype)
        else:
            raise NotImplementedError(f'Data type {self._col.dtype} not handled yet')
        assert col_data is not None and col_data.data is not None, ' '
        f'col_data(.data) should not be None when dtype = {dtype}'
        buffer = _CuDFBuffer(col_data.data, col_data.dtype, allow_copy=self._allow_copy)
        return (buffer, dtype)

class _CuDFDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    Instances of this (private) class are returned from
    ``cudf.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """

    def __init__(self, df: 'cudf.core.dataframe.DataFrame', nan_as_null: bool=True, allow_copy: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Constructor - an instance of this (private) class is returned from\n        `cudf.DataFrame.__dataframe__`.\n        '
        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(self, nan_as_null: bool=False, allow_copy: bool=True) -> '_CuDFDataFrame':
        if False:
            for i in range(10):
                print('nop')
        '\n        See the docstring of the `cudf.DataFrame.__dataframe__` for details\n        '
        return _CuDFDataFrame(self._df, nan_as_null=nan_as_null, allow_copy=allow_copy)

    @property
    def metadata(self):
        if False:
            print('Hello World!')
        return {'cudf.index': self._df.index}

    def num_columns(self) -> int:
        if False:
            return 10
        return len(self._df._column_names)

    def num_rows(self) -> int:
        if False:
            while True:
                i = 10
        return len(self._df)

    def num_chunks(self) -> int:
        if False:
            while True:
                i = 10
        return 1

    def column_names(self) -> Iterable[str]:
        if False:
            return 10
        return self._df._column_names

    def get_column(self, i: int) -> _CuDFColumn:
        if False:
            return 10
        return _CuDFColumn(as_column(self._df.iloc[:, i]), allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _CuDFColumn:
        if False:
            while True:
                i = 10
        return _CuDFColumn(as_column(self._df[name]), allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_CuDFColumn]:
        if False:
            return 10
        return [_CuDFColumn(as_column(self._df[name]), allow_copy=self._allow_copy) for name in self._df.columns]

    def select_columns(self, indices: Sequence[int]) -> '_CuDFDataFrame':
        if False:
            print('Hello World!')
        if not isinstance(indices, abc.Sequence):
            raise ValueError('`indices` is not a sequence')
        return _CuDFDataFrame(self._df.iloc[:, indices])

    def select_columns_by_name(self, names: Sequence[str]) -> '_CuDFDataFrame':
        if False:
            print('Hello World!')
        if not isinstance(names, abc.Sequence):
            raise ValueError('`names` is not a sequence')
        return _CuDFDataFrame(self._df.loc[:, names], self._nan_as_null, self._allow_copy)

    def get_chunks(self, n_chunks: Optional[int]=None) -> Iterable['_CuDFDataFrame']:
        if False:
            print('Hello World!')
        '\n        Return an iterator yielding the chunks.\n        '
        return (self,)

def __dataframe__(self, nan_as_null: bool=False, allow_copy: bool=True) -> _CuDFDataFrame:
    if False:
        print('Hello World!')
    '\n    The public method to attach to cudf.DataFrame.\n\n    ``nan_as_null`` is a keyword intended for the consumer to tell the\n    producer to overwrite null values in the data with ``NaN`` (or ``NaT``).\n    This currently has no effect; once support for nullable extension\n    dtypes is added, this value should be propagated to columns.\n\n    ``allow_copy`` is a keyword that defines whether or not the library is\n    allowed to make a copy of the data. For example, copying data would be\n    necessary if a library supports strided buffers, given that this protocol\n    specifies contiguous buffers.\n    '
    return _CuDFDataFrame(self, nan_as_null=nan_as_null, allow_copy=allow_copy)
"\nImplementation of the dataframe exchange protocol.\n\nPublic API\n----------\n\nfrom_dataframe : construct a cudf.DataFrame from an input data frame which\n                 implements the exchange protocol\n\nNotes\n-----\n\n- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and\n  unsafe to do in pure Python. It's more general but definitely less friendly\n  than having ``to_arrow`` and ``to_numpy`` methods. So for the buffers which\n  lack ``__dlpack__`` (e.g., because the column dtype isn't supported by\n  DLPack), this is worth looking at again.\n\n"
DataFrameObject = Any
ColumnObject = Any
_INTS = {8: cp.int8, 16: cp.int16, 32: cp.int32, 64: cp.int64}
_UINTS = {8: cp.uint8, 16: cp.uint16, 32: cp.uint32, 64: cp.uint64}
_FLOATS = {32: cp.float32, 64: cp.float64}
_CP_DTYPES = {0: _INTS, 1: _UINTS, 2: _FLOATS, 20: {8: bool}, 21: {8: cp.uint8}}

def from_dataframe(df: DataFrameObject, allow_copy: bool=False) -> _CuDFDataFrame:
    if False:
        print('Hello World!')
    "\n    Construct a ``DataFrame`` from ``df`` if it supports the\n    dataframe interchange protocol (``__dataframe__``).\n\n    Parameters\n    ----------\n    df : DataFrameObject\n        Object supporting dataframe interchange protocol\n    allow_copy : bool\n        If ``True``, allow copying of the data. If ``False``, a\n        ``TypeError`` is raised if data copying is required to\n        construct the ``DataFrame`` (e.g., if ``df`` lives in CPU\n        memory).\n\n    Returns\n    -------\n    DataFrame\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> pdf = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})\n    >>> df = cudf.from_dataframe(pdf, allow_copy=True)\n    >>> type(df)\n    cudf.core.dataframe.DataFrame\n    >>> df\n       a  b\n    0  1  x\n    1  2  y\n    2  3  z\n\n    Notes\n    -----\n    See https://data-apis.org/dataframe-protocol/latest/index.html\n    for the dataframe interchange protocol spec and API\n    "
    if isinstance(df, cudf.DataFrame):
        return df
    if not hasattr(df, '__dataframe__'):
        raise ValueError('`df` does not support __dataframe__')
    df = df.__dataframe__(allow_copy=allow_copy)
    if df.num_chunks() > 1:
        raise NotImplementedError('More than one chunk not handled yet')
    columns = dict()
    _buffers = []
    for name in df.column_names():
        col = df.get_column_by_name(name)
        if col.dtype[0] in (_DtypeKind.INT, _DtypeKind.UINT, _DtypeKind.FLOAT, _DtypeKind.BOOL):
            (columns[name], _buf) = _protocol_to_cudf_column_numeric(col, allow_copy)
        elif col.dtype[0] == _DtypeKind.CATEGORICAL:
            (columns[name], _buf) = _protocol_to_cudf_column_categorical(col, allow_copy)
        elif col.dtype[0] == _DtypeKind.STRING:
            (columns[name], _buf) = _protocol_to_cudf_column_string(col, allow_copy)
        else:
            raise NotImplementedError(f'Data type {col.dtype[0]} not handled yet')
        _buffers.append(_buf)
    df_new = cudf.DataFrame._from_data(columns)
    df_new._buffers = _buffers
    return df_new

def _protocol_to_cudf_column_numeric(col, allow_copy: bool) -> Tuple[cudf.core.column.ColumnBase, Mapping[str, Optional[Tuple[_CuDFBuffer, ProtoDtype]]]]:
    if False:
        print('Hello World!')
    '\n    Convert an int, uint, float or bool protocol column\n    to the corresponding cudf column\n    '
    if col.offset != 0:
        raise NotImplementedError('column.offset > 0 not handled yet')
    buffers = col.get_buffers()
    assert buffers['data'] is not None, 'data buffer should not be None'
    (_dbuffer, _ddtype) = buffers['data']
    _dbuffer = _ensure_gpu_buffer(_dbuffer, _ddtype, allow_copy)
    cudfcol_num = build_column(_dbuffer._buf, protocol_dtype_to_cupy_dtype(_ddtype))
    return (_set_missing_values(col, cudfcol_num, allow_copy), buffers)

def _ensure_gpu_buffer(buf, data_type, allow_copy: bool) -> _CuDFBuffer:
    if False:
        i = 10
        return i + 15
    if buf.__dlpack_device__()[0] != _Device.CUDA:
        if allow_copy:
            dbuf = rmm.DeviceBuffer(ptr=buf.ptr, size=buf.bufsize)
            return _CuDFBuffer(as_buffer(dbuf, exposed=True), protocol_dtype_to_cupy_dtype(data_type), allow_copy)
        else:
            raise TypeError('This operation must copy data from CPU to GPU. Set `allow_copy=True` to allow it.')
    return buf

def _set_missing_values(protocol_col, cudf_col: cudf.core.column.ColumnBase, allow_copy: bool) -> cudf.core.column.ColumnBase:
    if False:
        i = 10
        return i + 15
    valid_mask = protocol_col.get_buffers()['validity']
    if valid_mask is not None:
        (null, invalid) = protocol_col.describe_null
        if null == _MaskKind.BYTEMASK:
            valid_mask = _ensure_gpu_buffer(valid_mask[0], valid_mask[1], allow_copy)
            boolmask = as_column(valid_mask._buf, dtype='bool')
            bitmask = cudf._lib.transform.bools_to_mask(boolmask)
            return cudf_col.set_mask(bitmask)
        elif null == _MaskKind.BITMASK:
            valid_mask = _ensure_gpu_buffer(valid_mask[0], valid_mask[1], allow_copy)
            bitmask = valid_mask._buf
            return cudf_col.set_mask(bitmask)
    return cudf_col

def protocol_dtype_to_cupy_dtype(_dtype: ProtoDtype) -> cp.dtype:
    if False:
        print('Hello World!')
    kind = _dtype[0]
    bitwidth = _dtype[1]
    if _dtype[0] not in _SUPPORTED_KINDS:
        raise RuntimeError(f'Data type {_dtype[0]} not handled yet')
    return _CP_DTYPES[kind][bitwidth]

def _protocol_to_cudf_column_categorical(col, allow_copy: bool) -> Tuple[cudf.core.column.ColumnBase, Mapping[str, Optional[Tuple[_CuDFBuffer, ProtoDtype]]]]:
    if False:
        print('Hello World!')
    '\n    Convert a categorical column to a Series instance\n    '
    (ordered, is_dict, categories) = col.describe_categorical
    if not is_dict:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')
    buffers = col.get_buffers()
    assert buffers['data'] is not None, 'data buffer should not be None'
    (codes_buffer, codes_dtype) = buffers['data']
    codes_buffer = _ensure_gpu_buffer(codes_buffer, codes_dtype, allow_copy)
    cdtype = protocol_dtype_to_cupy_dtype(codes_dtype)
    codes = build_column(codes_buffer._buf, cdtype)
    cudfcol = build_categorical_column(categories=categories, codes=codes, mask=codes.base_mask, size=codes.size, ordered=ordered)
    return (_set_missing_values(col, cudfcol, allow_copy), buffers)

def _protocol_to_cudf_column_string(col, allow_copy: bool) -> Tuple[cudf.core.column.ColumnBase, Mapping[str, Optional[Tuple[_CuDFBuffer, ProtoDtype]]]]:
    if False:
        print('Hello World!')
    '\n    Convert a string ColumnObject to cudf Column object.\n    '
    buffers = col.get_buffers()
    assert buffers['data'] is not None, 'data buffer should never be None'
    (data_buffer, data_dtype) = buffers['data']
    data_buffer = _ensure_gpu_buffer(data_buffer, data_dtype, allow_copy)
    encoded_string = build_column(data_buffer._buf, protocol_dtype_to_cupy_dtype(data_dtype))
    assert buffers['offsets'] is not None, 'not possible for string column'
    (offset_buffer, offset_dtype) = buffers['offsets']
    offset_buffer = _ensure_gpu_buffer(offset_buffer, offset_dtype, allow_copy)
    offsets = build_column(offset_buffer._buf, protocol_dtype_to_cupy_dtype(offset_dtype))
    offsets = offsets.astype('int32')
    cudfcol_str = build_column(None, dtype=cp.dtype('O'), children=(offsets, encoded_string))
    return (_set_missing_values(col, cudfcol_str, allow_copy), buffers)

def _protocol_buffer_to_cudf_buffer(protocol_buffer):
    if False:
        return 10
    return as_buffer(rmm.DeviceBuffer(ptr=protocol_buffer.ptr, size=protocol_buffer.bufsize), exposed=True)