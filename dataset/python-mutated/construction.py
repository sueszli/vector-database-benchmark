"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""
from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar, dict_compat, maybe_cast_to_datetime, maybe_convert_platform, maybe_infer_to_datetimelike
from pandas.core.dtypes.common import is_1d_only_ea_dtype, is_integer_dtype, is_list_like, is_named_tuple, is_object_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core import algorithms, common as com
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import array as pd_array, ensure_wrapped_if_datetimelike, extract_array, range_to_ndarray, sanitize_array
from pandas.core.indexes.api import DatetimeIndex, Index, TimedeltaIndex, default_index, ensure_index, get_objs_combined_axis, union_indexes
from pandas.core.internals.array_manager import ArrayManager, SingleArrayManager
from pandas.core.internals.blocks import BlockPlacement, ensure_block_shape, new_block, new_block_2d
from pandas.core.internals.managers import BlockManager, SingleBlockManager, create_block_manager_from_blocks, create_block_manager_from_column_arrays
if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from pandas._typing import ArrayLike, DtypeObj, Manager, npt

def arrays_to_mgr(arrays, columns: Index, index, *, dtype: DtypeObj | None=None, verify_integrity: bool=True, typ: str | None=None, consolidate: bool=True) -> Manager:
    if False:
        while True:
            i = 10
    '\n    Segregate Series based on type and coerce into matrices.\n\n    Needs to handle a lot of exceptional cases.\n    '
    if verify_integrity:
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)
        (arrays, refs) = _homogenize(arrays, index, dtype)
    else:
        index = ensure_index(index)
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        refs = [None] * len(arrays)
        for arr in arrays:
            if not isinstance(arr, (np.ndarray, ExtensionArray)) or arr.ndim != 1 or len(arr) != len(index):
                raise ValueError('Arrays must be 1-dimensional np.ndarray or ExtensionArray with length matching len(index)')
    columns = ensure_index(columns)
    if len(columns) != len(arrays):
        raise ValueError('len(arrays) must match len(columns)')
    axes = [columns, index]
    if typ == 'block':
        return create_block_manager_from_column_arrays(arrays, axes, consolidate=consolidate, refs=refs)
    elif typ == 'array':
        return ArrayManager(arrays, [index, columns])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")

def rec_array_to_mgr(data: np.rec.recarray | np.ndarray, index, columns, dtype: DtypeObj | None, copy: bool, typ: str) -> Manager:
    if False:
        print('Hello World!')
    '\n    Extract from a masked rec array and create the manager.\n    '
    fdata = ma.getdata(data)
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)
    if columns is not None:
        columns = ensure_index(columns)
    (arrays, arr_columns) = to_arrays(fdata, columns)
    (arrays, arr_columns) = reorder_arrays(arrays, arr_columns, columns, len(index))
    if columns is None:
        columns = arr_columns
    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ)
    if copy:
        mgr = mgr.copy()
    return mgr

def mgr_to_mgr(mgr, typ: str, copy: bool=True) -> Manager:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert to specific type of Manager. Does not copy if the type is already\n    correct. Does not guarantee a copy otherwise. `copy` keyword only controls\n    whether conversion from Block->ArrayManager copies the 1D arrays.\n    '
    new_mgr: Manager
    if typ == 'block':
        if isinstance(mgr, BlockManager):
            new_mgr = mgr
        elif mgr.ndim == 2:
            new_mgr = arrays_to_mgr(mgr.arrays, mgr.axes[0], mgr.axes[1], typ='block')
        else:
            new_mgr = SingleBlockManager.from_array(mgr.arrays[0], mgr.index)
    elif typ == 'array':
        if isinstance(mgr, ArrayManager):
            new_mgr = mgr
        elif mgr.ndim == 2:
            arrays = [mgr.iget_values(i) for i in range(len(mgr.axes[0]))]
            if copy:
                arrays = [arr.copy() for arr in arrays]
            new_mgr = ArrayManager(arrays, [mgr.axes[1], mgr.axes[0]])
        else:
            array = mgr.internal_values()
            if copy:
                array = array.copy()
            new_mgr = SingleArrayManager([array], [mgr.index])
    else:
        raise ValueError(f"'typ' needs to be one of {{'block', 'array'}}, got '{typ}'")
    return new_mgr

def ndarray_to_mgr(values, index, columns, dtype: DtypeObj | None, copy: bool, typ: str) -> Manager:
    if False:
        while True:
            i = 10
    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = Index([values.name])
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)
    copy_on_sanitize = False if typ == 'array' else copy
    vdtype = getattr(values, 'dtype', None)
    refs = None
    if is_1d_only_ea_dtype(vdtype) or is_1d_only_ea_dtype(dtype):
        if isinstance(values, (np.ndarray, ExtensionArray)) and values.ndim > 1:
            values = [values[:, n] for n in range(values.shape[1])]
        else:
            values = [values]
        if columns is None:
            columns = Index(range(len(values)))
        else:
            columns = ensure_index(columns)
        return arrays_to_mgr(values, columns, index, dtype=dtype, typ=typ)
    elif isinstance(vdtype, ExtensionDtype):
        values = extract_array(values, extract_numpy=True)
        if copy:
            values = values.copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    elif isinstance(values, (ABCSeries, Index)):
        if not copy_on_sanitize and (dtype is None or astype_is_view(values.dtype, dtype)):
            refs = values._references
        if copy_on_sanitize:
            values = values._values.copy()
        else:
            values = values._values
        values = _ensure_2d(values)
    elif isinstance(values, (np.ndarray, ExtensionArray)):
        _copy = copy_on_sanitize if dtype is None or astype_is_view(values.dtype, dtype) else False
        values = np.array(values, copy=_copy)
        values = _ensure_2d(values)
    else:
        values = _prep_ndarraylike(values, copy=copy_on_sanitize)
    if dtype is not None and values.dtype != dtype:
        values = sanitize_array(values, None, dtype=dtype, copy=copy_on_sanitize, allow_2d=True)
    (index, columns) = _get_axes(values.shape[0], values.shape[1], index=index, columns=columns)
    _check_values_indices_shape_match(values, index, columns)
    if typ == 'array':
        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)
        if dtype is None and is_object_dtype(values.dtype):
            arrays = [ensure_wrapped_if_datetimelike(maybe_infer_to_datetimelike(values[:, i])) for i in range(values.shape[1])]
        else:
            if lib.is_np_dtype(values.dtype, 'mM'):
                values = ensure_wrapped_if_datetimelike(values)
            arrays = [values[:, i] for i in range(values.shape[1])]
        if copy:
            arrays = [arr.copy() for arr in arrays]
        return ArrayManager(arrays, [index, columns], verify_integrity=False)
    values = values.T
    if dtype is None and is_object_dtype(values.dtype):
        obj_columns = list(values)
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        if any((x is not y for (x, y) in zip(obj_columns, maybe_datetime))):
            dvals_list = [ensure_block_shape(dval, 2) for dval in maybe_datetime]
            block_values = [new_block_2d(dvals_list[n], placement=BlockPlacement(n)) for n in range(len(dvals_list))]
        else:
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == 'U' and using_pyarrow_string_dtype():
        dtype = StringDtype(storage='pyarrow_numpy')
        obj_columns = list(values)
        block_values = [new_block(dtype.construct_array_type()._from_sequence(data, dtype=dtype), BlockPlacement(slice(i, i + 1)), ndim=2) for (i, data) in enumerate(obj_columns)]
    else:
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]
    if len(columns) == 0:
        block_values = []
    return create_block_manager_from_blocks(block_values, [columns, index], verify_integrity=False)

def _check_values_indices_shape_match(values: np.ndarray, index: Index, columns: Index) -> None:
    if False:
        return 10
    '\n    Check that the shape implied by our axes matches the actual shape of the\n    data.\n    '
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        if values.shape[0] == 0 < len(index):
            raise ValueError('Empty data passed with indices specified.')
        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f'Shape of passed values is {passed}, indices imply {implied}')

def dict_to_mgr(data: dict, index, columns, *, dtype: DtypeObj | None=None, typ: str='block', copy: bool=True) -> Manager:
    if False:
        while True:
            i = 10
    '\n    Segregate Series based on type and coerce into matrices.\n    Needs to handle a lot of exceptional cases.\n\n    Used in DataFrame.__init__\n    '
    arrays: Sequence[Any] | Series
    if columns is not None:
        from pandas.core.series import Series
        arrays = Series(data, index=columns, dtype=object)
        missing = arrays.isna()
        if index is None:
            index = _extract_index(arrays[~missing])
        else:
            index = ensure_index(index)
        if missing.any() and (not is_integer_dtype(dtype)):
            nan_dtype: DtypeObj
            if dtype is not None:
                midxs = missing.values.nonzero()[0]
                for i in midxs:
                    arr = sanitize_array(arrays.iat[i], index, dtype=dtype)
                    arrays.iat[i] = arr
            else:
                nan_dtype = np.dtype('object')
                val = construct_1d_arraylike_from_scalar(np.nan, len(index), nan_dtype)
                nmissing = missing.sum()
                if copy:
                    rhs = [val] * nmissing
                else:
                    rhs = [val.copy() for _ in range(nmissing)]
                arrays.loc[missing] = rhs
        arrays = list(arrays)
        columns = ensure_index(columns)
    else:
        keys = list(data.keys())
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
    if copy:
        if typ == 'block':
            arrays = [x.copy() if isinstance(x, ExtensionArray) else x.copy(deep=True) if isinstance(x, Index) or (isinstance(x, ABCSeries) and is_1d_only_ea_dtype(x.dtype)) else x for x in arrays]
        else:
            arrays = [x.copy() if hasattr(x, 'dtype') else x for x in arrays]
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)

def nested_data_to_arrays(data: Sequence, columns: Index | None, index: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index, Index]:
    if False:
        print('Hello World!')
    '\n    Convert a single sequence of arrays to multiple arrays.\n    '
    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)
    (arrays, columns) = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)
    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))
    return (arrays, columns, index)

def treat_as_nested(data) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if we should use nested_data_to_arrays.\n    '
    return len(data) > 0 and is_list_like(data[0]) and (getattr(data[0], 'ndim', 1) == 1) and (not (isinstance(data, ExtensionArray) and data.ndim == 2))

def _prep_ndarraylike(values, copy: bool=True) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    if len(values) == 0:
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v):
        if False:
            while True:
                i = 10
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v
        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        return res
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)
    return _ensure_2d(values)

def _ensure_2d(values: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    Reshape 1D values, raise on anything else other than 2D.\n    '
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f'Must pass 2-d input. shape={values.shape}')
    return values

def _homogenize(data, index: Index, dtype: DtypeObj | None) -> tuple[list[ArrayLike], list[Any]]:
    if False:
        print('Hello World!')
    oindex = None
    homogenized = []
    refs: list[Any] = []
    for val in data:
        if isinstance(val, (ABCSeries, Index)):
            if dtype is not None:
                val = val.astype(dtype, copy=False)
            if isinstance(val, ABCSeries) and val.index is not index:
                val = val.reindex(index, copy=False)
            refs.append(val._references)
            val = val._values
        else:
            if isinstance(val, dict):
                if oindex is None:
                    oindex = index.astype('O')
                if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                    val = dict_compat(val)
                else:
                    val = dict(val)
                val = lib.fast_multiget(val, oindex._values, default=np.nan)
            val = sanitize_array(val, index, dtype=dtype, copy=False)
            com.require_length_match(val, index)
            refs.append(None)
        homogenized.append(val)
    return (homogenized, refs)

def _extract_index(data) -> Index:
    if False:
        i = 10
        return i + 15
    '\n    Try to infer an Index from the passed data, raise ValueError on failure.\n    '
    index: Index
    if len(data) == 0:
        return default_index(0)
    raw_lengths = []
    indexes: list[list[Hashable] | Index] = []
    have_raw_arrays = False
    have_series = False
    have_dicts = False
    for val in data:
        if isinstance(val, ABCSeries):
            have_series = True
            indexes.append(val.index)
        elif isinstance(val, dict):
            have_dicts = True
            indexes.append(list(val.keys()))
        elif is_list_like(val) and getattr(val, 'ndim', 1) == 1:
            have_raw_arrays = True
            raw_lengths.append(len(val))
        elif isinstance(val, np.ndarray) and val.ndim > 1:
            raise ValueError('Per-column arrays must each be 1-dimensional')
    if not indexes and (not raw_lengths):
        raise ValueError('If using all scalar values, you must pass an index')
    if have_series:
        index = union_indexes(indexes)
    elif have_dicts:
        index = union_indexes(indexes, sort=False)
    if have_raw_arrays:
        lengths = list(set(raw_lengths))
        if len(lengths) > 1:
            raise ValueError('All arrays must be of the same length')
        if have_dicts:
            raise ValueError('Mixing dicts with non-Series may lead to ambiguous ordering.')
        if have_series:
            if lengths[0] != len(index):
                msg = f'array length {lengths[0]} does not match index length {len(index)}'
                raise ValueError(msg)
        else:
            index = default_index(lengths[0])
    return ensure_index(index)

def reorder_arrays(arrays: list[ArrayLike], arr_columns: Index, columns: Index | None, length: int) -> tuple[list[ArrayLike], Index]:
    if False:
        print('Hello World!')
    '\n    Pre-emptively (cheaply) reindex arrays with new columns.\n    '
    if columns is not None:
        if not columns.equals(arr_columns):
            new_arrays: list[ArrayLike] = []
            indexer = arr_columns.get_indexer(columns)
            for (i, k) in enumerate(indexer):
                if k == -1:
                    arr = np.empty(length, dtype=object)
                    arr.fill(np.nan)
                else:
                    arr = arrays[k]
                new_arrays.append(arr)
            arrays = new_arrays
            arr_columns = columns
    return (arrays, arr_columns)

def _get_names_from_index(data) -> Index:
    if False:
        for i in range(10):
            print('nop')
    has_some_name = any((getattr(s, 'name', None) is not None for s in data))
    if not has_some_name:
        return default_index(len(data))
    index: list[Hashable] = list(range(len(data)))
    count = 0
    for (i, s) in enumerate(data):
        n = getattr(s, 'name', None)
        if n is not None:
            index[i] = n
        else:
            index[i] = f'Unnamed {count}'
            count += 1
    return Index(index)

def _get_axes(N: int, K: int, index: Index | None, columns: Index | None) -> tuple[Index, Index]:
    if False:
        while True:
            i = 10
    if index is None:
        index = default_index(N)
    else:
        index = ensure_index(index)
    if columns is None:
        columns = default_index(K)
    else:
        columns = ensure_index(columns)
    return (index, columns)

def dataclasses_to_dicts(data):
    if False:
        while True:
            i = 10
    "\n    Converts a list of dataclass instances to a list of dictionaries.\n\n    Parameters\n    ----------\n    data : List[Type[dataclass]]\n\n    Returns\n    --------\n    list_dict : List[dict]\n\n    Examples\n    --------\n    >>> from dataclasses import dataclass\n    >>> @dataclass\n    ... class Point:\n    ...     x: int\n    ...     y: int\n\n    >>> dataclasses_to_dicts([Point(1, 2), Point(2, 3)])\n    [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]\n\n    "
    from dataclasses import asdict
    return list(map(asdict, data))

def to_arrays(data, columns: Index | None, dtype: DtypeObj | None=None) -> tuple[list[ArrayLike], Index]:
    if False:
        while True:
            i = 10
    '\n    Return list of arrays, columns.\n\n    Returns\n    -------\n    list[ArrayLike]\n        These will become columns in a DataFrame.\n    Index\n        This will become frame.columns.\n\n    Notes\n    -----\n    Ensures that len(result_arrays) == len(result_index).\n    '
    if not len(data):
        if isinstance(data, np.ndarray):
            if data.dtype.names is not None:
                columns = ensure_index(data.dtype.names)
                arrays = [data[name] for name in columns]
                if len(data) == 0:
                    for (i, arr) in enumerate(arrays):
                        if arr.ndim == 2:
                            arrays[i] = arr[:, 0]
                return (arrays, columns)
        return ([], ensure_index([]))
    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        columns = Index(list(data.dtype.names))
        arrays = [data[k] for k in columns]
        return (arrays, columns)
    if isinstance(data[0], (list, tuple)):
        arr = _list_to_arrays(data)
    elif isinstance(data[0], abc.Mapping):
        (arr, columns) = _list_of_dict_to_arrays(data, columns)
    elif isinstance(data[0], ABCSeries):
        (arr, columns) = _list_of_series_to_arrays(data, columns)
    else:
        data = [tuple(x) for x in data]
        arr = _list_to_arrays(data)
    (content, columns) = _finalize_columns_and_data(arr, columns, dtype)
    return (content, columns)

def _list_to_arrays(data: list[tuple | list]) -> np.ndarray:
    if False:
        return 10
    if isinstance(data[0], tuple):
        content = lib.to_object_array_tuples(data)
    else:
        content = lib.to_object_array(data)
    return content

def _list_of_series_to_arrays(data: list, columns: Index | None) -> tuple[np.ndarray, Index]:
    if False:
        i = 10
        return i + 15
    if columns is None:
        pass_data = [x for x in data if isinstance(x, (ABCSeries, ABCDataFrame))]
        columns = get_objs_combined_axis(pass_data, sort=False)
    indexer_cache: dict[int, np.ndarray] = {}
    aligned_values = []
    for s in data:
        index = getattr(s, 'index', None)
        if index is None:
            index = default_index(len(s))
        if id(index) in indexer_cache:
            indexer = indexer_cache[id(index)]
        else:
            indexer = indexer_cache[id(index)] = index.get_indexer(columns)
        values = extract_array(s, extract_numpy=True)
        aligned_values.append(algorithms.take_nd(values, indexer))
    content = np.vstack(aligned_values)
    return (content, columns)

def _list_of_dict_to_arrays(data: list[dict], columns: Index | None) -> tuple[np.ndarray, Index]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert list of dicts to numpy arrays\n\n    if `columns` is not passed, column names are inferred from the records\n    - for OrderedDict and dicts, the column names match\n      the key insertion-order from the first record to the last.\n    - For other kinds of dict-likes, the keys are lexically sorted.\n\n    Parameters\n    ----------\n    data : iterable\n        collection of records (OrderedDict, dict)\n    columns: iterables or None\n\n    Returns\n    -------\n    content : np.ndarray[object, ndim=2]\n    columns : Index\n    '
    if columns is None:
        gen = (list(x.keys()) for x in data)
        sort = not any((isinstance(d, dict) for d in data))
        pre_cols = lib.fast_unique_multiple_list_gen(gen, sort=sort)
        columns = ensure_index(pre_cols)
    data = [d if type(d) is dict else dict(d) for d in data]
    content = lib.dicts_to_array(data, list(columns))
    return (content, columns)

def _finalize_columns_and_data(content: np.ndarray, columns: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure we have valid columns, cast object dtypes if possible.\n    '
    contents = list(content.T)
    try:
        columns = _validate_or_indexify_columns(contents, columns)
    except AssertionError as err:
        raise ValueError(err) from err
    if len(contents) and contents[0].dtype == np.object_:
        contents = convert_object_array(contents, dtype=dtype)
    return (contents, columns)

def _validate_or_indexify_columns(content: list[np.ndarray], columns: Index | None) -> Index:
    if False:
        i = 10
        return i + 15
    '\n    If columns is None, make numbers as column names; Otherwise, validate that\n    columns have valid length.\n\n    Parameters\n    ----------\n    content : list of np.ndarrays\n    columns : Index or None\n\n    Returns\n    -------\n    Index\n        If columns is None, assign positional column index value as columns.\n\n    Raises\n    ------\n    1. AssertionError when content is not composed of list of lists, and if\n        length of columns is not equal to length of content.\n    2. ValueError when content is list of lists, but length of each sub-list\n        is not equal\n    3. ValueError when content is list of lists, but length of sub-list is\n        not equal to length of content\n    '
    if columns is None:
        columns = default_index(len(content))
    else:
        is_mi_list = isinstance(columns, list) and all((isinstance(col, list) for col in columns))
        if not is_mi_list and len(columns) != len(content):
            raise AssertionError(f'{len(columns)} columns passed, passed data had {len(content)} columns')
        if is_mi_list:
            if len({len(col) for col in columns}) > 1:
                raise ValueError('Length of columns passed for MultiIndex columns is different')
            if columns and len(columns[0]) != len(content):
                raise ValueError(f'{len(columns[0])} columns passed, passed data had {len(content)} columns')
    return columns

def convert_object_array(content: list[npt.NDArray[np.object_]], dtype: DtypeObj | None, dtype_backend: str='numpy', coerce_float: bool=False) -> list[ArrayLike]:
    if False:
        return 10
    '\n    Internal function to convert object array.\n\n    Parameters\n    ----------\n    content: List[np.ndarray]\n    dtype: np.dtype or ExtensionDtype\n    dtype_backend: Controls if nullable/pyarrow dtypes are returned.\n    coerce_float: Cast floats that are integers to int.\n\n    Returns\n    -------\n    List[ArrayLike]\n    '

    def convert(arr):
        if False:
            while True:
                i = 10
        if dtype != np.dtype('O'):
            arr = lib.maybe_convert_objects(arr, try_float=coerce_float, convert_to_nullable_dtype=dtype_backend != 'numpy')
            if dtype is None:
                if arr.dtype == np.dtype('O'):
                    arr = maybe_infer_to_datetimelike(arr)
                    if dtype_backend != 'numpy' and arr.dtype == np.dtype('O'):
                        arr = StringDtype().construct_array_type()._from_sequence(arr)
                elif dtype_backend != 'numpy' and isinstance(arr, np.ndarray):
                    if arr.dtype.kind in 'iufb':
                        arr = pd_array(arr, copy=False)
            elif isinstance(dtype, ExtensionDtype):
                cls = dtype.construct_array_type()
                arr = cls._from_sequence(arr, dtype=dtype, copy=False)
            elif dtype.kind in 'mM':
                arr = maybe_cast_to_datetime(arr, dtype)
        return arr
    arrays = [convert(arr) for arr in content]
    return arrays