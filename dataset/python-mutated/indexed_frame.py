"""Base class for Frame types that have an index."""
from __future__ import annotations
import numbers
import operator
import textwrap
import warnings
from collections import Counter, abc
from functools import cached_property
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Type, TypeVar, Union, cast
from uuid import uuid4
import cupy as cp
import numpy as np
import pandas as pd
from typing_extensions import Self
import cudf
import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype
from cudf._typing import ColumnLike, DataFrameOrSeries, Dtype, NotImplementedType
from cudf.api.extensions import no_default
from cudf.api.types import _is_non_decimal_numeric_dtype, is_bool_dtype, is_categorical_dtype, is_decimal_dtype, is_dict_like, is_list_dtype, is_list_like, is_scalar
from cudf.core._base_index import BaseIndex
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import ColumnBase, as_column, full
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import BooleanMask, GatherMap
from cudf.core.dtypes import ListDtype
from cudf.core.frame import Frame
from cudf.core.groupby.groupby import GroupBy
from cudf.core.index import Index, RangeIndex, _index_from_columns
from cudf.core.missing import NA
from cudf.core.multiindex import MultiIndex
from cudf.core.resample import _Resampler
from cudf.core.udf.utils import _compile_or_get, _get_input_args_from_frame, _post_process_output_col, _return_arr_from_dtype
from cudf.utils import docutils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import _warn_no_dask_cudf
doc_reset_index_template = '\n        Reset the index of the {klass}, or a level of it.\n\n        Parameters\n        ----------\n        level : int, str, tuple, or list, default None\n            Only remove the given levels from the index. Removes all levels by\n            default.\n        drop : bool, default False\n            Do not try to insert index into dataframe columns. This resets\n            the index to the default integer index.\n{argument}\n        inplace : bool, default False\n            Modify the DataFrame in place (do not create a new object).\n\n        Returns\n        -------\n        {return_type}\n            {klass} with the new index or None if ``inplace=True``.{return_doc}\n\n        Examples\n        --------\n        {example}\n'
doc_binop_template = textwrap.dedent("\n    Get {operation} of DataFrame or Series and other, element-wise (binary\n    operator `{op_name}`).\n\n    Equivalent to ``frame + other``, but with support to substitute a\n    ``fill_value`` for missing data in one of the inputs.\n\n    Parameters\n    ----------\n    other : scalar, sequence, Series, or DataFrame\n        Any single or multiple element data structure, or list-like object.\n    axis : int or string\n        Only ``0`` is supported for series, ``1`` or ``columns`` supported\n        for dataframe\n    level : int or name\n        Broadcast across a level, matching Index values on the\n        passed MultiIndex level. Not yet supported.\n    fill_value  : float or None, default None\n        Fill existing missing (NaN) values, and any new element needed\n        for successful DataFrame alignment, with this value before\n        computation. If data in both corresponding DataFrame locations\n        is missing the result will be missing.\n\n    Returns\n    -------\n    DataFrame or Series\n        Result of the arithmetic operation.\n\n    Examples\n    --------\n\n    **DataFrame**\n\n    >>> df = cudf.DataFrame(\n    ...     {{'angles': [0, 3, 4], 'degrees': [360, 180, 360]}},\n    ...     index=['circle', 'triangle', 'rectangle']\n    ... )\n    {df_op_example}\n\n    **Series**\n\n    >>> a = cudf.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])\n    >>> b = cudf.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])\n    {ser_op_example}\n    ")

def _get_host_unique(array):
    if False:
        return 10
    if isinstance(array, (cudf.Series, cudf.Index, ColumnBase)):
        return array.unique.to_pandas()
    elif isinstance(array, (str, numbers.Number)):
        return [array]
    else:
        return set(array)

def _drop_columns(f: Frame, columns: abc.Iterable, errors: str):
    if False:
        while True:
            i = 10
    for c in columns:
        try:
            f._drop_column(c)
        except KeyError as e:
            if errors == 'ignore':
                pass
            else:
                raise e

def _indices_from_labels(obj, labels):
    if False:
        print('Hello World!')
    if not isinstance(labels, cudf.MultiIndex):
        labels = cudf.core.column.as_column(labels)
        if is_categorical_dtype(obj.index):
            labels = labels.astype('category')
            codes = labels.codes.astype(obj.index._values.codes.dtype)
            labels = cudf.core.column.build_categorical_column(categories=labels.dtype.categories, codes=codes, ordered=labels.dtype.ordered)
        else:
            labels = labels.astype(obj.index.dtype)
    lhs = cudf.DataFrame({'__': cudf.core.column.arange(len(labels))}, index=labels)
    rhs = cudf.DataFrame({'_': cudf.core.column.arange(len(obj))}, index=obj.index)
    return lhs.join(rhs).sort_values(by=['__', '_'])['_']

def _get_label_range_or_mask(index, start, stop, step):
    if False:
        while True:
            i = 10
    if not (start is None and stop is None) and type(index) is cudf.core.index.DatetimeIndex and (index.is_monotonic_increasing is False):
        start = pd.to_datetime(start)
        stop = pd.to_datetime(stop)
        if start is not None and stop is not None:
            if start > stop:
                return slice(0, 0, None)
            boolean_mask = cp.logical_and(index >= start, index <= stop)
        elif start is not None:
            boolean_mask = index >= start
        else:
            boolean_mask = index <= stop
        return boolean_mask
    else:
        return index.find_label_range(slice(start, stop, step))

class _FrameIndexer:
    """Parent class for indexers."""

    def __init__(self, frame):
        if False:
            while True:
                i = 10
        self._frame = frame
_LocIndexerClass = TypeVar('_LocIndexerClass', bound='_FrameIndexer')
_IlocIndexerClass = TypeVar('_IlocIndexerClass', bound='_FrameIndexer')

class IndexedFrame(Frame):
    """A frame containing an index.

    This class encodes the common behaviors for core user-facing classes like
    DataFrame and Series that consist of a sequence of columns along with a
    special set of index columns.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """
    _loc_indexer_type: Type[_LocIndexerClass]
    _iloc_indexer_type: Type[_IlocIndexerClass]
    _index: cudf.core.index.BaseIndex
    _groupby = GroupBy
    _resampler = _Resampler
    _VALID_SCANS = {'cumsum', 'cumprod', 'cummin', 'cummax'}
    _SCAN_DOCSTRINGS = {'cumsum': {'op_name': 'cumulative sum'}, 'cumprod': {'op_name': 'cumulative product'}, 'cummin': {'op_name': 'cumulative min'}, 'cummax': {'op_name': 'cumulative max'}}

    def __init__(self, data=None, index=None):
        if False:
            print('Hello World!')
        super().__init__(data=data)
        self._index = index

    @property
    def _num_rows(self) -> int:
        if False:
            print('Hello World!')
        return len(self._index)

    @property
    def _index_names(self) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        return self._index._data.names

    @classmethod
    def _from_data(cls, data: MutableMapping, index: Optional[BaseIndex]=None):
        if False:
            for i in range(10):
                print('nop')
        out = super()._from_data(data)
        out._index = RangeIndex(out._data.nrows) if index is None else index
        return out

    @_cudf_nvtx_annotate
    def _from_data_like_self(self, data: MutableMapping):
        if False:
            return 10
        out = self._from_data(data, self._index)
        out._data._level_names = self._data._level_names
        return out

    @classmethod
    @_cudf_nvtx_annotate
    def _from_columns(cls, columns: List[ColumnBase], column_names: List[str], index_names: Optional[List[str]]=None):
        if False:
            i = 10
            return i + 15
        'Construct a `Frame` object from a list of columns.\n\n        If `index_names` is set, the first `len(index_names)` columns are\n        used to construct the index of the frame.\n        '
        data_columns = columns
        index = None
        if index_names is not None:
            n_index_columns = len(index_names)
            data_columns = columns[n_index_columns:]
            index = _index_from_columns(columns[:n_index_columns])
            if isinstance(index, cudf.MultiIndex):
                index.names = index_names
            else:
                index.name = index_names[0]
        out = super()._from_columns(data_columns, column_names)
        if index is not None:
            out._index = index
        return out

    @_cudf_nvtx_annotate
    def _from_columns_like_self(self, columns: List[ColumnBase], column_names: Optional[abc.Iterable[str]]=None, index_names: Optional[List[str]]=None, *, override_dtypes: Optional[abc.Iterable[Optional[Dtype]]]=None) -> Self:
        if False:
            while True:
                i = 10
        'Construct a `Frame` from a list of columns with metadata from self.\n\n        If `index_names` is set, the first `len(index_names)` columns are\n        used to construct the index of the frame.\n\n        If override_dtypes is provided then any non-None entry will be\n        used for the dtype of the matching column in preference to the\n        dtype of the column in self.\n        '
        if column_names is None:
            column_names = self._column_names
        frame = self.__class__._from_columns(columns, column_names, index_names)
        return frame._copy_type_metadata(self, include_index=bool(index_names), override_dtypes=override_dtypes)

    def __round__(self, digits=0):
        if False:
            i = 10
            return i + 15
        return self.round(decimals=digits)

    def _mimic_inplace(self, result: Self, inplace: bool=False) -> Optional[Self]:
        if False:
            return 10
        if inplace:
            self._index = result._index
        return super()._mimic_inplace(result, inplace)

    @_cudf_nvtx_annotate
    def _scan(self, op, axis=None, skipna=True):
        if False:
            i = 10
            return i + 15
        "\n        Return {op_name} of the {cls}.\n\n        Parameters\n        ----------\n        axis: {{index (0), columns(1)}}\n            Axis for the function to be applied on.\n        skipna: bool, default True\n            Exclude NA/null values. If an entire row/column is NA,\n            the result will be NA.\n\n        Returns\n        -------\n        {cls}\n\n        Examples\n        --------\n        **Series**\n\n        >>> import cudf\n        >>> ser = cudf.Series([1, 5, 2, 4, 3])\n        >>> ser.cumsum()\n        0    1\n        1    6\n        2    8\n        3    12\n        4    15\n\n        **DataFrame**\n\n        >>> import cudf\n        >>> df = cudf.DataFrame({{'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]}})\n        >>> s.cumsum()\n            a   b\n        0   1   7\n        1   3  15\n        2   6  24\n        3  10  34\n        "
        cast_to_int = op in ('cumsum', 'cumprod')
        skipna = True if skipna is None else skipna
        results = {}
        for (name, col) in self._data.items():
            if skipna:
                try:
                    result_col = col.nans_to_nulls()
                except AttributeError:
                    result_col = col
            elif col.has_nulls(include_nan=True):
                first_index = col.isnull().find_first_value(True)
                result_col = col.copy()
                result_col[first_index:] = None
            else:
                result_col = col
            if cast_to_int and (not is_decimal_dtype(result_col.dtype)) and (np.issubdtype(result_col.dtype, np.integer) or np.issubdtype(result_col.dtype, np.bool_)):
                result_col = result_col.astype(np.int64)
            results[name] = getattr(result_col, op)()
        return self._from_data(results, self._index)

    def _check_data_index_length_match(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._data.nrows > 0 and self._data.nrows != len(self._index):
            raise ValueError(f'Length of values ({self._data.nrows}) does not match length of index ({len(self._index)})')

    @property
    @_cudf_nvtx_annotate
    def empty(self):
        if False:
            print('Hello World!')
        "\n        Indicator whether DataFrame or Series is empty.\n\n        True if DataFrame/Series is entirely empty (no items),\n        meaning any of the axes are of length 0.\n\n        Returns\n        -------\n        out : bool\n            If DataFrame/Series is empty, return True, if not return False.\n\n        Notes\n        -----\n        If DataFrame/Series contains only `null` values, it is still not\n        considered empty. See the example below.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame({'A' : []})\n        >>> df\n        Empty DataFrame\n        Columns: [A]\n        Index: []\n        >>> df.empty\n        True\n\n        If we only have `null` values in our DataFrame, it is\n        not considered empty! We will need to drop\n        the `null`'s to make the DataFrame empty:\n\n        >>> df = cudf.DataFrame({'A' : [None, None]})\n        >>> df\n              A\n        0  <NA>\n        1  <NA>\n        >>> df.empty\n        False\n        >>> df.dropna().empty\n        True\n\n        Non-empty and empty Series example:\n\n        >>> s = cudf.Series([1, 2, None])\n        >>> s\n        0       1\n        1       2\n        2    <NA>\n        dtype: int64\n        >>> s.empty\n        False\n        >>> s = cudf.Series([])\n        >>> s\n        Series([], dtype: float64)\n        >>> s.empty\n        True\n        "
        return self.size == 0

    def copy(self, deep: bool=True) -> Self:
        if False:
            print('Hello World!')
        'Make a copy of this object\'s indices and data.\n\n        When ``deep=True`` (default), a new object will be created with a\n        copy of the calling object\'s data and indices. Modifications to\n        the data or indices of the copy will not be reflected in the\n        original object (see notes below).\n        When ``deep=False``, a new object will be created without copying\n        the calling object\'s data or index (only references to the data\n        and index are copied). Any changes to the data of the original\n        will be reflected in the shallow copy (and vice versa).\n\n        Parameters\n        ----------\n        deep : bool, default True\n            Make a deep copy, including a copy of the data and the indices.\n            With ``deep=False`` neither the indices nor the data are copied.\n\n        Returns\n        -------\n        copy : Series or DataFrame\n            Object type matches caller.\n\n        Examples\n        --------\n        >>> s = cudf.Series([1, 2], index=["a", "b"])\n        >>> s\n        a    1\n        b    2\n        dtype: int64\n        >>> s_copy = s.copy()\n        >>> s_copy\n        a    1\n        b    2\n        dtype: int64\n\n        **Shallow copy versus default (deep) copy:**\n\n        >>> s = cudf.Series([1, 2], index=["a", "b"])\n        >>> deep = s.copy()\n        >>> shallow = s.copy(deep=False)\n\n        Updates to the data shared by shallow copy and original is reflected\n        in both; deep copy remains unchanged.\n\n        >>> s[\'a\'] = 3\n        >>> shallow[\'b\'] = 4\n        >>> s\n        a    3\n        b    4\n        dtype: int64\n        >>> shallow\n        a    3\n        b    4\n        dtype: int64\n        >>> deep\n        a    1\n        b    2\n        dtype: int64\n        '
        return self._from_data(self._data.copy(deep=deep), self._index.copy(deep=False))

    @_cudf_nvtx_annotate
    def equals(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not super().equals(other):
            return False
        return self._index.equals(other._index)

    @property
    def index(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the labels for the rows.'
        return self._index

    @index.setter
    def index(self, value):
        if False:
            return 10
        old_length = len(self)
        new_length = len(value)
        if len(self._data) > 0 and new_length != old_length:
            raise ValueError(f'Length mismatch: Expected axis has {old_length} elements, new values have {len(value)} elements')
        self._index = Index(value)

    @_cudf_nvtx_annotate
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method=None):
        if False:
            print('Hello World!')
        "Replace values given in ``to_replace`` with ``value``.\n\n        Parameters\n        ----------\n        to_replace : numeric, str or list-like\n            Value(s) to replace.\n\n            * numeric or str:\n                - values equal to ``to_replace`` will be replaced\n                  with ``value``\n            * list of numeric or str:\n                - If ``value`` is also list-like, ``to_replace`` and\n                  ``value`` must be of same length.\n            * dict:\n                - Dicts can be used to specify different replacement values\n                  for different existing values. For example, {'a': 'b',\n                  'y': 'z'} replaces the value 'a' with 'b' and\n                  'y' with 'z'.\n                  To use a dict in this way the ``value`` parameter should\n                  be ``None``.\n        value : scalar, dict, list-like, str, default None\n            Value to replace any values matching ``to_replace`` with.\n        inplace : bool, default False\n            If True, in place.\n\n        See Also\n        --------\n        Series.fillna\n\n        Raises\n        ------\n        TypeError\n            - If ``to_replace`` is not a scalar, array-like, dict, or None\n            - If ``to_replace`` is a dict and value is not a list, dict,\n              or Series\n        ValueError\n            - If a list is passed to ``to_replace`` and ``value`` but they\n              are not the same length.\n\n        Returns\n        -------\n        result : Series\n            Series after replacement. The mask and index are preserved.\n\n        Notes\n        -----\n        Parameters that are currently not supported are: `limit`, `regex`,\n        `method`\n\n        Examples\n        --------\n        **Series**\n\n        Scalar ``to_replace`` and ``value``\n\n        >>> import cudf\n        >>> s = cudf.Series([0, 1, 2, 3, 4])\n        >>> s\n        0    0\n        1    1\n        2    2\n        3    3\n        4    4\n        dtype: int64\n        >>> s.replace(0, 5)\n        0    5\n        1    1\n        2    2\n        3    3\n        4    4\n        dtype: int64\n\n        List-like ``to_replace``\n\n        >>> s.replace([1, 2], 10)\n        0     0\n        1    10\n        2    10\n        3     3\n        4     4\n        dtype: int64\n\n        dict-like ``to_replace``\n\n        >>> s.replace({1:5, 3:50})\n        0     0\n        1     5\n        2     2\n        3    50\n        4     4\n        dtype: int64\n        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])\n        >>> s\n        0     b\n        1     a\n        2     a\n        3     b\n        4     a\n        dtype: object\n        >>> s.replace({'a': None})\n        0       b\n        1    <NA>\n        2    <NA>\n        3       b\n        4    <NA>\n        dtype: object\n\n        If there is a mismatch in types of the values in\n        ``to_replace`` & ``value`` with the actual series, then\n        cudf exhibits different behavior with respect to pandas\n        and the pairs are ignored silently:\n\n        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])\n        >>> s\n        0    b\n        1    a\n        2    a\n        3    b\n        4    a\n        dtype: object\n        >>> s.replace('a', 1)\n        0    b\n        1    a\n        2    a\n        3    b\n        4    a\n        dtype: object\n        >>> s.replace(['a', 'c'], [1, 2])\n        0    b\n        1    a\n        2    a\n        3    b\n        4    a\n        dtype: object\n\n        **DataFrame**\n\n        Scalar ``to_replace`` and ``value``\n\n        >>> import cudf\n        >>> df = cudf.DataFrame({'A': [0, 1, 2, 3, 4],\n        ...                    'B': [5, 6, 7, 8, 9],\n        ...                    'C': ['a', 'b', 'c', 'd', 'e']})\n        >>> df\n           A  B  C\n        0  0  5  a\n        1  1  6  b\n        2  2  7  c\n        3  3  8  d\n        4  4  9  e\n        >>> df.replace(0, 5)\n           A  B  C\n        0  5  5  a\n        1  1  6  b\n        2  2  7  c\n        3  3  8  d\n        4  4  9  e\n\n        List-like ``to_replace``\n\n        >>> df.replace([0, 1, 2, 3], 4)\n           A  B  C\n        0  4  5  a\n        1  4  6  b\n        2  4  7  c\n        3  4  8  d\n        4  4  9  e\n        >>> df.replace([0, 1, 2, 3], [4, 3, 2, 1])\n           A  B  C\n        0  4  5  a\n        1  3  6  b\n        2  2  7  c\n        3  1  8  d\n        4  4  9  e\n\n        dict-like ``to_replace``\n\n        >>> df.replace({0: 10, 1: 100})\n             A  B  C\n        0   10  5  a\n        1  100  6  b\n        2    2  7  c\n        3    3  8  d\n        4    4  9  e\n        >>> df.replace({'A': 0, 'B': 5}, 100)\n             A    B  C\n        0  100  100  a\n        1    1    6  b\n        2    2    7  c\n        3    3    8  d\n        4    4    9  e\n        "
        if limit is not None:
            raise NotImplementedError('limit parameter is not implemented yet')
        if regex:
            raise NotImplementedError('regex parameter is not implemented yet')
        if method not in ('pad', None):
            raise NotImplementedError('method parameter is not implemented yet')
        if not (to_replace is None and value is None):
            copy_data = {}
            (all_na_per_column, to_replace_per_column, replacements_per_column) = _get_replacement_values_for_columns(to_replace=to_replace, value=value, columns_dtype_map=self._dtypes)
            for (name, col) in self._data.items():
                try:
                    copy_data[name] = col.find_and_replace(to_replace_per_column[name], replacements_per_column[name], all_na_per_column[name])
                except (KeyError, OverflowError):
                    copy_data[name] = col.copy(deep=True)
        else:
            copy_data = self._data.copy(deep=True)
        result = self._from_data(copy_data, self._index)
        return self._mimic_inplace(result, inplace=inplace)

    @_cudf_nvtx_annotate
    def clip(self, lower=None, upper=None, inplace=False, axis=1):
        if False:
            while True:
                i = 10
        '\n        Trim values at input threshold(s).\n\n        Assigns values outside boundary to boundary values.\n        Thresholds can be singular values or array like,\n        and in the latter case the clipping is performed\n        element-wise in the specified axis. Currently only\n        `axis=1` is supported.\n\n        Parameters\n        ----------\n        lower : scalar or array_like, default None\n            Minimum threshold value. All values below this\n            threshold will be set to it. If it is None,\n            there will be no clipping based on lower.\n            In case of Series/Index, lower is expected to be\n            a scalar or an array of size 1.\n        upper : scalar or array_like, default None\n            Maximum threshold value. All values below this\n            threshold will be set to it. If it is None,\n            there will be no clipping based on upper.\n            In case of Series, upper is expected to be\n            a scalar or an array of size 1.\n        inplace : bool, default False\n\n        Returns\n        -------\n        Clipped DataFrame/Series/Index/MultiIndex\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame({"a":[1, 2, 3, 4], "b":[\'a\', \'b\', \'c\', \'d\']})\n        >>> df.clip(lower=[2, \'b\'], upper=[3, \'c\'])\n           a  b\n        0  2  b\n        1  2  b\n        2  3  c\n        3  3  c\n\n        >>> df.clip(lower=None, upper=[3, \'c\'])\n           a  b\n        0  1  a\n        1  2  b\n        2  3  c\n        3  3  c\n\n        >>> df.clip(lower=[2, \'b\'], upper=None)\n           a  b\n        0  2  b\n        1  2  b\n        2  3  c\n        3  4  d\n\n        >>> df.clip(lower=2, upper=3, inplace=True)\n        >>> df\n           a  b\n        0  2  2\n        1  2  3\n        2  3  3\n        3  3  3\n\n        >>> import cudf\n        >>> sr = cudf.Series([1, 2, 3, 4])\n        >>> sr.clip(lower=2, upper=3)\n        0    2\n        1    2\n        2    3\n        3    3\n        dtype: int64\n\n        >>> sr.clip(lower=None, upper=3)\n        0    1\n        1    2\n        2    3\n        3    3\n        dtype: int64\n\n        >>> sr.clip(lower=2, upper=None, inplace=True)\n        >>> sr\n        0    2\n        1    2\n        2    3\n        3    4\n        dtype: int64\n        '
        if axis != 1:
            raise NotImplementedError('`axis is not yet supported in clip`')
        if lower is None and upper is None:
            return None if inplace is True else self.copy(deep=True)
        if is_scalar(lower):
            lower = np.full(self._num_columns, lower)
        if is_scalar(upper):
            upper = np.full(self._num_columns, upper)
        if len(lower) != len(upper):
            raise ValueError('Length of lower and upper should be equal')
        if len(lower) != self._num_columns:
            raise ValueError('Length of lower/upper should be equal to number of columns')
        if self.ndim == 1:
            if lower[0] is not None and upper[0] is not None and (lower[0] > upper[0]):
                (lower[0], upper[0]) = (upper[0], lower[0])
        data = {name: col.clip(lower[i], upper[i]) for (i, (name, col)) in enumerate(self._data.items())}
        output = self._from_data(data, self._index)
        output._copy_type_metadata(self, include_index=False)
        return self._mimic_inplace(output, inplace=inplace)

    def _copy_type_metadata(self, other: Self, include_index: bool=True, *, override_dtypes: Optional[abc.Iterable[Optional[Dtype]]]=None) -> Self:
        if False:
            print('Hello World!')
        '\n        Copy type metadata from each column of `other` to the corresponding\n        column of `self`.\n        See `ColumnBase._with_type_metadata` for more information.\n        '
        super()._copy_type_metadata(other, override_dtypes=override_dtypes)
        if include_index and self._index is not None and (other._index is not None):
            self._index._copy_type_metadata(other._index)
            if isinstance(other._index, cudf.core.index.CategoricalIndex) and (not isinstance(self._index, cudf.core.index.CategoricalIndex)):
                self._index = cudf.Index(cast(cudf.core.index.NumericIndex, self._index)._column, name=self._index.name)
            elif isinstance(other._index, cudf.MultiIndex) and (not isinstance(self._index, cudf.MultiIndex)):
                self._index = cudf.MultiIndex._from_data(self._index._data, name=self._index.name)
        return self

    @_cudf_nvtx_annotate
    def interpolate(self, method='linear', axis=0, limit=None, inplace=False, limit_direction=None, limit_area=None, downcast=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Interpolate data values between some points.\n\n        Parameters\n        ----------\n        method : str, default 'linear'\n            Interpolation technique to use. Currently,\n            only 'linear` is supported.\n            * 'linear': Ignore the index and treat the values as\n            equally spaced. This is the only method supported on MultiIndexes.\n            * 'index', 'values': linearly interpolate using the index as\n            an x-axis. Unsorted indices can lead to erroneous results.\n        axis : int, default 0\n            Axis to interpolate along. Currently,\n            only 'axis=0' is supported.\n        inplace : bool, default False\n            Update the data in place if possible.\n\n        Returns\n        -------\n        Series or DataFrame\n            Returns the same object type as the caller, interpolated at\n            some or all ``NaN`` values\n\n        "
        if method in {'pad', 'ffill'} and limit_direction != 'forward':
            raise ValueError(f"`limit_direction` must be 'forward' for method `{method}`")
        if method in {'backfill', 'bfill'} and limit_direction != 'backward':
            raise ValueError(f"`limit_direction` must be 'backward' for method `{method}`")
        data = self
        if not isinstance(data._index, cudf.RangeIndex):
            perm_sort = data._index.argsort()
            data = data._gather(GatherMap.from_column_unchecked(cudf.core.column.as_column(perm_sort), len(data), nullify=False))
        interpolator = cudf.core.algorithms.get_column_interpolator(method)
        columns = {}
        for (colname, col) in data._data.items():
            if col.nullable:
                col = col.astype('float64').fillna(np.nan)
            columns[colname] = interpolator(col, index=data._index)
        result = self._from_data(columns, index=data._index)
        return result if isinstance(data._index, cudf.RangeIndex) else result._gather(GatherMap.from_column_unchecked(cudf.core.column.as_column(perm_sort.argsort()), len(result), nullify=False))

    @_cudf_nvtx_annotate
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        if False:
            return 10
        'Shift values by `periods` positions.'
        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise ValueError('Only axis=0 is supported.')
        if freq is not None:
            raise ValueError('The freq argument is not yet supported.')
        data_columns = (col.shift(periods, fill_value) for col in self._columns)
        return self.__class__._from_data(zip(self._column_names, data_columns), self._index)

    @_cudf_nvtx_annotate
    def truncate(self, before=None, after=None, axis=0, copy=True):
        if False:
            while True:
                i = 10
        '\n        Truncate a Series or DataFrame before and after some index value.\n\n        This is a useful shorthand for boolean indexing based on index\n        values above or below certain thresholds.\n\n        Parameters\n        ----------\n        before : date, str, int\n            Truncate all rows before this index value.\n        after : date, str, int\n            Truncate all rows after this index value.\n        axis : {0 or \'index\', 1 or \'columns\'}, optional\n            Axis to truncate. Truncates the index (rows) by default.\n        copy : bool, default is True,\n            Return a copy of the truncated section.\n\n        Returns\n        -------\n            The truncated Series or DataFrame.\n\n        Notes\n        -----\n        If the index being truncated contains only datetime values,\n        `before` and `after` may be specified as strings instead of\n        Timestamps.\n\n        .. pandas-compat::\n            **DataFrame.truncate, Series.truncate**\n\n            The ``copy`` parameter is only present for API compatibility, but\n            ``copy=False`` is not supported. This method always generates a\n            copy.\n\n        Examples\n        --------\n        **Series**\n\n        >>> import cudf\n        >>> cs1 = cudf.Series([1, 2, 3, 4])\n        >>> cs1\n        0    1\n        1    2\n        2    3\n        3    4\n        dtype: int64\n\n        >>> cs1.truncate(before=1, after=2)\n        1    2\n        2    3\n        dtype: int64\n\n        >>> import cudf\n        >>> dates = cudf.date_range(\n        ...     \'2021-01-01 23:45:00\', \'2021-01-01 23:46:00\', freq=\'s\'\n        ... )\n        >>> cs2 = cudf.Series(range(len(dates)), index=dates)\n        >>> cs2\n        2021-01-01 23:45:00     0\n        2021-01-01 23:45:01     1\n        2021-01-01 23:45:02     2\n        2021-01-01 23:45:03     3\n        2021-01-01 23:45:04     4\n        2021-01-01 23:45:05     5\n        2021-01-01 23:45:06     6\n        2021-01-01 23:45:07     7\n        2021-01-01 23:45:08     8\n        2021-01-01 23:45:09     9\n        2021-01-01 23:45:10    10\n        2021-01-01 23:45:11    11\n        2021-01-01 23:45:12    12\n        2021-01-01 23:45:13    13\n        2021-01-01 23:45:14    14\n        2021-01-01 23:45:15    15\n        2021-01-01 23:45:16    16\n        2021-01-01 23:45:17    17\n        2021-01-01 23:45:18    18\n        2021-01-01 23:45:19    19\n        2021-01-01 23:45:20    20\n        2021-01-01 23:45:21    21\n        2021-01-01 23:45:22    22\n        2021-01-01 23:45:23    23\n        2021-01-01 23:45:24    24\n        ...\n        2021-01-01 23:45:56    56\n        2021-01-01 23:45:57    57\n        2021-01-01 23:45:58    58\n        2021-01-01 23:45:59    59\n        dtype: int64\n\n\n        >>> cs2.truncate(\n        ...     before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"\n        ... )\n        2021-01-01 23:45:18    18\n        2021-01-01 23:45:19    19\n        2021-01-01 23:45:20    20\n        2021-01-01 23:45:21    21\n        2021-01-01 23:45:22    22\n        2021-01-01 23:45:23    23\n        2021-01-01 23:45:24    24\n        2021-01-01 23:45:25    25\n        2021-01-01 23:45:26    26\n        2021-01-01 23:45:27    27\n        dtype: int64\n\n        >>> cs3 = cudf.Series({\'A\': 1, \'B\': 2, \'C\': 3, \'D\': 4})\n        >>> cs3\n        A    1\n        B    2\n        C    3\n        D    4\n        dtype: int64\n\n        >>> cs3.truncate(before=\'B\', after=\'C\')\n        B    2\n        C    3\n        dtype: int64\n\n        **DataFrame**\n\n        >>> df = cudf.DataFrame({\n        ...     \'A\': [\'a\', \'b\', \'c\', \'d\', \'e\'],\n        ...     \'B\': [\'f\', \'g\', \'h\', \'i\', \'j\'],\n        ...     \'C\': [\'k\', \'l\', \'m\', \'n\', \'o\']\n        ... }, index=[1, 2, 3, 4, 5])\n        >>> df\n           A  B  C\n        1  a  f  k\n        2  b  g  l\n        3  c  h  m\n        4  d  i  n\n        5  e  j  o\n\n        >>> df.truncate(before=2, after=4)\n           A  B  C\n        2  b  g  l\n        3  c  h  m\n        4  d  i  n\n\n        >>> df.truncate(before="A", after="B", axis="columns")\n           A  B\n        1  a  f\n        2  b  g\n        3  c  h\n        4  d  i\n        5  e  j\n\n        >>> import cudf\n        >>> dates = cudf.date_range(\n        ...     \'2021-01-01 23:45:00\', \'2021-01-01 23:46:00\', freq=\'s\'\n        ... )\n        >>> df2 = cudf.DataFrame(data={\'A\': 1, \'B\': 2}, index=dates)\n        >>> df2.head()\n                             A  B\n        2021-01-01 23:45:00  1  2\n        2021-01-01 23:45:01  1  2\n        2021-01-01 23:45:02  1  2\n        2021-01-01 23:45:03  1  2\n        2021-01-01 23:45:04  1  2\n\n        >>> df2.truncate(\n        ...     before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"\n        ... )\n                             A  B\n        2021-01-01 23:45:18  1  2\n        2021-01-01 23:45:19  1  2\n        2021-01-01 23:45:20  1  2\n        2021-01-01 23:45:21  1  2\n        2021-01-01 23:45:22  1  2\n        2021-01-01 23:45:23  1  2\n        2021-01-01 23:45:24  1  2\n        2021-01-01 23:45:25  1  2\n        2021-01-01 23:45:26  1  2\n        2021-01-01 23:45:27  1  2\n        '
        if not copy:
            raise ValueError('Truncating with copy=False is not supported.')
        axis = self._get_axis_from_axis_arg(axis)
        ax = self._index if axis == 0 else self._data.to_pandas_index()
        if not ax.is_monotonic_increasing and (not ax.is_monotonic_decreasing):
            raise ValueError('truncate requires a sorted index')
        if type(ax) is cudf.core.index.DatetimeIndex:
            before = pd.to_datetime(before)
            after = pd.to_datetime(after)
        if before is not None and after is not None and (before > after):
            raise ValueError(f'Truncate: {after} must be after {before}')
        if len(ax) > 1 and ax.is_monotonic_decreasing and (ax.nunique() > 1):
            (before, after) = (after, before)
        slicer = [slice(None, None)] * self.ndim
        slicer[axis] = slice(before, after)
        return self.loc[tuple(slicer)].copy()

    @cached_property
    def loc(self):
        if False:
            i = 10
            return i + 15
        "Select rows and columns by label or boolean mask.\n\n        Examples\n        --------\n        **Series**\n\n        >>> import cudf\n        >>> series = cudf.Series([10, 11, 12], index=['a', 'b', 'c'])\n        >>> series\n        a    10\n        b    11\n        c    12\n        dtype: int64\n        >>> series.loc['b']\n        11\n\n        **DataFrame**\n\n        DataFrame with string index.\n\n        >>> df\n           a  b\n        a  0  5\n        b  1  6\n        c  2  7\n        d  3  8\n        e  4  9\n\n        Select a single row by label.\n\n        >>> df.loc['a']\n        a    0\n        b    5\n        Name: a, dtype: int64\n\n        Select multiple rows and a single column.\n\n        >>> df.loc[['a', 'c', 'e'], 'b']\n        a    5\n        c    7\n        e    9\n        Name: b, dtype: int64\n\n        Selection by boolean mask.\n\n        >>> df.loc[df.a > 2]\n           a  b\n        d  3  8\n        e  4  9\n\n        Setting values using loc.\n\n        >>> df.loc[['a', 'c', 'e'], 'a'] = 0\n        >>> df\n           a  b\n        a  0  5\n        b  1  6\n        c  0  7\n        d  3  8\n        e  0  9\n\n        "
        return self._loc_indexer_type(self)

    @cached_property
    def iloc(self):
        if False:
            for i in range(10):
                print('nop')
        "Select values by position.\n\n        Examples\n        --------\n        **Series**\n\n        >>> import cudf\n        >>> s = cudf.Series([10, 20, 30])\n        >>> s\n        0    10\n        1    20\n        2    30\n        dtype: int64\n        >>> s.iloc[2]\n        30\n\n        **DataFrame**\n\n        Selecting rows and column by position.\n\n        >>> df = cudf.DataFrame({'a': range(20),\n        ...                      'b': range(20),\n        ...                      'c': range(20)})\n\n        Select a single row using an integer index.\n\n        >>> df.iloc[1]\n        a    1\n        b    1\n        c    1\n        Name: 1, dtype: int64\n\n        Select multiple rows using a list of integers.\n\n        >>> df.iloc[[0, 2, 9, 18]]\n              a    b    c\n         0    0    0    0\n         2    2    2    2\n         9    9    9    9\n        18   18   18   18\n\n        Select rows using a slice.\n\n        >>> df.iloc[3:10:2]\n             a    b    c\n        3    3    3    3\n        5    5    5    5\n        7    7    7    7\n        9    9    9    9\n\n        Select both rows and columns.\n\n        >>> df.iloc[[1, 3, 5, 7], 2]\n        1    1\n        3    3\n        5    5\n        7    7\n        Name: c, dtype: int64\n\n        Setting values in a column using iloc.\n\n        >>> df.iloc[:4] = 0\n        >>> df\n           a  b  c\n        0  0  0  0\n        1  0  0  0\n        2  0  0  0\n        3  0  0  0\n        4  4  4  4\n        5  5  5  5\n        6  6  6  6\n        7  7  7  7\n        8  8  8  8\n        9  9  9  9\n        [10 more rows]\n\n        "
        return self._iloc_indexer_type(self)

    @_cudf_nvtx_annotate
    def scale(self):
        if False:
            while True:
                i = 10
        '\n        Scale values to [0, 1] in float64\n\n        Returns\n        -------\n        DataFrame or Series\n            Values scaled to [0, 1].\n\n        Examples\n        --------\n        >>> import cudf\n        >>> series = cudf.Series([10, 11, 12, 0.5, 1])\n        >>> series\n        0    10.0\n        1    11.0\n        2    12.0\n        3     0.5\n        4     1.0\n        dtype: float64\n        >>> series.scale()\n        0    0.826087\n        1    0.913043\n        2    1.000000\n        3    0.000000\n        4    0.043478\n        dtype: float64\n        '
        vmin = self.min()
        vmax = self.max()
        scaled = (self - vmin) / (vmax - vmin)
        scaled._index = self._index.copy(deep=False)
        return scaled

    @_cudf_nvtx_annotate
    def sort_index(self, axis=0, level=None, ascending=True, inplace=False, kind=None, na_position='last', sort_remaining=True, ignore_index=False, key=None):
        if False:
            for i in range(10):
                print('nop')
        'Sort object by labels (along an axis).\n\n        Parameters\n        ----------\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            The axis along which to sort. The value 0 identifies the rows,\n            and 1 identifies the columns.\n        level : int or level name or list of ints or list of level names\n            If not None, sort on values in specified index level(s).\n            This is only useful in the case of MultiIndex.\n        ascending : bool, default True\n            Sort ascending vs. descending.\n        inplace : bool, default False\n            If True, perform operation in-place.\n        kind : sorting method such as `quick sort` and others.\n            Not yet supported.\n        na_position : {\'first\', \'last\'}, default \'last\'\n            Puts NaNs at the beginning if first; last puts NaNs at the end.\n        sort_remaining : bool, default True\n            When sorting a multiindex on a subset of its levels,\n            should entries be lexsorted by the remaining\n            (non-specified) levels as well?\n        ignore_index : bool, default False\n            if True, index will be replaced with RangeIndex.\n        key : callable, optional\n            If not None, apply the key function to the index values before\n            sorting. This is similar to the key argument in the builtin\n            sorted() function, with the notable difference that this key\n            function should be vectorized. It should expect an Index and return\n            an Index of the same shape. For MultiIndex inputs, the key is\n            applied per level.\n\n        Returns\n        -------\n        Frame or None\n\n        Notes\n        -----\n        Difference from pandas:\n          * Not supporting: kind, sort_remaining=False\n\n        Examples\n        --------\n        **Series**\n\n        >>> import cudf\n        >>> series = cudf.Series([\'a\', \'b\', \'c\', \'d\'], index=[3, 2, 1, 4])\n        >>> series\n        3    a\n        2    b\n        1    c\n        4    d\n        dtype: object\n        >>> series.sort_index()\n        1    c\n        2    b\n        3    a\n        4    d\n        dtype: object\n\n        Sort Descending\n\n        >>> series.sort_index(ascending=False)\n        4    d\n        3    a\n        2    b\n        1    c\n        dtype: object\n\n        **DataFrame**\n\n        >>> df = cudf.DataFrame(\n        ... {"b":[3, 2, 1], "a":[2, 1, 3]}, index=[1, 3, 2])\n        >>> df.sort_index(axis=0)\n           b  a\n        1  3  2\n        2  1  3\n        3  2  1\n        >>> df.sort_index(axis=1)\n           a  b\n        1  2  3\n        3  1  2\n        2  3  1\n        '
        if kind is not None:
            raise NotImplementedError('kind is not yet supported')
        if key is not None:
            raise NotImplementedError('key is not yet supported.')
        if na_position not in {'first', 'last'}:
            raise ValueError(f'invalid na_position: {na_position}')
        if axis in (0, 'index'):
            idx = self.index
            if isinstance(idx, MultiIndex):
                if level is not None:
                    na_position = 'first' if ascending is True else 'last'
                    if not is_list_like(level):
                        level = [level]
                    by = list(map(idx._get_level_label, level))
                    if sort_remaining:
                        handled = set(by)
                        by.extend(filter(lambda n: n not in handled, self.index._data.names))
                else:
                    by = list(idx._data.names)
                inds = idx._get_sorted_inds(by=by, ascending=ascending, na_position=na_position)
                out = self._gather(GatherMap.from_column_unchecked(inds, len(self), nullify=False))
                if isinstance(self, cudf.core.dataframe.DataFrame) and self._data.multiindex:
                    out._set_column_names_like(self)
            elif ascending and idx.is_monotonic_increasing or (not ascending and idx.is_monotonic_decreasing):
                out = self.copy()
            else:
                inds = idx.argsort(ascending=ascending, na_position=na_position)
                out = self._gather(GatherMap.from_column_unchecked(cudf.core.column.as_column(inds), len(self), nullify=False))
                if isinstance(self, cudf.core.dataframe.DataFrame) and self._data.multiindex:
                    out._set_column_names_like(self)
        else:
            labels = sorted(self._data.names, reverse=not ascending)
            out = self[labels]
        if ignore_index is True:
            out = out.reset_index(drop=True)
        return self._mimic_inplace(out, inplace=inplace)

    def memory_usage(self, index=True, deep=False):
        if False:
            while True:
                i = 10
        "Return the memory usage of an object.\n\n        Parameters\n        ----------\n        index : bool, default True\n            Specifies whether to include the memory usage of the index.\n        deep : bool, default False\n            The deep parameter is ignored and is only included for pandas\n            compatibility.\n\n        Returns\n        -------\n        Series or scalar\n            For DataFrame, a Series whose index is the original column names\n            and whose values is the memory usage of each column in bytes. For a\n            Series the total memory usage.\n\n        Examples\n        --------\n        **DataFrame**\n\n        >>> dtypes = ['int64', 'float64', 'object', 'bool']\n        >>> data = dict([(t, np.ones(shape=5000).astype(t))\n        ...              for t in dtypes])\n        >>> df = cudf.DataFrame(data)\n        >>> df.head()\n           int64  float64  object  bool\n        0      1      1.0     1.0  True\n        1      1      1.0     1.0  True\n        2      1      1.0     1.0  True\n        3      1      1.0     1.0  True\n        4      1      1.0     1.0  True\n        >>> df.memory_usage(index=False)\n        int64      40000\n        float64    40000\n        object     40000\n        bool        5000\n        dtype: int64\n\n        Use a Categorical for efficient storage of an object-dtype column with\n        many repeated values.\n\n        >>> df['object'].astype('category').memory_usage(deep=True)\n        5008\n\n        **Series**\n        >>> s = cudf.Series(range(3), index=['a','b','c'])\n        >>> s.memory_usage()\n        43\n\n        Not including the index gives the size of the rest of the data, which\n        is necessarily smaller:\n\n        >>> s.memory_usage(index=False)\n        24\n        "
        raise NotImplementedError

    def hash_values(self, method='murmur3', seed=None):
        if False:
            print('Hello World!')
        'Compute the hash of values in this column.\n\n        Parameters\n        ----------\n        method : {\'murmur3\', \'md5\'}, default \'murmur3\'\n            Hash function to use:\n            * murmur3: MurmurHash3 hash function.\n            * md5: MD5 hash function.\n\n        seed : int, optional\n            Seed value to use for the hash function.\n            Note - This only has effect for the following supported\n            hash functions:\n            * murmur3: MurmurHash3 hash function.\n\n        Returns\n        -------\n        Series\n            A Series with hash values.\n\n        Examples\n        --------\n        **Series**\n\n        >>> import cudf\n        >>> series = cudf.Series([10, 120, 30])\n        >>> series\n        0     10\n        1    120\n        2     30\n        dtype: int64\n        >>> series.hash_values(method="murmur3")\n        0   -1930516747\n        1     422619251\n        2    -941520876\n        dtype: int32\n        >>> series.hash_values(method="md5")\n        0    7be4bbacbfdb05fb3044e36c22b41e8b\n        1    947ca8d2c5f0f27437f156cfbfab0969\n        2    d0580ef52d27c043c8e341fd5039b166\n        dtype: object\n        >>> series.hash_values(method="murmur3", seed=42)\n        0    2364453205\n        1     422621911\n        2    3353449140\n        dtype: uint32\n\n        **DataFrame**\n\n        >>> import cudf\n        >>> df = cudf.DataFrame({"a": [10, 120, 30], "b": [0.0, 0.25, 0.50]})\n        >>> df\n             a     b\n        0   10  0.00\n        1  120  0.25\n        2   30  0.50\n        >>> df.hash_values(method="murmur3")\n        0    -330519225\n        1    -397962448\n        2   -1345834934\n        dtype: int32\n        >>> df.hash_values(method="md5")\n        0    57ce879751b5169c525907d5c563fae1\n        1    948d6221a7c4963d4be411bcead7e32b\n        2    fe061786ea286a515b772d91b0dfcd70\n        dtype: object\n        '
        seed_hash_methods = {'murmur3'}
        if seed is None:
            seed = 0
        elif method not in seed_hash_methods:
            warnings.warn(f'Provided seed value has no effect for hash method `{method}`. Refer to the docstring for information on hash methods that support the `seed` param')
        return cudf.Series._from_data({None: libcudf.hash.hash([*self._columns], method, seed)}, index=self.index)

    def _gather(self, gather_map: GatherMap, keep_index=True):
        if False:
            return 10
        'Gather rows of frame specified by indices in `gather_map`.\n\n        Maintain the index if keep_index is True.\n\n        This function does no expensive bounds checking, but does\n        check that the number of rows of self matches the validated\n        number of rows.\n        '
        if not gather_map.nullify and len(self) != gather_map.nrows:
            raise IndexError('Gather map is out of bounds')
        return self._from_columns_like_self(libcudf.copying.gather(list(self._index._columns + self._columns) if keep_index else list(self._columns), gather_map.column, nullify=gather_map.nullify), self._column_names, self._index.names if keep_index else None)

    def _slice(self, arg: slice, keep_index: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        'Slice a frame.\n\n        Parameters\n        ----------\n        arg\n            The slice\n        keep_index\n            Preserve the index when slicing?\n\n        Returns\n        -------\n        Sliced frame\n\n        Notes\n        -----\n        This slicing has normal python semantics.\n        '
        num_rows = len(self)
        if num_rows == 0:
            return self
        (start, stop, stride) = arg.indices(num_rows)
        index = self.index
        has_range_index = isinstance(index, RangeIndex)
        if len(range(start, stop, stride)) == 0:
            result = self._empty_like(keep_index=keep_index and (not has_range_index))
            if keep_index and has_range_index:
                lo = index.start + start * index.step
                hi = index.start + stop * index.step
                step = index.step * stride
                result.index = RangeIndex(start=lo, stop=hi, step=step, name=index.name)
            return result
        if start < 0:
            start = start + num_rows
        if stop < 0 and (not (stride < 0 and stop == -1)):
            stop = stop + num_rows
        stride = 1 if stride is None else stride
        if (stop - start) * stride <= 0:
            return self._empty_like(keep_index=True)
        start = min(start, num_rows)
        stop = min(stop, num_rows)
        if stride != 1:
            return self._gather(GatherMap.from_column_unchecked(cudf.core.column.arange(start, stop=stop, step=stride, dtype=libcudf.types.size_type_dtype), len(self), nullify=False), keep_index=keep_index)
        columns_to_slice = [*(self._index._data.columns if keep_index and (not has_range_index) else []), *self._columns]
        result = self._from_columns_like_self(libcudf.copying.columns_slice(columns_to_slice, [start, stop])[0], self._column_names, None if has_range_index or not keep_index else self._index.names)
        if keep_index and has_range_index:
            result.index = self.index[start:stop]
        return result

    def _positions_from_column_names(self, column_names, offset_by_index_columns=False):
        if False:
            return 10
        'Map each column name into their positions in the frame.\n\n        Return positions of the provided column names, offset by the number of\n        index columns if `offset_by_index_columns` is True. The order of\n        indices returned corresponds to the column order in this Frame.\n        '
        num_index_columns = len(self._index._data) if offset_by_index_columns else 0
        return [i + num_index_columns for (i, name) in enumerate(self._column_names) if name in set(column_names)]

    def drop_duplicates(self, subset=None, keep='first', nulls_are_equal=True, ignore_index=False):
        if False:
            return 10
        '\n        Drop duplicate rows in frame.\n\n        subset : list, optional\n            List of columns to consider when dropping rows.\n        keep : ["first", "last", False]\n            "first" will keep the first duplicate entry, "last" will keep the\n            last duplicate entry, and False will drop all duplicates.\n        nulls_are_equal: bool, default True\n            Null elements are considered equal to other null elements.\n        ignore_index: bool, default False\n            If True, the resulting axis will be labeled 0, 1, ..., n - 1.\n        '
        if not isinstance(ignore_index, (np.bool_, bool)):
            raise ValueError(f'ignore_index={ignore_index!r} must be bool, not {type(ignore_index).__name__}')
        subset = self._preprocess_subset(subset)
        subset_cols = [name for name in self._column_names if name in subset]
        if len(subset_cols) == 0:
            return self.copy(deep=True)
        keys = self._positions_from_column_names(subset, offset_by_index_columns=not ignore_index)
        return self._from_columns_like_self(libcudf.stream_compaction.drop_duplicates(list(self._columns) if ignore_index else list(self._index._columns + self._columns), keys=keys, keep=keep, nulls_are_equal=nulls_are_equal), self._column_names, self._index.names if not ignore_index else None)

    @_cudf_nvtx_annotate
    def duplicated(self, subset=None, keep='first'):
        if False:
            while True:
                i = 10
        "\n        Return boolean Series denoting duplicate rows.\n\n        Considering certain columns is optional.\n\n        Parameters\n        ----------\n        subset : column label or sequence of labels, optional\n            Only consider certain columns for identifying duplicates, by\n            default use all of the columns.\n        keep : {'first', 'last', False}, default 'first'\n            Determines which duplicates (if any) to mark.\n\n            - ``'first'`` : Mark duplicates as ``True`` except for the first\n                occurrence.\n            - ``'last'`` : Mark duplicates as ``True`` except for the last\n                occurrence.\n            - ``False`` : Mark all duplicates as ``True``.\n\n        Returns\n        -------\n        Series\n            Boolean series indicating duplicated rows.\n\n        See Also\n        --------\n        Index.duplicated : Equivalent method on index.\n        Series.duplicated : Equivalent method on Series.\n        Series.drop_duplicates : Remove duplicate values from Series.\n        DataFrame.drop_duplicates : Remove duplicate values from DataFrame.\n\n        Examples\n        --------\n        Consider a dataset containing ramen product ratings.\n\n        >>> import cudf\n        >>> df = cudf.DataFrame({\n        ...     'brand': ['Yum Yum', 'Yum Yum', 'Maggie', 'Maggie', 'Maggie'],\n        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],\n        ...     'rating': [4, 4, 3.5, 15, 5]\n        ... })\n        >>> df\n             brand style  rating\n        0  Yum Yum   cup     4.0\n        1  Yum Yum   cup     4.0\n        2   Maggie   cup     3.5\n        3   Maggie  pack    15.0\n        4   Maggie  pack     5.0\n\n        By default, for each set of duplicated values, the first occurrence\n        is set to False and all others to True.\n\n        >>> df.duplicated()\n        0    False\n        1     True\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        By using 'last', the last occurrence of each set of duplicated values\n        is set to False and all others to True.\n\n        >>> df.duplicated(keep='last')\n        0     True\n        1    False\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        By setting ``keep`` to False, all duplicates are True.\n\n        >>> df.duplicated(keep=False)\n        0     True\n        1     True\n        2    False\n        3    False\n        4    False\n        dtype: bool\n\n        To find duplicates on specific column(s), use ``subset``.\n\n        >>> df.duplicated(subset=['brand'])\n        0    False\n        1     True\n        2    False\n        3     True\n        4     True\n        dtype: bool\n        "
        subset = self._preprocess_subset(subset)
        if isinstance(self, cudf.Series):
            df = self.to_frame(name='None')
            subset = ['None']
        else:
            df = self.copy(deep=False)
        df._data['index'] = cudf.core.column.arange(0, len(self), dtype=size_type_dtype)
        new_df = df.drop_duplicates(subset=subset, keep=keep)
        idx = df.merge(new_df, how='inner')['index']
        s = cudf.Series._from_data({None: cudf.core.column.full(size=len(self), fill_value=True, dtype='bool')}, index=self.index)
        s.iloc[idx] = False
        return s

    @_cudf_nvtx_annotate
    def _empty_like(self, keep_index=True) -> Self:
        if False:
            return 10
        return self._from_columns_like_self(libcudf.copying.columns_empty_like([*(self._index._data.columns if keep_index else ()), *self._columns]), self._column_names, self._index.names if keep_index else None)

    def _split(self, splits, keep_index=True):
        if False:
            for i in range(10):
                print('nop')
        if self._num_rows == 0:
            return []
        columns_split = libcudf.copying.columns_split([*(self._index._data.columns if keep_index else []), *self._columns], splits)
        return [self._from_columns_like_self(columns_split[i], self._column_names, self._index.names if keep_index else None) for i in range(len(splits) + 1)]

    @_cudf_nvtx_annotate
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None):
        if False:
            return 10
        old_index = self._index
        ret = super().fillna(value, method, axis, inplace, limit)
        if inplace:
            self._index = old_index
        else:
            ret._index = old_index
        return ret

    @_cudf_nvtx_annotate
    def bfill(self, value=None, axis=None, inplace=None, limit=None):
        if False:
            print('Hello World!')
        "\n        Synonym for :meth:`Series.fillna` with ``method='bfill'``.\n\n        Returns\n        -------\n            Object with missing values filled or None if ``inplace=True``.\n        "
        return self.fillna(method='bfill', value=value, axis=axis, inplace=inplace, limit=limit)

    @_cudf_nvtx_annotate
    def backfill(self, value=None, axis=None, inplace=None, limit=None):
        if False:
            print('Hello World!')
        "\n        Synonym for :meth:`Series.fillna` with ``method='bfill'``.\n\n        .. deprecated:: 23.06\n           Use `DataFrame.bfill/Series.bfill` instead.\n\n        Returns\n        -------\n            Object with missing values filled or None if ``inplace=True``.\n        "
        warnings.warn('DataFrame.backfill/Series.backfill is deprecated. Use DataFrame.bfill/Series.bfill instead', FutureWarning)
        return self.bfill(value=value, axis=axis, inplace=inplace, limit=limit)

    @_cudf_nvtx_annotate
    def ffill(self, value=None, axis=None, inplace=None, limit=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Synonym for :meth:`Series.fillna` with ``method='ffill'``.\n\n        Returns\n        -------\n            Object with missing values filled or None if ``inplace=True``.\n        "
        return self.fillna(method='ffill', value=value, axis=axis, inplace=inplace, limit=limit)

    @_cudf_nvtx_annotate
    def pad(self, value=None, axis=None, inplace=None, limit=None):
        if False:
            print('Hello World!')
        "\n        Synonym for :meth:`Series.fillna` with ``method='ffill'``.\n\n        .. deprecated:: 23.06\n           Use `DataFrame.ffill/Series.ffill` instead.\n\n        Returns\n        -------\n            Object with missing values filled or None if ``inplace=True``.\n        "
        warnings.warn('DataFrame.pad/Series.pad is deprecated. Use DataFrame.ffill/Series.ffill instead', FutureWarning)
        return self.ffill(value=value, axis=axis, inplace=inplace, limit=limit)

    def add_prefix(self, prefix):
        if False:
            print('Hello World!')
        "\n        Prefix labels with string `prefix`.\n\n        For Series, the row labels are prefixed.\n        For DataFrame, the column labels are prefixed.\n\n        Parameters\n        ----------\n        prefix : str\n            The string to add before each label.\n\n        Returns\n        -------\n        Series or DataFrame\n            New Series with updated labels or DataFrame with updated labels.\n\n        See Also\n        --------\n        Series.add_suffix: Suffix row labels with string 'suffix'.\n        DataFrame.add_suffix: Suffix column labels with string 'suffix'.\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = cudf.Series([1, 2, 3, 4])\n        >>> s\n        0    1\n        1    2\n        2    3\n        3    4\n        dtype: int64\n        >>> s.add_prefix('item_')\n        item_0    1\n        item_1    2\n        item_2    3\n        item_3    4\n        dtype: int64\n\n        **DataFrame**\n\n        >>> df = cudf.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})\n        >>> df\n           A  B\n        0  1  3\n        1  2  4\n        2  3  5\n        3  4  6\n        >>> df.add_prefix('col_')\n             col_A  col_B\n        0       1       3\n        1       2       4\n        2       3       5\n        3       4       6\n        "
        raise NotImplementedError('`IndexedFrame.add_prefix` not currently implemented.                 Use `Series.add_prefix` or `DataFrame.add_prefix`')

    def add_suffix(self, suffix):
        if False:
            i = 10
            return i + 15
        "\n        Suffix labels with string `suffix`.\n\n        For Series, the row labels are suffixed.\n        For DataFrame, the column labels are suffixed.\n\n        Parameters\n        ----------\n        prefix : str\n            The string to add after each label.\n\n        Returns\n        -------\n        Series or DataFrame\n            New Series with updated labels or DataFrame with updated labels.\n\n        See Also\n        --------\n        Series.add_prefix: prefix row labels with string 'prefix'.\n        DataFrame.add_prefix: Prefix column labels with string 'prefix'.\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = cudf.Series([1, 2, 3, 4])\n        >>> s\n        0    1\n        1    2\n        2    3\n        3    4\n        dtype: int64\n        >>> s.add_suffix('_item')\n        0_item    1\n        1_item    2\n        2_item    3\n        3_item    4\n        dtype: int64\n\n        **DataFrame**\n\n        >>> df = cudf.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})\n        >>> df\n           A  B\n        0  1  3\n        1  2  4\n        2  3  5\n        3  4  6\n        >>> df.add_suffix('_col')\n             A_col  B_col\n        0       1       3\n        1       2       4\n        2       3       5\n        3       4       6\n        "
        raise NotImplementedError

    @acquire_spill_lock()
    @_cudf_nvtx_annotate
    def _apply(self, func, kernel_getter, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Apply `func` across the rows of the frame.'
        if kwargs:
            raise ValueError('UDFs using **kwargs are not yet supported.')
        try:
            (kernel, retty) = _compile_or_get(self, func, args, kernel_getter=kernel_getter)
        except Exception as e:
            raise ValueError('user defined function compilation failed.') from e
        ans_col = _return_arr_from_dtype(retty, len(self))
        ans_mask = cudf.core.column.full(size=len(self), fill_value=True, dtype='bool')
        output_args = [(ans_col, ans_mask), len(self)]
        input_args = _get_input_args_from_frame(self)
        launch_args = output_args + input_args + list(args)
        try:
            with _CUDFNumbaConfig():
                kernel.forall(len(self))(*launch_args)
        except Exception as e:
            raise RuntimeError('UDF kernel execution failed.') from e
        col = _post_process_output_col(ans_col, retty)
        col.set_base_mask(libcudf.transform.bools_to_mask(ans_mask))
        result = cudf.Series._from_data({None: col}, self._index)
        return result

    def sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False):
        if False:
            while True:
                i = 10
        "Sort by the values along either axis.\n\n        Parameters\n        ----------\n        by : str or list of str\n            Name or list of names to sort by.\n        ascending : bool or list of bool, default True\n            Sort ascending vs. descending. Specify list for multiple sort\n            orders. If this is a list of bools, must match the length of the\n            by.\n        na_position : {'first', 'last'}, default 'last'\n            'first' puts nulls at the beginning, 'last' puts nulls at the end\n        ignore_index : bool, default False\n            If True, index will not be sorted.\n\n        Returns\n        -------\n        Frame : Frame with sorted values.\n\n        Notes\n        -----\n        Difference from pandas:\n          * Support axis='index' only.\n          * Not supporting: inplace, kind\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame()\n        >>> df['a'] = [0, 1, 2]\n        >>> df['b'] = [-3, 2, 0]\n        >>> df.sort_values('b')\n           a  b\n        0  0 -3\n        2  2  0\n        1  1  2\n        "
        if na_position not in {'first', 'last'}:
            raise ValueError(f'invalid na_position: {na_position}')
        if inplace:
            raise NotImplementedError('`inplace` not currently implemented.')
        if kind != 'quicksort':
            if kind not in {'mergesort', 'heapsort', 'stable'}:
                raise AttributeError(f"{kind} is not a valid sorting algorithm for 'DataFrame' object")
            warnings.warn(f'GPU-accelerated {kind} is currently not supported, defaulting to quicksort.')
        if axis != 0:
            raise NotImplementedError('`axis` not currently implemented.')
        if len(self) == 0:
            return self
        out = self._gather(GatherMap.from_column_unchecked(self._get_columns_by_label(by)._get_sorted_inds(ascending=ascending, na_position=na_position), len(self), nullify=False), keep_index=not ignore_index)
        if isinstance(self, cudf.core.dataframe.DataFrame) and self._data.multiindex:
            out.columns = self._data.to_pandas_index()
        return out

    def _n_largest_or_smallest(self, largest, n, columns, keep):
        if False:
            print('Hello World!')
        if isinstance(columns, str):
            columns = [columns]
        method = 'nlargest' if largest else 'nsmallest'
        for col in columns:
            if isinstance(self._data[col], cudf.core.column.StringColumn):
                if isinstance(self, cudf.DataFrame):
                    error_msg = f"Column '{col}' has dtype {self._data[col].dtype}, cannot use method '{method}' with this dtype"
                else:
                    error_msg = f"Cannot use method '{method}' with dtype {self._data[col].dtype}"
                raise TypeError(error_msg)
        if len(self) == 0:
            return self
        if keep == 'first':
            if n < 0:
                n = 0
            return self._gather(GatherMap.from_column_unchecked(self._get_columns_by_label(columns)._get_sorted_inds(ascending=not largest).slice(*slice(None, n).indices(len(self))), len(self), nullify=False), keep_index=True)
        elif keep == 'last':
            indices = self._get_columns_by_label(columns)._get_sorted_inds(ascending=largest)
            if n <= 0:
                indices = indices.slice(0, 0)
            else:
                indices = indices.slice(*slice(None, -n - 1, -1).indices(len(self)))
            return self._gather(GatherMap.from_column_unchecked(indices, len(self), nullify=False), keep_index=True)
        else:
            raise ValueError('keep must be either "first", "last"')

    def _align_to_index(self, index: ColumnLike, how: str='outer', sort: bool=True, allow_non_unique: bool=False) -> Self:
        if False:
            while True:
                i = 10
        index = cudf.core.index.as_index(index)
        if self.index.equals(index):
            return self
        if not allow_non_unique:
            if not self.index.is_unique or not index.is_unique:
                raise ValueError('Cannot align indices with non-unique values')
        lhs = cudf.DataFrame._from_data(self._data, index=self.index)
        rhs = cudf.DataFrame._from_data({}, index=index)
        sort_col_id = str(uuid4())
        if how == 'left':
            lhs[sort_col_id] = cudf.core.column.arange(len(lhs))
        elif how == 'right':
            rhs[sort_col_id] = cudf.core.column.arange(len(rhs))
        result = lhs.join(rhs, how=how, sort=sort)
        if how in ('left', 'right'):
            result = result.sort_values(sort_col_id)
            del result[sort_col_id]
        result = self.__class__._from_data(data=result._data, index=result.index)
        result._data.multiindex = self._data.multiindex
        result._data._level_names = self._data._level_names
        result.index.names = self.index.names
        return result

    @_cudf_nvtx_annotate
    def _reindex(self, column_names, dtypes=None, deep=False, index=None, inplace=False, fill_value=NA):
        if False:
            while True:
                i = 10
        '\n        Helper for `.reindex`\n\n        Parameters\n        ----------\n        columns_names : array-like\n            The list of columns to select from the Frame,\n            if ``columns`` is a superset of ``Frame.columns`` new\n            columns are created.\n        dtypes : dict\n            Mapping of dtypes for the empty columns being created.\n        deep : boolean, optional, default False\n            Whether to make deep copy or shallow copy of the columns.\n        index : Index or array-like, default None\n            The ``index`` to be used to reindex the Frame with.\n        inplace : bool, default False\n            Whether to perform the operation in place on the data.\n        fill_value : value with which to replace nulls in the result\n\n        Returns\n        -------\n        Series or DataFrame\n        '
        if dtypes is None:
            dtypes = {}
        df = self
        if index is not None:
            if not df._index.is_unique:
                raise ValueError('cannot reindex on an axis with duplicate labels')
            index = cudf.core.index.as_index(index, name=getattr(index, 'name', self._index.name))
            idx_dtype_match = df.index.nlevels == index.nlevels and all((_is_same_dtype(left_dtype, right_dtype) for (left_dtype, right_dtype) in zip((col.dtype for col in df.index._data.columns), (col.dtype for col in index._data.columns))))
            if not idx_dtype_match:
                column_names = column_names if column_names is not None else list(df._column_names)
                df = cudf.DataFrame()
            else:
                lhs = cudf.DataFrame._from_data({}, index=index)
                rhs = cudf.DataFrame._from_data({name or 0 if isinstance(self, cudf.Series) else name: col for (name, col) in df._data.items()}, index=df._index)
                df = lhs.join(rhs, how='left', sort=True)
                df = df.take(index.argsort(ascending=True).argsort())
        index = index if index is not None else df.index
        names = column_names if column_names is not None else list(df._data.names)
        cols = {name: df._data[name].copy(deep=deep) if name in df._data else cudf.core.column.column.column_empty(dtype=dtypes.get(name, np.float64), masked=True, row_count=len(index)) for name in names}
        result = self.__class__._from_data(data=cudf.core.column_accessor.ColumnAccessor(cols, multiindex=self._data.multiindex, level_names=tuple(column_names.names) if isinstance(column_names, pd.Index) else None), index=index)
        result.fillna(fill_value, inplace=True)
        return self._mimic_inplace(result, inplace=inplace)

    def round(self, decimals=0, how='half_even'):
        if False:
            return 10
        '\n        Round to a variable number of decimal places.\n\n        Parameters\n        ----------\n        decimals : int, dict, Series\n            Number of decimal places to round each column to. This parameter\n            must be an int for a Series. For a DataFrame, a dict or a Series\n            are also valid inputs. If an int is given, round each column to the\n            same number of places. Otherwise dict and Series round to variable\n            numbers of places. Column names should be in the keys if\n            `decimals` is a dict-like, or in the index if `decimals` is a\n            Series. Any columns not included in `decimals` will be left as is.\n            Elements of `decimals` which are not columns of the input will be\n            ignored.\n        how : str, optional\n            Type of rounding. Can be either "half_even" (default)\n            or "half_up" rounding.\n\n        Returns\n        -------\n        Series or DataFrame\n            A Series or DataFrame with the affected columns rounded to the\n            specified number of decimal places.\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = cudf.Series([0.1, 1.4, 2.9])\n        >>> s.round()\n        0    0.0\n        1    1.0\n        2    3.0\n        dtype: float64\n\n        **DataFrame**\n\n        >>> df = cudf.DataFrame(\n        ...     [(.21, .32), (.01, .67), (.66, .03), (.21, .18)],\n        ...     columns=[\'dogs\', \'cats\'],\n        ... )\n        >>> df\n           dogs  cats\n        0  0.21  0.32\n        1  0.01  0.67\n        2  0.66  0.03\n        3  0.21  0.18\n\n        By providing an integer each column is rounded to the same number\n        of decimal places.\n\n        >>> df.round(1)\n           dogs  cats\n        0   0.2   0.3\n        1   0.0   0.7\n        2   0.7   0.0\n        3   0.2   0.2\n\n        With a dict, the number of places for specific columns can be\n        specified with the column names as keys and the number of decimal\n        places as values.\n\n        >>> df.round({\'dogs\': 1, \'cats\': 0})\n           dogs  cats\n        0   0.2   0.0\n        1   0.0   1.0\n        2   0.7   0.0\n        3   0.2   0.0\n\n        Using a Series, the number of places for specific columns can be\n        specified with the column names as the index and the number of\n        decimal places as the values.\n\n        >>> decimals = cudf.Series([0, 1], index=[\'cats\', \'dogs\'])\n        >>> df.round(decimals)\n           dogs  cats\n        0   0.2   0.0\n        1   0.0   1.0\n        2   0.7   0.0\n        3   0.2   0.0\n        '
        if isinstance(decimals, cudf.Series):
            decimals = decimals.to_pandas()
        if isinstance(decimals, pd.Series):
            if not decimals.index.is_unique:
                raise ValueError('Index of decimals must be unique')
            decimals = decimals.to_dict()
        elif isinstance(decimals, int):
            decimals = {name: decimals for name in self._column_names}
        elif not isinstance(decimals, abc.Mapping):
            raise TypeError('decimals must be an integer, a dict-like or a Series')
        cols = {name: col.round(decimals[name], how=how) if name in decimals and _is_non_decimal_numeric_dtype(col.dtype) and (not is_bool_dtype(col.dtype)) else col.copy(deep=True) for (name, col) in self._data.items()}
        return self.__class__._from_data(data=cudf.core.column_accessor.ColumnAccessor(cols, multiindex=self._data.multiindex, level_names=self._data.level_names), index=self._index)

    def resample(self, rule, axis=0, closed=None, label=None, convention='start', kind=None, loffset=None, base=None, on=None, level=None, origin='start_day', offset=None):
        if False:
            while True:
                i = 10
        '\n        Convert the frequency of ("resample") the given time series data.\n\n        Parameters\n        ----------\n        rule: str\n            The offset string representing the frequency to use.\n            Note that DateOffset objects are not yet supported.\n        closed: {"right", "left"}, default None\n            Which side of bin interval is closed. The default is\n            "left" for all frequency offsets except for "M" and "W",\n            which have a default of "right".\n        label: {"right", "left"}, default None\n            Which bin edge label to label bucket with. The default is\n            "left" for all frequency offsets except for "M" and "W",\n            which have a default of "right".\n        on: str, optional\n            For a DataFrame, column to use instead of the index for\n            resampling.  Column must be a datetime-like.\n        level: str or int, optional\n            For a MultiIndex, level to use instead of the index for\n            resampling.  The level must be a datetime-like.\n\n        Returns\n        -------\n        A Resampler object\n\n        Examples\n        --------\n        First, we create a time series with 1 minute intervals:\n\n        >>> index = cudf.date_range(start="2001-01-01", periods=10, freq="1T")\n        >>> sr = cudf.Series(range(10), index=index)\n        >>> sr\n        2001-01-01 00:00:00    0\n        2001-01-01 00:01:00    1\n        2001-01-01 00:02:00    2\n        2001-01-01 00:03:00    3\n        2001-01-01 00:04:00    4\n        2001-01-01 00:05:00    5\n        2001-01-01 00:06:00    6\n        2001-01-01 00:07:00    7\n        2001-01-01 00:08:00    8\n        2001-01-01 00:09:00    9\n        dtype: int64\n\n        Downsampling to 3 minute intervals, followed by a "sum" aggregation:\n\n        >>> sr.resample("3T").sum()\n        2001-01-01 00:00:00     3\n        2001-01-01 00:03:00    12\n        2001-01-01 00:06:00    21\n        2001-01-01 00:09:00     9\n        dtype: int64\n\n        Use the right side of each interval to label the bins:\n\n        >>> sr.resample("3T", label="right").sum()\n        2001-01-01 00:03:00     3\n        2001-01-01 00:06:00    12\n        2001-01-01 00:09:00    21\n        2001-01-01 00:12:00     9\n        dtype: int64\n\n        Close the right side of the interval instead of the left:\n\n        >>> sr.resample("3T", closed="right").sum()\n        2000-12-31 23:57:00     0\n        2001-01-01 00:00:00     6\n        2001-01-01 00:03:00    15\n        2001-01-01 00:06:00    24\n        dtype: int64\n\n        Upsampling to 30 second intervals:\n\n        >>> sr.resample("30s").asfreq()[:5]  # show the first 5 rows\n        2001-01-01 00:00:00       0\n        2001-01-01 00:00:30    <NA>\n        2001-01-01 00:01:00       1\n        2001-01-01 00:01:30    <NA>\n        2001-01-01 00:02:00       2\n        dtype: int64\n\n        Upsample and fill nulls using the "bfill" method:\n\n        >>> sr.resample("30s").bfill()[:5]\n        2001-01-01 00:00:00    0\n        2001-01-01 00:00:30    1\n        2001-01-01 00:01:00    1\n        2001-01-01 00:01:30    2\n        2001-01-01 00:02:00    2\n        dtype: int64\n\n        Resampling by a specified column of a Dataframe:\n\n        >>> df = cudf.DataFrame({\n        ...     "price": [10, 11, 9, 13, 14, 18, 17, 19],\n        ...     "volume": [50, 60, 40, 100, 50, 100, 40, 50],\n        ...     "week_starting": cudf.date_range(\n        ...         "2018-01-01", periods=8, freq="7D"\n        ...     )\n        ... })\n        >>> df\n        price  volume week_starting\n        0     10      50    2018-01-01\n        1     11      60    2018-01-08\n        2      9      40    2018-01-15\n        3     13     100    2018-01-22\n        4     14      50    2018-01-29\n        5     18     100    2018-02-05\n        6     17      40    2018-02-12\n        7     19      50    2018-02-19\n        >>> df.resample("M", on="week_starting").mean()\n                       price     volume\n        week_starting\n        2018-01-31      11.4  60.000000\n        2018-02-28      18.0  63.333333\n\n\n        Notes\n        -----\n        Note that the dtype of the index (or the \'on\' column if using\n        \'on=\') in the result will be of a frequency closest to the\n        resampled frequency.  For example, if resampling from\n        nanoseconds to milliseconds, the index will be of dtype\n        \'datetime64[ms]\'.\n        '
        import cudf.core.resample
        if (axis, convention, kind, loffset, base, origin, offset) != (0, 'start', None, None, None, 'start_day', None):
            raise NotImplementedError('The following arguments are not currently supported by resample:\n\n- axis\n- convention\n- kind\n- loffset\n- base\n- origin\n- offset')
        by = cudf.Grouper(key=on, freq=rule, closed=closed, label=label, level=level)
        return cudf.core.resample.SeriesResampler(self, by=by) if isinstance(self, cudf.Series) else cudf.core.resample.DataFrameResampler(self, by=by)

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        if False:
            return 10
        '\n        Drop rows (or columns) containing nulls from a Column.\n\n        Parameters\n        ----------\n        axis : {0, 1}, optional\n            Whether to drop rows (axis=0, default) or columns (axis=1)\n            containing nulls.\n        how : {"any", "all"}, optional\n            Specifies how to decide whether to drop a row (or column).\n            any (default) drops rows (or columns) containing at least\n            one null value. all drops only rows (or columns) containing\n            *all* null values.\n        thresh: int, optional\n            If specified, then drops every row (or column) containing\n            less than `thresh` non-null values\n        subset : list, optional\n            List of columns to consider when dropping rows (all columns\n            are considered by default). Alternatively, when dropping\n            columns, subset is a list of rows to consider.\n        inplace : bool, default False\n            If True, do operation inplace and return None.\n\n        Returns\n        -------\n        Copy of the DataFrame with rows/columns containing nulls dropped.\n\n        See Also\n        --------\n        cudf.DataFrame.isna\n            Indicate null values.\n        cudf.DataFrame.notna\n            Indicate non-null values.\n        cudf.DataFrame.fillna\n            Replace null values.\n        cudf.Series.dropna\n            Drop null values.\n        cudf.Index.dropna\n            Drop null indices.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame({"name": [\'Alfred\', \'Batman\', \'Catwoman\'],\n        ...                    "toy": [\'Batmobile\', None, \'Bullwhip\'],\n        ...                    "born": [np.datetime64("1940-04-25"),\n        ...                             np.datetime64("NaT"),\n        ...                             np.datetime64("NaT")]})\n        >>> df\n               name        toy                 born\n        0    Alfred  Batmobile  1940-04-25 00:00:00\n        1    Batman       <NA>                 <NA>\n        2  Catwoman   Bullwhip                 <NA>\n\n        Drop the rows where at least one element is null.\n\n        >>> df.dropna()\n             name        toy       born\n        0  Alfred  Batmobile 1940-04-25\n\n        Drop the columns where at least one element is null.\n\n        >>> df.dropna(axis=\'columns\')\n               name\n        0    Alfred\n        1    Batman\n        2  Catwoman\n\n        Drop the rows where all elements are null.\n\n        >>> df.dropna(how=\'all\')\n               name        toy                 born\n        0    Alfred  Batmobile  1940-04-25 00:00:00\n        1    Batman       <NA>                 <NA>\n        2  Catwoman   Bullwhip                 <NA>\n\n        Keep only the rows with at least 2 non-null values.\n\n        >>> df.dropna(thresh=2)\n               name        toy                 born\n        0    Alfred  Batmobile  1940-04-25 00:00:00\n        2  Catwoman   Bullwhip                 <NA>\n\n        Define in which columns to look for null values.\n\n        >>> df.dropna(subset=[\'name\', \'born\'])\n             name        toy       born\n        0  Alfred  Batmobile 1940-04-25\n\n        Keep the DataFrame with valid entries in the same variable.\n\n        >>> df.dropna(inplace=True)\n        >>> df\n             name        toy       born\n        0  Alfred  Batmobile 1940-04-25\n        '
        if axis == 0:
            result = self._drop_na_rows(how=how, subset=subset, thresh=thresh)
        else:
            result = self._drop_na_columns(how=how, subset=subset, thresh=thresh)
        return self._mimic_inplace(result, inplace=inplace)

    def _drop_na_rows(self, how='any', subset=None, thresh=None):
        if False:
            print('Hello World!')
        '\n        Drop null rows from `self`.\n\n        how : {"any", "all"}, optional\n            Specifies how to decide whether to drop a row.\n            any (default) drops rows containing at least\n            one null value. all drops only rows containing\n            *all* null values.\n        subset : list, optional\n            List of columns to consider when dropping rows.\n        thresh : int, optional\n            If specified, then drops every row containing\n            less than `thresh` non-null values.\n        '
        subset = self._preprocess_subset(subset)
        if len(subset) == 0:
            return self.copy(deep=True)
        data_columns = [col.nans_to_nulls() if isinstance(col, cudf.core.column.NumericalColumn) else col for col in self._columns]
        return self._from_columns_like_self(libcudf.stream_compaction.drop_nulls([*self._index._data.columns, *data_columns], how=how, keys=self._positions_from_column_names(subset, offset_by_index_columns=True), thresh=thresh), self._column_names, self._index.names)

    def _apply_boolean_mask(self, boolean_mask: BooleanMask, keep_index=True):
        if False:
            while True:
                i = 10
        'Apply boolean mask to each row of `self`.\n\n        Rows corresponding to `False` is dropped.\n\n        If keep_index is False, the index is not preserved.\n        '
        if len(boolean_mask.column) != len(self):
            raise IndexError(f'Boolean mask has wrong length: {len(boolean_mask.column)} not {len(self)}')
        return self._from_columns_like_self(libcudf.stream_compaction.apply_boolean_mask(list(self._index._columns + self._columns) if keep_index else list(self._columns), boolean_mask.column), column_names=self._column_names, index_names=self._index.names if keep_index else None)

    def take(self, indices, axis=0):
        if False:
            i = 10
            return i + 15
        "Return a new frame containing the rows specified by *indices*.\n\n        Parameters\n        ----------\n        indices : array-like\n            Array of ints indicating which positions to take.\n        axis : Unsupported\n\n        Returns\n        -------\n        out : Series or DataFrame\n            New object with desired subset of rows.\n\n        Examples\n        --------\n        **Series**\n        >>> s = cudf.Series(['a', 'b', 'c', 'd', 'e'])\n        >>> s.take([2, 0, 4, 3])\n        2    c\n        0    a\n        4    e\n        3    d\n        dtype: object\n\n        **DataFrame**\n\n        >>> a = cudf.DataFrame({'a': [1.0, 2.0, 3.0],\n        ...                    'b': cudf.Series(['a', 'b', 'c'])})\n        >>> a.take([0, 2, 2])\n             a  b\n        0  1.0  a\n        2  3.0  c\n        2  3.0  c\n        >>> a.take([True, False, True])\n             a  b\n        0  1.0  a\n        2  3.0  c\n        "
        if self._get_axis_from_axis_arg(axis) != 0:
            raise NotImplementedError('Only axis=0 is supported.')
        return self._gather(GatherMap(indices, len(self), nullify=False))

    def _reset_index(self, level, drop, col_level=0, col_fill=''):
        if False:
            return 10
        'Shared path for DataFrame.reset_index and Series.reset_index.'
        if level is not None and (not isinstance(level, (tuple, list))):
            level = (level,)
        _check_duplicate_level_names(level, self._index.names)
        (data_columns, index_columns, data_names, index_names) = self._index._split_columns_by_levels(level)
        if index_columns:
            index = _index_from_columns(index_columns, name=self._index.name)
            if isinstance(index, MultiIndex):
                index.names = index_names
            else:
                index.name = index_names[0]
        else:
            index = RangeIndex(len(self))
        if drop:
            return (self._data, index)
        new_column_data = {}
        for (name, col) in zip(data_names, data_columns):
            if name == 'index' and 'index' in self._data:
                name = 'level_0'
            name = tuple((name if i == col_level else col_fill for i in range(self._data.nlevels))) if self._data.multiindex else name
            new_column_data[name] = col
        return (ColumnAccessor({**new_column_data, **self._data}, self._data.multiindex, self._data._level_names), index)

    def _first_or_last(self, offset, idx: int, op: Callable, side: str, slice_func: Callable) -> 'IndexedFrame':
        if False:
            while True:
                i = 10
        'Shared code path for ``first`` and ``last``.'
        if not isinstance(self._index, cudf.core.index.DatetimeIndex):
            raise TypeError("'first' only supports a DatetimeIndex index.")
        if not isinstance(offset, str):
            raise NotImplementedError(f'Unsupported offset type {type(offset)}.')
        if len(self) == 0:
            return self.copy()
        pd_offset = pd.tseries.frequencies.to_offset(offset)
        to_search = op(pd.Timestamp(self._index._column.element_indexing(idx)), pd_offset)
        if idx == 0 and (not isinstance(pd_offset, pd.tseries.offsets.Tick)) and pd_offset.is_on_offset(pd.Timestamp(self._index[0])):
            to_search = to_search - pd_offset.base
            return self.loc[:to_search]
        end_point = int(self._index._column.searchsorted(to_search, side=side)[0])
        return slice_func(end_point)

    def first(self, offset):
        if False:
            i = 10
            return i + 15
        "Select initial periods of time series data based on a date offset.\n\n        When having a DataFrame with **sorted** dates as index, this function\n        can select the first few rows based on a date offset.\n\n        Parameters\n        ----------\n        offset: str\n            The offset length of the data that will be selected. For instance,\n            '1M' will display all rows having their index within the first\n            month.\n\n        Returns\n        -------\n        Series or DataFrame\n            A subset of the caller.\n\n        Raises\n        ------\n        TypeError\n            If the index is not a ``DatetimeIndex``\n\n        Examples\n        --------\n        >>> i = cudf.date_range('2018-04-09', periods=4, freq='2D')\n        >>> ts = cudf.DataFrame({'A': [1, 2, 3, 4]}, index=i)\n        >>> ts\n                    A\n        2018-04-09  1\n        2018-04-11  2\n        2018-04-13  3\n        2018-04-15  4\n        >>> ts.first('3D')\n                    A\n        2018-04-09  1\n        2018-04-11  2\n        "
        return self._first_or_last(offset, idx=0, op=operator.__add__, side='left', slice_func=lambda i: self.iloc[:i])

    def last(self, offset):
        if False:
            while True:
                i = 10
        "Select final periods of time series data based on a date offset.\n\n        When having a DataFrame with **sorted** dates as index, this function\n        can select the last few rows based on a date offset.\n\n        Parameters\n        ----------\n        offset: str\n            The offset length of the data that will be selected. For instance,\n            '3D' will display all rows having their index within the last 3\n            days.\n\n        Returns\n        -------\n        Series or DataFrame\n            A subset of the caller.\n\n        Raises\n        ------\n        TypeError\n            If the index is not a ``DatetimeIndex``\n\n        Examples\n        --------\n        >>> i = cudf.date_range('2018-04-09', periods=4, freq='2D')\n        >>> ts = cudf.DataFrame({'A': [1, 2, 3, 4]}, index=i)\n        >>> ts\n                    A\n        2018-04-09  1\n        2018-04-11  2\n        2018-04-13  3\n        2018-04-15  4\n        >>> ts.last('3D')\n                    A\n        2018-04-13  3\n        2018-04-15  4\n        "
        return self._first_or_last(offset, idx=-1, op=operator.__sub__, side='right', slice_func=lambda i: self.iloc[i:])

    @_cudf_nvtx_annotate
    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None, axis=None, ignore_index=False):
        if False:
            for i in range(10):
                print('nop')
        'Return a random sample of items from an axis of object.\n\n        If reproducible results are required, a random number generator may be\n        provided via the `random_state` parameter. This function will always\n        produce the same sample given an identical `random_state`.\n\n        Notes\n        -----\n        When sampling from ``axis=0/\'index\'``, ``random_state`` can be either\n        a numpy random state (``numpy.random.RandomState``) or a cupy random\n        state (``cupy.random.RandomState``). When a numpy random state is\n        used, the output is guaranteed to match the output of the corresponding\n        pandas method call, but generating the sample may be slow. If exact\n        pandas equivalence is not required, using a cupy random state will\n        achieve better performance, especially when sampling large number of\n        items. It\'s advised to use the matching `ndarray` type to the random\n        state for the `weights` array.\n\n        Parameters\n        ----------\n        n : int, optional\n            Number of items from axis to return. Cannot be used with `frac`.\n            Default = 1 if frac = None.\n        frac : float, optional\n            Fraction of axis items to return. Cannot be used with n.\n        replace : bool, default False\n            Allow or disallow sampling of the same row more than once.\n            `replace == True` is not supported for axis = 1/"columns".\n            `replace == False` is not supported for axis = 0/"index" given\n            `random_state` is `None` or a cupy random state, and `weights` is\n            specified.\n        weights : ndarray-like, optional\n            Default `None` for uniform probability distribution over rows to\n            sample from. If `ndarray` is passed, the length of `weights` should\n            equal to the number of rows to sample from, and will be normalized\n            to have a sum of 1. Unlike pandas, index alignment is not currently\n            not performed.\n        random_state : int, numpy/cupy RandomState, or None, default None\n            If None, default cupy random state is chosen.\n            If int, the seed for the default cupy random state.\n            If RandomState, rows-to-sample are generated from the RandomState.\n        axis : {0 or `index`, 1 or `columns`, None}, default None\n            Axis to sample. Accepts axis number or name.\n            Default is stat axis for given data type\n            (0 for Series and DataFrames). Series doesn\'t support axis=1.\n        ignore_index : bool, default False\n            If True, the resulting index will be labeled 0, 1, …, n - 1.\n\n        Returns\n        -------\n        Series or DataFrame\n            A new object of same type as caller containing n items\n            randomly sampled from the caller object.\n\n        Examples\n        --------\n        >>> import cudf as cudf\n        >>> df = cudf.DataFrame({"a":{1, 2, 3, 4, 5}})\n        >>> df.sample(3)\n           a\n        1  2\n        3  4\n        0  1\n\n        >>> sr = cudf.Series([1, 2, 3, 4, 5])\n        >>> sr.sample(10, replace=True)\n        1    4\n        3    1\n        2    4\n        0    5\n        0    1\n        4    5\n        4    1\n        0    2\n        0    3\n        3    2\n        dtype: int64\n\n        >>> df = cudf.DataFrame(\n        ...     {"a": [1, 2], "b": [2, 3], "c": [3, 4], "d": [4, 5]}\n        ... )\n        >>> df.sample(2, axis=1)\n           a  c\n        0  1  3\n        1  2  4\n        '
        axis = 0 if axis is None else self._get_axis_from_axis_arg(axis)
        size = self.shape[axis]
        if frac is None:
            n = 1 if n is None else n
        else:
            if frac > 1 and (not replace):
                raise ValueError('Replace has to be set to `True` when upsampling the population `frac` > 1.')
            if n is not None:
                raise ValueError('Please enter a value for `frac` OR `n`, not both.')
            n = int(round(size * frac))
        if n > 0 and size == 0:
            raise ValueError('Cannot take a sample larger than 0 when axis is empty.')
        if isinstance(random_state, cp.random.RandomState):
            lib = cp
        elif isinstance(random_state, np.random.RandomState):
            lib = np
        else:
            lib = cp if axis == 0 else np
            random_state = lib.random.RandomState(seed=random_state)
        if weights is not None:
            if isinstance(weights, str):
                raise NotImplementedError('Weights specified by string is unsupported yet.')
            if size != len(weights):
                raise ValueError('Weights and axis to be sampled must be of same length.')
            weights = lib.asarray(weights)
            weights = weights / weights.sum()
        if axis == 0:
            return self._sample_axis_0(n, weights, replace, random_state, ignore_index)
        else:
            if isinstance(random_state, cp.random.RandomState):
                raise ValueError("Sampling from `axis=1`/`columns` with cupy random stateisn't supported.")
            return self._sample_axis_1(n, weights, replace, random_state, ignore_index)

    def _sample_axis_0(self, n: int, weights: Optional[ColumnLike], replace: bool, random_state: Union[np.random.RandomState, cp.random.RandomState], ignore_index: bool):
        if False:
            i = 10
            return i + 15
        try:
            gather_map = GatherMap.from_column_unchecked(cudf.core.column.as_column(random_state.choice(len(self), size=n, replace=replace, p=weights)), len(self), nullify=False)
        except NotImplementedError as e:
            raise NotImplementedError('Random sampling with cupy does not support these inputs.') from e
        return self._gather(gather_map, keep_index=not ignore_index)

    def _sample_axis_1(self, n: int, weights: Optional[ColumnLike], replace: bool, random_state: np.random.RandomState, ignore_index: bool):
        if False:
            return 10
        raise NotImplementedError(f'Sampling from axis 1 is not implemented for {self.__class__}.')

    def _binaryop(self, other: Any, op: str, fill_value: Any=None, can_reindex: bool=False, *args, **kwargs):
        if False:
            return 10
        (reflect, op) = self._check_reflected_op(op)
        (operands, out_index, can_use_self_column_name) = self._make_operands_and_index_for_binop(other, op, fill_value, reflect, can_reindex)
        if operands is NotImplemented:
            return NotImplemented
        level_names = self._data._level_names if can_use_self_column_name else None
        return self._from_data(ColumnAccessor(type(self)._colwise_binop(operands, op), level_names=level_names), index=out_index)

    def _make_operands_and_index_for_binop(self, other: Any, fn: str, fill_value: Any=None, reflect: bool=False, can_reindex: bool=False, *args, **kwargs) -> Tuple[Union[Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]], NotImplementedType], Optional[cudf.BaseIndex], bool]:
        if False:
            print('Hello World!')
        raise NotImplementedError(f'Binary operations are not supported for {self.__class__}')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if False:
            return 10
        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        fname = ufunc.__name__
        if ret is not None:
            if 'bitwise' in fname:
                reflect = self is not inputs[0]
                other = inputs[0] if reflect else inputs[1]
                if isinstance(other, self.__class__) and (not self.index.equals(other.index)):
                    ret = ret.astype(bool)
            return ret
        cupy_func = getattr(cp, fname)
        if cupy_func:
            if ufunc.nin == 2:
                other = inputs[self is inputs[0]]
                (inputs, index, _) = self._make_operands_and_index_for_binop(other, fname)
            else:
                inputs = {name: (col, None, False, None) for (name, col) in self._data.items()}
                index = self._index
            data = self._apply_cupy_ufunc_to_operands(ufunc, cupy_func, inputs, **kwargs)
            out = tuple((self._from_data(out, index=index) for out in data))
            return out[0] if ufunc.nout == 1 else out
        return NotImplemented

    @_cudf_nvtx_annotate
    def repeat(self, repeats, axis=None):
        if False:
            print('Hello World!')
        "Repeats elements consecutively.\n\n        Returns a new object of caller type(DataFrame/Series) where each\n        element of the current object is repeated consecutively a given\n        number of times.\n\n        Parameters\n        ----------\n        repeats : int, or array of ints\n            The number of repetitions for each element. This should\n            be a non-negative integer. Repeating 0 times will return\n            an empty object.\n\n        Returns\n        -------\n        Series/DataFrame\n            A newly created object of same type as caller\n            with repeated elements.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})\n        >>> df\n           a   b\n        0  1  10\n        1  2  20\n        2  3  30\n        >>> df.repeat(3)\n           a   b\n        0  1  10\n        0  1  10\n        0  1  10\n        1  2  20\n        1  2  20\n        1  2  20\n        2  3  30\n        2  3  30\n        2  3  30\n\n        Repeat on Series\n\n        >>> s = cudf.Series([0, 2])\n        >>> s\n        0    0\n        1    2\n        dtype: int64\n        >>> s.repeat([3, 4])\n        0    0\n        0    0\n        0    0\n        1    2\n        1    2\n        1    2\n        1    2\n        dtype: int64\n        >>> s.repeat(2)\n        0    0\n        0    0\n        1    2\n        1    2\n        dtype: int64\n        "
        return self._from_columns_like_self(Frame._repeat([*self._index._data.columns, *self._columns], repeats, axis), self._column_names, self._index_names)

    def _append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('The append method is deprecated and will be removed in a future version. Use cudf.concat instead.', FutureWarning)
        if verify_integrity not in (None, False):
            raise NotImplementedError('verify_integrity parameter is not supported yet.')
        if is_list_like(other):
            to_concat = [self, *other]
        else:
            to_concat = [self, other]
        return cudf.concat(to_concat, ignore_index=ignore_index, sort=sort)

    def astype(self, dtype, copy=False, errors='raise', **kwargs):
        if False:
            while True:
                i = 10
        "Cast the object to the given dtype.\n\n        Parameters\n        ----------\n        dtype : data type, or dict of column name -> data type\n            Use a :class:`numpy.dtype` or Python type to cast entire DataFrame\n            object to the same type. Alternatively, use ``{col: dtype, ...}``,\n            where col is a column label and dtype is a :class:`numpy.dtype`\n            or Python type to cast one or more of the DataFrame's columns to\n            column-specific types.\n        copy : bool, default False\n            Return a deep-copy when ``copy=True``. Note by default\n            ``copy=False`` setting is used and hence changes to\n            values then may propagate to other cudf objects.\n        errors : {'raise', 'ignore', 'warn'}, default 'raise'\n            Control raising of exceptions on invalid data for provided dtype.\n\n            -   ``raise`` : allow exceptions to be raised\n            -   ``ignore`` : suppress exceptions. On error return original\n                object.\n        **kwargs : extra arguments to pass on to the constructor\n\n        Returns\n        -------\n        DataFrame/Series\n\n        Examples\n        --------\n        **DataFrame**\n\n        >>> import cudf\n        >>> df = cudf.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})\n        >>> df\n            a  b\n        0  10  1\n        1  20  2\n        2  30  3\n        >>> df.dtypes\n        a    int64\n        b    int64\n        dtype: object\n\n        Cast all columns to `int32`:\n\n        >>> df.astype('int32').dtypes\n        a    int32\n        b    int32\n        dtype: object\n\n        Cast `a` to `float32` using a dictionary:\n\n        >>> df.astype({'a': 'float32'}).dtypes\n        a    float32\n        b      int64\n        dtype: object\n        >>> df.astype({'a': 'float32'})\n              a  b\n        0  10.0  1\n        1  20.0  2\n        2  30.0  3\n\n        **Series**\n\n        >>> import cudf\n        >>> series = cudf.Series([1, 2], dtype='int32')\n        >>> series\n        0    1\n        1    2\n        dtype: int32\n        >>> series.astype('int64')\n        0    1\n        1    2\n        dtype: int64\n\n        Convert to categorical type:\n\n        >>> series.astype('category')\n        0    1\n        1    2\n        dtype: category\n        Categories (2, int64): [1, 2]\n\n        Convert to ordered categorical type with custom ordering:\n\n        >>> cat_dtype = cudf.CategoricalDtype(categories=[2, 1], ordered=True)\n        >>> series.astype(cat_dtype)\n        0    1\n        1    2\n        dtype: category\n        Categories (2, int64): [2 < 1]\n\n        Note that using ``copy=False`` (enabled by default)\n        and changing data on a new Series will\n        propagate changes:\n\n        >>> s1 = cudf.Series([1, 2])\n        >>> s1\n        0    1\n        1    2\n        dtype: int64\n        >>> s2 = s1.astype('int64', copy=False)\n        >>> s2[0] = 10\n        >>> s1\n        0    10\n        1     2\n        dtype: int64\n        "
        if errors not in ('ignore', 'raise'):
            raise ValueError('invalid error value specified')
        try:
            data = super().astype(dtype, copy, **kwargs)
        except Exception as e:
            if errors == 'raise':
                raise e
            return self
        return self._from_data(data, index=self._index)

    @_cudf_nvtx_annotate
    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        if False:
            print('Hello World!')
        'Drop specified labels from rows or columns.\n\n        Remove rows or columns by specifying label names and corresponding\n        axis, or by specifying directly index or column names. When using a\n        multi-index, labels on different levels can be removed by specifying\n        the level.\n\n        Parameters\n        ----------\n        labels : single label or list-like\n            Index or column labels to drop.\n        axis : {0 or \'index\', 1 or \'columns\'}, default 0\n            Whether to drop labels from the index (0 or \'index\') or\n            columns (1 or \'columns\').\n        index : single label or list-like\n            Alternative to specifying axis (``labels, axis=0``\n            is equivalent to ``index=labels``).\n        columns : single label or list-like\n            Alternative to specifying axis (``labels, axis=1``\n            is equivalent to ``columns=labels``).\n        level : int or level name, optional\n            For MultiIndex, level from which the labels will be removed.\n        inplace : bool, default False\n            If False, return a copy. Otherwise, do operation\n            inplace and return None.\n        errors : {\'ignore\', \'raise\'}, default \'raise\'\n            If \'ignore\', suppress error and only existing labels are\n            dropped.\n\n        Returns\n        -------\n        DataFrame or Series\n            DataFrame or Series without the removed index or column labels.\n\n        Raises\n        ------\n        KeyError\n            If any of the labels is not found in the selected axis.\n\n        See Also\n        --------\n        DataFrame.loc : Label-location based indexer for selection by label.\n        DataFrame.dropna : Return DataFrame with labels on given axis omitted\n            where (all or any) data are missing.\n        DataFrame.drop_duplicates : Return DataFrame with duplicate rows\n            removed, optionally only considering certain columns.\n        Series.reindex\n            Return only specified index labels of Series\n        Series.dropna\n            Return series without null values\n        Series.drop_duplicates\n            Return series with duplicate values removed\n\n        Examples\n        --------\n        **Series**\n\n        >>> s = cudf.Series([1,2,3], index=[\'x\', \'y\', \'z\'])\n        >>> s\n        x    1\n        y    2\n        z    3\n        dtype: int64\n\n        Drop labels x and z\n\n        >>> s.drop(labels=[\'x\', \'z\'])\n        y    2\n        dtype: int64\n\n        Drop a label from the second level in MultiIndex Series.\n\n        >>> midx = cudf.MultiIndex.from_product([[0, 1, 2], [\'x\', \'y\']])\n        >>> s = cudf.Series(range(6), index=midx)\n        >>> s\n        0  x    0\n           y    1\n        1  x    2\n           y    3\n        2  x    4\n           y    5\n        dtype: int64\n        >>> s.drop(labels=\'y\', level=1)\n        0  x    0\n        1  x    2\n        2  x    4\n        Name: 2, dtype: int64\n\n        **DataFrame**\n\n        >>> import cudf\n        >>> df = cudf.DataFrame({"A": [1, 2, 3, 4],\n        ...                      "B": [5, 6, 7, 8],\n        ...                      "C": [10, 11, 12, 13],\n        ...                      "D": [20, 30, 40, 50]})\n        >>> df\n           A  B   C   D\n        0  1  5  10  20\n        1  2  6  11  30\n        2  3  7  12  40\n        3  4  8  13  50\n\n        Drop columns\n\n        >>> df.drop([\'B\', \'C\'], axis=1)\n           A   D\n        0  1  20\n        1  2  30\n        2  3  40\n        3  4  50\n        >>> df.drop(columns=[\'B\', \'C\'])\n           A   D\n        0  1  20\n        1  2  30\n        2  3  40\n        3  4  50\n\n        Drop a row by index\n\n        >>> df.drop([0, 1])\n           A  B   C   D\n        2  3  7  12  40\n        3  4  8  13  50\n\n        Drop columns and/or rows of MultiIndex DataFrame\n\n        >>> midx = cudf.MultiIndex(levels=[[\'lama\', \'cow\', \'falcon\'],\n        ...                              [\'speed\', \'weight\', \'length\']],\n        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],\n        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])\n        >>> df = cudf.DataFrame(index=midx, columns=[\'big\', \'small\'],\n        ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],\n        ...                         [250, 150], [1.5, 0.8], [320, 250],\n        ...                         [1, 0.8], [0.3, 0.2]])\n        >>> df\n                         big  small\n        lama   speed    45.0   30.0\n               weight  200.0  100.0\n               length    1.5    1.0\n        cow    speed    30.0   20.0\n               weight  250.0  150.0\n               length    1.5    0.8\n        falcon speed   320.0  250.0\n               weight    1.0    0.8\n               length    0.3    0.2\n        >>> df.drop(index=\'cow\', columns=\'small\')\n                         big\n        lama   speed    45.0\n               weight  200.0\n               length    1.5\n        falcon speed   320.0\n               weight    1.0\n               length    0.3\n        >>> df.drop(index=\'length\', level=1)\n                         big  small\n        lama   speed    45.0   30.0\n               weight  200.0  100.0\n        cow    speed    30.0   20.0\n               weight  250.0  150.0\n        falcon speed   320.0  250.0\n               weight    1.0    0.8\n        '
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            target = labels
        elif index is not None:
            target = index
            axis = 0
        elif columns is not None:
            target = columns
            axis = 1
        else:
            raise ValueError("Need to specify at least one of 'labels', 'index' or 'columns'")
        if inplace:
            out = self
        else:
            out = self.copy()
        if axis in (1, 'columns'):
            target = _get_host_unique(target)
            _drop_columns(out, target, errors)
        elif axis in (0, 'index'):
            dropped = _drop_rows_by_labels(out, target, level, errors)
            if columns is not None:
                columns = _get_host_unique(columns)
                _drop_columns(dropped, columns, errors)
            out._data = dropped._data
            out._index = dropped._index
        if not inplace:
            return out

    @_cudf_nvtx_annotate
    def _explode(self, explode_column: Any, ignore_index: bool):
        if False:
            while True:
                i = 10
        if not is_list_dtype(self._data[explode_column].dtype):
            data = self._data.copy(deep=True)
            idx = None if ignore_index else self._index.copy(deep=True)
            return self.__class__._from_data(data, index=idx)
        column_index = self._column_names.index(explode_column)
        if not ignore_index and self._index is not None:
            index_offset = self._index.nlevels
        else:
            index_offset = 0
        exploded = libcudf.lists.explode_outer([*(self._index._data.columns if not ignore_index else ()), *self._columns], column_index + index_offset)
        exploded_dtype = cast(ListDtype, self._columns[column_index].dtype).element_type
        return self._from_columns_like_self(exploded, self._column_names, self._index_names if not ignore_index else None, override_dtypes=(exploded_dtype if i == column_index else None for i in range(len(self._columns))))

    @_cudf_nvtx_annotate
    def tile(self, count):
        if False:
            while True:
                i = 10
        'Repeats the rows `count` times to form a new Frame.\n\n        Parameters\n        ----------\n        self : input Table containing columns to interleave.\n        count : Number of times to tile "rows". Must be non-negative.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df  = cudf.Dataframe([[8, 4, 7], [5, 2, 3]])\n        >>> count = 2\n        >>> df.tile(df, count)\n           0  1  2\n        0  8  4  7\n        1  5  2  3\n        0  8  4  7\n        1  5  2  3\n\n        Returns\n        -------\n        The indexed frame containing the tiled "rows".\n        '
        return self._from_columns_like_self(libcudf.reshape.tile([*self._index._columns, *self._columns], count), column_names=self._column_names, index_names=self._index_names)

    @_cudf_nvtx_annotate
    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=no_default, group_keys=False, squeeze=False, observed=True, dropna=True):
        if False:
            while True:
                i = 10
        if sort is no_default:
            sort = cudf.get_option('mode.pandas_compatible')
        if axis not in (0, 'index'):
            raise NotImplementedError('axis parameter is not yet implemented')
        if squeeze is not False:
            raise NotImplementedError('squeeze parameter is not yet implemented')
        if not observed:
            raise NotImplementedError('observed parameter is not yet implemented')
        if by is None and level is None:
            raise TypeError('groupby() requires either by or level to be specified.')
        if group_keys is None:
            group_keys = False
        return self.__class__._resampler(self, by=by) if isinstance(by, cudf.Grouper) and by.freq else self.__class__._groupby(self, by=by, level=level, as_index=as_index, dropna=dropna, sort=sort, group_keys=group_keys)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Addition', op_name='add', equivalent_op='frame + other', df_op_example=textwrap.dedent('\n                >>> df.add(1)\n                        angles  degrees\n                circle          1      361\n                triangle        4      181\n                rectangle       5      361\n                '), ser_op_example=textwrap.dedent('\n                >>> a.add(b)\n                a       2\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.add(b, fill_value=0)\n                a       2\n                b       1\n                c       1\n                d       1\n                e    <NA>\n                dtype: int64\n                ')))
    def add(self, other, axis, level=None, fill_value=None):
        if False:
            print('Hello World!')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__add__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Addition', op_name='radd', equivalent_op='other + frame', df_op_example=textwrap.dedent('\n                >>> df.radd(1)\n                        angles  degrees\n                circle          1      361\n                triangle        4      181\n                rectangle       5      361\n                '), ser_op_example=textwrap.dedent('\n                >>> a.radd(b)\n                a       2\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.radd(b, fill_value=0)\n                a       2\n                b       1\n                c       1\n                d       1\n                e    <NA>\n                dtype: int64\n                ')))
    def radd(self, other, axis, level=None, fill_value=None):
        if False:
            return 10
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__radd__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Subtraction', op_name='sub', equivalent_op='frame - other', df_op_example=textwrap.dedent('\n                >>> df.sub(1)\n                        angles  degrees\n                circle         -1      359\n                triangle        2      179\n                rectangle       3      359\n                '), ser_op_example=textwrap.dedent('\n                >>> a.sub(b)\n                a       0\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.sub(b, fill_value=0)\n                a       2\n                b       1\n                c       1\n                d      -1\n                e    <NA>\n                dtype: int64\n                ')))
    def subtract(self, other, axis, level=None, fill_value=None):
        if False:
            i = 10
            return i + 15
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__sub__', fill_value)
    sub = subtract

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Subtraction', op_name='rsub', equivalent_op='other - frame', df_op_example=textwrap.dedent('\n                >>> df.rsub(1)\n                        angles  degrees\n                circle          1     -359\n                triangle       -2     -179\n                rectangle      -3     -359\n                '), ser_op_example=textwrap.dedent('\n                >>> a.rsub(b)\n                a       0\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.rsub(b, fill_value=0)\n                a       0\n                b      -1\n                c      -1\n                d       1\n                e    <NA>\n                dtype: int64\n                ')))
    def rsub(self, other, axis, level=None, fill_value=None):
        if False:
            print('Hello World!')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__rsub__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Multiplication', op_name='mul', equivalent_op='frame * other', df_op_example=textwrap.dedent('\n                >>> df.multiply(1)\n                        angles  degrees\n                circle          0      360\n                triangle        3      180\n                rectangle       4      360\n                '), ser_op_example=textwrap.dedent('\n                >>> a.multiply(b)\n                a       1\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.multiply(b, fill_value=0)\n                a       1\n                b       0\n                c       0\n                d       0\n                e    <NA>\n                dtype: int64\n                ')))
    def multiply(self, other, axis, level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__mul__', fill_value)
    mul = multiply

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Multiplication', op_name='rmul', equivalent_op='other * frame', df_op_example=textwrap.dedent('\n                >>> df.rmul(1)\n                        angles  degrees\n                circle          0      360\n                triangle        3      180\n                rectangle       4      360\n                '), ser_op_example=textwrap.dedent('\n                >>> a.rmul(b)\n                a       1\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.rmul(b, fill_value=0)\n                a       1\n                b       0\n                c       0\n                d       0\n                e    <NA>\n                dtype: int64\n                ')))
    def rmul(self, other, axis, level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__rmul__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Modulo', op_name='mod', equivalent_op='frame % other', df_op_example=textwrap.dedent('\n                >>> df.mod(1)\n                        angles  degrees\n                circle          0        0\n                triangle        0        0\n                rectangle       0        0\n                '), ser_op_example=textwrap.dedent('\n                >>> a.mod(b)\n                a       0\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.mod(b, fill_value=0)\n                a             0\n                b    4294967295\n                c    4294967295\n                d             0\n                e          <NA>\n                dtype: int64\n                ')))
    def mod(self, other, axis, level=None, fill_value=None):
        if False:
            while True:
                i = 10
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__mod__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Modulo', op_name='rmod', equivalent_op='other % frame', df_op_example=textwrap.dedent('\n                >>> df.rmod(1)\n                            angles  degrees\n                circle     4294967295        1\n                triangle            1        1\n                rectangle           1        1\n                '), ser_op_example=textwrap.dedent('\n                >>> a.rmod(b)\n                a       0\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.rmod(b, fill_value=0)\n                a             0\n                b             0\n                c             0\n                d    4294967295\n                e          <NA>\n                dtype: int64\n                ')))
    def rmod(self, other, axis, level=None, fill_value=None):
        if False:
            while True:
                i = 10
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__rmod__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Exponential', op_name='pow', equivalent_op='frame ** other', df_op_example=textwrap.dedent('\n                >>> df.pow(1)\n                        angles  degrees\n                circle          0      360\n                triangle        2      180\n                rectangle       4      360\n                '), ser_op_example=textwrap.dedent('\n                >>> a.pow(b)\n                a       1\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.pow(b, fill_value=0)\n                a       1\n                b       1\n                c       1\n                d       0\n                e    <NA>\n                dtype: int64\n                ')))
    def pow(self, other, axis, level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__pow__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Exponential', op_name='rpow', equivalent_op='other ** frame', df_op_example=textwrap.dedent('\n                >>> df.rpow(1)\n                        angles  degrees\n                circle          1        1\n                triangle        1        1\n                rectangle       1        1\n                '), ser_op_example=textwrap.dedent('\n                >>> a.rpow(b)\n                a       1\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.rpow(b, fill_value=0)\n                a       1\n                b       0\n                c       0\n                d       1\n                e    <NA>\n                dtype: int64\n                ')))
    def rpow(self, other, axis, level=None, fill_value=None):
        if False:
            print('Hello World!')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__rpow__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Integer division', op_name='floordiv', equivalent_op='frame // other', df_op_example=textwrap.dedent('\n                >>> df.floordiv(1)\n                        angles  degrees\n                circle          0      360\n                triangle        3      180\n                rectangle       4      360\n                '), ser_op_example=textwrap.dedent('\n                >>> a.floordiv(b)\n                a       1\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.floordiv(b, fill_value=0)\n                a                      1\n                b    9223372036854775807\n                c    9223372036854775807\n                d                      0\n                e                   <NA>\n                dtype: int64\n                ')))
    def floordiv(self, other, axis, level=None, fill_value=None):
        if False:
            return 10
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__floordiv__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Integer division', op_name='rfloordiv', equivalent_op='other // frame', df_op_example=textwrap.dedent('\n                >>> df.rfloordiv(1)\n                                        angles  degrees\n                circle     9223372036854775807        0\n                triangle                     0        0\n                rectangle                    0        0\n                '), ser_op_example=textwrap.dedent('\n                >>> a.rfloordiv(b)\n                a       1\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: int64\n                >>> a.rfloordiv(b, fill_value=0)\n                a                      1\n                b                      0\n                c                      0\n                d    9223372036854775807\n                e                   <NA>\n                dtype: int64\n                ')))
    def rfloordiv(self, other, axis, level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__rfloordiv__', fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Floating division', op_name='truediv', equivalent_op='frame / other', df_op_example=textwrap.dedent('\n                >>> df.truediv(1)\n                        angles  degrees\n                circle        0.0    360.0\n                triangle      3.0    180.0\n                rectangle     4.0    360.0\n                '), ser_op_example=textwrap.dedent('\n                >>> a.truediv(b)\n                a     1.0\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: float64\n                >>> a.truediv(b, fill_value=0)\n                a     1.0\n                b     Inf\n                c     Inf\n                d     0.0\n                e    <NA>\n                dtype: float64\n                ')))
    def truediv(self, other, axis, level=None, fill_value=None):
        if False:
            return 10
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__truediv__', fill_value)
    div = truediv
    divide = truediv

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Floating division', op_name='rtruediv', equivalent_op='other / frame', df_op_example=textwrap.dedent('\n                >>> df.rtruediv(1)\n                            angles   degrees\n                circle          inf  0.002778\n                triangle   0.333333  0.005556\n                rectangle  0.250000  0.002778\n                '), ser_op_example=textwrap.dedent('\n                >>> a.rtruediv(b)\n                a     1.0\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: float64\n                >>> a.rtruediv(b, fill_value=0)\n                a     1.0\n                b     0.0\n                c     0.0\n                d     Inf\n                e    <NA>\n                dtype: float64\n                ')))
    def rtruediv(self, other, axis, level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        if level is not None:
            raise NotImplementedError('level parameter is not supported yet.')
        return self._binaryop(other, '__rtruediv__', fill_value)
    rdiv = rtruediv

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Equal to', op_name='eq', equivalent_op='frame == other', df_op_example=textwrap.dedent('\n                >>> df.eq(1)\n                        angles  degrees\n                circle      False    False\n                triangle    False    False\n                rectangle   False    False\n                '), ser_op_example=textwrap.dedent('\n                >>> a.eq(b)\n                a    True\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: bool\n                >>> a.eq(b, fill_value=0)\n                a    True\n                b   False\n                c   False\n                d   False\n                e    <NA>\n                dtype: bool\n                ')))
    def eq(self, other, axis='columns', level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        return self._binaryop(other=other, op='__eq__', fill_value=fill_value, can_reindex=True)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Not equal to', op_name='ne', equivalent_op='frame != other', df_op_example=textwrap.dedent('\n                >>> df.ne(1)\n                        angles  degrees\n                circle       True     True\n                triangle     True     True\n                rectangle    True     True\n                '), ser_op_example=textwrap.dedent('\n                >>> a.ne(b)\n                a    False\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: bool\n                >>> a.ne(b, fill_value=0)\n                a   False\n                b    True\n                c    True\n                d    True\n                e    <NA>\n                dtype: bool\n                ')))
    def ne(self, other, axis='columns', level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        return self._binaryop(other=other, op='__ne__', fill_value=fill_value, can_reindex=True)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Less than', op_name='lt', equivalent_op='frame < other', df_op_example=textwrap.dedent('\n                >>> df.lt(1)\n                        angles  degrees\n                circle       True    False\n                triangle    False    False\n                rectangle   False    False\n                '), ser_op_example=textwrap.dedent('\n                >>> a.lt(b)\n                a   False\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: bool\n                >>> a.lt(b, fill_value=0)\n                a   False\n                b   False\n                c   False\n                d    True\n                e    <NA>\n                dtype: bool\n                ')))
    def lt(self, other, axis='columns', level=None, fill_value=None):
        if False:
            return 10
        return self._binaryop(other=other, op='__lt__', fill_value=fill_value, can_reindex=True)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Less than or equal to', op_name='le', equivalent_op='frame <= other', df_op_example=textwrap.dedent('\n                >>> df.le(1)\n                        angles  degrees\n                circle       True    False\n                triangle    False    False\n                rectangle   False    False\n                '), ser_op_example=textwrap.dedent('\n                >>> a.le(b)\n                a    True\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: bool\n                >>> a.le(b, fill_value=0)\n                a    True\n                b   False\n                c   False\n                d    True\n                e    <NA>\n                dtype: bool\n                ')))
    def le(self, other, axis='columns', level=None, fill_value=None):
        if False:
            print('Hello World!')
        return self._binaryop(other=other, op='__le__', fill_value=fill_value, can_reindex=True)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Greater than', op_name='gt', equivalent_op='frame > other', df_op_example=textwrap.dedent('\n                >>> df.gt(1)\n                        angles  degrees\n                circle      False     True\n                triangle     True     True\n                rectangle    True     True\n                '), ser_op_example=textwrap.dedent('\n                >>> a.gt(b)\n                a   False\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: bool\n                >>> a.gt(b, fill_value=0)\n                a   False\n                b    True\n                c    True\n                d   False\n                e    <NA>\n                dtype: bool\n                ')))
    def gt(self, other, axis='columns', level=None, fill_value=None):
        if False:
            i = 10
            return i + 15
        return self._binaryop(other=other, op='__gt__', fill_value=fill_value, can_reindex=True)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(doc_binop_template.format(operation='Greater than or equal to', op_name='ge', equivalent_op='frame >= other', df_op_example=textwrap.dedent('\n                >>> df.ge(1)\n                        angles  degrees\n                circle      False     True\n                triangle     True     True\n                rectangle    True     True\n                '), ser_op_example=textwrap.dedent('\n                >>> a.ge(b)\n                a    True\n                b    <NA>\n                c    <NA>\n                d    <NA>\n                e    <NA>\n                dtype: bool\n                >>> a.ge(b, fill_value=0)\n                a   True\n                b    True\n                c    True\n                d   False\n                e    <NA>\n                dtype: bool\n                ')))
    def ge(self, other, axis='columns', level=None, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        return self._binaryop(other=other, op='__ge__', fill_value=fill_value, can_reindex=True)

    def _preprocess_subset(self, subset):
        if False:
            print('Hello World!')
        if subset is None:
            subset = self._column_names
        elif not np.iterable(subset) or isinstance(subset, str) or (isinstance(subset, tuple) and subset in self._data.names):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError(f'columns {diff} do not exist')
        return subset

    @_cudf_nvtx_annotate
    def rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute numerical data ranks (1 through n) along axis.\n\n        By default, equal values are assigned a rank that is the average of the\n        ranks of those values.\n\n        Parameters\n        ----------\n        axis : {0 or 'index'}, default 0\n            Index to direct ranking.\n        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'\n            How to rank the group of records that have the same value\n            (i.e. ties):\n            * average: average rank of the group\n            * min: lowest rank in the group\n            * max: highest rank in the group\n            * first: ranks assigned in order they appear in the array\n            * dense: like 'min', but rank always increases by 1 between groups.\n        numeric_only : bool, optional\n            For DataFrame objects, rank only numeric columns if set to True.\n        na_option : {'keep', 'top', 'bottom'}, default 'keep'\n            How to rank NaN values:\n            * keep: assign NaN rank to NaN values\n            * top: assign smallest rank to NaN values if ascending\n            * bottom: assign highest rank to NaN values if ascending.\n        ascending : bool, default True\n            Whether or not the elements should be ranked in ascending order.\n        pct : bool, default False\n            Whether or not to display the returned rankings in percentile\n            form.\n\n        Returns\n        -------\n        same type as caller\n            Return a Series or DataFrame with data ranks as values.\n        "
        if method not in {'average', 'min', 'max', 'first', 'dense'}:
            raise KeyError(method)
        method_enum = libcudf.aggregation.RankMethod[method.upper()]
        if na_option not in {'keep', 'top', 'bottom'}:
            raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")
        if axis not in (0, 'index'):
            raise NotImplementedError(f'axis must be `0`/`index`, axis={axis} is not yet supported in rank')
        source = self
        if numeric_only:
            numeric_cols = (name for name in self._data.names if _is_non_decimal_numeric_dtype(self._data[name]))
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                return source.astype('float64')
        result_columns = libcudf.sort.rank_columns([*source._columns], method_enum, na_option, ascending, pct)
        return self.__class__._from_data(dict(zip(source._column_names, result_columns)), index=source._index).astype(np.float64)

    def convert_dtypes(self, infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True, convert_floating=True, dtype_backend=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert columns to the best possible nullable dtypes.\n\n        If the dtype is numeric, and consists of all integers, convert\n        to an appropriate integer extension type. Otherwise, convert\n        to an appropriate floating type.\n\n        All other dtypes are always returned as-is as all dtypes in\n        cudf are nullable.\n        '
        result = self.copy()
        if convert_floating:
            for (name, col) in result._data.items():
                if col.dtype.kind == 'f':
                    col = col.fillna(0)
                    if cp.allclose(col, col.astype('int64')):
                        result._data[name] = col.astype('int64')
        return result

    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        if False:
            print('Hello World!')
        return [type(self), self._dtypes, self.index, self.hash_values().values_host]

def _check_duplicate_level_names(specified, level_names):
    if False:
        for i in range(10):
            print('nop')
    'Raise if any of `specified` has duplicates in `level_names`.'
    if specified is None:
        return
    if len(set(level_names)) == len(level_names):
        return
    duplicates = {key for (key, val) in Counter(level_names).items() if val > 1}
    duplicates_specified = [spec for spec in specified if spec in duplicates]
    if not len(duplicates_specified) == 0:
        raise ValueError(f'The names {duplicates_specified} occurs multiple times, use a level number')

@_cudf_nvtx_annotate
def _get_replacement_values_for_columns(to_replace: Any, value: Any, columns_dtype_map: Dict[Any, Any]) -> Tuple[Dict[Any, bool], Dict[Any, Any], Dict[Any, Any]]:
    if False:
        i = 10
        return i + 15
    '\n    Returns a per column mapping for the values to be replaced, new\n    values to be replaced with and if all the values are empty.\n\n    Parameters\n    ----------\n    to_replace : numeric, str, list-like or dict\n        Contains the values to be replaced.\n    value : numeric, str, list-like, or dict\n        Contains the values to replace `to_replace` with.\n    columns_dtype_map : dict\n        A column to dtype mapping representing dtype of columns.\n\n    Returns\n    -------\n    all_na_columns : dict\n        A dict mapping of all columns if they contain all na values\n    to_replace_columns : dict\n        A dict mapping of all columns and the existing values that\n        have to be replaced.\n    values_columns : dict\n        A dict mapping of all columns and the corresponding values\n        to be replaced with.\n    '
    to_replace_columns: Dict[Any, Any] = {}
    values_columns: Dict[Any, Any] = {}
    all_na_columns: Dict[Any, Any] = {}
    if is_scalar(to_replace) and is_scalar(value):
        to_replace_columns = {col: [to_replace] for col in columns_dtype_map}
        values_columns = {col: [value] for col in columns_dtype_map}
    elif cudf.api.types.is_list_like(to_replace) or isinstance(to_replace, ColumnBase):
        if is_scalar(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {col: [value] if _is_non_decimal_numeric_dtype(columns_dtype_map[col]) else full(len(to_replace), value, cudf.dtype(type(value))) for col in columns_dtype_map}
        elif cudf.api.types.is_list_like(value):
            if len(to_replace) != len(value):
                raise ValueError(f'Replacement lists must be of same length. Expected {len(to_replace)}, got {len(value)}.')
            else:
                to_replace_columns = {col: to_replace for col in columns_dtype_map}
                values_columns = {col: value for col in columns_dtype_map}
        elif cudf.utils.dtypes.is_column_like(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {col: value for col in columns_dtype_map}
        else:
            raise TypeError('value argument must be scalar, list-like or Series')
    elif _is_series(to_replace):
        if value is None:
            to_replace_columns = {col: as_column(to_replace.index) for col in columns_dtype_map}
            values_columns = {col: to_replace for col in columns_dtype_map}
        elif is_dict_like(value):
            to_replace_columns = {col: to_replace[col] for col in columns_dtype_map if col in to_replace}
            values_columns = {col: value[col] for col in to_replace_columns if col in value}
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {col: to_replace[col] for col in columns_dtype_map if col in to_replace}
            values_columns = {col: [value] if is_scalar(value) else value[col] for col in to_replace_columns if col in value}
        else:
            raise ValueError('Series.replace cannot use dict-like to_replace and non-None value')
    elif is_dict_like(to_replace):
        if value is None:
            to_replace_columns = {col: list(to_replace.keys()) for col in columns_dtype_map}
            values_columns = {col: list(to_replace.values()) for col in columns_dtype_map}
        elif is_dict_like(value):
            to_replace_columns = {col: to_replace[col] for col in columns_dtype_map if col in to_replace}
            values_columns = {col: value[col] for col in columns_dtype_map if col in value}
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {col: to_replace[col] for col in columns_dtype_map if col in to_replace}
            values_columns = {col: [value] if is_scalar(value) else value for col in columns_dtype_map if col in to_replace}
        else:
            raise TypeError('value argument must be scalar, dict, or Series')
    else:
        raise TypeError(f"Expecting 'to_replace' to be either a scalar, array-like, dict or None, got invalid type '{type(to_replace).__name__}'")
    to_replace_columns = {key: [value] if is_scalar(value) else value for (key, value) in to_replace_columns.items()}
    values_columns = {key: [value] if is_scalar(value) else value for (key, value) in values_columns.items()}
    for i in to_replace_columns:
        if i in values_columns:
            if isinstance(values_columns[i], list):
                all_na = values_columns[i].count(None) == len(values_columns[i])
            else:
                all_na = False
            all_na_columns[i] = all_na
    return (all_na_columns, to_replace_columns, values_columns)

def _is_series(obj):
    if False:
        return 10
    '\n    Checks if the `obj` is of type `cudf.Series`\n    instead of checking for isinstance(obj, cudf.Series)\n    '
    return isinstance(obj, Frame) and obj.ndim == 1 and (obj._index is not None)

@_cudf_nvtx_annotate
def _drop_rows_by_labels(obj: DataFrameOrSeries, labels: Union[ColumnLike, abc.Iterable, str], level: Union[int, str], errors: str) -> DataFrameOrSeries:
    if False:
        print('Hello World!')
    'Remove rows specified by `labels`.\n\n    If `errors="raise"`, an error is raised if some items in `labels` do not\n    exist in `obj._index`.\n\n    Will raise if level(int) is greater or equal to index nlevels.\n    '
    if isinstance(level, int) and level >= obj.index.nlevels:
        raise ValueError('Param level out of bounds.')
    if not isinstance(labels, cudf.core.single_column_frame.SingleColumnFrame):
        labels = as_column(labels)
    if isinstance(obj.index, cudf.MultiIndex):
        if level is None:
            level = 0
        levels_index = obj.index.get_level_values(level)
        if errors == 'raise' and (not labels.isin(levels_index).all()):
            raise KeyError('One or more values not found in axis')
        if isinstance(level, int):
            ilevel = level
        else:
            ilevel = obj._index.names.index(level)
        idx_nlv = obj._index.nlevels
        working_df = obj._index.to_frame(index=False)
        working_df.columns = list(range(idx_nlv))
        for (i, col) in enumerate(obj._data):
            working_df[idx_nlv + i] = obj._data[col]
        working_df = working_df.set_index(level)
        to_join = cudf.DataFrame(index=cudf.Index(labels, name=level))
        join_res = working_df.join(to_join, how='leftanti')
        join_res._insert(ilevel, name=join_res._index.name, value=join_res._index)
        midx = cudf.MultiIndex.from_frame(join_res.iloc[:, 0:idx_nlv], names=obj._index.names)
        if isinstance(obj, cudf.Series):
            return obj.__class__._from_data(join_res.iloc[:, idx_nlv:]._data, index=midx, name=obj.name)
        else:
            return obj.__class__._from_data(join_res.iloc[:, idx_nlv:]._data, index=midx, columns=obj._data.to_pandas_index())
    else:
        if errors == 'raise' and (not labels.isin(obj.index).all()):
            raise KeyError('One or more values not found in axis')
        key_df = cudf.DataFrame._from_data(data={}, index=cudf.Index(labels, name=getattr(labels, 'name', obj.index.name)))
        if isinstance(obj, cudf.DataFrame):
            res = obj.join(key_df, how='leftanti')
        else:
            res = obj.to_frame(name='tmp').join(key_df, how='leftanti')['tmp']
            res.name = obj.name
        res._index = res.index.astype(obj.index.dtype)
        return res

def _is_same_dtype(lhs_dtype, rhs_dtype):
    if False:
        print('Hello World!')
    if lhs_dtype == rhs_dtype:
        return True
    elif is_categorical_dtype(lhs_dtype) and is_categorical_dtype(rhs_dtype) and (lhs_dtype.categories.dtype == rhs_dtype.categories.dtype):
        return True
    elif is_categorical_dtype(lhs_dtype) and (not is_categorical_dtype(rhs_dtype)) and (lhs_dtype.categories.dtype == rhs_dtype):
        return True
    elif is_categorical_dtype(rhs_dtype) and (not is_categorical_dtype(lhs_dtype)) and (rhs_dtype.categories.dtype == lhs_dtype):
        return True
    else:
        return False