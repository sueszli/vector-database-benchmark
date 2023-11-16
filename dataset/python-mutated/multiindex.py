from __future__ import annotations
import itertools
import numbers
import operator
import pickle
import warnings
from collections import abc
from functools import cached_property
from numbers import Integral
from typing import Any, List, MutableMapping, Tuple, Union
import cupy as cp
import numpy as np
import pandas as pd
from pandas._config import get_option
import cudf
from cudf import _lib as libcudf
from cudf._typing import DataFrameOrSeries
from cudf.api.extensions import no_default
from cudf.api.types import is_integer, is_list_like, is_object_dtype
from cudf.core import column
from cudf.core._compat import PANDAS_GE_150
from cudf.core.frame import Frame
from cudf.core.index import BaseIndex, _lexsorted_equal_range, as_index
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import NotIterable, _external_only_api, _is_same_name

def _maybe_indices_to_slice(indices: cp.ndarray) -> Union[slice, cp.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Makes best effort to convert an array of indices into a python slice.\n    If the conversion is not possible, return input. `indices` are expected\n    to be valid.\n    '
    if len(indices) == 1:
        x = indices[0].item()
        return slice(x, x + 1)
    if len(indices) == 2:
        (x1, x2) = (indices[0].item(), indices[1].item())
        return slice(x1, x2 + 1, x2 - x1)
    (start, step) = (indices[0].item(), (indices[1] - indices[0]).item())
    stop = start + step * len(indices)
    if (indices == cp.arange(start, stop, step)).all():
        return slice(start, stop, step)
    return indices

class MultiIndex(Frame, BaseIndex, NotIterable):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes: sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Not yet supported
    names: optional sequence of objects
        Names for each of the index levels.
    copy : bool, default False
        Copy the levels and codes.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.
        Not yet supported

    Attributes
    ----------
    names
    nlevels
    dtypes
    levels
    codes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_loc
    drop

    Returns
    -------
    MultiIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.MultiIndex(
    ... levels=[[1, 2], ['blue', 'red']], codes=[[0, 0, 1, 1], [1, 0, 1, 0]])
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               )
    """

    @_cudf_nvtx_annotate
    def __init__(self, levels=None, codes=None, sortorder=None, names=None, dtype=None, copy=False, name=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if sortorder is not None:
            raise NotImplementedError('sortorder is not yet supported')
        if name is not None:
            raise NotImplementedError('Use `names`, `name` is not yet supported')
        if len(levels) == 0:
            raise ValueError('Must pass non-zero number of levels/codes')
        if not isinstance(codes, cudf.DataFrame) and (not isinstance(codes[0], (abc.Sequence, np.ndarray, cp.ndarray))):
            raise TypeError('Codes is not a Sequence of sequences')
        if copy:
            if isinstance(codes, cudf.DataFrame):
                codes = codes.copy(deep=True)
            if len(levels) > 0 and isinstance(levels[0], cudf.Series):
                levels = [level.copy(deep=True) for level in levels]
        if not isinstance(codes, cudf.DataFrame):
            if len(levels) == len(codes):
                codes = cudf.DataFrame._from_data({i: column.as_column(code).astype(np.int64) for (i, code) in enumerate(codes)})
            else:
                raise ValueError('MultiIndex has unequal number of levels and codes and is inconsistent!')
        levels = [cudf.Series(level) for level in levels]
        if len(levels) != len(codes._data):
            raise ValueError('MultiIndex has unequal number of levels and codes and is inconsistent!')
        if len({c.size for c in codes._data.columns}) != 1:
            raise ValueError('MultiIndex length of codes does not match and is inconsistent!')
        for (level, code) in zip(levels, codes._data.columns):
            if code.max() > len(level) - 1:
                raise ValueError('MultiIndex code %d contains value %d larger than maximum level size at this position')
        source_data = {}
        for (i, (column_name, col)) in enumerate(codes._data.items()):
            if -1 in col:
                level = cudf.DataFrame({column_name: [None] + list(levels[i])}, index=range(-1, len(levels[i])))
            else:
                level = cudf.DataFrame({column_name: levels[i]})
            source_data[column_name] = libcudf.copying.gather([level._data[column_name]], col)[0]
        super().__init__(source_data)
        self._levels = levels
        self._codes = codes
        self._name = None
        self.names = names

    @property
    @_cudf_nvtx_annotate
    def names(self):
        if False:
            i = 10
            return i + 15
        return self._names

    @names.setter
    @_cudf_nvtx_annotate
    def names(self, value):
        if False:
            i = 10
            return i + 15
        if value is None:
            value = [None] * self.nlevels
        elif not is_list_like(value):
            raise ValueError('Names should be list-like for a MultiIndex')
        elif len(value) != self.nlevels:
            raise ValueError('Length of names must match number of levels in MultiIndex.')
        if len(value) == len(set(value)):
            self._data = self._data.__class__._create_unsafe(dict(zip(value, self._data.values())), level_names=self._data.level_names)
        self._names = pd.core.indexes.frozen.FrozenList(value)

    @_cudf_nvtx_annotate
    def to_series(self, index=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError("MultiIndex.to_series isn't implemented yet.")

    @_cudf_nvtx_annotate
    def astype(self, dtype, copy: bool=True):
        if False:
            i = 10
            return i + 15
        if not is_object_dtype(dtype):
            raise TypeError('Setting a MultiIndex dtype to anything other than object is not supported')
        return self

    @_cudf_nvtx_annotate
    def rename(self, names, inplace=False):
        if False:
            i = 10
            return i + 15
        "\n        Alter MultiIndex level names\n\n        Parameters\n        ----------\n        names : list of label\n            Names to set, length must be the same as number of levels\n        inplace : bool, default False\n            If True, modifies objects directly, otherwise returns a new\n            ``MultiIndex`` instance\n\n        Returns\n        -------\n        None or MultiIndex\n\n        Examples\n        --------\n        Renaming each levels of a MultiIndex to specified name:\n\n        >>> midx = cudf.MultiIndex.from_product(\n        ...     [('A', 'B'), (2020, 2021)], names=['c1', 'c2'])\n        >>> midx.rename(['lv1', 'lv2'])\n        MultiIndex([('A', 2020),\n                    ('A', 2021),\n                    ('B', 2020),\n                    ('B', 2021)],\n                names=['lv1', 'lv2'])\n        >>> midx.rename(['lv1', 'lv2'], inplace=True)\n        >>> midx\n        MultiIndex([('A', 2020),\n                    ('A', 2021),\n                    ('B', 2020),\n                    ('B', 2021)],\n                names=['lv1', 'lv2'])\n\n        ``names`` argument must be a list, and must have same length as\n        ``MultiIndex.levels``:\n\n        >>> midx.rename(['lv0'])\n        Traceback (most recent call last):\n        ValueError: Length of names must match number of levels in MultiIndex.\n\n        "
        return self.set_names(names, level=None, inplace=inplace)

    @_cudf_nvtx_annotate
    def set_names(self, names, level=None, inplace=False):
        if False:
            return 10
        names_is_list_like = is_list_like(names)
        level_is_list_like = is_list_like(level)
        if level is not None and (not level_is_list_like) and names_is_list_like:
            raise TypeError('Names must be a string when a single level is provided.')
        if not names_is_list_like and level is None and (self.nlevels > 1):
            raise TypeError('Must pass list-like as `names`.')
        if not names_is_list_like:
            names = [names]
        if level is not None and (not level_is_list_like):
            level = [level]
        if level is not None and len(names) != len(level):
            raise ValueError('Length of names must match length of level.')
        if level is None and len(names) != self.nlevels:
            raise ValueError('Length of names must match number of levels in MultiIndex.')
        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._level_index_from_level(lev) for lev in level]
        existing_names = list(self.names)
        for (i, lev) in enumerate(level):
            existing_names[lev] = names[i]
        names = existing_names
        return self._set_names(names=names, inplace=inplace)

    @classmethod
    @_cudf_nvtx_annotate
    def _from_data(cls, data: MutableMapping, name: Any=None) -> MultiIndex:
        if False:
            return 10
        obj = cls.from_frame(cudf.DataFrame._from_data(data=data))
        if name is not None:
            obj.name = name
        return obj

    @property
    @_cudf_nvtx_annotate
    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    @name.setter
    @_cudf_nvtx_annotate
    def name(self, value):
        if False:
            i = 10
            return i + 15
        self._name = value

    @_cudf_nvtx_annotate
    def copy(self, names=None, dtype=None, levels=None, codes=None, deep=False, name=None):
        if False:
            print('Hello World!')
        "Returns copy of MultiIndex object.\n\n        Returns a copy of `MultiIndex`. The `levels` and `codes` value can be\n        set to the provided parameters. When they are provided, the returned\n        MultiIndex is always newly constructed.\n\n        Parameters\n        ----------\n        names : sequence of objects, optional (default None)\n            Names for each of the index levels.\n        dtype : object, optional (default None)\n            MultiIndex dtype, only supports None or object type\n\n            .. deprecated:: 23.02\n\n               The `dtype` parameter is deprecated and will be removed in\n               a future version of cudf. Use the `astype` method instead.\n\n        levels : sequence of arrays, optional (default None)\n            The unique labels for each level. Original values used if None.\n\n            .. deprecated:: 23.02\n\n               The `levels` parameter is deprecated and will be removed in\n               a future version of cudf.\n\n        codes : sequence of arrays, optional (default None)\n            Integers for each level designating which label at each location.\n            Original values used if None.\n\n            .. deprecated:: 23.02\n\n               The `codes` parameter is deprecated and will be removed in\n               a future version of cudf.\n\n        deep : Bool (default False)\n            If True, `._data`, `._levels`, `._codes` will be copied. Ignored if\n            `levels` or `codes` are specified.\n        name : object, optional (default None)\n            To keep consistent with `Index.copy`, should not be used.\n\n        Returns\n        -------\n        Copy of MultiIndex Instance\n\n        Examples\n        --------\n        >>> df = cudf.DataFrame({'Close': [3400.00, 226.58, 3401.80, 228.91]})\n        >>> idx1 = cudf.MultiIndex(\n        ... levels=[['2020-08-27', '2020-08-28'], ['AMZN', 'MSFT']],\n        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],\n        ... names=['Date', 'Symbol'])\n        >>> idx2 = idx1.copy(\n        ... levels=[['day1', 'day2'], ['com1', 'com2']],\n        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],\n        ... names=['col1', 'col2'])\n\n        >>> df.index = idx1\n        >>> df\n                             Close\n        Date       Symbol\n        2020-08-27 AMZN    3400.00\n                   MSFT     226.58\n        2020-08-28 AMZN    3401.80\n                   MSFT     228.91\n\n        >>> df.index = idx2\n        >>> df\n                     Close\n        col1 col2\n        day1 com1  3400.00\n             com2   226.58\n        day2 com1  3401.80\n             com2   228.91\n\n        "
        if levels is not None:
            warnings.warn('parameter levels is deprecated and will be removed in a future version.', FutureWarning)
        if codes is not None:
            warnings.warn('parameter codes is deprecated and will be removed in a future version.', FutureWarning)
        if dtype is not None:
            warnings.warn('parameter dtype is deprecated and will be removed in a future version. Use the astype method instead.', FutureWarning)
        dtype = object if dtype is None else dtype
        if not pd.api.types.is_object_dtype(dtype):
            raise TypeError('Dtype for MultiIndex only supports object type.')
        if levels is not None or codes is not None:
            if self._levels is None or self._codes is None:
                self._compute_levels_and_codes()
            levels = self._levels if levels is None else levels
            codes = self._codes if codes is None else codes
            names = self.names if names is None else names
            mi = MultiIndex(levels=levels, codes=codes, names=names, copy=deep)
            return mi
        mi = MultiIndex._from_data(self._data.copy(deep=deep))
        if self._levels is not None:
            mi._levels = [s.copy(deep) for s in self._levels]
        if self._codes is not None:
            mi._codes = self._codes.copy(deep)
        if names is not None:
            mi.names = names
        elif self.names is not None:
            mi.names = self.names.copy()
        return mi

    @_cudf_nvtx_annotate
    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        max_seq_items = get_option('display.max_seq_items') or len(self)
        if len(self) > max_seq_items:
            n = int(max_seq_items / 2) + 1
            indices = column.arange(start=0, stop=n, step=1)
            indices = indices.append(column.arange(start=len(self) - n, stop=len(self), step=1))
            preprocess = self.take(indices)
        else:
            preprocess = self.copy(deep=False)
        if any((col.has_nulls() for col in preprocess._data.columns)):
            preprocess_df = preprocess.to_frame(index=False)
            for (name, col) in preprocess._data.items():
                if isinstance(col, (column.datetime.DatetimeColumn, column.timedelta.TimeDeltaColumn)):
                    preprocess_df[name] = col.astype('str').fillna(str(cudf.NaT))
            tuples_list = list(zip(*list((map(lambda val: pd.NA if val is None else val, col) for col in preprocess_df.to_arrow().to_pydict().values()))))
            if not PANDAS_GE_150:
                preprocess_pdf = pd.DataFrame({name: col.to_pandas(nullable=col.dtype.kind != 'f') for (name, col) in preprocess._data.items()})
                preprocess_pdf.columns = preprocess.names
                preprocess = pd.MultiIndex.from_frame(preprocess_pdf)
            else:
                preprocess = preprocess.to_pandas(nullable=True)
            preprocess.values[:] = tuples_list
        else:
            preprocess = preprocess.to_pandas(nullable=True)
        output = repr(preprocess)
        output_prefix = self.__class__.__name__ + '('
        output = output.lstrip(output_prefix)
        lines = output.split('\n')
        if len(lines) > 1:
            if 'length=' in lines[-1] and len(self) != len(preprocess):
                last_line = lines[-1]
                length_index = last_line.index('length=')
                last_line = last_line[:length_index] + f'length={len(self)})'
                lines = lines[:-1]
                lines.append(last_line)
        data_output = '\n'.join(lines)
        return output_prefix + data_output

    @property
    def _codes_frame(self):
        if False:
            print('Hello World!')
        if self._codes is None:
            self._compute_levels_and_codes()
        return self._codes

    @property
    @_external_only_api('Use ._codes_frame instead')
    @_cudf_nvtx_annotate
    def codes(self):
        if False:
            print('Hello World!')
        "\n        Returns the codes of the underlying MultiIndex.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 11, 12]})\n        >>> midx = cudf.MultiIndex.from_frame(df)\n        >>> midx\n        MultiIndex([(1, 10),\n                    (2, 11),\n                    (3, 12)],\n                names=['a', 'b'])\n        >>> midx.codes\n        FrozenList([[0, 1, 2], [0, 1, 2]])\n        "
        return pd.core.indexes.frozen.FrozenList((col.values for col in self._codes_frame._columns))

    def get_slice_bound(self, label, side, kind=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @property
    @_cudf_nvtx_annotate
    def nlevels(self):
        if False:
            i = 10
            return i + 15
        'Integer number of levels in this MultiIndex.'
        return len(self._data)

    @property
    @_cudf_nvtx_annotate
    def levels(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns list of levels in the MultiIndex\n\n        Returns\n        -------\n        List of Series objects\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 11, 12]})\n        >>> cudf.MultiIndex.from_frame(df)\n        MultiIndex([(1, 10),\n                    (2, 11),\n                    (3, 12)],\n                names=['a', 'b'])\n        >>> midx = cudf.MultiIndex.from_frame(df)\n        >>> midx\n        MultiIndex([(1, 10),\n                    (2, 11),\n                    (3, 12)],\n                names=['a', 'b'])\n        >>> midx.levels\n        [Int64Index([1, 2, 3], dtype='int64', name='a'), Int64Index([10, 11, 12], dtype='int64', name='b')]\n        "
        if self._levels is None:
            self._compute_levels_and_codes()
        return self._levels

    @property
    @_cudf_nvtx_annotate
    def ndim(self):
        if False:
            for i in range(10):
                print('nop')
        'Dimension of the data. For MultiIndex ndim is always 2.'
        return 2

    @_cudf_nvtx_annotate
    def _get_level_label(self, level):
        if False:
            i = 10
            return i + 15
        'Get name of the level.\n\n        Parameters\n        ----------\n        level : int or level name\n            if level is name, it will be returned as it is\n            else if level is index of the level, then level\n            label will be returned as per the index.\n        '
        if level in self._data.names:
            return level
        else:
            return self._data.names[level]

    @_cudf_nvtx_annotate
    def isin(self, values, level=None):
        if False:
            i = 10
            return i + 15
        "Return a boolean array where the index values are in values.\n\n        Compute boolean array of whether each index value is found in\n        the passed set of values. The length of the returned boolean\n        array matches the length of the index.\n\n        Parameters\n        ----------\n        values : set, list-like, Index or Multi-Index\n            Sought values.\n        level : str or int, optional\n            Name or position of the index level to use (if the index\n            is a MultiIndex).\n\n        Returns\n        -------\n        is_contained : cupy array\n            CuPy array of boolean values.\n\n        Notes\n        -----\n        When `level` is None, `values` can only be MultiIndex, or a\n        set/list-like tuples.\n        When `level` is provided, `values` can be Index or MultiIndex,\n        or a set/list-like tuples.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import pandas as pd\n        >>> midx = cudf.from_pandas(pd.MultiIndex.from_arrays([[1,2,3],\n        ...                                  ['red', 'blue', 'green']],\n        ...                                  names=('number', 'color')))\n        >>> midx\n        MultiIndex([(1,   'red'),\n                    (2,  'blue'),\n                    (3, 'green')],\n                   names=['number', 'color'])\n\n        Check whether the strings in the 'color' level of the MultiIndex\n        are in a list of colors.\n\n        >>> midx.isin(['red', 'orange', 'yellow'], level='color')\n        array([ True, False, False])\n\n        To check across the levels of a MultiIndex, pass a list of tuples:\n\n        >>> midx.isin([(1, 'red'), (3, 'red')])\n        array([ True, False, False])\n        "
        if level is None:
            if isinstance(values, cudf.MultiIndex):
                values_idx = values
            elif isinstance(values, (cudf.Series, cudf.Index, cudf.DataFrame, column.ColumnBase)) or not is_list_like(values) or (is_list_like(values) and len(values) > 0 and (not isinstance(values[0], tuple))):
                raise TypeError('values need to be a Multi-Index or set/list-like tuple squences  when `level=None`.')
            else:
                values_idx = cudf.MultiIndex.from_tuples(values, names=self.names)
            self_df = self.to_frame(index=False).reset_index()
            values_df = values_idx.to_frame(index=False)
            idx = self_df.merge(values_df)._data['index']
            res = cudf.core.column.full(size=len(self), fill_value=False)
            res[idx] = True
            result = res.values
        else:
            level_series = self.get_level_values(level)
            result = level_series.isin(values)
        return result

    def where(self, cond, other=None, inplace=False):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('.where is not supported for MultiIndex operations')

    @_cudf_nvtx_annotate
    def _compute_levels_and_codes(self):
        if False:
            i = 10
            return i + 15
        levels = []
        codes = {}
        for (name, col) in self._data.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                (code, cats) = cudf.Series._from_data({None: col}).factorize()
            cats.name = name
            codes[name] = code.astype(np.int64)
            levels.append(cats)
        self._levels = levels
        self._codes = cudf.DataFrame._from_data(codes)

    @_cudf_nvtx_annotate
    def _compute_validity_mask(self, index, row_tuple, max_length):
        if False:
            while True:
                i = 10
        'Computes the valid set of indices of values in the lookup'
        lookup = cudf.DataFrame()
        for (i, row) in enumerate(row_tuple):
            if isinstance(row, slice) and row == slice(None):
                continue
            lookup[i] = cudf.Series(row)
        frame = cudf.DataFrame(dict(enumerate(index._data.columns)))
        data_table = cudf.concat([frame, cudf.DataFrame({'idx': cudf.Series(column.arange(len(frame)))})], axis=1)
        if cudf.get_option('mode.pandas_compatible'):
            lookup_order = '_' + '_'.join(map(str, lookup._data.names))
            lookup[lookup_order] = column.arange(len(lookup))
            postprocess = operator.methodcaller('sort_values', by=[lookup_order, 'idx'])
        else:
            postprocess = lambda r: r
        result = postprocess(lookup.merge(data_table))['idx']
        if len(result) == 0:
            for (idx, row) in enumerate(row_tuple):
                if row == slice(None):
                    continue
                if row not in index.levels[idx]._column:
                    raise KeyError(row)
        return result

    @_cudf_nvtx_annotate
    def _get_valid_indices_by_tuple(self, index, row_tuple, max_length):
        if False:
            while True:
                i = 10
        if isinstance(row_tuple, slice):
            if isinstance(row_tuple.start, numbers.Number) or isinstance(row_tuple.stop, numbers.Number) or row_tuple == slice(None):
                stop = row_tuple.stop or max_length
                (start, stop, step) = row_tuple.indices(stop)
                return column.arange(start, stop, step)
            start_values = self._compute_validity_mask(index, row_tuple.start, max_length)
            stop_values = self._compute_validity_mask(index, row_tuple.stop, max_length)
            return column.arange(start_values.min(), stop_values.max() + 1)
        elif isinstance(row_tuple, numbers.Number):
            return row_tuple
        return self._compute_validity_mask(index, row_tuple, max_length)

    @_cudf_nvtx_annotate
    def _index_and_downcast(self, result, index, index_key):
        if False:
            while True:
                i = 10
        if isinstance(index_key, (numbers.Number, slice)):
            index_key = [index_key]
        if len(index_key) > 0 and (not isinstance(index_key, tuple)) or isinstance(index_key[0], slice):
            index_key = index_key[0]
        slice_access = isinstance(index_key, slice)
        out_index = cudf.DataFrame()
        size = 0
        if not isinstance(index_key, (numbers.Number, slice)):
            size = len(index_key)
        for k in range(size, len(index._data)):
            out_index.insert(out_index._num_columns, k, cudf.Series._from_data({None: index._data.columns[k]}))
        need_downcast = isinstance(result, cudf.DataFrame) and len(result) == 1 and (not slice_access) and (size == 0 or len(index_key) == self.nlevels)
        if need_downcast:
            result = result.T
            return result[result._data.names[0]]
        if len(result) == 0 and (not slice_access):
            result = cudf.Series._from_data({}, name=tuple((col[0] for col in index._data.columns)))
        elif out_index._num_columns == 1:
            (*_, last_column) = index._data.columns
            out_index = as_index(last_column)
            out_index.name = index.names[-1]
            index = out_index
        elif out_index._num_columns > 1:
            result.reset_index(drop=True)
            if index.names is not None:
                result.names = index.names[size:]
            index = MultiIndex(levels=index.levels[size:], codes=index._codes_frame.iloc[:, size:], names=index.names[size:])
        if isinstance(index_key, tuple):
            result.index = index
        return result

    @_cudf_nvtx_annotate
    def _get_row_major(self, df: DataFrameOrSeries, row_tuple: Union[numbers.Number, slice, Tuple[Any, ...], List[Tuple[Any, ...]]]) -> DataFrameOrSeries:
        if False:
            print('Hello World!')
        if pd.api.types.is_bool_dtype(list(row_tuple) if isinstance(row_tuple, tuple) else row_tuple):
            return df[row_tuple]
        if isinstance(row_tuple, slice):
            if row_tuple.start is None:
                row_tuple = slice(self[0], row_tuple.stop, row_tuple.step)
            if row_tuple.stop is None:
                row_tuple = slice(row_tuple.start, self[-1], row_tuple.step)
        self._validate_indexer(row_tuple)
        valid_indices = self._get_valid_indices_by_tuple(df.index, row_tuple, len(df.index))
        indices = cudf.Series(valid_indices)
        result = df.take(indices)
        final = self._index_and_downcast(result, result.index, row_tuple)
        return final

    @_cudf_nvtx_annotate
    def _validate_indexer(self, indexer: Union[numbers.Number, slice, Tuple[Any, ...], List[Tuple[Any, ...]]]):
        if False:
            print('Hello World!')
        if isinstance(indexer, numbers.Number):
            return
        if isinstance(indexer, tuple):
            indexer = tuple(itertools.dropwhile(lambda x: x == slice(None), reversed(indexer)))[::-1]
            if len(indexer) > self.nlevels:
                raise IndexError('Indexer size exceeds number of levels')
        elif isinstance(indexer, slice):
            self._validate_indexer(indexer.start)
            self._validate_indexer(indexer.stop)
        else:
            for i in indexer:
                self._validate_indexer(i)

    @_cudf_nvtx_annotate
    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, MultiIndex):
            return np.array([self_col.equals(other_col) for (self_col, other_col) in zip(self._data.values(), other._data.values())])
        return NotImplemented

    @property
    @_cudf_nvtx_annotate
    def size(self):
        if False:
            print('Hello World!')
        return self._num_rows

    @_cudf_nvtx_annotate
    def take(self, indices):
        if False:
            print('Hello World!')
        if isinstance(indices, cudf.Series) and indices.has_nulls:
            raise ValueError('Column must have no nulls.')
        obj = super().take(indices)
        obj.names = self.names
        return obj

    @_cudf_nvtx_annotate
    def serialize(self):
        if False:
            for i in range(10):
                print('nop')
        (header, frames) = super().serialize()
        header['column_names'] = pickle.dumps(self.names)
        return (header, frames)

    @classmethod
    @_cudf_nvtx_annotate
    def deserialize(cls, header, frames):
        if False:
            while True:
                i = 10
        column_names = pickle.loads(header['column_names'])
        header['column_names'] = pickle.dumps(range(0, len(column_names)))
        obj = super().deserialize(header, frames)
        return obj._set_names(column_names)

    @_cudf_nvtx_annotate
    def __getitem__(self, index):
        if False:
            return 10
        flatten = isinstance(index, int)
        if isinstance(index, (Integral, abc.Sequence)):
            index = np.array(index)
        elif isinstance(index, slice):
            (start, stop, step) = index.indices(len(self))
            index = column.arange(start, stop, step)
        result = MultiIndex.from_frame(self.to_frame(index=False, name=range(0, self.nlevels)).take(index), names=self.names)
        if flatten:
            return result.to_pandas()[0]
        if self._codes_frame is not None:
            result._codes = self._codes_frame.take(index)
        if self._levels is not None:
            result._levels = self._levels
        return result

    @_cudf_nvtx_annotate
    def to_frame(self, index=True, name=no_default, allow_duplicates=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a DataFrame with the levels of the MultiIndex as columns.\n\n        Column ordering is determined by the DataFrame constructor with data as\n        a dict.\n\n        Parameters\n        ----------\n        index : bool, default True\n            Set the index of the returned DataFrame as the original MultiIndex.\n        name : list / sequence of str, optional\n            The passed names should substitute index level names.\n        allow_duplicates : bool, optional default False\n            Allow duplicate column labels to be created. Note\n            that this parameter is non-functional because\n            duplicates column labels aren't supported in cudf.\n\n        Returns\n        -------\n        DataFrame\n\n        Examples\n        --------\n        >>> import cudf\n        >>> mi = cudf.MultiIndex.from_tuples([('a', 'c'), ('b', 'd')])\n        >>> mi\n        MultiIndex([('a', 'c'),\n                    ('b', 'd')],\n                   )\n\n        >>> df = mi.to_frame()\n        >>> df\n             0  1\n        a c  a  c\n        b d  b  d\n\n        >>> df = mi.to_frame(index=False)\n        >>> df\n           0  1\n        0  a  c\n        1  b  d\n\n        >>> df = mi.to_frame(name=['x', 'y'])\n        >>> df\n             x  y\n        a c  a  c\n        b d  b  d\n        "
        if name is None:
            warnings.warn("Explicitly passing `name=None` currently preserves the Index's name or uses a default name of 0. This behaviour is deprecated, and in the future `None` will be used as the name of the resulting DataFrame column.", FutureWarning)
            name = no_default
        if name is not no_default:
            if len(name) != len(self.levels):
                raise ValueError("'name' should have the same length as number of levels on index.")
            column_names = name
        else:
            column_names = self.names
        all_none_names = None
        if not (all_none_names := all((x is None for x in column_names))) and len(column_names) != len(set(column_names)):
            raise ValueError('Duplicate column names are not allowed')
        df = cudf.DataFrame._from_data(data=self._data, columns=column_names if name is not no_default and (not all_none_names) else None)
        if index:
            df = df.set_index(self)
        return df

    @_cudf_nvtx_annotate
    def get_level_values(self, level):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the values at the requested level\n\n        Parameters\n        ----------\n        level : int or label\n\n        Returns\n        -------\n        An Index containing the values at the requested level.\n        '
        colnames = self._data.names
        if level not in colnames:
            if isinstance(level, int):
                if level < 0:
                    level = level + len(colnames)
                if level < 0 or level >= len(colnames):
                    raise IndexError(f"Invalid level number: '{level}'")
                level_idx = level
                level = colnames[level_idx]
            elif level in self.names:
                level_idx = list(self.names).index(level)
                level = colnames[level_idx]
            else:
                raise KeyError(f"Level not found: '{level}'")
        else:
            level_idx = colnames.index(level)
        level_values = as_index(self._data[level], name=self.names[level_idx])
        return level_values

    def _is_numeric(self):
        if False:
            return 10
        return False

    def _is_boolean(self):
        if False:
            while True:
                i = 10
        return False

    def _is_integer(self):
        if False:
            print('Hello World!')
        return False

    def _is_floating(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def _is_object(self):
        if False:
            i = 10
            return i + 15
        return False

    def _is_categorical(self):
        if False:
            return 10
        return False

    def _is_interval(self):
        if False:
            return 10
        return False

    @classmethod
    @_cudf_nvtx_annotate
    def _concat(cls, objs):
        if False:
            while True:
                i = 10
        source_data = [o.to_frame(index=False) for o in objs]
        if len(source_data) > 1:
            colnames = source_data[0]._data.to_pandas_index()
            for obj in source_data[1:]:
                obj.columns = colnames
        source_data = cudf.DataFrame._concat(source_data)
        names = [None] * source_data._num_columns
        objs = list(filter(lambda o: o.names is not None, objs))
        for o in range(len(objs)):
            for (i, name) in enumerate(objs[o].names):
                names[i] = names[i] or name
        return cudf.MultiIndex.from_frame(source_data, names=names)

    @classmethod
    @_cudf_nvtx_annotate
    def from_tuples(cls, tuples, names=None):
        if False:
            while True:
                i = 10
        "\n        Convert list of tuples to MultiIndex.\n\n        Parameters\n        ----------\n        tuples : list / sequence of tuple-likes\n            Each tuple is the index of one row/column.\n        names : list / sequence of str, optional\n            Names for the levels in the index.\n\n        Returns\n        -------\n        MultiIndex\n\n        See Also\n        --------\n        MultiIndex.from_product : Make a MultiIndex from cartesian product\n                                  of iterables.\n        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.\n\n        Examples\n        --------\n        >>> tuples = [(1, 'red'), (1, 'blue'),\n        ...           (2, 'red'), (2, 'blue')]\n        >>> cudf.MultiIndex.from_tuples(tuples, names=('number', 'color'))\n        MultiIndex([(1,  'red'),\n                    (1, 'blue'),\n                    (2,  'red'),\n                    (2, 'blue')],\n                   names=['number', 'color'])\n        "
        pdi = pd.MultiIndex.from_tuples(tuples, names=names)
        return cls.from_pandas(pdi)

    @_cudf_nvtx_annotate
    def to_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.values_host

    @property
    @_cudf_nvtx_annotate
    def values_host(self):
        if False:
            return 10
        '\n        Return a numpy representation of the MultiIndex.\n\n        Only the values in the MultiIndex will be returned.\n\n        Returns\n        -------\n        out : numpy.ndarray\n            The values of the MultiIndex.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> midx = cudf.MultiIndex(\n        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],\n        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],\n        ...         names=["x", "y"],\n        ...     )\n        >>> midx.values_host\n        array([(1, 1), (1, 5), (3, 2), (4, 2), (5, 1)], dtype=object)\n        >>> type(midx.values_host)\n        <class \'numpy.ndarray\'>\n        '
        return self.to_pandas().values

    @property
    @_cudf_nvtx_annotate
    def values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a CuPy representation of the MultiIndex.\n\n        Only the values in the MultiIndex will be returned.\n\n        Returns\n        -------\n        out: cupy.ndarray\n            The values of the MultiIndex.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> midx = cudf.MultiIndex(\n        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],\n        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],\n        ...         names=["x", "y"],\n        ...     )\n        >>> midx.values\n        array([[1, 1],\n            [1, 5],\n            [3, 2],\n            [4, 2],\n            [5, 1]])\n        >>> type(midx.values)\n        <class \'cupy...ndarray\'>\n        '
        if cudf.get_option('mode.pandas_compatible'):
            raise NotImplementedError('Unable to create a cupy array with tuples.')
        return self.to_frame(index=False).values

    @classmethod
    @_cudf_nvtx_annotate
    def from_frame(cls, df, names=None):
        if False:
            i = 10
            return i + 15
        "\n        Make a MultiIndex from a DataFrame.\n\n        Parameters\n        ----------\n        df : DataFrame\n            DataFrame to be converted to MultiIndex.\n        names : list-like, optional\n            If no names are provided, use the column names, or tuple of column\n            names if the columns is a MultiIndex. If a sequence, overwrite\n            names with the given sequence.\n\n        Returns\n        -------\n        MultiIndex\n            The MultiIndex representation of the given DataFrame.\n\n        See Also\n        --------\n        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.\n        MultiIndex.from_product : Make a MultiIndex from cartesian product\n                                  of iterables.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> df = cudf.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],\n        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],\n        ...                   columns=['a', 'b'])\n        >>> df\n              a       b\n        0    HI    Temp\n        1    HI  Precip\n        2    NJ    Temp\n        3    NJ  Precip\n        >>> cudf.MultiIndex.from_frame(df)\n        MultiIndex([('HI',   'Temp'),\n                    ('HI', 'Precip'),\n                    ('NJ',   'Temp'),\n                    ('NJ', 'Precip')],\n                   names=['a', 'b'])\n\n        Using explicit names, instead of the column names\n\n        >>> cudf.MultiIndex.from_frame(df, names=['state', 'observation'])\n        MultiIndex([('HI',   'Temp'),\n                    ('HI', 'Precip'),\n                    ('NJ',   'Temp'),\n                    ('NJ', 'Precip')],\n                   names=['state', 'observation'])\n        "
        obj = cls.__new__(cls)
        super(cls, obj).__init__()
        source_data = df.copy(deep=False)
        source_data.reset_index(drop=True, inplace=True)
        if isinstance(source_data, pd.DataFrame):
            source_data = cudf.DataFrame.from_pandas(source_data)
        names = names if names is not None else source_data._data.names
        if len(dict.fromkeys(names)) == len(names):
            source_data.columns = names
        obj._name = None
        obj._data = source_data._data
        obj.names = names
        obj._codes = None
        obj._levels = None
        return obj

    @classmethod
    @_cudf_nvtx_annotate
    def from_product(cls, arrays, names=None):
        if False:
            while True:
                i = 10
        "\n        Make a MultiIndex from the cartesian product of multiple iterables.\n\n        Parameters\n        ----------\n        iterables : list / sequence of iterables\n            Each iterable has unique labels for each level of the index.\n        names : list / sequence of str, optional\n            Names for the levels in the index.\n            If not explicitly provided, names will be inferred from the\n            elements of iterables if an element has a name attribute\n\n        Returns\n        -------\n        MultiIndex\n\n        See Also\n        --------\n        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.\n        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.\n\n        Examples\n        --------\n        >>> numbers = [0, 1, 2]\n        >>> colors = ['green', 'purple']\n        >>> cudf.MultiIndex.from_product([numbers, colors],\n        ...                            names=['number', 'color'])\n        MultiIndex([(0,  'green'),\n                    (0, 'purple'),\n                    (1,  'green'),\n                    (1, 'purple'),\n                    (2,  'green'),\n                    (2, 'purple')],\n                   names=['number', 'color'])\n        "
        pdi = pd.MultiIndex.from_product(arrays, names=names)
        return cls.from_pandas(pdi)

    @_cudf_nvtx_annotate
    def _poplevels(self, level):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove and return the specified levels from self.\n\n        Parameters\n        ----------\n        level : level name or index, list\n            One or more levels to remove\n\n        Returns\n        -------\n        Index composed of the removed levels. If only a single level\n        is removed, a flat index is returned. If no levels are specified\n        (empty list), None is returned.\n        '
        if not pd.api.types.is_list_like(level):
            level = (level,)
        ilevels = sorted((self._level_index_from_level(lev) for lev in level))
        if not ilevels:
            return None
        popped_data = {}
        popped_names = []
        names = list(self.names)
        for i in ilevels:
            n = self._data.names[i]
            popped_data[n] = self._data[n]
            popped_names.append(self.names[i])
        for i in reversed(ilevels):
            n = self._data.names[i]
            names.pop(i)
            popped_data[n] = self._data.pop(n)
        popped = cudf.core.index._index_from_data(popped_data)
        popped.names = popped_names
        self.names = names
        self._compute_levels_and_codes()
        return popped

    @_cudf_nvtx_annotate
    def swaplevel(self, i=-2, j=-1):
        if False:
            print('Hello World!')
        "\n        Swap level i with level j.\n        Calling this method does not change the ordering of the values.\n\n        Parameters\n        ----------\n        i : int or str, default -2\n            First level of index to be swapped.\n        j : int or str, default -1\n            Second level of index to be swapped.\n\n        Returns\n        -------\n        MultiIndex\n            A new MultiIndex.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> mi = cudf.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],\n        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])\n        >>> mi\n        MultiIndex([('a', 'bb'),\n            ('a', 'aa'),\n            ('b', 'bb'),\n            ('b', 'aa')],\n           )\n        >>> mi.swaplevel(0, 1)\n        MultiIndex([('bb', 'a'),\n            ('aa', 'a'),\n            ('bb', 'b'),\n            ('aa', 'b')],\n           )\n        "
        name_i = self._data.names[i] if isinstance(i, int) else i
        name_j = self._data.names[j] if isinstance(j, int) else j
        new_data = {}
        for (k, v) in self._data.items():
            if k not in (name_i, name_j):
                new_data[k] = v
            elif k == name_i:
                new_data[name_j] = self._data[name_j]
            elif k == name_j:
                new_data[name_i] = self._data[name_i]
        midx = MultiIndex._from_data(new_data)
        if all((n is None for n in self.names)):
            midx = midx.set_names(self.names)
        return midx

    @_cudf_nvtx_annotate
    def droplevel(self, level=-1):
        if False:
            while True:
                i = 10
        '\n        Removes the specified levels from the MultiIndex.\n\n        Parameters\n        ----------\n        level : level name or index, list-like\n            Integer, name or list of such, specifying one or more\n            levels to drop from the MultiIndex\n\n        Returns\n        -------\n        A MultiIndex or Index object, depending on the number of remaining\n        levels.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.MultiIndex.from_frame(\n        ...     cudf.DataFrame(\n        ...         {\n        ...             "first": ["a", "a", "a", "b", "b", "b"],\n        ...             "second": [1, 1, 2, 2, 3, 3],\n        ...             "third": [0, 1, 2, 0, 1, 2],\n        ...         }\n        ...     )\n        ... )\n\n        Dropping level by index:\n\n        >>> idx.droplevel(0)\n        MultiIndex([(1, 0),\n                    (1, 1),\n                    (2, 2),\n                    (2, 0),\n                    (3, 1),\n                    (3, 2)],\n                   names=[\'second\', \'third\'])\n\n        Dropping level by name:\n\n        >>> idx.droplevel("first")\n        MultiIndex([(1, 0),\n                    (1, 1),\n                    (2, 2),\n                    (2, 0),\n                    (3, 1),\n                    (3, 2)],\n                   names=[\'second\', \'third\'])\n\n        Dropping multiple levels:\n\n        >>> idx.droplevel(["first", "second"])\n        Int64Index([0, 1, 2, 0, 1, 2], dtype=\'int64\', name=\'third\')\n        '
        mi = self.copy(deep=False)
        mi._poplevels(level)
        if mi.nlevels == 1:
            return mi.get_level_values(mi.names[0])
        else:
            return mi

    @_cudf_nvtx_annotate
    def to_pandas(self, nullable=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = self.to_frame(index=False, name=list(range(self.nlevels))).to_pandas(nullable=nullable)
        return pd.MultiIndex.from_frame(result, names=self.names)

    @classmethod
    @_cudf_nvtx_annotate
    def from_pandas(cls, multiindex, nan_as_null=no_default):
        if False:
            while True:
                i = 10
        "\n        Convert from a Pandas MultiIndex\n\n        Raises\n        ------\n        TypeError for invalid input type.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import pandas as pd\n        >>> pmi = pd.MultiIndex(levels=[['a', 'b'], ['c', 'd']],\n        ...                     codes=[[0, 1], [1, 1]])\n        >>> cudf.from_pandas(pmi)\n        MultiIndex([('a', 'd'),\n                    ('b', 'd')],\n                   )\n        "
        if not isinstance(multiindex, pd.MultiIndex):
            raise TypeError('not a pandas.MultiIndex')
        if nan_as_null is no_default:
            nan_as_null = False if cudf.get_option('mode.pandas_compatible') else None
        names = tuple(range(len(multiindex.names)))
        df = cudf.DataFrame.from_pandas(multiindex.to_frame(index=False, name=names), nan_as_null)
        return cls.from_frame(df, names=multiindex.names)

    @cached_property
    @_cudf_nvtx_annotate
    def is_unique(self):
        if False:
            return 10
        return len(self) == len(self.unique())

    @property
    def dtype(self):
        if False:
            return 10
        return np.dtype('O')

    @cached_property
    @_cudf_nvtx_annotate
    def is_monotonic_increasing(self):
        if False:
            while True:
                i = 10
        '\n        Return if the index is monotonic increasing\n        (only equal or increasing) values.\n        '
        return self._is_sorted(ascending=None, null_position=None)

    @cached_property
    @_cudf_nvtx_annotate
    def is_monotonic_decreasing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return if the index is monotonic decreasing\n        (only equal or decreasing) values.\n        '
        return self._is_sorted(ascending=[False] * len(self.levels), null_position=None)

    @_cudf_nvtx_annotate
    def fillna(self, value):
        if False:
            return 10
        '\n        Fill null values with the specified value.\n\n        Parameters\n        ----------\n        value : scalar\n            Scalar value to use to fill nulls. This value cannot be a\n            list-likes.\n\n        Returns\n        -------\n        filled : MultiIndex\n\n        Examples\n        --------\n        >>> import cudf\n        >>> index = cudf.MultiIndex(\n        ...         levels=[["a", "b", "c", None], ["1", None, "5"]],\n        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],\n        ...         names=["x", "y"],\n        ...       )\n        >>> index\n        MultiIndex([( \'a\',  \'1\'),\n                    ( \'a\',  \'5\'),\n                    ( \'b\', <NA>),\n                    ( \'c\', <NA>),\n                    (<NA>,  \'1\')],\n                   names=[\'x\', \'y\'])\n        >>> index.fillna(\'hello\')\n        MultiIndex([(    \'a\',     \'1\'),\n                    (    \'a\',     \'5\'),\n                    (    \'b\', \'hello\'),\n                    (    \'c\', \'hello\'),\n                    (\'hello\',     \'1\')],\n                   names=[\'x\', \'y\'])\n        '
        return super().fillna(value=value)

    @_cudf_nvtx_annotate
    def unique(self):
        if False:
            for i in range(10):
                print('nop')
        return self.drop_duplicates(keep='first')

    def _clean_nulls_from_index(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert all na values(if any) in MultiIndex object\n        to `<NA>` as a preprocessing step to `__repr__` methods.\n        '
        index_df = self.to_frame(index=False, name=list(range(self.nlevels)))
        return MultiIndex.from_frame(index_df._clean_nulls_from_dataframe(index_df), names=self.names)

    @_cudf_nvtx_annotate
    def memory_usage(self, deep=False):
        if False:
            i = 10
            return i + 15
        usage = sum((col.memory_usage for col in self._data.columns))
        if self.levels:
            for level in self.levels:
                usage += level.memory_usage(deep=deep)
        if self._codes_frame:
            for col in self._codes_frame._data.columns:
                usage += col.memory_usage
        return usage

    @_cudf_nvtx_annotate
    def difference(self, other, sort=None):
        if False:
            return 10
        if hasattr(other, 'to_pandas'):
            other = other.to_pandas()
        return cudf.from_pandas(self.to_pandas().difference(other, sort))

    @_cudf_nvtx_annotate
    def append(self, other):
        if False:
            print('Hello World!')
        "\n        Append a collection of MultiIndex objects together\n\n        Parameters\n        ----------\n        other : MultiIndex or list/tuple of MultiIndex objects\n\n        Returns\n        -------\n        appended : Index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx1 = cudf.MultiIndex(\n        ...     levels=[[1, 2], ['blue', 'red']],\n        ...     codes=[[0, 0, 1, 1], [1, 0, 1, 0]]\n        ... )\n        >>> idx2 = cudf.MultiIndex(\n        ...     levels=[[3, 4], ['blue', 'red']],\n        ...     codes=[[0, 0, 1, 1], [1, 0, 1, 0]]\n        ... )\n        >>> idx1\n        MultiIndex([(1,  'red'),\n                    (1, 'blue'),\n                    (2,  'red'),\n                    (2, 'blue')],\n                   )\n        >>> idx2\n        MultiIndex([(3,  'red'),\n                    (3, 'blue'),\n                    (4,  'red'),\n                    (4, 'blue')],\n                   )\n        >>> idx1.append(idx2)\n        MultiIndex([(1,  'red'),\n                    (1, 'blue'),\n                    (2,  'red'),\n                    (2, 'blue'),\n                    (3,  'red'),\n                    (3, 'blue'),\n                    (4,  'red'),\n                    (4, 'blue')],\n                   )\n        "
        if isinstance(other, (list, tuple)):
            to_concat = [self]
            to_concat.extend(other)
        else:
            to_concat = [self, other]
        for obj in to_concat:
            if not isinstance(obj, MultiIndex):
                raise TypeError(f'all objects should be of type MultiIndex for MultiIndex.append, found object of type: {type(obj)}')
        return MultiIndex._concat(to_concat)

    @_cudf_nvtx_annotate
    def __array_function__(self, func, types, args, kwargs):
        if False:
            while True:
                i = 10
        cudf_df_module = MultiIndex
        for submodule in func.__module__.split('.')[1:]:
            if hasattr(cudf_df_module, submodule):
                cudf_df_module = getattr(cudf_df_module, submodule)
            else:
                return NotImplemented
        fname = func.__name__
        handled_types = [cudf_df_module, np.ndarray]
        for t in types:
            if t not in handled_types:
                return NotImplemented
        if hasattr(cudf_df_module, fname):
            cudf_func = getattr(cudf_df_module, fname)
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)
        else:
            return NotImplemented

    def _level_index_from_level(self, level):
        if False:
            print('Hello World!')
        '\n        Return level index from given level name or index\n        '
        try:
            return self.names.index(level)
        except ValueError:
            if not is_integer(level):
                raise KeyError(f'Level {level} not found')
            if level < 0:
                level += self.nlevels
            if level >= self.nlevels:
                raise IndexError(f'Level {level} out of bounds. Index has {self.nlevels} levels.') from None
            return level

    @_cudf_nvtx_annotate
    def get_loc(self, key, method=None, tolerance=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get location for a label or a tuple of labels.\n\n        The location is returned as an integer/slice or boolean mask.\n\n        Parameters\n        ----------\n        key : label or tuple of labels (one for each level)\n        method : None\n\n        Returns\n        -------\n        loc : int, slice object or boolean mask\n            - If index is unique, search result is unique, return a single int.\n            - If index is monotonic, index is returned as a slice object.\n            - Otherwise, cudf attempts a best effort to convert the search\n              result into a slice object, and will return a boolean mask if\n              failed to do so. Notice this can deviate from Pandas behavior\n              in some situations.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> mi = cudf.MultiIndex.from_tuples(\n        ...     [('a', 'd'), ('b', 'e'), ('b', 'f')])\n        >>> mi.get_loc('b')\n        slice(1, 3, None)\n        >>> mi.get_loc(('b', 'e'))\n        1\n        >>> non_monotonic_non_unique_idx = cudf.MultiIndex.from_tuples(\n        ...     [('c', 'd'), ('b', 'e'), ('a', 'f'), ('b', 'e')])\n        >>> non_monotonic_non_unique_idx.get_loc('b') # differ from pandas\n        slice(1, 4, 2)\n\n        .. pandas-compat::\n            **MultiIndex.get_loc**\n\n            The return types of this function may deviates from the\n            method provided by Pandas. If the index is neither\n            lexicographically sorted nor unique, a best effort attempt is made\n            to coerce the found indices into a slice. For example:\n\n            .. code-block::\n\n                >>> import pandas as pd\n                >>> import cudf\n                >>> x = pd.MultiIndex.from_tuples([\n                ...     (2, 1, 1), (1, 2, 3), (1, 2, 1),\n                ...     (1, 1, 1), (1, 1, 1), (2, 2, 1),\n                ... ])\n                >>> x.get_loc(1)\n                array([False,  True,  True,  True,  True, False])\n                >>> cudf.from_pandas(x).get_loc(1)\n                slice(1, 5, 1)\n        "
        if tolerance is not None:
            raise NotImplementedError('Parameter tolerance is not supported yet.')
        if method is not None:
            raise NotImplementedError('only the default get_loc method is currently supported for MultiIndex')
        is_sorted = self.is_monotonic_increasing or self.is_monotonic_decreasing
        is_unique = self.is_unique
        key = (key,) if not isinstance(key, tuple) else key
        key_as_table = cudf.core.frame.Frame({i: column.as_column(k, length=1) for (i, k) in enumerate(key)})
        partial_index = self.__class__._from_data(data=self._data.select_by_index(slice(key_as_table._num_columns)))
        (lower_bound, upper_bound, sort_inds) = _lexsorted_equal_range(partial_index, key_as_table, is_sorted)
        if lower_bound == upper_bound:
            raise KeyError(key)
        if is_unique and lower_bound + 1 == upper_bound:
            return lower_bound if is_sorted else sort_inds.element_indexing(lower_bound)
        if is_sorted:
            return slice(lower_bound, upper_bound)
        true_inds = sort_inds.slice(lower_bound, upper_bound).values
        true_inds = _maybe_indices_to_slice(true_inds)
        if isinstance(true_inds, slice):
            return true_inds
        mask = cp.full(self._data.nrows, False)
        mask[true_inds] = True
        return mask

    def _get_reconciled_name_object(self, other) -> MultiIndex:
        if False:
            while True:
                i = 10
        '\n        If the result of a set operation will be self,\n        return self, unless the names change, in which\n        case make a shallow copy of self.\n        '
        names = self._maybe_match_names(other)
        if self.names != names:
            return self.rename(names)
        return self

    def _maybe_match_names(self, other):
        if False:
            while True:
                i = 10
        '\n        Try to find common names to attach to the result of an operation\n        between a and b. Return a consensus list of names if they match\n        at least partly or list of None if they have completely\n        different names.\n        '
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        return [self_name if _is_same_name(self_name, other_name) else None for (self_name, other_name) in zip(self.names, other.names)]

    @_cudf_nvtx_annotate
    def union(self, other, sort=None):
        if False:
            return 10
        if not isinstance(other, MultiIndex):
            msg = 'other must be a MultiIndex or a list of tuples'
            try:
                other = MultiIndex.from_tuples(other, names=self.names)
            except (ValueError, TypeError) as err:
                raise TypeError(msg) from err
        if sort not in {None, False}:
            raise ValueError(f"The 'sort' keyword only takes the values of None or False; {sort} was passed.")
        if not len(other) or self.equals(other):
            return self._get_reconciled_name_object(other)
        elif not len(self):
            return other._get_reconciled_name_object(self)
        return self._union(other, sort=sort)

    @_cudf_nvtx_annotate
    def _union(self, other, sort=None):
        if False:
            while True:
                i = 10
        other_df = other.copy(deep=True).to_frame(index=False)
        self_df = self.copy(deep=True).to_frame(index=False)
        col_names = list(range(0, self.nlevels))
        self_df.columns = col_names
        other_df.columns = col_names
        self_df['order'] = self_df.index
        other_df['order'] = other_df.index
        result_df = self_df.merge(other_df, on=col_names, how='outer')
        result_df = result_df.sort_values(by=result_df._data.to_pandas_index()[self.nlevels:], ignore_index=True)
        midx = MultiIndex.from_frame(result_df.iloc[:, :self.nlevels])
        midx.names = self.names if self.names == other.names else None
        if sort is None and len(other):
            return midx.sort_values()
        return midx

    @_cudf_nvtx_annotate
    def _intersection(self, other, sort=None):
        if False:
            for i in range(10):
                print('nop')
        if self.names != other.names:
            deep = True
            col_names = list(range(0, self.nlevels))
            res_name = (None,) * self.nlevels
        else:
            deep = False
            col_names = None
            res_name = self.names
        other_df = other.copy(deep=deep).to_frame(index=False)
        self_df = self.copy(deep=deep).to_frame(index=False)
        if col_names is not None:
            other_df.columns = col_names
            self_df.columns = col_names
        result_df = cudf.merge(self_df, other_df, how='inner')
        midx = self.__class__.from_frame(result_df, names=res_name)
        if sort is None and len(other):
            return midx.sort_values()
        return midx

    @_cudf_nvtx_annotate
    def _copy_type_metadata(self: MultiIndex, other: MultiIndex, *, override_dtypes=None) -> MultiIndex:
        if False:
            i = 10
            return i + 15
        res = super()._copy_type_metadata(other)
        res._names = other._names
        return res

    @_cudf_nvtx_annotate
    def _split_columns_by_levels(self, levels):
        if False:
            return 10
        if levels is None:
            return (list(self._data.columns), [], [f'level_{i}' if name is None else name for (i, name) in enumerate(self.names)], [])
        level_names = list(self.names)
        level_indices = {lv if isinstance(lv, int) else level_names.index(lv) for lv in levels}
        (data_columns, index_columns) = ([], [])
        (data_names, index_names) = ([], [])
        for (i, (name, col)) in enumerate(zip(self.names, self._data.columns)):
            if i in level_indices:
                name = f'level_{i}' if name is None else name
                data_columns.append(col)
                data_names.append(name)
            else:
                index_columns.append(col)
                index_names.append(name)
        return (data_columns, index_columns, data_names, index_names)

    def repeat(self, repeats, axis=None):
        if False:
            while True:
                i = 10
        return self._from_columns_like_self(Frame._repeat([*self._columns], repeats, axis), self._column_names)