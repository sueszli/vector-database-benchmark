"""
Experimental manager based on storing a collection of 1D arrays
"""
from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from pandas._libs import NaT, lib
from pandas.core.dtypes.astype import astype_array, astype_array_safe
from pandas.core.dtypes.cast import ensure_dtype_can_hold_na, find_common_type, infer_dtype_from_scalar, np_find_common_type
from pandas.core.dtypes.common import ensure_platform_int, is_datetime64_ns_dtype, is_integer, is_numeric_dtype, is_object_dtype, is_timedelta64_ns_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import array_equals, isna, na_value_for_dtype
import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import DatetimeArray, ExtensionArray, NumpyExtensionArray, TimedeltaArray
from pandas.core.construction import ensure_wrapped_if_datetimelike, extract_array, sanitize_array
from pandas.core.indexers import maybe_convert_indices, validate_indices
from pandas.core.indexes.api import Index, ensure_index
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import DataManager, SingleDataManager, ensure_np_dtype, interleaved_dtype
from pandas.core.internals.blocks import BlockPlacement, ensure_block_shape, external_values, extract_pandas_array, maybe_coerce_values, new_block
from pandas.core.internals.managers import make_na_array
if TYPE_CHECKING:
    from collections.abc import Hashable
    from pandas._typing import ArrayLike, AxisInt, DtypeObj, QuantileInterpolation, Self, npt

class BaseArrayManager(DataManager):
    """
    Core internal data structure to implement DataFrame and Series.

    Alternative to the BlockManager, storing a list of 1D arrays instead of
    Blocks.

    This is *not* a public API class

    Parameters
    ----------
    arrays : Sequence of arrays
    axes : Sequence of Index
    verify_integrity : bool, default True

    """
    __slots__ = ['_axes', 'arrays']
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool=True) -> None:
        if False:
            return 10
        raise NotImplementedError

    def make_empty(self, axes=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return an empty ArrayManager with the items axis of len 0 (no columns)'
        if axes is None:
            axes = [self.axes[1:], Index([])]
        arrays: list[np.ndarray | ExtensionArray] = []
        return type(self)(arrays, axes)

    @property
    def items(self) -> Index:
        if False:
            for i in range(10):
                print('nop')
        return self._axes[-1]

    @property
    def axes(self) -> list[Index]:
        if False:
            print('Hello World!')
        'Axes is BlockManager-compatible order (columns, rows)'
        return [self._axes[1], self._axes[0]]

    @property
    def shape_proper(self) -> tuple[int, ...]:
        if False:
            print('Hello World!')
        return tuple((len(ax) for ax in self._axes))

    @staticmethod
    def _normalize_axis(axis: AxisInt) -> int:
        if False:
            for i in range(10):
                print('nop')
        axis = 1 if axis == 0 else 0
        return axis

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._validate_set_axis(axis, new_labels)
        axis = self._normalize_axis(axis)
        self._axes[axis] = new_labels

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        if False:
            return 10
        return np.array([arr.dtype for arr in self.arrays], dtype='object')

    def add_references(self, mgr: BaseArrayManager) -> None:
        if False:
            while True:
                i = 10
        '\n        Only implemented on the BlockManager level\n        '
        return

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return (self.arrays, self._axes)

    def __setstate__(self, state) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.arrays = state[0]
        self._axes = state[1]

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        output = type(self).__name__
        output += f'\nIndex: {self._axes[0]}'
        if self.ndim == 2:
            output += f'\nColumns: {self._axes[1]}'
        output += f'\n{len(self.arrays)} arrays:'
        for arr in self.arrays:
            output += f'\n{arr.dtype}'
        return output

    def apply(self, f, align_keys: list[str] | None=None, **kwargs) -> Self:
        if False:
            return 10
        '\n        Iterate over the arrays, collect and create a new ArrayManager.\n\n        Parameters\n        ----------\n        f : str or callable\n            Name of the Array method to apply.\n        align_keys: List[str] or None, default None\n        **kwargs\n            Keywords to pass to `f`\n\n        Returns\n        -------\n        ArrayManager\n        '
        assert 'filter' not in kwargs
        align_keys = align_keys or []
        result_arrays: list[ArrayLike] = []
        aligned_args = {k: kwargs[k] for k in align_keys}
        if f == 'apply':
            f = kwargs.pop('func')
        for (i, arr) in enumerate(self.arrays):
            if aligned_args:
                for (k, obj) in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[i]
                        else:
                            kwargs[k] = obj.iloc[:, i]._values
                    else:
                        kwargs[k] = obj[i]
            if callable(f):
                applied = f(arr, **kwargs)
            else:
                applied = getattr(arr, f)(**kwargs)
            result_arrays.append(applied)
        new_axes = self._axes
        return type(self)(result_arrays, new_axes)

    def apply_with_block(self, f, align_keys=None, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        swap_axis = True
        if f == 'interpolate':
            swap_axis = False
        if swap_axis and 'axis' in kwargs and (self.ndim == 2):
            kwargs['axis'] = 1 if kwargs['axis'] == 0 else 0
        align_keys = align_keys or []
        aligned_args = {k: kwargs[k] for k in align_keys}
        result_arrays = []
        for (i, arr) in enumerate(self.arrays):
            if aligned_args:
                for (k, obj) in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        if obj.ndim == 1:
                            if self.ndim == 2:
                                kwargs[k] = obj.iloc[slice(i, i + 1)]._values
                            else:
                                kwargs[k] = obj.iloc[:]._values
                        else:
                            kwargs[k] = obj.iloc[:, [i]]._values
                    elif obj.ndim == 2:
                        kwargs[k] = obj[[i]]
            if isinstance(arr.dtype, np.dtype) and (not isinstance(arr, np.ndarray)):
                arr = np.asarray(arr)
            arr = maybe_coerce_values(arr)
            if self.ndim == 2:
                arr = ensure_block_shape(arr, 2)
                bp = BlockPlacement(slice(0, 1, 1))
                block = new_block(arr, placement=bp, ndim=2)
            else:
                bp = BlockPlacement(slice(0, len(self), 1))
                block = new_block(arr, placement=bp, ndim=1)
            applied = getattr(block, f)(**kwargs)
            if isinstance(applied, list):
                applied = applied[0]
            arr = applied.values
            if self.ndim == 2 and arr.ndim == 2:
                assert len(arr) == 1
                arr = arr[0, :]
            result_arrays.append(arr)
        return type(self)(result_arrays, self._axes)

    def setitem(self, indexer, value) -> Self:
        if False:
            while True:
                i = 10
        return self.apply_with_block('setitem', indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        if False:
            i = 10
            return i + 15
        assert self.ndim == 2
        return self.apply(algos.diff, n=n)

    def astype(self, dtype, copy: bool | None=False, errors: str='raise') -> Self:
        if False:
            print('Hello World!')
        if copy is None:
            copy = True
        return self.apply(astype_array_safe, dtype=dtype, copy=copy, errors=errors)

    def convert(self, copy: bool | None) -> Self:
        if False:
            print('Hello World!')
        if copy is None:
            copy = True

        def _convert(arr):
            if False:
                return 10
            if is_object_dtype(arr.dtype):
                arr = np.asarray(arr)
                result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
                if result is arr and copy:
                    return arr.copy()
                return result
            else:
                return arr.copy() if copy else arr
        return self.apply(_convert)

    def get_values_for_csv(self, *, float_format, date_format, decimal, na_rep: str='nan', quoting=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self.apply(get_values_for_csv, na_rep=na_rep, quoting=quoting, float_format=float_format, date_format=date_format, decimal=decimal)

    @property
    def any_extension_types(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether any of the blocks in this manager are extension blocks'
        return False

    @property
    def is_view(self) -> bool:
        if False:
            i = 10
            return i + 15
        'return a boolean if we are a single block and are a view'
        return False

    @property
    def is_single_block(self) -> bool:
        if False:
            i = 10
            return i + 15
        return len(self.arrays) == 1

    def _get_data_subset(self, predicate: Callable) -> Self:
        if False:
            print('Hello World!')
        indices = [i for (i, arr) in enumerate(self.arrays) if predicate(arr)]
        arrays = [self.arrays[i] for i in indices]
        taker = np.array(indices, dtype='intp')
        new_cols = self._axes[1].take(taker)
        new_axes = [self._axes[0], new_cols]
        return type(self)(arrays, new_axes, verify_integrity=False)

    def get_bool_data(self, copy: bool=False) -> Self:
        if False:
            print('Hello World!')
        '\n        Select columns that are bool-dtype and object-dtype columns that are all-bool.\n\n        Parameters\n        ----------\n        copy : bool, default False\n            Whether to copy the blocks\n        '
        return self._get_data_subset(lambda x: x.dtype == np.dtype(bool))

    def get_numeric_data(self, copy: bool=False) -> Self:
        if False:
            return 10
        '\n        Select columns that have a numeric dtype.\n\n        Parameters\n        ----------\n        copy : bool, default False\n            Whether to copy the blocks\n        '
        return self._get_data_subset(lambda arr: is_numeric_dtype(arr.dtype) or getattr(arr.dtype, '_is_numeric', False))

    def copy(self, deep: bool | Literal['all'] | None=True) -> Self:
        if False:
            while True:
                i = 10
        "\n        Make deep or shallow copy of ArrayManager\n\n        Parameters\n        ----------\n        deep : bool or string, default True\n            If False, return shallow copy (do not copy data)\n            If 'all', copy data and a deep copy of the index\n\n        Returns\n        -------\n        BlockManager\n        "
        if deep is None:
            deep = True
        if deep:

            def copy_func(ax):
                if False:
                    i = 10
                    return i + 15
                return ax.copy(deep=True) if deep == 'all' else ax.view()
            new_axes = [copy_func(ax) for ax in self._axes]
        else:
            new_axes = list(self._axes)
        if deep:
            new_arrays = [arr.copy() for arr in self.arrays]
        else:
            new_arrays = list(self.arrays)
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def reindex_indexer(self, new_axis, indexer, axis: AxisInt, fill_value=None, allow_dups: bool=False, copy: bool | None=True, only_slice: bool=False, use_na_proxy: bool=False) -> Self:
        if False:
            return 10
        axis = self._normalize_axis(axis)
        return self._reindex_indexer(new_axis, indexer, axis, fill_value, allow_dups, copy, use_na_proxy)

    def _reindex_indexer(self, new_axis, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value=None, allow_dups: bool=False, copy: bool | None=True, use_na_proxy: bool=False) -> Self:
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        new_axis : Index\n        indexer : ndarray[intp] or None\n        axis : int\n        fill_value : object, default None\n        allow_dups : bool, default False\n        copy : bool, default True\n\n\n        pandas-indexer with -1's only.\n        "
        if copy is None:
            copy = True
        if indexer is None:
            if new_axis is self._axes[axis] and (not copy):
                return self
            result = self.copy(deep=copy)
            result._axes = list(self._axes)
            result._axes[axis] = new_axis
            return result
        if not allow_dups:
            self._axes[axis]._validate_can_reindex(indexer)
        if axis >= self.ndim:
            raise IndexError('Requested axis not found in manager')
        if axis == 1:
            new_arrays = []
            for i in indexer:
                if i == -1:
                    arr = self._make_na_array(fill_value=fill_value, use_na_proxy=use_na_proxy)
                else:
                    arr = self.arrays[i]
                    if copy:
                        arr = arr.copy()
                new_arrays.append(arr)
        else:
            validate_indices(indexer, len(self._axes[0]))
            indexer = ensure_platform_int(indexer)
            mask = indexer == -1
            needs_masking = mask.any()
            new_arrays = [take_1d(arr, indexer, allow_fill=needs_masking, fill_value=fill_value, mask=mask) for arr in self.arrays]
        new_axes = list(self._axes)
        new_axes[axis] = new_axis
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def take(self, indexer: npt.NDArray[np.intp], axis: AxisInt=1, verify: bool=True) -> Self:
        if False:
            while True:
                i = 10
        '\n        Take items along any axis.\n        '
        assert isinstance(indexer, np.ndarray), type(indexer)
        assert indexer.dtype == np.intp, indexer.dtype
        axis = self._normalize_axis(axis)
        if not indexer.ndim == 1:
            raise ValueError('indexer should be 1-dimensional')
        n = self.shape_proper[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)
        new_labels = self._axes[axis].take(indexer)
        return self._reindex_indexer(new_axis=new_labels, indexer=indexer, axis=axis, allow_dups=True)

    def _make_na_array(self, fill_value=None, use_na_proxy: bool=False):
        if False:
            for i in range(10):
                print('nop')
        if use_na_proxy:
            assert fill_value is None
            return NullArrayProxy(self.shape_proper[0])
        if fill_value is None:
            fill_value = np.nan
        (dtype, fill_value) = infer_dtype_from_scalar(fill_value)
        array_values = make_na_array(dtype, self.shape_proper[:1], fill_value)
        return array_values

    def _equal_values(self, other) -> bool:
        if False:
            return 10
        '\n        Used in .equals defined in base class. Only check the column values\n        assuming shape and indexes have already been checked.\n        '
        for (left, right) in zip(self.arrays, other.arrays):
            if not array_equals(left, right):
                return False
        return True

class ArrayManager(BaseArrayManager):

    @property
    def ndim(self) -> Literal[2]:
        if False:
            i = 10
            return i + 15
        return 2

    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool=True) -> None:
        if False:
            print('Hello World!')
        self._axes = axes
        self.arrays = arrays
        if verify_integrity:
            self._axes = [ensure_index(ax) for ax in axes]
            arrays = [extract_pandas_array(x, None, 1)[0] for x in arrays]
            self.arrays = [maybe_coerce_values(arr) for arr in arrays]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        if False:
            return 10
        (n_rows, n_columns) = self.shape_proper
        if not len(self.arrays) == n_columns:
            raise ValueError(f'Number of passed arrays must equal the size of the column Index: {len(self.arrays)} arrays vs {n_columns} columns.')
        for arr in self.arrays:
            if not len(arr) == n_rows:
                raise ValueError(f'Passed arrays should have the same length as the rows Index: {len(arr)} vs {n_rows} rows')
            if not isinstance(arr, (np.ndarray, ExtensionArray)):
                raise ValueError(f'Passed arrays should be np.ndarray or ExtensionArray instances, got {type(arr)} instead')
            if not arr.ndim == 1:
                raise ValueError(f'Passed arrays should be 1-dimensional, got array with {arr.ndim} dimensions instead.')

    def fast_xs(self, loc: int) -> SingleArrayManager:
        if False:
            i = 10
            return i + 15
        '\n        Return the array corresponding to `frame.iloc[loc]`.\n\n        Parameters\n        ----------\n        loc : int\n\n        Returns\n        -------\n        np.ndarray or ExtensionArray\n        '
        dtype = interleaved_dtype([arr.dtype for arr in self.arrays])
        values = [arr[loc] for arr in self.arrays]
        if isinstance(dtype, ExtensionDtype):
            result = dtype.construct_array_type()._from_sequence(values, dtype=dtype)
        elif is_datetime64_ns_dtype(dtype):
            result = DatetimeArray._from_sequence(values, dtype=dtype)._ndarray
        elif is_timedelta64_ns_dtype(dtype):
            result = TimedeltaArray._from_sequence(values, dtype=dtype)._ndarray
        else:
            result = np.array(values, dtype=dtype)
        return SingleArrayManager([result], [self._axes[1]])

    def get_slice(self, slobj: slice, axis: AxisInt=0) -> ArrayManager:
        if False:
            print('Hello World!')
        axis = self._normalize_axis(axis)
        if axis == 0:
            arrays = [arr[slobj] for arr in self.arrays]
        elif axis == 1:
            arrays = self.arrays[slobj]
        new_axes = list(self._axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)
        return type(self)(arrays, new_axes, verify_integrity=False)

    def iget(self, i: int) -> SingleArrayManager:
        if False:
            print('Hello World!')
        '\n        Return the data as a SingleArrayManager.\n        '
        values = self.arrays[i]
        return SingleArrayManager([values], [self._axes[0]])

    def iget_values(self, i: int) -> ArrayLike:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the data for column i as the values (ndarray or ExtensionArray).\n        '
        return self.arrays[i]

    @property
    def column_arrays(self) -> list[ArrayLike]:
        if False:
            i = 10
            return i + 15
        '\n        Used in the JSON C code to access column arrays.\n        '
        return [np.asarray(arr) for arr in self.arrays]

    def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool=False, refs=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set new column(s).\n\n        This changes the ArrayManager in-place, but replaces (an) existing\n        column(s), not changing column values in-place).\n\n        Parameters\n        ----------\n        loc : integer, slice or boolean mask\n            Positional location (already bounds checked)\n        value : np.ndarray or ExtensionArray\n        inplace : bool, default False\n            Whether overwrite existing array as opposed to replacing it.\n        '
        if lib.is_integer(loc):
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] == 1
                value = value[:, 0]
            value = maybe_coerce_values(value)
            assert isinstance(value, (np.ndarray, ExtensionArray))
            assert value.ndim == 1
            assert len(value) == len(self._axes[0])
            self.arrays[loc] = value
            return
        elif isinstance(loc, slice):
            indices: range | np.ndarray = range(loc.start if loc.start is not None else 0, loc.stop if loc.stop is not None else self.shape_proper[1], loc.step if loc.step is not None else 1)
        else:
            assert isinstance(loc, np.ndarray)
            assert loc.dtype == 'bool'
            indices = np.nonzero(loc)[0]
        assert value.ndim == 2
        assert value.shape[0] == len(self._axes[0])
        for (value_idx, mgr_idx) in enumerate(indices):
            value_arr = value[:, value_idx]
            self.arrays[mgr_idx] = value_arr
        return

    def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Set values ("setitem") into a single column (not setting the full column).\n\n        This is a method on the ArrayManager level, to avoid creating an\n        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)\n        '
        if not is_integer(loc):
            raise TypeError('The column index should be an integer')
        arr = self.arrays[loc]
        mgr = SingleArrayManager([arr], [self._axes[0]])
        if inplace_only:
            mgr.setitem_inplace(idx, value)
        else:
            new_mgr = mgr.setitem((idx,), value)
            self.arrays[loc] = new_mgr.arrays[0]

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        if False:
            return 10
        '\n        Insert item at selected position.\n\n        Parameters\n        ----------\n        loc : int\n        item : hashable\n        value : np.ndarray or ExtensionArray\n        '
        new_axis = self.items.insert(loc, item)
        value = extract_array(value, extract_numpy=True)
        if value.ndim == 2:
            if value.shape[0] == 1:
                value = value[0, :]
            else:
                raise ValueError(f'Expected a 1D array, got an array with shape {value.shape}')
        value = maybe_coerce_values(value)
        arrays = self.arrays.copy()
        arrays.insert(loc, value)
        self.arrays = arrays
        self._axes[1] = new_axis

    def idelete(self, indexer) -> ArrayManager:
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete selected locations in-place (new block and array, same BlockManager)\n        '
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False
        self.arrays = [self.arrays[i] for i in np.nonzero(to_keep)[0]]
        self._axes = [self._axes[0], self._axes[1][to_keep]]
        return self

    def grouped_reduce(self, func: Callable) -> Self:
        if False:
            i = 10
            return i + 15
        '\n        Apply grouped reduction function columnwise, returning a new ArrayManager.\n\n        Parameters\n        ----------\n        func : grouped reduction function\n\n        Returns\n        -------\n        ArrayManager\n        '
        result_arrays: list[np.ndarray] = []
        result_indices: list[int] = []
        for (i, arr) in enumerate(self.arrays):
            arr = ensure_block_shape(arr, ndim=2)
            res = func(arr)
            if res.ndim == 2:
                assert res.shape[0] == 1
                res = res[0]
            result_arrays.append(res)
            result_indices.append(i)
        if len(result_arrays) == 0:
            nrows = 0
        else:
            nrows = result_arrays[0].shape[0]
        index = Index(range(nrows))
        columns = self.items
        return type(self)(result_arrays, [index, columns])

    def reduce(self, func: Callable) -> Self:
        if False:
            print('Hello World!')
        '\n        Apply reduction function column-wise, returning a single-row ArrayManager.\n\n        Parameters\n        ----------\n        func : reduction function\n\n        Returns\n        -------\n        ArrayManager\n        '
        result_arrays: list[np.ndarray] = []
        for (i, arr) in enumerate(self.arrays):
            res = func(arr, axis=0)
            dtype = arr.dtype if res is NaT else None
            result_arrays.append(sanitize_array([res], None, dtype=dtype))
        index = Index._simple_new(np.array([None], dtype=object))
        columns = self.items
        new_mgr = type(self)(result_arrays, [index, columns])
        return new_mgr

    def operate_blockwise(self, other: ArrayManager, array_op) -> ArrayManager:
        if False:
            i = 10
            return i + 15
        '\n        Apply array_op blockwise with another (aligned) BlockManager.\n        '
        left_arrays = self.arrays
        right_arrays = other.arrays
        result_arrays = [array_op(left, right) for (left, right) in zip(left_arrays, right_arrays)]
        return type(self)(result_arrays, self._axes)

    def quantile(self, *, qs: Index, transposed: bool=False, interpolation: QuantileInterpolation='linear') -> ArrayManager:
        if False:
            i = 10
            return i + 15
        arrs = [ensure_block_shape(x, 2) for x in self.arrays]
        new_arrs = [quantile_compat(x, np.asarray(qs._values), interpolation) for x in arrs]
        for (i, arr) in enumerate(new_arrs):
            if arr.ndim == 2:
                assert arr.shape[0] == 1, arr.shape
                new_arrs[i] = arr[0]
        axes = [qs, self._axes[1]]
        return type(self)(new_arrs, axes)

    def unstack(self, unstacker, fill_value) -> ArrayManager:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a BlockManager with all blocks unstacked.\n\n        Parameters\n        ----------\n        unstacker : reshape._Unstacker\n        fill_value : Any\n            fill_value for newly introduced missing values.\n\n        Returns\n        -------\n        unstacked : BlockManager\n        '
        (indexer, _) = unstacker._indexer_and_to_sort
        if unstacker.mask.all():
            new_indexer = indexer
            allow_fill = False
            new_mask2D = None
            needs_masking = None
        else:
            new_indexer = np.full(unstacker.mask.shape, -1)
            new_indexer[unstacker.mask] = indexer
            allow_fill = True
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        new_indexer2D = new_indexer.reshape(*unstacker.full_shape)
        new_indexer2D = ensure_platform_int(new_indexer2D)
        new_arrays = []
        for arr in self.arrays:
            for i in range(unstacker.full_shape[1]):
                if allow_fill:
                    new_arr = take_1d(arr, new_indexer2D[:, i], allow_fill=needs_masking[i], fill_value=fill_value, mask=new_mask2D[:, i])
                else:
                    new_arr = take_1d(arr, new_indexer2D[:, i], allow_fill=False)
                new_arrays.append(new_arr)
        new_index = unstacker.new_index
        new_columns = unstacker.get_new_columns(self._axes[1])
        new_axes = [new_index, new_columns]
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def as_array(self, dtype=None, copy: bool=False, na_value: object=lib.no_default) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Convert the blockmanager data into an numpy array.\n\n        Parameters\n        ----------\n        dtype : object, default None\n            Data type of the return array.\n        copy : bool, default False\n            If True then guarantee that a copy is returned. A value of\n            False does not guarantee that the underlying data is not\n            copied.\n        na_value : object, default lib.no_default\n            Value to be used as the missing value sentinel.\n\n        Returns\n        -------\n        arr : ndarray\n        '
        if len(self.arrays) == 0:
            empty_arr = np.empty(self.shape, dtype=float)
            return empty_arr.transpose()
        copy = copy or na_value is not lib.no_default
        if not dtype:
            dtype = interleaved_dtype([arr.dtype for arr in self.arrays])
        dtype = ensure_np_dtype(dtype)
        result = np.empty(self.shape_proper, dtype=dtype)
        for (i, arr) in enumerate(self.arrays):
            arr = arr.astype(dtype, copy=copy)
            result[:, i] = arr
        if na_value is not lib.no_default:
            result[isna(result)] = na_value
        return result

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        if False:
            return 10
        '\n        Concatenate uniformly-indexed ArrayManagers horizontally.\n        '
        arrays = list(itertools.chain.from_iterable([mgr.arrays for mgr in mgrs]))
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        if False:
            for i in range(10):
                print('nop')
        '\n        Concatenate uniformly-indexed ArrayManagers vertically.\n        '
        arrays = [concat_arrays([mgrs[i].arrays[j] for i in range(len(mgrs))]) for j in range(len(mgrs[0].arrays))]
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr

class SingleArrayManager(BaseArrayManager, SingleDataManager):
    __slots__ = ['_axes', 'arrays']
    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    @property
    def ndim(self) -> Literal[1]:
        if False:
            print('Hello World!')
        return 1

    def __init__(self, arrays: list[np.ndarray | ExtensionArray], axes: list[Index], verify_integrity: bool=True) -> None:
        if False:
            print('Hello World!')
        self._axes = axes
        self.arrays = arrays
        if verify_integrity:
            assert len(axes) == 1
            assert len(arrays) == 1
            self._axes = [ensure_index(ax) for ax in self._axes]
            arr = arrays[0]
            arr = maybe_coerce_values(arr)
            arr = extract_pandas_array(arr, None, 1)[0]
            self.arrays = [arr]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        if False:
            print('Hello World!')
        (n_rows,) = self.shape
        assert len(self.arrays) == 1
        arr = self.arrays[0]
        assert len(arr) == n_rows
        if not arr.ndim == 1:
            raise ValueError(f'Passed array should be 1-dimensional, got array with {arr.ndim} dimensions instead.')

    @staticmethod
    def _normalize_axis(axis):
        if False:
            i = 10
            return i + 15
        return axis

    def make_empty(self, axes=None) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Return an empty ArrayManager with index/array of length 0'
        if axes is None:
            axes = [Index([], dtype=object)]
        array: np.ndarray = np.array([], dtype=self.dtype)
        return type(self)([array], axes)

    @classmethod
    def from_array(cls, array, index) -> SingleArrayManager:
        if False:
            for i in range(10):
                print('nop')
        return cls([array], [index])

    @property
    def axes(self) -> list[Index]:
        if False:
            for i in range(10):
                print('nop')
        return self._axes

    @property
    def index(self) -> Index:
        if False:
            for i in range(10):
                print('nop')
        return self._axes[0]

    @property
    def dtype(self):
        if False:
            return 10
        return self.array.dtype

    def external_values(self):
        if False:
            return 10
        'The array that Series.values returns'
        return external_values(self.array)

    def internal_values(self):
        if False:
            i = 10
            return i + 15
        'The array that Series._values returns'
        return self.array

    def array_values(self):
        if False:
            return 10
        'The array that Series.array returns'
        arr = self.array
        if isinstance(arr, np.ndarray):
            arr = NumpyExtensionArray(arr)
        return arr

    @property
    def _can_hold_na(self) -> bool:
        if False:
            return 10
        if isinstance(self.array, np.ndarray):
            return self.array.dtype.kind not in 'iub'
        else:
            return self.array._can_hold_na

    @property
    def is_single_block(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def fast_xs(self, loc: int) -> SingleArrayManager:
        if False:
            while True:
                i = 10
        raise NotImplementedError('Use series._values[loc] instead')

    def get_slice(self, slobj: slice, axis: AxisInt=0) -> SingleArrayManager:
        if False:
            i = 10
            return i + 15
        if axis >= self.ndim:
            raise IndexError('Requested axis not found in manager')
        new_array = self.array[slobj]
        new_index = self.index._getitem_slice(slobj)
        return type(self)([new_array], [new_index], verify_integrity=False)

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> SingleArrayManager:
        if False:
            return 10
        new_array = self.array[indexer]
        new_index = self.index[indexer]
        return type(self)([new_array], [new_index])

    def apply(self, func, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if callable(func):
            new_array = func(self.array, **kwargs)
        else:
            new_array = getattr(self.array, func)(**kwargs)
        return type(self)([new_array], self._axes)

    def setitem(self, indexer, value) -> SingleArrayManager:
        if False:
            i = 10
            return i + 15
        "\n        Set values with indexer.\n\n        For SingleArrayManager, this backs s[indexer] = value\n\n        See `setitem_inplace` for a version that works inplace and doesn't\n        return a new Manager.\n        "
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f'Cannot set values with ndim > {self.ndim}')
        return self.apply_with_block('setitem', indexer=indexer, value=value)

    def idelete(self, indexer) -> SingleArrayManager:
        if False:
            while True:
                i = 10
        '\n        Delete selected locations in-place (new array, same ArrayManager)\n        '
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False
        self.arrays = [self.arrays[0][to_keep]]
        self._axes = [self._axes[0][to_keep]]
        return self

    def _get_data_subset(self, predicate: Callable) -> SingleArrayManager:
        if False:
            for i in range(10):
                print('nop')
        if predicate(self.array):
            return type(self)(self.arrays, self._axes, verify_integrity=False)
        else:
            return self.make_empty()

    def set_values(self, values: ArrayLike) -> None:
        if False:
            while True:
                i = 10
        '\n        Set (replace) the values of the SingleArrayManager in place.\n\n        Use at your own risk! This does not check if the passed values are\n        valid for the current SingleArrayManager (length, dtype, etc).\n        '
        self.arrays[0] = values

    def to_2d_mgr(self, columns: Index) -> ArrayManager:
        if False:
            print('Hello World!')
        '\n        Manager analogue of Series.to_frame\n        '
        arrays = [self.arrays[0]]
        axes = [self.axes[0], columns]
        return ArrayManager(arrays, axes, verify_integrity=False)

class NullArrayProxy:
    """
    Proxy object for an all-NA array.

    Only stores the length of the array, and not the dtype. The dtype
    will only be known when actually concatenating (after determining the
    common dtype, for which this proxy is ignored).
    Using this object avoids that the internals/concat.py needs to determine
    the proper dtype and array type.
    """
    ndim = 1

    def __init__(self, n: int) -> None:
        if False:
            return 10
        self.n = n

    @property
    def shape(self) -> tuple[int]:
        if False:
            i = 10
            return i + 15
        return (self.n,)

    def to_array(self, dtype: DtypeObj) -> ArrayLike:
        if False:
            while True:
                i = 10
        '\n        Helper function to create the actual all-NA array from the NullArrayProxy\n        object.\n\n        Parameters\n        ----------\n        arr : NullArrayProxy\n        dtype : the dtype for the resulting array\n\n        Returns\n        -------\n        np.ndarray or ExtensionArray\n        '
        if isinstance(dtype, ExtensionDtype):
            empty = dtype.construct_array_type()._from_sequence([], dtype=dtype)
            indexer = -np.ones(self.n, dtype=np.intp)
            return empty.take(indexer, allow_fill=True)
        else:
            dtype = ensure_dtype_can_hold_na(dtype)
            fill_value = na_value_for_dtype(dtype)
            arr = np.empty(self.n, dtype=dtype)
            arr.fill(fill_value)
            return ensure_wrapped_if_datetimelike(arr)

def concat_arrays(to_concat: list) -> ArrayLike:
    if False:
        return 10
    '\n    Alternative for concat_compat but specialized for use in the ArrayManager.\n\n    Differences: only deals with 1D arrays (no axis keyword), assumes\n    ensure_wrapped_if_datetimelike and does not skip empty arrays to determine\n    the dtype.\n    In addition ensures that all NullArrayProxies get replaced with actual\n    arrays.\n\n    Parameters\n    ----------\n    to_concat : list of arrays\n\n    Returns\n    -------\n    np.ndarray or ExtensionArray\n    '
    to_concat_no_proxy = [x for x in to_concat if not isinstance(x, NullArrayProxy)]
    dtypes = {x.dtype for x in to_concat_no_proxy}
    single_dtype = len(dtypes) == 1
    if single_dtype:
        target_dtype = to_concat_no_proxy[0].dtype
    elif all((lib.is_np_dtype(x, 'iub') for x in dtypes)):
        target_dtype = np_find_common_type(*dtypes)
    else:
        target_dtype = find_common_type([arr.dtype for arr in to_concat_no_proxy])
    to_concat = [arr.to_array(target_dtype) if isinstance(arr, NullArrayProxy) else astype_array(arr, target_dtype, copy=False) for arr in to_concat]
    if isinstance(to_concat[0], ExtensionArray):
        cls = type(to_concat[0])
        return cls._concat_same_type(to_concat)
    result = np.concatenate(to_concat)
    if len(result) == 0:
        kinds = {obj.dtype.kind for obj in to_concat_no_proxy}
        if len(kinds) != 1:
            if 'b' in kinds:
                result = result.astype(object)
    return result