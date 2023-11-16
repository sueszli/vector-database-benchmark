from __future__ import annotations
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, cast, overload
import numpy as np
from pandas._libs import lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import get_unit_from_dtype, is_supported_unit
from pandas._typing import ArrayLike, AxisInt, Dtype, F, FillnaOptions, PositionalIndexer2D, PositionalIndexerTuple, ScalarIndexer, Self, SequenceIndexer, Shape, TakeIndexer, npt
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import validate_bool_kwarg, validate_fillna_kwargs, validate_insert_loc
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype, ExtensionDtype, PeriodDtype
from pandas.core.dtypes.missing import array_equivalent
from pandas.core import missing
from pandas.core.algorithms import take, unique, value_counts_internal as value_counts
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.sorting import nargminmax
if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import NumpySorter, NumpyValueArrayLike
    from pandas import Series

def ravel_compat(meth: F) -> F:
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator to ravel a 2D array before passing it to a cython operation,\n    then reshape the result to our own shape.\n    '

    @wraps(meth)
    def method(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.ndim == 1:
            return meth(self, *args, **kwargs)
        flags = self._ndarray.flags
        flat = self.ravel('K')
        result = meth(flat, *args, **kwargs)
        order = 'F' if flags.f_contiguous else 'C'
        return result.reshape(self.shape, order=order)
    return cast(F, method)

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    """
    ExtensionArray that is backed by a single NumPy ndarray.
    """
    _ndarray: np.ndarray
    _internal_fill_value: Any

    def _box_func(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrap numpy type in our dtype.type if necessary.\n        '
        return x

    def _validate_scalar(self, value):
        if False:
            print('Hello World!')
        raise AbstractMethodError(self)

    def view(self, dtype: Dtype | None=None) -> ArrayLike:
        if False:
            print('Hello World!')
        if dtype is None or dtype is self.dtype:
            return self._from_backing_data(self._ndarray)
        if isinstance(dtype, type):
            return self._ndarray.view(dtype)
        dtype = pandas_dtype(dtype)
        arr = self._ndarray
        if isinstance(dtype, PeriodDtype):
            cls = dtype.construct_array_type()
            return cls(arr.view('i8'), dtype=dtype)
        elif isinstance(dtype, DatetimeTZDtype):
            cls = dtype.construct_array_type()
            dt64_values = arr.view(f'M8[{dtype.unit}]')
            return cls(dt64_values, dtype=dtype)
        elif lib.is_np_dtype(dtype, 'M') and is_supported_unit(get_unit_from_dtype(dtype)):
            from pandas.core.arrays import DatetimeArray
            dt64_values = arr.view(dtype)
            return DatetimeArray(dt64_values, dtype=dtype)
        elif lib.is_np_dtype(dtype, 'm') and is_supported_unit(get_unit_from_dtype(dtype)):
            from pandas.core.arrays import TimedeltaArray
            td64_values = arr.view(dtype)
            return TimedeltaArray(td64_values, dtype=dtype)
        return arr.view(dtype=dtype)

    def take(self, indices: TakeIndexer, *, allow_fill: bool=False, fill_value: Any=None, axis: AxisInt=0) -> Self:
        if False:
            for i in range(10):
                print('nop')
        if allow_fill:
            fill_value = self._validate_scalar(fill_value)
        new_data = take(self._ndarray, indices, allow_fill=allow_fill, fill_value=fill_value, axis=axis)
        return self._from_backing_data(new_data)

    def equals(self, other) -> bool:
        if False:
            return 10
        if type(self) is not type(other):
            return False
        if self.dtype != other.dtype:
            return False
        return bool(array_equivalent(self._ndarray, other._ndarray, dtype_equal=True))

    @classmethod
    def _from_factorized(cls, values, original):
        if False:
            print('Hello World!')
        assert values.dtype == original._ndarray.dtype
        return original._from_backing_data(values)

    def _values_for_argsort(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        return self._ndarray

    def _values_for_factorize(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._ndarray, self._internal_fill_value)

    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> npt.NDArray[np.uint64]:
        if False:
            print('Hello World!')
        from pandas.core.util.hashing import hash_array
        values = self._ndarray
        return hash_array(values, encoding=encoding, hash_key=hash_key, categorize=categorize)

    def argmin(self, axis: AxisInt=0, skipna: bool=True):
        if False:
            for i in range(10):
                print('nop')
        validate_bool_kwarg(skipna, 'skipna')
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, 'argmin', axis=axis)

    def argmax(self, axis: AxisInt=0, skipna: bool=True):
        if False:
            for i in range(10):
                print('nop')
        validate_bool_kwarg(skipna, 'skipna')
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, 'argmax', axis=axis)

    def unique(self) -> Self:
        if False:
            print('Hello World!')
        new_data = unique(self._ndarray)
        return self._from_backing_data(new_data)

    @classmethod
    @doc(ExtensionArray._concat_same_type)
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt=0) -> Self:
        if False:
            i = 10
            return i + 15
        if not lib.dtypes_all_equal([x.dtype for x in to_concat]):
            dtypes = {str(x.dtype) for x in to_concat}
            raise ValueError('to_concat must have the same dtype', dtypes)
        return super()._concat_same_type(to_concat, axis=axis)

    @doc(ExtensionArray.searchsorted)
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right']='left', sorter: NumpySorter | None=None) -> npt.NDArray[np.intp] | np.intp:
        if False:
            while True:
                i = 10
        npvalue = self._validate_setitem_value(value)
        return self._ndarray.searchsorted(npvalue, side=side, sorter=sorter)

    @doc(ExtensionArray.shift)
    def shift(self, periods: int=1, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        axis = 0
        fill_value = self._validate_scalar(fill_value)
        new_values = shift(self._ndarray, periods, axis, fill_value)
        return self._from_backing_data(new_values)

    def __setitem__(self, key, value) -> None:
        if False:
            print('Hello World!')
        key = check_array_indexer(self, key)
        value = self._validate_setitem_value(value)
        self._ndarray[key] = value

    def _validate_setitem_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value

    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __getitem__(self, key: SequenceIndexer | PositionalIndexerTuple) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __getitem__(self, key: PositionalIndexer2D) -> Self | Any:
        if False:
            i = 10
            return i + 15
        if lib.is_integer(key):
            result = self._ndarray[key]
            if self.ndim == 1:
                return self._box_func(result)
            return self._from_backing_data(result)
        key = extract_array(key, extract_numpy=True)
        key = check_array_indexer(self, key)
        result = self._ndarray[key]
        if lib.is_scalar(result):
            return self._box_func(result)
        result = self._from_backing_data(result)
        return result

    def _fill_mask_inplace(self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]) -> None:
        if False:
            while True:
                i = 10
        func = missing.get_fill_func(method, ndim=self.ndim)
        func(self._ndarray.T, limit=limit, mask=mask.T)

    def _pad_or_backfill(self, *, method: FillnaOptions, limit: int | None=None, copy: bool=True) -> Self:
        if False:
            for i in range(10):
                print('nop')
        mask = self.isna()
        if mask.any():
            func = missing.get_fill_func(method, ndim=self.ndim)
            npvalues = self._ndarray.T
            if copy:
                npvalues = npvalues.copy()
            func(npvalues, limit=limit, mask=mask.T)
            npvalues = npvalues.T
            if copy:
                new_values = self._from_backing_data(npvalues)
            else:
                new_values = self
        elif copy:
            new_values = self.copy()
        else:
            new_values = self
        return new_values

    @doc(ExtensionArray.fillna)
    def fillna(self, value=None, method=None, limit: int | None=None, copy: bool=True) -> Self:
        if False:
            print('Hello World!')
        (value, method) = validate_fillna_kwargs(value, method, validate_scalar_dict_value=False)
        mask = self.isna()
        value = missing.check_value_size(value, mask, len(self))
        if mask.any():
            if method is not None:
                func = missing.get_fill_func(method, ndim=self.ndim)
                npvalues = self._ndarray.T
                if copy:
                    npvalues = npvalues.copy()
                func(npvalues, limit=limit, mask=mask.T)
                npvalues = npvalues.T
                new_values = self._from_backing_data(npvalues)
            else:
                if copy:
                    new_values = self.copy()
                else:
                    new_values = self[:]
                new_values[mask] = value
        else:
            if value is not None:
                self._validate_setitem_value(value)
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values

    def _wrap_reduction_result(self, axis: AxisInt | None, result):
        if False:
            return 10
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Analogue to np.putmask(self, mask, value)\n\n        Parameters\n        ----------\n        mask : np.ndarray[bool]\n        value : scalar or listlike\n\n        Raises\n        ------\n        TypeError\n            If value cannot be cast to self.dtype.\n        '
        value = self._validate_setitem_value(value)
        np.putmask(self._ndarray, mask, value)

    def _where(self: Self, mask: npt.NDArray[np.bool_], value) -> Self:
        if False:
            print('Hello World!')
        '\n        Analogue to np.where(mask, self, value)\n\n        Parameters\n        ----------\n        mask : np.ndarray[bool]\n        value : scalar or listlike\n\n        Raises\n        ------\n        TypeError\n            If value cannot be cast to self.dtype.\n        '
        value = self._validate_setitem_value(value)
        res_values = np.where(mask, self._ndarray, value)
        return self._from_backing_data(res_values)

    def insert(self, loc: int, item) -> Self:
        if False:
            print('Hello World!')
        '\n        Make new ExtensionArray inserting new item at location. Follows\n        Python list.append semantics for negative values.\n\n        Parameters\n        ----------\n        loc : int\n        item : object\n\n        Returns\n        -------\n        type(self)\n        '
        loc = validate_insert_loc(loc, len(self))
        code = self._validate_scalar(item)
        new_vals = np.concatenate((self._ndarray[:loc], np.asarray([code], dtype=self._ndarray.dtype), self._ndarray[loc:]))
        return self._from_backing_data(new_vals)

    def value_counts(self, dropna: bool=True) -> Series:
        if False:
            i = 10
            return i + 15
        "\n        Return a Series containing counts of unique values.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include counts of NA values.\n\n        Returns\n        -------\n        Series\n        "
        if self.ndim != 1:
            raise NotImplementedError
        from pandas import Index, Series
        if dropna:
            values = self[~self.isna()]._ndarray
        else:
            values = self._ndarray
        result = value_counts(values, sort=False, dropna=dropna)
        index_arr = self._from_backing_data(np.asarray(result.index._data))
        index = Index(index_arr, name=result.index.name)
        return Series(result._values, index=index, name=result.name, copy=False)

    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        if False:
            for i in range(10):
                print('nop')
        mask = np.asarray(self.isna())
        arr = self._ndarray
        fill_value = self._internal_fill_value
        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        res_values = self._cast_quantile_result(res_values)
        return self._from_backing_data(res_values)

    def _cast_quantile_result(self, res_values: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Cast the result of quantile_with_mask to an appropriate dtype\n        to pass to _from_backing_data in _quantile.\n        '
        return res_values

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        if False:
            for i in range(10):
                print('nop')
        '\n        Analogous to np.empty(shape, dtype=dtype)\n\n        Parameters\n        ----------\n        shape : tuple[int]\n        dtype : ExtensionDtype\n        '
        arr = cls._from_sequence([], dtype=dtype)
        backing = np.empty(shape, dtype=arr._ndarray.dtype)
        return arr._from_backing_data(backing)