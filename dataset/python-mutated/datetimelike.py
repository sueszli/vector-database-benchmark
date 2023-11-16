from __future__ import annotations
from datetime import datetime, timedelta
from functools import wraps
import operator
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast, final, overload
import warnings
import numpy as np
from pandas._libs import algos, lib
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset, IncompatibleFrequency, NaT, NaTType, Period, Resolution, Tick, Timedelta, Timestamp, astype_overflowsafe, get_unit_from_dtype, iNaT, ints_to_pydatetime, ints_to_pytimedelta, periods_per_day, to_offset
from pandas._libs.tslibs.fields import RoundTo, round_nsint64
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions
from pandas._libs.tslibs.timedeltas import get_unit_for_round
from pandas._libs.tslibs.timestamps import integer_op_not_supported
from pandas._typing import ArrayLike, AxisInt, DatetimeLikeScalar, Dtype, DtypeObj, F, InterpolateOptions, NpDtype, PositionalIndexer2D, PositionalIndexerTuple, ScalarIndexer, Self, SequenceIndexer, TimeAmbiguous, TimeNonexistent, npt
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError, InvalidComparison, PerformanceWarning
from pandas.util._decorators import Appender, Substitution, cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import is_all_strings, is_integer_dtype, is_list_like, is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCCategorical, ABCMultiIndex
from pandas.core.dtypes.missing import is_valid_na_for_dtype, isna
from pandas.core import algorithms, missing, nanops, ops
from pandas.core.algorithms import checked_add_with_arr, isin, map_array, unique1d
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray, ravel_compat
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.integer import IntegerArray
import pandas.core.common as com
from pandas.core.construction import array as pd_array, ensure_wrapped_if_datetimelike, extract_array
from pandas.core.indexers import check_array_indexer, check_setitem_lengths
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.ops.invalid import invalid_comparison, make_invalid_op
from pandas.tseries import frequencies
if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pandas import Index
    from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
DTScalarOrNaT = Union[DatetimeLikeScalar, NaTType]

def _make_unpacked_invalid_op(op_name: str):
    if False:
        while True:
            i = 10
    op = make_invalid_op(op_name)
    return unpack_zerodim_and_defer(op_name)(op)

def _period_dispatch(meth: F) -> F:
    if False:
        i = 10
        return i + 15
    '\n    For PeriodArray methods, dispatch to DatetimeArray and re-wrap the results\n    in PeriodArray.  We cannot use ._ndarray directly for the affected\n    methods because the i8 data has different semantics on NaT values.\n    '

    @wraps(meth)
    def new_meth(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if not isinstance(self.dtype, PeriodDtype):
            return meth(self, *args, **kwargs)
        arr = self.view('M8[ns]')
        result = meth(arr, *args, **kwargs)
        if result is NaT:
            return NaT
        elif isinstance(result, Timestamp):
            return self._box_func(result._value)
        res_i8 = result.view('i8')
        return self._from_backing_data(res_i8)
    return cast(F, new_meth)

class DatetimeLikeArrayMixin(OpsMixin, NDArrayBackedExtensionArray):
    """
    Shared Base/Mixin class for DatetimeArray, TimedeltaArray, PeriodArray

    Assumes that __new__/__init__ defines:
        _ndarray

    and that inheriting subclass implements:
        freq
    """
    _infer_matches: tuple[str, ...]
    _is_recognized_dtype: Callable[[DtypeObj], bool]
    _recognized_scalars: tuple[type, ...]
    _ndarray: np.ndarray
    freq: BaseOffset | None

    @cache_readonly
    def _can_hold_na(self) -> bool:
        if False:
            return 10
        return True

    def __init__(self, data, dtype: Dtype | None=None, freq=None, copy: bool=False) -> None:
        if False:
            while True:
                i = 10
        raise AbstractMethodError(self)

    @property
    def _scalar_type(self) -> type[DatetimeLikeScalar]:
        if False:
            print('Hello World!')
        '\n        The scalar associated with this datelike\n\n        * PeriodArray : Period\n        * DatetimeArray : Timestamp\n        * TimedeltaArray : Timedelta\n        '
        raise AbstractMethodError(self)

    def _scalar_from_string(self, value: str) -> DTScalarOrNaT:
        if False:
            return 10
        '\n        Construct a scalar type from a string.\n\n        Parameters\n        ----------\n        value : str\n\n        Returns\n        -------\n        Period, Timestamp, or Timedelta, or NaT\n            Whatever the type of ``self._scalar_type`` is.\n\n        Notes\n        -----\n        This should call ``self._check_compatible_with`` before\n        unboxing the result.\n        '
        raise AbstractMethodError(self)

    def _unbox_scalar(self, value: DTScalarOrNaT) -> np.int64 | np.datetime64 | np.timedelta64:
        if False:
            print('Hello World!')
        "\n        Unbox the integer value of a scalar `value`.\n\n        Parameters\n        ----------\n        value : Period, Timestamp, Timedelta, or NaT\n            Depending on subclass.\n\n        Returns\n        -------\n        int\n\n        Examples\n        --------\n        >>> arr = pd.arrays.DatetimeArray(np.array(['1970-01-01'], 'datetime64[ns]'))\n        >>> arr._unbox_scalar(arr[0])\n        numpy.datetime64('1970-01-01T00:00:00.000000000')\n        "
        raise AbstractMethodError(self)

    def _check_compatible_with(self, other: DTScalarOrNaT) -> None:
        if False:
            return 10
        '\n        Verify that `self` and `other` are compatible.\n\n        * DatetimeArray verifies that the timezones (if any) match\n        * PeriodArray verifies that the freq matches\n        * Timedelta has no verification\n\n        In each case, NaT is considered compatible.\n\n        Parameters\n        ----------\n        other\n\n        Raises\n        ------\n        Exception\n        '
        raise AbstractMethodError(self)

    def _box_func(self, x):
        if False:
            return 10
        '\n        box function to get object from internal representation\n        '
        raise AbstractMethodError(self)

    def _box_values(self, values) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        apply box func to passed values\n        '
        return lib.map_infer(values, self._box_func, convert=False)

    def __iter__(self) -> Iterator:
        if False:
            i = 10
            return i + 15
        if self.ndim > 1:
            return (self[n] for n in range(len(self)))
        else:
            return (self._box_func(v) for v in self.asi8)

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        if False:
            return 10
        '\n        Integer representation of the values.\n\n        Returns\n        -------\n        ndarray\n            An ndarray with int64 dtype.\n        '
        return self._ndarray.view('i8')

    def _format_native_types(self, *, na_rep: str | float='NaT', date_format=None) -> npt.NDArray[np.object_]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method for astype when converting to strings.\n\n        Returns\n        -------\n        ndarray[str]\n        '
        raise AbstractMethodError(self)

    def _formatter(self, boxed: bool=False):
        if False:
            return 10
        return "'{}'".format

    def __array__(self, dtype: NpDtype | None=None) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        if is_object_dtype(dtype):
            return np.array(list(self), dtype=object)
        return self._ndarray

    @overload
    def __getitem__(self, item: ScalarIndexer) -> DTScalarOrNaT:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer | PositionalIndexerTuple) -> Self:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __getitem__(self, key: PositionalIndexer2D) -> Self | DTScalarOrNaT:
        if False:
            i = 10
            return i + 15
        '\n        This getitem defers to the underlying array, which by-definition can\n        only handle list-likes, slices, and integer scalars\n        '
        result = cast('Union[Self, DTScalarOrNaT]', super().__getitem__(key))
        if lib.is_scalar(result):
            return result
        else:
            result = cast(Self, result)
        result._freq = self._get_getitem_freq(key)
        return result

    def _get_getitem_freq(self, key) -> BaseOffset | None:
        if False:
            while True:
                i = 10
        '\n        Find the `freq` attribute to assign to the result of a __getitem__ lookup.\n        '
        is_period = isinstance(self.dtype, PeriodDtype)
        if is_period:
            freq = self.freq
        elif self.ndim != 1:
            freq = None
        else:
            key = check_array_indexer(self, key)
            freq = None
            if isinstance(key, slice):
                if self.freq is not None and key.step is not None:
                    freq = key.step * self.freq
                else:
                    freq = self.freq
            elif key is Ellipsis:
                freq = self.freq
            elif com.is_bool_indexer(key):
                new_key = lib.maybe_booleans_to_slice(key.view(np.uint8))
                if isinstance(new_key, slice):
                    return self._get_getitem_freq(new_key)
        return freq

    def __setitem__(self, key: int | Sequence[int] | Sequence[bool] | slice, value: NaTType | Any | Sequence[Any]) -> None:
        if False:
            print('Hello World!')
        no_op = check_setitem_lengths(key, value, self)
        super().__setitem__(key, value)
        if no_op:
            return
        self._maybe_clear_freq()

    def _maybe_clear_freq(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def astype(self, dtype, copy: bool=True):
        if False:
            while True:
                i = 10
        dtype = pandas_dtype(dtype)
        if dtype == object:
            if self.dtype.kind == 'M':
                self = cast('DatetimeArray', self)
                i8data = self.asi8
                converted = ints_to_pydatetime(i8data, tz=self.tz, box='timestamp', reso=self._creso)
                return converted
            elif self.dtype.kind == 'm':
                return ints_to_pytimedelta(self._ndarray, box=True)
            return self._box_values(self.asi8.ravel()).reshape(self.shape)
        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)
        elif is_string_dtype(dtype):
            return self._format_native_types()
        elif dtype.kind in 'iu':
            values = self.asi8
            if dtype != np.int64:
                raise TypeError(f"Converting from {self.dtype} to {dtype} is not supported. Do obj.astype('int64').astype(dtype) instead")
            if copy:
                values = values.copy()
            return values
        elif dtype.kind in 'mM' and self.dtype != dtype or dtype.kind == 'f':
            msg = f'Cannot cast {type(self).__name__} to dtype {dtype}'
            raise TypeError(msg)
        else:
            return np.asarray(self, dtype=dtype)

    @overload
    def view(self) -> Self:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def view(self, dtype: Literal['M8[ns]']) -> DatetimeArray:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def view(self, dtype: Literal['m8[ns]']) -> TimedeltaArray:
        if False:
            return 10
        ...

    @overload
    def view(self, dtype: Dtype | None=...) -> ArrayLike:
        if False:
            while True:
                i = 10
        ...

    def view(self, dtype: Dtype | None=None) -> ArrayLike:
        if False:
            i = 10
            return i + 15
        return super().view(dtype)

    def _validate_comparison_value(self, other):
        if False:
            return 10
        if isinstance(other, str):
            try:
                other = self._scalar_from_string(other)
            except (ValueError, IncompatibleFrequency):
                raise InvalidComparison(other)
        if isinstance(other, self._recognized_scalars) or other is NaT:
            other = self._scalar_type(other)
            try:
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                raise InvalidComparison(other) from err
        elif not is_list_like(other):
            raise InvalidComparison(other)
        elif len(other) != len(self):
            raise ValueError('Lengths must match')
        else:
            try:
                other = self._validate_listlike(other, allow_object=True)
                self._check_compatible_with(other)
            except (TypeError, IncompatibleFrequency) as err:
                if is_object_dtype(getattr(other, 'dtype', None)):
                    pass
                else:
                    raise InvalidComparison(other) from err
        return other

    def _validate_scalar(self, value, *, allow_listlike: bool=False, unbox: bool=True):
        if False:
            return 10
        '\n        Validate that the input value can be cast to our scalar_type.\n\n        Parameters\n        ----------\n        value : object\n        allow_listlike: bool, default False\n            When raising an exception, whether the message should say\n            listlike inputs are allowed.\n        unbox : bool, default True\n            Whether to unbox the result before returning.  Note: unbox=False\n            skips the setitem compatibility check.\n\n        Returns\n        -------\n        self._scalar_type or NaT\n        '
        if isinstance(value, self._scalar_type):
            pass
        elif isinstance(value, str):
            try:
                value = self._scalar_from_string(value)
            except ValueError as err:
                msg = self._validation_error_message(value, allow_listlike)
                raise TypeError(msg) from err
        elif is_valid_na_for_dtype(value, self.dtype):
            value = NaT
        elif isna(value):
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        elif isinstance(value, self._recognized_scalars):
            value = self._scalar_type(value)
        else:
            msg = self._validation_error_message(value, allow_listlike)
            raise TypeError(msg)
        if not unbox:
            return value
        return self._unbox_scalar(value)

    def _validation_error_message(self, value, allow_listlike: bool=False) -> str:
        if False:
            return 10
        '\n        Construct an exception message on validation error.\n\n        Some methods allow only scalar inputs, while others allow either scalar\n        or listlike.\n\n        Parameters\n        ----------\n        allow_listlike: bool, default False\n\n        Returns\n        -------\n        str\n        '
        if hasattr(value, 'dtype') and getattr(value, 'ndim', 0) > 0:
            msg_got = f'{value.dtype} array'
        else:
            msg_got = f"'{type(value).__name__}'"
        if allow_listlike:
            msg = f"value should be a '{self._scalar_type.__name__}', 'NaT', or array of those. Got {msg_got} instead."
        else:
            msg = f"value should be a '{self._scalar_type.__name__}' or 'NaT'. Got {msg_got} instead."
        return msg

    def _validate_listlike(self, value, allow_object: bool=False):
        if False:
            return 10
        if isinstance(value, type(self)):
            return value
        if isinstance(value, list) and len(value) == 0:
            return type(self)._from_sequence([], dtype=self.dtype)
        if hasattr(value, 'dtype') and value.dtype == object:
            if lib.infer_dtype(value) in self._infer_matches:
                try:
                    value = type(self)._from_sequence(value)
                except (ValueError, TypeError):
                    if allow_object:
                        return value
                    msg = self._validation_error_message(value, True)
                    raise TypeError(msg)
        value = extract_array(value, extract_numpy=True)
        value = pd_array(value)
        value = extract_array(value, extract_numpy=True)
        if is_all_strings(value):
            try:
                value = type(self)._from_sequence(value, dtype=self.dtype)
            except ValueError:
                pass
        if isinstance(value.dtype, CategoricalDtype):
            if value.categories.dtype == self.dtype:
                value = value._internal_get_values()
                value = extract_array(value, extract_numpy=True)
        if allow_object and is_object_dtype(value.dtype):
            pass
        elif not type(self)._is_recognized_dtype(value.dtype):
            msg = self._validation_error_message(value, True)
            raise TypeError(msg)
        return value

    def _validate_setitem_value(self, value):
        if False:
            while True:
                i = 10
        if is_list_like(value):
            value = self._validate_listlike(value)
        else:
            return self._validate_scalar(value, allow_listlike=True)
        return self._unbox(value)

    @final
    def _unbox(self, other) -> np.int64 | np.datetime64 | np.timedelta64 | np.ndarray:
        if False:
            return 10
        '\n        Unbox either a scalar with _unbox_scalar or an instance of our own type.\n        '
        if lib.is_scalar(other):
            other = self._unbox_scalar(other)
        else:
            self._check_compatible_with(other)
            other = other._ndarray
        return other

    @ravel_compat
    def map(self, mapper, na_action=None):
        if False:
            while True:
                i = 10
        from pandas import Index
        result = map_array(self, mapper, na_action=na_action)
        result = Index(result)
        if isinstance(result, ABCMultiIndex):
            return result.to_numpy()
        else:
            return result.array

    def isin(self, values) -> npt.NDArray[np.bool_]:
        if False:
            print('Hello World!')
        '\n        Compute boolean array of whether each value is found in the\n        passed set of values.\n\n        Parameters\n        ----------\n        values : set or sequence of values\n\n        Returns\n        -------\n        ndarray[bool]\n        '
        if not hasattr(values, 'dtype'):
            values = np.asarray(values)
        if values.dtype.kind in 'fiuc':
            return np.zeros(self.shape, dtype=bool)
        if not isinstance(values, type(self)):
            inferable = ['timedelta', 'timedelta64', 'datetime', 'datetime64', 'date', 'period']
            if values.dtype == object:
                inferred = lib.infer_dtype(values, skipna=False)
                if inferred not in inferable:
                    if inferred == 'string':
                        pass
                    elif 'mixed' in inferred:
                        return isin(self.astype(object), values)
                    else:
                        return np.zeros(self.shape, dtype=bool)
            try:
                values = type(self)._from_sequence(values)
            except ValueError:
                return isin(self.astype(object), values)
        if self.dtype.kind in 'mM':
            self = cast('DatetimeArray | TimedeltaArray', self)
            values = values.as_unit(self.unit)
        try:
            self._check_compatible_with(values)
        except (TypeError, ValueError):
            return np.zeros(self.shape, dtype=bool)
        return isin(self.asi8, values.asi8)

    def isna(self) -> npt.NDArray[np.bool_]:
        if False:
            while True:
                i = 10
        return self._isnan

    @property
    def _isnan(self) -> npt.NDArray[np.bool_]:
        if False:
            return 10
        '\n        return if each value is nan\n        '
        return self.asi8 == iNaT

    @property
    def _hasna(self) -> bool:
        if False:
            print('Hello World!')
        '\n        return if I have any nans; enables various perf speedups\n        '
        return bool(self._isnan.any())

    def _maybe_mask_results(self, result: np.ndarray, fill_value=iNaT, convert=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        result : np.ndarray\n        fill_value : object, default iNaT\n        convert : str, dtype or None\n\n        Returns\n        -------\n        result : ndarray with values replace by the fill_value\n\n        mask the result if needed, convert to the provided dtype if its not\n        None\n\n        This is an internal routine.\n        '
        if self._hasna:
            if convert:
                result = result.astype(convert)
            if fill_value is None:
                fill_value = np.nan
            np.putmask(result, self._isnan, fill_value)
        return result

    @property
    def freqstr(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the frequency object as a string if it\'s set, otherwise None.\n\n        Examples\n        --------\n        For DatetimeIndex:\n\n        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00"], freq="D")\n        >>> idx.freqstr\n        \'D\'\n\n        The frequency can be inferred if there are more than 2 points:\n\n        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"],\n        ...                        freq="infer")\n        >>> idx.freqstr\n        \'2D\'\n\n        For PeriodIndex:\n\n        >>> idx = pd.PeriodIndex(["2023-1", "2023-2", "2023-3"], freq="M")\n        >>> idx.freqstr\n        \'M\'\n        '
        if self.freq is None:
            return None
        return self.freq.freqstr

    @property
    def inferred_freq(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tries to return a string representing a frequency generated by infer_freq.\n\n        Returns None if it can\'t autodetect the frequency.\n\n        Examples\n        --------\n        For DatetimeIndex:\n\n        >>> idx = pd.DatetimeIndex(["2018-01-01", "2018-01-03", "2018-01-05"])\n        >>> idx.inferred_freq\n        \'2D\'\n\n        For TimedeltaIndex:\n\n        >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])\n        >>> tdelta_idx\n        TimedeltaIndex([\'0 days\', \'10 days\', \'20 days\'],\n                       dtype=\'timedelta64[ns]\', freq=None)\n        >>> tdelta_idx.inferred_freq\n        \'10D\'\n        '
        if self.ndim != 1:
            return None
        try:
            return frequencies.infer_freq(self)
        except ValueError:
            return None

    @property
    def _resolution_obj(self) -> Resolution | None:
        if False:
            i = 10
            return i + 15
        freqstr = self.freqstr
        if freqstr is None:
            return None
        try:
            return Resolution.get_reso_from_freqstr(freqstr)
        except KeyError:
            return None

    @property
    def resolution(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns day, hour, minute, second, millisecond or microsecond\n        '
        return self._resolution_obj.attrname

    @property
    def _is_monotonic_increasing(self) -> bool:
        if False:
            print('Hello World!')
        return algos.is_monotonic(self.asi8, timelike=True)[0]

    @property
    def _is_monotonic_decreasing(self) -> bool:
        if False:
            i = 10
            return i + 15
        return algos.is_monotonic(self.asi8, timelike=True)[1]

    @property
    def _is_unique(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(unique1d(self.asi8.ravel('K'))) == self.size

    def _cmp_method(self, other, op):
        if False:
            while True:
                i = 10
        if self.ndim > 1 and getattr(other, 'shape', None) == self.shape:
            return op(self.ravel(), other.ravel()).reshape(self.shape)
        try:
            other = self._validate_comparison_value(other)
        except InvalidComparison:
            return invalid_comparison(self, other, op)
        dtype = getattr(other, 'dtype', None)
        if is_object_dtype(dtype):
            result = ops.comp_method_OBJECT_ARRAY(op, np.asarray(self.astype(object)), other)
            return result
        if other is NaT:
            if op is operator.ne:
                result = np.ones(self.shape, dtype=bool)
            else:
                result = np.zeros(self.shape, dtype=bool)
            return result
        if not isinstance(self.dtype, PeriodDtype):
            self = cast(TimelikeOps, self)
            if self._creso != other._creso:
                if not isinstance(other, type(self)):
                    try:
                        other = other.as_unit(self.unit, round_ok=False)
                    except ValueError:
                        other_arr = np.array(other.asm8)
                        return compare_mismatched_resolutions(self._ndarray, other_arr, op)
                else:
                    other_arr = other._ndarray
                    return compare_mismatched_resolutions(self._ndarray, other_arr, op)
        other_vals = self._unbox(other)
        result = op(self._ndarray.view('i8'), other_vals.view('i8'))
        o_mask = isna(other)
        mask = self._isnan | o_mask
        if mask.any():
            nat_result = op is operator.ne
            np.putmask(result, mask, nat_result)
        return result
    __pow__ = _make_unpacked_invalid_op('__pow__')
    __rpow__ = _make_unpacked_invalid_op('__rpow__')
    __mul__ = _make_unpacked_invalid_op('__mul__')
    __rmul__ = _make_unpacked_invalid_op('__rmul__')
    __truediv__ = _make_unpacked_invalid_op('__truediv__')
    __rtruediv__ = _make_unpacked_invalid_op('__rtruediv__')
    __floordiv__ = _make_unpacked_invalid_op('__floordiv__')
    __rfloordiv__ = _make_unpacked_invalid_op('__rfloordiv__')
    __mod__ = _make_unpacked_invalid_op('__mod__')
    __rmod__ = _make_unpacked_invalid_op('__rmod__')
    __divmod__ = _make_unpacked_invalid_op('__divmod__')
    __rdivmod__ = _make_unpacked_invalid_op('__rdivmod__')

    @final
    def _get_i8_values_and_mask(self, other) -> tuple[int | npt.NDArray[np.int64], None | npt.NDArray[np.bool_]]:
        if False:
            return 10
        '\n        Get the int64 values and b_mask to pass to checked_add_with_arr.\n        '
        if isinstance(other, Period):
            i8values = other.ordinal
            mask = None
        elif isinstance(other, (Timestamp, Timedelta)):
            i8values = other._value
            mask = None
        else:
            mask = other._isnan
            i8values = other.asi8
        return (i8values, mask)

    @final
    def _get_arithmetic_result_freq(self, other) -> BaseOffset | None:
        if False:
            print('Hello World!')
        '\n        Check if we can preserve self.freq in addition or subtraction.\n        '
        if isinstance(self.dtype, PeriodDtype):
            return self.freq
        elif not lib.is_scalar(other):
            return None
        elif isinstance(self.freq, Tick):
            return self.freq
        return None

    @final
    def _add_datetimelike_scalar(self, other) -> DatetimeArray:
        if False:
            return 10
        if not lib.is_np_dtype(self.dtype, 'm'):
            raise TypeError(f'cannot add {type(self).__name__} and {type(other).__name__}')
        self = cast('TimedeltaArray', self)
        from pandas.core.arrays import DatetimeArray
        from pandas.core.arrays.datetimes import tz_to_dtype
        assert other is not NaT
        if isna(other):
            result = self._ndarray + NaT.to_datetime64().astype(f'M8[{self.unit}]')
            return DatetimeArray._simple_new(result, dtype=result.dtype)
        other = Timestamp(other)
        (self, other) = self._ensure_matching_resos(other)
        self = cast('TimedeltaArray', self)
        (other_i8, o_mask) = self._get_i8_values_and_mask(other)
        result = checked_add_with_arr(self.asi8, other_i8, arr_mask=self._isnan, b_mask=o_mask)
        res_values = result.view(f'M8[{self.unit}]')
        dtype = tz_to_dtype(tz=other.tz, unit=self.unit)
        res_values = result.view(f'M8[{self.unit}]')
        new_freq = self._get_arithmetic_result_freq(other)
        return DatetimeArray._simple_new(res_values, dtype=dtype, freq=new_freq)

    @final
    def _add_datetime_arraylike(self, other: DatetimeArray) -> DatetimeArray:
        if False:
            print('Hello World!')
        if not lib.is_np_dtype(self.dtype, 'm'):
            raise TypeError(f'cannot add {type(self).__name__} and {type(other).__name__}')
        return other + self

    @final
    def _sub_datetimelike_scalar(self, other: datetime | np.datetime64) -> TimedeltaArray:
        if False:
            for i in range(10):
                print('nop')
        if self.dtype.kind != 'M':
            raise TypeError(f'cannot subtract a datelike from a {type(self).__name__}')
        self = cast('DatetimeArray', self)
        if isna(other):
            return self - NaT
        ts = Timestamp(other)
        (self, ts) = self._ensure_matching_resos(ts)
        return self._sub_datetimelike(ts)

    @final
    def _sub_datetime_arraylike(self, other: DatetimeArray) -> TimedeltaArray:
        if False:
            i = 10
            return i + 15
        if self.dtype.kind != 'M':
            raise TypeError(f'cannot subtract a datelike from a {type(self).__name__}')
        if len(self) != len(other):
            raise ValueError('cannot add indices of unequal length')
        self = cast('DatetimeArray', self)
        (self, other) = self._ensure_matching_resos(other)
        return self._sub_datetimelike(other)

    @final
    def _sub_datetimelike(self, other: Timestamp | DatetimeArray) -> TimedeltaArray:
        if False:
            for i in range(10):
                print('nop')
        self = cast('DatetimeArray', self)
        from pandas.core.arrays import TimedeltaArray
        try:
            self._assert_tzawareness_compat(other)
        except TypeError as err:
            new_message = str(err).replace('compare', 'subtract')
            raise type(err)(new_message) from err
        (other_i8, o_mask) = self._get_i8_values_and_mask(other)
        res_values = checked_add_with_arr(self.asi8, -other_i8, arr_mask=self._isnan, b_mask=o_mask)
        res_m8 = res_values.view(f'timedelta64[{self.unit}]')
        new_freq = self._get_arithmetic_result_freq(other)
        new_freq = cast('Tick | None', new_freq)
        return TimedeltaArray._simple_new(res_m8, dtype=res_m8.dtype, freq=new_freq)

    @final
    def _add_period(self, other: Period) -> PeriodArray:
        if False:
            print('Hello World!')
        if not lib.is_np_dtype(self.dtype, 'm'):
            raise TypeError(f'cannot add Period to a {type(self).__name__}')
        from pandas.core.arrays.period import PeriodArray
        i8vals = np.broadcast_to(other.ordinal, self.shape)
        dtype = PeriodDtype(other.freq)
        parr = PeriodArray(i8vals, dtype=dtype)
        return parr + self

    def _add_offset(self, offset):
        if False:
            i = 10
            return i + 15
        raise AbstractMethodError(self)

    def _add_timedeltalike_scalar(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a delta of a timedeltalike\n\n        Returns\n        -------\n        Same type as self\n        '
        if isna(other):
            new_values = np.empty(self.shape, dtype='i8').view(self._ndarray.dtype)
            new_values.fill(iNaT)
            return type(self)._simple_new(new_values, dtype=self.dtype)
        self = cast('DatetimeArray | TimedeltaArray', self)
        other = Timedelta(other)
        (self, other) = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    def _add_timedelta_arraylike(self, other: TimedeltaArray):
        if False:
            i = 10
            return i + 15
        '\n        Add a delta of a TimedeltaIndex\n\n        Returns\n        -------\n        Same type as self\n        '
        if len(self) != len(other):
            raise ValueError('cannot add indices of unequal length')
        self = cast('DatetimeArray | TimedeltaArray', self)
        (self, other) = self._ensure_matching_resos(other)
        return self._add_timedeltalike(other)

    @final
    def _add_timedeltalike(self, other: Timedelta | TimedeltaArray):
        if False:
            return 10
        self = cast('DatetimeArray | TimedeltaArray', self)
        (other_i8, o_mask) = self._get_i8_values_and_mask(other)
        new_values = checked_add_with_arr(self.asi8, other_i8, arr_mask=self._isnan, b_mask=o_mask)
        res_values = new_values.view(self._ndarray.dtype)
        new_freq = self._get_arithmetic_result_freq(other)
        return type(self)._simple_new(res_values, dtype=self.dtype, freq=new_freq)

    @final
    def _add_nat(self):
        if False:
            while True:
                i = 10
        '\n        Add pd.NaT to self\n        '
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(f'Cannot add {type(self).__name__} and {type(NaT).__name__}')
        self = cast('TimedeltaArray | DatetimeArray', self)
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        result = result.view(self._ndarray.dtype)
        return type(self)._simple_new(result, dtype=self.dtype, freq=None)

    @final
    def _sub_nat(self):
        if False:
            while True:
                i = 10
        '\n        Subtract pd.NaT from self\n        '
        result = np.empty(self.shape, dtype=np.int64)
        result.fill(iNaT)
        if self.dtype.kind in 'mM':
            self = cast('DatetimeArray| TimedeltaArray', self)
            return result.view(f'timedelta64[{self.unit}]')
        else:
            return result.view('timedelta64[ns]')

    @final
    def _sub_periodlike(self, other: Period | PeriodArray) -> npt.NDArray[np.object_]:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.dtype, PeriodDtype):
            raise TypeError(f'cannot subtract {type(other).__name__} from {type(self).__name__}')
        self = cast('PeriodArray', self)
        self._check_compatible_with(other)
        (other_i8, o_mask) = self._get_i8_values_and_mask(other)
        new_i8_data = checked_add_with_arr(self.asi8, -other_i8, arr_mask=self._isnan, b_mask=o_mask)
        new_data = np.array([self.freq.base * x for x in new_i8_data])
        if o_mask is None:
            mask = self._isnan
        else:
            mask = self._isnan | o_mask
        new_data[mask] = NaT
        return new_data

    @final
    def _addsub_object_array(self, other: npt.NDArray[np.object_], op):
        if False:
            print('Hello World!')
        '\n        Add or subtract array-like of DateOffset objects\n\n        Parameters\n        ----------\n        other : np.ndarray[object]\n        op : {operator.add, operator.sub}\n\n        Returns\n        -------\n        np.ndarray[object]\n            Except in fastpath case with length 1 where we operate on the\n            contained scalar.\n        '
        assert op in [operator.add, operator.sub]
        if len(other) == 1 and self.ndim == 1:
            return op(self, other[0])
        warnings.warn(f'Adding/subtracting object-dtype array to {type(self).__name__} not vectorized.', PerformanceWarning, stacklevel=find_stack_level())
        assert self.shape == other.shape, (self.shape, other.shape)
        res_values = op(self.astype('O'), np.asarray(other))
        return res_values

    def _accumulate(self, name: str, *, skipna: bool=True, **kwargs) -> Self:
        if False:
            print('Hello World!')
        if name not in {'cummin', 'cummax'}:
            raise TypeError(f'Accumulation {name} not supported for {type(self)}')
        op = getattr(datetimelike_accumulations, name)
        result = op(self.copy(), skipna=skipna, **kwargs)
        return type(self)._simple_new(result, dtype=self.dtype)

    @unpack_zerodim_and_defer('__add__')
    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        other_dtype = getattr(other, 'dtype', None)
        other = ensure_wrapped_if_datetimelike(other)
        if other is NaT:
            result = self._add_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(other)
        elif isinstance(other, BaseOffset):
            result = self._add_offset(other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._add_datetimelike_scalar(other)
        elif isinstance(other, Period) and lib.is_np_dtype(self.dtype, 'm'):
            result = self._add_period(other)
        elif lib.is_integer(other):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        elif lib.is_np_dtype(other_dtype, 'm'):
            result = self._add_timedelta_arraylike(other)
        elif is_object_dtype(other_dtype):
            result = self._addsub_object_array(other, operator.add)
        elif lib.is_np_dtype(other_dtype, 'M') or isinstance(other_dtype, DatetimeTZDtype):
            return self._add_datetime_arraylike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.add)
        else:
            return NotImplemented
        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray(result)
        return result

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        return self.__add__(other)

    @unpack_zerodim_and_defer('__sub__')
    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        other_dtype = getattr(other, 'dtype', None)
        other = ensure_wrapped_if_datetimelike(other)
        if other is NaT:
            result = self._sub_nat()
        elif isinstance(other, (Tick, timedelta, np.timedelta64)):
            result = self._add_timedeltalike_scalar(-other)
        elif isinstance(other, BaseOffset):
            result = self._add_offset(-other)
        elif isinstance(other, (datetime, np.datetime64)):
            result = self._sub_datetimelike_scalar(other)
        elif lib.is_integer(other):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        elif isinstance(other, Period):
            result = self._sub_periodlike(other)
        elif lib.is_np_dtype(other_dtype, 'm'):
            result = self._add_timedelta_arraylike(-other)
        elif is_object_dtype(other_dtype):
            result = self._addsub_object_array(other, operator.sub)
        elif lib.is_np_dtype(other_dtype, 'M') or isinstance(other_dtype, DatetimeTZDtype):
            result = self._sub_datetime_arraylike(other)
        elif isinstance(other_dtype, PeriodDtype):
            result = self._sub_periodlike(other)
        elif is_integer_dtype(other_dtype):
            if not isinstance(self.dtype, PeriodDtype):
                raise integer_op_not_supported(self)
            obj = cast('PeriodArray', self)
            result = obj._addsub_int_array_or_scalar(other * obj.dtype._n, operator.sub)
        else:
            return NotImplemented
        if isinstance(result, np.ndarray) and lib.is_np_dtype(result.dtype, 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray(result)
        return result

    def __rsub__(self, other):
        if False:
            return 10
        other_dtype = getattr(other, 'dtype', None)
        other_is_dt64 = lib.is_np_dtype(other_dtype, 'M') or isinstance(other_dtype, DatetimeTZDtype)
        if other_is_dt64 and lib.is_np_dtype(self.dtype, 'm'):
            if lib.is_scalar(other):
                return Timestamp(other) - self
            if not isinstance(other, DatetimeLikeArrayMixin):
                from pandas.core.arrays import DatetimeArray
                other = DatetimeArray(other)
            return other - self
        elif self.dtype.kind == 'M' and hasattr(other, 'dtype') and (not other_is_dt64):
            raise TypeError(f'cannot subtract {type(self).__name__} from {type(other).__name__}')
        elif isinstance(self.dtype, PeriodDtype) and lib.is_np_dtype(other_dtype, 'm'):
            raise TypeError(f'cannot subtract {type(self).__name__} from {other.dtype}')
        elif lib.is_np_dtype(self.dtype, 'm'):
            self = cast('TimedeltaArray', self)
            return -self + other
        return -(self - other)

    def __iadd__(self, other) -> Self:
        if False:
            i = 10
            return i + 15
        result = self + other
        self[:] = result[:]
        if not isinstance(self.dtype, PeriodDtype):
            self._freq = result.freq
        return self

    def __isub__(self, other) -> Self:
        if False:
            while True:
                i = 10
        result = self - other
        self[:] = result[:]
        if not isinstance(self.dtype, PeriodDtype):
            self._freq = result.freq
        return self

    @_period_dispatch
    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        if False:
            return 10
        return super()._quantile(qs=qs, interpolation=interpolation)

    @_period_dispatch
    def min(self, *, axis: AxisInt | None=None, skipna: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the minimum value of the Array or minimum along\n        an axis.\n\n        See Also\n        --------\n        numpy.ndarray.min\n        Index.min : Return the minimum value in an Index.\n        Series.min : Return the minimum value in a Series.\n        '
        nv.validate_min((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)
        result = nanops.nanmin(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def max(self, *, axis: AxisInt | None=None, skipna: bool=True, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Return the maximum value of the Array or maximum along\n        an axis.\n\n        See Also\n        --------\n        numpy.ndarray.max\n        Index.max : Return the maximum value in an Index.\n        Series.max : Return the maximum value in a Series.\n        '
        nv.validate_max((), kwargs)
        nv.validate_minmax_axis(axis, self.ndim)
        result = nanops.nanmax(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def mean(self, *, skipna: bool=True, axis: AxisInt | None=0):
        if False:
            print('Hello World!')
        "\n        Return the mean value of the Array.\n\n        Parameters\n        ----------\n        skipna : bool, default True\n            Whether to ignore any NaT elements.\n        axis : int, optional, default 0\n\n        Returns\n        -------\n        scalar\n            Timestamp or Timedelta.\n\n        See Also\n        --------\n        numpy.ndarray.mean : Returns the average of array elements along a given axis.\n        Series.mean : Return the mean value in a Series.\n\n        Notes\n        -----\n        mean is only defined for Datetime and Timedelta dtypes, not for Period.\n\n        Examples\n        --------\n        For :class:`pandas.DatetimeIndex`:\n\n        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)\n        >>> idx\n        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],\n                      dtype='datetime64[ns]', freq='D')\n        >>> idx.mean()\n        Timestamp('2001-01-02 00:00:00')\n\n        For :class:`pandas.TimedeltaIndex`:\n\n        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')\n        >>> tdelta_idx\n        TimedeltaIndex(['1 days', '2 days', '3 days'],\n                        dtype='timedelta64[ns]', freq=None)\n        >>> tdelta_idx.mean()\n        Timedelta('2 days 00:00:00')\n        "
        if isinstance(self.dtype, PeriodDtype):
            raise TypeError(f"mean is not implemented for {type(self).__name__} since the meaning is ambiguous.  An alternative is obj.to_timestamp(how='start').mean()")
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())
        return self._wrap_reduction_result(axis, result)

    @_period_dispatch
    def median(self, *, axis: AxisInt | None=None, skipna: bool=True, **kwargs):
        if False:
            i = 10
            return i + 15
        nv.validate_median((), kwargs)
        if axis is not None and abs(axis) >= self.ndim:
            raise ValueError('abs(axis) must be less than ndim')
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def _mode(self, dropna: bool=True):
        if False:
            print('Hello World!')
        mask = None
        if dropna:
            mask = self.isna()
        i8modes = algorithms.mode(self.view('i8'), mask=mask)
        npmodes = i8modes.view(self._ndarray.dtype)
        npmodes = cast(np.ndarray, npmodes)
        return self._from_backing_data(npmodes)

    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: npt.NDArray[np.intp], **kwargs):
        if False:
            return 10
        dtype = self.dtype
        if dtype.kind == 'M':
            if how in ['sum', 'prod', 'cumsum', 'cumprod', 'var', 'skew']:
                raise TypeError(f'datetime64 type does not support {how} operations')
            if how in ['any', 'all']:
                warnings.warn(f"'{how}' with datetime64 dtypes is deprecated and will raise in a future version. Use (obj != pd.Timestamp(0)).{how}() instead.", FutureWarning, stacklevel=find_stack_level())
        elif isinstance(dtype, PeriodDtype):
            if how in ['sum', 'prod', 'cumsum', 'cumprod', 'var', 'skew']:
                raise TypeError(f'Period type does not support {how} operations')
            if how in ['any', 'all']:
                warnings.warn(f"'{how}' with PeriodDtype is deprecated and will raise in a future version. Use (obj != pd.Period(0, freq)).{how}() instead.", FutureWarning, stacklevel=find_stack_level())
        elif how in ['prod', 'cumprod', 'skew', 'var']:
            raise TypeError(f'timedelta64 type does not support {how} operations')
        npvalues = self._ndarray.view('M8[ns]')
        from pandas.core.groupby.ops import WrappedCythonOp
        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)
        res_values = op._cython_op_ndim_compat(npvalues, min_count=min_count, ngroups=ngroups, comp_ids=ids, mask=None, **kwargs)
        if op.how in op.cast_blocklist:
            return res_values
        assert res_values.dtype == 'M8[ns]'
        if how in ['std', 'sem']:
            from pandas.core.arrays import TimedeltaArray
            if isinstance(self.dtype, PeriodDtype):
                raise TypeError("'std' and 'sem' are not valid for PeriodDtype")
            self = cast('DatetimeArray | TimedeltaArray', self)
            new_dtype = f'm8[{self.unit}]'
            res_values = res_values.view(new_dtype)
            return TimedeltaArray(res_values)
        res_values = res_values.view(self._ndarray.dtype)
        return self._from_backing_data(res_values)

class DatelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for DatetimeIndex/PeriodIndex, but not TimedeltaIndex.
    """

    @Substitution(URL='https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior')
    def strftime(self, date_format: str) -> npt.NDArray[np.object_]:
        if False:
            return 10
        '\n        Convert to Index using specified date_format.\n\n        Return an Index of formatted strings specified by date_format, which\n        supports the same string format as the python standard library. Details\n        of the string format can be found in `python string format\n        doc <%(URL)s>`__.\n\n        Formats supported by the C `strftime` API but not by the python string format\n        doc (such as `"%%R"`, `"%%r"`) are not officially supported and should be\n        preferably replaced with their supported equivalents (such as `"%%H:%%M"`,\n        `"%%I:%%M:%%S %%p"`).\n\n        Note that `PeriodIndex` support additional directives, detailed in\n        `Period.strftime`.\n\n        Parameters\n        ----------\n        date_format : str\n            Date format string (e.g. "%%Y-%%m-%%d").\n\n        Returns\n        -------\n        ndarray[object]\n            NumPy ndarray of formatted strings.\n\n        See Also\n        --------\n        to_datetime : Convert the given argument to datetime.\n        DatetimeIndex.normalize : Return DatetimeIndex with times to midnight.\n        DatetimeIndex.round : Round the DatetimeIndex to the specified freq.\n        DatetimeIndex.floor : Floor the DatetimeIndex to the specified freq.\n        Timestamp.strftime : Format a single Timestamp.\n        Period.strftime : Format a single Period.\n\n        Examples\n        --------\n        >>> rng = pd.date_range(pd.Timestamp("2018-03-10 09:00"),\n        ...                     periods=3, freq=\'s\')\n        >>> rng.strftime(\'%%B %%d, %%Y, %%r\')\n        Index([\'March 10, 2018, 09:00:00 AM\', \'March 10, 2018, 09:00:01 AM\',\n               \'March 10, 2018, 09:00:02 AM\'],\n              dtype=\'object\')\n        '
        result = self._format_native_types(date_format=date_format, na_rep=np.nan)
        return result.astype(object, copy=False)
_round_doc = '\n    Perform {op} operation on the data to the specified `freq`.\n\n    Parameters\n    ----------\n    freq : str or Offset\n        The frequency level to {op} the index to. Must be a fixed\n        frequency like \'S\' (second) not \'ME\' (month end). See\n        :ref:`frequency aliases <timeseries.offset_aliases>` for\n        a list of possible `freq` values.\n    ambiguous : \'infer\', bool-ndarray, \'NaT\', default \'raise\'\n        Only relevant for DatetimeIndex:\n\n        - \'infer\' will attempt to infer fall dst-transition hours based on\n          order\n        - bool-ndarray where True signifies a DST time, False designates\n          a non-DST time (note that this flag is only applicable for\n          ambiguous times)\n        - \'NaT\' will return NaT where there are ambiguous times\n        - \'raise\' will raise an AmbiguousTimeError if there are ambiguous\n          times.\n\n    nonexistent : \'shift_forward\', \'shift_backward\', \'NaT\', timedelta, default \'raise\'\n        A nonexistent time does not exist in a particular timezone\n        where clocks moved forward due to DST.\n\n        - \'shift_forward\' will shift the nonexistent time forward to the\n          closest existing time\n        - \'shift_backward\' will shift the nonexistent time backward to the\n          closest existing time\n        - \'NaT\' will return NaT where there are nonexistent times\n        - timedelta objects will shift nonexistent times by the timedelta\n        - \'raise\' will raise an NonExistentTimeError if there are\n          nonexistent times.\n\n    Returns\n    -------\n    DatetimeIndex, TimedeltaIndex, or Series\n        Index of the same type for a DatetimeIndex or TimedeltaIndex,\n        or a Series with the same index for a Series.\n\n    Raises\n    ------\n    ValueError if the `freq` cannot be converted.\n\n    Notes\n    -----\n    If the timestamps have a timezone, {op}ing will take place relative to the\n    local ("wall") time and re-localized to the same timezone. When {op}ing\n    near daylight savings time, use ``nonexistent`` and ``ambiguous`` to\n    control the re-localization behavior.\n\n    Examples\n    --------\n    **DatetimeIndex**\n\n    >>> rng = pd.date_range(\'1/1/2018 11:59:00\', periods=3, freq=\'min\')\n    >>> rng\n    DatetimeIndex([\'2018-01-01 11:59:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 12:01:00\'],\n                  dtype=\'datetime64[ns]\', freq=\'min\')\n    '
_round_example = '>>> rng.round(\'h\')\n    DatetimeIndex([\'2018-01-01 12:00:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 12:00:00\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n\n    **Series**\n\n    >>> pd.Series(rng).dt.round("h")\n    0   2018-01-01 12:00:00\n    1   2018-01-01 12:00:00\n    2   2018-01-01 12:00:00\n    dtype: datetime64[ns]\n\n    When rounding near a daylight savings time transition, use ``ambiguous`` or\n    ``nonexistent`` to control how the timestamp should be re-localized.\n\n    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")\n\n    >>> rng_tz.floor("2h", ambiguous=False)\n    DatetimeIndex([\'2021-10-31 02:00:00+01:00\'],\n                  dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)\n\n    >>> rng_tz.floor("2h", ambiguous=True)\n    DatetimeIndex([\'2021-10-31 02:00:00+02:00\'],\n                  dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)\n    '
_floor_example = '>>> rng.floor(\'h\')\n    DatetimeIndex([\'2018-01-01 11:00:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 12:00:00\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n\n    **Series**\n\n    >>> pd.Series(rng).dt.floor("h")\n    0   2018-01-01 11:00:00\n    1   2018-01-01 12:00:00\n    2   2018-01-01 12:00:00\n    dtype: datetime64[ns]\n\n    When rounding near a daylight savings time transition, use ``ambiguous`` or\n    ``nonexistent`` to control how the timestamp should be re-localized.\n\n    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 03:30:00"], tz="Europe/Amsterdam")\n\n    >>> rng_tz.floor("2h", ambiguous=False)\n    DatetimeIndex([\'2021-10-31 02:00:00+01:00\'],\n                 dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)\n\n    >>> rng_tz.floor("2h", ambiguous=True)\n    DatetimeIndex([\'2021-10-31 02:00:00+02:00\'],\n                  dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)\n    '
_ceil_example = '>>> rng.ceil(\'h\')\n    DatetimeIndex([\'2018-01-01 12:00:00\', \'2018-01-01 12:00:00\',\n                   \'2018-01-01 13:00:00\'],\n                  dtype=\'datetime64[ns]\', freq=None)\n\n    **Series**\n\n    >>> pd.Series(rng).dt.ceil("h")\n    0   2018-01-01 12:00:00\n    1   2018-01-01 12:00:00\n    2   2018-01-01 13:00:00\n    dtype: datetime64[ns]\n\n    When rounding near a daylight savings time transition, use ``ambiguous`` or\n    ``nonexistent`` to control how the timestamp should be re-localized.\n\n    >>> rng_tz = pd.DatetimeIndex(["2021-10-31 01:30:00"], tz="Europe/Amsterdam")\n\n    >>> rng_tz.ceil("h", ambiguous=False)\n    DatetimeIndex([\'2021-10-31 02:00:00+01:00\'],\n                  dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)\n\n    >>> rng_tz.ceil("h", ambiguous=True)\n    DatetimeIndex([\'2021-10-31 02:00:00+02:00\'],\n                  dtype=\'datetime64[ns, Europe/Amsterdam]\', freq=None)\n    '

class TimelikeOps(DatetimeLikeArrayMixin):
    """
    Common ops for TimedeltaIndex/DatetimeIndex, but not PeriodIndex.
    """
    _default_dtype: np.dtype

    def __init__(self, values, dtype=None, freq=lib.no_default, copy: bool=False) -> None:
        if False:
            print('Hello World!')
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        values = extract_array(values, extract_numpy=True)
        if isinstance(values, IntegerArray):
            values = values.to_numpy('int64', na_value=iNaT)
        inferred_freq = getattr(values, '_freq', None)
        explicit_none = freq is None
        freq = freq if freq is not lib.no_default else None
        if isinstance(values, type(self)):
            if explicit_none:
                pass
            elif freq is None:
                freq = values.freq
            elif freq and values.freq:
                freq = to_offset(freq)
                (freq, _) = validate_inferred_freq(freq, values.freq, False)
            if dtype is not None and dtype != values.dtype:
                raise TypeError(f'dtype={dtype} does not match data dtype {values.dtype}')
            dtype = values.dtype
            values = values._ndarray
        elif dtype is None:
            if isinstance(values, np.ndarray) and values.dtype.kind in 'Mm':
                dtype = values.dtype
            else:
                dtype = self._default_dtype
                if isinstance(values, np.ndarray) and values.dtype == 'i8':
                    values = values.view(dtype)
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Unexpected type '{type(values).__name__}'. 'values' must be a {type(self).__name__}, ndarray, or Series or Index containing one of those.")
        if values.ndim not in [1, 2]:
            raise ValueError('Only 1-dimensional input arrays are supported.')
        if values.dtype == 'i8':
            if dtype is None:
                dtype = self._default_dtype
                values = values.view(self._default_dtype)
            elif lib.is_np_dtype(dtype, 'mM'):
                values = values.view(dtype)
            elif isinstance(dtype, DatetimeTZDtype):
                kind = self._default_dtype.kind
                new_dtype = f'{kind}8[{dtype.unit}]'
                values = values.view(new_dtype)
        dtype = self._validate_dtype(values, dtype)
        if freq == 'infer':
            raise ValueError(f"Frequency inference not allowed in {type(self).__name__}.__init__. Use 'pd.array()' instead.")
        if copy:
            values = values.copy()
        if freq:
            freq = to_offset(freq)
            if values.dtype.kind == 'm' and (not isinstance(freq, Tick)):
                raise TypeError('TimedeltaArray/Index freq must be a Tick')
        NDArrayBacked.__init__(self, values=values, dtype=dtype)
        self._freq = freq
        if inferred_freq is None and freq is not None:
            type(self)._validate_frequency(self, freq)

    @classmethod
    def _validate_dtype(cls, values, dtype):
        if False:
            return 10
        raise AbstractMethodError(cls)

    @property
    def freq(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the frequency object if it is set, otherwise None.\n        '
        return self._freq

    @freq.setter
    def freq(self, value) -> None:
        if False:
            while True:
                i = 10
        if value is not None:
            value = to_offset(value)
            self._validate_frequency(self, value)
            if self.dtype.kind == 'm' and (not isinstance(value, Tick)):
                raise TypeError('TimedeltaArray/Index freq must be a Tick')
            if self.ndim > 1:
                raise ValueError('Cannot set freq with ndim > 1')
        self._freq = value

    @final
    @classmethod
    def _validate_frequency(cls, index, freq: BaseOffset, **kwargs):
        if False:
            return 10
        '\n        Validate that a frequency is compatible with the values of a given\n        Datetime Array/Index or Timedelta Array/Index\n\n        Parameters\n        ----------\n        index : DatetimeIndex or TimedeltaIndex\n            The index on which to determine if the given frequency is valid\n        freq : DateOffset\n            The frequency to validate\n        '
        inferred = index.inferred_freq
        if index.size == 0 or inferred == freq.freqstr:
            return None
        try:
            on_freq = cls._generate_range(start=index[0], end=None, periods=len(index), freq=freq, unit=index.unit, **kwargs)
            if not np.array_equal(index.asi8, on_freq.asi8):
                raise ValueError
        except ValueError as err:
            if 'non-fixed' in str(err):
                raise err
            raise ValueError(f'Inferred frequency {inferred} from passed values does not conform to passed frequency {freq.freqstr}') from err

    @classmethod
    def _generate_range(cls, start, end, periods, freq, *args, **kwargs) -> Self:
        if False:
            for i in range(10):
                print('nop')
        raise AbstractMethodError(cls)

    @cache_readonly
    def _creso(self) -> int:
        if False:
            i = 10
            return i + 15
        return get_unit_from_dtype(self._ndarray.dtype)

    @cache_readonly
    def unit(self) -> str:
        if False:
            print('Hello World!')
        return dtype_to_unit(self.dtype)

    def as_unit(self, unit: str) -> Self:
        if False:
            while True:
                i = 10
        if unit not in ['s', 'ms', 'us', 'ns']:
            raise ValueError("Supported units are 's', 'ms', 'us', 'ns'")
        dtype = np.dtype(f'{self.dtype.kind}8[{unit}]')
        new_values = astype_overflowsafe(self._ndarray, dtype, round_ok=True)
        if isinstance(self.dtype, np.dtype):
            new_dtype = new_values.dtype
        else:
            tz = cast('DatetimeArray', self).tz
            new_dtype = DatetimeTZDtype(tz=tz, unit=unit)
        return type(self)._simple_new(new_values, dtype=new_dtype, freq=self.freq)

    def _ensure_matching_resos(self, other):
        if False:
            return 10
        if self._creso != other._creso:
            if self._creso < other._creso:
                self = self.as_unit(other.unit)
            else:
                other = other.as_unit(self.unit)
        return (self, other)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if ufunc in [np.isnan, np.isinf, np.isfinite] and len(inputs) == 1 and (inputs[0] is self):
            return getattr(ufunc, method)(self._ndarray, **kwargs)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _round(self, freq, mode, ambiguous, nonexistent):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.dtype, DatetimeTZDtype):
            self = cast('DatetimeArray', self)
            naive = self.tz_localize(None)
            result = naive._round(freq, mode, ambiguous, nonexistent)
            return result.tz_localize(self.tz, ambiguous=ambiguous, nonexistent=nonexistent)
        values = self.view('i8')
        values = cast(np.ndarray, values)
        nanos = get_unit_for_round(freq, self._creso)
        if nanos == 0:
            return self.copy()
        result_i8 = round_nsint64(values, mode, nanos)
        result = self._maybe_mask_results(result_i8, fill_value=iNaT)
        result = result.view(self._ndarray.dtype)
        return self._simple_new(result, dtype=self.dtype)

    @Appender((_round_doc + _round_example).format(op='round'))
    def round(self, freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        if False:
            while True:
                i = 10
        return self._round(freq, RoundTo.NEAREST_HALF_EVEN, ambiguous, nonexistent)

    @Appender((_round_doc + _floor_example).format(op='floor'))
    def floor(self, freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        if False:
            for i in range(10):
                print('nop')
        return self._round(freq, RoundTo.MINUS_INFTY, ambiguous, nonexistent)

    @Appender((_round_doc + _ceil_example).format(op='ceil'))
    def ceil(self, freq, ambiguous: TimeAmbiguous='raise', nonexistent: TimeNonexistent='raise') -> Self:
        if False:
            while True:
                i = 10
        return self._round(freq, RoundTo.PLUS_INFTY, ambiguous, nonexistent)

    def any(self, *, axis: AxisInt | None=None, skipna: bool=True) -> bool:
        if False:
            return 10
        return nanops.nanany(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def all(self, *, axis: AxisInt | None=None, skipna: bool=True) -> bool:
        if False:
            print('Hello World!')
        return nanops.nanall(self._ndarray, axis=axis, skipna=skipna, mask=self.isna())

    def _maybe_clear_freq(self) -> None:
        if False:
            print('Hello World!')
        self._freq = None

    def _with_freq(self, freq) -> Self:
        if False:
            print('Hello World!')
        '\n        Helper to get a view on the same data, with a new freq.\n\n        Parameters\n        ----------\n        freq : DateOffset, None, or "infer"\n\n        Returns\n        -------\n        Same type as self\n        '
        if freq is None:
            pass
        elif len(self) == 0 and isinstance(freq, BaseOffset):
            if self.dtype.kind == 'm' and (not isinstance(freq, Tick)):
                raise TypeError('TimedeltaArray/Index freq must be a Tick')
        else:
            assert freq == 'infer'
            freq = to_offset(self.inferred_freq)
        arr = self.view()
        arr._freq = freq
        return arr

    def _values_for_json(self) -> np.ndarray:
        if False:
            print('Hello World!')
        if isinstance(self.dtype, np.dtype):
            return self._ndarray
        return super()._values_for_json()

    def factorize(self, use_na_sentinel: bool=True, sort: bool=False):
        if False:
            for i in range(10):
                print('nop')
        if self.freq is not None:
            codes = np.arange(len(self), dtype=np.intp)
            uniques = self.copy()
            if sort and self.freq.n < 0:
                codes = codes[::-1]
                uniques = uniques[::-1]
            return (codes, uniques)
        if sort:
            raise NotImplementedError(f"The 'sort' keyword in {type(self).__name__}.factorize is ignored unless arr.freq is not None. To factorize with sort, call pd.factorize(obj, sort=True) instead.")
        return super().factorize(use_na_sentinel=use_na_sentinel)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt=0) -> Self:
        if False:
            return 10
        new_obj = super()._concat_same_type(to_concat, axis)
        obj = to_concat[0]
        if axis == 0:
            to_concat = [x for x in to_concat if len(x)]
            if obj.freq is not None and all((x.freq == obj.freq for x in to_concat)):
                pairs = zip(to_concat[:-1], to_concat[1:])
                if all((pair[0][-1] + obj.freq == pair[1][0] for pair in pairs)):
                    new_freq = obj.freq
                    new_obj._freq = new_freq
        return new_obj

    def copy(self, order: str='C') -> Self:
        if False:
            i = 10
            return i + 15
        new_obj = super().copy(order=order)
        new_obj._freq = self.freq
        return new_obj

    def interpolate(self, *, method: InterpolateOptions, axis: int, index: Index, limit, limit_direction, limit_area, copy: bool, **kwargs) -> Self:
        if False:
            while True:
                i = 10
        '\n        See NDFrame.interpolate.__doc__.\n        '
        if method != 'linear':
            raise NotImplementedError
        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()
        missing.interpolate_2d_inplace(out_data, method=method, axis=axis, index=index, limit=limit, limit_direction=limit_direction, limit_area=limit_area, **kwargs)
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    @property
    def _is_dates_only(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check if we are round times at midnight (and no timezone), which will\n        be given a more compact __repr__ than other cases. For TimedeltaArray\n        we are checking for multiples of 24H.\n        '
        if not lib.is_np_dtype(self.dtype):
            return False
        values_int = self.asi8
        consider_values = values_int != iNaT
        reso = get_unit_from_dtype(self.dtype)
        ppd = periods_per_day(reso)
        even_days = np.logical_and(consider_values, values_int % ppd != 0).sum() == 0
        return even_days

def ensure_arraylike_for_datetimelike(data, copy: bool, cls_name: str):
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(data, 'dtype'):
        if not isinstance(data, (list, tuple)) and np.ndim(data) == 0:
            data = list(data)
        data = construct_1d_object_array_from_listlike(data)
        copy = False
    elif isinstance(data, ABCMultiIndex):
        raise TypeError(f'Cannot create a {cls_name} from a MultiIndex.')
    else:
        data = extract_array(data, extract_numpy=True)
    if isinstance(data, IntegerArray) or (isinstance(data, ArrowExtensionArray) and data.dtype.kind in 'iu'):
        data = data.to_numpy('int64', na_value=iNaT)
        copy = False
    elif isinstance(data, ArrowExtensionArray):
        data = data._maybe_convert_datelike_array()
        data = data.to_numpy()
        copy = False
    elif not isinstance(data, (np.ndarray, ExtensionArray)):
        data = np.asarray(data)
    elif isinstance(data, ABCCategorical):
        data = data.categories.take(data.codes, fill_value=NaT)._values
        copy = False
    return (data, copy)

@overload
def validate_periods(periods: None) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def validate_periods(periods: int | float) -> int:
    if False:
        while True:
            i = 10
    ...

def validate_periods(periods: int | float | None) -> int | None:
    if False:
        while True:
            i = 10
    '\n    If a `periods` argument is passed to the Datetime/Timedelta Array/Index\n    constructor, cast it to an integer.\n\n    Parameters\n    ----------\n    periods : None, float, int\n\n    Returns\n    -------\n    periods : None or int\n\n    Raises\n    ------\n    TypeError\n        if periods is None, float, or int\n    '
    if periods is not None:
        if lib.is_float(periods):
            periods = int(periods)
        elif not lib.is_integer(periods):
            raise TypeError(f'periods must be a number, got {periods}')
    return periods

def validate_inferred_freq(freq, inferred_freq, freq_infer) -> tuple[BaseOffset | None, bool]:
    if False:
        while True:
            i = 10
    '\n    If the user passes a freq and another freq is inferred from passed data,\n    require that they match.\n\n    Parameters\n    ----------\n    freq : DateOffset or None\n    inferred_freq : DateOffset or None\n    freq_infer : bool\n\n    Returns\n    -------\n    freq : DateOffset or None\n    freq_infer : bool\n\n    Notes\n    -----\n    We assume at this point that `maybe_infer_freq` has been called, so\n    `freq` is either a DateOffset object or None.\n    '
    if inferred_freq is not None:
        if freq is not None and freq != inferred_freq:
            raise ValueError(f'Inferred frequency {inferred_freq} from passed values does not conform to passed frequency {freq.freqstr}')
        if freq is None:
            freq = inferred_freq
        freq_infer = False
    return (freq, freq_infer)

def maybe_infer_freq(freq):
    if False:
        while True:
            i = 10
    '\n    Comparing a DateOffset to the string "infer" raises, so we need to\n    be careful about comparisons.  Make a dummy variable `freq_infer` to\n    signify the case where the given freq is "infer" and set freq to None\n    to avoid comparison trouble later on.\n\n    Parameters\n    ----------\n    freq : {DateOffset, None, str}\n\n    Returns\n    -------\n    freq : {DateOffset, None}\n    freq_infer : bool\n        Whether we should inherit the freq of passed data.\n    '
    freq_infer = False
    if not isinstance(freq, BaseOffset):
        if freq != 'infer':
            freq = to_offset(freq)
        else:
            freq_infer = True
            freq = None
    return (freq, freq_infer)

def dtype_to_unit(dtype: DatetimeTZDtype | np.dtype) -> str:
    if False:
        i = 10
        return i + 15
    "\n    Return the unit str corresponding to the dtype's resolution.\n\n    Parameters\n    ----------\n    dtype : DatetimeTZDtype or np.dtype\n        If np.dtype, we assume it is a datetime64 dtype.\n\n    Returns\n    -------\n    str\n    "
    if isinstance(dtype, DatetimeTZDtype):
        return dtype.unit
    return np.datetime_data(dtype)[0]