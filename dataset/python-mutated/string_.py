from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Literal
import numpy as np
from pandas._config import get_option
from pandas._libs import lib, missing as libmissing
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import pa_version_under10p1
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc
from pandas.core.dtypes.base import ExtensionDtype, StorageExtensionDtype, register_extension_dtype
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_integer_dtype, is_object_dtype, is_string_dtype, pandas_dtype
from pandas.core import ops
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import FloatingArray, FloatingDtype
from pandas.core.arrays.integer import IntegerArray, IntegerDtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna
if TYPE_CHECKING:
    import pyarrow
    from pandas._typing import AxisInt, Dtype, DtypeObj, NumpySorter, NumpyValueArrayLike, Scalar, Self, npt, type_t
    from pandas import Series

@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    """
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow", "pyarrow_numpy"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    """
    name: ClassVar[str] = 'string'

    @property
    def na_value(self) -> libmissing.NAType | float:
        if False:
            print('Hello World!')
        if self.storage == 'pyarrow_numpy':
            return np.nan
        else:
            return libmissing.NA
    _metadata = ('storage',)

    def __init__(self, storage=None) -> None:
        if False:
            i = 10
            return i + 15
        if storage is None:
            infer_string = get_option('future.infer_string')
            if infer_string:
                storage = 'pyarrow_numpy'
            else:
                storage = get_option('mode.string_storage')
        if storage not in {'python', 'pyarrow', 'pyarrow_numpy'}:
            raise ValueError(f"Storage must be 'python', 'pyarrow' or 'pyarrow_numpy'. Got {storage} instead.")
        if storage in ('pyarrow', 'pyarrow_numpy') and pa_version_under10p1:
            raise ImportError('pyarrow>=10.0.1 is required for PyArrow backed StringArray.')
        self.storage = storage

    @property
    def type(self) -> type[str]:
        if False:
            return 10
        return str

    @classmethod
    def construct_from_string(cls, string) -> Self:
        if False:
            while True:
                i = 10
        "\n        Construct a StringDtype from a string.\n\n        Parameters\n        ----------\n        string : str\n            The type of the name. The storage type will be taking from `string`.\n            Valid options and their storage types are\n\n            ========================== ==============================================\n            string                     result storage\n            ========================== ==============================================\n            ``'string'``               pd.options.mode.string_storage, default python\n            ``'string[python]'``       python\n            ``'string[pyarrow]'``      pyarrow\n            ========================== ==============================================\n\n        Returns\n        -------\n        StringDtype\n\n        Raise\n        -----\n        TypeError\n            If the string is not a valid option.\n        "
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string == 'string':
            return cls()
        elif string == 'string[python]':
            return cls(storage='python')
        elif string == 'string[pyarrow]':
            return cls(storage='pyarrow')
        elif string == 'string[pyarrow_numpy]':
            return cls(storage='pyarrow_numpy')
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def construct_array_type(self) -> type_t[BaseStringArray]:
        if False:
            return 10
        '\n        Return the array type associated with this dtype.\n\n        Returns\n        -------\n        type\n        '
        from pandas.core.arrays.string_arrow import ArrowStringArray, ArrowStringArrayNumpySemantics
        if self.storage == 'python':
            return StringArray
        elif self.storage == 'pyarrow':
            return ArrowStringArray
        else:
            return ArrowStringArrayNumpySemantics

    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BaseStringArray:
        if False:
            print('Hello World!')
        '\n        Construct StringArray from pyarrow Array/ChunkedArray.\n        '
        if self.storage == 'pyarrow':
            from pandas.core.arrays.string_arrow import ArrowStringArray
            return ArrowStringArray(array)
        elif self.storage == 'pyarrow_numpy':
            from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
            return ArrowStringArrayNumpySemantics(array)
        else:
            import pyarrow
            if isinstance(array, pyarrow.Array):
                chunks = [array]
            else:
                chunks = array.chunks
            results = []
            for arr in chunks:
                arr = arr.to_numpy(zero_copy_only=False)
                arr = ensure_string_array(arr, na_value=libmissing.NA)
                results.append(arr)
        if len(chunks) == 0:
            arr = np.array([], dtype=object)
        else:
            arr = np.concatenate(results)
        new_string_array = StringArray.__new__(StringArray)
        NDArrayBacked.__init__(new_string_array, arr, StringDtype(storage='python'))
        return new_string_array

class BaseStringArray(ExtensionArray):
    """
    Mixin class for StringArray, ArrowStringArray.
    """

    @doc(ExtensionArray.tolist)
    def tolist(self):
        if False:
            print('Hello World!')
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self.to_numpy())

    @classmethod
    def _from_scalars(cls, scalars, dtype: DtypeObj) -> Self:
        if False:
            print('Hello World!')
        if lib.infer_dtype(scalars, skipna=True) != 'string':
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

class StringArray(BaseStringArray, NumpyExtensionArray):
    """
    Extension array for string data.

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings
           or nan-likes (``None``, ``np.nan``, ``NA``).
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

        .. versionchanged:: 1.5.0

           StringArray now accepts array-likes containing
           nan-likes(``None``, ``np.nan``) for the ``values`` parameter
           in addition to strings and :attr:`pandas.NA`

    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`pandas.array`
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string")
    <StringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array(['1', 1], dtype="object")
    <NumpyExtensionArray>
    ['1', 1]
    Length: 2, dtype: object
    >>> pd.array(['1', 1], dtype="string")
    <StringArray>
    ['1', '1']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    """
    _typ = 'extension'

    def __init__(self, values, copy: bool=False) -> None:
        if False:
            while True:
                i = 10
        values = extract_array(values)
        super().__init__(values, copy=copy)
        if not isinstance(values, type(self)):
            self._validate()
        NDArrayBacked.__init__(self, self._ndarray, StringDtype(storage='python'))

    def _validate(self):
        if False:
            return 10
        'Validate that we only store NA or strings.'
        if len(self._ndarray) and (not lib.is_string_array(self._ndarray, skipna=True)):
            raise ValueError('StringArray requires a sequence of strings or pandas.NA')
        if self._ndarray.dtype != 'object':
            raise ValueError(f"StringArray requires a sequence of strings or pandas.NA. Got '{self._ndarray.dtype}' dtype instead.")
        if self._ndarray.ndim > 2:
            lib.convert_nans_to_NA(self._ndarray.ravel('K'))
        else:
            lib.convert_nans_to_NA(self._ndarray)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None=None, copy: bool=False):
        if False:
            while True:
                i = 10
        if dtype and (not (isinstance(dtype, str) and dtype == 'string')):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == 'python'
        from pandas.core.arrays.masked import BaseMaskedArray
        if isinstance(scalars, BaseMaskedArray):
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            result[na_values] = libmissing.NA
        else:
            if lib.is_pyarrow_array(scalars):
                scalars = np.array(scalars)
            result = lib.ensure_string_array(scalars, na_value=libmissing.NA, copy=copy)
        new_string_array = cls.__new__(cls)
        NDArrayBacked.__init__(new_string_array, result, StringDtype(storage='python'))
        return new_string_array

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None=None, copy: bool=False):
        if False:
            for i in range(10):
                print('nop')
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _empty(cls, shape, dtype) -> StringArray:
        if False:
            print('Hello World!')
        values = np.empty(shape, dtype=object)
        values[:] = libmissing.NA
        return cls(values).astype(dtype, copy=False)

    def __arrow_array__(self, type=None):
        if False:
            print('Hello World!')
        '\n        Convert myself into a pyarrow Array.\n        '
        import pyarrow as pa
        if type is None:
            type = pa.string()
        values = self._ndarray.copy()
        values[self.isna()] = None
        return pa.array(values, type=type, from_pandas=True)

    def _values_for_factorize(self):
        if False:
            return 10
        arr = self._ndarray.copy()
        mask = self.isna()
        arr[mask] = None
        return (arr, None)

    def __setitem__(self, key, value) -> None:
        if False:
            return 10
        value = extract_array(value, extract_numpy=True)
        if isinstance(value, type(self)):
            value = value._ndarray
        key = check_array_indexer(self, key)
        scalar_key = lib.is_scalar(key)
        scalar_value = lib.is_scalar(value)
        if scalar_key and (not scalar_value):
            raise ValueError('setting an array element with a sequence.')
        if scalar_value:
            if isna(value):
                value = libmissing.NA
            elif not isinstance(value, str):
                raise TypeError(f"Cannot set non-string value '{value}' into a StringArray.")
        else:
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            if len(value) and (not lib.is_string_array(value, skipna=True)):
                raise TypeError('Must provide strings.')
            mask = isna(value)
            if mask.any():
                value = value.copy()
                value[isna(value)] = libmissing.NA
        super().__setitem__(key, value)

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        if False:
            return 10
        ExtensionArray._putmask(self, mask, value)

    def astype(self, dtype, copy: bool=True):
        if False:
            for i in range(10):
                print('nop')
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, IntegerDtype):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype.numpy_dtype)
            return IntegerArray(values, mask, copy=False)
        elif isinstance(dtype, FloatingDtype):
            arr = self.copy()
            mask = self.isna()
            arr[mask] = '0'
            values = arr.astype(dtype.numpy_dtype)
            return FloatingArray(values, mask, copy=False)
        elif isinstance(dtype, ExtensionDtype):
            return ExtensionArray.astype(self, dtype, copy)
        elif np.issubdtype(dtype, np.floating):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype)
            values[mask] = np.nan
            return values
        return super().astype(dtype, copy)

    def _reduce(self, name: str, *, skipna: bool=True, axis: AxisInt | None=0, **kwargs):
        if False:
            print('Hello World!')
        if name in ['min', 'max']:
            return getattr(self, name)(skipna=skipna, axis=axis)
        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    def min(self, axis=None, skipna: bool=True, **kwargs) -> Scalar:
        if False:
            while True:
                i = 10
        nv.validate_min((), kwargs)
        result = masked_reductions.min(values=self.to_numpy(), mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def max(self, axis=None, skipna: bool=True, **kwargs) -> Scalar:
        if False:
            while True:
                i = 10
        nv.validate_max((), kwargs)
        result = masked_reductions.max(values=self.to_numpy(), mask=self.isna(), skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def value_counts(self, dropna: bool=True) -> Series:
        if False:
            print('Hello World!')
        from pandas.core.algorithms import value_counts_internal as value_counts
        result = value_counts(self._ndarray, dropna=dropna).astype('Int64')
        result.index = result.index.astype(self.dtype)
        return result

    def memory_usage(self, deep: bool=False) -> int:
        if False:
            print('Hello World!')
        result = self._ndarray.nbytes
        if deep:
            return result + lib.memory_usage_of_objects(self._ndarray)
        return result

    @doc(ExtensionArray.searchsorted)
    def searchsorted(self, value: NumpyValueArrayLike | ExtensionArray, side: Literal['left', 'right']='left', sorter: NumpySorter | None=None) -> npt.NDArray[np.intp] | np.intp:
        if False:
            while True:
                i = 10
        if self._hasna:
            raise ValueError('searchsorted requires array to be sorted, which is impossible with NAs present.')
        return super().searchsorted(value=value, side=side, sorter=sorter)

    def _cmp_method(self, other, op):
        if False:
            while True:
                i = 10
        from pandas.arrays import BooleanArray
        if isinstance(other, StringArray):
            other = other._ndarray
        mask = isna(self) | isna(other)
        valid = ~mask
        if not lib.is_scalar(other):
            if len(other) != len(self):
                raise ValueError(f'Lengths of operands do not match: {len(self)} != {len(other)}')
            other = np.asarray(other)
            other = other[valid]
        if op.__name__ in ops.ARITHMETIC_BINOPS:
            result = np.empty_like(self._ndarray, dtype='object')
            result[mask] = libmissing.NA
            result[valid] = op(self._ndarray[valid], other)
            return StringArray(result)
        else:
            result = np.zeros(len(self._ndarray), dtype='bool')
            result[valid] = op(self._ndarray[valid], other)
            return BooleanArray(result, mask)
    _arith_method = _cmp_method
    _str_na_value = libmissing.NA

    def _str_map(self, f, na_value=None, dtype: Dtype | None=None, convert: bool=True):
        if False:
            i = 10
            return i + 15
        from pandas.arrays import BooleanArray
        if dtype is None:
            dtype = StringDtype(storage='python')
        if na_value is None:
            na_value = self.dtype.na_value
        mask = isna(self)
        arr = np.asarray(self)
        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            constructor: type[IntegerArray | BooleanArray]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray
            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            elif dtype == np.dtype('bool'):
                na_value = bool(na_value)
            result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value, dtype=np.dtype(dtype))
            if not na_value_is_na:
                mask[:] = False
            return constructor(result, mask)
        elif is_string_dtype(dtype) and (not is_object_dtype(dtype)):
            result = lib.map_infer_mask(arr, f, mask.view('uint8'), convert=False, na_value=na_value)
            return StringArray(result)
        else:
            return lib.map_infer_mask(arr, f, mask.view('uint8'))