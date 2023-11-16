"""Base class for Frame types that only have a single column."""
from __future__ import annotations
import warnings
from typing import Any, Dict, Optional, Tuple, Union
import cupy
import numpy
import cudf
from cudf._typing import Dtype, NotImplementedType, ScalarLike
from cudf.api.extensions import no_default
from cudf.api.types import _is_scalar_or_zero_d_array, is_bool_dtype, is_integer, is_integer_dtype
from cudf.core.column import ColumnBase, as_column
from cudf.core.frame import Frame
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import NotIterable

class SingleColumnFrame(Frame, NotIterable):
    """A one-dimensional frame.

    Frames with only a single column share certain logic that is encoded in
    this class.
    """
    _SUPPORT_AXIS_LOOKUP = {0: 0, 'index': 0}

    @_cudf_nvtx_annotate
    def _reduce(self, op, axis=no_default, level=None, numeric_only=None, **kwargs):
        if False:
            print('Hello World!')
        if axis not in (None, 0, no_default):
            raise NotImplementedError('axis parameter is not implemented yet')
        if level is not None:
            raise NotImplementedError('level parameter is not implemented yet')
        if numeric_only and (not isinstance(self._column, cudf.core.column.numerical_base.NumericalBaseColumn)):
            raise NotImplementedError(f'Series.{op} does not implement numeric_only.')
        try:
            return getattr(self._column, op)(**kwargs)
        except AttributeError:
            raise TypeError(f'cannot perform {op} with type {self.dtype}')

    @_cudf_nvtx_annotate
    def _scan(self, op, axis=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        if axis not in (None, 0):
            raise NotImplementedError('axis parameter is not implemented yet')
        return super()._scan(op, *args, axis=axis, **kwargs)

    @property
    @_cudf_nvtx_annotate
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the name of this object.'
        return next(iter(self._data.names))

    @name.setter
    @_cudf_nvtx_annotate
    def name(self, value):
        if False:
            i = 10
            return i + 15
        self._data[value] = self._data.pop(self.name)

    @property
    @_cudf_nvtx_annotate
    def ndim(self):
        if False:
            i = 10
            return i + 15
        'Number of dimensions of the underlying data, by definition 1.'
        return 1

    @property
    @_cudf_nvtx_annotate
    def shape(self):
        if False:
            i = 10
            return i + 15
        'Get a tuple representing the dimensionality of the Index.'
        return (len(self),)

    def __bool__(self):
        if False:
            while True:
                i = 10
        raise TypeError(f'The truth value of a {type(self)} is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().')

    @property
    @_cudf_nvtx_annotate
    def _num_columns(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    @property
    @_cudf_nvtx_annotate
    def _column(self):
        if False:
            i = 10
            return i + 15
        return self._data[self.name]

    @_column.setter
    @_cudf_nvtx_annotate
    def _column(self, value):
        if False:
            i = 10
            return i + 15
        self._data[self.name] = value

    @property
    @_cudf_nvtx_annotate
    def values(self):
        if False:
            i = 10
            return i + 15
        return self._column.values

    @property
    @_cudf_nvtx_annotate
    def values_host(self):
        if False:
            print('Hello World!')
        return self._column.values_host

    @_cudf_nvtx_annotate
    def to_cupy(self, dtype: Union[Dtype, None]=None, copy: bool=True, na_value=None) -> cupy.ndarray:
        if False:
            i = 10
            return i + 15
        return super().to_cupy(dtype, copy, na_value).flatten()

    @_cudf_nvtx_annotate
    def to_numpy(self, dtype: Union[Dtype, None]=None, copy: bool=True, na_value=None) -> numpy.ndarray:
        if False:
            i = 10
            return i + 15
        return super().to_numpy(dtype, copy, na_value).flatten()

    @classmethod
    @_cudf_nvtx_annotate
    def from_arrow(cls, array):
        if False:
            for i in range(10):
                print('nop')
        'Create from PyArrow Array/ChunkedArray.\n\n        Parameters\n        ----------\n        array : PyArrow Array/ChunkedArray\n            PyArrow Object which has to be converted.\n\n        Raises\n        ------\n        TypeError for invalid input type.\n\n        Returns\n        -------\n        SingleColumnFrame\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import pyarrow as pa\n        >>> cudf.Index.from_arrow(pa.array(["a", "b", None]))\n        StringIndex([\'a\' \'b\' None], dtype=\'object\')\n        >>> cudf.Series.from_arrow(pa.array(["a", "b", None]))\n        0       a\n        1       b\n        2    <NA>\n        dtype: object\n        '
        return cls(ColumnBase.from_arrow(array))

    @_cudf_nvtx_annotate
    def to_arrow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert to a PyArrow Array.\n\n        Returns\n        -------\n        PyArrow Array\n\n        Examples\n        --------\n        >>> import cudf\n        >>> sr = cudf.Series(["a", "b", None])\n        >>> sr.to_arrow()\n        <pyarrow.lib.StringArray object at 0x7f796b0e7600>\n        [\n          "a",\n          "b",\n          null\n        ]\n        >>> ind = cudf.Index(["a", "b", None])\n        >>> ind.to_arrow()\n        <pyarrow.lib.StringArray object at 0x7f796b0e7750>\n        [\n          "a",\n          "b",\n          null\n        ]\n        '
        return self._column.to_arrow()

    @property
    @_cudf_nvtx_annotate
    def is_monotonic(self):
        if False:
            while True:
                i = 10
        'Return boolean if values in the object are monotonically increasing.\n\n        This property is an alias for :attr:`is_monotonic_increasing`.\n\n        Returns\n        -------\n        bool\n        '
        warnings.warn('is_monotonic is deprecated and will be removed in a future version. Use is_monotonic_increasing instead.', FutureWarning)
        return self.is_monotonic_increasing

    @property
    @_cudf_nvtx_annotate
    def is_monotonic_increasing(self):
        if False:
            while True:
                i = 10
        'Return boolean if values in the object are monotonically increasing.\n\n        Returns\n        -------\n        bool\n        '
        return self._column.is_monotonic_increasing

    @property
    @_cudf_nvtx_annotate
    def is_monotonic_decreasing(self):
        if False:
            for i in range(10):
                print('nop')
        'Return boolean if values in the object are monotonically decreasing.\n\n        Returns\n        -------\n        bool\n        '
        return self._column.is_monotonic_decreasing

    @property
    @_cudf_nvtx_annotate
    def __cuda_array_interface__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._column.__cuda_array_interface__
        except NotImplementedError:
            raise AttributeError

    @_cudf_nvtx_annotate
    def factorize(self, sort=False, na_sentinel=None, use_na_sentinel=None):
        if False:
            return 10
        "Encode the input values as integer labels.\n\n        Parameters\n        ----------\n        sort : bool, default True\n            Sort uniques and shuffle codes to maintain the relationship.\n        na_sentinel : number, default -1\n            Value to indicate missing category.\n\n            .. deprecated:: 23.04\n\n               The na_sentinel argument is deprecated and will be removed in\n               a future version of cudf. Specify use_na_sentinel as\n               either True or False.\n        use_na_sentinel : bool, default True\n            If True, the sentinel -1 will be used for NA values.\n            If False, NA values will be encoded as non-negative\n            integers and will not drop the NA from the uniques\n            of the values.\n\n        Returns\n        -------\n        (labels, cats) : (cupy.ndarray, cupy.ndarray or Index)\n            - *labels* contains the encoded values\n            - *cats* contains the categories in order that the N-th\n              item corresponds to the (N-1) code.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> s = cudf.Series(['a', 'a', 'c'])\n        >>> codes, uniques = s.factorize()\n        >>> codes\n        array([0, 0, 1], dtype=int8)\n        >>> uniques\n        StringIndex(['a' 'c'], dtype='object')\n        "
        return cudf.core.algorithms.factorize(self, sort=sort, na_sentinel=na_sentinel, use_na_sentinel=use_na_sentinel)

    @_cudf_nvtx_annotate
    def _make_operands_for_binop(self, other: Any, fill_value: Any=None, reflect: bool=False, *args, **kwargs) -> Union[Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]], NotImplementedType]:
        if False:
            while True:
                i = 10
        'Generate the dictionary of operands used for a binary operation.\n\n        Parameters\n        ----------\n        other : SingleColumnFrame\n            The second operand.\n        fill_value : Any, default None\n            The value to replace null values with. If ``None``, nulls are not\n            filled before the operation.\n        reflect : bool, default False\n            If ``True``, swap the order of the operands. See\n            https://docs.python.org/3/reference/datamodel.html#object.__ror__\n            for more information on when this is necessary.\n\n        Returns\n        -------\n        Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]]\n            The operands to be passed to _colwise_binop.\n        '
        if isinstance(other, SingleColumnFrame) and (not cudf.utils.utils._is_same_name(self.name, other.name)):
            result_name = None
        else:
            result_name = self.name
        if isinstance(other, SingleColumnFrame):
            other = other._column
        elif not _is_scalar_or_zero_d_array(other):
            if not hasattr(other, '__cuda_array_interface__') and (not isinstance(other, cudf.RangeIndex)):
                return NotImplemented
            try:
                other = as_column(other)
            except Exception:
                return NotImplemented
        return {result_name: (self._column, other, reflect, fill_value)}

    @_cudf_nvtx_annotate
    def nunique(self, dropna: bool=True):
        if False:
            while True:
                i = 10
        "\n        Return count of unique values for the column.\n\n        Parameters\n        ----------\n        dropna : bool, default True\n            Don't include NaN in the counts.\n\n        Returns\n        -------\n        int\n            Number of unique values in the column.\n        "
        if self._column.null_count == len(self):
            return 0
        return self._column.distinct_count(dropna=dropna)

    def _get_elements_from_column(self, arg) -> Union[ScalarLike, ColumnBase]:
        if False:
            for i in range(10):
                print('nop')
        if _is_scalar_or_zero_d_array(arg):
            if not is_integer(arg):
                raise ValueError(f'Can only select elements with an integer, not a {type(arg).__name__}')
            return self._column.element_indexing(int(arg))
        elif isinstance(arg, slice):
            (start, stop, stride) = arg.indices(len(self))
            return self._column.slice(start, stop, stride)
        else:
            arg = as_column(arg)
            if len(arg) == 0:
                arg = as_column([], dtype='int32')
            if is_integer_dtype(arg.dtype):
                return self._column.take(arg)
            if is_bool_dtype(arg.dtype):
                if (bn := len(arg)) != (n := len(self)):
                    raise IndexError(f'Boolean mask has wrong length: {bn} not {n}')
                return self._column.apply_boolean_mask(arg)
            raise NotImplementedError(f'Unknown indexer {type(arg)}')

    @_cudf_nvtx_annotate
    def where(self, cond, other=None, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        from cudf.core._internals.where import _check_and_cast_columns_with_other, _make_categorical_like
        if isinstance(other, cudf.DataFrame):
            raise NotImplementedError('cannot align with a higher dimensional Frame')
        cond = as_column(cond)
        if len(cond) != len(self):
            raise ValueError('Array conditional must be same shape as self')
        if not cudf.api.types.is_scalar(other):
            other = cudf.core.column.as_column(other)
        self_column = self._column
        (input_col, other) = _check_and_cast_columns_with_other(source_col=self_column, other=other, inplace=inplace)
        result = cudf._lib.copying.copy_if_else(input_col, other, cond)
        return _make_categorical_like(result, self_column)