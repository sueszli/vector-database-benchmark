from __future__ import annotations
import builtins
import pickle
import warnings
from functools import cached_property
from typing import Any, Set, Tuple
import pandas as pd
from typing_extensions import Self
import cudf
from cudf._lib.copying import _gather_map_is_valid, gather
from cudf._lib.stream_compaction import apply_boolean_mask, drop_duplicates, drop_nulls
from cudf._lib.types import size_type_dtype
from cudf.api.extensions import no_default
from cudf.api.types import is_bool_dtype, is_integer, is_integer_dtype, is_list_like, is_scalar, is_signed_integer_dtype, is_unsigned_integer_dtype
from cudf.core.abc import Serializable
from cudf.core.column import ColumnBase, column
from cudf.core.column_accessor import ColumnAccessor
from cudf.errors import MixedTypeError
from cudf.utils import ioutils
from cudf.utils.dtypes import can_convert_to_column, is_mixed_with_object_dtype
from cudf.utils.utils import _is_same_name

class BaseIndex(Serializable):
    """Base class for all cudf Index types."""
    _accessors: Set[Any] = set()
    _data: ColumnAccessor

    @property
    def _columns(self) -> Tuple[Any, ...]:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @cached_property
    def _values(self) -> ColumnBase:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def copy(self, deep: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __len__(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return len(self)

    def astype(self, dtype, copy: bool=True):
        if False:
            i = 10
            return i + 15
        "Create an Index with values cast to dtypes.\n\n        The class of a new Index is determined by dtype. When conversion is\n        impossible, a ValueError exception is raised.\n\n        Parameters\n        ----------\n        dtype : :class:`numpy.dtype`\n            Use a :class:`numpy.dtype` to cast entire Index object to.\n        copy : bool, default False\n            By default, astype always returns a newly allocated object.\n            If copy is set to False and internal requirements on dtype are\n            satisfied, the original data is used to create a new Index\n            or the original Index is returned.\n\n        Returns\n        -------\n        Index\n            Index with values cast to specified dtype.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> index = cudf.Index([1, 2, 3])\n        >>> index\n        Int64Index([1, 2, 3], dtype='int64')\n        >>> index.astype('float64')\n        Float64Index([1.0, 2.0, 3.0], dtype='float64')\n        "
        raise NotImplementedError

    def argsort(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Return the integer indices that would sort the index.\n\n        Parameters vary by subclass.\n        '
        raise NotImplementedError

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @property
    def empty(self):
        if False:
            while True:
                i = 10
        return self.size == 0

    @property
    def is_unique(self):
        if False:
            i = 10
            return i + 15
        'Return if the index has unique values.'
        raise NotImplementedError

    def memory_usage(self, deep=False):
        if False:
            print('Hello World!')
        'Return the memory usage of an object.\n\n        Parameters\n        ----------\n        deep : bool\n            The deep parameter is ignored and is only included for pandas\n            compatibility.\n\n        Returns\n        -------\n        The total bytes used.\n        '
        raise NotImplementedError

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('cuDF does not support conversion to host memory via the `tolist()` method. Consider using `.to_arrow().to_pylist()` to construct a Python list.')
    to_list = tolist

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'Returns the name of the Index.'
        raise NotImplementedError

    @property
    def ndim(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of dimensions of the underlying data, by definition 1.'
        return 1

    def equals(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if two Index objects contain the same elements.\n\n        Returns\n        -------\n        out: bool\n            True if "other" is an Index and it has the same elements\n            as calling index; False otherwise.\n        '
        raise NotImplementedError

    def shift(self, periods=1, freq=None):
        if False:
            while True:
                i = 10
        'Not yet implemented'
        raise NotImplementedError

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        'Get a tuple representing the dimensionality of the data.'
        return (len(self),)

    @property
    def str(self):
        if False:
            print('Hello World!')
        'Not yet implemented.'
        raise NotImplementedError

    @property
    def values(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def max(self):
        if False:
            i = 10
            return i + 15
        'The maximum value of the index.'
        raise NotImplementedError

    def min(self):
        if False:
            i = 10
            return i + 15
        'The minimum value of the index.'
        raise NotImplementedError

    def get_loc(self, key, method=None, tolerance=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def __contains__(self, item):
        if False:
            return 10
        return item in self._values

    def _copy_type_metadata(self, other: Self, *, override_dtypes=None) -> Self:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def get_level_values(self, level):
        if False:
            return 10
        '\n        Return an Index of values for requested level.\n\n        This is primarily useful to get an individual level of values from a\n        MultiIndex, but is provided on Index as well for compatibility.\n\n        Parameters\n        ----------\n        level : int or str\n            It is either the integer position or the name of the level.\n\n        Returns\n        -------\n        Index\n            Calling object, as there is only one level in the Index.\n\n        See Also\n        --------\n        cudf.MultiIndex.get_level_values : Get values for\n            a level of a MultiIndex.\n\n        Notes\n        -----\n        For Index, level should be 0, since there are no multiple levels.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index(["a", "b", "c"])\n        >>> idx.get_level_values(0)\n        StringIndex([\'a\' \'b\' \'c\'], dtype=\'object\')\n        '
        if level == self.name:
            return self
        elif is_integer(level):
            if level != 0:
                raise IndexError(f'Cannot get level: {level} for index with 1 level')
            return self
        else:
            raise KeyError(f'Requested level with name {level} not found')

    @classmethod
    def deserialize(cls, header, frames):
        if False:
            for i in range(10):
                print('nop')
        idx_type = pickle.loads(header['type-serialized'])
        return idx_type.deserialize(header, frames)

    @property
    def names(self):
        if False:
            return 10
        '\n        Returns a tuple containing the name of the Index.\n        '
        return (self.name,)

    @names.setter
    def names(self, values):
        if False:
            return 10
        if not is_list_like(values):
            raise ValueError('Names must be a list-like')
        num_values = len(values)
        if num_values > 1:
            raise ValueError('Length of new names must be 1, got %d' % num_values)
        self.name = values[0]

    def _clean_nulls_from_index(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert all na values(if any) in Index object\n        to `<NA>` as a preprocessing step to `__repr__` methods.\n\n        This will involve changing type of Index object\n        to StringIndex but it is the responsibility of the `__repr__`\n        methods using this method to replace or handle representation\n        of the actual types correctly.\n        '
        raise NotImplementedError

    @property
    def is_monotonic(self):
        if False:
            print('Hello World!')
        'Return boolean if values in the object are monotonic_increasing.\n\n        This property is an alias for :attr:`is_monotonic_increasing`.\n\n        Returns\n        -------\n        bool\n        '
        warnings.warn('is_monotonic is deprecated and will be removed in a future version. Use is_monotonic_increasing instead.', FutureWarning)
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        if False:
            for i in range(10):
                print('nop')
        'Return boolean if values in the object are monotonically increasing.\n\n        Returns\n        -------\n        bool\n        '
        raise NotImplementedError

    @property
    def is_monotonic_decreasing(self):
        if False:
            return 10
        'Return boolean if values in the object are monotonically decreasing.\n\n        Returns\n        -------\n        bool\n        '
        raise NotImplementedError

    @property
    def hasnans(self):
        if False:
            print('Hello World!')
        "\n        Return True if there are any NaNs or nulls.\n\n        Returns\n        -------\n        out : bool\n            If Series has at least one NaN or null value, return True,\n            if not return False.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import numpy as np\n        >>> index = cudf.Index([1, 2, np.nan, 3, 4], nan_as_null=False)\n        >>> index\n        Float64Index([1.0, 2.0, nan, 3.0, 4.0], dtype='float64')\n        >>> index.hasnans\n        True\n\n        `hasnans` returns `True` for the presence of any `NA` values:\n\n        >>> index = cudf.Index([1, 2, None, 3, 4])\n        >>> index\n        Int64Index([1, 2, <NA>, 3, 4], dtype='int64')\n        >>> index.hasnans\n        True\n        "
        raise NotImplementedError

    @property
    def nlevels(self):
        if False:
            return 10
        '\n        Number of levels.\n        '
        return 1

    def _set_names(self, names, inplace=False):
        if False:
            while True:
                i = 10
        if inplace:
            idx = self
        else:
            idx = self.copy(deep=False)
        idx.names = names
        if not inplace:
            return idx

    def set_names(self, names, level=None, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set Index or MultiIndex name.\n        Able to set new names partially and by level.\n\n        Parameters\n        ----------\n        names : label or list of label\n            Name(s) to set.\n        level : int, label or list of int or label, optional\n            If the index is a MultiIndex, level(s) to set (None for all\n            levels). Otherwise level must be None.\n        inplace : bool, default False\n            Modifies the object directly, instead of creating a new Index or\n            MultiIndex.\n\n        Returns\n        -------\n        Index\n            The same type as the caller or None if inplace is True.\n\n        See Also\n        --------\n        cudf.Index.rename : Able to set new names without level.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([1, 2, 3, 4])\n        >>> idx\n        Int64Index([1, 2, 3, 4], dtype='int64')\n        >>> idx.set_names('quarter')\n        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')\n        >>> idx = cudf.MultiIndex.from_product([['python', 'cobra'],\n        ... [2018, 2019]])\n        >>> idx\n        MultiIndex([('python', 2018),\n                    ('python', 2019),\n                    ( 'cobra', 2018),\n                    ( 'cobra', 2019)],\n                   )\n        >>> idx.names\n        FrozenList([None, None])\n        >>> idx.set_names(['kind', 'year'], inplace=True)\n        >>> idx.names\n        FrozenList(['kind', 'year'])\n        >>> idx.set_names('species', level=0, inplace=True)\n        >>> idx.names\n        FrozenList(['species', 'year'])\n        "
        if level is not None:
            raise ValueError('Level must be None for non-MultiIndex')
        if not is_list_like(names):
            names = [names]
        return self._set_names(names=names, inplace=inplace)

    @property
    def has_duplicates(self):
        if False:
            return 10
        return not self.is_unique

    def where(self, cond, other=None, inplace=False):
        if False:
            print('Hello World!')
        '\n        Replace values where the condition is False.\n\n        The replacement is taken from other.\n\n        Parameters\n        ----------\n        cond : bool array-like with the same length as self\n            Condition to select the values on.\n        other : scalar, or array-like, default None\n            Replacement if the condition is False.\n\n        Returns\n        -------\n        cudf.Index\n            A copy of self with values replaced from other\n            where the condition is False.\n        '
        raise NotImplementedError

    def factorize(self, sort=False, na_sentinel=None, use_na_sentinel=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def union(self, other, sort=None):
        if False:
            print('Hello World!')
        '\n        Form the union of two Index objects.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : bool or None, default None\n            Whether to sort the resulting Index.\n\n            * None : Sort the result, except when\n\n              1. `self` and `other` are equal.\n              2. `self` or `other` has length 0.\n\n            * False : do not sort the result.\n\n        Returns\n        -------\n        union : Index\n\n        Examples\n        --------\n        Union of an Index\n        >>> import cudf\n        >>> import pandas as pd\n        >>> idx1 = cudf.Index([1, 2, 3, 4])\n        >>> idx2 = cudf.Index([3, 4, 5, 6])\n        >>> idx1.union(idx2)\n        Int64Index([1, 2, 3, 4, 5, 6], dtype=\'int64\')\n\n        MultiIndex case\n\n        >>> idx1 = cudf.MultiIndex.from_pandas(\n        ...    pd.MultiIndex.from_arrays(\n        ...         [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]\n        ...    )\n        ... )\n        >>> idx1\n        MultiIndex([(1,  \'Red\'),\n                    (1, \'Blue\'),\n                    (2,  \'Red\'),\n                    (2, \'Blue\')],\n                   )\n        >>> idx2 = cudf.MultiIndex.from_pandas(\n        ...    pd.MultiIndex.from_arrays(\n        ...         [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]\n        ...    )\n        ... )\n        >>> idx2\n        MultiIndex([(3,   \'Red\'),\n                    (3, \'Green\'),\n                    (2,   \'Red\'),\n                    (2, \'Green\')],\n                   )\n        >>> idx1.union(idx2)\n        MultiIndex([(1,  \'Blue\'),\n                    (1,   \'Red\'),\n                    (2,  \'Blue\'),\n                    (2, \'Green\'),\n                    (2,   \'Red\'),\n                    (3, \'Green\'),\n                    (3,   \'Red\')],\n                   )\n        >>> idx1.union(idx2, sort=False)\n        MultiIndex([(1,   \'Red\'),\n                    (1,  \'Blue\'),\n                    (2,   \'Red\'),\n                    (2,  \'Blue\'),\n                    (3,   \'Red\'),\n                    (3, \'Green\'),\n                    (2, \'Green\')],\n                   )\n        '
        if not isinstance(other, BaseIndex):
            other = cudf.Index(other, name=self.name)
        if sort not in {None, False}:
            raise ValueError(f"The 'sort' keyword only takes the values of None or False; {sort} was passed.")
        if cudf.get_option('mode.pandas_compatible'):
            if is_bool_dtype(self.dtype) and (not is_bool_dtype(other.dtype)) or (not is_bool_dtype(self.dtype) and is_bool_dtype(other.dtype)):
                raise MixedTypeError('Cannot perform union with mixed types')
            if is_signed_integer_dtype(self.dtype) and is_unsigned_integer_dtype(other.dtype) or (is_unsigned_integer_dtype(self.dtype) and is_signed_integer_dtype(other.dtype)):
                raise MixedTypeError('Cannot perform union with mixed types')
        if not len(other) or self.equals(other):
            common_dtype = cudf.utils.dtypes.find_common_type([self.dtype, other.dtype])
            return self._get_reconciled_name_object(other).astype(common_dtype)
        elif not len(self):
            common_dtype = cudf.utils.dtypes.find_common_type([self.dtype, other.dtype])
            return other._get_reconciled_name_object(self).astype(common_dtype)
        result = self._union(other, sort=sort)
        result.name = _get_result_name(self.name, other.name)
        return result

    def intersection(self, other, sort=False):
        if False:
            print('Hello World!')
        '\n        Form the intersection of two Index objects.\n\n        This returns a new Index with elements common to the index and `other`.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : False or None, default False\n            Whether to sort the resulting index.\n\n            * False : do not sort the result.\n            * None : sort the result, except when `self` and `other` are equal\n              or when the values cannot be compared.\n\n        Returns\n        -------\n        intersection : Index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import pandas as pd\n        >>> idx1 = cudf.Index([1, 2, 3, 4])\n        >>> idx2 = cudf.Index([3, 4, 5, 6])\n        >>> idx1.intersection(idx2)\n        Int64Index([3, 4], dtype=\'int64\')\n\n        MultiIndex case\n\n        >>> idx1 = cudf.MultiIndex.from_pandas(\n        ...    pd.MultiIndex.from_arrays(\n        ...         [[1, 1, 3, 4], ["Red", "Blue", "Red", "Blue"]]\n        ...    )\n        ... )\n        >>> idx2 = cudf.MultiIndex.from_pandas(\n        ...    pd.MultiIndex.from_arrays(\n        ...         [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]\n        ...    )\n        ... )\n        >>> idx1\n        MultiIndex([(1,  \'Red\'),\n                    (1, \'Blue\'),\n                    (3,  \'Red\'),\n                    (4, \'Blue\')],\n                )\n        >>> idx2\n        MultiIndex([(1,  \'Red\'),\n                    (1, \'Blue\'),\n                    (2,  \'Red\'),\n                    (2, \'Blue\')],\n                )\n        >>> idx1.intersection(idx2)\n        MultiIndex([(1,  \'Red\'),\n                    (1, \'Blue\')],\n                )\n        >>> idx1.intersection(idx2, sort=False)\n        MultiIndex([(1,  \'Red\'),\n                    (1, \'Blue\')],\n                )\n        '
        if not can_convert_to_column(other):
            raise TypeError('Input must be Index or array-like')
        if not isinstance(other, BaseIndex):
            other = cudf.Index(other, name=getattr(other, 'name', self.name))
        if sort not in {None, False}:
            raise ValueError(f"The 'sort' keyword only takes the values of None or False; {sort} was passed.")
        if not len(self) or not len(other) or self.equals(other):
            common_dtype = cudf.utils.dtypes._dtype_pandas_compatible(cudf.utils.dtypes.find_common_type([self.dtype, other.dtype]))
            lhs = self.unique() if self.has_duplicates else self
            rhs = other
            if not len(other):
                (lhs, rhs) = (rhs, lhs)
            return lhs._get_reconciled_name_object(rhs).astype(common_dtype)
        res_name = _get_result_name(self.name, other.name)
        if self._is_boolean() and other._is_numeric() or (self._is_numeric() and other._is_boolean()):
            if isinstance(self, cudf.MultiIndex):
                return self[:0].rename(res_name)
            else:
                return cudf.Index([], name=res_name)
        if self.has_duplicates:
            lhs = self.unique()
        else:
            lhs = self
        if other.has_duplicates:
            rhs = other.unique()
        else:
            rhs = other
        result = lhs._intersection(rhs, sort=sort)
        result.name = res_name
        return result

    def _get_reconciled_name_object(self, other):
        if False:
            print('Hello World!')
        '\n        If the result of a set operation will be self,\n        return self, unless the name changes, in which\n        case make a shallow copy of self.\n        '
        name = _get_result_name(self.name, other.name)
        if not _is_same_name(self.name, name):
            return self.rename(name)
        return self

    def fillna(self, value, downcast=None):
        if False:
            print('Hello World!')
        "\n        Fill null values with the specified value.\n\n        Parameters\n        ----------\n        value : scalar\n            Scalar value to use to fill nulls. This value cannot be a\n            list-likes.\n\n        downcast : dict, default is None\n            This Parameter is currently NON-FUNCTIONAL.\n\n        Returns\n        -------\n        filled : Index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> index = cudf.Index([1, 2, None, 4])\n        >>> index\n        Int64Index([1, 2, <NA>, 4], dtype='int64')\n        >>> index.fillna(3)\n        Int64Index([1, 2, 3, 4], dtype='int64')\n        "
        if downcast is not None:
            raise NotImplementedError('`downcast` parameter is not yet supported')
        return super().fillna(value=value)

    def to_frame(self, index=True, name=no_default):
        if False:
            for i in range(10):
                print('nop')
        "Create a DataFrame with a column containing this Index\n\n        Parameters\n        ----------\n        index : boolean, default True\n            Set the index of the returned DataFrame as the original Index\n        name : object, defaults to index.name\n            The passed name should substitute for the index name (if it has\n            one).\n\n        Returns\n        -------\n        DataFrame\n            DataFrame containing the original Index data.\n\n        See Also\n        --------\n        Index.to_series : Convert an Index to a Series.\n        Series.to_frame : Convert Series to DataFrame.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index(['Ant', 'Bear', 'Cow'], name='animal')\n        >>> idx.to_frame()\n               animal\n        animal\n        Ant       Ant\n        Bear     Bear\n        Cow       Cow\n\n        By default, the original Index is reused. To enforce a new Index:\n\n        >>> idx.to_frame(index=False)\n            animal\n        0   Ant\n        1  Bear\n        2   Cow\n\n        To override the name of the resulting column, specify `name`:\n\n        >>> idx.to_frame(index=False, name='zoo')\n            zoo\n        0   Ant\n        1  Bear\n        2   Cow\n        "
        if name is None:
            warnings.warn("Explicitly passing `name=None` currently preserves the Index's name or uses a default name of 0. This behaviour is deprecated, and in the future `None` will be used as the name of the resulting DataFrame column.", FutureWarning)
            name = no_default
        if name is not no_default:
            col_name = name
        elif self.name is None:
            col_name = 0
        else:
            col_name = self.name
        return cudf.DataFrame({col_name: self._values}, index=self if index else None)

    def to_arrow(self):
        if False:
            i = 10
            return i + 15
        'Convert to a suitable Arrow object.'
        raise NotImplementedError

    def to_cupy(self):
        if False:
            while True:
                i = 10
        'Convert to a cupy array.'
        raise NotImplementedError

    def to_numpy(self):
        if False:
            return 10
        'Convert to a numpy array.'
        raise NotImplementedError

    def any(self):
        if False:
            print('Hello World!')
        '\n        Return whether any elements is True in Index.\n        '
        raise NotImplementedError

    def isna(self):
        if False:
            i = 10
            return i + 15
        '\n        Detect missing values.\n\n        Return a boolean same-sized object indicating if the values are NA.\n        NA values, such as ``None``, `numpy.NAN` or `cudf.NA`, get\n        mapped to ``True`` values.\n        Everything else get mapped to ``False`` values.\n\n        Returns\n        -------\n        numpy.ndarray[bool]\n            A boolean array to indicate which entries are NA.\n\n        '
        raise NotImplementedError

    def notna(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detect existing (non-missing) values.\n\n        Return a boolean same-sized object indicating if the values are not NA.\n        Non-missing values get mapped to ``True``.\n        NA values, such as None or `numpy.NAN`, get mapped to ``False``\n        values.\n\n        Returns\n        -------\n        numpy.ndarray[bool]\n            A boolean array to indicate which entries are not NA.\n        '
        raise NotImplementedError

    def to_pandas(self, nullable=False):
        if False:
            return 10
        "\n        Convert to a Pandas Index.\n\n        Parameters\n        ----------\n        nullable : bool, Default False\n            If ``nullable`` is ``True``, the resulting index will have\n            a corresponding nullable Pandas dtype.\n            If there is no corresponding nullable Pandas dtype present,\n            the resulting dtype will be a regular pandas dtype.\n            If ``nullable`` is ``False``, the resulting index will\n            either convert null values to ``np.nan`` or ``None``\n            depending on the dtype.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([-3, 10, 15, 20])\n        >>> idx\n        Int64Index([-3, 10, 15, 20], dtype='int64')\n        >>> idx.to_pandas()\n        Int64Index([-3, 10, 15, 20], dtype='int64')\n        >>> type(idx.to_pandas())\n        <class 'pandas.core.indexes.numeric.Int64Index'>\n        >>> type(idx)\n        <class 'cudf.core.index.Int64Index'>\n        "
        raise NotImplementedError

    def isin(self, values):
        if False:
            print('Hello World!')
        "Return a boolean array where the index values are in values.\n\n        Compute boolean array of whether each index value is found in\n        the passed set of values. The length of the returned boolean\n        array matches the length of the index.\n\n        Parameters\n        ----------\n        values : set, list-like, Index\n            Sought values.\n\n        Returns\n        -------\n        is_contained : cupy array\n            CuPy array of boolean values.\n\n        Examples\n        --------\n        >>> idx = cudf.Index([1,2,3])\n        >>> idx\n        Int64Index([1, 2, 3], dtype='int64')\n\n        Check whether each index value in a list of values.\n\n        >>> idx.isin([1, 4])\n        array([ True, False, False])\n        "
        raise NotImplementedError

    def unique(self):
        if False:
            while True:
                i = 10
        '\n        Return unique values in the index.\n\n        Returns\n        -------\n        Index without duplicates\n        '
        raise NotImplementedError

    def to_series(self, index=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a Series with both index and values equal to the index keys.\n        Useful with map for returning an indexer based on an index.\n\n        Parameters\n        ----------\n        index : Index, optional\n            Index of resulting Series. If None, defaults to original index.\n        name : str, optional\n            Name of resulting Series. If None, defaults to name of original\n            index.\n\n        Returns\n        -------\n        Series\n            The dtype will be based on the type of the Index values.\n        '
        return cudf.Series._from_data(self._data, index=self.copy(deep=False) if index is None else index, name=self.name if name is None else name)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        if False:
            i = 10
            return i + 15
        '{docstring}'
        return cudf.io.dlpack.to_dlpack(self)

    def append(self, other):
        if False:
            i = 10
            return i + 15
        "\n        Append a collection of Index objects together.\n\n        Parameters\n        ----------\n        other : Index or list/tuple of indices\n\n        Returns\n        -------\n        appended : Index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([1, 2, 10, 100])\n        >>> idx\n        Int64Index([1, 2, 10, 100], dtype='int64')\n        >>> other = cudf.Index([200, 400, 50])\n        >>> other\n        Int64Index([200, 400, 50], dtype='int64')\n        >>> idx.append(other)\n        Int64Index([1, 2, 10, 100, 200, 400, 50], dtype='int64')\n\n        append accepts list of Index objects\n\n        >>> idx.append([other, other])\n        Int64Index([1, 2, 10, 100, 200, 400, 50, 200, 400, 50], dtype='int64')\n        "
        raise NotImplementedError

    def difference(self, other, sort=None):
        if False:
            i = 10
            return i + 15
        "\n        Return a new Index with elements from the index that are not in\n        `other`.\n\n        This is the set difference of two Index objects.\n\n        Parameters\n        ----------\n        other : Index or array-like\n        sort : False or None, default None\n            Whether to sort the resulting index. By default, the\n            values are attempted to be sorted, but any TypeError from\n            incomparable elements is caught by cudf.\n\n            * None : Attempt to sort the result, but catch any TypeErrors\n              from comparing incomparable elements.\n            * False : Do not sort the result.\n\n        Returns\n        -------\n        difference : Index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx1 = cudf.Index([2, 1, 3, 4])\n        >>> idx1\n        Int64Index([2, 1, 3, 4], dtype='int64')\n        >>> idx2 = cudf.Index([3, 4, 5, 6])\n        >>> idx2\n        Int64Index([3, 4, 5, 6], dtype='int64')\n        >>> idx1.difference(idx2)\n        Int64Index([1, 2], dtype='int64')\n        >>> idx1.difference(idx2, sort=False)\n        Int64Index([2, 1], dtype='int64')\n        "
        if not can_convert_to_column(other):
            raise TypeError('Input must be Index or array-like')
        if sort not in {None, False}:
            raise ValueError(f"The 'sort' keyword only takes the values of None or False; {sort} was passed.")
        other = cudf.Index(other, name=getattr(other, 'name', self.name))
        if not len(other):
            return self._get_reconciled_name_object(other)
        elif self.equals(other):
            return self[:0]._get_reconciled_name_object(other)
        res_name = _get_result_name(self.name, other.name)
        if is_mixed_with_object_dtype(self, other):
            difference = self.copy()
        else:
            other = other.copy(deep=False)
            difference = cudf.core.index._index_from_data(cudf.DataFrame._from_data({'None': self._column}).merge(cudf.DataFrame._from_data({'None': other._column}), how='leftanti', on='None')._data)
            if self.dtype != other.dtype:
                difference = difference.astype(self.dtype)
        difference.name = res_name
        if sort is None and len(other):
            return difference.sort_values()
        return difference

    def is_numeric(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the Index only consists of numeric data.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_any_real_numeric_dtype` instead.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of numeric data.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_numeric()\n        True\n        >>> idx = cudf.Index([1, 2, 3, 4.0])\n        >>> idx.is_numeric()\n        True\n        >>> idx = cudf.Index([1, 2, 3, 4])\n        >>> idx.is_numeric()\n        True\n        >>> idx = cudf.Index([1, 2, 3, 4.0, np.nan])\n        >>> idx.is_numeric()\n        True\n        >>> idx = cudf.Index(["Apple", "cold"])\n        >>> idx.is_numeric()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_numeric is deprecated. Use cudf.api.types.is_any_real_numeric_dtype instead', FutureWarning)
        return self._is_numeric()

    def _is_numeric(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def is_boolean(self):
        if False:
            print('Hello World!')
        '\n        Check if the Index only consists of booleans.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_bool_dtype` instead.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of booleans.\n\n        See Also\n        --------\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([True, False, True])\n        >>> idx.is_boolean()\n        True\n        >>> idx = cudf.Index(["True", "False", "True"])\n        >>> idx.is_boolean()\n        False\n        >>> idx = cudf.Index([1, 2, 3])\n        >>> idx.is_boolean()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_boolean is deprecated. Use cudf.api.types.is_bool_dtype instead', FutureWarning)
        return self._is_boolean()

    def _is_boolean(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def is_integer(self):
        if False:
            i = 10
            return i + 15
        '\n        Check if the Index only consists of integers.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_integer_dtype` instead.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of integers.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([1, 2, 3, 4])\n        >>> idx.is_integer()\n        True\n        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_integer()\n        False\n        >>> idx = cudf.Index(["Apple", "Mango", "Watermelon"])\n        >>> idx.is_integer()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_integer is deprecated. Use cudf.api.types.is_integer_dtype instead', FutureWarning)
        return self._is_integer()

    def _is_integer(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def is_floating(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the Index is a floating type.\n\n        The Index may consist of only floats, NaNs, or a mix of floats,\n        integers, or NaNs.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_float_dtype` instead.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index only consists of only consists\n            of floats, NaNs, or a mix of floats, integers, or NaNs.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_floating()\n        True\n        >>> idx = cudf.Index([1.0, 2.0, np.nan, 4.0])\n        >>> idx.is_floating()\n        True\n        >>> idx = cudf.Index([1, 2, 3, 4, np.nan], nan_as_null=False)\n        >>> idx.is_floating()\n        True\n        >>> idx = cudf.Index([1, 2, 3, 4])\n        >>> idx.is_floating()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_floating is deprecated. Use cudf.api.types.is_float_dtype instead', FutureWarning)
        return self._is_floating()

    def _is_floating(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def is_object(self):
        if False:
            while True:
                i = 10
        '\n        Check if the Index is of the object dtype.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_object_dtype` instead.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index is of the object dtype.\n\n        See Also\n        --------\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_categorical : Check if the Index holds categorical data.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index(["Apple", "Mango", "Watermelon"])\n        >>> idx.is_object()\n        True\n        >>> idx = cudf.Index(["Watermelon", "Orange", "Apple",\n        ...                 "Watermelon"]).astype("category")\n        >>> idx.is_object()\n        False\n        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])\n        >>> idx.is_object()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_object is deprecated. Use cudf.api.types.is_object_dtype instead', FutureWarning)
        return self._is_object()

    def _is_object(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def is_categorical(self):
        if False:
            while True:
                i = 10
        '\n        Check if the Index holds categorical data.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_categorical_dtype` instead.\n\n        Returns\n        -------\n        bool\n            True if the Index is categorical.\n\n        See Also\n        --------\n        CategoricalIndex : Index for categorical data.\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_interval : Check if the Index holds Interval objects.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index(["Watermelon", "Orange", "Apple",\n        ...                 "Watermelon"]).astype("category")\n        >>> idx.is_categorical()\n        True\n        >>> idx = cudf.Index([1, 3, 5, 7])\n        >>> idx.is_categorical()\n        False\n        >>> s = cudf.Series(["Peter", "Victor", "Elisabeth", "Mar"])\n        >>> s\n        0        Peter\n        1       Victor\n        2    Elisabeth\n        3          Mar\n        dtype: object\n        >>> s.index.is_categorical()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_categorical is deprecated. Use cudf.api.types.is_categorical_dtype instead', FutureWarning)
        return self._is_categorical()

    def _is_categorical(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def is_interval(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the Index holds Interval objects.\n\n        .. deprecated:: 23.04\n           Use `cudf.api.types.is_interval_dtype` instead.\n\n        Returns\n        -------\n        bool\n            Whether or not the Index holds Interval objects.\n\n        See Also\n        --------\n        IntervalIndex : Index for Interval objects.\n        is_boolean : Check if the Index only consists of booleans.\n        is_integer : Check if the Index only consists of integers.\n        is_floating : Check if the Index is a floating type.\n        is_numeric : Check if the Index only consists of numeric data.\n        is_object : Check if the Index is of the object dtype.\n        is_categorical : Check if the Index holds categorical data.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import pandas as pd\n        >>> idx = cudf.from_pandas(\n        ...     pd.Index([pd.Interval(left=0, right=5),\n        ...               pd.Interval(left=5, right=10)])\n        ... )\n        >>> idx.is_interval()\n        True\n        >>> idx = cudf.Index([1, 3, 5, 7])\n        >>> idx.is_interval()\n        False\n        '
        warnings.warn(f'{type(self).__name__}.is_interval is deprecated. Use cudf.api.types.is_interval_dtype instead', FutureWarning)
        return self._is_interval()

    def _is_interval(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _union(self, other, sort=None):
        if False:
            i = 10
            return i + 15
        self_df = self.to_frame(index=False, name=0)
        other_df = other.to_frame(index=False, name=0)
        self_df['order'] = self_df.index
        other_df['order'] = other_df.index
        res = self_df.merge(other_df, on=[0], how='outer')
        res = res.sort_values(by=res._data.to_pandas_index()[1:], ignore_index=True)
        union_result = cudf.core.index._index_from_data({0: res._data[0]})
        if sort is None and len(other):
            return union_result.sort_values()
        return union_result

    def _intersection(self, other, sort=None):
        if False:
            for i in range(10):
                print('nop')
        intersection_result = cudf.core.index._index_from_data(cudf.DataFrame._from_data({'None': self.unique()._column}).merge(cudf.DataFrame._from_data({'None': other.unique()._column}), how='inner', on='None')._data)
        if sort is None and len(other):
            return intersection_result.sort_values()
        return intersection_result

    def sort_values(self, return_indexer=False, ascending=True, na_position='last', key=None):
        if False:
            while True:
                i = 10
        '\n        Return a sorted copy of the index, and optionally return the indices\n        that sorted the index itself.\n\n        Parameters\n        ----------\n        return_indexer : bool, default False\n            Should the indices that would sort the index be returned.\n        ascending : bool, default True\n            Should the index values be sorted in an ascending order.\n        na_position : {\'first\' or \'last\'}, default \'last\'\n            Argument \'first\' puts NaNs at the beginning, \'last\' puts NaNs at\n            the end.\n        key : None, optional\n            This parameter is NON-FUNCTIONAL.\n\n        Returns\n        -------\n        sorted_index : Index\n            Sorted copy of the index.\n        indexer : cupy.ndarray, optional\n            The indices that the index itself was sorted by.\n\n        See Also\n        --------\n        cudf.Series.min : Sort values of a Series.\n        cudf.DataFrame.sort_values : Sort values in a DataFrame.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> idx = cudf.Index([10, 100, 1, 1000])\n        >>> idx\n        Int64Index([10, 100, 1, 1000], dtype=\'int64\')\n\n        Sort values in ascending order (default behavior).\n\n        >>> idx.sort_values()\n        Int64Index([1, 10, 100, 1000], dtype=\'int64\')\n\n        Sort values in descending order, and also get the indices `idx` was\n        sorted by.\n\n        >>> idx.sort_values(ascending=False, return_indexer=True)\n        (Int64Index([1000, 100, 10, 1], dtype=\'int64\'), array([3, 1, 0, 2],\n                                                            dtype=int32))\n\n        Sorting values in a MultiIndex:\n\n        >>> midx = cudf.MultiIndex(\n        ...      levels=[[1, 3, 4, -10], [1, 11, 5]],\n        ...      codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],\n        ...      names=["x", "y"],\n        ... )\n        >>> midx\n        MultiIndex([(  1,  1),\n                    (  1,  5),\n                    (  3, 11),\n                    (  4, 11),\n                    (-10,  1)],\n                   names=[\'x\', \'y\'])\n        >>> midx.sort_values()\n        MultiIndex([(-10,  1),\n                    (  1,  1),\n                    (  1,  5),\n                    (  3, 11),\n                    (  4, 11)],\n                   names=[\'x\', \'y\'])\n        >>> midx.sort_values(ascending=False)\n        MultiIndex([(  4, 11),\n                    (  3, 11),\n                    (  1,  5),\n                    (  1,  1),\n                    (-10,  1)],\n                   names=[\'x\', \'y\'])\n        '
        if key is not None:
            raise NotImplementedError('key parameter is not yet implemented.')
        if na_position not in {'first', 'last'}:
            raise ValueError(f'invalid na_position: {na_position}')
        indices = self.argsort(ascending=ascending, na_position=na_position)
        index_sorted = self.take(indices)
        if return_indexer:
            return (index_sorted, indices)
        else:
            return index_sorted

    def join(self, other, how='left', level=None, return_indexers=False, sort=False):
        if False:
            return 10
        '\n        Compute join_index and indexers to conform data structures\n        to the new index.\n\n        Parameters\n        ----------\n        other : Index.\n        how : {\'left\', \'right\', \'inner\', \'outer\'}\n        return_indexers : bool, default False\n        sort : bool, default False\n            Sort the join keys lexicographically in the result Index. If False,\n            the order of the join keys depends on the join type (how keyword).\n\n        Returns: index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> lhs = cudf.DataFrame({\n        ...     "a": [2, 3, 1],\n        ...     "b": [3, 4, 2],\n        ... }).set_index([\'a\', \'b\']).index\n        >>> lhs\n        MultiIndex([(2, 3),\n                    (3, 4),\n                    (1, 2)],\n                   names=[\'a\', \'b\'])\n        >>> rhs = cudf.DataFrame({"a": [1, 4, 3]}).set_index(\'a\').index\n        >>> rhs\n        Int64Index([1, 4, 3], dtype=\'int64\', name=\'a\')\n        >>> lhs.join(rhs, how=\'inner\')\n        MultiIndex([(3, 4),\n                    (1, 2)],\n                   names=[\'a\', \'b\'])\n        '
        if return_indexers is not False:
            raise NotImplementedError('return_indexers is not implemented')
        self_is_multi = isinstance(self, cudf.MultiIndex)
        other_is_multi = isinstance(other, cudf.MultiIndex)
        if level is not None:
            if self_is_multi and other_is_multi:
                raise TypeError('Join on level between two MultiIndex objects is ambiguous')
            if not is_scalar(level):
                raise ValueError('level should be an int or a label only')
        if other_is_multi:
            if how == 'left':
                how = 'right'
            elif how == 'right':
                how = 'left'
            rhs = self.copy(deep=False)
            lhs = other.copy(deep=False)
        else:
            lhs = self.copy(deep=False)
            rhs = other.copy(deep=False)
        same_names = lhs.names == rhs.names
        if isinstance(lhs, cudf.MultiIndex):
            on = lhs._data.select_by_index(level).names[0] if isinstance(level, int) else level
            if on is not None:
                rhs.names = (on,)
            on = rhs.names[0]
            if how == 'outer':
                how = 'left'
            elif how == 'right':
                how = 'inner'
        else:
            on = lhs.names[0]
            rhs.names = lhs.names
        lhs = lhs.to_frame()
        rhs = rhs.to_frame()
        output = lhs.merge(rhs, how=how, on=on, sort=sort)
        if self_is_multi and other_is_multi:
            return cudf.MultiIndex._from_data(output._data)
        else:
            idx = cudf.core.index._index_from_data(output._data)
            idx.name = self.name if same_names else None
            return idx

    def rename(self, name, inplace=False):
        if False:
            while True:
                i = 10
        "\n        Alter Index name.\n\n        Defaults to returning new index.\n\n        Parameters\n        ----------\n        name : label\n            Name(s) to set.\n\n        Returns\n        -------\n        Index\n\n        Examples\n        --------\n        >>> import cudf\n        >>> index = cudf.Index([1, 2, 3], name='one')\n        >>> index\n        Int64Index([1, 2, 3], dtype='int64', name='one')\n        >>> index.name\n        'one'\n        >>> renamed_index = index.rename('two')\n        >>> renamed_index\n        Int64Index([1, 2, 3], dtype='int64', name='two')\n        >>> renamed_index.name\n        'two'\n        "
        if inplace is True:
            self.name = name
            return None
        else:
            out = self.copy(deep=True)
            out.name = name
            return out

    def _indices_of(self, value) -> cudf.core.column.NumericalColumn:
        if False:
            return 10
        '\n        Return indices corresponding to value\n\n        Parameters\n        ----------\n        value\n            Value to look for in index\n\n        Returns\n        -------\n        Column of indices\n        '
        raise NotImplementedError

    def find_label_range(self, loc: slice) -> slice:
        if False:
            print('Hello World!')
        '\n        Translate a label-based slice to an index-based slice\n\n        Parameters\n        ----------\n        loc\n            slice to search for.\n\n        Notes\n        -----\n        As with all label-based searches, the slice is right-closed.\n\n        Returns\n        -------\n        New slice translated into integer indices of the index (right-open).\n        '
        start = loc.start
        stop = loc.stop
        step = 1 if loc.step is None else loc.step
        if step < 0:
            (start_side, stop_side) = ('right', 'left')
        else:
            (start_side, stop_side) = ('left', 'right')
        istart = None if start is None else self.get_slice_bound(start, side=start_side)
        istop = None if stop is None else self.get_slice_bound(stop, side=stop_side)
        if step < 0:
            istart = None if istart is None else max(istart - 1, 0)
            istop = None if istop is None or istop == 0 else istop - 1
        return slice(istart, istop, step)

    def searchsorted(self, value, side: builtins.str='left', ascending: bool=True, na_position: builtins.str='last'):
        if False:
            return 10
        "Find index where elements should be inserted to maintain order\n\n        Parameters\n        ----------\n        value :\n            Value to be hypothetically inserted into Self\n        side : str {'left', 'right'} optional, default 'left'\n            If 'left', the index of the first suitable location found is given\n            If 'right', return the last such index\n        ascending : bool optional, default True\n            Index is in ascending order (otherwise descending)\n        na_position : str {'last', 'first'} optional, default 'last'\n            Position of null values in sorted order\n\n        Returns\n        -------\n        Insertion point.\n\n        Notes\n        -----\n        As a precondition the index must be sorted in the same order\n        as requested by the `ascending` flag.\n        "
        raise NotImplementedError

    def get_slice_bound(self, label, side: builtins.str, kind=None) -> int:
        if False:
            print('Hello World!')
        "\n        Calculate slice bound that corresponds to given label.\n        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position\n        of given label.\n\n        Parameters\n        ----------\n        label : object\n        side : {'left', 'right'}\n        kind : {'ix', 'loc', 'getitem'}\n\n        Returns\n        -------\n        int\n            Index of label.\n        "
        if kind is not None:
            warnings.warn("'kind' argument in get_slice_bound is deprecated and will be removed in a future version.", FutureWarning)
        if side not in {'left', 'right'}:
            raise ValueError(f'Invalid side argument {side}')
        if self.is_monotonic_increasing or self.is_monotonic_decreasing:
            return self.searchsorted(label, side=side, ascending=self.is_monotonic_increasing)
        else:
            try:
                (left, right) = self._values._find_first_and_last(label)
            except ValueError:
                raise KeyError(f'label={label!r} not in index')
            if left != right:
                raise KeyError(f'Cannot get slice bound for non-unique label label={label!r}')
            if side == 'left':
                return left
            else:
                return right + 1

    def __array_function__(self, func, types, args, kwargs):
        if False:
            while True:
                i = 10
        cudf_index_module = type(self)
        for submodule in func.__module__.split('.')[1:]:
            if hasattr(cudf_index_module, submodule):
                cudf_index_module = getattr(cudf_index_module, submodule)
            else:
                return NotImplemented
        fname = func.__name__
        handled_types = [BaseIndex, cudf.Series]
        for t in types:
            if not any((issubclass(t, handled_type) for handled_type in handled_types)):
                return NotImplemented
        if hasattr(cudf_index_module, fname):
            cudf_func = getattr(cudf_index_module, fname)
            if cudf_func is func:
                return NotImplemented
            else:
                result = cudf_func(*args, **kwargs)
                if fname == 'unique':
                    result = result.sort_values()
                return result
        else:
            return NotImplemented

    @classmethod
    def from_pandas(cls, index, nan_as_null=no_default):
        if False:
            i = 10
            return i + 15
        "\n        Convert from a Pandas Index.\n\n        Parameters\n        ----------\n        index : Pandas Index object\n            A Pandas Index object which has to be converted\n            to cuDF Index.\n        nan_as_null : bool, Default None\n            If ``None``/``True``, converts ``np.nan`` values\n            to ``null`` values.\n            If ``False``, leaves ``np.nan`` values as is.\n\n        Raises\n        ------\n        TypeError for invalid input type.\n\n        Examples\n        --------\n        >>> import cudf\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>> data = [10, 20, 30, np.nan]\n        >>> pdi = pd.Index(data)\n        >>> cudf.Index.from_pandas(pdi)\n        Float64Index([10.0, 20.0, 30.0, <NA>], dtype='float64')\n        >>> cudf.Index.from_pandas(pdi, nan_as_null=False)\n        Float64Index([10.0, 20.0, 30.0, nan], dtype='float64')\n        "
        if nan_as_null is no_default:
            nan_as_null = False if cudf.get_option('mode.pandas_compatible') else None
        if not isinstance(index, pd.Index):
            raise TypeError('not a pandas.Index')
        ind = cudf.Index(column.as_column(index, nan_as_null=nan_as_null))
        ind.name = index.name
        return ind

    @property
    def _constructor_expanddim(self):
        if False:
            i = 10
            return i + 15
        return cudf.MultiIndex

    def drop_duplicates(self, keep='first', nulls_are_equal=True):
        if False:
            i = 10
            return i + 15
        '\n        Drop duplicate rows in index.\n\n        keep : {"first", "last", False}, default "first"\n            - \'first\' : Drop duplicates except for the first occurrence.\n            - \'last\' : Drop duplicates except for the last occurrence.\n            - ``False`` : Drop all duplicates.\n        nulls_are_equal: bool, default True\n            Null elements are considered equal to other null elements.\n        '
        return self._from_columns_like_self(drop_duplicates(list(self._columns), keys=range(len(self._data)), keep=keep, nulls_are_equal=nulls_are_equal), self._column_names)

    def duplicated(self, keep='first'):
        if False:
            print('Hello World!')
        "\n        Indicate duplicate index values.\n\n        Duplicated values are indicated as ``True`` values in the resulting\n        array. Either all duplicates, all except the first, or all except the\n        last occurrence of duplicates can be indicated.\n\n        Parameters\n        ----------\n        keep : {'first', 'last', False}, default 'first'\n            The value or values in a set of duplicates to mark as missing.\n\n            - ``'first'`` : Mark duplicates as ``True`` except for the first\n              occurrence.\n            - ``'last'`` : Mark duplicates as ``True`` except for the last\n              occurrence.\n            - ``False`` : Mark all duplicates as ``True``.\n\n        Returns\n        -------\n        cupy.ndarray[bool]\n\n        See Also\n        --------\n        Series.duplicated : Equivalent method on cudf.Series.\n        DataFrame.duplicated : Equivalent method on cudf.DataFrame.\n        Index.drop_duplicates : Remove duplicate values from Index.\n\n        Examples\n        --------\n        By default, for each set of duplicated values, the first occurrence is\n        set to False and all others to True:\n\n        >>> import cudf\n        >>> idx = cudf.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])\n        >>> idx.duplicated()\n        array([False, False,  True, False,  True])\n\n        which is equivalent to\n\n        >>> idx.duplicated(keep='first')\n        array([False, False,  True, False,  True])\n\n        By using 'last', the last occurrence of each set of duplicated values\n        is set to False and all others to True:\n\n        >>> idx.duplicated(keep='last')\n        array([ True, False,  True, False, False])\n\n        By setting keep to ``False``, all duplicates are True:\n\n        >>> idx.duplicated(keep=False)\n        array([ True, False,  True, False,  True])\n        "
        return self.to_series().duplicated(keep=keep).to_cupy()

    def dropna(self, how='any'):
        if False:
            return 10
        '\n        Drop null rows from Index.\n\n        how : {"any", "all"}, default "any"\n            Specifies how to decide whether to drop a row.\n            "any" (default) drops rows containing at least\n            one null value. "all" drops only rows containing\n            *all* null values.\n        '
        data_columns = [col.nans_to_nulls() if isinstance(col, cudf.core.column.NumericalColumn) else col for col in self._columns]
        return self._from_columns_like_self(drop_nulls(data_columns, how=how, keys=range(len(data_columns))), self._column_names)

    def _gather(self, gather_map, nullify=False, check_bounds=True):
        if False:
            for i in range(10):
                print('nop')
        'Gather rows of index specified by indices in `gather_map`.\n\n        Skip bounds checking if check_bounds is False.\n        Set rows to null for all out of bound indices if nullify is `True`.\n        '
        gather_map = cudf.core.column.as_column(gather_map)
        if not is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype(size_type_dtype)
        if not _gather_map_is_valid(gather_map, len(self), check_bounds, nullify):
            raise IndexError('Gather map index is out of bounds.')
        return self._from_columns_like_self(gather(list(self._columns), gather_map, nullify=nullify), self._column_names)

    def take(self, indices, axis=0, allow_fill=True, fill_value=None):
        if False:
            for i in range(10):
                print('nop')
        "Return a new index containing the rows specified by *indices*\n\n        Parameters\n        ----------\n        indices : array-like\n            Array of ints indicating which positions to take.\n        axis : int\n            The axis over which to select values, always 0.\n        allow_fill : Unsupported\n        fill_value : Unsupported\n\n        Returns\n        -------\n        out : Index\n            New object with desired subset of rows.\n\n        Examples\n        --------\n        >>> idx = cudf.Index(['a', 'b', 'c', 'd', 'e'])\n        >>> idx.take([2, 0, 4, 3])\n        StringIndex(['c' 'a' 'e' 'd'], dtype='object')\n        "
        if axis not in {0, 'index'}:
            raise NotImplementedError('Gather along column axis is not yet supported.')
        if not allow_fill or fill_value is not None:
            raise NotImplementedError('`allow_fill` and `fill_value` are unsupported.')
        return self._gather(indices)

    def _apply_boolean_mask(self, boolean_mask):
        if False:
            while True:
                i = 10
        'Apply boolean mask to each row of `self`.\n\n        Rows corresponding to `False` is dropped.\n        '
        boolean_mask = cudf.core.column.as_column(boolean_mask)
        if not is_bool_dtype(boolean_mask.dtype):
            raise ValueError('boolean_mask is not boolean type.')
        return self._from_columns_like_self(apply_boolean_mask(list(self._columns), boolean_mask), column_names=self._column_names)

    def repeat(self, repeats, axis=None):
        if False:
            return 10
        "Repeat elements of a Index.\n\n        Returns a new Index where each element of the current Index is repeated\n        consecutively a given number of times.\n\n        Parameters\n        ----------\n        repeats : int, or array of ints\n            The number of repetitions for each element. This should\n            be a non-negative integer. Repeating 0 times will return\n            an empty object.\n\n        Returns\n        -------\n        Index\n            A newly created object of same type as caller with repeated\n            elements.\n\n        Examples\n        --------\n        >>> index = cudf.Index([10, 22, 33, 55])\n        >>> index\n        Int64Index([10, 22, 33, 55], dtype='int64')\n        >>> index.repeat(5)\n        Int64Index([10, 10, 10, 10, 10, 22, 22, 22, 22, 22, 33,\n                    33, 33, 33, 33, 55, 55, 55, 55, 55],\n                dtype='int64')\n        "
        raise NotImplementedError

    def _split_columns_by_levels(self, levels):
        if False:
            return 10
        if isinstance(levels, int) and levels > 0:
            raise ValueError(f'Out of bound level: {levels}')
        return ([self._data[self.name]], [], ['index' if self.name is None else self.name], [])

    def _split(self, splits):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

def _get_result_name(left_name, right_name):
    if False:
        i = 10
        return i + 15
    return left_name if _is_same_name(left_name, right_name) else None