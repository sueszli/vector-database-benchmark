"""Accessors for arrow-backed data."""
from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from pandas.compat import pa_version_under10p1, pa_version_under11p0
if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc
    from pandas.core.dtypes.dtypes import ArrowDtype
if TYPE_CHECKING:
    from collections.abc import Iterator
    from pandas import DataFrame, Series

class ArrowAccessor(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, data, validation_msg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._data = data
        self._validation_msg = validation_msg
        self._validate(data)

    @abstractmethod
    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass

    def _validate(self, data):
        if False:
            for i in range(10):
                print('nop')
        dtype = data.dtype
        if not isinstance(dtype, ArrowDtype):
            raise AttributeError(self._validation_msg.format(dtype=dtype))
        if not self._is_valid_pyarrow_dtype(dtype.pyarrow_dtype):
            raise AttributeError(self._validation_msg.format(dtype=dtype))

    @property
    def _pa_array(self):
        if False:
            while True:
                i = 10
        return self._data.array._pa_array

class ListAccessor(ArrowAccessor):
    """
    Accessor object for list data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow list data.
    """

    def __init__(self, data=None) -> None:
        if False:
            return 10
        super().__init__(data, validation_msg="Can only use the '.list' accessor with 'list[pyarrow]' dtype, not {dtype}.")

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        if False:
            while True:
                i = 10
        return pa.types.is_list(pyarrow_dtype) or pa.types.is_fixed_size_list(pyarrow_dtype) or pa.types.is_large_list(pyarrow_dtype)

    def len(self) -> Series:
        if False:
            while True:
                i = 10
        '\n        Return the length of each list in the Series.\n\n        Returns\n        -------\n        pandas.Series\n            The length of each list.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> s = pd.Series(\n        ...     [\n        ...         [1, 2, 3],\n        ...         [3],\n        ...     ],\n        ...     dtype=pd.ArrowDtype(pa.list_(\n        ...         pa.int64()\n        ...     ))\n        ... )\n        >>> s.list.len()\n        0    3\n        1    1\n        dtype: int32[pyarrow]\n        '
        from pandas import Series
        value_lengths = pc.list_value_length(self._pa_array)
        return Series(value_lengths, dtype=ArrowDtype(value_lengths.type))

    def __getitem__(self, key: int | slice) -> Series:
        if False:
            print('Hello World!')
        '\n        Index or slice lists in the Series.\n\n        Parameters\n        ----------\n        key : int | slice\n            Index or slice of indices to access from each list.\n\n        Returns\n        -------\n        pandas.Series\n            The list at requested index.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> s = pd.Series(\n        ...     [\n        ...         [1, 2, 3],\n        ...         [3],\n        ...     ],\n        ...     dtype=pd.ArrowDtype(pa.list_(\n        ...         pa.int64()\n        ...     ))\n        ... )\n        >>> s.list[0]\n        0    1\n        1    3\n        dtype: int64[pyarrow]\n        '
        from pandas import Series
        if isinstance(key, int):
            element = pc.list_element(self._pa_array, key)
            return Series(element, dtype=ArrowDtype(element.type))
        elif isinstance(key, slice):
            if pa_version_under11p0:
                raise NotImplementedError(f'List slice not supported by pyarrow {pa.__version__}.')
            (start, stop, step) = (key.start, key.stop, key.step)
            if start is None:
                start = 0
            if step is None:
                step = 1
            sliced = pc.list_slice(self._pa_array, start, stop, step)
            return Series(sliced, dtype=ArrowDtype(sliced.type))
        else:
            raise ValueError(f'key must be an int or slice, got {type(key).__name__}')

    def __iter__(self) -> Iterator:
        if False:
            print('Hello World!')
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def flatten(self) -> Series:
        if False:
            i = 10
            return i + 15
        '\n        Flatten list values.\n\n        Returns\n        -------\n        pandas.Series\n            The data from all lists in the series flattened.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> s = pd.Series(\n        ...     [\n        ...         [1, 2, 3],\n        ...         [3],\n        ...     ],\n        ...     dtype=pd.ArrowDtype(pa.list_(\n        ...         pa.int64()\n        ...     ))\n        ... )\n        >>> s.list.flatten()\n        0    1\n        1    2\n        2    3\n        3    3\n        dtype: int64[pyarrow]\n        '
        from pandas import Series
        flattened = pc.list_flatten(self._pa_array)
        return Series(flattened, dtype=ArrowDtype(flattened.type))

class StructAccessor(ArrowAccessor):
    """
    Accessor object for structured data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow struct data.
    """

    def __init__(self, data=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(data, validation_msg="Can only use the '.struct' accessor with 'struct[pyarrow]' dtype, not {dtype}.")

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        if False:
            return 10
        return pa.types.is_struct(pyarrow_dtype)

    @property
    def dtypes(self) -> Series:
        if False:
            print('Hello World!')
        '\n        Return the dtype object of each child field of the struct.\n\n        Returns\n        -------\n        pandas.Series\n            The data type of each child field.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> s = pd.Series(\n        ...     [\n        ...         {"version": 1, "project": "pandas"},\n        ...         {"version": 2, "project": "pandas"},\n        ...         {"version": 1, "project": "numpy"},\n        ...     ],\n        ...     dtype=pd.ArrowDtype(pa.struct(\n        ...         [("version", pa.int64()), ("project", pa.string())]\n        ...     ))\n        ... )\n        >>> s.struct.dtypes\n        version     int64[pyarrow]\n        project    string[pyarrow]\n        dtype: object\n        '
        from pandas import Index, Series
        pa_type = self._data.dtype.pyarrow_dtype
        types = [ArrowDtype(struct.type) for struct in pa_type]
        names = [struct.name for struct in pa_type]
        return Series(types, index=Index(names))

    def field(self, name_or_index: str | int) -> Series:
        if False:
            while True:
                i = 10
        '\n        Extract a child field of a struct as a Series.\n\n        Parameters\n        ----------\n        name_or_index : str | int\n            Name or index of the child field to extract.\n\n        Returns\n        -------\n        pandas.Series\n            The data corresponding to the selected child field.\n\n        See Also\n        --------\n        Series.struct.explode : Return all child fields as a DataFrame.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> s = pd.Series(\n        ...     [\n        ...         {"version": 1, "project": "pandas"},\n        ...         {"version": 2, "project": "pandas"},\n        ...         {"version": 1, "project": "numpy"},\n        ...     ],\n        ...     dtype=pd.ArrowDtype(pa.struct(\n        ...         [("version", pa.int64()), ("project", pa.string())]\n        ...     ))\n        ... )\n\n        Extract by field name.\n\n        >>> s.struct.field("project")\n        0    pandas\n        1    pandas\n        2     numpy\n        Name: project, dtype: string[pyarrow]\n\n        Extract by field index.\n\n        >>> s.struct.field(0)\n        0    1\n        1    2\n        2    1\n        Name: version, dtype: int64[pyarrow]\n        '
        from pandas import Series
        pa_arr = self._data.array._pa_array
        if isinstance(name_or_index, int):
            index = name_or_index
        elif isinstance(name_or_index, str):
            index = pa_arr.type.get_field_index(name_or_index)
        else:
            raise ValueError(f'name_or_index must be an int or str, got {type(name_or_index).__name__}')
        pa_field = pa_arr.type[index]
        field_arr = pc.struct_field(pa_arr, [index])
        return Series(field_arr, dtype=ArrowDtype(field_arr.type), index=self._data.index, name=pa_field.name)

    def explode(self) -> DataFrame:
        if False:
            return 10
        '\n        Extract all child fields of a struct as a DataFrame.\n\n        Returns\n        -------\n        pandas.DataFrame\n            The data corresponding to all child fields.\n\n        See Also\n        --------\n        Series.struct.field : Return a single child field as a Series.\n\n        Examples\n        --------\n        >>> import pyarrow as pa\n        >>> s = pd.Series(\n        ...     [\n        ...         {"version": 1, "project": "pandas"},\n        ...         {"version": 2, "project": "pandas"},\n        ...         {"version": 1, "project": "numpy"},\n        ...     ],\n        ...     dtype=pd.ArrowDtype(pa.struct(\n        ...         [("version", pa.int64()), ("project", pa.string())]\n        ...     ))\n        ... )\n\n        >>> s.struct.explode()\n           version project\n        0        1  pandas\n        1        2  pandas\n        2        1   numpy\n        '
        from pandas import concat
        pa_type = self._pa_array.type
        return concat([self.field(i) for i in range(pa_type.num_fields)], axis='columns')