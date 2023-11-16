"""
Implementation of nlargest and nsmallest.
"""
from __future__ import annotations
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, cast, final
import numpy as np
from pandas._libs import algos as libalgos
from pandas.core.dtypes.common import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_list_like, is_numeric_dtype, needs_i8_conversion
from pandas.core.dtypes.dtypes import BaseMaskedDtype
if TYPE_CHECKING:
    from pandas._typing import DtypeObj, IndexLabel
    from pandas import DataFrame, Series

class SelectN:

    def __init__(self, obj, n: int, keep: str) -> None:
        if False:
            while True:
                i = 10
        self.obj = obj
        self.n = n
        self.keep = keep
        if self.keep not in ('first', 'last', 'all'):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> DataFrame | Series:
        if False:
            print('Hello World!')
        raise NotImplementedError

    @final
    def nlargest(self):
        if False:
            for i in range(10):
                print('nop')
        return self.compute('nlargest')

    @final
    def nsmallest(self):
        if False:
            print('Hello World!')
        return self.compute('nsmallest')

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        if False:
            while True:
                i = 10
        '\n        Helper function to determine if dtype is valid for\n        nsmallest/nlargest methods\n        '
        if is_numeric_dtype(dtype):
            return not is_complex_dtype(dtype)
        return needs_i8_conversion(dtype)

class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series:
        if False:
            return 10
        from pandas.core.reshape.concat import concat
        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")
        if n <= 0:
            return self.obj[[]]
        dropped = self.obj.dropna()
        nan_index = self.obj.drop(dropped.index)
        if n >= len(self.obj):
            ascending = method == 'nsmallest'
            return self.obj.sort_values(ascending=ascending).head(n)
        new_dtype = dropped.dtype
        arr = dropped._values
        if needs_i8_conversion(arr.dtype):
            arr = arr.view('i8')
        elif isinstance(arr.dtype, BaseMaskedDtype):
            arr = arr._data
        else:
            arr = np.asarray(arr)
        if arr.dtype.kind == 'b':
            arr = arr.view(np.uint8)
        if method == 'nlargest':
            arr = -arr
            if is_integer_dtype(new_dtype):
                arr -= 1
            elif is_bool_dtype(new_dtype):
                arr = 1 - -arr
        if self.keep == 'last':
            arr = arr[::-1]
        nbase = n
        narr = len(arr)
        n = min(n, narr)
        kth_val = libalgos.kth_smallest(arr.copy(order='C'), n - 1)
        (ns,) = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind='mergesort')]
        if self.keep != 'all':
            inds = inds[:n]
            findex = nbase
        elif len(inds) < nbase <= len(nan_index) + len(inds):
            findex = len(nan_index) + len(inds)
        else:
            findex = len(inds)
        if self.keep == 'last':
            inds = narr - 1 - inds
        return concat([dropped.iloc[inds], nan_index]).iloc[:findex]

class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj: DataFrame, n: int, keep: str, columns: IndexLabel) -> None:
        if False:
            while True:
                i = 10
        super().__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]
        columns = cast(Sequence[Hashable], columns)
        columns = list(columns)
        self.columns = columns

    def compute(self, method: str) -> DataFrame:
        if False:
            while True:
                i = 10
        from pandas.core.api import Index
        n = self.n
        frame = self.obj
        columns = self.columns
        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(f'Column {repr(column)} has dtype {dtype}, cannot use method {repr(method)} with this dtype')

        def get_indexer(current_indexer, other_indexer):
            if False:
                return 10
            '\n            Helper function to concat `current_indexer` and `other_indexer`\n            depending on `method`\n            '
            if method == 'nsmallest':
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Index([], dtype=np.int64)
        for (i, column) in enumerate(columns):
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(cur_n, keep=self.keep if is_last_column else 'all')
            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break
            border_value = values == values[values.index[-1]]
            unsafe_values = values[border_value]
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)
        frame = frame.take(indexer)
        frame.index = original_index.take(indexer)
        if len(columns) == 1:
            return frame
        ascending = method == 'nsmallest'
        return frame.sort_values(columns, ascending=ascending, kind='mergesort')