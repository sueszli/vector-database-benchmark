from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Tuple, cast
import numpy as np
import cudf
from cudf.api.types import is_decimal_dtype, is_dtype_equal
from cudf.core.column import CategoricalColumn
from cudf.core.dtypes import CategoricalDtype
if TYPE_CHECKING:
    from cudf.core.column import ColumnBase

class _Indexer:

    def __init__(self, name: Any):
        if False:
            while True:
                i = 10
        self.name = name

class _ColumnIndexer(_Indexer):

    def get(self, obj: cudf.DataFrame) -> ColumnBase:
        if False:
            while True:
                i = 10
        return obj._data[self.name]

    def set(self, obj: cudf.DataFrame, value: ColumnBase, validate=False):
        if False:
            return 10
        obj._data.set_by_label(self.name, value, validate=validate)

class _IndexIndexer(_Indexer):

    def get(self, obj: cudf.DataFrame) -> ColumnBase:
        if False:
            i = 10
            return i + 15
        return obj._index._data[self.name]

    def set(self, obj: cudf.DataFrame, value: ColumnBase, validate=False):
        if False:
            for i in range(10):
                print('nop')
        obj._index._data.set_by_label(self.name, value, validate=validate)

def _match_join_keys(lcol: ColumnBase, rcol: ColumnBase, how: str) -> Tuple[ColumnBase, ColumnBase]:
    if False:
        return 10
    common_type = None
    ltype = lcol.dtype
    rtype = rcol.dtype
    left_is_categorical = isinstance(ltype, CategoricalDtype)
    right_is_categorical = isinstance(rtype, CategoricalDtype)
    if left_is_categorical and right_is_categorical:
        return _match_categorical_dtypes_both(cast(CategoricalColumn, lcol), cast(CategoricalColumn, rcol), how)
    elif left_is_categorical or right_is_categorical:
        if left_is_categorical:
            if how in {'left', 'leftsemi', 'leftanti'}:
                return (lcol, rcol.astype(ltype))
            common_type = ltype.categories.dtype
        else:
            common_type = rtype.categories.dtype
        common_type = cudf.utils.dtypes._dtype_pandas_compatible(common_type)
        return (lcol.astype(common_type), rcol.astype(common_type))
    if is_dtype_equal(ltype, rtype):
        return (lcol, rcol)
    if is_decimal_dtype(ltype) or is_decimal_dtype(rtype):
        raise TypeError('Decimal columns can only be merged with decimal columns of the same precision and scale')
    if np.issubdtype(ltype, np.number) and np.issubdtype(rtype, np.number) and (not (np.issubdtype(ltype, np.timedelta64) or np.issubdtype(rtype, np.timedelta64))):
        common_type = max(ltype, rtype) if ltype.kind == rtype.kind else np.find_common_type([], (ltype, rtype))
    elif np.issubdtype(ltype, np.datetime64) and np.issubdtype(rtype, np.datetime64) or (np.issubdtype(ltype, np.timedelta64) and np.issubdtype(rtype, np.timedelta64)):
        common_type = max(ltype, rtype)
    elif (np.issubdtype(ltype, np.datetime64) or np.issubdtype(ltype, np.timedelta64)) and (not rcol.fillna(0).can_cast_safely(ltype)):
        raise TypeError(f'Cannot join between {ltype} and {rtype}, please type-cast both columns to the same type.')
    elif (np.issubdtype(rtype, np.datetime64) or np.issubdtype(rtype, np.timedelta64)) and (not lcol.fillna(0).can_cast_safely(rtype)):
        raise TypeError(f'Cannot join between {rtype} and {ltype}, please type-cast both columns to the same type.')
    if how == 'left' and rcol.fillna(0).can_cast_safely(ltype):
        return (lcol, rcol.astype(ltype))
    return (lcol.astype(common_type), rcol.astype(common_type))

def _match_categorical_dtypes_both(lcol: CategoricalColumn, rcol: CategoricalColumn, how: str) -> Tuple[ColumnBase, ColumnBase]:
    if False:
        i = 10
        return i + 15
    (ltype, rtype) = (lcol.dtype, rcol.dtype)
    if ltype == rtype:
        return (lcol, rcol)
    if ltype.ordered != rtype.ordered:
        raise TypeError('Merging on categorical variables with mismatched ordering is ambiguous')
    if ltype.ordered and rtype.ordered:
        raise TypeError(f'{how} merge between categoricals with different categories is only valid when neither side is ordered')
    if how == 'inner':
        return _match_join_keys(lcol._get_decategorized_column(), rcol._get_decategorized_column(), how)
    elif how in {'left', 'leftanti', 'leftsemi'}:
        return (lcol, rcol.astype(ltype))
    else:
        merged_categories = cudf.concat([ltype.categories, rtype.categories]).unique()
        common_type = cudf.CategoricalDtype(categories=merged_categories, ordered=False)
        return (lcol.astype(common_type), rcol.astype(common_type))

def _coerce_to_tuple(obj):
    if False:
        while True:
            i = 10
    if isinstance(obj, abc.Iterable) and (not isinstance(obj, str)):
        return tuple(obj)
    else:
        return (obj,)