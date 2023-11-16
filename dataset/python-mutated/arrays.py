from __future__ import annotations
import operator
from functools import partial
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy
import ibis.expr.operations as ops
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.dispatch import execute_node
if TYPE_CHECKING:
    from collections.abc import Collection

@execute_node.register(ops.ArrayColumn, tuple)
def execute_array_column(op, cols, **kwargs):
    if False:
        i = 10
        return i + 15
    cols = [execute(arg, **kwargs) for arg in cols]
    df = pd.concat(cols, axis=1)
    return df.apply(lambda row: np.array(row, dtype=object), axis=1)

@execute_node.register(ops.ArrayLength, pd.Series)
def execute_array_length(op, data, **kwargs):
    if False:
        i = 10
        return i + 15
    return data.apply(len)

@execute_node.register(ops.ArrayLength, np.ndarray)
def execute_array_length_scalar(op, data, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return len(data)

@execute_node.register(ops.ArraySlice, pd.Series, int, (int, type(None)))
def execute_array_slice(op, data, start, stop, **kwargs):
    if False:
        while True:
            i = 10
    return data.apply(operator.itemgetter(slice(start, stop)))

@execute_node.register(ops.ArraySlice, np.ndarray, int, (int, type(None)))
def execute_array_slice_scalar(op, data, start, stop, **kwargs):
    if False:
        i = 10
        return i + 15
    return data[start:stop]

@execute_node.register(ops.ArrayIndex, pd.Series, int)
def execute_array_index(op, data, index, **kwargs):
    if False:
        while True:
            i = 10
    return data.apply(lambda array, index=index: array[index] if -len(array) <= index < len(array) else None)

@execute_node.register(ops.ArrayIndex, np.ndarray, int)
def execute_array_index_scalar(op, data, index, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    try:
        return data[index]
    except IndexError:
        return None

@execute_node.register(ops.ArrayContains, np.ndarray, object)
def execute_node_contains_value_array(op, haystack, needle, **kwargs):
    if False:
        while True:
            i = 10
    return needle in haystack

def _concat_iterables_to_series(*iters: Collection[Any]) -> pd.Series:
    if False:
        i = 10
        return i + 15
    'Concatenate two collections to create a Series.\n\n    The two collections are assumed to have the same length.\n\n    Used for ArrayConcat implementation.\n    '
    (first, *rest) = iters
    assert all((len(series) == len(first) for series in rest))
    return pd.Series(map(lambda *args: np.concatenate(args), first, *rest))

@execute_node.register(ops.ArrayConcat, tuple)
def execute_array_concat(op, args, **kwargs):
    if False:
        while True:
            i = 10
    return execute_node(op, *map(partial(execute, **kwargs), args), **kwargs)

@execute_node.register(ops.ArrayConcat, pd.Series, pd.Series, [pd.Series])
def execute_array_concat_series(op, first, second, *args, **kwargs):
    if False:
        return 10
    return _concat_iterables_to_series(first, second, *args)

@execute_node.register(ops.ArrayConcat, np.ndarray, pd.Series, [(pd.Series, np.ndarray)])
def execute_array_concat_mixed_left(op, left, right, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    left = np.tile(left, (len(right), 1))
    return _concat_iterables_to_series(left, right)

@execute_node.register(ops.ArrayConcat, pd.Series, np.ndarray, [(pd.Series, np.ndarray)])
def execute_array_concat_mixed_right(op, left, right, *args, **kwargs):
    if False:
        while True:
            i = 10
    right = np.tile(right, (len(left), 1))
    return _concat_iterables_to_series(left, right)

@execute_node.register(ops.ArrayConcat, np.ndarray, np.ndarray, [np.ndarray])
def execute_array_concat_scalar(op, left, right, *args, **kwargs):
    if False:
        print('Hello World!')
    return np.concatenate([left, right, *args])

@execute_node.register(ops.ArrayRepeat, pd.Series, int)
def execute_array_repeat(op, data, n, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    n = max(n, 0)
    return pd.Series((np.tile(arr, n) for arr in data))

@execute_node.register(ops.ArrayRepeat, np.ndarray, int)
def execute_array_repeat_scalar(op, data, n, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return np.tile(data, max(n, 0))

@execute_node.register(ops.ArrayCollect, pd.Series, (type(None), pd.Series))
def execute_array_collect(op, data, where, aggcontext=None, **kwargs):
    if False:
        while True:
            i = 10
    return aggcontext.agg(data.loc[where] if where is not None else data, np.array)

@execute_node.register(ops.ArrayCollect, SeriesGroupBy, (type(None), pd.Series))
def execute_array_collect_groupby(op, data, where, aggcontext=None, **kwargs):
    if False:
        print('Hello World!')
    return aggcontext.agg(data.obj.loc[where].groupby(data.grouping.grouper) if where is not None else data, np.array)

@execute_node.register(ops.Unnest, pd.Series)
def execute_unnest(op, data, **kwargs):
    if False:
        while True:
            i = 10
    return data.explode()