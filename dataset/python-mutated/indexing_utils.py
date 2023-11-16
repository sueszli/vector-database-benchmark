from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Union
from typing_extensions import TypeAlias
import cudf
from cudf.api.types import _is_scalar_or_zero_d_array, is_bool_dtype, is_integer, is_integer_dtype
from cudf.core.copy_types import BooleanMask, GatherMap

class EmptyIndexer:
    """An indexer that will produce an empty result."""
    pass

@dataclass
class MapIndexer:
    """An indexer for a gather map."""
    key: GatherMap

@dataclass
class MaskIndexer:
    """An indexer for a boolean mask."""
    key: BooleanMask

@dataclass
class SliceIndexer:
    """An indexer for a slice."""
    key: slice

@dataclass
class ScalarIndexer:
    """An indexer for a scalar value."""
    key: GatherMap
IndexingSpec: TypeAlias = Union[EmptyIndexer, MapIndexer, MaskIndexer, ScalarIndexer, SliceIndexer]
ColumnLabels: TypeAlias = List[str]

def destructure_iloc_key(key: Any, frame: Union[cudf.Series, cudf.DataFrame]) -> tuple[Any, ...]:
    if False:
        while True:
            i = 10
    '\n    Destructure a potentially tuple-typed key into row and column indexers.\n\n    Tuple arguments to iloc indexing are treated specially. They are\n    picked apart into indexers for the row and column. If the number\n    of entries is less than the number of modes of the frame, missing\n    entries are slice-expanded.\n\n    If the user-provided key is not a tuple, it is treated as if it\n    were a singleton tuple, and then slice-expanded.\n\n    Once this destructuring has occurred, any entries that are\n    callables are then called with the indexed frame. This should\n    return a valid indexing object for the rows (respectively\n    columns), namely one of:\n\n    - A boolean mask of the same length as the frame in the given\n      dimension\n    - A scalar integer that indexes the frame\n    - An array-like of integers that index the frame\n    - A slice that indexes the frame\n\n    Integer and slice-based indexing follows usual Python conventions.\n\n    Parameters\n    ----------\n    key\n        The key to destructure\n    frame\n        DataFrame or Series to provide context\n\n    Returns\n    -------\n    tuple\n        Indexers with length equal to the dimension of the frame\n\n    Raises\n    ------\n    IndexError\n        If there are too many indexers, or any individual indexer is a tuple.\n    '
    n = len(frame.shape)
    if isinstance(key, tuple):
        indexers = key + (slice(None),) * (n - len(key))
        if len(indexers) > n:
            raise IndexError(f'Too many indexers: got {len(indexers)} expected {n}')
    else:
        indexers = (key, *(slice(None),) * (n - 1))
    indexers = tuple((k(frame) if callable(k) else k for k in indexers))
    if any((isinstance(k, tuple) for k in indexers)):
        raise IndexError("Too many indexers: can't have nested tuples in iloc indexing")
    return indexers

def destructure_dataframe_iloc_indexer(key: Any, frame: cudf.DataFrame) -> Tuple[Any, Tuple[bool, ColumnLabels]]:
    if False:
        while True:
            i = 10
    'Destructure an index key for DataFrame iloc getitem.\n\n    Parameters\n    ----------\n    key\n        Key to destructure\n    frame\n        DataFrame to provide context context\n\n    Returns\n    -------\n    tuple\n        2-tuple of a key for the rows and tuple of\n        (column_index_is_scalar, column_names) for the columns\n\n    Raises\n    ------\n    TypeError\n        If the column indexer is invalid\n    IndexError\n        If the provided key does not destructure correctly\n    NotImplementedError\n        If the requested column indexer repeats columns\n    '
    (rows, cols) = destructure_iloc_key(key, frame)
    if cols is Ellipsis:
        cols = slice(None)
    scalar = is_integer(cols)
    try:
        column_names: ColumnLabels = list(frame._data.get_labels_by_index(cols))
        if len(set(column_names)) != len(column_names):
            raise NotImplementedError('cudf DataFrames do not support repeated column names')
    except TypeError:
        raise TypeError('Column indices must be integers, slices, or list-like of integers')
    if scalar:
        assert len(column_names) == 1, 'Scalar column indexer should not produce more than one column'
    return (rows, (scalar, column_names))

def destructure_series_iloc_indexer(key: Any, frame: cudf.Series) -> Any:
    if False:
        return 10
    'Destructure an index key for Series iloc getitem.\n\n    Parameters\n    ----------\n    key\n        Key to destructure\n    frame\n        Series for unpacking context\n\n    Returns\n    -------\n    Single key that will index the rows\n    '
    (rows,) = destructure_iloc_key(key, frame)
    return rows

def parse_row_iloc_indexer(key: Any, n: int) -> IndexingSpec:
    if False:
        for i in range(10):
            print('nop')
    '\n    Normalize and produce structured information about a row indexer.\n\n    Given a row indexer that has already been destructured by\n    :func:`destructure_iloc_key`, inspect further and produce structured\n    information for indexing operations to act upon.\n\n    Parameters\n    ----------\n    key\n        Suitably destructured key for row indexing\n    n\n        Length of frame to index\n\n    Returns\n    -------\n    IndexingSpec\n        Structured data for indexing. A tag + parsed data.\n\n    Raises\n    ------\n    IndexError\n        If a valid type of indexer is provided, but it is out of\n        bounds\n    TypeError\n        If the indexing key is otherwise invalid.\n    '
    if key is Ellipsis:
        return SliceIndexer(slice(None))
    elif isinstance(key, slice):
        return SliceIndexer(key)
    elif _is_scalar_or_zero_d_array(key):
        return ScalarIndexer(GatherMap(key, n, nullify=False))
    else:
        key = cudf.core.column.as_column(key)
        if isinstance(key, cudf.core.column.CategoricalColumn):
            key = key.as_numerical_column(key.codes.dtype)
        if is_bool_dtype(key.dtype):
            return MaskIndexer(BooleanMask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        elif is_integer_dtype(key.dtype):
            return MapIndexer(GatherMap(key, n, nullify=False))
        else:
            raise TypeError(f'Cannot index by location with non-integer key of type {type(key)}')