"""
Interaction with scipy.sparse matrices.

Currently only includes to_coo helpers.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from pandas._libs import lib
from pandas.core.dtypes.missing import notna
from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series
if TYPE_CHECKING:
    from collections.abc import Iterable
    import numpy as np
    import scipy.sparse
    from pandas._typing import IndexLabel, npt

def _check_is_partition(parts: Iterable, whole: Iterable):
    if False:
        i = 10
        return i + 15
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise ValueError('Is not a partition because intersection is not null.')
    if set.union(*parts) != whole:
        raise ValueError('Is not a partition because union is not the whole.')

def _levels_to_axis(ss, levels: tuple[int] | list[int], valid_ilocs: npt.NDArray[np.intp], sort_labels: bool=False) -> tuple[npt.NDArray[np.intp], list[IndexLabel]]:
    if False:
        print('Hello World!')
    "\n    For a MultiIndexed sparse Series `ss`, return `ax_coords` and `ax_labels`,\n    where `ax_coords` are the coordinates along one of the two axes of the\n    destination sparse matrix, and `ax_labels` are the labels from `ss`' Index\n    which correspond to these coordinates.\n\n    Parameters\n    ----------\n    ss : Series\n    levels : tuple/list\n    valid_ilocs : numpy.ndarray\n        Array of integer positions of valid values for the sparse matrix in ss.\n    sort_labels : bool, default False\n        Sort the axis labels before forming the sparse matrix. When `levels`\n        refers to a single level, set to True for a faster execution.\n\n    Returns\n    -------\n    ax_coords : numpy.ndarray (axis coordinates)\n    ax_labels : list (axis labels)\n    "
    if sort_labels and len(levels) == 1:
        ax_coords = ss.index.codes[levels[0]][valid_ilocs]
        ax_labels = ss.index.levels[levels[0]]
    else:
        levels_values = lib.fast_zip([ss.index.get_level_values(lvl).to_numpy() for lvl in levels])
        (codes, ax_labels) = factorize(levels_values, sort=sort_labels)
        ax_coords = codes[valid_ilocs]
    ax_labels = ax_labels.tolist()
    return (ax_coords, ax_labels)

def _to_ijv(ss, row_levels: tuple[int] | list[int]=(0,), column_levels: tuple[int] | list[int]=(1,), sort_labels: bool=False) -> tuple[np.ndarray, npt.NDArray[np.intp], npt.NDArray[np.intp], list[IndexLabel], list[IndexLabel]]:
    if False:
        return 10
    '\n    For an arbitrary MultiIndexed sparse Series return (v, i, j, ilabels,\n    jlabels) where (v, (i, j)) is suitable for passing to scipy.sparse.coo\n    constructor, and ilabels and jlabels are the row and column labels\n    respectively.\n\n    Parameters\n    ----------\n    ss : Series\n    row_levels : tuple/list\n    column_levels : tuple/list\n    sort_labels : bool, default False\n        Sort the row and column labels before forming the sparse matrix.\n        When `row_levels` and/or `column_levels` refer to a single level,\n        set to `True` for a faster execution.\n\n    Returns\n    -------\n    values : numpy.ndarray\n        Valid values to populate a sparse matrix, extracted from\n        ss.\n    i_coords : numpy.ndarray (row coordinates of the values)\n    j_coords : numpy.ndarray (column coordinates of the values)\n    i_labels : list (row labels)\n    j_labels : list (column labels)\n    '
    _check_is_partition([row_levels, column_levels], range(ss.index.nlevels))
    sp_vals = ss.array.sp_values
    na_mask = notna(sp_vals)
    values = sp_vals[na_mask]
    valid_ilocs = ss.array.sp_index.indices[na_mask]
    (i_coords, i_labels) = _levels_to_axis(ss, row_levels, valid_ilocs, sort_labels=sort_labels)
    (j_coords, j_labels) = _levels_to_axis(ss, column_levels, valid_ilocs, sort_labels=sort_labels)
    return (values, i_coords, j_coords, i_labels, j_labels)

def sparse_series_to_coo(ss: Series, row_levels: Iterable[int]=(0,), column_levels: Iterable[int]=(1,), sort_labels: bool=False) -> tuple[scipy.sparse.coo_matrix, list[IndexLabel], list[IndexLabel]]:
    if False:
        while True:
            i = 10
    '\n    Convert a sparse Series to a scipy.sparse.coo_matrix using index\n    levels row_levels, column_levels as the row and column\n    labels respectively. Returns the sparse_matrix, row and column labels.\n    '
    import scipy.sparse
    if ss.index.nlevels < 2:
        raise ValueError('to_coo requires MultiIndex with nlevels >= 2.')
    if not ss.index.is_unique:
        raise ValueError('Duplicate index entries are not allowed in to_coo transformation.')
    row_levels = [ss.index._get_level_number(x) for x in row_levels]
    column_levels = [ss.index._get_level_number(x) for x in column_levels]
    (v, i, j, rows, columns) = _to_ijv(ss, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels)
    sparse_matrix = scipy.sparse.coo_matrix((v, (i, j)), shape=(len(rows), len(columns)))
    return (sparse_matrix, rows, columns)

def coo_to_sparse_series(A: scipy.sparse.coo_matrix, dense_index: bool=False) -> Series:
    if False:
        i = 10
        return i + 15
    '\n    Convert a scipy.sparse.coo_matrix to a Series with type sparse.\n\n    Parameters\n    ----------\n    A : scipy.sparse.coo_matrix\n    dense_index : bool, default False\n\n    Returns\n    -------\n    Series\n\n    Raises\n    ------\n    TypeError if A is not a coo_matrix\n    '
    from pandas import SparseDtype
    try:
        ser = Series(A.data, MultiIndex.from_arrays((A.row, A.col)), copy=False)
    except AttributeError as err:
        raise TypeError(f'Expected coo_matrix. Got {type(A).__name__} instead.') from err
    ser = ser.sort_index()
    ser = ser.astype(SparseDtype(ser.dtype))
    if dense_index:
        ind = MultiIndex.from_product([A.row, A.col])
        ser = ser.reindex(ind)
    return ser