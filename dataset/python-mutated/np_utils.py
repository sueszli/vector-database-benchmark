"""
High-level operations for numpy structured arrays.

Some code and inspiration taken from numpy.lib.recfunctions.join_by().
Redistribution license restrictions apply.
"""
import collections
from collections import Counter, OrderedDict
from collections.abc import Sequence
import numpy as np
__all__ = ['TableMergeError']

class TableMergeError(ValueError):
    pass

def get_col_name_map(arrays, common_names, uniq_col_name='{col_name}_{table_name}', table_names=None):
    if False:
        print('Hello World!')
    '\n    Find the column names mapping when merging the list of structured ndarrays\n    ``arrays``.  It is assumed that col names in ``common_names`` are to be\n    merged into a single column while the rest will be uniquely represented\n    in the output.  The args ``uniq_col_name`` and ``table_names`` specify\n    how to rename columns in case of conflicts.\n\n    Returns a dict mapping each output column name to the input(s).  This takes the form\n    {outname : (col_name_0, col_name_1, ...), ... }.  For key columns all of input names\n    will be present, while for the other non-key columns the value will be (col_name_0,\n    None, ..) or (None, col_name_1, ..) etc.\n    '
    col_name_map = collections.defaultdict(lambda : [None] * len(arrays))
    col_name_list = []
    if table_names is None:
        table_names = [str(ii + 1) for ii in range(len(arrays))]
    for (idx, array) in enumerate(arrays):
        table_name = table_names[idx]
        for name in array.dtype.names:
            out_name = name
            if name in common_names:
                if name not in col_name_list:
                    col_name_list.append(name)
            else:
                others = list(arrays)
                others.pop(idx)
                if any((name in other.dtype.names for other in others)):
                    out_name = uniq_col_name.format(table_name=table_name, col_name=name)
                col_name_list.append(out_name)
            col_name_map[out_name][idx] = name
    col_name_count = Counter(col_name_list)
    repeated_names = [name for (name, count) in col_name_count.items() if count > 1]
    if repeated_names:
        raise TableMergeError(f'Merging column names resulted in duplicates: {repeated_names}.  Change uniq_col_name or table_names args to fix this.')
    col_name_map = OrderedDict(((name, col_name_map[name]) for name in col_name_list))
    return col_name_map

def get_descrs(arrays, col_name_map):
    if False:
        i = 10
        return i + 15
    "\n    Find the dtypes descrs resulting from merging the list of arrays' dtypes,\n    using the column name mapping ``col_name_map``.\n\n    Return a list of descrs for the output.\n    "
    out_descrs = []
    for (out_name, in_names) in col_name_map.items():
        in_cols = [arr[name] for (arr, name) in zip(arrays, in_names) if name is not None]
        names = [name for name in in_names if name is not None]
        try:
            dtype = common_dtype(in_cols)
        except TableMergeError as tme:
            raise TableMergeError("The '{}' columns have incompatible types: {}".format(names[0], tme._incompat_types)) from tme
        uniq_shapes = {col.shape[1:] for col in in_cols}
        if len(uniq_shapes) != 1:
            raise TableMergeError('Key columns have different shape')
        shape = uniq_shapes.pop()
        if out_name is not None:
            out_name = str(out_name)
        out_descrs.append((out_name, dtype, shape))
    return out_descrs

def common_dtype(cols):
    if False:
        print('Hello World!')
    '\n    Use numpy to find the common dtype for a list of structured ndarray columns.\n\n    Only allow columns within the following fundamental numpy data types:\n    np.bool_, np.object_, np.number, np.character, np.void\n    '
    np_types = (np.bool_, np.object_, np.number, np.character, np.void)
    uniq_types = {tuple((issubclass(col.dtype.type, np_type) for np_type in np_types)) for col in cols}
    if len(uniq_types) > 1:
        incompat_types = [col.dtype.name for col in cols]
        tme = TableMergeError(f'Columns have incompatible types {incompat_types}')
        tme._incompat_types = incompat_types
        raise tme
    arrs = [np.empty(1, dtype=col.dtype) for col in cols]
    for arr in arrs:
        if arr.dtype.kind in ('S', 'U'):
            arr[0] = '0' * arr.itemsize
    arr_common = np.array([arr[0] for arr in arrs])
    return arr_common.dtype.str

def _check_for_sequence_of_structured_arrays(arrays):
    if False:
        while True:
            i = 10
    err = '`arrays` arg must be a sequence (e.g. list) of structured arrays'
    if not isinstance(arrays, Sequence):
        raise TypeError(err)
    for array in arrays:
        if not isinstance(array, np.ndarray) or array.dtype.names is None:
            raise TypeError(err)
    if len(arrays) == 0:
        raise ValueError('`arrays` arg must include at least one array')