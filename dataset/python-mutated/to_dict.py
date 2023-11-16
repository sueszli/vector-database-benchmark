from __future__ import annotations
from typing import TYPE_CHECKING, Literal, overload
import warnings
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import maybe_box_native
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core import common as com
if TYPE_CHECKING:
    from pandas._typing import MutableMappingT
    from pandas import DataFrame

@overload
def to_dict(df: DataFrame, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'index']=..., *, into: type[MutableMappingT] | MutableMappingT, index: bool=...) -> MutableMappingT:
    if False:
        print('Hello World!')
    ...

@overload
def to_dict(df: DataFrame, orient: Literal['records'], *, into: type[MutableMappingT] | MutableMappingT, index: bool=...) -> list[MutableMappingT]:
    if False:
        return 10
    ...

@overload
def to_dict(df: DataFrame, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'index']=..., *, into: type[dict]=..., index: bool=...) -> dict:
    if False:
        return 10
    ...

@overload
def to_dict(df: DataFrame, orient: Literal['records'], *, into: type[dict]=..., index: bool=...) -> list[dict]:
    if False:
        print('Hello World!')
    ...

def to_dict(df: DataFrame, orient: Literal['dict', 'list', 'series', 'split', 'tight', 'records', 'index']='dict', *, into: type[MutableMappingT] | MutableMappingT=dict, index: bool=True) -> MutableMappingT | list[MutableMappingT]:
    if False:
        i = 10
        return i + 15
    "\n    Convert the DataFrame to a dictionary.\n\n    The type of the key-value pairs can be customized with the parameters\n    (see below).\n\n    Parameters\n    ----------\n    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}\n        Determines the type of the values of the dictionary.\n\n        - 'dict' (default) : dict like {column -> {index -> value}}\n        - 'list' : dict like {column -> [values]}\n        - 'series' : dict like {column -> Series(values)}\n        - 'split' : dict like\n          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}\n        - 'tight' : dict like\n          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],\n          'index_names' -> [index.names], 'column_names' -> [column.names]}\n        - 'records' : list like\n          [{column -> value}, ... , {column -> value}]\n        - 'index' : dict like {index -> {column -> value}}\n\n        .. versionadded:: 1.4.0\n            'tight' as an allowed value for the ``orient`` argument\n\n    into : class, default dict\n        The collections.abc.MutableMapping subclass used for all Mappings\n        in the return value.  Can be the actual class or an empty\n        instance of the mapping type you want.  If you want a\n        collections.defaultdict, you must pass it initialized.\n\n    index : bool, default True\n        Whether to include the index item (and index_names item if `orient`\n        is 'tight') in the returned dictionary. Can only be ``False``\n        when `orient` is 'split' or 'tight'.\n\n        .. versionadded:: 2.0.0\n\n    Returns\n    -------\n    dict, list or collections.abc.Mapping\n        Return a collections.abc.MutableMapping object representing the\n        DataFrame. The resulting transformation depends on the `orient` parameter.\n    "
    if not df.columns.is_unique:
        warnings.warn('DataFrame columns are not unique, some columns will be omitted.', UserWarning, stacklevel=find_stack_level())
    into_c = com.standardize_mapping(into)
    orient = orient.lower()
    if not index and orient not in ['split', 'tight']:
        raise ValueError("'index=False' is only valid when 'orient' is 'split' or 'tight'")
    if orient == 'series':
        return into_c(((k, v) for (k, v) in df.items()))
    box_native_indices = [i for (i, col_dtype) in enumerate(df.dtypes.values) if col_dtype == np.dtype(object) or isinstance(col_dtype, ExtensionDtype)]
    are_all_object_dtype_cols = len(box_native_indices) == len(df.dtypes)
    if orient == 'dict':
        return into_c(((k, v.to_dict(into=into)) for (k, v) in df.items()))
    elif orient == 'list':
        object_dtype_indices_as_set: set[int] = set(box_native_indices)
        return into_c(((k, list(map(maybe_box_native, v.to_numpy().tolist())) if i in object_dtype_indices_as_set else v.to_numpy().tolist()) for (i, (k, v)) in enumerate(df.items())))
    elif orient == 'split':
        data = df._create_data_for_split_and_tight_to_dict(are_all_object_dtype_cols, box_native_indices)
        return into_c(((('index', df.index.tolist()),) if index else ()) + (('columns', df.columns.tolist()), ('data', data)))
    elif orient == 'tight':
        data = df._create_data_for_split_and_tight_to_dict(are_all_object_dtype_cols, box_native_indices)
        return into_c(((('index', df.index.tolist()),) if index else ()) + (('columns', df.columns.tolist()), ('data', [list(map(maybe_box_native, t)) for t in df.itertuples(index=False, name=None)])) + ((('index_names', list(df.index.names)),) if index else ()) + (('column_names', list(df.columns.names)),))
    elif orient == 'records':
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            rows = (dict(zip(columns, row)) for row in df.itertuples(index=False, name=None))
            return [into_c(((k, maybe_box_native(v)) for (k, v) in row.items())) for row in rows]
        else:
            data = [into_c(zip(columns, t)) for t in df.itertuples(index=False, name=None)]
            if box_native_indices:
                object_dtype_indices_as_set = set(box_native_indices)
                object_dtype_cols = {col for (i, col) in enumerate(df.columns) if i in object_dtype_indices_as_set}
                for row in data:
                    for col in object_dtype_cols:
                        row[col] = maybe_box_native(row[col])
            return data
    elif orient == 'index':
        if not df.index.is_unique:
            raise ValueError("DataFrame index must be unique for orient='index'.")
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            return into_c(((t[0], dict(zip(df.columns, map(maybe_box_native, t[1:])))) for t in df.itertuples(name=None)))
        elif box_native_indices:
            object_dtype_indices_as_set = set(box_native_indices)
            is_object_dtype_by_index = [i in object_dtype_indices_as_set for i in range(len(df.columns))]
            return into_c(((t[0], {columns[i]: maybe_box_native(v) if is_object_dtype_by_index[i] else v for (i, v) in enumerate(t[1:])}) for t in df.itertuples(name=None)))
        else:
            return into_c(((t[0], dict(zip(df.columns, t[1:]))) for t in df.itertuples(name=None)))
    else:
        raise ValueError(f"orient '{orient}' not understood")