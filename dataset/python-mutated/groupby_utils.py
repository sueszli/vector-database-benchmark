import cupy as cp
import numpy as np
from numba import cuda, types
from numba.core.errors import TypingError
from numba.cuda.cudadrv.devices import get_context
from numba.np import numpy_support
import cudf.core.udf.utils
from cudf.core.udf.groupby_typing import SUPPORTED_GROUPBY_NUMPY_TYPES, Group, GroupType
from cudf.core.udf.templates import group_initializer_template, groupby_apply_kernel_template
from cudf.core.udf.utils import Row, _compile_or_get, _get_extensionty_size, _get_kernel, _get_udf_return_type, _supported_cols_from_frame, _supported_dtypes_from_frame
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate

def _get_frame_groupby_type(dtype, index_dtype):
    if False:
        i = 10
        return i + 15
    '\n    Get the Numba type corresponding to a row of grouped data. Models the\n    column as a Record-like data structure containing GroupTypes. See\n    numba.np.numpy_support.from_struct_dtype for details.\n\n    Parameters\n    ----------\n    level : np.dtype\n        A numpy structured array dtype associating field names\n        to scalar dtypes\n    index_dtype : np.dtype\n        A numpy scalar dtype associated with the index of the\n        incoming grouped data\n    '
    fields = []
    offset = 0
    sizes = [val[0].itemsize for val in dtype.fields.values()]
    for (i, (name, info)) in enumerate(dtype.fields.items()):
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None
        ty = numpy_support.from_dtype(elemdtype)
        indexty = numpy_support.from_dtype(index_dtype)
        groupty = GroupType(ty, indexty)
        infos = {'type': groupty, 'offset': offset, 'title': title}
        fields.append((name, infos))
        offset += _get_extensionty_size(groupty)
        if i < len(sizes) - 1:
            alignment = offset % 8
            if alignment != 0:
                offset += 8 - alignment
    _is_aligned_struct = True
    return Row(fields, offset, _is_aligned_struct)

def _groupby_apply_kernel_string_from_template(frame, args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function to write numba kernels for `Groupby.apply` as a string.\n    Workaround until numba supports functions that use `*args`\n    '
    frame = _supported_cols_from_frame(frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES)
    input_columns = ', '.join([f'input_col_{i}' for i in range(len(frame))])
    extra_args = ', '.join([f'extra_arg_{i}' for i in range(len(args))])
    initializers = []
    for (i, colname) in enumerate(frame.keys()):
        initializers.append(group_initializer_template.format(idx=i, name=colname))
    return groupby_apply_kernel_template.format(input_columns=input_columns, extra_args=extra_args, group_initializers='\n'.join(initializers))

def _get_groupby_apply_kernel(frame, func, args):
    if False:
        for i in range(10):
            print('nop')
    np_field_types = np.dtype(list(_supported_dtypes_from_frame(frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES).items()))
    dataframe_group_type = _get_frame_groupby_type(np_field_types, frame.index.dtype)
    return_type = _get_udf_return_type(dataframe_group_type, func, args)
    global_exec_context = {'cuda': cuda, 'Group': Group, 'dataframe_group_type': dataframe_group_type, 'types': types}
    kernel_string = _groupby_apply_kernel_string_from_template(frame, args)
    kernel = _get_kernel(kernel_string, global_exec_context, None, func)
    return (kernel, return_type)

@_cudf_nvtx_annotate
def jit_groupby_apply(offsets, grouped_values, function, *args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Main entrypoint for JIT Groupby.apply via Numba.\n\n    Parameters\n    ----------\n    offsets : list\n        A list of integers denoting the indices of the group\n        boundaries in grouped_values\n    grouped_values : DataFrame\n        A DataFrame representing the source data\n        sorted by group keys\n    function : callable\n        The user-defined function to execute\n    '
    (kernel, return_type) = _compile_or_get(grouped_values, function, args, kernel_getter=_get_groupby_apply_kernel, suffix='__GROUPBY_APPLY_UDF')
    offsets = cp.asarray(offsets)
    ngroups = len(offsets) - 1
    output = cudf.core.column.column_empty(ngroups, dtype=return_type)
    launch_args = [offsets, output, grouped_values.index]
    launch_args += list(_supported_cols_from_frame(grouped_values, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES).values())
    launch_args += list(args)
    max_group_size = cp.diff(offsets).max()
    if max_group_size >= 256:
        blocklim = 256
    else:
        blocklim = (max_group_size + 32 - 1) // 32 * 32
    if kernel.specialized:
        specialized = kernel
    else:
        specialized = kernel.specialize(*launch_args)
    ctx = get_context()
    (kern_def,) = specialized.overloads.values()
    (grid, tpb) = ctx.get_max_potential_block_size(func=kern_def._codelibrary.get_cufunc(), b2d_func=0, memsize=0, blocksizelimit=int(blocklim))
    with _CUDFNumbaConfig():
        specialized[ngroups, tpb](*launch_args)
    return output

def _can_be_jitted(frame, func, args):
    if False:
        i = 10
        return i + 15
    '\n    Determine if this UDF is supported through the JIT engine\n    by attempting to compile just the function to PTX using the\n    target set of types\n    '
    if not hasattr(func, '__code__'):
        return False
    np_field_types = np.dtype(list(_supported_dtypes_from_frame(frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES).items()))
    dataframe_group_type = _get_frame_groupby_type(np_field_types, frame.index.dtype)
    try:
        _get_udf_return_type(dataframe_group_type, func, args)
        return True
    except TypingError:
        return False