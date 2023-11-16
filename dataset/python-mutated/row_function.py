import math
import numpy as np
from numba import cuda
from numba.np import numpy_support
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import string_view
from cudf.core.udf.templates import masked_input_initializer_template, row_initializer_template, row_kernel_template, unmasked_input_initializer_template
from cudf.core.udf.utils import Row, _all_dtypes_from_frame, _construct_signature, _get_extensionty_size, _get_kernel, _get_udf_return_type, _mask_get, _supported_cols_from_frame, _supported_dtypes_from_frame

def _get_frame_row_type(dtype):
    if False:
        return 10
    '\n    Get the Numba type of a row in a frame. Models each column and its mask as\n    a MaskedType and models the row as a dictionary like data structure\n    containing these MaskedTypes. Large parts of this function are copied with\n    comments from the Numba internals and slightly modified to account for\n    validity bools to be present in the final struct. See\n    numba.np.numpy_support.from_struct_dtype for details.\n    '
    fields = []
    offset = 0
    sizes = [_get_extensionty_size(string_view) if val[0] == np.dtype('O') else val[0].itemsize for val in dtype.fields.values()]
    for (i, (name, info)) in enumerate(dtype.fields.items()):
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None
        ty = string_view if elemdtype == np.dtype('O') else numpy_support.from_dtype(elemdtype)
        infos = {'type': MaskedType(ty), 'offset': offset, 'title': title}
        fields.append((name, infos))
        itemsize = _get_extensionty_size(string_view) if elemdtype == np.dtype('O') else elemdtype.itemsize
        offset += itemsize + 1
        if i < len(sizes) - 1:
            next_itemsize = sizes[i + 1]
            offset = int(math.ceil(offset / next_itemsize) * next_itemsize)
    _is_aligned_struct = True
    return Row(fields, offset, _is_aligned_struct)

def _row_kernel_string_from_template(frame, row_type, args):
    if False:
        i = 10
        return i + 15
    "\n    Function to write numba kernels for `DataFrame.apply` as a string.\n    Workaround until numba supports functions that use `*args`\n\n    `DataFrame.apply` expects functions of a dict like row as well as\n    possibly one or more scalar arguments\n\n    def f(row, c, k):\n        return (row['x'] + c) / k\n\n    Both the number of input columns as well as their nullability and any\n    scalar arguments may vary, so the kernels vary significantly. See\n    templates.py for the full row kernel template and more details.\n    "
    frame = _supported_cols_from_frame(frame)
    input_columns = ', '.join([f'input_col_{i}' for i in range(len(frame))])
    input_offsets = ', '.join([f'offset_{i}' for i in range(len(frame))])
    extra_args = ', '.join([f'extra_arg_{i}' for i in range(len(args))])
    initializers = []
    row_initializers = []
    for (i, (colname, col)) in enumerate(frame.items()):
        idx = str(i)
        template = masked_input_initializer_template if col.mask is not None else unmasked_input_initializer_template
        initializers.append(template.format(idx=idx))
        row_initializers.append(row_initializer_template.format(idx=idx, name=colname))
    return row_kernel_template.format(input_columns=input_columns, input_offsets=input_offsets, extra_args=extra_args, masked_input_initializers='\n'.join(initializers), row_initializers='\n'.join(row_initializers), numba_rectype=row_type)

def _get_row_kernel(frame, func, args):
    if False:
        print('Hello World!')
    row_type = _get_frame_row_type(np.dtype(list(_all_dtypes_from_frame(frame).items())))
    scalar_return_type = _get_udf_return_type(row_type, func, args)
    sig = _construct_signature(frame, scalar_return_type, args)
    np_field_types = np.dtype(list(_supported_dtypes_from_frame(frame).items()))
    row_type = _get_frame_row_type(np_field_types)
    global_exec_context = {'cuda': cuda, 'Masked': Masked, '_mask_get': _mask_get, 'pack_return': pack_return, 'row_type': row_type}
    kernel_string = _row_kernel_string_from_template(frame, row_type, args)
    kernel = _get_kernel(kernel_string, global_exec_context, sig, func)
    return (kernel, scalar_return_type)