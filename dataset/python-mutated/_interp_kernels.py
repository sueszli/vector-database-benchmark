import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
math_constants_preamble = '\n// workaround for HIP: line begins with #include\n#include <cupy/math_constants.h>\n'
spline_weights_inline = _spline_kernel_weights.spline_weights_inline

def _get_coord_map(ndim, nprepad=0):
    if False:
        return 10
    'Extract target coordinate from coords array (for map_coordinates).\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        coords (ndarray): array of shape (ncoords, ndim) containing the target\n            coordinates.\n        c_j: variables to hold the target coordinates\n\n    computes::\n\n        c_j = coords[i + j * ncoords];\n\n    ncoords is determined by the size of the output array, y.\n    y will be indexed by the CIndexer, _ind.\n    Thus ncoords = _ind.size();\n\n    '
    ops = []
    ops.append('ptrdiff_t ncoords = _ind.size();')
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = coords[i + {j} * ncoords]{pre};')
    return ops

def _get_coord_zoom_and_shift(ndim, nprepad=0):
    if False:
        return 10
    'Compute target coordinate based on a shift followed by a zoom.\n\n    This version zooms from the center of the edge pixels.\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        in_coord[ndim]: array containing the source coordinate\n        zoom[ndim]: array containing the zoom for each axis\n        shift[ndim]: array containing the zoom for each axis\n\n    computes::\n\n        c_j = zoom[j] * (in_coord[j] - shift[j])\n\n    '
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[{j}]){pre};')
    return ops

def _get_coord_zoom_and_shift_grid(ndim, nprepad=0):
    if False:
        i = 10
        return i + 15
    'Compute target coordinate based on a shift followed by a zoom.\n\n    This version zooms from the outer edges of the grid pixels.\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        in_coord[ndim]: array containing the source coordinate\n        zoom[ndim]: array containing the zoom for each axis\n        shift[ndim]: array containing the zoom for each axis\n\n    computes::\n\n        c_j = zoom[j] * (in_coord[j] - shift[j] + 0.5) - 0.5\n\n    '
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[j] + 0.5) - 0.5{pre};')
    return ops

def _get_coord_zoom(ndim, nprepad=0):
    if False:
        for i in range(10):
            print('nop')
    'Compute target coordinate based on a zoom.\n\n    This version zooms from the center of the edge pixels.\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        in_coord[ndim]: array containing the source coordinate\n        zoom[ndim]: array containing the zoom for each axis\n\n    computes::\n\n        c_j = zoom[j] * in_coord[j]\n\n    '
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = zoom[{j}] * (W)in_coord[{j}]{pre};')
    return ops

def _get_coord_zoom_grid(ndim, nprepad=0):
    if False:
        print('Hello World!')
    'Compute target coordinate based on a zoom (grid_mode=True version).\n\n    This version zooms from the outer edges of the grid pixels.\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        in_coord[ndim]: array containing the source coordinate\n        zoom[ndim]: array containing the zoom for each axis\n\n    computes::\n\n        c_j = zoom[j] * (in_coord[j] + 0.5) - 0.5\n\n    '
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] + 0.5) - 0.5{pre};')
    return ops

def _get_coord_shift(ndim, nprepad=0):
    if False:
        for i in range(10):
            print('nop')
    'Compute target coordinate based on a shift.\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        in_coord[ndim]: array containing the source coordinate\n        shift[ndim]: array containing the zoom for each axis\n\n    computes::\n\n        c_j = in_coord[j] - shift[j]\n\n    '
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = (W)in_coord[{j}] - shift[{j}]{pre};')
    return ops

def _get_coord_affine(ndim, nprepad=0):
    if False:
        for i in range(10):
            print('nop')
    'Compute target coordinate based on a homogeneous transformation matrix.\n\n    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to\n    affine matrix where the last row of the affine is assumed to be:\n    ``[0] * ndim + [1]``.\n\n    Notes\n    -----\n    Assumes the following variables have been initialized on the device::\n\n        mat(array): array containing the (ndim, ndim + 1) transform matrix.\n        in_coords(array): coordinates of the input\n\n    For example, in 2D:\n\n        c_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];\n        c_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];\n\n    '
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    ncol = ndim + 1
    for j in range(ndim):
        ops.append(f'\n            W c_{j} = (W)0.0;')
        for k in range(ndim):
            ops.append(f'\n            c_{j} += mat[{ncol * j + k}] * (W)in_coord[{k}];')
        ops.append(f'\n            c_{j} += mat[{ncol * j + ndim}]{pre};')
    return ops

def _unravel_loop_index(shape, uint_t='unsigned int'):
    if False:
        i = 10
        return i + 15
    '\n    declare a multi-index array in_coord and unravel the 1D index, i into it.\n    This code assumes that the array is a C-ordered array.\n    '
    ndim = len(shape)
    code = [f'\n        {uint_t} in_coord[{ndim}];\n        {uint_t} s, t, idx = i;']
    for j in range(ndim - 1, 0, -1):
        code.append(f'\n        s = {shape[j]};\n        t = idx / s;\n        in_coord[{j}] = idx - t * s;\n        idx = t;')
    code.append('\n        in_coord[0] = idx;')
    return '\n'.join(code)

def _generate_interp_custom(coord_func, ndim, large_int, yshape, mode, cval, order, name='', integer_output=False, nprepad=0, omit_in_coord=False):
    if False:
        print('Hello World!')
    "\n    Args:\n        coord_func (function): generates code to do the coordinate\n            transformation. See for example, `_get_coord_shift`.\n        ndim (int): The number of dimensions.\n        large_int (bool): If true use Py_ssize_t instead of int for indexing.\n        yshape (tuple): Shape of the output array.\n        mode (str): Signal extension mode to use at the array boundaries\n        cval (float): constant value used when `mode == 'constant'`.\n        name (str): base name for the interpolation kernel\n        integer_output (bool): boolean indicating whether the output has an\n            integer type.\n        nprepad (int): integer indicating the amount of prepadding at the\n            boundaries.\n\n    Returns:\n        operation (str): code body for the ElementwiseKernel\n        name (str): name for the ElementwiseKernel\n    "
    ops = []
    internal_dtype = 'double' if integer_output else 'Y'
    ops.append(f'{internal_dtype} out = 0.0;')
    if large_int:
        uint_t = 'size_t'
        int_t = 'ptrdiff_t'
    else:
        uint_t = 'unsigned int'
        int_t = 'int'
    for j in range(ndim):
        ops.append(f'const {int_t} xsize_{j} = x.shape()[{j}];')
    ops.append(f'const {uint_t} sx_{ndim - 1} = 1;')
    for j in range(ndim - 1, 0, -1):
        ops.append(f'const {uint_t} sx_{j - 1} = sx_{j} * xsize_{j};')
    if not omit_in_coord:
        ops.append(_unravel_loop_index(yshape, uint_t))
    ops = ops + coord_func(ndim, nprepad)
    if cval is numpy.nan:
        cval = '(Y)CUDART_NAN'
    elif cval == numpy.inf:
        cval = '(Y)CUDART_INF'
    elif cval == -numpy.inf:
        cval = '(Y)(-CUDART_INF)'
    else:
        cval = f'({internal_dtype}){cval}'
    if mode == 'constant':
        _cond = ' || '.join([f'(c_{j} < 0) || (c_{j} > xsize_{j} - 1)' for j in range(ndim)])
        ops.append(f'\n        if ({_cond})\n        {{\n            out = {cval};\n        }}\n        else\n        {{')
    if order == 0:
        if mode == 'wrap':
            ops.append('double dcoord;')
        for j in range(ndim):
            if mode == 'wrap':
                ops.append(f'\n                dcoord = c_{j};')
            else:
                ops.append(f'\n                {int_t} cf_{j} = ({int_t})floor((double)c_{j} + 0.5);')
            if mode != 'constant':
                if mode == 'wrap':
                    ixvar = 'dcoord'
                    float_ix = True
                else:
                    ixvar = f'cf_{j}'
                    float_ix = False
                ops.append(_util._generate_boundary_condition_ops(mode, ixvar, f'xsize_{j}', int_t, float_ix))
                if mode == 'wrap':
                    ops.append(f'\n                {int_t} cf_{j} = ({int_t})floor(dcoord + 0.5);')
            ops.append(f'\n            {int_t} ic_{j} = cf_{j} * sx_{j};')
        _coord_idx = ' + '.join([f'ic_{j}' for j in range(ndim)])
        if mode == 'grid-constant':
            _cond = ' || '.join([f'(ic_{j} < 0)' for j in range(ndim)])
            ops.append(f'\n            if ({_cond}) {{\n                out = {cval};\n            }} else {{\n                out = ({internal_dtype})x[{_coord_idx}];\n            }}')
        else:
            ops.append(f'\n            out = ({internal_dtype})x[{_coord_idx}];')
    elif order == 1:
        for j in range(ndim):
            ops.append(f'\n            {int_t} cf_{j} = ({int_t})floor((double)c_{j});\n            {int_t} cc_{j} = cf_{j} + 1;\n            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed\n            ')
            if mode == 'wrap':
                ops.append(f'\n                double dcoordf = c_{j};\n                double dcoordc = c_{j} + 1;')
            else:
                ops.append(f'\n                {int_t} cf_bounded_{j} = cf_{j};\n                {int_t} cc_bounded_{j} = cc_{j};')
            if mode != 'constant':
                if mode == 'wrap':
                    ixvar = 'dcoordf'
                    float_ix = True
                else:
                    ixvar = f'cf_bounded_{j}'
                    float_ix = False
                ops.append(_util._generate_boundary_condition_ops(mode, ixvar, f'xsize_{j}', int_t, float_ix))
                ixvar = 'dcoordc' if mode == 'wrap' else f'cc_bounded_{j}'
                ops.append(_util._generate_boundary_condition_ops(mode, ixvar, f'xsize_{j}', int_t, float_ix))
                if mode == 'wrap':
                    ops.append(f'\n                    {int_t} cf_bounded_{j} = ({int_t})floor(dcoordf);;\n                    {int_t} cc_bounded_{j} = ({int_t})floor(dcoordf + 1);;\n                    ')
            ops.append(f'\n            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)\n                {{\n                    W w_{j};\n                    {int_t} ic_{j};\n                    if (s_{j} == 0)\n                    {{\n                        w_{j} = (W)cc_{j} - c_{j};\n                        ic_{j} = cf_bounded_{j} * sx_{j};\n                    }} else\n                    {{\n                        w_{j} = c_{j} - (W)cf_{j};\n                        ic_{j} = cc_bounded_{j} * sx_{j};\n                    }}')
    elif order > 1:
        if mode == 'grid-constant':
            spline_mode = 'constant'
        elif mode == 'nearest':
            spline_mode = 'nearest'
        else:
            spline_mode = _spline_prefilter_core._get_spline_mode(mode)
        ops.append(f'\n            W wx, wy;\n            {int_t} start;')
        for j in range(ndim):
            ops.append(f'\n            W weights_{j}[{order + 1}];')
            ops.append(spline_weights_inline[order].format(j=j, order=order))
            if mode in ['wrap']:
                ops.append(f'double dcoord = c_{j};')
                coord_var = 'dcoord'
                ops.append(_util._generate_boundary_condition_ops(mode, coord_var, f'xsize_{j}', int_t, True))
            else:
                coord_var = f'(double)c_{j}'
            if order & 1:
                op_str = '\n                start = ({int_t})floor({coord_var}) - {order_2};'
            else:
                op_str = '\n                start = ({int_t})floor({coord_var} + 0.5) - {order_2};'
            ops.append(op_str.format(int_t=int_t, coord_var=coord_var, order_2=order // 2))
            ops.append(f'{int_t} ci_{j}[{order + 1}];')
            for k in range(order + 1):
                ixvar = f'ci_{j}[{k}]'
                ops.append(f'\n                {ixvar} = start + {k};')
                ops.append(_util._generate_boundary_condition_ops(spline_mode, ixvar, f'xsize_{j}', int_t))
            ops.append(f'\n            W w_{j};\n            {int_t} ic_{j};\n            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)\n                {{\n                    w_{j} = weights_{j}[k_{j}];\n                    ic_{j} = ci_{j}[k_{j}] * sx_{j};\n            ')
    if order > 0:
        _weight = ' * '.join([f'w_{j}' for j in range(ndim)])
        _coord_idx = ' + '.join([f'ic_{j}' for j in range(ndim)])
        if mode == 'grid-constant' or (order > 1 and mode == 'constant'):
            _cond = ' || '.join([f'(ic_{j} < 0)' for j in range(ndim)])
            ops.append(f'\n            if ({_cond}) {{\n                out += {cval} * ({internal_dtype})({_weight});\n            }} else {{\n                {internal_dtype} val = ({internal_dtype})x[{_coord_idx}];\n                out += val * ({internal_dtype})({_weight});\n            }}')
        else:
            ops.append(f'\n            {internal_dtype} val = ({internal_dtype})x[{_coord_idx}];\n            out += val * ({internal_dtype})({_weight});')
        ops.append('}' * ndim)
    if mode == 'constant':
        ops.append('}')
    if integer_output:
        ops.append('y = (Y)rint((double)out);')
    else:
        ops.append('y = (Y)out;')
    operation = '\n'.join(ops)
    mode_str = mode.replace('-', '_')
    name = 'cupyx_scipy_ndimage_interpolate_{}_order{}_{}_{}d_y{}'.format(name, order, mode_str, ndim, '_'.join([f'{j}' for j in yshape]))
    if uint_t == 'size_t':
        name += '_i64'
    return (operation, name)

@cupy._util.memoize(for_each_device=True)
def _get_map_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, nprepad=0):
    if False:
        for i in range(10):
            print('nop')
    in_params = 'raw X x, raw W coords'
    out_params = 'Y y'
    (operation, name) = _generate_interp_custom(coord_func=_get_coord_map, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='map', integer_output=integer_output, nprepad=nprepad, omit_in_coord=True)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)

@cupy._util.memoize(for_each_device=True)
def _get_shift_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, nprepad=0):
    if False:
        for i in range(10):
            print('nop')
    in_params = 'raw X x, raw W shift'
    out_params = 'Y y'
    (operation, name) = _generate_interp_custom(coord_func=_get_coord_shift, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='shift', integer_output=integer_output, nprepad=nprepad)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)

@cupy._util.memoize(for_each_device=True)
def _get_zoom_shift_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, grid_mode=False, nprepad=0):
    if False:
        for i in range(10):
            print('nop')
    in_params = 'raw X x, raw W shift, raw W zoom'
    out_params = 'Y y'
    if grid_mode:
        zoom_shift_func = _get_coord_zoom_and_shift_grid
    else:
        zoom_shift_func = _get_coord_zoom_and_shift
    (operation, name) = _generate_interp_custom(coord_func=zoom_shift_func, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='zoom_shift_grid' if grid_mode else 'zoom_shift', integer_output=integer_output, nprepad=nprepad)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)

@cupy._util.memoize(for_each_device=True)
def _get_zoom_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, grid_mode=False, nprepad=0):
    if False:
        print('Hello World!')
    in_params = 'raw X x, raw W zoom'
    out_params = 'Y y'
    (operation, name) = _generate_interp_custom(coord_func=_get_coord_zoom_grid if grid_mode else _get_coord_zoom, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='zoom_grid' if grid_mode else 'zoom', integer_output=integer_output, nprepad=nprepad)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)

@cupy._util.memoize(for_each_device=True)
def _get_affine_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, nprepad=0):
    if False:
        i = 10
        return i + 15
    in_params = 'raw X x, raw W mat'
    out_params = 'Y y'
    (operation, name) = _generate_interp_custom(coord_func=_get_coord_affine, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='affine', integer_output=integer_output, nprepad=nprepad)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)