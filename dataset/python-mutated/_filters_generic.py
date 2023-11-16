import cupy
from cupy_backends.cuda.api import runtime
from cupy import _util
from cupyx.scipy.ndimage import _filters_core

def _get_sub_kernel(f):
    if False:
        print('Hello World!')
    '\n    Takes the "function" given to generic_filter and returns the "sub-kernel"\n    that will be called, one of RawKernel or ReductionKernel.\n\n    This supports:\n     * cupy.RawKernel\n       no checks are possible\n     * cupy.ReductionKernel\n       checks that there is a single input and output\n    '
    if isinstance(f, cupy.RawKernel):
        return f
    elif isinstance(f, cupy.ReductionKernel):
        if f.nin != 1 or f.nout != 1:
            raise TypeError('ReductionKernel must have 1 input and output')
        return f
    elif isinstance(f, cupy.ElementwiseKernel):
        raise TypeError('only ReductionKernel allowed (not ElementwiseKernel)')
    else:
        raise TypeError('bad function type')

@_util.memoize(for_each_device=True)
def _get_generic_filter_red(rk, in_dtype, out_dtype, filter_size, mode, wshape, offsets, cval, int_type):
    if False:
        print('Hello World!')
    'Generic filter implementation based on a reduction kernel.'
    (in_param, out_param) = (rk.in_params[0], rk.out_params[0])
    out_ctype = out_param.ctype
    if out_param.dtype is None:
        out_ctype = cupy._core._scalar.get_typename(in_dtype if out_param.ctype == in_param.ctype else out_dtype)
    setup = '\n    int iv = 0;\n    X values[{size}];\n    CArray<X, 1, true, true> sub_in(values, {{{size}}});\n    {out_ctype} val_out;\n    CArray<{out_ctype}, 1, true, true> sub_out(&val_out, {{1}});\n    '.format(size=filter_size, out_ctype=out_ctype)
    sub_call = 'reduction_kernel::{}(sub_in, sub_out);\n    y = cast<Y>(val_out);'.format(rk.name)
    sub_kernel = _reduction_kernel_code(rk, filter_size, out_dtype, in_dtype)
    return _filters_core._generate_nd_kernel('generic_{}_{}'.format(filter_size, rk.name), setup, 'values[iv++] = {value};', sub_call, mode, wshape, int_type, offsets, cval, preamble=sub_kernel, options=getattr(rk, 'options', ()))

def _reduction_kernel_code(rk, filter_size, out_dtype, in_dtype):
    if False:
        i = 10
        return i + 15
    types = {}
    (in_param, out_param) = (rk.in_params[0], rk.out_params[0])
    in_ctype = _get_type_info(in_param, in_dtype, types)
    out_ctype = _get_type_info(out_param, out_dtype, types)
    types = '\n'.join(('typedef {} {};'.format(typ, name) for (name, typ) in types.items()))
    return 'namespace reduction_kernel {{\n{type_preamble}\n{preamble}\n__device__\nvoid {name}({in_const} CArray<{in_ctype}, 1, true, true>& _raw_{in_name},\n            CArray<{out_ctype}, 1, true, true>& _raw_{out_name}) {{\n    // these are just provided so if they are available for the RK\n    CIndexer<1> _in_ind({{{size}}});\n    CIndexer<0> _out_ind;\n\n    #define REDUCE(a, b) ({reduce_expr})\n    #define POST_MAP(a) ({post_map_expr})\n    typedef {reduce_type} _type_reduce;\n    _type_reduce _s = _type_reduce({identity});\n    for (int _j = 0; _j < {size}; ++_j) {{\n        _in_ind.set(_j);\n        {in_const} {in_ctype}& {in_name} = _raw_{in_name}[_j];\n        _type_reduce _a = static_cast<_type_reduce>({pre_map_expr});\n        _s = REDUCE(_s, _a);\n    }}\n    _out_ind.set(0);\n    {out_ctype} &{out_name} = _raw_{out_name}[0];\n    POST_MAP(_s);\n    #undef REDUCE\n    #undef POST_MAP\n}}\n}}'.format(name=rk.name, type_preamble=types, preamble=rk.preamble, in_const='const' if in_param.is_const else '', in_ctype=in_ctype, in_name=in_param.name, out_ctype=out_ctype, out_name=out_param.name, pre_map_expr=rk.map_expr, identity='' if rk.identity is None else rk.identity, size=filter_size, reduce_type=rk.reduce_type, reduce_expr=rk.reduce_expr, post_map_expr=rk.post_map_expr)

def _get_type_info(param, dtype, types):
    if False:
        i = 10
        return i + 15
    if param.dtype is not None:
        return param.ctype
    ctype = cupy._core._scalar.get_typename(dtype)
    types.setdefault(param.ctype, ctype)
    return ctype

@_util.memoize(for_each_device=True)
def _get_generic_filter_raw(rk, filter_size, mode, wshape, offsets, cval, int_type):
    if False:
        i = 10
        return i + 15
    'Generic filter implementation based on a raw kernel.'
    setup = '\n    int iv = 0;\n    double values[{}];\n    double val_out;'.format(filter_size)
    sub_call = 'raw_kernel::{}(values, {}, &val_out);\n    y = cast<Y>(val_out);'.format(rk.name, filter_size)
    return _filters_core._generate_nd_kernel('generic_{}_{}'.format(filter_size, rk.name), setup, 'values[iv++] = cast<double>({value});', sub_call, mode, wshape, int_type, offsets, cval, preamble='namespace raw_kernel {{\n{}\n}}'.format(rk.code.replace('__global__', '__device__')), options=rk.options)

@_util.memoize(for_each_device=True)
def _get_generic_filter1d(rk, length, n_lines, filter_size, origin, mode, cval, in_ctype, out_ctype, int_type):
    if False:
        return 10
    "\n    The generic 1d filter is different than all other filters and thus is the\n    only filter that doesn't use _generate_nd_kernel() and has a completely\n    custom raw kernel.\n    "
    in_length = length + filter_size - 1
    start = filter_size // 2 + origin
    end = start + length
    if mode == 'constant':
        (boundary, boundary_early) = ('', '\n        for (idx_t j = 0; j < {start}; ++j) {{ input_line[j] = {cval}; }}\n        for (idx_t j = {end}; j<{in_length}; ++j) {{ input_line[j] = {cval}; }}\n        '.format(start=start, end=end, in_length=in_length, cval=cval))
    else:
        if length == 1:
            a = b = 'j_ = 0;'
        elif mode == 'reflect':
            j = 'j_ = ({j}) % ({length} * 2);\nj_ = min(j_, 2 * {length} - 1 - j_);'
            a = j.format(j='-1 - j_', length=length)
            b = j.format(j='j_', length=length)
        elif mode == 'mirror':
            j = 'j_ = 1 + (({j}) - 1) % (({length} - 1) * 2);\nj_ = min(j_, 2 * {length} - 2 - j_);'
            a = j.format(j='-j_', length=length)
            b = j.format(j='j_', length=length)
        elif mode == 'nearest':
            (a, b) = ('j_ = 0;', 'j_ = {length}-1;'.format(length=length))
        elif mode == 'wrap':
            a = 'j_ = j_ % {length} + {length};'.format(length=length)
            b = 'j_ = j_ % {length};'.format(length=length)
        loop = 'for (idx_t j = {{}}; j < {{}}; ++j) {{{{\n            idx_t j_ = j - {start};\n            {{}}\n            input_line[j] = input_line[j_ + {start}];\n        }}}}'.format(start=start)
        boundary_early = ''
        boundary = loop.format(0, start, a) + '\n' + loop.format(end, in_length, b)
    name = 'generic1d_{}_{}_{}'.format(length, filter_size, rk.name)
    code = '#include "cupy/carray.cuh"\n#include "cupy/complex.cuh"\n{include_type_traits}  // let Jitify handle this\n\nnamespace raw_kernel {{\n{rk_code}\n}}\n\n{CAST}\n\ntypedef unsigned char byte;\ntypedef {in_ctype} X;\ntypedef {out_ctype} Y;\ntypedef {int_type} idx_t;\n\n__device__ idx_t offset(idx_t i, idx_t axis, idx_t ndim,\n                        const idx_t* shape, const idx_t* strides) {{\n    idx_t index = 0;\n    for (idx_t a = ndim; --a > 0; ) {{\n        if (a == axis) {{ continue; }}\n        index += (i % shape[a]) * strides[a];\n        i /= shape[a];\n    }}\n    return index + strides[0] * i;\n}}\n\nextern "C" __global__\nvoid {name}(const byte* input, byte* output, const idx_t* x) {{\n    const idx_t axis = x[0], ndim = x[1],\n        *shape = x+2, *in_strides = x+2+ndim, *out_strides = x+2+2*ndim;\n\n    const idx_t in_elem_stride = in_strides[axis];\n    const idx_t out_elem_stride = out_strides[axis];\n\n    double input_line[{in_length}];\n    double output_line[{length}];\n    {boundary_early}\n\n    for (idx_t i = ((idx_t)blockIdx.x) * blockDim.x + threadIdx.x;\n            i < {n_lines};\n            i += ((idx_t)blockDim.x) * gridDim.x) {{\n        // Copy line from input (with boundary filling)\n        const byte* input_ = input + offset(i, axis, ndim, shape, in_strides);\n        for (idx_t j = 0; j < {length}; ++j) {{\n            input_line[j+{start}] = (double)*(X*)(input_+j*in_elem_stride);\n        }}\n        {boundary}\n\n        raw_kernel::{rk_name}(input_line, {in_length}, output_line, {length});\n\n        // Copy line to output\n        byte* output_ = output + offset(i, axis, ndim, shape, out_strides);\n        for (idx_t j = 0; j < {length}; ++j) {{\n            *(Y*)(output_+j*out_elem_stride) = cast<Y>(output_line[j]);\n        }}\n    }}\n}}'.format(n_lines=n_lines, length=length, in_length=in_length, start=start, in_ctype=in_ctype, out_ctype=out_ctype, int_type=int_type, boundary_early=boundary_early, boundary=boundary, name=name, rk_name=rk.name, rk_code=rk.code.replace('__global__', '__device__'), include_type_traits='' if runtime.is_hip else '#include <type_traits>\n', CAST=_filters_core._CAST_FUNCTION)
    return cupy.RawKernel(code, name, ('--std=c++11',) + rk.options, jitify=True)