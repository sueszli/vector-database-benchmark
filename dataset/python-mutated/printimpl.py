from functools import singledispatch
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.errors import NumbaWarning
from numba.core.imputils import Registry
from numba.cuda import nvvmutils
from warnings import warn
registry = Registry()
lower = registry.lower
voidptr = ir.PointerType(ir.IntType(8))

@singledispatch
def print_item(ty, context, builder, val):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handle printing of a single value of the given Numba type.\n    A (format string, [list of arguments]) is returned that will allow\n    forming the final printf()-like call.\n    '
    raise NotImplementedError('printing unimplemented for values of type %s' % (ty,))

@print_item.register(types.Integer)
@print_item.register(types.IntegerLiteral)
def int_print_impl(ty, context, builder, val):
    if False:
        return 10
    if ty in types.unsigned_domain:
        rawfmt = '%llu'
        dsttype = types.uint64
    else:
        rawfmt = '%lld'
        dsttype = types.int64
    lld = context.cast(builder, val, ty, dsttype)
    return (rawfmt, [lld])

@print_item.register(types.Float)
def real_print_impl(ty, context, builder, val):
    if False:
        i = 10
        return i + 15
    lld = context.cast(builder, val, ty, types.float64)
    return ('%f', [lld])

@print_item.register(types.StringLiteral)
def const_print_impl(ty, context, builder, sigval):
    if False:
        for i in range(10):
            print('nop')
    pyval = ty.literal_value
    assert isinstance(pyval, str)
    rawfmt = '%s'
    val = context.insert_string_const_addrspace(builder, pyval)
    return (rawfmt, [val])

@lower(print, types.VarArg(types.Any))
def print_varargs(context, builder, sig, args):
    if False:
        print('Hello World!')
    "This function is a generic 'print' wrapper for arbitrary types.\n    It dispatches to the appropriate 'print' implementations above\n    depending on the detected real types in the signature."
    vprint = nvvmutils.declare_vprint(builder.module)
    formats = []
    values = []
    for (i, (argtype, argval)) in enumerate(zip(sig.args, args)):
        (argfmt, argvals) = print_item(argtype, context, builder, argval)
        formats.append(argfmt)
        values.extend(argvals)
    rawfmt = ' '.join(formats) + '\n'
    if len(args) > 32:
        msg = 'CUDA print() cannot print more than 32 items. The raw format string will be emitted by the kernel instead.'
        warn(msg, NumbaWarning)
        rawfmt = rawfmt.replace('%', '%%')
    fmt = context.insert_string_const_addrspace(builder, rawfmt)
    array = cgutils.make_anonymous_struct(builder, values)
    arrayptr = cgutils.alloca_once_value(builder, array)
    vprint = nvvmutils.declare_vprint(builder.module)
    builder.call(vprint, (fmt, builder.bitcast(arrayptr, voidptr)))
    return context.get_dummy_value()