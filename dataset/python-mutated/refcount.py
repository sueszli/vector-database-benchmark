"""
Helpers to see the refcount information of an object
"""
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import intrinsic
from numba.core.runtime.nrtdynmod import _meminfo_struct_type

@intrinsic
def dump_refcount(typingctx, obj):
    if False:
        print('Hello World!')
    'Dump the refcount of an object to stdout.\n\n    Returns True if and only if object is reference-counted and NRT is enabled.\n    '

    def codegen(context, builder, signature, args):
        if False:
            for i in range(10):
                print('nop')
        [obj] = args
        [ty] = signature.args
        meminfos = []
        if context.enable_nrt:
            tmp_mis = context.nrt.get_meminfos(builder, ty, obj)
            meminfos.extend(tmp_mis)
        if meminfos:
            pyapi = context.get_python_api(builder)
            gil_state = pyapi.gil_ensure()
            pyapi.print_string('dump refct of {}'.format(ty))
            for (ty, mi) in meminfos:
                miptr = builder.bitcast(mi, _meminfo_struct_type.as_pointer())
                refctptr = cgutils.gep_inbounds(builder, miptr, 0, 0)
                refct = builder.load(refctptr)
                pyapi.print_string(' | {} refct='.format(ty))
                refct_32bit = builder.trunc(refct, ir.IntType(32))
                printed = cgutils.snprintf_stackbuffer(builder, 30, '%d [%p]', refct_32bit, miptr)
                pyapi.sys_write_stdout(printed)
            pyapi.print_string(';\n')
            pyapi.gil_release(gil_state)
            return cgutils.true_bit
        else:
            return cgutils.false_bit
    sig = types.bool_(obj)
    return (sig, codegen)

@intrinsic
def get_refcount(typingctx, obj):
    if False:
        print('Hello World!')
    'Get the current refcount of an object.\n\n    FIXME: only handles the first object\n    '

    def codegen(context, builder, signature, args):
        if False:
            print('Hello World!')
        [obj] = args
        [ty] = signature.args
        meminfos = []
        if context.enable_nrt:
            tmp_mis = context.nrt.get_meminfos(builder, ty, obj)
            meminfos.extend(tmp_mis)
        refcounts = []
        if meminfos:
            for (ty, mi) in meminfos:
                miptr = builder.bitcast(mi, _meminfo_struct_type.as_pointer())
                refctptr = cgutils.gep_inbounds(builder, miptr, 0, 0)
                refct = builder.load(refctptr)
                refct_32bit = builder.trunc(refct, ir.IntType(32))
                refcounts.append(refct_32bit)
        return refcounts[0]
    sig = types.int32(obj)
    return (sig, codegen)