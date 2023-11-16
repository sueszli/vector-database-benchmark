""" Common compiler level utilities for typed dict and list. """
import operator
import warnings
from llvmlite import ir
from numba.core import types, cgutils
from numba.core import typing
from numba.core.registry import cpu_target
from numba.core.typeconv import Conversion
from numba.core.extending import intrinsic
from numba.core.errors import TypingError, NumbaTypeSafetyWarning

def _as_bytes(builder, ptr):
    if False:
        return 10
    'Helper to do (void*)ptr\n    '
    return builder.bitcast(ptr, cgutils.voidptr_t)

@intrinsic
def _cast(typingctx, val, typ):
    if False:
        for i in range(10):
            print('nop')
    'Cast *val* to *typ*\n    '

    def codegen(context, builder, signature, args):
        if False:
            for i in range(10):
                print('nop')
        [val, typ] = args
        context.nrt.incref(builder, signature.return_type, val)
        return val
    casted = typ.instance_type
    _sentry_safe_cast(val, casted)
    sig = casted(casted, typ)
    return (sig, codegen)

def _sentry_safe_cast(fromty, toty):
    if False:
        print('Hello World!')
    'Check and raise TypingError if *fromty* cannot be safely cast to *toty*\n    '
    tyctxt = cpu_target.typing_context
    (fromty, toty) = map(types.unliteral, (fromty, toty))
    by = tyctxt.can_convert(fromty, toty)

    def warn():
        if False:
            print('Hello World!')
        m = 'unsafe cast from {} to {}. Precision may be lost.'
        warnings.warn(m.format(fromty, toty), category=NumbaTypeSafetyWarning)
    isint = lambda x: isinstance(x, types.Integer)
    isflt = lambda x: isinstance(x, types.Float)
    iscmplx = lambda x: isinstance(x, types.Complex)
    isdict = lambda x: isinstance(x, types.DictType)
    if by is None or by > Conversion.safe:
        if isint(fromty) and isint(toty):
            warn()
        elif isint(fromty) and isflt(toty):
            warn()
        elif isflt(fromty) and isflt(toty):
            warn()
        elif iscmplx(fromty) and iscmplx(toty):
            warn()
        elif isdict(fromty) and isdict(toty):
            pass
        elif not isinstance(toty, types.Number):
            warn()
        else:
            m = 'cannot safely cast {} to {}. Please cast explicitly.'
            raise TypingError(m.format(fromty, toty))

def _sentry_safe_cast_default(default, valty):
    if False:
        while True:
            i = 10
    'Similar to _sentry_safe_cast but handle default value.\n    '
    if default is None:
        return
    if isinstance(default, (types.Omitted, types.NoneType)):
        return
    return _sentry_safe_cast(default, valty)

@intrinsic
def _nonoptional(typingctx, val):
    if False:
        i = 10
        return i + 15
    'Typing trick to cast Optional[T] to T\n    '
    if not isinstance(val, types.Optional):
        raise TypeError('expected an optional')

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        context.nrt.incref(builder, sig.return_type, args[0])
        return args[0]
    casted = val.type
    sig = casted(casted)
    return (sig, codegen)

def _container_get_data(context, builder, container_ty, c):
    if False:
        return 10
    'Helper to get the C list pointer in a numba containers.\n    '
    ctor = cgutils.create_struct_proxy(container_ty)
    conatainer_struct = ctor(context, builder, value=c)
    return conatainer_struct.data

def _container_get_meminfo(context, builder, container_ty, c):
    if False:
        while True:
            i = 10
    'Helper to get the meminfo for a container\n    '
    ctor = cgutils.create_struct_proxy(container_ty)
    conatainer_struct = ctor(context, builder, value=c)
    return conatainer_struct.meminfo

def _get_incref_decref(context, module, datamodel, container_element_type):
    if False:
        return 10
    assert datamodel.contains_nrt_meminfo()
    fe_type = datamodel.fe_type
    data_ptr_ty = datamodel.get_data_type().as_pointer()
    refct_fnty = ir.FunctionType(ir.VoidType(), [data_ptr_ty])
    incref_fn = cgutils.get_or_insert_function(module, refct_fnty, '.numba_{}.{}_incref'.format(context.fndesc.mangled_name, container_element_type))
    builder = ir.IRBuilder(incref_fn.append_basic_block())
    context.nrt.incref(builder, fe_type, datamodel.load_from_data_pointer(builder, incref_fn.args[0]))
    builder.ret_void()
    decref_fn = cgutils.get_or_insert_function(module, refct_fnty, name='.numba_{}.{}_decref'.format(context.fndesc.mangled_name, container_element_type))
    builder = ir.IRBuilder(decref_fn.append_basic_block())
    context.nrt.decref(builder, fe_type, datamodel.load_from_data_pointer(builder, decref_fn.args[0]))
    builder.ret_void()
    return (incref_fn, decref_fn)

def _get_equal(context, module, datamodel, container_element_type):
    if False:
        return 10
    assert datamodel.contains_nrt_meminfo()
    fe_type = datamodel.fe_type
    data_ptr_ty = datamodel.get_data_type().as_pointer()
    wrapfnty = context.call_conv.get_function_type(types.int32, [fe_type, fe_type])
    argtypes = [fe_type, fe_type]

    def build_wrapper(fn):
        if False:
            return 10
        builder = ir.IRBuilder(fn.append_basic_block())
        args = context.call_conv.decode_arguments(builder, argtypes, fn)
        sig = typing.signature(types.boolean, fe_type, fe_type)
        op = operator.eq
        fnop = context.typing_context.resolve_value_type(op)
        fnop.get_call_type(context.typing_context, sig.args, {})
        eqfn = context.get_function(fnop, sig)
        res = eqfn(builder, args)
        intres = context.cast(builder, res, types.boolean, types.int32)
        context.call_conv.return_value(builder, intres)
    wrapfn = cgutils.get_or_insert_function(module, wrapfnty, name='.numba_{}.{}_equal.wrap'.format(context.fndesc.mangled_name, container_element_type))
    build_wrapper(wrapfn)
    equal_fnty = ir.FunctionType(ir.IntType(32), [data_ptr_ty, data_ptr_ty])
    equal_fn = cgutils.get_or_insert_function(module, equal_fnty, name='.numba_{}.{}_equal'.format(context.fndesc.mangled_name, container_element_type))
    builder = ir.IRBuilder(equal_fn.append_basic_block())
    lhs = datamodel.load_from_data_pointer(builder, equal_fn.args[0])
    rhs = datamodel.load_from_data_pointer(builder, equal_fn.args[1])
    (status, retval) = context.call_conv.call_function(builder, wrapfn, types.boolean, argtypes, [lhs, rhs])
    with builder.if_then(status.is_ok, likely=True):
        with builder.if_then(status.is_none):
            builder.ret(context.get_constant(types.int32, 0))
        retval = context.cast(builder, retval, types.boolean, types.int32)
        builder.ret(retval)
    builder.ret(context.get_constant(types.int32, -1))
    return equal_fn