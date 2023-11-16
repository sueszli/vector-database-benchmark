import operator
from llvmlite import ir
from numba.core import cgutils
from numba.core.typing import signature as nb_signature
from numba.cuda.cudaimpl import lower as cuda_lower, registry as cuda_lowering_registry
from numba.extending import lower_builtin, types
from cudf.core.udf import api
from cudf.core.udf._ops import arith_ops, bitwise_ops, comparison_ops, unary_ops
from cudf.core.udf.masked_typing import MaskedType, NAType, _supported_masked_types

@cuda_lowering_registry.lower_constant(NAType)
def constant_na(context, builder, ty, pyval):
    if False:
        print('Hello World!')
    return context.get_dummy_value()

def make_arithmetic_op(op):
    if False:
        return 10
    '\n    Make closures that implement arithmetic operations. See\n    register_arithmetic_op for details.\n    '

    def masked_scalar_op_impl(context, builder, sig, args):
        if False:
            print('Hello World!')
        '\n        Implement `MaskedType` <op> `MaskedType`\n        '
        (masked_type_1, masked_type_2) = sig.args
        masked_return_type = sig.return_type
        m1 = cgutils.create_struct_proxy(masked_type_1)(context, builder, value=args[0])
        m2 = cgutils.create_struct_proxy(masked_type_2)(context, builder, value=args[1])
        result = cgutils.create_struct_proxy(masked_return_type)(context, builder)
        valid = builder.and_(m1.valid, m2.valid)
        result.valid = valid
        with builder.if_then(valid):
            result.value = context.compile_internal(builder, lambda x, y: op(x, y), nb_signature(masked_return_type.value_type, masked_type_1.value_type, masked_type_2.value_type), (m1.value, m2.value))
        return result._getvalue()
    return masked_scalar_op_impl

def make_unary_op(op):
    if False:
        i = 10
        return i + 15
    '\n    Make closures that implement unary operations. See register_unary_op for\n    details.\n    '

    def masked_scalar_unary_op_impl(context, builder, sig, args):
        if False:
            while True:
                i = 10
        '\n        Implement <op> `MaskedType`\n        '
        masked_type_1 = sig.args[0]
        masked_return_type = sig.return_type
        m1 = cgutils.create_struct_proxy(masked_type_1)(context, builder, value=args[0])
        result = cgutils.create_struct_proxy(masked_return_type)(context, builder)
        result.valid = m1.valid
        with builder.if_then(m1.valid):
            result.value = context.compile_internal(builder, lambda x: op(x), nb_signature(masked_return_type.value_type, masked_type_1.value_type), (m1.value,))
        return result._getvalue()
    return masked_scalar_unary_op_impl

def register_arithmetic_op(op):
    if False:
        print('Hello World!')
    '\n    Register a lowering implementation for the\n    arithmetic op `op`.\n\n    Because the lowering implementations compile the final\n    op separately using a lambda and compile_internal, `op`\n    needs to be tied to each lowering implementation using\n    a closure.\n\n    This function makes and lowers a closure for one op.\n\n    '
    to_lower_op = make_arithmetic_op(op)
    cuda_lower(op, MaskedType, MaskedType)(to_lower_op)

def register_unary_op(op):
    if False:
        while True:
            i = 10
    '\n    Register a lowering implementation for the\n    unary op `op`.\n\n    Because the lowering implementations compile the final\n    op separately using a lambda and compile_internal, `op`\n    needs to be tied to each lowering implementation using\n    a closure.\n\n    This function makes and lowers a closure for one op.\n\n    '
    to_lower_op = make_unary_op(op)
    cuda_lower(op, MaskedType)(to_lower_op)

def masked_scalar_null_op_impl(context, builder, sig, args):
    if False:
        print('Hello World!')
    '\n    Implement `MaskedType` <op> `NAType`\n    or `NAType` <op> `MaskedType`\n    The answer to this is known up front so no actual operation\n    needs to take place\n    '
    return_type = sig.return_type
    result = cgutils.create_struct_proxy(MaskedType(return_type.value_type))(context, builder)
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()

def make_const_op(op):
    if False:
        i = 10
        return i + 15

    def masked_scalar_const_op_impl(context, builder, sig, args):
        if False:
            i = 10
            return i + 15
        return_type = sig.return_type
        result = cgutils.create_struct_proxy(return_type)(context, builder)
        result.valid = context.get_constant(types.boolean, 0)
        if isinstance(sig.args[0], MaskedType):
            (masked_type, const_type) = sig.args
            (masked_value, const_value) = args
            indata = cgutils.create_struct_proxy(masked_type)(context, builder, value=masked_value)
            nb_sig = nb_signature(return_type.value_type, masked_type.value_type, const_type)
            compile_args = (indata.value, const_value)
        else:
            (const_type, masked_type) = sig.args
            (const_value, masked_value) = args
            indata = cgutils.create_struct_proxy(masked_type)(context, builder, value=masked_value)
            nb_sig = nb_signature(return_type.value_type, const_type, masked_type.value_type)
            compile_args = (const_value, indata.value)
        with builder.if_then(indata.valid):
            result.value = context.compile_internal(builder, lambda x, y: op(x, y), nb_sig, compile_args)
            result.valid = context.get_constant(types.boolean, 1)
        return result._getvalue()
    return masked_scalar_const_op_impl

def register_const_op(op):
    if False:
        while True:
            i = 10
    to_lower_op = make_const_op(op)
    cuda_lower(op, MaskedType, types.Number)(to_lower_op)
    cuda_lower(op, types.Number, MaskedType)(to_lower_op)
    cuda_lower(op, MaskedType, types.Boolean)(to_lower_op)
    cuda_lower(op, types.Boolean, MaskedType)(to_lower_op)
    cuda_lower(op, MaskedType, types.NPDatetime)(to_lower_op)
    cuda_lower(op, types.NPDatetime, MaskedType)(to_lower_op)
    cuda_lower(op, MaskedType, types.NPTimedelta)(to_lower_op)
    cuda_lower(op, types.NPTimedelta, MaskedType)(to_lower_op)
for binary_op in arith_ops + bitwise_ops + comparison_ops:
    register_arithmetic_op(binary_op)
    register_const_op(binary_op)
    cuda_lower(binary_op, MaskedType, NAType)(masked_scalar_null_op_impl)
    cuda_lower(binary_op, NAType, MaskedType)(masked_scalar_null_op_impl)
for unary_op in unary_ops:
    register_unary_op(unary_op)
register_unary_op(abs)

@cuda_lower(operator.is_, MaskedType, NAType)
@cuda_lower(operator.is_, NAType, MaskedType)
def masked_scalar_is_null_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    '\n    Implement `MaskedType` is `NA`\n    '
    if isinstance(sig.args[1], NAType):
        (masked_type, na) = sig.args
        value = args[0]
    else:
        (na, masked_type) = sig.args
        value = args[1]
    indata = cgutils.create_struct_proxy(masked_type)(context, builder, value=value)
    result = cgutils.alloca_once(builder, ir.IntType(1))
    with builder.if_else(indata.valid) as (then, otherwise):
        with then:
            builder.store(context.get_constant(types.boolean, 0), result)
        with otherwise:
            builder.store(context.get_constant(types.boolean, 1), result)
    return builder.load(result)

@cuda_lower(api.pack_return, MaskedType)
def pack_return_masked_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    return args[0]

@cuda_lower(api.pack_return, types.Boolean)
@cuda_lower(api.pack_return, types.Number)
@cuda_lower(api.pack_return, types.NPDatetime)
@cuda_lower(api.pack_return, types.NPTimedelta)
def pack_return_scalar_impl(context, builder, sig, args):
    if False:
        while True:
            i = 10
    outdata = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    outdata.value = args[0]
    outdata.valid = context.get_constant(types.boolean, 1)
    return outdata._getvalue()

@cuda_lower(operator.truth, MaskedType)
@cuda_lower(bool, MaskedType)
def masked_scalar_bool_impl(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    indata = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
    result = cgutils.alloca_once(builder, ir.IntType(1))
    with builder.if_else(indata.valid) as (then, otherwise):
        with then:
            builder.store(context.cast(builder, indata.value, sig.args[0].value_type, types.boolean), result)
        with otherwise:
            builder.store(context.get_constant(types.boolean, 0), result)
    return builder.load(result)

@cuda_lower(float, MaskedType)
@cuda_lower(int, MaskedType)
def masked_scalar_cast_impl(context, builder, sig, args):
    if False:
        return 10
    input = cgutils.create_struct_proxy(sig.args[0])(context, builder, value=args[0])
    result = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    casted = context.cast(builder, input.value, sig.args[0].value_type, sig.return_type.value_type)
    result.value = casted
    result.valid = input.valid
    return result._getvalue()

@cuda_lowering_registry.lower_cast(types.Any, MaskedType)
def cast_primitive_to_masked(context, builder, fromty, toty, val):
    if False:
        while True:
            i = 10
    casted = context.cast(builder, val, fromty, toty.value_type)
    ext = cgutils.create_struct_proxy(toty)(context, builder)
    ext.value = casted
    ext.valid = context.get_constant(types.boolean, 1)
    return ext._getvalue()

@cuda_lowering_registry.lower_cast(NAType, MaskedType)
def cast_na_to_masked(context, builder, fromty, toty, val):
    if False:
        return 10
    result = cgutils.create_struct_proxy(toty)(context, builder)
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()

@cuda_lowering_registry.lower_cast(MaskedType, MaskedType)
def cast_masked_to_masked(context, builder, fromty, toty, val):
    if False:
        return 10
    "\n    When numba encounters an op that expects a certain type and\n    the input to the op is not of the expected type it will try\n    to cast the input to the appropriate type. But, in our case\n    the input may be a MaskedType, which numba doesn't natively\n    know how to cast to a different MaskedType with a different\n    `value_type`. This implements and registers that cast.\n    "
    operand = cgutils.create_struct_proxy(fromty)(context, builder, value=val)
    casted = context.cast(builder, operand.value, fromty.value_type, toty.value_type)
    ext = cgutils.create_struct_proxy(toty)(context, builder)
    ext.value = casted
    ext.valid = operand.valid
    return ext._getvalue()

def masked_constructor(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    ty = sig.return_type
    (value, valid) = args
    masked = cgutils.create_struct_proxy(ty)(context, builder)
    masked.value = value
    masked.valid = valid
    return masked._getvalue()
for ty in _supported_masked_types:
    lower_builtin(api.Masked, ty, types.boolean)(masked_constructor)

@cuda_lowering_registry.lower_constant(MaskedType)
def lower_constant_masked(context, builder, ty, val):
    if False:
        return 10
    masked = cgutils.create_struct_proxy(ty)(context, builder)
    masked.value = context.get_constant(ty.value_type, val.value)
    masked.valid = context.get_constant(types.boolean, val.valid)
    return masked._getvalue()