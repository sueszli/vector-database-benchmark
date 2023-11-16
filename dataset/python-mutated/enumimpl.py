"""
Implementation of enums.
"""
import operator
from numba.core.imputils import lower_builtin, lower_getattr, lower_getattr_generic, lower_cast, lower_constant, impl_ret_untracked
from numba.core import types
from numba.core.extending import overload_method

@lower_builtin(operator.eq, types.EnumMember, types.EnumMember)
def enum_eq(context, builder, sig, args):
    if False:
        return 10
    (tu, tv) = sig.args
    (u, v) = args
    res = context.generic_compare(builder, operator.eq, (tu.dtype, tv.dtype), (u, v))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.is_, types.EnumMember, types.EnumMember)
def enum_is(context, builder, sig, args):
    if False:
        print('Hello World!')
    (tu, tv) = sig.args
    (u, v) = args
    if tu == tv:
        res = context.generic_compare(builder, operator.eq, (tu.dtype, tv.dtype), (u, v))
    else:
        res = context.get_constant(sig.return_type, False)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.ne, types.EnumMember, types.EnumMember)
def enum_ne(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    (tu, tv) = sig.args
    (u, v) = args
    res = context.generic_compare(builder, operator.ne, (tu.dtype, tv.dtype), (u, v))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_getattr(types.EnumMember, 'value')
def enum_value(context, builder, ty, val):
    if False:
        return 10
    return val

@lower_cast(types.IntEnumMember, types.Integer)
def int_enum_to_int(context, builder, fromty, toty, val):
    if False:
        while True:
            i = 10
    '\n    Convert an IntEnum member to its raw integer value.\n    '
    return context.cast(builder, val, fromty.dtype, toty)

@lower_constant(types.EnumMember)
def enum_constant(context, builder, ty, pyval):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a LLVM constant representing enum member *pyval*.\n    '
    return context.get_constant_generic(builder, ty.dtype, pyval.value)

@lower_getattr_generic(types.EnumClass)
def enum_class_getattr(context, builder, ty, val, attr):
    if False:
        print('Hello World!')
    '\n    Return an enum member by attribute name.\n    '
    member = getattr(ty.instance_class, attr)
    return context.get_constant_generic(builder, ty.dtype, member.value)

@lower_builtin('static_getitem', types.EnumClass, types.StringLiteral)
def enum_class_getitem(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return an enum member by index name.\n    '
    (enum_cls_typ, idx) = sig.args
    member = enum_cls_typ.instance_class[idx.literal_value]
    return context.get_constant_generic(builder, enum_cls_typ.dtype, member.value)

@overload_method(types.IntEnumMember, '__hash__')
def intenum_hash(val):
    if False:
        return 10

    def hash_impl(val):
        if False:
            i = 10
            return i + 15
        return hash(val.value)
    return hash_impl