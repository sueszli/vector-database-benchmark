"""
This file implements the lowering for `dict()`
"""
from numba.core import types
from numba.core.imputils import lower_builtin
_message_dict_support = '\nUnsupported use of `dict()` with keyword argument(s). The only supported uses are `dict()` or `dict(*iterable)`.\n'.strip()

@lower_builtin(dict, types.IterableType)
def dict_constructor(context, builder, sig, args):
    if False:
        while True:
            i = 10
    from numba.typed import Dict
    dicttype = sig.return_type
    (kt, vt) = (dicttype.key_type, dicttype.value_type)

    def dict_impl(iterable):
        if False:
            while True:
                i = 10
        res = Dict.empty(kt, vt)
        for (k, v) in iterable:
            res[k] = v
        return res
    return context.compile_internal(builder, dict_impl, sig, args)

@lower_builtin(dict)
def impl_dict(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    '\n    The `dict()` implementation simply forwards the work to `Dict.empty()`.\n    '
    from numba.typed import Dict
    dicttype = sig.return_type
    (kt, vt) = (dicttype.key_type, dicttype.value_type)

    def call_ctor():
        if False:
            print('Hello World!')
        return Dict.empty(kt, vt)
    return context.compile_internal(builder, call_ctor, sig, args)