"""
Exception handling intrinsics.
"""
from numba.core import types, errors, cgutils
from numba.core.extending import intrinsic

@intrinsic
def exception_check(typingctx):
    if False:
        return 10
    'An intrinsic to check if an exception is raised\n    '

    def codegen(context, builder, signature, args):
        if False:
            i = 10
            return i + 15
        nrt = context.nrt
        return nrt.eh_check(builder)
    restype = types.boolean
    return (restype(), codegen)

@intrinsic
def mark_try_block(typingctx):
    if False:
        i = 10
        return i + 15
    'An intrinsic to mark the start of a *try* block.\n    '

    def codegen(context, builder, signature, args):
        if False:
            return 10
        nrt = context.nrt
        nrt.eh_try(builder)
        return context.get_dummy_value()
    restype = types.none
    return (restype(), codegen)

@intrinsic
def end_try_block(typingctx):
    if False:
        while True:
            i = 10
    'An intrinsic to mark the end of a *try* block.\n    '

    def codegen(context, builder, signature, args):
        if False:
            i = 10
            return i + 15
        nrt = context.nrt
        nrt.eh_end_try(builder)
        return context.get_dummy_value()
    restype = types.none
    return (restype(), codegen)

@intrinsic
def exception_match(typingctx, exc_value, exc_class):
    if False:
        while True:
            i = 10
    'Basically do ``isinstance(exc_value, exc_class)`` for exception objects.\n    Used in ``except Exception:`` syntax.\n    '
    if exc_class.exc_class is not Exception:
        msg = 'Exception matching is limited to {}'
        raise errors.UnsupportedError(msg.format(Exception))

    def codegen(context, builder, signature, args):
        if False:
            return 10
        return cgutils.true_bit
    restype = types.boolean
    return (restype(exc_value, exc_class), codegen)