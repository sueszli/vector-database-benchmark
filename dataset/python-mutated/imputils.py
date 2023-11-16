"""
Utilities to simplify the boilerplate for native lowering.
"""
import collections
import contextlib
import inspect
import functools
from enum import Enum
from numba.core import typing, types, utils, cgutils
from numba.core.typing.templates import BaseRegistryLoader

class Registry(object):
    """
    A registry of function and attribute implementations.
    """

    def __init__(self, name='unspecified'):
        if False:
            print('Hello World!')
        self.name = name
        self.functions = []
        self.getattrs = []
        self.setattrs = []
        self.casts = []
        self.constants = []

    def lower(self, func, *argtys):
        if False:
            while True:
                i = 10
        '\n        Decorate an implementation of *func* for the given argument types.\n        *func* may be an actual global function object, or any\n        pseudo-function supported by Numba, such as "getitem".\n\n        The decorated implementation has the signature\n        (context, builder, sig, args).\n        '

        def decorate(impl):
            if False:
                for i in range(10):
                    print('nop')
            self.functions.append((impl, func, argtys))
            return impl
        return decorate

    def _decorate_attr(self, impl, ty, attr, impl_list, decorator):
        if False:
            print('Hello World!')
        real_impl = decorator(impl, ty, attr)
        impl_list.append((real_impl, attr, real_impl.signature))
        return impl

    def lower_getattr(self, ty, attr):
        if False:
            return 10
        '\n        Decorate an implementation of __getattr__ for type *ty* and\n        the attribute *attr*.\n\n        The decorated implementation will have the signature\n        (context, builder, typ, val).\n        '

        def decorate(impl):
            if False:
                i = 10
                return i + 15
            return self._decorate_attr(impl, ty, attr, self.getattrs, _decorate_getattr)
        return decorate

    def lower_getattr_generic(self, ty):
        if False:
            print('Hello World!')
        "\n        Decorate the fallback implementation of __getattr__ for type *ty*.\n\n        The decorated implementation will have the signature\n        (context, builder, typ, val, attr).  The implementation is\n        called for attributes which haven't been explicitly registered\n        with lower_getattr().\n        "
        return self.lower_getattr(ty, None)

    def lower_setattr(self, ty, attr):
        if False:
            while True:
                i = 10
        '\n        Decorate an implementation of __setattr__ for type *ty* and\n        the attribute *attr*.\n\n        The decorated implementation will have the signature\n        (context, builder, sig, args).\n        '

        def decorate(impl):
            if False:
                for i in range(10):
                    print('nop')
            return self._decorate_attr(impl, ty, attr, self.setattrs, _decorate_setattr)
        return decorate

    def lower_setattr_generic(self, ty):
        if False:
            return 10
        "\n        Decorate the fallback implementation of __setattr__ for type *ty*.\n\n        The decorated implementation will have the signature\n        (context, builder, sig, args, attr).  The implementation is\n        called for attributes which haven't been explicitly registered\n        with lower_setattr().\n        "
        return self.lower_setattr(ty, None)

    def lower_cast(self, fromty, toty):
        if False:
            print('Hello World!')
        '\n        Decorate the implementation of implicit conversion between\n        *fromty* and *toty*.\n\n        The decorated implementation will have the signature\n        (context, builder, fromty, toty, val).\n        '

        def decorate(impl):
            if False:
                return 10
            self.casts.append((impl, (fromty, toty)))
            return impl
        return decorate

    def lower_constant(self, ty):
        if False:
            return 10
        '\n        Decorate the implementation for creating a constant of type *ty*.\n\n        The decorated implementation will have the signature\n        (context, builder, ty, pyval).\n        '

        def decorate(impl):
            if False:
                return 10
            self.constants.append((impl, (ty,)))
            return impl
        return decorate

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Lowering Registry<{self.name}>'

class RegistryLoader(BaseRegistryLoader):
    """
    An incremental loader for a target registry.
    """
    registry_items = ('functions', 'getattrs', 'setattrs', 'casts', 'constants')
builtin_registry = Registry('builtin_registry')
lower_builtin = builtin_registry.lower
lower_getattr = builtin_registry.lower_getattr
lower_getattr_generic = builtin_registry.lower_getattr_generic
lower_setattr = builtin_registry.lower_setattr
lower_setattr_generic = builtin_registry.lower_setattr_generic
lower_cast = builtin_registry.lower_cast
lower_constant = builtin_registry.lower_constant

def _decorate_getattr(impl, ty, attr):
    if False:
        return 10
    real_impl = impl
    if attr is not None:

        def res(context, builder, typ, value, attr):
            if False:
                i = 10
                return i + 15
            return real_impl(context, builder, typ, value)
    else:

        def res(context, builder, typ, value, attr):
            if False:
                i = 10
                return i + 15
            return real_impl(context, builder, typ, value, attr)
    res.signature = (ty,)
    res.attr = attr
    return res

def _decorate_setattr(impl, ty, attr):
    if False:
        while True:
            i = 10
    real_impl = impl
    if attr is not None:

        def res(context, builder, sig, args, attr):
            if False:
                return 10
            return real_impl(context, builder, sig, args)
    else:

        def res(context, builder, sig, args, attr):
            if False:
                for i in range(10):
                    print('nop')
            return real_impl(context, builder, sig, args, attr)
    res.signature = (ty, types.Any)
    res.attr = attr
    return res

def fix_returning_optional(context, builder, sig, status, retval):
    if False:
        print('Hello World!')
    if isinstance(sig.return_type, types.Optional):
        value_type = sig.return_type.type
        optional_none = context.make_optional_none(builder, value_type)
        retvalptr = cgutils.alloca_once_value(builder, optional_none)
        with builder.if_then(builder.not_(status.is_none)):
            optional_value = context.make_optional_value(builder, value_type, retval)
            builder.store(optional_value, retvalptr)
        retval = builder.load(retvalptr)
    return retval

def user_function(fndesc, libs):
    if False:
        while True:
            i = 10
    '\n    A wrapper inserting code calling Numba-compiled *fndesc*.\n    '

    def imp(context, builder, sig, args):
        if False:
            print('Hello World!')
        func = context.declare_function(builder.module, fndesc)
        (status, retval) = context.call_conv.call_function(builder, func, fndesc.restype, fndesc.argtypes, args)
        with cgutils.if_unlikely(builder, status.is_error):
            context.call_conv.return_status_propagate(builder, status)
        assert sig.return_type == fndesc.restype
        retval = fix_returning_optional(context, builder, sig, status, retval)
        if retval.type != context.get_value_type(sig.return_type):
            msg = 'function returned {0} but expect {1}'
            raise TypeError(msg.format(retval.type, sig.return_type))
        return impl_ret_new_ref(context, builder, fndesc.restype, retval)
    imp.signature = fndesc.argtypes
    imp.libs = tuple(libs)
    return imp

def user_generator(gendesc, libs):
    if False:
        print('Hello World!')
    '\n    A wrapper inserting code calling Numba-compiled *gendesc*.\n    '

    def imp(context, builder, sig, args):
        if False:
            for i in range(10):
                print('nop')
        func = context.declare_function(builder.module, gendesc)
        (status, retval) = context.call_conv.call_function(builder, func, gendesc.restype, gendesc.argtypes, args)
        return (status, retval)
    imp.libs = tuple(libs)
    return imp

def iterator_impl(iterable_type, iterator_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator a given class as implementing *iterator_type*\n    (by providing an `iternext()` method).\n    '

    def wrapper(cls):
        if False:
            return 10
        iternext = cls.iternext

        @iternext_impl(RefType.BORROWED)
        def iternext_wrapper(context, builder, sig, args, result):
            if False:
                for i in range(10):
                    print('nop')
            (value,) = args
            iterobj = cls(context, builder, value)
            return iternext(iterobj, context, builder, result)
        lower_builtin('iternext', iterator_type)(iternext_wrapper)
        return cls
    return wrapper

class _IternextResult(object):
    """
    A result wrapper for iteration, passed by iternext_impl() into the
    wrapped function.
    """
    __slots__ = ('_context', '_builder', '_pairobj')

    def __init__(self, context, builder, pairobj):
        if False:
            while True:
                i = 10
        self._context = context
        self._builder = builder
        self._pairobj = pairobj

    def set_exhausted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mark the iterator as exhausted.\n        '
        self._pairobj.second = self._context.get_constant(types.boolean, False)

    def set_valid(self, is_valid=True):
        if False:
            print('Hello World!')
        '\n        Mark the iterator as valid according to *is_valid* (which must\n        be either a Python boolean or a LLVM inst).\n        '
        if is_valid in (False, True):
            is_valid = self._context.get_constant(types.boolean, is_valid)
        self._pairobj.second = is_valid

    def yield_(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mark the iterator as yielding the given *value* (a LLVM inst).\n        '
        self._pairobj.first = value

    def is_valid(self):
        if False:
            return 10
        '\n        Return whether the iterator is marked valid.\n        '
        return self._context.get_argument_value(self._builder, types.boolean, self._pairobj.second)

    def yielded_value(self):
        if False:
            print('Hello World!')
        "\n        Return the iterator's yielded value, if any.\n        "
        return self._pairobj.first

class RefType(Enum):
    """
    Enumerate the reference type
    """
    '\n    A new reference\n    '
    NEW = 1
    '\n    A borrowed reference\n    '
    BORROWED = 2
    '\n    An untracked reference\n    '
    UNTRACKED = 3

def iternext_impl(ref_type=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap the given iternext() implementation so that it gets passed\n    an _IternextResult() object easing the returning of the iternext()\n    result pair.\n\n    ref_type: a numba.targets.imputils.RefType value, the reference type used is\n    that specified through the RefType enum.\n\n    The wrapped function will be called with the following signature:\n        (context, builder, sig, args, iternext_result)\n    '
    if ref_type not in [x for x in RefType]:
        raise ValueError('ref_type must be an enum member of imputils.RefType')

    def outer(func):
        if False:
            i = 10
            return i + 15

        def wrapper(context, builder, sig, args):
            if False:
                for i in range(10):
                    print('nop')
            pair_type = sig.return_type
            pairobj = context.make_helper(builder, pair_type)
            func(context, builder, sig, args, _IternextResult(context, builder, pairobj))
            if ref_type == RefType.NEW:
                impl_ret = impl_ret_new_ref
            elif ref_type == RefType.BORROWED:
                impl_ret = impl_ret_borrowed
            elif ref_type == RefType.UNTRACKED:
                impl_ret = impl_ret_untracked
            else:
                raise ValueError('Unknown ref_type encountered')
            return impl_ret(context, builder, pair_type, pairobj._getvalue())
        return wrapper
    return outer

def call_getiter(context, builder, iterable_type, val):
    if False:
        while True:
            i = 10
    '\n    Call the `getiter()` implementation for the given *iterable_type*\n    of value *val*, and return the corresponding LLVM inst.\n    '
    getiter_sig = typing.signature(iterable_type.iterator_type, iterable_type)
    getiter_impl = context.get_function('getiter', getiter_sig)
    return getiter_impl(builder, (val,))

def call_iternext(context, builder, iterator_type, val):
    if False:
        i = 10
        return i + 15
    '\n    Call the `iternext()` implementation for the given *iterator_type*\n    of value *val*, and return a convenience _IternextResult() object\n    reflecting the results.\n    '
    itemty = iterator_type.yield_type
    pair_type = types.Pair(itemty, types.boolean)
    iternext_sig = typing.signature(pair_type, iterator_type)
    iternext_impl = context.get_function('iternext', iternext_sig)
    val = iternext_impl(builder, (val,))
    pairobj = context.make_helper(builder, pair_type, val)
    return _IternextResult(context, builder, pairobj)

def call_len(context, builder, ty, val):
    if False:
        for i in range(10):
            print('nop')
    "\n    Call len() on the given value.  Return None if len() isn't defined on\n    this type.\n    "
    try:
        len_impl = context.get_function(len, typing.signature(types.intp, ty))
    except NotImplementedError:
        return None
    else:
        return len_impl(builder, (val,))
_ForIterLoop = collections.namedtuple('_ForIterLoop', ('value', 'do_break'))

@contextlib.contextmanager
def for_iter(context, builder, iterable_type, val):
    if False:
        i = 10
        return i + 15
    '\n    Simulate a for loop on the given iterable.  Yields a namedtuple with\n    the given members:\n    - `value` is the value being yielded\n    - `do_break` is a callable to early out of the loop\n    '
    iterator_type = iterable_type.iterator_type
    iterval = call_getiter(context, builder, iterable_type, val)
    bb_body = builder.append_basic_block('for_iter.body')
    bb_end = builder.append_basic_block('for_iter.end')

    def do_break():
        if False:
            return 10
        builder.branch(bb_end)
    builder.branch(bb_body)
    with builder.goto_block(bb_body):
        res = call_iternext(context, builder, iterator_type, iterval)
        with builder.if_then(builder.not_(res.is_valid()), likely=False):
            builder.branch(bb_end)
        yield _ForIterLoop(res.yielded_value(), do_break)
        builder.branch(bb_body)
    builder.position_at_end(bb_end)
    if context.enable_nrt:
        context.nrt.decref(builder, iterator_type, iterval)

def impl_ret_new_ref(ctx, builder, retty, ret):
    if False:
        i = 10
        return i + 15
    '\n    The implementation returns a new reference.\n    '
    return ret

def impl_ret_borrowed(ctx, builder, retty, ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    The implementation returns a borrowed reference.\n    This function automatically incref so that the implementation is\n    returning a new reference.\n    '
    if ctx.enable_nrt:
        ctx.nrt.incref(builder, retty, ret)
    return ret

def impl_ret_untracked(ctx, builder, retty, ret):
    if False:
        return 10
    '\n    The return type is not a NRT object.\n    '
    return ret

@contextlib.contextmanager
def force_error_model(context, model_name='numpy'):
    if False:
        while True:
            i = 10
    "\n    Temporarily change the context's error model.\n    "
    from numba.core import callconv
    old_error_model = context.error_model
    context.error_model = callconv.create_error_model(model_name, context)
    try:
        yield
    finally:
        context.error_model = old_error_model

def numba_typeref_ctor(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'A stub for use internally by Numba when a call is emitted\n    on a TypeRef.\n    '
    raise NotImplementedError('This function should not be executed.')