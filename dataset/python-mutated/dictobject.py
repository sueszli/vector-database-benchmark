"""
Compiler-side implementation of the dictionary.
"""
import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import overload, overload_method, overload_attribute, intrinsic, register_model, models, lower_builtin, lower_cast, make_attribute_wrapper
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import DictType, DictItemsIterableType, DictKeysIterableType, DictValuesIterableType, DictIteratorType, Type
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import _as_bytes, _cast, _nonoptional, _sentry_safe_cast_default, _get_incref_decref, _get_equal, _container_get_data
ll_dict_type = cgutils.voidptr_t
ll_dictiter_type = cgutils.voidptr_t
ll_voidptr_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t
ll_hash = ll_ssize_t
ll_bytes = cgutils.voidptr_t
_meminfo_dictptr = types.MemInfoPointer(types.voidptr)

class DKIX(IntEnum):
    """Special return value of dict lookup.
    """
    EMPTY = -1

class Status(IntEnum):
    """Status code for other dict operations.
    """
    OK = 0
    OK_REPLACED = 1
    ERR_NO_MEMORY = -1
    ERR_DICT_MUTATED = -2
    ERR_ITER_EXHAUSTED = -3
    ERR_DICT_EMPTY = -4
    ERR_CMP_FAILED = -5

def new_dict(key, value, n_keys=0):
    if False:
        return 10
    'Construct a new dict with enough space for *n_keys* without a resize.\n\n    Parameters\n    ----------\n    key, value : TypeRef\n        Key type and value type of the new dict.\n    n_keys : int, default 0\n        The number of keys to insert without needing a resize.\n        A value of 0 creates a dict with minimum size.\n    '
    return dict()

@register_model(DictType)
class DictModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            print('Hello World!')
        members = [('meminfo', _meminfo_dictptr), ('data', types.voidptr)]
        super(DictModel, self).__init__(dmm, fe_type, members)

@register_model(DictItemsIterableType)
@register_model(DictKeysIterableType)
@register_model(DictValuesIterableType)
@register_model(DictIteratorType)
class DictIterModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        if False:
            while True:
                i = 10
        members = [('parent', fe_type.parent), ('state', types.voidptr)]
        super(DictIterModel, self).__init__(dmm, fe_type, members)
make_attribute_wrapper(DictItemsIterableType, 'parent', '_parent')
make_attribute_wrapper(DictKeysIterableType, 'parent', '_parent')
make_attribute_wrapper(DictValuesIterableType, 'parent', '_parent')

def _raise_if_error(context, builder, status, msg):
    if False:
        while True:
            i = 10
    'Raise an internal error depending on the value of *status*\n    '
    ok_status = status.type(int(Status.OK))
    with builder.if_then(builder.icmp_signed('!=', status, ok_status)):
        context.call_conv.return_user_exc(builder, RuntimeError, (msg,))

@intrinsic
def _as_meminfo(typingctx, dctobj):
    if False:
        print('Hello World!')
    'Returns the MemInfoPointer of a dictionary.\n    '
    if not isinstance(dctobj, types.DictType):
        raise TypingError('expected *dctobj* to be a DictType')

    def codegen(context, builder, sig, args):
        if False:
            for i in range(10):
                print('nop')
        [td] = sig.args
        [d] = args
        context.nrt.incref(builder, td, d)
        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        return dstruct.meminfo
    sig = _meminfo_dictptr(dctobj)
    return (sig, codegen)

@intrinsic
def _from_meminfo(typingctx, mi, dicttyperef):
    if False:
        return 10
    'Recreate a dictionary from a MemInfoPointer\n    '
    if mi != _meminfo_dictptr:
        raise TypingError('expected a MemInfoPointer for dict.')
    dicttype = dicttyperef.instance_type
    if not isinstance(dicttype, DictType):
        raise TypingError('expected a {}'.format(DictType))

    def codegen(context, builder, sig, args):
        if False:
            i = 10
            return i + 15
        [tmi, tdref] = sig.args
        td = tdref.instance_type
        [mi, _] = args
        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder)
        data_pointer = context.nrt.meminfo_data(builder, mi)
        data_pointer = builder.bitcast(data_pointer, ll_dict_type.as_pointer())
        dstruct.data = builder.load(data_pointer)
        dstruct.meminfo = mi
        return impl_ret_borrowed(context, builder, dicttype, dstruct._getvalue())
    sig = dicttype(mi, dicttyperef)
    return (sig, codegen)

def _call_dict_free(context, builder, ptr):
    if False:
        for i in range(10):
            print('nop')
    'Call numba_dict_free(ptr)\n    '
    fnty = ir.FunctionType(ir.VoidType(), [ll_dict_type])
    free = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_free')
    builder.call(free, [ptr])

def _imp_dtor(context, module):
    if False:
        for i in range(10):
            print('nop')
    'Define the dtor for dictionary\n    '
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    fnty = ir.FunctionType(ir.VoidType(), [llvoidptr, llsize, llvoidptr])
    fname = '_numba_dict_dtor'
    fn = cgutils.get_or_insert_function(module, fnty, fname)
    if fn.is_declaration:
        fn.linkage = 'linkonce_odr'
        builder = ir.IRBuilder(fn.append_basic_block())
        dp = builder.bitcast(fn.args[0], ll_dict_type.as_pointer())
        d = builder.load(dp)
        _call_dict_free(context, builder, d)
        builder.ret_void()
    return fn

@intrinsic
def _dict_new_sized(typingctx, n_keys, keyty, valty):
    if False:
        while True:
            i = 10
    'Wrap numba_dict_new_sized.\n\n    Allocate a new dictionary object with enough space to hold\n    *n_keys* keys without needing a resize.\n\n    Parameters\n    ----------\n    keyty, valty: Type\n        Type of the key and value, respectively.\n    n_keys: int\n        The number of keys to insert without needing a resize.\n        A value of 0 creates a dict with minimum size.\n    '
    resty = types.voidptr
    sig = resty(n_keys, keyty, valty)

    def codegen(context, builder, sig, args):
        if False:
            return 10
        n_keys = builder.bitcast(args[0], ll_ssize_t)
        ll_key = context.get_data_type(keyty.instance_type)
        ll_val = context.get_data_type(valty.instance_type)
        sz_key = context.get_abi_sizeof(ll_key)
        sz_val = context.get_abi_sizeof(ll_val)
        refdp = cgutils.alloca_once(builder, ll_dict_type, zfill=True)
        argtys = [ll_dict_type.as_pointer(), ll_ssize_t, ll_ssize_t, ll_ssize_t]
        fnty = ir.FunctionType(ll_status, argtys)
        fn = ir.Function(builder.module, fnty, 'numba_dict_new_sized')
        args = [refdp, n_keys, ll_ssize_t(sz_key), ll_ssize_t(sz_val)]
        status = builder.call(fn, args)
        allocated_failed_msg = 'Failed to allocate dictionary'
        _raise_if_error(context, builder, status, msg=allocated_failed_msg)
        dp = builder.load(refdp)
        return dp
    return (sig, codegen)

@intrinsic
def _dict_set_method_table(typingctx, dp, keyty, valty):
    if False:
        i = 10
        return i + 15
    'Wrap numba_dict_set_method_table\n    '
    resty = types.void
    sig = resty(dp, keyty, valty)

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        vtablety = ir.LiteralStructType([ll_voidptr_type, ll_voidptr_type, ll_voidptr_type, ll_voidptr_type, ll_voidptr_type])
        setmethod_fnty = ir.FunctionType(ir.VoidType(), [ll_dict_type, vtablety.as_pointer()])
        setmethod_fn = ir.Function(builder.module, setmethod_fnty, name='numba_dict_set_method_table')
        dp = args[0]
        vtable = cgutils.alloca_once(builder, vtablety, zfill=True)
        key_equal_ptr = cgutils.gep_inbounds(builder, vtable, 0, 0)
        key_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 1)
        key_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 2)
        val_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 3)
        val_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 4)
        dm_key = context.data_model_manager[keyty.instance_type]
        if dm_key.contains_nrt_meminfo():
            equal = _get_equal(context, builder.module, dm_key, 'dict_key')
            (key_incref, key_decref) = _get_incref_decref(context, builder.module, dm_key, 'dict_key')
            builder.store(builder.bitcast(equal, key_equal_ptr.type.pointee), key_equal_ptr)
            builder.store(builder.bitcast(key_incref, key_incref_ptr.type.pointee), key_incref_ptr)
            builder.store(builder.bitcast(key_decref, key_decref_ptr.type.pointee), key_decref_ptr)
        dm_val = context.data_model_manager[valty.instance_type]
        if dm_val.contains_nrt_meminfo():
            (val_incref, val_decref) = _get_incref_decref(context, builder.module, dm_val, 'dict_value')
            builder.store(builder.bitcast(val_incref, val_incref_ptr.type.pointee), val_incref_ptr)
            builder.store(builder.bitcast(val_decref, val_decref_ptr.type.pointee), val_decref_ptr)
        builder.call(setmethod_fn, [dp, vtable])
    return (sig, codegen)

@intrinsic
def _dict_insert(typingctx, d, key, hashval, val):
    if False:
        while True:
            i = 10
    'Wrap numba_dict_insert\n    '
    resty = types.int32
    sig = resty(d, d.key_type, types.intp, d.value_type)

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(ll_status, [ll_dict_type, ll_bytes, ll_hash, ll_bytes, ll_bytes])
        [d, key, hashval, val] = args
        [td, tkey, thashval, tval] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_insert')
        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[tval]
        data_key = dm_key.as_data(builder, key)
        data_val = dm_val.as_data(builder, val)
        ptr_key = cgutils.alloca_once_value(builder, data_key)
        cgutils.memset_padding(builder, ptr_key)
        ptr_val = cgutils.alloca_once_value(builder, data_val)
        ptr_oldval = cgutils.alloca_once(builder, data_val.type)
        dp = _container_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, _as_bytes(builder, ptr_key), hashval, _as_bytes(builder, ptr_val), _as_bytes(builder, ptr_oldval)])
        return status
    return (sig, codegen)

@intrinsic
def _dict_length(typingctx, d):
    if False:
        i = 10
        return i + 15
    'Wrap numba_dict_length\n\n    Returns the length of the dictionary.\n    '
    resty = types.intp
    sig = resty(d)

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(ll_ssize_t, [ll_dict_type])
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_length')
        [d] = args
        [td] = sig.args
        dp = _container_get_data(context, builder, td, d)
        n = builder.call(fn, [dp])
        return n
    return (sig, codegen)

@intrinsic
def _dict_dump(typingctx, d):
    if False:
        return 10
    'Dump the dictionary keys and values.\n    Wraps numba_dict_dump for debugging.\n    '
    resty = types.void
    sig = resty(d)

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        fnty = ir.FunctionType(ir.VoidType(), [ll_dict_type])
        [td] = sig.args
        [d] = args
        dp = _container_get_data(context, builder, td, d)
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_dump')
        builder.call(fn, [dp])
    return (sig, codegen)

@intrinsic
def _dict_lookup(typingctx, d, key, hashval):
    if False:
        return 10
    'Wrap numba_dict_lookup\n\n    Returns 2-tuple of (intp, ?value_type)\n    '
    resty = types.Tuple([types.intp, types.Optional(d.value_type)])
    sig = resty(d, key, hashval)

    def codegen(context, builder, sig, args):
        if False:
            return 10
        fnty = ir.FunctionType(ll_ssize_t, [ll_dict_type, ll_bytes, ll_hash, ll_bytes])
        [td, tkey, thashval] = sig.args
        [d, key, hashval] = args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_lookup')
        dm_key = context.data_model_manager[tkey]
        dm_val = context.data_model_manager[td.value_type]
        data_key = dm_key.as_data(builder, key)
        ptr_key = cgutils.alloca_once_value(builder, data_key)
        cgutils.memset_padding(builder, ptr_key)
        ll_val = context.get_data_type(td.value_type)
        ptr_val = cgutils.alloca_once(builder, ll_val)
        dp = _container_get_data(context, builder, td, d)
        ix = builder.call(fn, [dp, _as_bytes(builder, ptr_key), hashval, _as_bytes(builder, ptr_val)])
        found = builder.icmp_signed('>', ix, ix.type(int(DKIX.EMPTY)))
        out = context.make_optional_none(builder, td.value_type)
        pout = cgutils.alloca_once_value(builder, out)
        with builder.if_then(found):
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            context.nrt.incref(builder, td.value_type, val)
            loaded = context.make_optional_value(builder, td.value_type, val)
            builder.store(loaded, pout)
        out = builder.load(pout)
        return context.make_tuple(builder, resty, [ix, out])
    return (sig, codegen)

@intrinsic
def _dict_popitem(typingctx, d):
    if False:
        while True:
            i = 10
    'Wrap numba_dict_popitem\n    '
    keyvalty = types.Tuple([d.key_type, d.value_type])
    resty = types.Tuple([types.int32, types.Optional(keyvalty)])
    sig = resty(d)

    def codegen(context, builder, sig, args):
        if False:
            for i in range(10):
                print('nop')
        fnty = ir.FunctionType(ll_status, [ll_dict_type, ll_bytes, ll_bytes])
        [d] = args
        [td] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_popitem')
        dm_key = context.data_model_manager[td.key_type]
        dm_val = context.data_model_manager[td.value_type]
        ptr_key = cgutils.alloca_once(builder, dm_key.get_data_type())
        ptr_val = cgutils.alloca_once(builder, dm_val.get_data_type())
        dp = _container_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, _as_bytes(builder, ptr_key), _as_bytes(builder, ptr_val)])
        out = context.make_optional_none(builder, keyvalty)
        pout = cgutils.alloca_once_value(builder, out)
        cond = builder.icmp_signed('==', status, status.type(int(Status.OK)))
        with builder.if_then(cond):
            key = dm_key.load_from_data_pointer(builder, ptr_key)
            val = dm_val.load_from_data_pointer(builder, ptr_val)
            keyval = context.make_tuple(builder, keyvalty, [key, val])
            optkeyval = context.make_optional_value(builder, keyvalty, keyval)
            builder.store(optkeyval, pout)
        out = builder.load(pout)
        return cgutils.pack_struct(builder, [status, out])
    return (sig, codegen)

@intrinsic
def _dict_delitem(typingctx, d, hk, ix):
    if False:
        return 10
    'Wrap numba_dict_delitem\n    '
    resty = types.int32
    sig = resty(d, hk, types.intp)

    def codegen(context, builder, sig, args):
        if False:
            i = 10
            return i + 15
        fnty = ir.FunctionType(ll_status, [ll_dict_type, ll_hash, ll_ssize_t])
        [d, hk, ix] = args
        [td, thk, tix] = sig.args
        fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_delitem')
        dp = _container_get_data(context, builder, td, d)
        status = builder.call(fn, [dp, hk, ix])
        return status
    return (sig, codegen)

def _iterator_codegen(resty):
    if False:
        for i in range(10):
            print('nop')
    'The common codegen for iterator intrinsics.\n\n    Populates the iterator struct and increfs.\n    '

    def codegen(context, builder, sig, args):
        if False:
            i = 10
            return i + 15
        [d] = args
        [td] = sig.args
        iterhelper = context.make_helper(builder, resty)
        iterhelper.parent = d
        iterhelper.state = iterhelper.state.type(None)
        return impl_ret_borrowed(context, builder, resty, iterhelper._getvalue())
    return codegen

@intrinsic
def _dict_items(typingctx, d):
    if False:
        while True:
            i = 10
    'Get dictionary iterator for .items()'
    resty = types.DictItemsIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return (sig, codegen)

@intrinsic
def _dict_keys(typingctx, d):
    if False:
        while True:
            i = 10
    'Get dictionary iterator for .keys()'
    resty = types.DictKeysIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return (sig, codegen)

@intrinsic
def _dict_values(typingctx, d):
    if False:
        while True:
            i = 10
    'Get dictionary iterator for .values()'
    resty = types.DictValuesIterableType(d)
    sig = resty(d)
    codegen = _iterator_codegen(resty)
    return (sig, codegen)

@intrinsic
def _make_dict(typingctx, keyty, valty, ptr):
    if False:
        i = 10
        return i + 15
    'Make a dictionary struct with the given *ptr*\n\n    Parameters\n    ----------\n    keyty, valty: Type\n        Type of the key and value, respectively.\n    ptr : llvm pointer value\n        Points to the dictionary object.\n    '
    dict_ty = types.DictType(keyty.instance_type, valty.instance_type)

    def codegen(context, builder, signature, args):
        if False:
            print('Hello World!')
        [_, _, ptr] = args
        ctor = cgutils.create_struct_proxy(dict_ty)
        dstruct = ctor(context, builder)
        dstruct.data = ptr
        alloc_size = context.get_abi_sizeof(context.get_value_type(types.voidptr))
        dtor = _imp_dtor(context, builder.module)
        meminfo = context.nrt.meminfo_alloc_dtor(builder, context.get_constant(types.uintp, alloc_size), dtor)
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, ll_dict_type.as_pointer())
        builder.store(ptr, data_pointer)
        dstruct.meminfo = meminfo
        return dstruct._getvalue()
    sig = dict_ty(keyty, valty, ptr)
    return (sig, codegen)

@overload(new_dict)
def impl_new_dict(key, value, n_keys=0):
    if False:
        for i in range(10):
            print('nop')
    'Creates a new dictionary with *key* and *value* as the type\n    of the dictionary key and value, respectively. *n_keys* is the\n    number of keys to insert without requiring a resize, where a\n    value of 0 creates a dictionary with minimum size.\n    '
    if any([not isinstance(key, Type), not isinstance(value, Type)]):
        raise TypeError('expecting *key* and *value* to be a numba Type')
    (keyty, valty) = (key, value)

    def imp(key, value, n_keys=0):
        if False:
            i = 10
            return i + 15
        if n_keys < 0:
            raise RuntimeError('expecting *n_keys* to be >= 0')
        dp = _dict_new_sized(n_keys, keyty, valty)
        _dict_set_method_table(dp, keyty, valty)
        d = _make_dict(keyty, valty, dp)
        return d
    return imp

@overload(len)
def impl_len(d):
    if False:
        i = 10
        return i + 15
    'len(dict)\n    '
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        if False:
            while True:
                i = 10
        return _dict_length(d)
    return impl

@overload(len)
def impl_len_iters(d):
    if False:
        print('Hello World!')
    'len(dict.keys()), len(dict.values()), len(dict.items())\n    '
    if not isinstance(d, (DictKeysIterableType, DictValuesIterableType, DictItemsIterableType)):
        return

    def impl(d):
        if False:
            print('Hello World!')
        return _dict_length(d._parent)
    return impl

@overload_method(types.DictType, '__setitem__')
@overload(operator.setitem)
def impl_setitem(d, key, value):
    if False:
        return 10
    if not isinstance(d, types.DictType):
        return
    (keyty, valty) = (d.key_type, d.value_type)

    def impl(d, key, value):
        if False:
            i = 10
            return i + 15
        castedkey = _cast(key, keyty)
        castedval = _cast(value, valty)
        status = _dict_insert(d, castedkey, hash(castedkey), castedval)
        if status == Status.OK:
            return
        elif status == Status.OK_REPLACED:
            return
        elif status == Status.ERR_CMP_FAILED:
            raise ValueError('key comparison failed')
        else:
            raise RuntimeError('dict.__setitem__ failed unexpectedly')
    if d.is_precise():
        return impl
    else:
        d = d.refine(key, value)
        (keyty, valty) = (d.key_type, d.value_type)
        sig = typing.signature(types.void, d, keyty, valty)
        return (sig, impl)

@overload_method(types.DictType, 'get')
def impl_get(dct, key, default=None):
    if False:
        return 10
    if not isinstance(dct, types.DictType):
        return
    keyty = dct.key_type
    valty = dct.value_type
    _sentry_safe_cast_default(default, valty)

    def impl(dct, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        castedkey = _cast(key, keyty)
        (ix, val) = _dict_lookup(dct, castedkey, hash(castedkey))
        if ix > DKIX.EMPTY:
            return val
        return default
    return impl

@overload_attribute(types.DictType, '__hash__')
def impl_hash(dct):
    if False:
        return 10
    if not isinstance(dct, types.DictType):
        return
    return lambda dct: None

@overload(operator.getitem)
def impl_getitem(d, key):
    if False:
        i = 10
        return i + 15
    if not isinstance(d, types.DictType):
        return
    keyty = d.key_type

    def impl(d, key):
        if False:
            for i in range(10):
                print('nop')
        castedkey = _cast(key, keyty)
        (ix, val) = _dict_lookup(d, castedkey, hash(castedkey))
        if ix == DKIX.EMPTY:
            raise KeyError()
        elif ix < DKIX.EMPTY:
            raise AssertionError('internal dict error during lookup')
        else:
            return _nonoptional(val)
    return impl

@overload_method(types.DictType, 'popitem')
def impl_popitem(d):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        if False:
            for i in range(10):
                print('nop')
        (status, keyval) = _dict_popitem(d)
        if status == Status.OK:
            return _nonoptional(keyval)
        elif status == Status.ERR_DICT_EMPTY:
            raise KeyError()
        else:
            raise AssertionError('internal dict error during popitem')
    return impl

@overload_method(types.DictType, 'pop')
def impl_pop(dct, key, default=None):
    if False:
        return 10
    if not isinstance(dct, types.DictType):
        return
    keyty = dct.key_type
    valty = dct.value_type
    should_raise = isinstance(default, types.Omitted)
    _sentry_safe_cast_default(default, valty)

    def impl(dct, key, default=None):
        if False:
            while True:
                i = 10
        castedkey = _cast(key, keyty)
        hashed = hash(castedkey)
        (ix, val) = _dict_lookup(dct, castedkey, hashed)
        if ix == DKIX.EMPTY:
            if should_raise:
                raise KeyError()
            else:
                return default
        elif ix < DKIX.EMPTY:
            raise AssertionError('internal dict error during lookup')
        else:
            status = _dict_delitem(dct, hashed, ix)
            if status != Status.OK:
                raise AssertionError('internal dict error during delitem')
            return val
    return impl

@overload(operator.delitem)
def impl_delitem(d, k):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(d, types.DictType):
        return

    def impl(d, k):
        if False:
            return 10
        d.pop(k)
    return impl

@overload(operator.contains)
def impl_contains(d, k):
    if False:
        return 10
    if not isinstance(d, types.DictType):
        return
    keyty = d.key_type

    def impl(d, k):
        if False:
            i = 10
            return i + 15
        k = _cast(k, keyty)
        (ix, val) = _dict_lookup(d, k, hash(k))
        return ix > DKIX.EMPTY
    return impl

@overload_method(types.DictType, 'clear')
def impl_clear(d):
    if False:
        i = 10
        return i + 15
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        if False:
            for i in range(10):
                print('nop')
        while len(d):
            d.popitem()
    return impl

@overload_method(types.DictType, 'copy')
def impl_copy(d):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(d, types.DictType):
        return
    (key_type, val_type) = (d.key_type, d.value_type)

    def impl(d):
        if False:
            print('Hello World!')
        newd = new_dict(key_type, val_type, n_keys=len(d))
        for (k, v) in d.items():
            newd[k] = v
        return newd
    return impl

@overload_method(types.DictType, 'setdefault')
def impl_setdefault(dct, key, default=None):
    if False:
        while True:
            i = 10
    if not isinstance(dct, types.DictType):
        return

    def impl(dct, key, default=None):
        if False:
            while True:
                i = 10
        if key not in dct:
            dct[key] = default
        return dct[key]
    return impl

@overload_method(types.DictType, 'items')
def impl_items(d):
    if False:
        while True:
            i = 10
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        if False:
            for i in range(10):
                print('nop')
        it = _dict_items(d)
        return it
    return impl

@overload_method(types.DictType, 'keys')
def impl_keys(d):
    if False:
        return 10
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        if False:
            while True:
                i = 10
        return _dict_keys(d)
    return impl

@overload_method(types.DictType, 'values')
def impl_values(d):
    if False:
        return 10
    if not isinstance(d, types.DictType):
        return

    def impl(d):
        if False:
            i = 10
            return i + 15
        return _dict_values(d)
    return impl

@overload_method(types.DictType, 'update')
def ol_dict_update(d, other):
    if False:
        while True:
            i = 10
    if not isinstance(d, types.DictType):
        return
    if not isinstance(other, types.DictType):
        return

    def impl(d, other):
        if False:
            while True:
                i = 10
        for (k, v) in other.items():
            d[k] = v
    return impl

@overload(operator.eq)
def impl_equal(da, db):
    if False:
        while True:
            i = 10
    if not isinstance(da, types.DictType):
        return
    if not isinstance(db, types.DictType):

        def impl_type_mismatch(da, db):
            if False:
                while True:
                    i = 10
            return False
        return impl_type_mismatch
    otherkeyty = db.key_type

    def impl_type_matched(da, db):
        if False:
            return 10
        if len(da) != len(db):
            return False
        for (ka, va) in da.items():
            kb = _cast(ka, otherkeyty)
            (ix, vb) = _dict_lookup(db, kb, hash(kb))
            if ix <= DKIX.EMPTY:
                return False
            if va != vb:
                return False
        return True
    return impl_type_matched

@overload(operator.ne)
def impl_not_equal(da, db):
    if False:
        print('Hello World!')
    if not isinstance(da, types.DictType):
        return

    def impl(da, db):
        if False:
            print('Hello World!')
        return not da == db
    return impl

@lower_builtin('getiter', types.DictItemsIterableType)
@lower_builtin('getiter', types.DictKeysIterableType)
@lower_builtin('getiter', types.DictValuesIterableType)
def impl_iterable_getiter(context, builder, sig, args):
    if False:
        print('Hello World!')
    'Implement iter() for .keys(), .values(), .items()\n    '
    iterablety = sig.args[0]
    it = context.make_helper(builder, iterablety.iterator_type, args[0])
    fnty = ir.FunctionType(ir.VoidType(), [ll_dictiter_type, ll_dict_type])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_iter')
    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    dictiter_sizeof = proto(_helperlib.c_helpers['dict_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), dictiter_sizeof())
    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)
    dp = _container_get_data(context, builder, iterablety.parent, it.parent)
    builder.call(fn, [it.state, dp])
    return impl_ret_borrowed(context, builder, sig.return_type, it._getvalue())

@lower_builtin('getiter', types.DictType)
def impl_dict_getiter(context, builder, sig, args):
    if False:
        return 10
    'Implement iter(Dict).  Semantically equivalent to dict.keys()\n    '
    [td] = sig.args
    [d] = args
    iterablety = types.DictKeysIterableType(td)
    it = context.make_helper(builder, iterablety.iterator_type)
    fnty = ir.FunctionType(ir.VoidType(), [ll_dictiter_type, ll_dict_type])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_iter')
    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    dictiter_sizeof = proto(_helperlib.c_helpers['dict_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), dictiter_sizeof())
    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)
    it.parent = d
    dp = _container_get_data(context, builder, iterablety.parent, args[0])
    builder.call(fn, [it.state, dp])
    return impl_ret_borrowed(context, builder, sig.return_type, it._getvalue())

@lower_builtin('iternext', types.DictIteratorType)
@iternext_impl(RefType.BORROWED)
def impl_iterator_iternext(context, builder, sig, args, result):
    if False:
        return 10
    iter_type = sig.args[0]
    it = context.make_helper(builder, iter_type, args[0])
    p2p_bytes = ll_bytes.as_pointer()
    iternext_fnty = ir.FunctionType(ll_status, [ll_bytes, p2p_bytes, p2p_bytes])
    iternext = cgutils.get_or_insert_function(builder.module, iternext_fnty, 'numba_dict_iter_next')
    key_raw_ptr = cgutils.alloca_once(builder, ll_bytes)
    val_raw_ptr = cgutils.alloca_once(builder, ll_bytes)
    status = builder.call(iternext, (it.state, key_raw_ptr, val_raw_ptr))
    is_valid = builder.icmp_unsigned('==', status, status.type(0))
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        yield_type = iter_type.yield_type
        (key_ty, val_ty) = iter_type.parent.keyvalue_type
        dm_key = context.data_model_manager[key_ty]
        dm_val = context.data_model_manager[val_ty]
        key_ptr = builder.bitcast(builder.load(key_raw_ptr), dm_key.get_data_type().as_pointer())
        val_ptr = builder.bitcast(builder.load(val_raw_ptr), dm_val.get_data_type().as_pointer())
        key = dm_key.load_from_data_pointer(builder, key_ptr)
        val = dm_val.load_from_data_pointer(builder, val_ptr)
        if isinstance(iter_type.iterable, DictItemsIterableType):
            tup = context.make_tuple(builder, yield_type, [key, val])
            result.yield_(tup)
        elif isinstance(iter_type.iterable, DictKeysIterableType):
            result.yield_(key)
        elif isinstance(iter_type.iterable, DictValuesIterableType):
            result.yield_(val)
        else:
            raise AssertionError('unknown type: {}'.format(iter_type.iterable))

def build_map(context, builder, dict_type, item_types, items):
    if False:
        i = 10
        return i + 15
    if isinstance(dict_type, types.LiteralStrKeyDict):
        unliteral_tys = [x for x in dict_type.literal_value.values()]
        nbty = types.NamedTuple(unliteral_tys, dict_type.tuple_ty)
        values = [x[1] for x in items]
        tup = context.get_constant_undef(nbty)
        literal_tys = [x for x in dict_type.literal_value.values()]
        value_index = dict_type.value_index
        if value_index is None:
            value_indexer = range(len(values))
        else:
            value_indexer = value_index.values()
        for (i, ix) in enumerate(value_indexer):
            val = values[ix]
            casted = context.cast(builder, val, literal_tys[i], unliteral_tys[i])
            tup = builder.insert_value(tup, casted, i)
        d = tup
        context.nrt.incref(builder, nbty, d)
    else:
        from numba.typed import Dict
        dt = types.DictType(dict_type.key_type, dict_type.value_type)
        (kt, vt) = (dict_type.key_type, dict_type.value_type)
        sig = typing.signature(dt)

        def make_dict():
            if False:
                while True:
                    i = 10
            return Dict.empty(kt, vt)
        d = context.compile_internal(builder, make_dict, sig, ())
        if items:
            for ((kt, vt), (k, v)) in zip(item_types, items):
                sig = typing.signature(types.void, dt, kt, vt)
                args = (d, k, v)

                def put(d, k, v):
                    if False:
                        i = 10
                        return i + 15
                    d[k] = v
                context.compile_internal(builder, put, sig, args)
    return d

@intrinsic
def _mixed_values_to_tuple(tyctx, d):
    if False:
        while True:
            i = 10
    keys = [x for x in d.literal_value.keys()]
    literal_tys = [x for x in d.literal_value.values()]

    def impl(cgctx, builder, sig, args):
        if False:
            while True:
                i = 10
        (lld,) = args
        impl = cgctx.get_function('static_getitem', types.none(d, types.literal('dummy')))
        items = []
        for k in range(len(keys)):
            item = impl(builder, (lld, k))
            casted = cgctx.cast(builder, item, literal_tys[k], d.types[k])
            items.append(casted)
            cgctx.nrt.incref(builder, d.types[k], item)
        ret = cgctx.make_tuple(builder, sig.return_type, items)
        return ret
    sig = types.Tuple(d.types)(d)
    return (sig, impl)

@overload_method(types.LiteralStrKeyDict, 'values')
def literalstrkeydict_impl_values(d):
    if False:
        i = 10
        return i + 15
    if not isinstance(d, types.LiteralStrKeyDict):
        return

    def impl(d):
        if False:
            print('Hello World!')
        return _mixed_values_to_tuple(d)
    return impl

@overload_method(types.LiteralStrKeyDict, 'keys')
def literalstrkeydict_impl_keys(d):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(d, types.LiteralStrKeyDict):
        return
    t = tuple([x.literal_value for x in d.literal_value.keys()])

    def impl(d):
        if False:
            i = 10
            return i + 15
        d = dict()
        for x in t:
            d[x] = 0
        return d.keys()
    return impl

@lower_builtin(operator.eq, types.LiteralStrKeyDict, types.LiteralStrKeyDict)
def literalstrkeydict_impl_equals(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    (tu, tv) = sig.args
    (u, v) = args
    pred = tu.literal_value == tv.literal_value
    res = context.get_constant(types.boolean, pred)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@overload(operator.getitem)
@overload_method(types.LiteralStrKeyDict, 'get')
def literalstrkeydict_impl_get(dct, *args):
    if False:
        print('Hello World!')
    if not isinstance(dct, types.LiteralStrKeyDict):
        return
    msg = 'Cannot get{item}() on a literal dictionary, return type cannot be statically determined'
    raise TypingError(msg)

@overload_method(types.LiteralStrKeyDict, 'copy')
def literalstrkeydict_impl_copy(d):
    if False:
        return 10
    if not isinstance(d, types.LiteralStrKeyDict):
        return

    def impl(d):
        if False:
            i = 10
            return i + 15
        return d
    return impl

@intrinsic
def _str_items_mixed_values_to_tuple(tyctx, d):
    if False:
        for i in range(10):
            print('nop')
    keys = [x for x in d.literal_value.keys()]
    literal_tys = [x for x in d.literal_value.values()]

    def impl(cgctx, builder, sig, args):
        if False:
            return 10
        (lld,) = args
        impl = cgctx.get_function('static_getitem', types.none(d, types.literal('dummy')))
        items = []
        from numba.cpython.unicode import make_string_from_constant
        for k in range(len(keys)):
            item = impl(builder, (lld, k))
            casted = cgctx.cast(builder, item, literal_tys[k], d.types[k])
            cgctx.nrt.incref(builder, d.types[k], item)
            keydata = make_string_from_constant(cgctx, builder, types.unicode_type, keys[k].literal_value)
            pair = cgctx.make_tuple(builder, types.Tuple([types.unicode_type, d.types[k]]), (keydata, casted))
            items.append(pair)
        ret = cgctx.make_tuple(builder, sig.return_type, items)
        return ret
    kvs = [types.Tuple((types.unicode_type, x)) for x in d.types]
    sig = types.Tuple(kvs)(d)
    return (sig, impl)

@overload_method(types.LiteralStrKeyDict, 'items')
def literalstrkeydict_impl_items(d):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(d, types.LiteralStrKeyDict):
        return

    def impl(d):
        if False:
            for i in range(10):
                print('nop')
        return _str_items_mixed_values_to_tuple(d)
    return impl

@overload(operator.contains)
def literalstrkeydict_impl_contains(d, k):
    if False:
        return 10
    if not isinstance(d, types.LiteralStrKeyDict):
        return

    def impl(d, k):
        if False:
            for i in range(10):
                print('nop')
        for key in d.keys():
            if k == key:
                return True
        return False
    return impl

@overload(len)
def literalstrkeydict_impl_len(d):
    if False:
        i = 10
        return i + 15
    if not isinstance(d, types.LiteralStrKeyDict):
        return
    l = d.count
    return lambda d: l

@overload(operator.setitem)
def literalstrkeydict_banned_impl_setitem(d, key, value):
    if False:
        print('Hello World!')
    if not isinstance(d, types.LiteralStrKeyDict):
        return
    raise TypingError('Cannot mutate a literal dictionary')

@overload(operator.delitem)
def literalstrkeydict_banned_impl_delitem(d, k):
    if False:
        return 10
    if not isinstance(d, types.LiteralStrKeyDict):
        return
    raise TypingError('Cannot mutate a literal dictionary')

@overload_method(types.LiteralStrKeyDict, 'popitem')
@overload_method(types.LiteralStrKeyDict, 'pop')
@overload_method(types.LiteralStrKeyDict, 'clear')
@overload_method(types.LiteralStrKeyDict, 'setdefault')
@overload_method(types.LiteralStrKeyDict, 'update')
def literalstrkeydict_banned_impl_mutators(d, *args):
    if False:
        while True:
            i = 10
    if not isinstance(d, types.LiteralStrKeyDict):
        return
    raise TypingError('Cannot mutate a literal dictionary')

@lower_cast(types.LiteralStrKeyDict, types.LiteralStrKeyDict)
def cast_LiteralStrKeyDict_LiteralStrKeyDict(context, builder, fromty, toty, val):
    if False:
        while True:
            i = 10
    for ((k1, v1), (k2, v2)) in zip(fromty.literal_value.items(), toty.literal_value.items()):
        if k1 != k2:
            msg = 'LiteralDictionary keys are not the same {} != {}'
            raise LoweringError(msg.format(k1, k2))
        if context.typing_context.unify_pairs(v1, v2) is None:
            msg = 'LiteralDictionary values cannot by unified, have {} and {}'
            raise LoweringError(msg.format(v1, v2))
    else:
        fromty = types.Tuple(fromty.types)
        toty = types.Tuple(toty.types)
        olditems = cgutils.unpack_tuple(builder, val, len(fromty))
        items = [context.cast(builder, v, f, t) for (v, f, t) in zip(olditems, fromty, toty)]
        return context.make_tuple(builder, toty, items)

@lower_cast(types.DictType, types.DictType)
def cast_DictType_DictType(context, builder, fromty, toty, val):
    if False:
        while True:
            i = 10
    return val