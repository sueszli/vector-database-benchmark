"""
Support for native homogeneous sets.
"""
import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import lower_builtin, lower_cast, iternext_impl, impl_ret_borrowed, impl_ret_new_ref, impl_ret_untracked, for_iter, call_len, RefType
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic

def get_payload_struct(context, builder, set_type, ptr):
    if False:
        while True:
            i = 10
    '\n    Given a set value and type, get its payload structure (as a\n    reference, so that mutations are seen by all).\n    '
    payload_type = types.SetPayload(set_type)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(ptr, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)

def get_entry_size(context, set_type):
    if False:
        i = 10
        return i + 15
    '\n    Return the entry size for the given set type.\n    '
    llty = context.get_data_type(types.SetEntry(set_type))
    return context.get_abi_sizeof(llty)
EMPTY = -1
DELETED = -2
FALLBACK = -43
MINSIZE = 16
LINEAR_PROBES = 3
DEBUG_ALLOCS = False

def get_hash_value(context, builder, typ, value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the hash of the given value.\n    '
    typingctx = context.typing_context
    fnty = typingctx.resolve_value_type(hash)
    sig = fnty.get_call_type(typingctx, (typ,), {})
    fn = context.get_function(fnty, sig)
    h = fn(builder, (value,))
    is_ok = is_hash_used(context, builder, h)
    fallback = ir.Constant(h.type, FALLBACK)
    return builder.select(is_ok, h, fallback)

@intrinsic
def _get_hash_value_intrinsic(typingctx, value):
    if False:
        return 10

    def impl(context, builder, typ, args):
        if False:
            while True:
                i = 10
        return get_hash_value(context, builder, value, args[0])
    fnty = typingctx.resolve_value_type(hash)
    sig = fnty.get_call_type(typingctx, (value,), {})
    return (sig, impl)

def is_hash_empty(context, builder, h):
    if False:
        print('Hello World!')
    '\n    Whether the hash value denotes an empty entry.\n    '
    empty = ir.Constant(h.type, EMPTY)
    return builder.icmp_unsigned('==', h, empty)

def is_hash_deleted(context, builder, h):
    if False:
        return 10
    '\n    Whether the hash value denotes a deleted entry.\n    '
    deleted = ir.Constant(h.type, DELETED)
    return builder.icmp_unsigned('==', h, deleted)

def is_hash_used(context, builder, h):
    if False:
        while True:
            i = 10
    '\n    Whether the hash value denotes an active entry.\n    '
    deleted = ir.Constant(h.type, DELETED)
    return builder.icmp_unsigned('<', h, deleted)

def check_all_set(*args):
    if False:
        return 10
    if not all([isinstance(typ, types.Set) for typ in args]):
        raise TypingError(f'All arguments must be Sets, got {args}')
    if not all([args[0].dtype == s.dtype for s in args]):
        raise TypingError(f'All Sets must be of the same type, got {args}')
SetLoop = collections.namedtuple('SetLoop', ('index', 'entry', 'do_break'))

class _SetPayload(object):

    def __init__(self, context, builder, set_type, ptr):
        if False:
            while True:
                i = 10
        payload = get_payload_struct(context, builder, set_type, ptr)
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._payload = payload
        self._entries = payload._get_ptr_by_name('entries')
        self._ptr = ptr

    @property
    def mask(self):
        if False:
            return 10
        return self._payload.mask

    @mask.setter
    def mask(self, value):
        if False:
            return 10
        self._payload.mask = value

    @property
    def used(self):
        if False:
            for i in range(10):
                print('nop')
        return self._payload.used

    @used.setter
    def used(self, value):
        if False:
            print('Hello World!')
        self._payload.used = value

    @property
    def fill(self):
        if False:
            for i in range(10):
                print('nop')
        return self._payload.fill

    @fill.setter
    def fill(self, value):
        if False:
            while True:
                i = 10
        self._payload.fill = value

    @property
    def finger(self):
        if False:
            for i in range(10):
                print('nop')
        return self._payload.finger

    @finger.setter
    def finger(self, value):
        if False:
            i = 10
            return i + 15
        self._payload.finger = value

    @property
    def dirty(self):
        if False:
            print('Hello World!')
        return self._payload.dirty

    @dirty.setter
    def dirty(self, value):
        if False:
            while True:
                i = 10
        self._payload.dirty = value

    @property
    def entries(self):
        if False:
            return 10
        '\n        A pointer to the start of the entries array.\n        '
        return self._entries

    @property
    def ptr(self):
        if False:
            print('Hello World!')
        '\n        A pointer to the start of the NRT-allocated area.\n        '
        return self._ptr

    def get_entry(self, idx):
        if False:
            while True:
                i = 10
        '\n        Get entry number *idx*.\n        '
        entry_ptr = cgutils.gep(self._builder, self._entries, idx)
        entry = self._context.make_data_helper(self._builder, types.SetEntry(self._ty), ref=entry_ptr)
        return entry

    def _lookup(self, item, h, for_insert=False):
        if False:
            i = 10
            return i + 15
        '\n        Lookup the *item* with the given hash values in the entries.\n\n        Return a (found, entry index) tuple:\n        - If found is true, <entry index> points to the entry containing\n          the item.\n        - If found is false, <entry index> points to the empty entry that\n          the item can be written to (only if *for_insert* is true)\n        '
        context = self._context
        builder = self._builder
        intp_t = h.type
        mask = self.mask
        dtype = self._ty.dtype
        tyctx = context.typing_context
        fnty = tyctx.resolve_value_type(operator.eq)
        sig = fnty.get_call_type(tyctx, (dtype, dtype), {})
        eqfn = context.get_function(fnty, sig)
        one = ir.Constant(intp_t, 1)
        five = ir.Constant(intp_t, 5)
        perturb = cgutils.alloca_once_value(builder, h)
        index = cgutils.alloca_once_value(builder, builder.and_(h, mask))
        if for_insert:
            free_index_sentinel = mask.type(-1)
            free_index = cgutils.alloca_once_value(builder, free_index_sentinel)
        bb_body = builder.append_basic_block('lookup.body')
        bb_found = builder.append_basic_block('lookup.found')
        bb_not_found = builder.append_basic_block('lookup.not_found')
        bb_end = builder.append_basic_block('lookup.end')

        def check_entry(i):
            if False:
                return 10
            '\n            Check entry *i* against the value being searched for.\n            '
            entry = self.get_entry(i)
            entry_hash = entry.hash
            with builder.if_then(builder.icmp_unsigned('==', h, entry_hash)):
                eq = eqfn(builder, (item, entry.key))
                with builder.if_then(eq):
                    builder.branch(bb_found)
            with builder.if_then(is_hash_empty(context, builder, entry_hash)):
                builder.branch(bb_not_found)
            if for_insert:
                with builder.if_then(is_hash_deleted(context, builder, entry_hash)):
                    j = builder.load(free_index)
                    j = builder.select(builder.icmp_unsigned('==', j, free_index_sentinel), i, j)
                    builder.store(j, free_index)
        with cgutils.for_range(builder, ir.Constant(intp_t, LINEAR_PROBES)):
            i = builder.load(index)
            check_entry(i)
            i = builder.add(i, one)
            i = builder.and_(i, mask)
            builder.store(i, index)
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            i = builder.load(index)
            check_entry(i)
            p = builder.load(perturb)
            p = builder.lshr(p, five)
            i = builder.add(one, builder.mul(i, five))
            i = builder.and_(mask, builder.add(i, p))
            builder.store(i, index)
            builder.store(p, perturb)
            builder.branch(bb_body)
        with builder.goto_block(bb_not_found):
            if for_insert:
                i = builder.load(index)
                j = builder.load(free_index)
                i = builder.select(builder.icmp_unsigned('==', j, free_index_sentinel), i, j)
                builder.store(i, index)
            builder.branch(bb_end)
        with builder.goto_block(bb_found):
            builder.branch(bb_end)
        builder.position_at_end(bb_end)
        found = builder.phi(ir.IntType(1), 'found')
        found.add_incoming(cgutils.true_bit, bb_found)
        found.add_incoming(cgutils.false_bit, bb_not_found)
        return (found, builder.load(index))

    @contextlib.contextmanager
    def _iterate(self, start=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Iterate over the payload's entries.  Yield a SetLoop.\n        "
        context = self._context
        builder = self._builder
        intp_t = context.get_value_type(types.intp)
        one = ir.Constant(intp_t, 1)
        size = builder.add(self.mask, one)
        with cgutils.for_range(builder, size, start=start) as range_loop:
            entry = self.get_entry(range_loop.index)
            is_used = is_hash_used(context, builder, entry.hash)
            with builder.if_then(is_used):
                loop = SetLoop(index=range_loop.index, entry=entry, do_break=range_loop.do_break)
                yield loop

    @contextlib.contextmanager
    def _next_entry(self):
        if False:
            print('Hello World!')
        "\n        Yield a random entry from the payload.  Caller must ensure the\n        set isn't empty, otherwise the function won't end.\n        "
        context = self._context
        builder = self._builder
        intp_t = context.get_value_type(types.intp)
        zero = ir.Constant(intp_t, 0)
        one = ir.Constant(intp_t, 1)
        mask = self.mask
        bb_body = builder.append_basic_block('next_entry_body')
        bb_end = builder.append_basic_block('next_entry_end')
        index = cgutils.alloca_once_value(builder, self.finger)
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            i = builder.load(index)
            i = builder.and_(mask, builder.add(i, one))
            builder.store(i, index)
            entry = self.get_entry(i)
            is_used = is_hash_used(context, builder, entry.hash)
            builder.cbranch(is_used, bb_end, bb_body)
        builder.position_at_end(bb_end)
        i = builder.load(index)
        self.finger = i
        yield self.get_entry(i)

class SetInstance(object):

    def __init__(self, context, builder, set_type, set_val):
        if False:
            while True:
                i = 10
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._entrysize = get_entry_size(context, set_type)
        self._set = context.make_helper(builder, set_type, set_val)

    @property
    def dtype(self):
        if False:
            for i in range(10):
                print('nop')
        return self._ty.dtype

    @property
    def payload(self):
        if False:
            print('Hello World!')
        '\n        The _SetPayload for this set.\n        '
        context = self._context
        builder = self._builder
        ptr = self._context.nrt.meminfo_data(builder, self.meminfo)
        return _SetPayload(context, builder, self._ty, ptr)

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self._set._getvalue()

    @property
    def meminfo(self):
        if False:
            return 10
        return self._set.meminfo

    @property
    def parent(self):
        if False:
            print('Hello World!')
        return self._set.parent

    @parent.setter
    def parent(self, value):
        if False:
            return 10
        self._set.parent = value

    def get_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the number of elements in the size.\n        '
        return self.payload.used

    def set_dirty(self, val):
        if False:
            return 10
        if self._ty.reflected:
            self.payload.dirty = cgutils.true_bit if val else cgutils.false_bit

    def _add_entry(self, payload, entry, item, h, do_resize=True):
        if False:
            return 10
        context = self._context
        builder = self._builder
        old_hash = entry.hash
        entry.hash = h
        self.incref_value(item)
        entry.key = item
        used = payload.used
        one = ir.Constant(used.type, 1)
        used = payload.used = builder.add(used, one)
        with builder.if_then(is_hash_empty(context, builder, old_hash), likely=True):
            payload.fill = builder.add(payload.fill, one)
        if do_resize:
            self.upsize(used)
        self.set_dirty(True)

    def _add_key(self, payload, item, h, do_resize=True, do_incref=True):
        if False:
            while True:
                i = 10
        context = self._context
        builder = self._builder
        (found, i) = payload._lookup(item, h, for_insert=True)
        not_found = builder.not_(found)
        with builder.if_then(not_found):
            entry = payload.get_entry(i)
            old_hash = entry.hash
            entry.hash = h
            if do_incref:
                self.incref_value(item)
            entry.key = item
            used = payload.used
            one = ir.Constant(used.type, 1)
            used = payload.used = builder.add(used, one)
            with builder.if_then(is_hash_empty(context, builder, old_hash), likely=True):
                payload.fill = builder.add(payload.fill, one)
            if do_resize:
                self.upsize(used)
            self.set_dirty(True)

    def _remove_entry(self, payload, entry, do_resize=True, do_decref=True):
        if False:
            while True:
                i = 10
        entry.hash = ir.Constant(entry.hash.type, DELETED)
        if do_decref:
            self.decref_value(entry.key)
        used = payload.used
        one = ir.Constant(used.type, 1)
        used = payload.used = self._builder.sub(used, one)
        if do_resize:
            self.downsize(used)
        self.set_dirty(True)

    def _remove_key(self, payload, item, h, do_resize=True):
        if False:
            print('Hello World!')
        context = self._context
        builder = self._builder
        (found, i) = payload._lookup(item, h)
        with builder.if_then(found):
            entry = payload.get_entry(i)
            self._remove_entry(payload, entry, do_resize)
        return found

    def add(self, item, do_resize=True):
        if False:
            return 10
        context = self._context
        builder = self._builder
        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        self._add_key(payload, item, h, do_resize)

    def add_pyapi(self, pyapi, item, do_resize=True):
        if False:
            for i in range(10):
                print('nop')
        'A version of .add for use inside functions following Python calling\n        convention.\n        '
        context = self._context
        builder = self._builder
        payload = self.payload
        h = self._pyapi_get_hash_value(pyapi, context, builder, item)
        self._add_key(payload, item, h, do_resize)

    def _pyapi_get_hash_value(self, pyapi, context, builder, item):
        if False:
            return 10
        'Python API compatible version of `get_hash_value()`.\n        '
        argtypes = [self._ty.dtype]
        resty = types.intp

        def wrapper(val):
            if False:
                print('Hello World!')
            return _get_hash_value_intrinsic(val)
        args = [item]
        sig = typing.signature(resty, *argtypes)
        (is_error, retval) = pyapi.call_jit_code(wrapper, sig, args)
        with builder.if_then(is_error, likely=False):
            builder.ret(pyapi.get_null_object())
        return retval

    def contains(self, item):
        if False:
            i = 10
            return i + 15
        context = self._context
        builder = self._builder
        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        (found, i) = payload._lookup(item, h)
        return found

    def discard(self, item):
        if False:
            print('Hello World!')
        context = self._context
        builder = self._builder
        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        found = self._remove_key(payload, item, h)
        return found

    def pop(self):
        if False:
            return 10
        context = self._context
        builder = self._builder
        lty = context.get_value_type(self._ty.dtype)
        key = cgutils.alloca_once(builder, lty)
        payload = self.payload
        with payload._next_entry() as entry:
            builder.store(entry.key, key)
            self._remove_entry(payload, entry, do_decref=False)
        return builder.load(key)

    def clear(self):
        if False:
            while True:
                i = 10
        context = self._context
        builder = self._builder
        intp_t = context.get_value_type(types.intp)
        minsize = ir.Constant(intp_t, MINSIZE)
        self._replace_payload(minsize)
        self.set_dirty(True)

    def copy(self):
        if False:
            return 10
        '\n        Return a copy of this set.\n        '
        context = self._context
        builder = self._builder
        payload = self.payload
        used = payload.used
        fill = payload.fill
        other = type(self)(context, builder, self._ty, None)
        no_deleted_entries = builder.icmp_unsigned('==', used, fill)
        with builder.if_else(no_deleted_entries, likely=True) as (if_no_deleted, if_deleted):
            with if_no_deleted:
                ok = other._copy_payload(payload)
                with builder.if_then(builder.not_(ok), likely=False):
                    context.call_conv.return_user_exc(builder, MemoryError, ('cannot copy set',))
            with if_deleted:
                nentries = self.choose_alloc_size(context, builder, used)
                ok = other._allocate_payload(nentries)
                with builder.if_then(builder.not_(ok), likely=False):
                    context.call_conv.return_user_exc(builder, MemoryError, ('cannot copy set',))
                other_payload = other.payload
                with payload._iterate() as loop:
                    entry = loop.entry
                    other._add_key(other_payload, entry.key, entry.hash, do_resize=False)
        return other

    def intersect(self, other):
        if False:
            return 10
        '\n        In-place intersection with *other* set.\n        '
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        with payload._iterate() as loop:
            entry = loop.entry
            (found, _) = other_payload._lookup(entry.key, entry.hash)
            with builder.if_then(builder.not_(found)):
                self._remove_entry(payload, entry, do_resize=False)
        self.downsize(payload.used)

    def difference(self, other):
        if False:
            return 10
        '\n        In-place difference with *other* set.\n        '
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        with other_payload._iterate() as loop:
            entry = loop.entry
            self._remove_key(payload, entry.key, entry.hash, do_resize=False)
        self.downsize(payload.used)

    def symmetric_difference(self, other):
        if False:
            while True:
                i = 10
        '\n        In-place symmetric difference with *other* set.\n        '
        context = self._context
        builder = self._builder
        other_payload = other.payload
        with other_payload._iterate() as loop:
            key = loop.entry.key
            h = loop.entry.hash
            payload = self.payload
            (found, i) = payload._lookup(key, h, for_insert=True)
            entry = payload.get_entry(i)
            with builder.if_else(found) as (if_common, if_not_common):
                with if_common:
                    self._remove_entry(payload, entry, do_resize=False)
                with if_not_common:
                    self._add_entry(payload, entry, key, h)
        self.downsize(self.payload.used)

    def issubset(self, other, strict=False):
        if False:
            i = 10
            return i + 15
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        cmp_op = '<' if strict else '<='
        res = cgutils.alloca_once_value(builder, cgutils.true_bit)
        with builder.if_else(builder.icmp_unsigned(cmp_op, payload.used, other_payload.used)) as (if_smaller, if_larger):
            with if_larger:
                builder.store(cgutils.false_bit, res)
            with if_smaller:
                with payload._iterate() as loop:
                    entry = loop.entry
                    (found, _) = other_payload._lookup(entry.key, entry.hash)
                    with builder.if_then(builder.not_(found)):
                        builder.store(cgutils.false_bit, res)
                        loop.do_break()
        return builder.load(res)

    def isdisjoint(self, other):
        if False:
            print('Hello World!')
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        res = cgutils.alloca_once_value(builder, cgutils.true_bit)

        def check(smaller, larger):
            if False:
                for i in range(10):
                    print('nop')
            with smaller._iterate() as loop:
                entry = loop.entry
                (found, _) = larger._lookup(entry.key, entry.hash)
                with builder.if_then(found):
                    builder.store(cgutils.false_bit, res)
                    loop.do_break()
        with builder.if_else(builder.icmp_unsigned('>', payload.used, other_payload.used)) as (if_larger, otherwise):
            with if_larger:
                check(other_payload, payload)
            with otherwise:
                check(payload, other_payload)
        return builder.load(res)

    def equals(self, other):
        if False:
            for i in range(10):
                print('nop')
        context = self._context
        builder = self._builder
        payload = self.payload
        other_payload = other.payload
        res = cgutils.alloca_once_value(builder, cgutils.true_bit)
        with builder.if_else(builder.icmp_unsigned('==', payload.used, other_payload.used)) as (if_same_size, otherwise):
            with if_same_size:
                with payload._iterate() as loop:
                    entry = loop.entry
                    (found, _) = other_payload._lookup(entry.key, entry.hash)
                    with builder.if_then(builder.not_(found)):
                        builder.store(cgutils.false_bit, res)
                        loop.do_break()
            with otherwise:
                builder.store(cgutils.false_bit, res)
        return builder.load(res)

    @classmethod
    def allocate_ex(cls, context, builder, set_type, nitems=None):
        if False:
            i = 10
            return i + 15
        "\n        Allocate a SetInstance with its storage.\n        Return a (ok, instance) tuple where *ok* is a LLVM boolean and\n        *instance* is a SetInstance object (the object's contents are\n        only valid when *ok* is true).\n        "
        intp_t = context.get_value_type(types.intp)
        if nitems is None:
            nentries = ir.Constant(intp_t, MINSIZE)
        else:
            if isinstance(nitems, int):
                nitems = ir.Constant(intp_t, nitems)
            nentries = cls.choose_alloc_size(context, builder, nitems)
        self = cls(context, builder, set_type, None)
        ok = self._allocate_payload(nentries)
        return (ok, self)

    @classmethod
    def allocate(cls, context, builder, set_type, nitems=None):
        if False:
            return 10
        "\n        Allocate a SetInstance with its storage.  Same as allocate_ex(),\n        but return an initialized *instance*.  If allocation failed,\n        control is transferred to the caller using the target's current\n        call convention.\n        "
        (ok, self) = cls.allocate_ex(context, builder, set_type, nitems)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, ('cannot allocate set',))
        return self

    @classmethod
    def from_meminfo(cls, context, builder, set_type, meminfo):
        if False:
            i = 10
            return i + 15
        '\n        Allocate a new set instance pointing to an existing payload\n        (a meminfo pointer).\n        Note the parent field has to be filled by the caller.\n        '
        self = cls(context, builder, set_type, None)
        self._set.meminfo = meminfo
        self._set.parent = context.get_constant_null(types.pyobject)
        context.nrt.incref(builder, set_type, self.value)
        return self

    @classmethod
    def choose_alloc_size(cls, context, builder, nitems):
        if False:
            i = 10
            return i + 15
        '\n        Choose a suitable number of entries for the given number of items.\n        '
        intp_t = nitems.type
        one = ir.Constant(intp_t, 1)
        minsize = ir.Constant(intp_t, MINSIZE)
        min_entries = builder.shl(nitems, one)
        size_p = cgutils.alloca_once_value(builder, minsize)
        bb_body = builder.append_basic_block('calcsize.body')
        bb_end = builder.append_basic_block('calcsize.end')
        builder.branch(bb_body)
        with builder.goto_block(bb_body):
            size = builder.load(size_p)
            is_large_enough = builder.icmp_unsigned('>=', size, min_entries)
            with builder.if_then(is_large_enough, likely=False):
                builder.branch(bb_end)
            next_size = builder.shl(size, one)
            builder.store(next_size, size_p)
            builder.branch(bb_body)
        builder.position_at_end(bb_end)
        return builder.load(size_p)

    def upsize(self, nitems):
        if False:
            print('Hello World!')
        '\n        When adding to the set, ensure it is properly sized for the given\n        number of used entries.\n        '
        context = self._context
        builder = self._builder
        intp_t = nitems.type
        one = ir.Constant(intp_t, 1)
        two = ir.Constant(intp_t, 2)
        payload = self.payload
        min_entries = builder.shl(nitems, one)
        size = builder.add(payload.mask, one)
        need_resize = builder.icmp_unsigned('>=', min_entries, size)
        with builder.if_then(need_resize, likely=False):
            new_size_p = cgutils.alloca_once_value(builder, size)
            bb_body = builder.append_basic_block('calcsize.body')
            bb_end = builder.append_basic_block('calcsize.end')
            builder.branch(bb_body)
            with builder.goto_block(bb_body):
                new_size = builder.load(new_size_p)
                new_size = builder.shl(new_size, two)
                builder.store(new_size, new_size_p)
                is_too_small = builder.icmp_unsigned('>=', min_entries, new_size)
                builder.cbranch(is_too_small, bb_body, bb_end)
            builder.position_at_end(bb_end)
            new_size = builder.load(new_size_p)
            if DEBUG_ALLOCS:
                context.printf(builder, 'upsize to %zd items: current size = %zd, min entries = %zd, new size = %zd\n', nitems, size, min_entries, new_size)
            self._resize(payload, new_size, 'cannot grow set')

    def downsize(self, nitems):
        if False:
            return 10
        '\n        When removing from the set, ensure it is properly sized for the given\n        number of used entries.\n        '
        context = self._context
        builder = self._builder
        intp_t = nitems.type
        one = ir.Constant(intp_t, 1)
        two = ir.Constant(intp_t, 2)
        minsize = ir.Constant(intp_t, MINSIZE)
        payload = self.payload
        min_entries = builder.shl(nitems, one)
        min_entries = builder.select(builder.icmp_unsigned('>=', min_entries, minsize), min_entries, minsize)
        max_size = builder.shl(min_entries, two)
        size = builder.add(payload.mask, one)
        need_resize = builder.and_(builder.icmp_unsigned('<=', max_size, size), builder.icmp_unsigned('<', minsize, size))
        with builder.if_then(need_resize, likely=False):
            new_size_p = cgutils.alloca_once_value(builder, size)
            bb_body = builder.append_basic_block('calcsize.body')
            bb_end = builder.append_basic_block('calcsize.end')
            builder.branch(bb_body)
            with builder.goto_block(bb_body):
                new_size = builder.load(new_size_p)
                new_size = builder.lshr(new_size, one)
                is_too_small = builder.icmp_unsigned('>', min_entries, new_size)
                with builder.if_then(is_too_small):
                    builder.branch(bb_end)
                builder.store(new_size, new_size_p)
                builder.branch(bb_body)
            builder.position_at_end(bb_end)
            new_size = builder.load(new_size_p)
            if DEBUG_ALLOCS:
                context.printf(builder, 'downsize to %zd items: current size = %zd, min entries = %zd, new size = %zd\n', nitems, size, min_entries, new_size)
            self._resize(payload, new_size, 'cannot shrink set')

    def _resize(self, payload, nentries, errmsg):
        if False:
            while True:
                i = 10
        '\n        Resize the payload to the given number of entries.\n\n        CAUTION: *nentries* must be a power of 2!\n        '
        context = self._context
        builder = self._builder
        old_payload = payload
        ok = self._allocate_payload(nentries, realloc=True)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, (errmsg,))
        payload = self.payload
        with old_payload._iterate() as loop:
            entry = loop.entry
            self._add_key(payload, entry.key, entry.hash, do_resize=False, do_incref=False)
        self._free_payload(old_payload.ptr)

    def _replace_payload(self, nentries):
        if False:
            i = 10
            return i + 15
        '\n        Replace the payload with a new empty payload with the given number\n        of entries.\n\n        CAUTION: *nentries* must be a power of 2!\n        '
        context = self._context
        builder = self._builder
        with self.payload._iterate() as loop:
            entry = loop.entry
            self.decref_value(entry.key)
        self._free_payload(self.payload.ptr)
        ok = self._allocate_payload(nentries, realloc=True)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, ('cannot reallocate set',))

    def _allocate_payload(self, nentries, realloc=False):
        if False:
            print('Hello World!')
        '\n        Allocate and initialize payload for the given number of entries.\n        If *realloc* is True, the existing meminfo is reused.\n\n        CAUTION: *nentries* must be a power of 2!\n        '
        context = self._context
        builder = self._builder
        ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
        intp_t = context.get_value_type(types.intp)
        zero = ir.Constant(intp_t, 0)
        one = ir.Constant(intp_t, 1)
        payload_type = context.get_data_type(types.SetPayload(self._ty))
        payload_size = context.get_abi_sizeof(payload_type)
        entry_size = self._entrysize
        payload_size -= entry_size
        (allocsize, ovf) = cgutils.muladd_with_overflow(builder, nentries, ir.Constant(intp_t, entry_size), ir.Constant(intp_t, payload_size))
        with builder.if_then(ovf, likely=False):
            builder.store(cgutils.false_bit, ok)
        with builder.if_then(builder.load(ok), likely=True):
            if realloc:
                meminfo = self._set.meminfo
                ptr = context.nrt.meminfo_varsize_alloc_unchecked(builder, meminfo, size=allocsize)
                alloc_ok = cgutils.is_null(builder, ptr)
            else:
                dtor = self._imp_dtor(context, builder.module)
                meminfo = context.nrt.meminfo_new_varsize_dtor_unchecked(builder, allocsize, builder.bitcast(dtor, cgutils.voidptr_t))
                alloc_ok = cgutils.is_null(builder, meminfo)
            with builder.if_else(alloc_ok, likely=False) as (if_error, if_ok):
                with if_error:
                    builder.store(cgutils.false_bit, ok)
                with if_ok:
                    if not realloc:
                        self._set.meminfo = meminfo
                        self._set.parent = context.get_constant_null(types.pyobject)
                    payload = self.payload
                    cgutils.memset(builder, payload.ptr, allocsize, 255)
                    payload.used = zero
                    payload.fill = zero
                    payload.finger = zero
                    new_mask = builder.sub(nentries, one)
                    payload.mask = new_mask
                    if DEBUG_ALLOCS:
                        context.printf(builder, 'allocated %zd bytes for set at %p: mask = %zd\n', allocsize, payload.ptr, new_mask)
        return builder.load(ok)

    def _free_payload(self, ptr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Free an allocated old payload at *ptr*.\n        '
        self._context.nrt.meminfo_varsize_free(self._builder, self.meminfo, ptr)

    def _copy_payload(self, src_payload):
        if False:
            for i in range(10):
                print('nop')
        '\n        Raw-copy the given payload into self.\n        '
        context = self._context
        builder = self._builder
        ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
        intp_t = context.get_value_type(types.intp)
        zero = ir.Constant(intp_t, 0)
        one = ir.Constant(intp_t, 1)
        payload_type = context.get_data_type(types.SetPayload(self._ty))
        payload_size = context.get_abi_sizeof(payload_type)
        entry_size = self._entrysize
        payload_size -= entry_size
        mask = src_payload.mask
        nentries = builder.add(one, mask)
        allocsize = builder.add(ir.Constant(intp_t, payload_size), builder.mul(ir.Constant(intp_t, entry_size), nentries))
        with builder.if_then(builder.load(ok), likely=True):
            dtor = self._imp_dtor(context, builder.module)
            meminfo = context.nrt.meminfo_new_varsize_dtor_unchecked(builder, allocsize, builder.bitcast(dtor, cgutils.voidptr_t))
            alloc_ok = cgutils.is_null(builder, meminfo)
            with builder.if_else(alloc_ok, likely=False) as (if_error, if_ok):
                with if_error:
                    builder.store(cgutils.false_bit, ok)
                with if_ok:
                    self._set.meminfo = meminfo
                    payload = self.payload
                    payload.used = src_payload.used
                    payload.fill = src_payload.fill
                    payload.finger = zero
                    payload.mask = mask
                    cgutils.raw_memcpy(builder, payload.entries, src_payload.entries, nentries, entry_size)
                    with src_payload._iterate() as loop:
                        self.incref_value(loop.entry.key)
                    if DEBUG_ALLOCS:
                        context.printf(builder, 'allocated %zd bytes for set at %p: mask = %zd\n', allocsize, payload.ptr, mask)
        return builder.load(ok)

    def _imp_dtor(self, context, module):
        if False:
            i = 10
            return i + 15
        'Define the dtor for set\n        '
        llvoidptr = cgutils.voidptr_t
        llsize_t = context.get_value_type(types.size_t)
        fnty = ir.FunctionType(ir.VoidType(), [llvoidptr, llsize_t, llvoidptr])
        fname = f'.dtor.set.{self._ty.dtype}'
        fn = cgutils.get_or_insert_function(module, fnty, name=fname)
        if fn.is_declaration:
            fn.linkage = 'linkonce_odr'
            builder = ir.IRBuilder(fn.append_basic_block())
            payload = _SetPayload(context, builder, self._ty, fn.args[0])
            with payload._iterate() as loop:
                entry = loop.entry
                context.nrt.decref(builder, self._ty.dtype, entry.key)
            builder.ret_void()
        return fn

    def incref_value(self, val):
        if False:
            for i in range(10):
                print('nop')
        'Incref an element value\n        '
        self._context.nrt.incref(self._builder, self._ty.dtype, val)

    def decref_value(self, val):
        if False:
            return 10
        'Decref an element value\n        '
        self._context.nrt.decref(self._builder, self._ty.dtype, val)

class SetIterInstance(object):

    def __init__(self, context, builder, iter_type, iter_val):
        if False:
            for i in range(10):
                print('nop')
        self._context = context
        self._builder = builder
        self._ty = iter_type
        self._iter = context.make_helper(builder, iter_type, iter_val)
        ptr = self._context.nrt.meminfo_data(builder, self.meminfo)
        self._payload = _SetPayload(context, builder, self._ty.container, ptr)

    @classmethod
    def from_set(cls, context, builder, iter_type, set_val):
        if False:
            print('Hello World!')
        set_inst = SetInstance(context, builder, iter_type.container, set_val)
        self = cls(context, builder, iter_type, None)
        index = context.get_constant(types.intp, 0)
        self._iter.index = cgutils.alloca_once_value(builder, index)
        self._iter.meminfo = set_inst.meminfo
        return self

    @property
    def value(self):
        if False:
            return 10
        return self._iter._getvalue()

    @property
    def meminfo(self):
        if False:
            while True:
                i = 10
        return self._iter.meminfo

    @property
    def index(self):
        if False:
            print('Hello World!')
        return self._builder.load(self._iter.index)

    @index.setter
    def index(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._builder.store(value, self._iter.index)

    def iternext(self, result):
        if False:
            return 10
        index = self.index
        payload = self._payload
        one = ir.Constant(index.type, 1)
        result.set_exhausted()
        with payload._iterate(start=index) as loop:
            entry = loop.entry
            result.set_valid()
            result.yield_(entry.key)
            self.index = self._builder.add(loop.index, one)
            loop.do_break()

def build_set(context, builder, set_type, items):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build a set of the given type, containing the given items.\n    '
    nitems = len(items)
    inst = SetInstance.allocate(context, builder, set_type, nitems)
    if nitems > 0:
        array = cgutils.pack_array(builder, items)
        array_ptr = cgutils.alloca_once_value(builder, array)
        count = context.get_constant(types.intp, nitems)
        with cgutils.for_range(builder, count) as loop:
            item = builder.load(cgutils.gep(builder, array_ptr, 0, loop.index))
            inst.add(item)
    return impl_ret_new_ref(context, builder, set_type, inst.value)

@lower_builtin(set)
def set_empty_constructor(context, builder, sig, args):
    if False:
        print('Hello World!')
    set_type = sig.return_type
    inst = SetInstance.allocate(context, builder, set_type)
    return impl_ret_new_ref(context, builder, set_type, inst.value)

@lower_builtin(set, types.IterableType)
def set_constructor(context, builder, sig, args):
    if False:
        return 10
    set_type = sig.return_type
    (items_type,) = sig.args
    (items,) = args
    n = call_len(context, builder, items_type, items)
    inst = SetInstance.allocate(context, builder, set_type, n)
    with for_iter(context, builder, items_type, items) as loop:
        inst.add(loop.value)
        context.nrt.decref(builder, set_type.dtype, loop.value)
    return impl_ret_new_ref(context, builder, set_type, inst.value)

@lower_builtin(len, types.Set)
def set_len(context, builder, sig, args):
    if False:
        while True:
            i = 10
    inst = SetInstance(context, builder, sig.args[0], args[0])
    return inst.get_size()

@lower_builtin(operator.contains, types.Set, types.Any)
def in_set(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    inst = SetInstance(context, builder, sig.args[0], args[0])
    return inst.contains(args[1])

@lower_builtin('getiter', types.Set)
def getiter_set(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    inst = SetIterInstance.from_set(context, builder, sig.return_type, args[0])
    return impl_ret_borrowed(context, builder, sig.return_type, inst.value)

@lower_builtin('iternext', types.SetIter)
@iternext_impl(RefType.BORROWED)
def iternext_listiter(context, builder, sig, args, result):
    if False:
        return 10
    inst = SetIterInstance(context, builder, sig.args[0], args[0])
    inst.iternext(result)

@lower_builtin('set.add', types.Set, types.Any)
def set_add(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    inst = SetInstance(context, builder, sig.args[0], args[0])
    item = args[1]
    inst.add(item)
    return context.get_dummy_value()

@intrinsic
def _set_discard(typingctx, s, item):
    if False:
        for i in range(10):
            print('nop')
    sig = types.none(s, item)

    def set_discard(context, builder, sig, args):
        if False:
            return 10
        inst = SetInstance(context, builder, sig.args[0], args[0])
        item = args[1]
        inst.discard(item)
        return context.get_dummy_value()
    return (sig, set_discard)

@overload_method(types.Set, 'discard')
def ol_set_discard(s, item):
    if False:
        for i in range(10):
            print('nop')
    return lambda s, item: _set_discard(s, item)

@intrinsic
def _set_pop(typingctx, s):
    if False:
        while True:
            i = 10
    sig = s.dtype(s)

    def set_pop(context, builder, sig, args):
        if False:
            return 10
        inst = SetInstance(context, builder, sig.args[0], args[0])
        used = inst.payload.used
        with builder.if_then(cgutils.is_null(builder, used), likely=False):
            context.call_conv.return_user_exc(builder, KeyError, ('set.pop(): empty set',))
        return inst.pop()
    return (sig, set_pop)

@overload_method(types.Set, 'pop')
def ol_set_pop(s):
    if False:
        return 10
    return lambda s: _set_pop(s)

@intrinsic
def _set_remove(typingctx, s, item):
    if False:
        while True:
            i = 10
    sig = types.none(s, item)

    def set_remove(context, builder, sig, args):
        if False:
            print('Hello World!')
        inst = SetInstance(context, builder, sig.args[0], args[0])
        item = args[1]
        found = inst.discard(item)
        with builder.if_then(builder.not_(found), likely=False):
            context.call_conv.return_user_exc(builder, KeyError, ('set.remove(): key not in set',))
        return context.get_dummy_value()
    return (sig, set_remove)

@overload_method(types.Set, 'remove')
def ol_set_remove(s, item):
    if False:
        while True:
            i = 10
    if s.dtype == item:
        return lambda s, item: _set_remove(s, item)

@intrinsic
def _set_clear(typingctx, s):
    if False:
        return 10
    sig = types.none(s)

    def set_clear(context, builder, sig, args):
        if False:
            while True:
                i = 10
        inst = SetInstance(context, builder, sig.args[0], args[0])
        inst.clear()
        return context.get_dummy_value()
    return (sig, set_clear)

@overload_method(types.Set, 'clear')
def ol_set_clear(s):
    if False:
        while True:
            i = 10
    return lambda s: _set_clear(s)

@intrinsic
def _set_copy(typingctx, s):
    if False:
        return 10
    sig = s(s)

    def set_copy(context, builder, sig, args):
        if False:
            while True:
                i = 10
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = inst.copy()
        return impl_ret_new_ref(context, builder, sig.return_type, other.value)
    return (sig, set_copy)

@overload_method(types.Set, 'copy')
def ol_set_copy(s):
    if False:
        print('Hello World!')
    return lambda s: _set_copy(s)

def set_difference_update(context, builder, sig, args):
    if False:
        print('Hello World!')
    inst = SetInstance(context, builder, sig.args[0], args[0])
    other = SetInstance(context, builder, sig.args[1], args[1])
    inst.difference(other)
    return context.get_dummy_value()

@intrinsic
def _set_difference_update(typingctx, a, b):
    if False:
        while True:
            i = 10
    sig = types.none(a, b)
    return (sig, set_difference_update)

@overload_method(types.Set, 'difference_update')
def set_difference_update_impl(a, b):
    if False:
        print('Hello World!')
    check_all_set(a, b)
    return lambda a, b: _set_difference_update(a, b)

def set_intersection_update(context, builder, sig, args):
    if False:
        while True:
            i = 10
    inst = SetInstance(context, builder, sig.args[0], args[0])
    other = SetInstance(context, builder, sig.args[1], args[1])
    inst.intersect(other)
    return context.get_dummy_value()

@intrinsic
def _set_intersection_update(typingctx, a, b):
    if False:
        print('Hello World!')
    sig = types.none(a, b)
    return (sig, set_intersection_update)

@overload_method(types.Set, 'intersection_update')
def set_intersection_update_impl(a, b):
    if False:
        for i in range(10):
            print('nop')
    check_all_set(a, b)
    return lambda a, b: _set_intersection_update(a, b)

def set_symmetric_difference_update(context, builder, sig, args):
    if False:
        return 10
    inst = SetInstance(context, builder, sig.args[0], args[0])
    other = SetInstance(context, builder, sig.args[1], args[1])
    inst.symmetric_difference(other)
    return context.get_dummy_value()

@intrinsic
def _set_symmetric_difference_update(typingctx, a, b):
    if False:
        return 10
    sig = types.none(a, b)
    return (sig, set_symmetric_difference_update)

@overload_method(types.Set, 'symmetric_difference_update')
def set_symmetric_difference_update_impl(a, b):
    if False:
        return 10
    check_all_set(a, b)
    return lambda a, b: _set_symmetric_difference_update(a, b)

@lower_builtin('set.update', types.Set, types.IterableType)
def set_update(context, builder, sig, args):
    if False:
        return 10
    inst = SetInstance(context, builder, sig.args[0], args[0])
    items_type = sig.args[1]
    items = args[1]
    n = call_len(context, builder, items_type, items)
    if n is not None:
        new_size = builder.add(inst.payload.used, n)
        inst.upsize(new_size)
    with for_iter(context, builder, items_type, items) as loop:
        casted = context.cast(builder, loop.value, items_type.dtype, inst.dtype)
        inst.add(casted)
        context.nrt.decref(builder, items_type.dtype, loop.value)
    if n is not None:
        inst.downsize(inst.payload.used)
    return context.get_dummy_value()

def gen_operator_impl(op, impl):
    if False:
        return 10

    @intrinsic
    def _set_operator_intr(typingctx, a, b):
        if False:
            print('Hello World!')
        sig = a(a, b)

        def codegen(context, builder, sig, args):
            if False:
                return 10
            assert sig.return_type == sig.args[0]
            impl(context, builder, sig, args)
            return impl_ret_borrowed(context, builder, sig.args[0], args[0])
        return (sig, codegen)

    @overload(op)
    def _ol_set_operator(a, b):
        if False:
            while True:
                i = 10
        check_all_set(a, b)
        return lambda a, b: _set_operator_intr(a, b)
for (op_, op_impl) in [(operator.iand, set_intersection_update), (operator.ior, set_update), (operator.isub, set_difference_update), (operator.ixor, set_symmetric_difference_update)]:
    gen_operator_impl(op_, op_impl)

@overload(operator.sub)
@overload_method(types.Set, 'difference')
def impl_set_difference(a, b):
    if False:
        i = 10
        return i + 15
    check_all_set(a, b)

    def difference_impl(a, b):
        if False:
            while True:
                i = 10
        s = a.copy()
        s.difference_update(b)
        return s
    return difference_impl

@overload(operator.and_)
@overload_method(types.Set, 'intersection')
def set_intersection(a, b):
    if False:
        return 10
    check_all_set(a, b)

    def intersection_impl(a, b):
        if False:
            while True:
                i = 10
        if len(a) < len(b):
            s = a.copy()
            s.intersection_update(b)
            return s
        else:
            s = b.copy()
            s.intersection_update(a)
            return s
    return intersection_impl

@overload(operator.xor)
@overload_method(types.Set, 'symmetric_difference')
def set_symmetric_difference(a, b):
    if False:
        return 10
    check_all_set(a, b)

    def symmetric_difference_impl(a, b):
        if False:
            for i in range(10):
                print('nop')
        if len(a) > len(b):
            s = a.copy()
            s.symmetric_difference_update(b)
            return s
        else:
            s = b.copy()
            s.symmetric_difference_update(a)
            return s
    return symmetric_difference_impl

@overload(operator.or_)
@overload_method(types.Set, 'union')
def set_union(a, b):
    if False:
        return 10
    check_all_set(a, b)

    def union_impl(a, b):
        if False:
            i = 10
            return i + 15
        if len(a) > len(b):
            s = a.copy()
            s.update(b)
            return s
        else:
            s = b.copy()
            s.update(a)
            return s
    return union_impl

@intrinsic
def _set_isdisjoint(typingctx, a, b):
    if False:
        print('Hello World!')
    sig = types.boolean(a, b)

    def codegen(context, builder, sig, args):
        if False:
            print('Hello World!')
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = SetInstance(context, builder, sig.args[1], args[1])
        return inst.isdisjoint(other)
    return (sig, codegen)

@overload_method(types.Set, 'isdisjoint')
def set_isdisjoint(a, b):
    if False:
        i = 10
        return i + 15
    check_all_set(a, b)
    return lambda a, b: _set_isdisjoint(a, b)

@intrinsic
def _set_issubset(typingctx, a, b):
    if False:
        print('Hello World!')
    sig = types.boolean(a, b)

    def codegen(context, builder, sig, args):
        if False:
            i = 10
            return i + 15
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = SetInstance(context, builder, sig.args[1], args[1])
        return inst.issubset(other)
    return (sig, codegen)

@overload(operator.le)
@overload_method(types.Set, 'issubset')
def set_issubset(a, b):
    if False:
        print('Hello World!')
    check_all_set(a, b)
    return lambda a, b: _set_issubset(a, b)

@overload(operator.ge)
@overload_method(types.Set, 'issuperset')
def set_issuperset(a, b):
    if False:
        while True:
            i = 10
    check_all_set(a, b)

    def superset_impl(a, b):
        if False:
            print('Hello World!')
        return b.issubset(a)
    return superset_impl

@intrinsic
def _set_eq(typingctx, a, b):
    if False:
        print('Hello World!')
    sig = types.boolean(a, b)

    def codegen(context, builder, sig, args):
        if False:
            return 10
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = SetInstance(context, builder, sig.args[1], args[1])
        return inst.equals(other)
    return (sig, codegen)

@overload(operator.eq)
def set_eq(a, b):
    if False:
        for i in range(10):
            print('nop')
    check_all_set(a, b)
    return lambda a, b: _set_eq(a, b)

@overload(operator.ne)
def set_ne(a, b):
    if False:
        i = 10
        return i + 15
    check_all_set(a, b)

    def ne_impl(a, b):
        if False:
            print('Hello World!')
        return not a == b
    return ne_impl

@intrinsic
def _set_lt(typingctx, a, b):
    if False:
        print('Hello World!')
    sig = types.boolean(a, b)

    def codegen(context, builder, sig, args):
        if False:
            return 10
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = SetInstance(context, builder, sig.args[1], args[1])
        return inst.issubset(other, strict=True)
    return (sig, codegen)

@overload(operator.lt)
def set_lt(a, b):
    if False:
        return 10
    check_all_set(a, b)
    return lambda a, b: _set_lt(a, b)

@overload(operator.gt)
def set_gt(a, b):
    if False:
        i = 10
        return i + 15
    check_all_set(a, b)

    def gt_impl(a, b):
        if False:
            print('Hello World!')
        return b < a
    return gt_impl

@lower_builtin(operator.is_, types.Set, types.Set)
def set_is(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    a = SetInstance(context, builder, sig.args[0], args[0])
    b = SetInstance(context, builder, sig.args[1], args[1])
    ma = builder.ptrtoint(a.meminfo, cgutils.intp_t)
    mb = builder.ptrtoint(b.meminfo, cgutils.intp_t)
    return builder.icmp_signed('==', ma, mb)

@lower_cast(types.Set, types.Set)
def set_to_set(context, builder, fromty, toty, val):
    if False:
        return 10
    assert fromty.dtype == toty.dtype
    return val