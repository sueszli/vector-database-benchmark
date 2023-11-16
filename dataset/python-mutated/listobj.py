"""
Support for native homogeneous lists.
"""
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import lower_builtin, lower_cast, iternext_impl, impl_ret_borrowed, impl_ret_new_ref, impl_ret_untracked, RefType
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll

def get_list_payload(context, builder, list_type, value):
    if False:
        i = 10
        return i + 15
    '\n    Given a list value and type, get its payload structure (as a\n    reference, so that mutations are seen by all).\n    '
    payload_type = types.ListPayload(list_type)
    payload = context.nrt.meminfo_data(builder, value.meminfo)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)

def get_itemsize(context, list_type):
    if False:
        return 10
    '\n    Return the item size for the given list type.\n    '
    llty = context.get_data_type(list_type.dtype)
    return context.get_abi_sizeof(llty)

class _ListPayloadMixin(object):

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        return self._payload.size

    @size.setter
    def size(self, value):
        if False:
            while True:
                i = 10
        self._payload.size = value

    @property
    def dirty(self):
        if False:
            return 10
        return self._payload.dirty

    @property
    def data(self):
        if False:
            return 10
        return self._payload._get_ptr_by_name('data')

    def _gep(self, idx):
        if False:
            while True:
                i = 10
        return cgutils.gep(self._builder, self.data, idx)

    def getitem(self, idx):
        if False:
            return 10
        ptr = self._gep(idx)
        data_item = self._builder.load(ptr)
        return self._datamodel.from_data(self._builder, data_item)

    def fix_index(self, idx):
        if False:
            i = 10
            return i + 15
        '\n        Fix negative indices by adding the size to them.  Positive\n        indices are left untouched.\n        '
        is_negative = self._builder.icmp_signed('<', idx, ir.Constant(idx.type, 0))
        wrapped_index = self._builder.add(idx, self.size)
        return self._builder.select(is_negative, wrapped_index, idx)

    def is_out_of_bounds(self, idx):
        if False:
            return 10
        '\n        Return whether the index is out of bounds.\n        '
        underflow = self._builder.icmp_signed('<', idx, ir.Constant(idx.type, 0))
        overflow = self._builder.icmp_signed('>=', idx, self.size)
        return self._builder.or_(underflow, overflow)

    def clamp_index(self, idx):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clamp the index in [0, size].\n        '
        builder = self._builder
        idxptr = cgutils.alloca_once_value(builder, idx)
        zero = ir.Constant(idx.type, 0)
        size = self.size
        underflow = self._builder.icmp_signed('<', idx, zero)
        with builder.if_then(underflow, likely=False):
            builder.store(zero, idxptr)
        overflow = self._builder.icmp_signed('>=', idx, size)
        with builder.if_then(overflow, likely=False):
            builder.store(size, idxptr)
        return builder.load(idxptr)

    def guard_index(self, idx, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Raise an error if the index is out of bounds.\n        '
        with self._builder.if_then(self.is_out_of_bounds(idx), likely=False):
            self._context.call_conv.return_user_exc(self._builder, IndexError, (msg,))

    def fix_slice(self, slice):
        if False:
            i = 10
            return i + 15
        '\n        Fix slice start and stop to be valid (inclusive and exclusive, resp)\n        indexing bounds.\n        '
        return slicing.fix_slice(self._builder, slice, self.size)

    def incref_value(self, val):
        if False:
            i = 10
            return i + 15
        'Incref an element value'
        self._context.nrt.incref(self._builder, self.dtype, val)

    def decref_value(self, val):
        if False:
            while True:
                i = 10
        'Decref an element value'
        self._context.nrt.decref(self._builder, self.dtype, val)

class ListPayloadAccessor(_ListPayloadMixin):
    """
    A helper object to access the list attributes given the pointer to the
    payload type.
    """

    def __init__(self, context, builder, list_type, payload_ptr):
        if False:
            while True:
                i = 10
        self._context = context
        self._builder = builder
        self._ty = list_type
        self._datamodel = context.data_model_manager[list_type.dtype]
        payload_type = types.ListPayload(list_type)
        ptrty = context.get_data_type(payload_type).as_pointer()
        payload_ptr = builder.bitcast(payload_ptr, ptrty)
        payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)
        self._payload = payload

class ListInstance(_ListPayloadMixin):

    def __init__(self, context, builder, list_type, list_val):
        if False:
            i = 10
            return i + 15
        self._context = context
        self._builder = builder
        self._ty = list_type
        self._list = context.make_helper(builder, list_type, list_val)
        self._itemsize = get_itemsize(context, list_type)
        self._datamodel = context.data_model_manager[list_type.dtype]

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self._ty.dtype

    @property
    def _payload(self):
        if False:
            return 10
        return get_list_payload(self._context, self._builder, self._ty, self._list)

    @property
    def parent(self):
        if False:
            i = 10
            return i + 15
        return self._list.parent

    @parent.setter
    def parent(self, value):
        if False:
            return 10
        self._list.parent = value

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        return self._list._getvalue()

    @property
    def meminfo(self):
        if False:
            i = 10
            return i + 15
        return self._list.meminfo

    def set_dirty(self, val):
        if False:
            i = 10
            return i + 15
        if self._ty.reflected:
            self._payload.dirty = cgutils.true_bit if val else cgutils.false_bit

    def clear_value(self, idx):
        if False:
            i = 10
            return i + 15
        'Remove the value at the location\n        '
        self.decref_value(self.getitem(idx))
        self.zfill(idx, self._builder.add(idx, idx.type(1)))

    def setitem(self, idx, val, incref, decref_old_value=True):
        if False:
            print('Hello World!')
        if decref_old_value:
            self.decref_value(self.getitem(idx))
        ptr = self._gep(idx)
        data_item = self._datamodel.as_data(self._builder, val)
        self._builder.store(data_item, ptr)
        self.set_dirty(True)
        if incref:
            self.incref_value(val)

    def inititem(self, idx, val, incref=True):
        if False:
            for i in range(10):
                print('nop')
        ptr = self._gep(idx)
        data_item = self._datamodel.as_data(self._builder, val)
        self._builder.store(data_item, ptr)
        if incref:
            self.incref_value(val)

    def zfill(self, start, stop):
        if False:
            return 10
        'Zero-fill the memory at index *start* to *stop*\n\n        *stop* MUST not be smaller than *start*.\n        '
        builder = self._builder
        base = self._gep(start)
        end = self._gep(stop)
        intaddr_t = self._context.get_value_type(types.intp)
        size = builder.sub(builder.ptrtoint(end, intaddr_t), builder.ptrtoint(base, intaddr_t))
        cgutils.memset(builder, base, size, ir.IntType(8)(0))

    @classmethod
    def allocate_ex(cls, context, builder, list_type, nitems):
        if False:
            for i in range(10):
                print('nop')
        "\n        Allocate a ListInstance with its storage.\n        Return a (ok, instance) tuple where *ok* is a LLVM boolean and\n        *instance* is a ListInstance object (the object's contents are\n        only valid when *ok* is true).\n        "
        intp_t = context.get_value_type(types.intp)
        if isinstance(nitems, int):
            nitems = ir.Constant(intp_t, nitems)
        payload_type = context.get_data_type(types.ListPayload(list_type))
        payload_size = context.get_abi_sizeof(payload_type)
        itemsize = get_itemsize(context, list_type)
        payload_size -= itemsize
        ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
        self = cls(context, builder, list_type, None)
        (allocsize, ovf) = cgutils.muladd_with_overflow(builder, nitems, ir.Constant(intp_t, itemsize), ir.Constant(intp_t, payload_size))
        with builder.if_then(ovf, likely=False):
            builder.store(cgutils.false_bit, ok)
        with builder.if_then(builder.load(ok), likely=True):
            meminfo = context.nrt.meminfo_new_varsize_dtor_unchecked(builder, size=allocsize, dtor=self.get_dtor())
            with builder.if_else(cgutils.is_null(builder, meminfo), likely=False) as (if_error, if_ok):
                with if_error:
                    builder.store(cgutils.false_bit, ok)
                with if_ok:
                    self._list.meminfo = meminfo
                    self._list.parent = context.get_constant_null(types.pyobject)
                    self._payload.allocated = nitems
                    self._payload.size = ir.Constant(intp_t, 0)
                    self._payload.dirty = cgutils.false_bit
                    self.zfill(self.size.type(0), nitems)
        return (builder.load(ok), self)

    def define_dtor(self):
        if False:
            i = 10
            return i + 15
        'Define the destructor if not already defined'
        context = self._context
        builder = self._builder
        mod = builder.module
        fnty = ir.FunctionType(ir.VoidType(), [cgutils.voidptr_t])
        fn = cgutils.get_or_insert_function(mod, fnty, '.dtor.list.{}'.format(self.dtype))
        if not fn.is_declaration:
            return fn
        fn.linkage = 'linkonce_odr'
        builder = ir.IRBuilder(fn.append_basic_block())
        base_ptr = fn.args[0]
        payload = ListPayloadAccessor(context, builder, self._ty, base_ptr)
        intp = payload.size.type
        with cgutils.for_range_slice(builder, start=intp(0), stop=payload.size, step=intp(1), intp=intp) as (idx, _):
            val = payload.getitem(idx)
            context.nrt.decref(builder, self.dtype, val)
        builder.ret_void()
        return fn

    def get_dtor(self):
        if False:
            return 10
        '"Get the element dtor function pointer as void pointer.\n\n        It\'s safe to be called multiple times.\n        '
        dtor = self.define_dtor()
        dtor_fnptr = self._builder.bitcast(dtor, cgutils.voidptr_t)
        return dtor_fnptr

    @classmethod
    def allocate(cls, context, builder, list_type, nitems):
        if False:
            while True:
                i = 10
        "\n        Allocate a ListInstance with its storage.  Same as allocate_ex(),\n        but return an initialized *instance*.  If allocation failed,\n        control is transferred to the caller using the target's current\n        call convention.\n        "
        (ok, self) = cls.allocate_ex(context, builder, list_type, nitems)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError, ('cannot allocate list',))
        return self

    @classmethod
    def from_meminfo(cls, context, builder, list_type, meminfo):
        if False:
            i = 10
            return i + 15
        '\n        Allocate a new list instance pointing to an existing payload\n        (a meminfo pointer).\n        Note the parent field has to be filled by the caller.\n        '
        self = cls(context, builder, list_type, None)
        self._list.meminfo = meminfo
        self._list.parent = context.get_constant_null(types.pyobject)
        context.nrt.incref(builder, list_type, self.value)
        return self

    def resize(self, new_size):
        if False:
            print('Hello World!')
        '\n        Ensure the list is properly sized for the new size.\n        '

        def _payload_realloc(new_allocated):
            if False:
                return 10
            payload_type = context.get_data_type(types.ListPayload(self._ty))
            payload_size = context.get_abi_sizeof(payload_type)
            payload_size -= itemsize
            (allocsize, ovf) = cgutils.muladd_with_overflow(builder, new_allocated, ir.Constant(intp_t, itemsize), ir.Constant(intp_t, payload_size))
            with builder.if_then(ovf, likely=False):
                context.call_conv.return_user_exc(builder, MemoryError, ('cannot resize list',))
            ptr = context.nrt.meminfo_varsize_realloc_unchecked(builder, self._list.meminfo, size=allocsize)
            cgutils.guard_memory_error(context, builder, ptr, 'cannot resize list')
            self._payload.allocated = new_allocated
        context = self._context
        builder = self._builder
        intp_t = new_size.type
        itemsize = get_itemsize(context, self._ty)
        allocated = self._payload.allocated
        two = ir.Constant(intp_t, 2)
        eight = ir.Constant(intp_t, 8)
        is_too_small = builder.icmp_signed('<', allocated, new_size)
        is_too_large = builder.icmp_signed('>', builder.ashr(allocated, two), new_size)
        with builder.if_then(is_too_large, likely=False):
            _payload_realloc(new_size)
        with builder.if_then(is_too_small, likely=False):
            new_allocated = builder.add(eight, builder.add(new_size, builder.ashr(new_size, two)))
            _payload_realloc(new_allocated)
            self.zfill(self.size, new_allocated)
        self._payload.size = new_size
        self.set_dirty(True)

    def move(self, dest_idx, src_idx, count):
        if False:
            print('Hello World!')
        '\n        Move `count` elements from `src_idx` to `dest_idx`.\n        '
        dest_ptr = self._gep(dest_idx)
        src_ptr = self._gep(src_idx)
        cgutils.raw_memmove(self._builder, dest_ptr, src_ptr, count, itemsize=self._itemsize)
        self.set_dirty(True)

class ListIterInstance(_ListPayloadMixin):

    def __init__(self, context, builder, iter_type, iter_val):
        if False:
            while True:
                i = 10
        self._context = context
        self._builder = builder
        self._ty = iter_type
        self._iter = context.make_helper(builder, iter_type, iter_val)
        self._datamodel = context.data_model_manager[iter_type.yield_type]

    @classmethod
    def from_list(cls, context, builder, iter_type, list_val):
        if False:
            i = 10
            return i + 15
        list_inst = ListInstance(context, builder, iter_type.container, list_val)
        self = cls(context, builder, iter_type, None)
        index = context.get_constant(types.intp, 0)
        self._iter.index = cgutils.alloca_once_value(builder, index)
        self._iter.meminfo = list_inst.meminfo
        return self

    @property
    def _payload(self):
        if False:
            for i in range(10):
                print('nop')
        return get_list_payload(self._context, self._builder, self._ty.container, self._iter)

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self._iter._getvalue()

    @property
    def index(self):
        if False:
            i = 10
            return i + 15
        return self._builder.load(self._iter.index)

    @index.setter
    def index(self, value):
        if False:
            while True:
                i = 10
        self._builder.store(value, self._iter.index)

def build_list(context, builder, list_type, items):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build a list of the given type, containing the given items.\n    '
    nitems = len(items)
    inst = ListInstance.allocate(context, builder, list_type, nitems)
    inst.size = context.get_constant(types.intp, nitems)
    for (i, val) in enumerate(items):
        inst.setitem(context.get_constant(types.intp, i), val, incref=True)
    return impl_ret_new_ref(context, builder, list_type, inst.value)

@lower_builtin(list, types.IterableType)
def list_constructor(context, builder, sig, args):
    if False:
        i = 10
        return i + 15

    def list_impl(iterable):
        if False:
            return 10
        res = []
        res.extend(iterable)
        return res
    return context.compile_internal(builder, list_impl, sig, args)

@lower_builtin(list)
def list_constructor(context, builder, sig, args):
    if False:
        print('Hello World!')
    list_type = sig.return_type
    list_len = 0
    inst = ListInstance.allocate(context, builder, list_type, list_len)
    return impl_ret_new_ref(context, builder, list_type, inst.value)

@lower_builtin(len, types.List)
def list_len(context, builder, sig, args):
    if False:
        print('Hello World!')
    inst = ListInstance(context, builder, sig.args[0], args[0])
    return inst.size

@lower_builtin('getiter', types.List)
def getiter_list(context, builder, sig, args):
    if False:
        print('Hello World!')
    inst = ListIterInstance.from_list(context, builder, sig.return_type, args[0])
    return impl_ret_borrowed(context, builder, sig.return_type, inst.value)

@lower_builtin('iternext', types.ListIter)
@iternext_impl(RefType.BORROWED)
def iternext_listiter(context, builder, sig, args, result):
    if False:
        for i in range(10):
            print('nop')
    inst = ListIterInstance(context, builder, sig.args[0], args[0])
    index = inst.index
    nitems = inst.size
    is_valid = builder.icmp_signed('<', index, nitems)
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        result.yield_(inst.getitem(index))
        inst.index = builder.add(index, context.get_constant(types.intp, 1))

@lower_builtin(operator.getitem, types.List, types.Integer)
def getitem_list(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    inst = ListInstance(context, builder, sig.args[0], args[0])
    index = args[1]
    index = inst.fix_index(index)
    inst.guard_index(index, msg='getitem out of range')
    result = inst.getitem(index)
    return impl_ret_borrowed(context, builder, sig.return_type, result)

@lower_builtin(operator.setitem, types.List, types.Integer, types.Any)
def setitem_list(context, builder, sig, args):
    if False:
        print('Hello World!')
    inst = ListInstance(context, builder, sig.args[0], args[0])
    index = args[1]
    value = args[2]
    index = inst.fix_index(index)
    inst.guard_index(index, msg='setitem out of range')
    inst.setitem(index, value, incref=True)
    return context.get_dummy_value()

@lower_builtin(operator.getitem, types.List, types.SliceType)
def getslice_list(context, builder, sig, args):
    if False:
        return 10
    inst = ListInstance(context, builder, sig.args[0], args[0])
    slice = context.make_helper(builder, sig.args[1], args[1])
    slicing.guard_invalid_slice(context, builder, sig.args[1], slice)
    inst.fix_slice(slice)
    result_size = slicing.get_slice_length(builder, slice)
    result = ListInstance.allocate(context, builder, sig.return_type, result_size)
    result.size = result_size
    with cgutils.for_range_slice_generic(builder, slice.start, slice.stop, slice.step) as (pos_range, neg_range):
        with pos_range as (idx, count):
            value = inst.getitem(idx)
            result.inititem(count, value, incref=True)
        with neg_range as (idx, count):
            value = inst.getitem(idx)
            result.inititem(count, value, incref=True)
    return impl_ret_new_ref(context, builder, sig.return_type, result.value)

@lower_builtin(operator.setitem, types.List, types.SliceType, types.Any)
def setitem_list(context, builder, sig, args):
    if False:
        return 10
    dest = ListInstance(context, builder, sig.args[0], args[0])
    src = ListInstance(context, builder, sig.args[2], args[2])
    slice = context.make_helper(builder, sig.args[1], args[1])
    slicing.guard_invalid_slice(context, builder, sig.args[1], slice)
    dest.fix_slice(slice)
    src_size = src.size
    avail_size = slicing.get_slice_length(builder, slice)
    size_delta = builder.sub(src.size, avail_size)
    zero = ir.Constant(size_delta.type, 0)
    one = ir.Constant(size_delta.type, 1)
    with builder.if_else(builder.icmp_signed('==', slice.step, one)) as (then, otherwise):
        with then:
            real_stop = builder.add(slice.start, avail_size)
            tail_size = builder.sub(dest.size, real_stop)
            with builder.if_then(builder.icmp_signed('>', size_delta, zero)):
                dest.resize(builder.add(dest.size, size_delta))
                dest.move(builder.add(real_stop, size_delta), real_stop, tail_size)
            with builder.if_then(builder.icmp_signed('<', size_delta, zero)):
                dest.move(builder.add(real_stop, size_delta), real_stop, tail_size)
                dest.resize(builder.add(dest.size, size_delta))
            dest_offset = slice.start
            with cgutils.for_range(builder, src_size) as loop:
                value = src.getitem(loop.index)
                dest.setitem(builder.add(loop.index, dest_offset), value, incref=True)
        with otherwise:
            with builder.if_then(builder.icmp_signed('!=', size_delta, zero)):
                msg = 'cannot resize extended list slice with step != 1'
                context.call_conv.return_user_exc(builder, ValueError, (msg,))
            with cgutils.for_range_slice_generic(builder, slice.start, slice.stop, slice.step) as (pos_range, neg_range):
                with pos_range as (index, count):
                    value = src.getitem(count)
                    dest.setitem(index, value, incref=True)
                with neg_range as (index, count):
                    value = src.getitem(count)
                    dest.setitem(index, value, incref=True)
    return context.get_dummy_value()

@lower_builtin(operator.delitem, types.List, types.Integer)
def delitem_list_index(context, builder, sig, args):
    if False:
        return 10

    def list_delitem_impl(lst, i):
        if False:
            while True:
                i = 10
        lst.pop(i)
    return context.compile_internal(builder, list_delitem_impl, sig, args)

@lower_builtin(operator.delitem, types.List, types.SliceType)
def delitem_list(context, builder, sig, args):
    if False:
        while True:
            i = 10
    inst = ListInstance(context, builder, sig.args[0], args[0])
    slice = context.make_helper(builder, sig.args[1], args[1])
    slicing.guard_invalid_slice(context, builder, sig.args[1], slice)
    inst.fix_slice(slice)
    slice_len = slicing.get_slice_length(builder, slice)
    one = ir.Constant(slice_len.type, 1)
    with builder.if_then(builder.icmp_signed('!=', slice.step, one), likely=False):
        msg = 'unsupported del list[start:stop:step] with step != 1'
        context.call_conv.return_user_exc(builder, NotImplementedError, (msg,))
    start = slice.start
    real_stop = builder.add(start, slice_len)
    with cgutils.for_range_slice(builder, start, real_stop, start.type(1)) as (idx, _):
        inst.decref_value(inst.getitem(idx))
    tail_size = builder.sub(inst.size, real_stop)
    inst.move(start, real_stop, tail_size)
    inst.resize(builder.sub(inst.size, slice_len))
    return context.get_dummy_value()

@lower_builtin(operator.contains, types.Sequence, types.Any)
def in_seq(context, builder, sig, args):
    if False:
        while True:
            i = 10

    def seq_contains_impl(lst, value):
        if False:
            return 10
        for elem in lst:
            if elem == value:
                return True
        return False
    return context.compile_internal(builder, seq_contains_impl, sig, args)

@lower_builtin(bool, types.Sequence)
def sequence_bool(context, builder, sig, args):
    if False:
        return 10

    def sequence_bool_impl(seq):
        if False:
            i = 10
            return i + 15
        return len(seq) != 0
    return context.compile_internal(builder, sequence_bool_impl, sig, args)

@overload(operator.truth)
def sequence_truth(seq):
    if False:
        while True:
            i = 10
    if isinstance(seq, types.Sequence):

        def impl(seq):
            if False:
                i = 10
                return i + 15
            return len(seq) != 0
        return impl

@lower_builtin(operator.add, types.List, types.List)
def list_add(context, builder, sig, args):
    if False:
        return 10
    a = ListInstance(context, builder, sig.args[0], args[0])
    b = ListInstance(context, builder, sig.args[1], args[1])
    a_size = a.size
    b_size = b.size
    nitems = builder.add(a_size, b_size)
    dest = ListInstance.allocate(context, builder, sig.return_type, nitems)
    dest.size = nitems
    with cgutils.for_range(builder, a_size) as loop:
        value = a.getitem(loop.index)
        value = context.cast(builder, value, a.dtype, dest.dtype)
        dest.setitem(loop.index, value, incref=True)
    with cgutils.for_range(builder, b_size) as loop:
        value = b.getitem(loop.index)
        value = context.cast(builder, value, b.dtype, dest.dtype)
        dest.setitem(builder.add(loop.index, a_size), value, incref=True)
    return impl_ret_new_ref(context, builder, sig.return_type, dest.value)

@lower_builtin(operator.iadd, types.List, types.List)
def list_add_inplace(context, builder, sig, args):
    if False:
        return 10
    assert sig.args[0].dtype == sig.return_type.dtype
    dest = _list_extend_list(context, builder, sig, args)
    return impl_ret_borrowed(context, builder, sig.return_type, dest.value)

@lower_builtin(operator.mul, types.List, types.Integer)
@lower_builtin(operator.mul, types.Integer, types.List)
def list_mul(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(sig.args[0], types.List):
        (list_idx, int_idx) = (0, 1)
    else:
        (list_idx, int_idx) = (1, 0)
    src = ListInstance(context, builder, sig.args[list_idx], args[list_idx])
    src_size = src.size
    mult = args[int_idx]
    zero = ir.Constant(mult.type, 0)
    mult = builder.select(cgutils.is_neg_int(builder, mult), zero, mult)
    nitems = builder.mul(mult, src_size)
    dest = ListInstance.allocate(context, builder, sig.return_type, nitems)
    dest.size = nitems
    with cgutils.for_range_slice(builder, zero, nitems, src_size, inc=True) as (dest_offset, _):
        with cgutils.for_range(builder, src_size) as loop:
            value = src.getitem(loop.index)
            dest.setitem(builder.add(loop.index, dest_offset), value, incref=True)
    return impl_ret_new_ref(context, builder, sig.return_type, dest.value)

@lower_builtin(operator.imul, types.List, types.Integer)
def list_mul_inplace(context, builder, sig, args):
    if False:
        return 10
    inst = ListInstance(context, builder, sig.args[0], args[0])
    src_size = inst.size
    mult = args[1]
    zero = ir.Constant(mult.type, 0)
    mult = builder.select(cgutils.is_neg_int(builder, mult), zero, mult)
    nitems = builder.mul(mult, src_size)
    inst.resize(nitems)
    with cgutils.for_range_slice(builder, src_size, nitems, src_size, inc=True) as (dest_offset, _):
        with cgutils.for_range(builder, src_size) as loop:
            value = inst.getitem(loop.index)
            inst.setitem(builder.add(loop.index, dest_offset), value, incref=True)
    return impl_ret_borrowed(context, builder, sig.return_type, inst.value)

@lower_builtin(operator.is_, types.List, types.List)
def list_is(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    a = ListInstance(context, builder, sig.args[0], args[0])
    b = ListInstance(context, builder, sig.args[1], args[1])
    ma = builder.ptrtoint(a.meminfo, cgutils.intp_t)
    mb = builder.ptrtoint(b.meminfo, cgutils.intp_t)
    return builder.icmp_signed('==', ma, mb)

@lower_builtin(operator.eq, types.List, types.List)
def list_eq(context, builder, sig, args):
    if False:
        while True:
            i = 10
    (aty, bty) = sig.args
    a = ListInstance(context, builder, aty, args[0])
    b = ListInstance(context, builder, bty, args[1])
    a_size = a.size
    same_size = builder.icmp_signed('==', a_size, b.size)
    res = cgutils.alloca_once_value(builder, same_size)
    with builder.if_then(same_size):
        with cgutils.for_range(builder, a_size) as loop:
            v = a.getitem(loop.index)
            w = b.getitem(loop.index)
            itemres = context.generic_compare(builder, operator.eq, (aty.dtype, bty.dtype), (v, w))
            with builder.if_then(builder.not_(itemres)):
                builder.store(cgutils.false_bit, res)
                loop.do_break()
    return builder.load(res)

def all_list(*args):
    if False:
        return 10
    return all([isinstance(typ, types.List) for typ in args])

@overload(operator.ne)
def impl_list_ne(a, b):
    if False:
        return 10
    if not all_list(a, b):
        return

    def list_ne_impl(a, b):
        if False:
            for i in range(10):
                print('nop')
        return not a == b
    return list_ne_impl

@overload(operator.le)
def impl_list_le(a, b):
    if False:
        i = 10
        return i + 15
    if not all_list(a, b):
        return

    def list_le_impl(a, b):
        if False:
            return 10
        m = len(a)
        n = len(b)
        for i in range(min(m, n)):
            if a[i] < b[i]:
                return True
            elif a[i] > b[i]:
                return False
        return m <= n
    return list_le_impl

@overload(operator.lt)
def impl_list_lt(a, b):
    if False:
        for i in range(10):
            print('nop')
    if not all_list(a, b):
        return

    def list_lt_impl(a, b):
        if False:
            print('Hello World!')
        m = len(a)
        n = len(b)
        for i in range(min(m, n)):
            if a[i] < b[i]:
                return True
            elif a[i] > b[i]:
                return False
        return m < n
    return list_lt_impl

@overload(operator.ge)
def impl_list_ge(a, b):
    if False:
        for i in range(10):
            print('nop')
    if not all_list(a, b):
        return

    def list_ge_impl(a, b):
        if False:
            for i in range(10):
                print('nop')
        return b <= a
    return list_ge_impl

@overload(operator.gt)
def impl_list_gt(a, b):
    if False:
        print('Hello World!')
    if not all_list(a, b):
        return

    def list_gt_impl(a, b):
        if False:
            i = 10
            return i + 15
        return b < a
    return list_gt_impl

@lower_builtin('list.append', types.List, types.Any)
def list_append(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    inst = ListInstance(context, builder, sig.args[0], args[0])
    item = args[1]
    n = inst.size
    new_size = builder.add(n, ir.Constant(n.type, 1))
    inst.resize(new_size)
    inst.setitem(n, item, incref=True)
    return context.get_dummy_value()

@lower_builtin('list.clear', types.List)
def list_clear(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    inst = ListInstance(context, builder, sig.args[0], args[0])
    inst.resize(context.get_constant(types.intp, 0))
    return context.get_dummy_value()

@overload_method(types.List, 'copy')
def list_copy(lst):
    if False:
        for i in range(10):
            print('nop')

    def list_copy_impl(lst):
        if False:
            for i in range(10):
                print('nop')
        return list(lst)
    return list_copy_impl

@overload_method(types.List, 'count')
def list_count(lst, value):
    if False:
        while True:
            i = 10

    def list_count_impl(lst, value):
        if False:
            while True:
                i = 10
        res = 0
        for elem in lst:
            if elem == value:
                res += 1
        return res
    return list_count_impl

def _list_extend_list(context, builder, sig, args):
    if False:
        print('Hello World!')
    src = ListInstance(context, builder, sig.args[1], args[1])
    dest = ListInstance(context, builder, sig.args[0], args[0])
    src_size = src.size
    dest_size = dest.size
    nitems = builder.add(src_size, dest_size)
    dest.resize(nitems)
    dest.size = nitems
    with cgutils.for_range(builder, src_size) as loop:
        value = src.getitem(loop.index)
        value = context.cast(builder, value, src.dtype, dest.dtype)
        dest.setitem(builder.add(loop.index, dest_size), value, incref=True)
    return dest

@lower_builtin('list.extend', types.List, types.IterableType)
def list_extend(context, builder, sig, args):
    if False:
        return 10
    if isinstance(sig.args[1], types.List):
        _list_extend_list(context, builder, sig, args)
        return context.get_dummy_value()

    def list_extend(lst, iterable):
        if False:
            for i in range(10):
                print('nop')
        meth = lst.append
        for v in iterable:
            meth(v)
    return context.compile_internal(builder, list_extend, sig, args)
intp_max = types.intp.maxval

@overload_method(types.List, 'index')
def list_index(lst, value, start=0, stop=intp_max):
    if False:
        while True:
            i = 10
    if not isinstance(start, (int, types.Integer, types.Omitted)):
        raise errors.TypingError(f'arg "start" must be an Integer. Got {start}')
    if not isinstance(stop, (int, types.Integer, types.Omitted)):
        raise errors.TypingError(f'arg "stop" must be an Integer. Got {stop}')

    def list_index_impl(lst, value, start=0, stop=intp_max):
        if False:
            i = 10
            return i + 15
        n = len(lst)
        if start < 0:
            start += n
            if start < 0:
                start = 0
        if stop < 0:
            stop += n
        if stop > n:
            stop = n
        for i in range(start, stop):
            if lst[i] == value:
                return i
        raise ValueError('value not in list')
    return list_index_impl

@lower_builtin('list.insert', types.List, types.Integer, types.Any)
def list_insert(context, builder, sig, args):
    if False:
        i = 10
        return i + 15
    inst = ListInstance(context, builder, sig.args[0], args[0])
    index = inst.fix_index(args[1])
    index = inst.clamp_index(index)
    value = args[2]
    n = inst.size
    one = ir.Constant(n.type, 1)
    new_size = builder.add(n, one)
    inst.resize(new_size)
    inst.move(builder.add(index, one), index, builder.sub(n, index))
    inst.setitem(index, value, incref=True, decref_old_value=False)
    return context.get_dummy_value()

@lower_builtin('list.pop', types.List)
def list_pop(context, builder, sig, args):
    if False:
        while True:
            i = 10
    inst = ListInstance(context, builder, sig.args[0], args[0])
    n = inst.size
    cgutils.guard_zero(context, builder, n, (IndexError, 'pop from empty list'))
    n = builder.sub(n, ir.Constant(n.type, 1))
    res = inst.getitem(n)
    inst.incref_value(res)
    inst.clear_value(n)
    inst.resize(n)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@lower_builtin('list.pop', types.List, types.Integer)
def list_pop(context, builder, sig, args):
    if False:
        for i in range(10):
            print('nop')
    inst = ListInstance(context, builder, sig.args[0], args[0])
    idx = inst.fix_index(args[1])
    n = inst.size
    cgutils.guard_zero(context, builder, n, (IndexError, 'pop from empty list'))
    inst.guard_index(idx, 'pop index out of range')
    res = inst.getitem(idx)
    one = ir.Constant(n.type, 1)
    n = builder.sub(n, ir.Constant(n.type, 1))
    inst.move(idx, builder.add(idx, one), builder.sub(n, idx))
    inst.resize(n)
    return impl_ret_new_ref(context, builder, sig.return_type, res)

@overload_method(types.List, 'remove')
def list_remove(lst, value):
    if False:
        return 10

    def list_remove_impl(lst, value):
        if False:
            return 10
        for i in range(len(lst)):
            if lst[i] == value:
                lst.pop(i)
                return
        raise ValueError('list.remove(x): x not in list')
    return list_remove_impl

@overload_method(types.List, 'reverse')
def list_reverse(lst):
    if False:
        i = 10
        return i + 15

    def list_reverse_impl(lst):
        if False:
            print('Hello World!')
        for a in range(0, len(lst) // 2):
            b = -a - 1
            (lst[a], lst[b]) = (lst[b], lst[a])
    return list_reverse_impl

def gt(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a > b
sort_forwards = quicksort.make_jit_quicksort().run_quicksort
sort_backwards = quicksort.make_jit_quicksort(lt=gt).run_quicksort
arg_sort_forwards = quicksort.make_jit_quicksort(is_argsort=True, is_list=True).run_quicksort
arg_sort_backwards = quicksort.make_jit_quicksort(is_argsort=True, lt=gt, is_list=True).run_quicksort

def _sort_check_reverse(reverse):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(reverse, types.Omitted):
        rty = reverse.value
    elif isinstance(reverse, types.Optional):
        rty = reverse.type
    else:
        rty = reverse
    if not isinstance(rty, (types.Boolean, types.Integer, int, bool)):
        msg = "an integer is required for 'reverse' (got type %s)" % reverse
        raise errors.TypingError(msg)
    return rty

def _sort_check_key(key):
    if False:
        return 10
    if isinstance(key, types.Optional):
        msg = 'Key must concretely be None or a Numba JIT compiled function, an Optional (union of None and a value) was found'
        raise errors.TypingError(msg)
    if not (cgutils.is_nonelike(key) or isinstance(key, types.Dispatcher)):
        msg = 'Key must be None or a Numba JIT compiled function'
        raise errors.TypingError(msg)

@overload_method(types.List, 'sort')
def ol_list_sort(lst, key=None, reverse=False):
    if False:
        for i in range(10):
            print('nop')
    _sort_check_key(key)
    _sort_check_reverse(reverse)
    if cgutils.is_nonelike(key):
        KEY = False
        sort_f = sort_forwards
        sort_b = sort_backwards
    elif isinstance(key, types.Dispatcher):
        KEY = True
        sort_f = arg_sort_forwards
        sort_b = arg_sort_backwards

    def impl(lst, key=None, reverse=False):
        if False:
            i = 10
            return i + 15
        if KEY is True:
            _lst = [key(x) for x in lst]
        else:
            _lst = lst
        if reverse is False or reverse == 0:
            tmp = sort_f(_lst)
        else:
            tmp = sort_b(_lst)
        if KEY is True:
            lst[:] = [lst[i] for i in tmp]
    return impl

@overload(sorted)
def ol_sorted(iterable, key=None, reverse=False):
    if False:
        i = 10
        return i + 15
    if not isinstance(iterable, types.IterableType):
        return False
    _sort_check_key(key)
    _sort_check_reverse(reverse)

    def impl(iterable, key=None, reverse=False):
        if False:
            while True:
                i = 10
        lst = list(iterable)
        lst.sort(key=key, reverse=reverse)
        return lst
    return impl

@lower_cast(types.List, types.List)
def list_to_list(context, builder, fromty, toty, val):
    if False:
        print('Hello World!')
    assert fromty.dtype == toty.dtype
    return val
_banned_error = errors.TypingError('Cannot mutate a literal list')

@overload_method(types.LiteralList, 'append')
def literal_list_banned_append(lst, obj):
    if False:
        while True:
            i = 10
    raise _banned_error

@overload_method(types.LiteralList, 'extend')
def literal_list_banned_extend(lst, iterable):
    if False:
        return 10
    raise _banned_error

@overload_method(types.LiteralList, 'insert')
def literal_list_banned_insert(lst, index, obj):
    if False:
        print('Hello World!')
    raise _banned_error

@overload_method(types.LiteralList, 'remove')
def literal_list_banned_remove(lst, value):
    if False:
        for i in range(10):
            print('nop')
    raise _banned_error

@overload_method(types.LiteralList, 'pop')
def literal_list_banned_pop(lst, index=-1):
    if False:
        return 10
    raise _banned_error

@overload_method(types.LiteralList, 'clear')
def literal_list_banned_clear(lst):
    if False:
        return 10
    raise _banned_error

@overload_method(types.LiteralList, 'sort')
def literal_list_banned_sort(lst, key=None, reverse=False):
    if False:
        for i in range(10):
            print('nop')
    raise _banned_error

@overload_method(types.LiteralList, 'reverse')
def literal_list_banned_reverse(lst):
    if False:
        for i in range(10):
            print('nop')
    raise _banned_error
_index_end = types.intp.maxval

@overload_method(types.LiteralList, 'index')
def literal_list_index(lst, x, start=0, end=_index_end):
    if False:
        return 10
    if isinstance(lst, types.LiteralList):
        msg = 'list.index is unsupported for literal lists'
        raise errors.TypingError(msg)

@overload_method(types.LiteralList, 'count')
def literal_list_count(lst, x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(lst, types.LiteralList):

        def impl(lst, x):
            if False:
                print('Hello World!')
            count = 0
            for val in literal_unroll(lst):
                if val == x:
                    count += 1
            return count
        return impl

@overload_method(types.LiteralList, 'copy')
def literal_list_count(lst):
    if False:
        i = 10
        return i + 15
    if isinstance(lst, types.LiteralList):

        def impl(lst):
            if False:
                for i in range(10):
                    print('nop')
            return lst
        return impl

@overload(operator.delitem)
def literal_list_delitem(lst, index):
    if False:
        return 10
    if isinstance(lst, types.LiteralList):
        raise _banned_error

@overload(operator.setitem)
def literal_list_setitem(lst, index, value):
    if False:
        while True:
            i = 10
    if isinstance(lst, types.LiteralList):
        raise errors.TypingError('Cannot mutate a literal list')

@overload(operator.getitem)
def literal_list_getitem(lst, *args):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(lst, types.LiteralList):
        return
    msg = 'Cannot __getitem__ on a literal list, return type cannot be statically determined.'
    raise errors.TypingError(msg)

@overload(len)
def literal_list_len(lst):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(lst, types.LiteralList):
        return
    l = lst.count
    return lambda lst: l

@overload(operator.contains)
def literal_list_contains(lst, item):
    if False:
        return 10
    if isinstance(lst, types.LiteralList):

        def impl(lst, item):
            if False:
                for i in range(10):
                    print('nop')
            for val in literal_unroll(lst):
                if val == item:
                    return True
            return False
        return impl

@lower_cast(types.LiteralList, types.LiteralList)
def literallist_to_literallist(context, builder, fromty, toty, val):
    if False:
        i = 10
        return i + 15
    if len(fromty) != len(toty):
        raise NotImplementedError
    olditems = cgutils.unpack_tuple(builder, val, len(fromty))
    items = [context.cast(builder, v, f, t) for (v, f, t) in zip(olditems, fromty, toty)]
    return context.make_tuple(builder, toty, items)