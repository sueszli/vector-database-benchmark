"""
Implementation of the range object for fixed-size integers.
"""
import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import lower_builtin, lower_cast, iterator_impl, impl_ret_untracked
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange

def make_range_iterator(typ):
    if False:
        print('Hello World!')
    '\n    Return the Structure representation of the given *typ* (an\n    instance of types.RangeIteratorType).\n    '
    return cgutils.create_struct_proxy(typ)

def make_range_impl(int_type, range_state_type, range_iter_type):
    if False:
        for i in range(10):
            print('nop')
    RangeState = cgutils.create_struct_proxy(range_state_type)

    @lower_builtin(range, int_type)
    @lower_builtin(prange, int_type)
    @lower_builtin(internal_prange, int_type)
    def range1_impl(context, builder, sig, args):
        if False:
            return 10
        '\n        range(stop: int) -> range object\n        '
        [stop] = args
        state = RangeState(context, builder)
        state.start = context.get_constant(int_type, 0)
        state.stop = stop
        state.step = context.get_constant(int_type, 1)
        return impl_ret_untracked(context, builder, range_state_type, state._getvalue())

    @lower_builtin(range, int_type, int_type)
    @lower_builtin(prange, int_type, int_type)
    @lower_builtin(internal_prange, int_type, int_type)
    def range2_impl(context, builder, sig, args):
        if False:
            while True:
                i = 10
        '\n        range(start: int, stop: int) -> range object\n        '
        (start, stop) = args
        state = RangeState(context, builder)
        state.start = start
        state.stop = stop
        state.step = context.get_constant(int_type, 1)
        return impl_ret_untracked(context, builder, range_state_type, state._getvalue())

    @lower_builtin(range, int_type, int_type, int_type)
    @lower_builtin(prange, int_type, int_type, int_type)
    @lower_builtin(internal_prange, int_type, int_type, int_type)
    def range3_impl(context, builder, sig, args):
        if False:
            print('Hello World!')
        '\n        range(start: int, stop: int, step: int) -> range object\n        '
        [start, stop, step] = args
        state = RangeState(context, builder)
        state.start = start
        state.stop = stop
        state.step = step
        return impl_ret_untracked(context, builder, range_state_type, state._getvalue())

    @lower_builtin(len, range_state_type)
    def range_len(context, builder, sig, args):
        if False:
            print('Hello World!')
        '\n        len(range)\n        '
        (value,) = args
        state = RangeState(context, builder, value)
        res = RangeIter.from_range_state(context, builder, state)
        return impl_ret_untracked(context, builder, int_type, builder.load(res.count))

    @lower_builtin('getiter', range_state_type)
    def getiter_range32_impl(context, builder, sig, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        range.__iter__\n        '
        (value,) = args
        state = RangeState(context, builder, value)
        res = RangeIter.from_range_state(context, builder, state)._getvalue()
        return impl_ret_untracked(context, builder, range_iter_type, res)

    @iterator_impl(range_state_type, range_iter_type)
    class RangeIter(make_range_iterator(range_iter_type)):

        @classmethod
        def from_range_state(cls, context, builder, state):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Create a RangeIter initialized from the given RangeState *state*.\n            '
            self = cls(context, builder)
            start = state.start
            stop = state.stop
            step = state.step
            startptr = cgutils.alloca_once(builder, start.type)
            builder.store(start, startptr)
            countptr = cgutils.alloca_once(builder, start.type)
            self.iter = startptr
            self.stop = stop
            self.step = step
            self.count = countptr
            diff = builder.sub(stop, start)
            zero = context.get_constant(int_type, 0)
            one = context.get_constant(int_type, 1)
            pos_diff = builder.icmp_signed('>', diff, zero)
            pos_step = builder.icmp_signed('>', step, zero)
            sign_differs = builder.xor(pos_diff, pos_step)
            zero_step = builder.icmp_unsigned('==', step, zero)
            with cgutils.if_unlikely(builder, zero_step):
                context.call_conv.return_user_exc(builder, ValueError, ('range() arg 3 must not be zero',))
            with builder.if_else(sign_differs) as (then, orelse):
                with then:
                    builder.store(zero, self.count)
                with orelse:
                    rem = builder.srem(diff, step)
                    rem = builder.select(pos_diff, rem, builder.neg(rem))
                    uneven = builder.icmp_signed('>', rem, zero)
                    newcount = builder.add(builder.sdiv(diff, step), builder.select(uneven, one, zero))
                    builder.store(newcount, self.count)
            return self

        def iternext(self, context, builder, result):
            if False:
                i = 10
                return i + 15
            zero = context.get_constant(int_type, 0)
            countptr = self.count
            count = builder.load(countptr)
            is_valid = builder.icmp_signed('>', count, zero)
            result.set_valid(is_valid)
            with builder.if_then(is_valid):
                value = builder.load(self.iter)
                result.yield_(value)
                one = context.get_constant(int_type, 1)
                builder.store(builder.sub(count, one, flags=['nsw']), countptr)
                builder.store(builder.add(value, self.step), self.iter)
range_impl_map = {types.int32: (types.range_state32_type, types.range_iter32_type), types.int64: (types.range_state64_type, types.range_iter64_type), types.uint64: (types.unsigned_range_state64_type, types.unsigned_range_iter64_type)}
for (int_type, state_types) in range_impl_map.items():
    make_range_impl(int_type, *state_types)

@lower_cast(types.RangeType, types.RangeType)
def range_to_range(context, builder, fromty, toty, val):
    if False:
        while True:
            i = 10
    olditems = cgutils.unpack_tuple(builder, val, 3)
    items = [context.cast(builder, v, fromty.dtype, toty.dtype) for v in olditems]
    return cgutils.make_anonymous_struct(builder, items)

@intrinsic
def length_of_iterator(typingctx, val):
    if False:
        return 10
    '\n    An implementation of len(iter) for internal use.\n    Primary use is for array comprehensions (see inline_closurecall).\n    '
    if isinstance(val, types.RangeIteratorType):
        val_type = val.yield_type

        def codegen(context, builder, sig, args):
            if False:
                print('Hello World!')
            (value,) = args
            iter_type = range_impl_map[val_type][1]
            iterobj = cgutils.create_struct_proxy(iter_type)(context, builder, value)
            int_type = iterobj.count.type
            return impl_ret_untracked(context, builder, int_type, builder.load(iterobj.count))
        return (signature(val_type, val), codegen)
    elif isinstance(val, types.ListIter):

        def codegen(context, builder, sig, args):
            if False:
                return 10
            (value,) = args
            intp_t = context.get_value_type(types.intp)
            iterobj = ListIterInstance(context, builder, sig.args[0], value)
            return impl_ret_untracked(context, builder, intp_t, iterobj.size)
        return (signature(types.intp, val), codegen)
    elif isinstance(val, types.ArrayIterator):

        def codegen(context, builder, sig, args):
            if False:
                print('Hello World!')
            (iterty,) = sig.args
            (value,) = args
            intp_t = context.get_value_type(types.intp)
            iterobj = context.make_helper(builder, iterty, value=value)
            arrayty = iterty.array_type
            ary = make_array(arrayty)(context, builder, value=iterobj.array)
            shape = cgutils.unpack_tuple(builder, ary.shape)
            return impl_ret_untracked(context, builder, intp_t, shape[0])
        return (signature(types.intp, val), codegen)
    elif isinstance(val, types.UniTupleIter):

        def codegen(context, builder, sig, args):
            if False:
                i = 10
                return i + 15
            (iterty,) = sig.args
            tuplety = iterty.container
            intp_t = context.get_value_type(types.intp)
            count_const = intp_t(tuplety.count)
            return impl_ret_untracked(context, builder, intp_t, count_const)
        return (signature(types.intp, val), codegen)
    elif isinstance(val, types.ListTypeIteratorType):

        def codegen(context, builder, sig, args):
            if False:
                while True:
                    i = 10
            (value,) = args
            intp_t = context.get_value_type(types.intp)
            from numba.typed.listobject import ListIterInstance
            iterobj = ListIterInstance(context, builder, sig.args[0], value)
            return impl_ret_untracked(context, builder, intp_t, iterobj.size)
        return (signature(types.intp, val), codegen)
    else:
        msg = 'Unsupported iterator found in array comprehension, try preallocating the array and filling manually.'
        raise errors.TypingError(msg)

def make_range_attr(index, attribute):
    if False:
        return 10

    @intrinsic
    def rangetype_attr_getter(typingctx, a):
        if False:
            while True:
                i = 10
        if isinstance(a, types.RangeType):

            def codegen(context, builder, sig, args):
                if False:
                    for i in range(10):
                        print('nop')
                (val,) = args
                items = cgutils.unpack_tuple(builder, val, 3)
                return impl_ret_untracked(context, builder, sig.return_type, items[index])
            return (signature(a.dtype, a), codegen)

    @overload_attribute(types.RangeType, attribute)
    def range_attr(rnge):
        if False:
            for i in range(10):
                print('nop')

        def get(rnge):
            if False:
                print('Hello World!')
            return rangetype_attr_getter(rnge)
        return get

@register_jitable
def impl_contains_helper(robj, val):
    if False:
        while True:
            i = 10
    if robj.step > 0 and (val < robj.start or val >= robj.stop):
        return False
    elif robj.step < 0 and (val <= robj.stop or val > robj.start):
        return False
    return (val - robj.start) % robj.step == 0

@overload(operator.contains)
def impl_contains(robj, val):
    if False:
        return 10

    def impl_false(robj, val):
        if False:
            i = 10
            return i + 15
        return False
    if not isinstance(robj, types.RangeType):
        return
    elif isinstance(val, (types.Integer, types.Boolean)):
        return impl_contains_helper
    elif isinstance(val, types.Float):

        def impl(robj, val):
            if False:
                i = 10
                return i + 15
            if val % 1 != 0:
                return False
            else:
                return impl_contains_helper(robj, int(val))
        return impl
    elif isinstance(val, types.Complex):

        def impl(robj, val):
            if False:
                return 10
            if val.imag != 0:
                return False
            elif val.real % 1 != 0:
                return False
            else:
                return impl_contains_helper(robj, int(val.real))
        return impl
    elif not isinstance(val, types.Number):
        return impl_false
for (ix, attr) in enumerate(('start', 'stop', 'step')):
    make_range_attr(index=ix, attribute=attr)