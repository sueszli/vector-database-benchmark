import operator
from numba.core import types
from .templates import ConcreteTemplate, AbstractTemplate, AttributeTemplate, CallableTemplate, Registry, signature, bound_function, make_callable_template
from numba.core.typing import collections
registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr

@infer_global(set)
class SetBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            print('Hello World!')
        assert not kws
        if args:
            (iterable,) = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                if isinstance(dtype, types.Hashable):
                    return signature(types.Set(dtype), iterable)
        else:
            return signature(types.Set(types.undefined))

@infer_getattr
class SetAttribute(AttributeTemplate):
    key = types.Set

    @bound_function('set.add')
    def resolve_add(self, set, args, kws):
        if False:
            while True:
                i = 10
        (item,) = args
        assert not kws
        unified = self.context.unify_pairs(set.dtype, item)
        if unified is not None:
            sig = signature(types.none, unified)
            sig = sig.replace(recvr=set.copy(dtype=unified))
            return sig

    @bound_function('set.update')
    def resolve_update(self, set, args, kws):
        if False:
            print('Hello World!')
        (iterable,) = args
        assert not kws
        if not isinstance(iterable, types.IterableType):
            return
        dtype = iterable.iterator_type.yield_type
        unified = self.context.unify_pairs(set.dtype, dtype)
        if unified is not None:
            sig = signature(types.none, iterable)
            sig = sig.replace(recvr=set.copy(dtype=unified))
            return sig

    def _resolve_operator(self, set, args, kws):
        if False:
            for i in range(10):
                print('nop')
        assert not kws
        (iterable,) = args
        if isinstance(iterable, types.Set) and iterable.dtype == set.dtype:
            return signature(set, iterable)

    def _resolve_comparator(self, set, args, kws):
        if False:
            while True:
                i = 10
        assert not kws
        (arg,) = args
        if arg == set:
            return signature(types.boolean, arg)

class SetOperator(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            while True:
                i = 10
        if len(args) != 2:
            return
        (a, b) = args
        if isinstance(a, types.Set) and isinstance(b, types.Set) and (a.dtype == b.dtype):
            return signature(a, *args)

class SetComparison(AbstractTemplate):

    def generic(self, args, kws):
        if False:
            return 10
        if len(args) != 2:
            return
        (a, b) = args
        if isinstance(a, types.Set) and isinstance(b, types.Set) and (a == b):
            return signature(types.boolean, *args)
for op_key in (operator.add, operator.invert):

    @infer_global(op_key)
    class ConcreteSetOperator(SetOperator):
        key = op_key
for op_key in (operator.iadd,):

    @infer_global(op_key)
    class ConcreteInplaceSetOperator(SetOperator):
        key = op_key