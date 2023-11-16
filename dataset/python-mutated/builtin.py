import functools
import inspect
import itertools
import logging
import math
import operator
import types
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import AttributeMutationError, unimplemented, Unsupported, UserError, UserErrorType
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import build_checkpoint_variable, check_constant_args, check_numpy_ndarray_args, check_unspec_python_args, extract_fake_example_value, get_fake_value, guard_if_dyn, is_utils_checkpoint, istype, numpy_operator_wrapper, proxy_args_kwargs, tensortype_to_dtype
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, SetVariable
from .lists import BaseListVariable, ListIteratorVariable, ListVariable, SizeVariable, TupleIteratorVariable, TupleVariable
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
log = logging.getLogger(__name__)
IN_PLACE_DESUGARING_MAP = {operator.iadd: operator.add, operator.isub: operator.sub, operator.imul: operator.mul, operator.ifloordiv: operator.floordiv, operator.itruediv: operator.truediv, operator.imod: operator.mod, operator.imatmul: operator.imatmul, operator.ilshift: operator.lshift, operator.irshift: operator.rshift, operator.ipow: operator.pow, operator.iand: operator.and_, operator.ior: operator.or_, operator.ixor: operator.xor}

class BuiltinVariable(VariableTracker):

    @staticmethod
    @functools.lru_cache(None)
    def _constant_fold_functions():
        if False:
            i = 10
            return i + 15
        fns = {abs, all, any, bool, callable, chr, divmod, float, int, len, max, min, ord, pow, repr, round, str, str.format, sum, type, operator.pos, operator.neg, operator.not_, operator.invert, operator.pow, operator.mul, operator.matmul, operator.floordiv, operator.truediv, operator.mod, operator.add, operator.sub, operator.getitem, operator.lshift, operator.rshift, operator.and_, operator.or_, operator.xor, operator.ipow, operator.imul, operator.imatmul, operator.ifloordiv, operator.itruediv, operator.imod, operator.iadd, operator.isub, operator.ilshift, operator.irshift, operator.iand, operator.ixor, operator.ior, operator.index}
        fns.update((x for x in math.__dict__.values() if isinstance(x, type(math.sqrt))))
        return fns

    def can_constant_fold_through(self):
        if False:
            i = 10
            return i + 15
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.lru_cache(None)
    def _fx_graph_functions():
        if False:
            for i in range(10):
                print('nop')
        fns = {operator.pos, operator.neg, operator.not_, operator.invert, operator.pow, operator.mul, operator.matmul, operator.floordiv, operator.truediv, operator.mod, operator.add, operator.lt, operator.gt, operator.ge, operator.le, operator.ne, operator.eq, operator.sub, operator.getitem, operator.lshift, operator.rshift, operator.and_, operator.or_, operator.xor, operator.ipow, operator.imul, operator.imatmul, operator.ifloordiv, operator.itruediv, operator.imod, operator.iadd, operator.isub, operator.ilshift, operator.irshift, operator.iand, operator.ixor, operator.ior}
        return fns

    @staticmethod
    @functools.lru_cache(None)
    def _binops():
        if False:
            print('Hello World!')
        fns = {operator.add: (['__add__', '__radd__', '__iadd__'], operator.iadd), operator.sub: (['__sub__', '__rsub__', '__isub__'], operator.isub), operator.mul: (['__mul__', '__rmul__', '__imul__'], operator.imul), operator.truediv: (['__truediv__', '__rtruediv__', '__itruediv__'], operator.itruediv), operator.floordiv: (['__floordiv__', '__rfloordiv__', '__ifloordiv__'], operator.ifloordiv), operator.mod: (['__mod__', '__rmod__', '__imod__'], operator.imod), pow: (['__pow__', '__rpow__', '__ipow__'], operator.ipow), operator.pow: (['__pow__', '__rpow__', '__ipow__'], operator.ipow), operator.lshift: (['__lshift__', '__rlshift__', '__ilshift__'], operator.ilshift), operator.rshift: (['__rshift__', '__rrshift__', '__irshift__'], operator.irshift)}
        return fns

    @staticmethod
    @functools.lru_cache(None)
    def _binop_handlers():
        if False:
            for i in range(10):
                print('nop')
        op_handlers = {}
        for (op, (magic_method_names, in_place_op)) in BuiltinVariable._binops().items():
            op_handlers[op] = []
            op_handlers[in_place_op] = []
            (forward_name, reverse_name, inplace_name) = magic_method_names

            def user_defined_handler(tx, a, b, options, forward_name=forward_name, reverse_name=reverse_name):
                if False:
                    print('Hello World!')
                if isinstance(a, UserDefinedVariable):
                    return a.call_method(tx, forward_name, [b], {})
                else:
                    return b.call_method(tx, reverse_name, [a], {})
            op_handlers[op].append(((UserDefinedVariable, VariableTracker), user_defined_handler))
            op_handlers[op].append(((VariableTracker, UserDefinedVariable), user_defined_handler))

            def user_defined_inplace_handler(tx, a, b, options, forward_name=inplace_name):
                if False:
                    print('Hello World!')
                return a.call_method(tx, forward_name, [b], {})
            op_handlers[in_place_op].append(((UserDefinedVariable, VariableTracker), user_defined_inplace_handler))
            op_handlers[in_place_op].append(((VariableTracker, UserDefinedVariable), user_defined_inplace_handler))

            def dynamic_handler(tx, a, b, options, fn=op):
                if False:
                    return 10
                from .builder import wrap_fx_proxy
                return wrap_fx_proxy(tx, tx.output.create_proxy('call_function', fn, *proxy_args_kwargs([a, b], {})), **options)
            op_handlers[op].append(((SymNodeVariable, VariableTracker), dynamic_handler))
            op_handlers[op].append(((VariableTracker, SymNodeVariable), dynamic_handler))
            op_handlers[in_place_op].append(((SymNodeVariable, VariableTracker), dynamic_handler))
            op_handlers[in_place_op].append(((VariableTracker, SymNodeVariable), dynamic_handler))

        def tuple_add_handler(tx, a, b, options):
            if False:
                print('Hello World!')
            return TupleVariable(a.items + list(b.unpack_var_sequence(tx)), **options)

        def size_add_handler(tx, a, b, options):
            if False:
                print('Hello World!')
            return SizeVariable(a.items + list(b.unpack_var_sequence(tx)), **options)
        list_like_addition_handlers = [((SizeVariable, SizeVariable), size_add_handler), ((TupleVariable, TupleVariable), tuple_add_handler), ((TupleVariable, ConstantVariable), tuple_add_handler), ((ConstantVariable, TupleVariable), lambda tx, a, b, options: TupleVariable(list(a.unpack_var_sequence(tx)) + b.items, **options)), ((BaseListVariable, BaseListVariable), lambda tx, a, b, options: type(a)(a.items + b.items, **options))]
        op_handlers[operator.add].extend(list_like_addition_handlers)

        def list_iadd_handler(tx, a, b, options):
            if False:
                print('Hello World!')
            if not a.mutable_local or not b.has_unpack_var_sequence(tx):
                return None
            return tx.replace_all(a, ListVariable(list(a.items) + list(b.unpack_var_sequence(tx)), **options))
        list_like_iadd_handlers = [((ListVariable, VariableTracker), list_iadd_handler), ((TupleVariable, TupleVariable), tuple_add_handler), ((TupleVariable, ConstantVariable), tuple_add_handler)]
        op_handlers[operator.iadd].extend(list_like_iadd_handlers)

        def expand_list_like(tx, lst, const, options):
            if False:
                print('Hello World!')
            return lst.__class__(items=lst.items * const.as_python_constant(), mutable_local=MutableLocal(), **options)
        list_like_expansion_handlers = [((ListVariable, ConstantVariable), expand_list_like), ((TupleVariable, ConstantVariable), expand_list_like), ((ConstantVariable, ListVariable), lambda tx, a, b, options: expand_list_like(tx, b, a, options)), ((ConstantVariable, TupleVariable), lambda tx, a, b, options: expand_list_like(tx, b, a, options))]
        op_handlers[operator.mul].extend(list_like_expansion_handlers)
        return op_handlers

    @staticmethod
    def _find_binop_handler(op, a, b):
        if False:
            return 10
        handlers = BuiltinVariable._binop_handlers()
        if op not in handlers:
            return None
        for ((type1, type2), handler) in handlers[op]:
            if isinstance(a, type1) and isinstance(b, type2):
                return handler
        return None

    def can_insert_in_graph(self):
        if False:
            i = 10
            return i + 15
        return self.fn in self._fx_graph_functions()

    def __init__(self, fn, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.fn = fn

    def __str__(self):
        if False:
            print('Hello World!')
        if self.fn is None:
            name = 'None'
        else:
            name = self.fn.__name__
        return f'{self.__class__.__name__}({name})'

    def python_type(self):
        if False:
            i = 10
            return i + 15
        return type(self.fn)

    def as_python_constant(self):
        if False:
            print('Hello World!')
        return self.fn

    def as_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        DTYPE = {bool: torch.bool, int: torch.int64, float: torch.float64}
        if self.fn in DTYPE:
            return DTYPE[self.fn]
        return super().as_proxy()

    def reconstruct(self, codegen):
        if False:
            for i in range(10):
                print('nop')
        name = self.fn.__name__
        assert self.fn.__module__ == 'builtins'
        assert name not in codegen.tx.f_globals, 'shadowed global'
        return [codegen.create_load_global(name, False, add=True)]

    def constant_args(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return any((isinstance(i, variables.TensorVariable) for i in itertools.chain(args, kwargs.values()))) and (not any((isinstance(i, variables.GetAttrVariable) for i in itertools.chain(args, kwargs.values()))))

    def unspec_python_args(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return check_unspec_python_args(args, kwargs)

    @staticmethod
    def unwrap_unspec_args_kwargs(args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return ([x.as_python_constant() for x in args], {k: v.as_python_constant() for (k, v) in kwargs.items()})

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            while True:
                i = 10
        from . import UserFunctionVariable
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls
        args = [v.realize() for v in args]
        kwargs = {k: v.realize() for (k, v) in kwargs.items()}
        constant_args = check_constant_args(args, kwargs)
        tensor_args = self.tensor_args(*args, **kwargs)
        unspec_python_args = self.unspec_python_args(*args, **kwargs)
        has_constant_handler = self.can_constant_fold_through() and (constant_args or unspec_python_args)
        assert isinstance(args, (list, tuple))
        assert isinstance(kwargs, dict)
        if self.fn is operator.getitem and (not isinstance(args[0], variables.TensorVariable)):
            tensor_args = False
        if self.can_insert_in_graph() and tensor_args and (not (self.fn is operator.getitem and isinstance(args[0], ConstDictVariable) and isinstance(args[1], variables.TensorVariable))):
            try:
                fn = self.fn
                if self.fn in IN_PLACE_DESUGARING_MAP and isinstance(args[0], variables.ConstantVariable):
                    (fn, args) = (IN_PLACE_DESUGARING_MAP[self.fn], [args[0], args[1]])
                if self.fn is operator.getitem and isinstance(args[1], SymNodeVariable):
                    (fn, args) = (torch.select, [args[0], variables.ConstantVariable.create(0), args[1]])
                if check_numpy_ndarray_args(args, kwargs) and (not any((type(arg) == variables.TensorVariable for arg in args))):
                    proxy = tx.output.create_proxy('call_function', numpy_operator_wrapper(self.fn), *proxy_args_kwargs(args, kwargs))
                    return wrap_fx_proxy_cls(variables.NumpyNdarrayVariable, tx, proxy)
                proxy = tx.output.create_proxy('call_function', fn, *proxy_args_kwargs(args, kwargs))
                if any((isinstance(arg, FakeItemVariable) for arg in args)):
                    return wrap_fx_proxy_cls(FakeItemVariable, tx, proxy)
                elif self.unspec_python_args(*args, **kwargs):
                    (_args, _kwargs) = self.unwrap_unspec_args_kwargs(args, kwargs)
                    raw_value = self.fn(*_args, **_kwargs)
                    need_unwrap = any((x.need_unwrap for x in itertools.chain(args, kwargs.values()) if isinstance(x, variables.UnspecializedPythonVariable)))
                    return wrap_fx_proxy_cls(UnspecializedPythonVariable, tx, proxy, raw_value=raw_value, need_unwrap=need_unwrap)
                elif all((isinstance(x, SymNodeVariable) for x in args)):
                    return SymNodeVariable.create(tx, proxy, None)
                else:
                    if self.fn is operator.truediv and isinstance(args[0], variables.UnspecializedPythonVariable):
                        args[0] = args[0].convert_to_constant(tx)
                    return wrap_fx_proxy(tx, proxy)
            except NotImplementedError:
                unimplemented(f'partial tensor op: {self} {args} {kwargs}')
        if self.fn in (int, float) and isinstance(args[0], (SymNodeVariable, variables.TensorVariable)):
            if isinstance(args[0], variables.TensorVariable):
                item = args[0].call_method(tx, 'item', [], {})
            else:
                item = args[0]
            fn_ = sym_int if self.fn is int else sym_float
            out = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', fn_, (item.as_proxy(),), {}))
            return out
        if self.fn == str and args and isinstance(args[0], UserFunctionVariable):
            return variables.ConstantVariable.create(value=str(args[0].fn))
        if len(kwargs) == 0 and len(args) == 2:
            binop_handler = BuiltinVariable._find_binop_handler(self.fn, args[0], args[1])
            if binop_handler:
                res = binop_handler(tx, args[0], args[1], {})
                if res is not None:
                    return res
        handler = getattr(self, f'call_{self.fn.__name__}', None)
        if handler:
            try:
                inspect.signature(handler).bind(tx, *args, **kwargs)
            except TypeError as exc:
                if not has_constant_handler:
                    log.warning('incorrect arg count %s %s and no constant handler', handler, exc)
                handler = None
        if handler:
            try:
                result = handler(tx, *args, **kwargs)
                if result is not None:
                    return result
            except Unsupported as exc:
                if not has_constant_handler:
                    raise
                exc.remove_from_stats()
        if has_constant_handler:
            return variables.ConstantVariable.create(self.as_python_constant()(*[x.as_python_constant() for x in args], **{k: v.as_python_constant() for (k, v) in kwargs.items()}))
        if self.fn is round:
            if len(args) > 0 and isinstance(args[0], SymNodeVariable):
                raise UserError(UserErrorType.STANDARD_LIBRARY, 'Calling round() on symbolic value is not supported. You can use floor() to implement this functionality', case_name='dynamic_shape_round')
        return super().call_function(tx, args, kwargs)

    def _call_min_max(self, tx, *args):
        if False:
            print('Hello World!')
        if len(args) == 1 and args[0].has_unpack_var_sequence(tx):
            items = args[0].unpack_var_sequence(tx)
            return self._call_min_max_seq(tx, items)
        elif len(args) == 2:
            return self._call_min_max_binary(tx, args[0], args[1])
        elif len(args) > 2:
            return self._call_min_max_seq(tx, args)

    def _call_min_max_seq(self, tx, items):
        if False:
            i = 10
            return i + 15
        assert len(items) > 0
        if len(items) == 1:
            return items[0]
        return functools.reduce(functools.partial(self._call_min_max_binary, tx), items)

    def _call_min_max_binary(self, tx, a, b):
        if False:
            while True:
                i = 10
        if self.tensor_args(a, b):
            if not isinstance(a, variables.TensorVariable):
                (a, b) = (b, a)
            assert isinstance(a, variables.TensorVariable)
            if isinstance(a, FakeItemVariable):
                a = variables.TorchVariable(torch.tensor).call_function(tx, [a], {})
            if isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
                from .builder import wrap_fx_proxy_cls
                return wrap_fx_proxy_cls(type(a), tx=tx, proxy=tx.output.create_proxy('call_function', self.fn, *proxy_args_kwargs([a, b], {})))
            if b.is_python_constant():
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np
                    fn = variables.NumpyVariable(np.clip)
                else:
                    fn = variables.TorchVariable(torch.clamp)
                kwargs = {'min': b} if self.fn is max else {'max': b}
                result = fn.call_function(tx, [a], kwargs)
            else:
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np
                    fn = {max: np.maximum, min: np.minimum}[self.fn]
                    fn = variables.NumpyVariable(fn)
                else:
                    fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                    fn = variables.TorchVariable(fn)
                result = fn.call_function(tx, [a, b], {})
            if all((isinstance(i, (variables.UnspecializedPythonVariable, variables.ConstantVariable)) for i in [a, b])):
                if any((isinstance(val, FakeItemVariable) for val in [a, b])):
                    return variables.FakeItemVariable.from_tensor_variable(result)
                if b.is_python_constant():
                    raw_b = b.as_python_constant()
                else:
                    raw_b = b.raw_value
                if self.fn is max:
                    raw_res = max(a.raw_value, raw_b)
                else:
                    raw_res = min(a.raw_value, raw_b)
                need_unwrap = any((x.need_unwrap for x in [a, b] if isinstance(x, variables.UnspecializedPythonVariable)))
                return variables.UnspecializedPythonVariable.from_tensor_variable(result, raw_res, need_unwrap)
            else:
                return result
        elif isinstance(a, variables.ConstantVariable) and isinstance(b, variables.ConstantVariable):
            if self.fn is max:
                return variables.ConstantVariable.create(max(a.value, b.value))
            else:
                return variables.ConstantVariable.create(min(a.value, b.value))
        elif isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
            proxy = tx.output.create_proxy('call_function', self.fn, *proxy_args_kwargs([a, b], {}))
            return SymNodeVariable.create(tx, proxy, None)
        else:
            unimplemented(f'unsupported min / max over args {str(a)}, {str(b)}')
    call_min = _call_min_max
    call_max = _call_min_max

    def call_abs(self, tx, arg: 'VariableTracker'):
        if False:
            for i in range(10):
                print('nop')
        abs_method = BuiltinVariable(getattr).call_function(tx, [arg, ConstantVariable.create('__abs__')], {})
        return abs_method.call_function(tx, [], {})

    def call_range(self, tx, *args):
        if False:
            while True:
                i = 10
        if self.unspec_python_args(*args) or self.constant_args(*args):
            return variables.RangeVariable(args)
        elif self._dynamic_args(*args):
            args = [variables.ConstantVariable.create(guard_if_dyn(arg)) for arg in args]
            return variables.RangeVariable(args)
        return None

    def _dynamic_args(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return any((isinstance(x, SymNodeVariable) for x in args)) or any((isinstance(x, SymNodeVariable) for x in kwargs.values()))

    def call_slice(self, tx, *args):
        if False:
            print('Hello World!')
        return variables.SliceVariable(args)

    def _dyn_proxy(self, tx, *args, **kwargs):
        if False:
            return 10
        from .builder import wrap_fx_proxy
        return wrap_fx_proxy(tx, tx.output.create_proxy('call_function', self.fn, *proxy_args_kwargs(args, kwargs)))

    def _call_iter_tuple_list(self, tx, obj=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self._dynamic_args(*args, **kwargs):
            return self._dyn_proxy(tx, *args, **kwargs)
        if isinstance(obj, variables.IteratorVariable):
            return obj
        if self.fn == set:
            cls = SetVariable
        else:
            cls = variables.BaseListVariable.cls_for(self.fn)
        if obj is None:
            if cls is SetVariable:
                return cls([], mutable_local=MutableLocal())
            else:
                return cls([], mutable_local=MutableLocal())
        elif obj.has_unpack_var_sequence(tx):
            if obj.source and (not is_constant_source(obj.source)):
                if isinstance(obj, TupleIteratorVariable):
                    install_guard(obj.source.make_guard(GuardBuilder.TUPLE_ITERATOR_LEN))
                else:
                    install_guard(obj.source.make_guard(GuardBuilder.LIST_LENGTH))
            if cls is SetVariable:
                return cls(list(obj.unpack_var_sequence(tx)), mutable_local=MutableLocal())
            return cls(list(obj.unpack_var_sequence(tx)), mutable_local=MutableLocal())
    call_iter = _call_iter_tuple_list
    call_tuple = _call_iter_tuple_list
    call_list = _call_iter_tuple_list
    call_set = _call_iter_tuple_list

    def call_callable(self, tx, arg):
        if False:
            return 10
        from .functions import BaseUserFunctionVariable
        if isinstance(arg, (variables.UserDefinedClassVariable, BaseUserFunctionVariable)):
            return variables.ConstantVariable.create(True)

    def call_cast(self, _, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if len(args) == 2:
            return args[1]
        unimplemented(f'unsupported args to builtin cast(): {args} {kwargs}')

    def call_dict(self, tx, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return BuiltinVariable.call_custom_dict(tx, dict, *args, **kwargs)

    @staticmethod
    def call_custom_dict(tx, user_cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not kwargs:
            if not args:
                args = ({},)
            assert len(args) == 1
            arg = args[0]
            if isinstance(arg, dict):
                return ConstDictVariable(arg, user_cls, mutable_local=MutableLocal())
            elif isinstance(arg, variables.ConstDictVariable):
                return arg.clone(user_cls=user_cls, mutable_local=MutableLocal())
            elif isinstance(arg, (ListVariable, TupleVariable, ListIteratorVariable)):
                items = user_cls()
                for x in arg.unpack_var_sequence(tx):
                    (k, v) = x.unpack_var_sequence(tx)
                    k = ConstDictVariable.get_key(k)
                    items.update({k: v})
                return ConstDictVariable(items, user_cls, mutable_local=MutableLocal())
        elif not args and kwargs:
            return variables.ConstDictVariable(dict(kwargs), user_cls=user_cls, mutable_local=MutableLocal())
        unimplemented(f'dict(): {args} {kwargs}')

    def call_zip(self, tx, *args):
        if False:
            i = 10
            return i + 15
        if all((x.has_unpack_var_sequence(tx) for x in args)):
            items = [variables.TupleVariable(list(item)) for item in zip(*[arg.unpack_var_sequence(tx) for arg in args])]
            return variables.TupleVariable(items)

    def call_enumerate(self, tx, *args):
        if False:
            return 10
        if len(args) == 1:
            start = 0
        else:
            assert len(args) == 2
            assert isinstance(args[1], variables.ConstantVariable)
            start = args[1].as_python_constant()
        if args[0].has_unpack_var_sequence(tx):
            items = [variables.TupleVariable([variables.ConstantVariable.create(idx), var]) for (idx, var) in enumerate(args[0].unpack_var_sequence(tx), start)]
            return variables.TupleVariable(items)

    def call_len(self, tx, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return args[0].call_method(tx, '__len__', args[1:], kwargs)

    def call_getitem(self, tx, *args, **kwargs):
        if False:
            while True:
                i = 10
        return args[0].call_method(tx, '__getitem__', args[1:], kwargs)

    def call_isinstance(self, tx, arg, isinstance_type):
        if False:
            print('Hello World!')
        arg_type = arg.python_type()
        isinstance_type = isinstance_type.as_python_constant()
        if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:

            def _tensor_isinstance(tensor_var, tensor_type):
                if False:
                    i = 10
                    return i + 15

                def check_type(ty):
                    if False:
                        i = 10
                        return i + 15
                    if ty not in tensortype_to_dtype:
                        return issubclass(arg.python_type(), ty)
                    dtypes = tensortype_to_dtype[ty]
                    return arg.dtype in dtypes
                if type(tensor_type) is tuple:
                    return any((check_type(ty) for ty in tensor_type))
                else:
                    return check_type(tensor_type)
            return variables.ConstantVariable.create(_tensor_isinstance(arg, isinstance_type))
        if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(arg.value, types.MemberDescriptorType):
            unimplemented(f'isinstance called on UserDefinedClass {arg} {isinstance_type}')
        if isinstance(arg, variables.UserDefinedObjectVariable) and '__instancecheck__' in isinstance_type.__class__.__dict__:
            return variables.ConstantVariable.create(isinstance_type.__class__.__instancecheck__(isinstance_type, arg.value))
        try:
            val = issubclass(arg_type, isinstance_type)
        except TypeError:
            val = arg_type is isinstance_type
        return variables.ConstantVariable.create(val)

    def call_issubclass(self, tx, left_ty, right_ty):
        if False:
            i = 10
            return i + 15
        'Checks if first arg is subclass of right arg'
        left_ty = left_ty.as_python_constant()
        right_ty = right_ty.as_python_constant()
        return variables.ConstantVariable(issubclass(left_ty, right_ty))

    def call_super(self, tx, a, b):
        if False:
            i = 10
            return i + 15
        return variables.SuperVariable(a, b)

    def call_next(self, tx, arg):
        if False:
            while True:
                i = 10
        if isinstance(arg, (variables.ListIteratorVariable, variables.IteratorVariable)):
            (val, next_iter) = arg.next_variables(tx)
            return val
        elif isinstance(arg, variables.BaseListVariable):
            return arg.items[0]

    def call_hasattr(self, tx, obj, attr):
        if False:
            print('Hello World!')
        if attr.is_python_constant():
            name = attr.as_python_constant()
            return obj.call_hasattr(tx, name)

    def call_map(self, tx, fn, seq):
        if False:
            for i in range(10):
                print('nop')
        if seq.has_unpack_var_sequence(tx):
            items = [fn.call_function(tx, [x], {}) for x in seq.unpack_var_sequence(tx)]
            return variables.TupleVariable(items)

    def call_sum(self, tx, seq, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(seq, (variables.ListVariable, variables.TupleVariable)) and all((isinstance(x, variables.ConstantVariable) and isinstance(x.value, (int, float)) for x in seq.items)) and (not kwargs):
            new_list = [x.value for x in seq.items]
            return variables.ConstantVariable.create(sum(new_list))
        if seq.has_unpack_var_sequence(tx):
            start = kwargs.pop('start', variables.ConstantVariable.create(0)).as_python_constant()
            assert not kwargs
            items = seq.unpack_var_sequence(tx)[start:]
            return BuiltinVariable(functools.reduce).call_function(tx, [BuiltinVariable(operator.add), variables.TupleVariable(items), variables.ConstantVariable.create(0)], {})

    def call_reduce(self, tx, function, iterable, initializer=None):
        if False:
            i = 10
            return i + 15
        if iterable.has_unpack_var_sequence(tx):
            items = iterable.unpack_var_sequence(tx)
            if initializer is None:
                (value, items) = (items[0], items[1:])
            else:
                value = initializer
            for element in items:
                value = function.call_function(tx, [value, element], {})
            return value

    def call_getattr(self, tx, obj: VariableTracker, name_var: VariableTracker, default=None):
        if False:
            i = 10
            return i + 15
        from .. import trace_rules
        from . import ConstantVariable, GetAttrVariable, PythonModuleVariable, TorchVariable, UserFunctionVariable
        from .builder import SourcelessBuilder, VariableBuilder
        name = name_var.as_python_constant()
        if not name_var.is_python_constant():
            unimplemented('non-const getattr() name')
        if tx.output.side_effects.is_attribute_mutation(obj):
            try:
                return tx.output.side_effects.load_attr(obj, name)
            except KeyError:
                pass
        if default is not None:
            hasattr_var = self.call_hasattr(tx, obj, name_var)
            assert hasattr_var.as_python_constant() in (True, False)
            if not hasattr_var.as_python_constant():
                return default
        options = {}
        if obj.source:
            source = AttrSource(obj.source, name)
            options['source'] = source
        else:
            source = None
        if name == '__bases__':
            try:
                value = obj.as_python_constant()
                if isinstance(value, type):
                    bases = value.__bases__
                    if source is not None:
                        tuple_args = [VariableBuilder(tx, GetItemSource(source, i))(b) for (i, b) in enumerate(bases)]
                    else:
                        tuple_args = [SourcelessBuilder()(tx, b) for b in bases]
                    return variables.TupleVariable(tuple_args, **options)
            except NotImplementedError:
                pass
        if isinstance(obj, variables.NNModuleVariable):
            return obj.var_getattr(tx, name)
        elif isinstance(obj, variables.TensorVariable) and name == 'grad':
            if source:
                for grapharg in tx.output.graphargs:
                    if grapharg.source == source.base:
                        old_grad = grapharg.example.grad
                        new_grad = obj.as_proxy().node.meta['example_value'].grad

                        def _grad_changed(old, new):
                            if False:
                                i = 10
                                return i + 15
                            if old is None or new is None:
                                return new is not old
                            try:
                                if old.shape != new.shape:
                                    return True
                                if old.stride() != new.stride():
                                    return True
                                return False
                            except TypeError as te:
                                unimplemented(str(te))
                        if _grad_changed(old_grad, new_grad):
                            if new_grad is not None:
                                grad_shape_specialized = [int(x) for x in new_grad.shape]
                                grapharg.example.grad = torch.zeros(grad_shape_specialized, device=new_grad.device)
                            else:
                                grapharg.example.grad = None
                        return VariableBuilder(tx, source)(grapharg.example.grad)
                unimplemented('tensor grad')
            else:
                unimplemented('tensor grad')
        elif isinstance(obj, (variables.TensorVariable, variables.NamedTupleVariable, variables.ConstantVariable, variables.UserDefinedClassVariable, variables.UserDefinedObjectVariable)):
            try:
                return obj.var_getattr(tx, name).clone(source=source)
            except NotImplementedError:
                return GetAttrVariable(obj, name, **options)
        elif isinstance(obj, TorchVariable):
            member = getattr(obj.value, name)
            if is_utils_checkpoint(member):
                options['source'] = source
                return build_checkpoint_variable(**options)
            elif trace_rules.lookup(member) is not None:
                return trace_rules.lookup(member)(member, **options)
            elif source is not None:
                return VariableBuilder(tx, source)(member)
            else:
                return SourcelessBuilder()(tx, member)
        elif isinstance(obj, (PythonModuleVariable, DummyModule)):
            member = obj.value.__dict__[name]
            if config.replay_record_enabled:
                tx.exec_recorder.record_module_access(obj.value, name, member)
            return VariableBuilder(tx, source)(member)
        elif istype(obj, UserFunctionVariable) and name in ('__name__', '__module__'):
            return ConstantVariable.create(getattr(obj.fn, name))
        else:
            try:
                return obj.var_getattr(tx, name).clone(source=source)
            except NotImplementedError:
                return GetAttrVariable(obj, name, **options)

    def call_setattr(self, tx, obj: VariableTracker, name_var: VariableTracker, val: VariableTracker):
        if False:
            i = 10
            return i + 15
        from .distributed import PlacementVariable
        if isinstance(obj, (variables.DataClassVariable, variables.CustomizedDictVariable, PlacementVariable)):
            return obj.call_method(tx, '__setattr__', [name_var, val], {})
        elif tx.output.side_effects.is_attribute_mutation(obj) and name_var.is_python_constant():
            name = name_var.as_python_constant()
            if name == 'requires_grad' and isinstance(obj, variables.TensorVariable):
                unimplemented('mutating requires_grad can introduce a new leaf from non-leaf or vice versa in the middle of the graph, which aot_autograd does not currently know how to handle. ')
            tx.output.side_effects.store_attr(obj, name, val)
            return val
        elif isinstance(obj, variables.UserDefinedObjectVariable):
            unimplemented(f'setattr(UserDefinedObjectVariable) {type(obj.value).__setattr__}')
        elif isinstance(obj, variables.NNModuleVariable):
            if not tx.output.is_root_tracer():
                raise AttributeMutationError("Can't inplace modify module params/buffers inside HigherOrderOp")
            if name_var.is_python_constant() and isinstance(val, variables.TensorVariable):
                assigning_fake_val = get_fake_value(val.as_proxy().node, tx)
                try:
                    getattr_var = obj.var_getattr(tx, name_var.as_python_constant())
                except AttributeError:
                    getattr_var = None
                if isinstance(getattr_var, variables.TensorVariable):
                    existing_fake_attr = get_fake_value(getattr_var.as_proxy().node, tx)
                    mod_setattr = inspect.getattr_static(obj.module_type, '__setattr__')
                    if existing_fake_attr is assigning_fake_val and mod_setattr is torch.nn.Module.__setattr__:
                        return getattr_var
            obj.convert_to_unspecialized(tx)
        elif isinstance(obj, variables.dicts.HFPretrainedConfigVariable) and tx.export:
            if name_var.is_python_constant() and isinstance(val, variables.ConstantVariable):
                setattr(obj.obj, name_var.as_python_constant(), val.as_python_constant())
                return ConstantVariable(None)

    def call_delattr(self, tx, obj: VariableTracker, name_var: VariableTracker):
        if False:
            while True:
                i = 10
        return self.call_setattr(tx, obj, name_var, variables.DeletedVariable())

    def call_type(self, tx, obj: VariableTracker):
        if False:
            i = 10
            return i + 15
        from .builder import SourcelessBuilder, VariableBuilder
        try:
            py_type = obj.python_type()
        except NotImplementedError as error:
            raise UserError(UserErrorType.INVALID_INPUT, str(error), case_name='unknown_python_type') from None
        if obj.source is None:
            return SourcelessBuilder()(tx, py_type)
        else:
            return VariableBuilder(tx, TypeSource(obj.source))(py_type)

    def call_reversed(self, tx, obj: VariableTracker):
        if False:
            return 10
        if obj.has_unpack_var_sequence(tx):
            items = list(reversed(obj.unpack_var_sequence(tx)))
            return variables.TupleVariable(items)

    def call_sorted(self, tx, obj: VariableTracker, **kwargs):
        if False:
            print('Hello World!')
        if obj.has_unpack_var_sequence(tx) and (not isinstance(obj, variables.TensorVariable)) and all((x.is_python_constant() for x in obj.unpack_var_sequence(tx))):
            function = kwargs.pop('key', None)
            reverse = kwargs.pop('reverse', ConstantVariable.create(False)).as_python_constant()
            assert len(kwargs) == 0
            if function:
                items = sorted(obj.unpack_var_sequence(tx), key=lambda x: function.call_function(tx, [x], {}).as_python_constant(), reverse=reverse)
            else:
                items = sorted(obj.unpack_var_sequence(tx), key=lambda x: x.as_python_constant(), reverse=reverse)
            return variables.ListVariable(items)

    def call_chain(self, tx, *args):
        if False:
            i = 10
            return i + 15
        if all((obj.has_unpack_var_sequence(tx) for obj in args)):
            items = []
            for obj in args:
                items.extend(obj.unpack_var_sequence(tx))
            return variables.TupleVariable(items)

    def call_islice(self, tx, iterable, *args):
        if False:
            for i in range(10):
                print('nop')
        if iterable.has_unpack_var_sequence(tx) and all((x.is_python_constant() for x in args)):
            const_args = [x.as_python_constant() for x in args]
            items = iterable.unpack_var_sequence(tx)
            items = list(itertools.islice(items, *const_args))
            return variables.TupleVariable(items)

    def call_neg(self, tx, a):
        if False:
            i = 10
            return i + 15
        if isinstance(a, SymNodeVariable):
            return SymNodeVariable.create(tx, operator.neg(a.as_proxy()), sym_num=None)
        return None

    def call_id(self, tx, *args):
        if False:
            for i in range(10):
                print('nop')
        if len(args) > 0 and isinstance(args[0], variables.NNModuleVariable):
            nn_mod_variable = args[0]
            mod = tx.output.get_submodule(nn_mod_variable.module_key)
            return variables.ConstantVariable.create(id(mod))
        else:
            unimplemented(f'call_id with args {args}')

    def _comparison(self, tx, left, right):
        if False:
            for i in range(10):
                print('nop')
        '\n        Used to implement comparison operators for different types.\n        For example, list1 < list2 is implemented differently from tensor1 < tensor2\n        '
        from . import BaseListVariable, ConstantVariable, NNModuleVariable, TensorVariable, UserDefinedObjectVariable, UserFunctionVariable
        from .lists import SizeVariable
        from .tensor import supported_const_comparison_ops, supported_tensor_comparison_ops
        op = self.fn

        def _unimplemented():
            if False:
                for i in range(10):
                    print('nop')
            unimplemented(f'comparison {typestr(left)} {op} {typestr(right)}')
        if all((isinstance(x, (NNModuleVariable, ConstantVariable)) for x in [left, right])) and op in supported_const_comparison_ops.values():
            left = tx.output.get_submodule(left.module_key) if isinstance(left, NNModuleVariable) else left.as_python_constant()
            right = tx.output.get_submodule(right.module_key) if isinstance(right, NNModuleVariable) else right.as_python_constant()
            return ConstantVariable.create(op(left, right))
        if isinstance(left, UserFunctionVariable):
            if op not in supported_const_comparison_ops.values():
                _unimplemented()
            if not isinstance(right, UserFunctionVariable):
                _unimplemented()
            return ConstantVariable.create(op(left.fn, right.fn))
        if isinstance(left, (SizeVariable, TupleVariable)) and isinstance(right, (TupleVariable, SizeVariable)):
            return BaseListVariable.list_compare(tx, op, left, right)
        if isinstance(left, BaseListVariable):
            if not type(left) == type(right):
                _unimplemented()
            return BaseListVariable.list_compare(tx, op, left, right)
        if isinstance(left, SetVariable):
            if not type(left) == type(right):
                _unimplemented()
            return ConstantVariable.create(op(left._underlying_items, right._underlying_items))
        if isinstance(left, TensorVariable) or isinstance(right, TensorVariable):
            from .builder import wrap_fx_proxy_cls
            if op is operator.is_ and isinstance(right, TensorVariable):
                return ConstantVariable.create(id(extract_fake_example_value(left.as_proxy().node)) == id(extract_fake_example_value(right.as_proxy().node)))
            if op not in supported_tensor_comparison_ops.values():
                _unimplemented()
            if isinstance(left, TensorVariable) and isinstance(right, TensorVariable) and ((left.size and right.size) is not None) and (left.size != right.size):
                try:
                    torch.broadcast_shapes(left.size, right.size)
                except RuntimeError:
                    _unimplemented()
            tensor_cls = left if isinstance(left, TensorVariable) else right
            return wrap_fx_proxy_cls(type(tensor_cls), tx, proxy)
        if isinstance(left, SymNodeVariable) or isinstance(right, SymNodeVariable):
            if op not in supported_tensor_comparison_ops.values():
                _unimplemented()
            proxy = tx.output.create_proxy('call_function', op, (left.as_proxy(), right.as_proxy()), {})
            return SymNodeVariable.create(tx, proxy, sym_num=None)
        if isinstance(left, ConstantVariable) and isinstance(right, ConstantVariable):
            return ConstantVariable.create(op(left.value, right.value))
        if isinstance(left, UserDefinedObjectVariable) and isinstance(right, UserDefinedObjectVariable):
            return ConstantVariable.create(op(left.value, right.value))
        if (isinstance(left, StreamVariable) and isinstance(right, StreamVariable) or (isinstance(left, EventVariable) and isinstance(right, EventVariable))) and op is operator.eq:
            return ConstantVariable(op(left.value, right.value))
        if op.__name__ == 'is_':
            if type(left) is not type(right):
                return ConstantVariable.create(False)
        _unimplemented()

    def call_and_(self, tx, a, b):
        if False:
            while True:
                i = 10
        if isinstance(a, (SymNodeVariable, ConstantVariable)) and isinstance(b, (SymNodeVariable, ConstantVariable)):
            return SymNodeVariable.create(tx, tx.output.create_proxy('call_function', operator.and_, *proxy_args_kwargs([a, b], {})), sym_num=None)
        return None

    def call_or_(self, tx, a, b):
        if False:
            return 10
        if isinstance(a, (SymNodeVariable, ConstantVariable)) and isinstance(b, (SymNodeVariable, ConstantVariable)):
            return SymNodeVariable.create(tx, tx.output.create_proxy('call_function', operator.or_, *proxy_args_kwargs([a, b], {})), sym_num=None)
        return None

    def call_not_(self, tx, a):
        if False:
            while True:
                i = 10
        if isinstance(a, SymNodeVariable):
            return SymNodeVariable.create(tx, tx.output.create_proxy('call_function', operator.not_, *proxy_args_kwargs([a], {})), sym_num=None)
        if isinstance(a, ListVariable):
            return ConstantVariable.create(len(a.items) == 0)
        return None
    call_eq = _comparison
    call_gt = _comparison
    call_lt = _comparison
    call_ge = _comparison
    call_le = _comparison
    call_ne = _comparison
    call_is_ = _comparison
    call_is_not = _comparison

    def call_all(self, tx, *args, **kwargs):
        if False:
            print('Hello World!')
        from .builder import SourcelessBuilder
        return tx.inline_user_function_return(SourcelessBuilder()(tx, polyfill.all), args, kwargs)