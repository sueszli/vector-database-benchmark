import collections
import functools
import inspect
import operator
from typing import Dict, List, Optional
import torch
import torch.fx
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import GetItemSource
from ..utils import get_fake_value, guard_if_dyn, is_namedtuple, iter_contains, namedtuple_fields, odict_values
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable

class BaseListVariable(VariableTracker):

    @staticmethod
    def cls_for_instance(obj):
        if False:
            print('Hello World!')
        if is_namedtuple(obj):
            return functools.partial(NamedTupleVariable, tuple_cls=type(obj))
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj):
        if False:
            while True:
                i = 10
        return {iter: ListIteratorVariable, list: ListVariable, slice: SliceVariable, torch.Size: SizeVariable, tuple: TupleVariable, odict_values: ListVariable, torch.nn.ParameterList: ListVariable, torch.nn.ModuleList: ListVariable, collections.deque: DequeVariable}[obj]

    def __init__(self, items: List[VariableTracker], **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        assert isinstance(items, list)
        assert all((isinstance(x, VariableTracker) for x in items))
        self.items: List[VariableTracker] = items

    def _as_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        return [x.as_proxy() for x in self.items]

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        return self.as_python_constant()

    def as_python_constant(self):
        if False:
            print('Hello World!')
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        if False:
            while True:
                i = 10
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: VariableTracker):
        if False:
            while True:
                i = 10
        from .tensor import SymNodeVariable
        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()
        if isinstance(index, slice):
            if self.source is not None:
                return self.clone(items=self.items[index], source=GetItemSource(self.source, index), mutable_local=MutableLocal() if self.mutable_local else None)
            else:
                return self.clone(items=self.items[index], mutable_local=MutableLocal() if self.mutable_local else None)
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    def unpack_var_sequence(self, tx):
        if False:
            print('Hello World!')
        return list(self.items)

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if False:
            print('Hello World!')
        if name == '__getitem__':
            from .tensor import TensorVariable
            assert not kwargs and len(args) == 1
            if isinstance(args[0], TensorVariable):
                value = get_fake_value(args[0].as_proxy().node, tx)
                if value.constant is not None and value.constant.numel() == 1:
                    value = variables.ConstantVariable.create(value.constant.item())
                else:
                    unimplemented('__getitem__ with non-constant tensor')
            else:
                value = args[0]
            return self.getitem_const(value)
        elif name == '__contains__':
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.items, args[0], tx)
        elif name == 'index':
            from .builder import SourcelessBuilder
            return tx.inline_user_function_return(SourcelessBuilder()(tx, polyfill.index), [self] + list(args), kwargs)
        return super().call_method(tx, name, args, kwargs)

    @staticmethod
    def list_compare(tx, op, left, right):
        if False:
            i = 10
            return i + 15
        from .builtin import BuiltinVariable
        eq_result = BaseListVariable.list_eq(tx, left, right)
        if op is operator.eq:
            return eq_result
        elif op is operator.ne:
            return BuiltinVariable(operator.not_).call_function(tx, [eq_result], {})
        else:
            unimplemented(f'list_compare {left} {op} {right}')

    @staticmethod
    def list_eq(tx, left, right):
        if False:
            i = 10
            return i + 15
        from .builtin import BuiltinVariable
        if len(left.items) != len(right.items):
            return ConstantVariable.create(False)
        if len(left.items) == 0:
            return ConstantVariable.create(True)
        comps = []
        for (l, r) in zip(left.items, right.items):
            comp = BuiltinVariable(operator.eq).call_function(tx, [l, r], {})
            if comp.is_python_constant() and (not comp.as_python_constant()):
                return comp
            comps.append(comp)
        return functools.reduce(lambda a, b: BuiltinVariable(operator.and_).call_function(tx, [a, b], {}), comps)

class RangeVariable(BaseListVariable):

    def __init__(self, items, **kwargs):
        if False:
            i = 10
            return i + 15
        items_to_map = items
        start = variables.ConstantVariable.create(0)
        stop = None
        step = variables.ConstantVariable.create(1)
        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            (start, stop) = items_to_map
        elif len(items_to_map) == 3:
            (start, stop, step) = items_to_map
        else:
            raise AssertionError()
        assert stop is not None
        super().__init__([start, stop, step], **kwargs)

    def python_type(self):
        if False:
            i = 10
            return i + 15
        return range

    def as_python_constant(self):
        if False:
            for i in range(10):
                print('nop')
        return range(*[x.as_python_constant() for x in self.items])

    def as_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(self, tx):
        if False:
            while True:
                i = 10
        return [variables.ConstantVariable.create(x) for x in self.as_python_constant()]

    def reconstruct(self, codegen):
        if False:
            print('Hello World!')
        assert 'range' not in codegen.tx.f_globals
        codegen.append_output(codegen.create_load_python_module(range, True))
        codegen.foreach(self.items)
        return create_call_function(3, False)

    def var_getattr(self, tx, name):
        if False:
            while True:
                i = 10
        fields = ['start', 'stop', 'step']
        if name not in fields:
            unimplemented(f'range.{name}')
        return self.items[fields.index(name)]

class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if False:
            i = 10
            return i + 15
        if name == 'append' and self.mutable_local:
            assert not kwargs
            (arg,) = args
            tx.replace_all(self, type(self)(self.items + [arg]))
            return ConstantVariable.create(None)
        elif name == 'extend' and self.mutable_local and args and args[0].has_unpack_var_sequence(tx):
            assert not kwargs
            (arg,) = args
            return tx.replace_all(self, type(self)(list(self.items) + list(arg.unpack_var_sequence(tx))))
        elif name == 'insert' and self.mutable_local:
            assert not kwargs
            (idx, value) = args
            items = list(self.items)
            items.insert(idx.as_python_constant(), value)
            return tx.replace_all(self, type(self)(items))
        elif name == 'pop' and self.mutable_local:
            assert not kwargs
            items = list(self.items)
            result = items.pop(*[a.as_python_constant() for a in args])
            tx.replace_all(self, type(self)(items))
            return result
        elif name == 'clear' and self.mutable_local:
            assert not kwargs and (not args)
            return tx.replace_all(self, type(self)([]))
        elif name == '__setitem__' and self.mutable_local and args and args[0].is_python_constant():
            assert not kwargs
            (key, value) = args
            items = list(self.items)
            if isinstance(key, SliceVariable):
                items[key.as_python_constant()] = list(value.items)
            else:
                items[key.as_python_constant()] = value
            result = ListVariable(items)
            return tx.replace_all(self, result)
        elif name == 'copy':
            assert not kwargs
            assert not args
            items = list(self.items)
            return type(self)(items, mutable_local=MutableLocal())
        else:
            return super().call_method(tx, name, args, kwargs)

class ListVariable(CommonListMethodsVariable):

    def python_type(self):
        if False:
            i = 10
            return i + 15
        return list

    def reconstruct(self, codegen):
        if False:
            for i in range(10):
                print('nop')
        codegen.foreach(self.items)
        return [create_instruction('BUILD_LIST', arg=len(self.items))]

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if False:
            while True:
                i = 10
        if name == '__setitem__' and self.mutable_local and args and args[0].is_python_constant():
            assert not kwargs
            (key, value) = args
            items = list(self.items)
            if isinstance(key, SliceVariable):
                if not value.has_unpack_var_sequence(tx):
                    unimplemented(f'Missing dynamo support for expanding {value} into a list for slice assignment.')
                items[key.as_python_constant()] = value.unpack_var_sequence(tx)
            else:
                items[key.as_python_constant()] = value
            result = ListVariable(items)
            return tx.replace_all(self, result)
        else:
            return super().call_method(tx, name, args, kwargs)

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        if False:
            while True:
                i = 10
        if self.python_type() is not list:
            return super().call_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr([], name))

class DequeVariable(CommonListMethodsVariable):

    def python_type(self):
        if False:
            while True:
                i = 10
        return collections.deque

    def reconstruct(self, codegen):
        if False:
            i = 10
            return i + 15
        assert 'deque' not in codegen.tx.f_globals
        codegen.append_output(codegen.create_load_python_module(collections.deque, True))
        codegen.foreach(self.items)
        return create_call_function(len(self.items), False)

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if False:
            for i in range(10):
                print('nop')
        if name == '__setitem__' and self.mutable_local and args and args[0].is_python_constant():
            assert not kwargs
            (key, value) = args
            assert key.is_python_constant() and isinstance(key.as_python_constant(), int)
            items = list(self.items)
            items[key.as_python_constant()] = value
            result = DequeVariable(items)
            return tx.replace_all(self, result)
        elif name == 'extendleft' and self.mutable_local:
            assert not kwargs
            (arg,) = args
            return tx.replace_all(self, DequeVariable(list(arg.unpack_var_sequence(tx)) + list(self.items)))
        elif name == 'popleft' and self.mutable_local:
            assert not args
            assert not kwargs
            items = collections.deque(self.items)
            result = items.popleft()
            tx.replace_all(self, DequeVariable(list(items)))
            return result
        elif name == 'appendleft' and self.mutable_local:
            assert not kwargs
            return tx.replace_all(self, DequeVariable([args[0]] + list(self.items)))
        else:
            return super().call_method(tx, name, args, kwargs)

class TupleVariable(BaseListVariable):

    def python_type(self):
        if False:
            while True:
                i = 10
        return tuple

    def reconstruct(self, codegen):
        if False:
            i = 10
            return i + 15
        codegen.foreach(self.items)
        return [create_instruction('BUILD_TUPLE', arg=len(self.items))]

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if False:
            i = 10
            return i + 15
        return super().call_method(tx, name, args, kwargs)

class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    def __init__(self, items: List[VariableTracker], proxy: Optional[torch.fx.Proxy]=None, **kwargs):
        if False:
            return 10
        self.proxy = proxy
        super().__init__(items, **kwargs)

    def python_type(self):
        if False:
            while True:
                i = 10
        return torch.Size

    def as_proxy(self):
        if False:
            i = 10
            return i + 15
        if self.proxy is not None:
            return self.proxy
        tracer = None
        proxies = self._as_proxy()
        for proxy in proxies:
            if isinstance(proxy, torch.fx.Proxy):
                tracer = proxy.tracer
                break
        if tracer is None:
            return torch.Size(proxies)
        proxy = tracer.create_proxy('call_function', torch.Size, (proxies,), {})
        proxy.node.meta['example_value'] = torch.Size([p.node.meta['example_value'] if not isinstance(p, int) else p for p in proxies])
        return proxy

    def reconstruct(self, codegen):
        if False:
            print('Hello World!')
        codegen.load_import_from('torch', 'Size')
        codegen.foreach(self.items)
        build_torch_size = [create_instruction('BUILD_TUPLE', arg=len(self.items))] + create_call_function(1, True)
        return build_torch_size

    def unpack_var_sequence(self, tx):
        if False:
            print('Hello World!')
        return list(self.items)

    def numel(self, tx):
        if False:
            for i in range(10):
                print('nop')
        from .builtin import BuiltinVariable
        from .tensor import SymNodeVariable
        const_result = 1
        sym_sizes = []
        for v in self.items:
            if isinstance(v, ConstantVariable):
                const_result *= v.value
            else:
                assert isinstance(v, SymNodeVariable), type(v)
                sym_sizes.append(v)
        result = ConstantVariable.create(const_result)
        if sym_sizes and const_result == 1:
            (result, *sym_sizes) = sym_sizes
        if not sym_sizes or const_result == 0:
            return result
        mul = BuiltinVariable(operator.mul)
        for v in sym_sizes:
            result = mul.call_function(tx, [result, v], {})
        return result

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if False:
            i = 10
            return i + 15
        if name == '__getitem__':
            assert not kwargs and len(args) == 1
            out = self.get_item_dyn(tx, args[0])
            return out
        elif name == 'numel':
            assert not args and (not kwargs)
            return self.numel(tx)
        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(self, tx, arg: VariableTracker):
        if False:
            return 10
        from .tensor import SymNodeVariable
        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()
        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

class NamedTupleVariable(TupleVariable):

    def __init__(self, items, tuple_cls, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        if False:
            print('Hello World!')
        return self.tuple_cls

    def as_python_constant(self):
        if False:
            for i in range(10):
                print('nop')
        return self.python_type()(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        if False:
            return 10
        create_fn = getattr(self.tuple_cls, '_make', self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        return [create_instruction('BUILD_TUPLE', arg=len(self.items))] + create_call_function(1, True)

    def var_getattr(self, tx, name):
        if False:
            for i in range(10):
                print('nop')

        def check_and_create_method():
            if False:
                i = 10
                return i + 15
            method = inspect.getattr_static(self.tuple_cls, name, None)
            if isinstance(method, classmethod):
                return UserMethodVariable(method.__func__, variables.UserDefinedClassVariable(self.tuple_cls))
            elif isinstance(method, staticmethod):
                return UserFunctionVariable(method.__func__)
            elif inspect.isfunction(method):
                return UserMethodVariable(method, self)
            else:
                return None
        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            method = check_and_create_method()
            if not method:
                super().var_getattr(tx, name)
            return method
        return self.items[fields.index(name)]

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        if False:
            print('Hello World!')
        fields = namedtuple_fields(self.tuple_cls)
        return variables.ConstantVariable.create(name in fields)

class SliceVariable(BaseListVariable):

    def __init__(self, items, **kwargs):
        if False:
            return 10
        items_to_map = items
        (start, stop, step) = [variables.ConstantVariable.create(None)] * 3
        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            (start, stop) = items_to_map
        elif len(items_to_map) == 3:
            (start, stop, step) = items_to_map
        else:
            raise AssertionError()
        if isinstance(start, variables.TensorVariable) or isinstance(stop, variables.TensorVariable):
            unimplemented('Dynamic slicing on data-dependent value is not supported')
        super().__init__([start, stop, step], **kwargs)

    def as_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        return slice(*self._as_proxy())

    def python_type(self):
        if False:
            print('Hello World!')
        return slice

    def as_python_constant(self):
        if False:
            return 10
        return slice(*[guard_if_dyn(x) for x in self.items])

    def reconstruct(self, codegen):
        if False:
            return 10
        codegen.foreach(self.items)
        return [create_instruction('BUILD_SLICE', arg=len(self.items))]

    def var_getattr(self, tx, name):
        if False:
            for i in range(10):
                print('nop')
        fields = ['start', 'stop', 'step']
        if name not in fields:
            unimplemented(f'slice.{name}')
        return self.items[fields.index(name)]

class ListIteratorVariable(VariableTracker):

    def __init__(self, items, index: int=0, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        assert isinstance(items, list)
        self.items = items
        self.index = index

    def next_variables(self, tx):
        if False:
            print('Hello World!')
        assert self.mutable_local
        if self.index >= len(self.items):
            raise StopIteration()
        next_iter = ListIteratorVariable(self.items, self.index + 1, mutable_local=MutableLocal())
        tx.replace_all(self, next_iter)
        return (self.items[self.index], next_iter)

    def as_python_constant(self):
        if False:
            while True:
                i = 10
        if self.index > 0:
            raise NotImplementedError()
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        if False:
            return 10
        return list(self.items[self.index:])

    def reconstruct(self, codegen):
        if False:
            print('Hello World!')
        remaining_items = self.items[self.index:]
        codegen.foreach(remaining_items)
        return [create_instruction('BUILD_TUPLE', arg=len(remaining_items)), create_instruction('GET_ITER')]

class TupleIteratorVariable(ListIteratorVariable):
    pass