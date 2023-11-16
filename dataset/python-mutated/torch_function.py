import inspect
from typing import Dict, List
import torch.utils._pytree as pytree
from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalSource
from ..utils import is_tensor_base_attr_getter
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable
from .tensor import TensorVariable
from .user_defined import UserDefinedClassVariable
banned_attrs = [fn.__self__.__name__ for fn in get_default_nowrap_functions() if is_tensor_base_attr_getter(fn)]

def is_torch_function_user_object(obj):
    if False:
        print('Hello World!')
    return hasattr(obj, '__torch_function__')

def _is_attr_overidden(tx, var, name):
    if False:
        i = 10
        return i + 15
    import torch
    overridden = False
    try:
        attr_val = inspect.getattr_static(var.python_type(), name)
        overridden |= attr_val != getattr(torch.Tensor, name)
    except AttributeError:
        pass
    return overridden

def call_torch_function(tx, torch_function_type, torch_function_var, fn, types, args, kwargs):
    if False:
        print('Hello World!')
    tf_args = (torch_function_type, fn, types, TupleVariable(list(args)))
    return tx.inline_user_function_return(torch_function_var, tf_args, kwargs)

def build_torch_function_fn(tx, value, source):
    if False:
        return 10
    from .builder import SourcelessBuilder, VariableBuilder
    if not source:
        return VariableBuilder(tx, AttrSource(AttrSource(source, '__torch_function__'), '__func__'))(value.__torch_function__.__func__)
    else:
        return SourcelessBuilder()(tx, value.__torch_function__.__func__)

def can_dispatch_torch_function(tx, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    if tx.output.torch_function_enabled:
        all_args = pytree.arg_tree_leaves(*args, **kwargs)
        return any((isinstance(arg, TensorWithTFOverrideVariable) for arg in all_args))
    else:
        return False

def dispatch_torch_function(tx, fn, args, kwargs):
    if False:
        return 10
    'Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args'
    all_args = pytree.arg_tree_leaves(*args, **kwargs)
    overloaded_args = _get_overloaded_args([arg for arg in all_args if isinstance(arg, TensorWithTFOverrideVariable)], lambda x: x.class_type)
    for arg in overloaded_args:
        res = arg.call_torch_function(tx, fn, TupleVariable([arg.subclass_type_var() for arg in overloaded_args]), args, kwargs)
        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res
    unimplemented(f'All __torch_function__ overrides for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented')

class TensorWithTFOverrideVariable(TensorVariable):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.torch_function_fn = kwargs.pop('torch_function_fn')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_tensor_var(cls, tx, tensor_var, class_type, torch_function_fn):
        if False:
            return 10
        import torch
        kwargs = dict(tensor_var.__dict__)
        assert kwargs.pop('class_type') is torch.Tensor, 'invalid class type in TensorWithTFOverrideVariable.from_tensor_var'
        var = cls(torch_function_fn=torch_function_fn, class_type=class_type, **kwargs)
        var.install_global(tx)
        return var

    def install_global(self, tx):
        if False:
            return 10
        if self.global_mangled_class_name() not in tx.output.global_scope:
            tx.output.install_global(self.global_mangled_class_name(), self.class_type)

    def python_type(self):
        if False:
            return 10
        return self.class_type

    def subclass_type_var(self):
        if False:
            i = 10
            return i + 15
        return UserDefinedClassVariable(self.class_type, source=GlobalSource(self.global_mangled_class_name()))

    def global_mangled_class_name(self):
        if False:
            return 10
        return f'__subclass_{self.class_type.__name__}_{id(self.class_type)}'

    def var_getattr(self, tx, name):
        if False:
            for i in range(10):
                print('nop')
        import torch
        from .builder import SourcelessBuilder
        if name in banned_attrs or not hasattr(torch.Tensor, name):
            unimplemented(f'Accessing {name} on a tensor subclass with a __torch_function__ override is not supported')
        if _is_attr_overidden(tx, self, name):
            unimplemented(f'Accessing overridden method/attribute {name} on a tensor subclass with a __torch_function__ override is not supported')
        if tx.output.torch_function_enabled:
            if self.source:
                install_guard(AttrSource(AttrSource(self.source, '__class__'), name).make_guard(GuardBuilder.FUNCTION_MATCH))
            get_fn = SourcelessBuilder()(tx, getattr(torch.Tensor, name).__get__)
            return self.call_torch_function(tx, get_fn, TupleVariable([self.subclass_type_var()]), [self], {})
        else:
            return super().var_getattr(tx, name)

    def call_torch_function(self, tx, fn, types, args, kwargs):
        if False:
            while True:
                i = 10
        return call_torch_function(tx, self.subclass_type_var(), self.torch_function_fn, fn, types, args, kwargs)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            return 10
        if tx.output.torch_function_enabled:
            import torch
            from .builder import SourcelessBuilder, VariableBuilder
            if _is_attr_overidden(tx, self, name):
                unimplemented(f'Calling overridden method {name} on a tensor subclass with a __torch_function__ override is not supported')
            if self.source:
                func_var = VariableBuilder(tx, AttrSource(AttrSource(self.source, '__class__'), name))(inspect.getattr_static(self.python_type(), name))
            else:
                func_var = SourcelessBuilder()(tx, getattr(torch.Tensor, name))
            return dispatch_torch_function(tx, func_var, [self] + args, kwargs)
        else:
            return super().call_method(tx, name, args, kwargs)