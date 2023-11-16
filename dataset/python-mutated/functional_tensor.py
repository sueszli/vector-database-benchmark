import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
not_implemented_log = torch._logging.getArtifactLogger(__name__, 'not_implemented')

class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensormode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """
    elem: torch.Tensor
    _mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
    _extra_dispatch_keys = torch._C._additional_keys_to_prop_for_wrapper_tensors.add(torch._C.DispatchKey.ZeroTensor)
    metadata_fns = [torch.ops.aten.is_contiguous.default, torch.ops.aten.is_contiguous.memory_format, torch.ops.aten.is_strides_like_format.default, torch.ops.aten.is_non_overlapping_and_dense.default, torch.ops.aten.size.default, torch.ops.aten.sym_size.default, torch.ops.aten.stride.default, torch.ops.aten.sym_stride.default, torch.ops.aten.storage_offset.default, torch.ops.aten.sym_storage_offset.default, torch.ops.aten.numel.default, torch.ops.aten.sym_numel.default, torch.ops.aten.dim.default]

    def __new__(cls, elem):
        if False:
            while True:
                i = 10
        assert torch._is_functional_tensor(elem)
        extra_dispatch_keys = FunctionalTensor._extra_dispatch_keys & torch._C._dispatch_keys(elem)
        out = torch.Tensor._make_wrapper_subclass(cls, elem.shape, elem.stride(), elem.storage_offset(), None, elem.dtype, elem.layout, elem.device, False, elem.requires_grad, 'sizes', False, False, extra_dispatch_keys)
        out.elem = elem
        return out
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            return 10
        unrecognized_types = [t for t in types if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]]
        if unrecognized_types:
            not_implemented_log.debug('FunctionalTensor unrecognized subclass(es): %s', unrecognized_types)
            return NotImplemented
        if kwargs is None:
            kwargs = {}
        if func in FunctionalTensor.metadata_fns:

            def unwrap(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.elem
            assert len(args) == 1 and isinstance(args[0], FunctionalTensor)
            assert len(kwargs) == 0
            return func(args[0].elem)
        raise RuntimeError('Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()')

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'FunctionalTensor({repr(self.elem)})'

    @staticmethod
    def to_functional(x):
        if False:
            for i in range(10):
                print('nop')
        assert not torch._is_functional_tensor(x)
        x_functional = torch._to_functional_tensor(x)
        with FunctionalTensorMode():
            torch._mirror_autograd_meta_to(x, x_functional)
            out = FunctionalTensor(x_functional)
            torch._mirror_autograd_meta_to(x_functional, out)
        return out

    def from_functional(self):
        if False:
            i = 10
            return i + 15
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)

    def replace_(self, output) -> None:
        if False:
            i = 10
            return i + 15
        torch._functionalize_replace(self.elem, output)

    def commit_update(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        torch._functionalize_commit_update(self.elem)

    def sync(self) -> None:
        if False:
            return 10
        torch._functionalize_sync(self.elem)

    def mark_mutation_hidden_from_autograd(self) -> None:
        if False:
            i = 10
            return i + 15
        torch._functionalize_mark_mutation_hidden_from_autograd(self.elem)

class FunctionalTensorMode(TorchDispatchMode):

    def __init__(self):
        if False:
            print('Hello World!')
        self.is_on_stack = False
        self.enter_stack = []
        self._mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
        self.decompose_composite_implicit_ops = True

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL) is None:
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            self.enter_stack.append(False)
            return self

    def __exit__(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        is_on_stack = self.enter_stack.pop()
        if is_on_stack:
            super().__exit__(a, b, c)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        if kwargs is None:
            kwargs = {}
        unrecognized_types = [t for t in types if not issubclass(t, torch._subclasses.FakeTensor) and t not in [torch.Tensor, FunctionalTensor]]
        if unrecognized_types:
            not_implemented_log.debug('FunctionalTensor unrecognized subclass(es): %s', unrecognized_types)
            return NotImplemented
        if func not in FunctionalTensor.metadata_fns and self.decompose_composite_implicit_ops and torch._C._dispatch_has_kernel(func.name()):
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        def assert_is_functional(x):
            if False:
                return 10
            assert torch._is_functional_tensor(x)

        def wrap(x):
            if False:
                while True:
                    i = 10
            assert not isinstance(x, FunctionalTensor)
            if isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
                return FunctionalTensor(x)
            return x
        any_functional_inputs = False

        def unwrap(x):
            if False:
                print('Hello World!')
            any_functional_inputs = True
            return x.elem
        (args_unwrapped, kwargs_unwrapped) = pytree.tree_map_only(FunctionalTensor, unwrap, (args, kwargs))
        is_included = torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Functionalize)
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.Functionalize)
        assert is_excluded or not is_included
        include_to_set = torch._C._dispatch_tls_local_include_set() | torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set().remove(torch._C.DispatchKey.Functionalize) - FunctionalTensor._extra_dispatch_keys
        with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
            try:
                old_apply_views = torch._functionalize_enable_reapply_views(True)
                outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                outs_wrapped = pytree.tree_map_only(torch.Tensor, wrap, outs_unwrapped)
            finally:
                torch._disable_functionalization()
                torch._functionalize_enable_reapply_views(old_apply_views)
        is_included = torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Functionalize)
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.Functionalize)
        assert is_excluded or not is_included
        if not any((isinstance(x, FunctionalTensor) for x in pytree.tree_leaves(outs_wrapped))) or func == torch.ops.aten.lift_fresh.default:
            return outs_wrapped
        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)

@contextlib.contextmanager
def maybe_disable_functional_mode():
    if False:
        while True:
            i = 10
    maybe_func_mode = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    try:
        yield
    finally:
        if maybe_func_mode is not None:
            torch._C._set_dispatch_mode(maybe_func_mode)

@contextlib.contextmanager
def unset_functional_temporarily():
    if False:
        print('Hello World!')
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    try:
        yield old
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)

def dispatch_functionalize(func):
    if False:
        return 10

    def to_fun(t):
        if False:
            return 10
        if isinstance(t, torch.Tensor):
            return FunctionalTensor.to_functional(t)
        return t

    def from_fun(t):
        if False:
            return 10
        if not isinstance(t, FunctionalTensor):
            if isinstance(t, torch.Tensor):
                assert not torch._is_functional_tensor(t)
            return t
        torch._sync(t)
        return torch._from_functional_tensor(t.elem)

    def inner(*args, **kwargs):
        if False:
            while True:
                i = 10
        func_args = pytree.tree_map_only(torch.Tensor, to_fun, args)
        func_kwargs = pytree.tree_map_only(torch.Tensor, to_fun, kwargs)
        flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
        flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)
        disable_above = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with disable_above, FunctionalTensorMode():
            func_outputs = func(*func_args, **func_kwargs)
            outputs = pytree.tree_map_only(FunctionalTensor, from_fun, func_outputs)
            return outputs
    return inner

class BaseFunctionalizeAPI(ABC):

    @abstractmethod
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def functionalize(self, inner_f: Callable) -> Callable:
        if False:
            return 10
        pass

    @abstractmethod
    def redispatch_to_next(self) -> ContextManager:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def replace(self, input_tensor, output_tensor) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def commit_update(self, tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def sync(self, tensor) -> None:
        if False:
            return 10
        pass

    @abstractmethod
    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        if False:
            return 10
        pass

class PythonFunctionalizeAPI(BaseFunctionalizeAPI):

    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            for i in range(10):
                print('nop')
        return torch.utils._pytree.tree_map_only(FunctionalTensor, FunctionalTensor.to_functional, args)

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        return torch.utils._pytree.tree_map_only(FunctionalTensor, FunctionalTensor.from_functional, args)

    def functionalize(self, inner_f: Callable) -> Callable:
        if False:
            while True:
                i = 10
        return dispatch_functionalize(inner_f)

    def redispatch_to_next(self) -> ContextManager:
        if False:
            while True:
                i = 10
        return unset_functional_temporarily()

    def replace(self, input_tensor, output_tensor) -> None:
        if False:
            return 10
        assert isinstance(input_tensor, FunctionalTensor)
        assert not isinstance(output_tensor, FunctionalTensor)
        input_tensor.replace_(output_tensor)

    def commit_update(self, tensor) -> None:
        if False:
            while True:
                i = 10
        assert isinstance(tensor, FunctionalTensor)
        tensor.commit_update()

    def sync(self, tensor) -> None:
        if False:
            while True:
                i = 10
        assert isinstance(tensor, FunctionalTensor)
        tensor.sync()

    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        if False:
            i = 10
            return i + 15
        assert isinstance(tensor, FunctionalTensor)
        tensor.mark_mutation_hidden_from_autograd()

class CppFunctionalizeAPI(BaseFunctionalizeAPI):

    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            i = 10
            return i + 15
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional
        return _wrap_all_tensors_to_functional(args, level=0)

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            return 10
        from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional
        return _unwrap_all_tensors_from_functional(args, reapply_views=_reapply_views())

    def functionalize(self, inner_f: Callable) -> Callable:
        if False:
            print('Hello World!')
        return torch.func.functionalize(inner_f)

    def redispatch_to_next(self) -> ContextManager:
        if False:
            i = 10
            return i + 15
        return torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))

    def replace(self, input_tensor, output_tensor) -> None:
        if False:
            while True:
                i = 10
        torch._functionalize_replace(input_tensor, output_tensor)

    def commit_update(self, tensor) -> None:
        if False:
            for i in range(10):
                print('nop')
        torch._functionalize_commit_update(tensor)

    def sync(self, tensor) -> None:
        if False:
            return 10
        torch._functionalize_sync(tensor)

    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        if False:
            print('Hello World!')
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)

class FunctorchFunctionalizeAPI(BaseFunctionalizeAPI):

    def __init__(self, interpreter):
        if False:
            return 10
        self.interpreter = interpreter

    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            return 10
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional
        return _wrap_all_tensors_to_functional(args, level=self.interpreter.level())

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        if False:
            print('Hello World!')
        from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional
        return _unwrap_all_tensors_from_functional(args, reapply_views=self.interpreter.functionalize_add_back_views())

    def functionalize(self, inner_f: Callable) -> Callable:
        if False:
            return 10
        return torch.func.functionalize(inner_f, remove='mutations_and_views' if self.interpreter.functionalize_add_back_views() else 'mutations')

    def redispatch_to_next(self) -> ContextManager:
        if False:
            while True:
                i = 10
        return self.interpreter.lower()

    def replace(self, input_tensor, output_tensor) -> None:
        if False:
            i = 10
            return i + 15
        torch._functionalize_replace(input_tensor, output_tensor)

    def commit_update(self, tensor) -> None:
        if False:
            return 10
        torch._functionalize_commit_update(tensor)

    def sync(self, tensor) -> None:
        if False:
            while True:
                i = 10
        torch._functionalize_sync(tensor)

    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        if False:
            print('Hello World!')
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)