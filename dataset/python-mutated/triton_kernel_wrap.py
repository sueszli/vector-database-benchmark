import threading
from typing import Any, Dict
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, ProxyTorchDispatchMode, track_tensor_tree

class KernelSideTable:
    id_to_kernel: Dict[int, Any] = dict()
    kernel_to_id: Dict[Any, int] = dict()
    lock = threading.Lock()

    def add_kernel(self, kernel) -> int:
        if False:
            return 10
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]
            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    def get_kernel(self, idx: int):
        if False:
            print('Hello World!')
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    def reset_table(self) -> None:
        if False:
            print('Hello World!')
        self.id_to_kernel = dict()
        self.kernel_to_id = dict()
kernel_side_table = KernelSideTable()

class TritonKernelWrapperMutation(HigherOrderOperator):

    def __init__(self):
        if False:
            return 10
        super().__init__('triton_kernel_wrapper_mutation')
triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()

class TritonKernelWrapperFunctional(HigherOrderOperator):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('triton_kernel_wrapper_functional')
triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()

@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(*, kernel_idx, grid, kwargs):
    if False:
        i = 10
        return i + 15
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code
    kernel = kernel_side_table.get_kernel(kernel_idx)
    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        (fn_name, code) = user_defined_kernel_grid_fn_code(kernel.fn.__name__, kernel.configs, grid)
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]
    kernel[grid_fn](**kwargs)

@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(mode, *, kernel_idx, grid, kwargs):
    if False:
        return 10
    with mode:
        return None

def trace_triton_kernel_wrapper(proxy_mode, func_overload, node_args):
    if False:
        i = 10
        return i + 15
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, (), proxy_args, name=func_overload.__name__ + '_proxy')
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)

@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(mode, *, kernel_idx, grid, kwargs):
    if False:
        while True:
            i = 10
    if mode.enable_tracing:
        trace_triton_kernel_wrapper(mode, triton_kernel_wrapper_mutation, {'kernel_idx': kernel_idx, 'grid': grid, 'kwargs': kwargs})
    else:
        triton_kernel_wrapper_mutation(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)
    return None

@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(ctx, kernel_idx, grid, kwargs):
    if False:
        print('Hello World!')
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    tensors_to_clone = [key for (key, value) in unwrapped_kwargs.items() if isinstance(value, Tensor)]
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(kernel_idx=kernel_idx, grid=grid, kwargs=unwrapped_kwargs, tensors_to_clone=tensors_to_clone)
    assert unwrapped_outputs.keys() == kwargs.keys()
    for (key, output_arg) in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        assert isinstance(input_arg, Tensor)
        ctx.replace(input_arg, output_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
    return None

@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(*, kernel_idx, grid, kwargs, tensors_to_clone):
    if False:
        return 10
    kwargs = {key: clone_preserve_strides(val) if key in tensors_to_clone else val for (key, val) in kwargs.items()}
    triton_kernel_wrapper_mutation(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)
    return kwargs

@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
def triton_kernel_wrapper_functional_fake_tensor_mode(mode, *, kernel_idx, grid, kwargs, tensors_to_clone):
    if False:
        return 10
    with mode:
        return {key: clone_preserve_strides(val) if key in tensors_to_clone else val for (key, val) in kwargs.items()}

@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(mode, *, kernel_idx, grid, kwargs, tensors_to_clone):
    if False:
        while True:
            i = 10
    if mode.enable_tracing:
        return trace_triton_kernel_wrapper(mode, triton_kernel_wrapper_functional, {'kernel_idx': kernel_idx, 'grid': grid, 'kwargs': kwargs, 'tensors_to_clone': tensors_to_clone})
    else:
        return triton_kernel_wrapper_functional(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs, tensors_to_clone=tensors_to_clone)

@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(ctx, kernel_idx, grid, kwargs, tensors_to_clone):
    if False:
        return 10
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        outputs = triton_kernel_wrapper_functional(kernel_idx=kernel_idx, grid=grid, kwargs=unwrapped_kwargs, tensors_to_clone=tensors_to_clone)
        return ctx.wrap_tensors(outputs)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonDispatcher)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonTLSSnapshot)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCPU)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCUDA)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCPU)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonDispatcher)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonTLSSnapshot)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCPU)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCPU)