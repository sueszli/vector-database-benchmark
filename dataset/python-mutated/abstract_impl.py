import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle

class AbstractImplHolder:
    """A holder where one can register an abstract impl to."""

    def __init__(self, qualname: str):
        if False:
            for i in range(10):
                print('nop')
        self.qualname: str = qualname
        self.kernel: Optional[Kernel] = None
        self.lib: Optional[torch.library.Library] = None

    def register(self, func: Callable, source: str) -> RegistrationHandle:
        if False:
            print('Hello World!')
        'Register an abstract impl.\n\n        Returns a RegistrationHandle that one can use to de-register this\n        abstract impl.\n        '
        if self.kernel is not None:
            raise RuntimeError(f'impl_abstract(...): the operator {self.qualname} already has an abstract impl registered at {self.kernel.source}.')
        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, 'Meta'):
            raise RuntimeError(f"impl_abstract(...): the operator {self.qualname} already has an DispatchKey::Meta implementation via a pre-existing torch.library or TORCH_LIBRARY registration. Please either remove that registration or don't call impl_abstract.")
        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, 'CompositeImplicitAutograd'):
            raise RuntimeError(f'impl_abstract(...): the operator {self.qualname} already has an implementation for this device type via a pre-existing registration to DispatchKey::CompositeImplicitAutograd.CompositeImplicitAutograd operators do not need an abstract impl; instead, the operator will decompose into its constituents and those can have abstract impls defined on them.')
        self.kernel = Kernel(func, source)
        if self.lib is None:
            ns = self.qualname.split('::')[0]
            self.lib = torch.library.Library(ns, 'FRAGMENT')
        meta_kernel = construct_meta_kernel(self.qualname, self)
        self.lib.impl(self.qualname, meta_kernel, 'Meta')

        def deregister_abstract_impl():
            if False:
                for i in range(10):
                    print('nop')
            if self.lib:
                self.lib._destroy()
                self.lib = None
            self.kernel = None
        return RegistrationHandle(deregister_abstract_impl)

def construct_meta_kernel(qualname: str, abstract_impl_holder: AbstractImplHolder) -> Callable:
    if False:
        i = 10
        return i + 15
    assert abstract_impl_holder.kernel is not None

    @functools.wraps(abstract_impl_holder.kernel.func)
    def meta_kernel(*args, **kwargs):
        if False:
            return 10
        assert abstract_impl_holder.kernel is not None
        source = abstract_impl_holder.kernel.source

        def error_on_ctx():
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError(f'Attempted to call get_ctx() for the meta implementation for {qualname} (implemented at {source})You have presumably called get_ctx() because the operator has a data-dependent output shape; if so, there is no such meta implementation and this error is the correct behavior.')
        with set_ctx_getter(error_on_ctx):
            return abstract_impl_holder.kernel(*args, **kwargs)
    return meta_kernel

def get_none():
    if False:
        while True:
            i = 10
    return None
global_ctx_getter: Callable = get_none

@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    if False:
        for i in range(10):
            print('nop')
    global global_ctx_getter
    prev = global_ctx_getter
    try:
        global_ctx_getter = ctx_getter
        yield
    finally:
        global_ctx_getter = prev

class AbstractImplCtx:
    """
    Context object for writing abstract implementations for custom operators.
    """

    def __init__(self, _shape_env, _op):
        if False:
            return 10
        self._shape_env = _shape_env
        self._op = _op

    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        if False:
            return 10
        warnings.warn('create_unbacked_symint is deprecated, please use new_dynamic_size instead')
        return self.new_dynamic_size(min=min, max=max)

    def new_dynamic_size(self, *, min=0, max=None) -> torch.SymInt:
        if False:
            i = 10
            return i + 15
        'Constructs a new symint (symbolic int) representing a data-dependent value.\n\n        This is useful for writing the abstract implementation (which is necessary\n        for torch.compile) for a CustomOp where an output Tensor has a size\n        that depends on the data of the input Tensors.\n\n        Args:\n            min (int): A statically known inclusive lower bound for this symint. Default: 0\n            max (Optional[int]): A statically known inclusive upper bound for this\n                symint. Default: None\n\n        .. warning:\n\n            It is important that the ``min`` and ``max`` (if not None) values are set\n            correctly, otherwise, there will be undefined behavior under\n            torch.compile. The default value of ``min`` is 2 due to torch.compile\n            specializing on 0/1 sizes.\n\n            You must also verify that your implementation on concrete Tensors\n            (e.g. CPU/CUDA) only returns Tensors where the size that corresponds\n            to the symint also has respects these constraint.\n            The easiest way to do this is to add an assertion in the CPU/CUDA/etc\n            implementation that the size follows these bounds.\n\n        Example::\n\n            >>> # An operator with data-dependent output shape\n            >>> lib = torch.library.Library("mymodule", "FRAGMENT")\n            >>> lib.define("mymodule::custom_nonzero(Tensor x) -> Tensor")\n            >>>\n            >>> @torch.library.impl_abstract("mymodule::custom_nonzero")\n            >>> def custom_nonzero_abstract(x):\n            >>>     # Number of nonzero-elements is data-dependent.\n            >>>     # Since we cannot peek at the data in an abstract impl,\n            >>>     # we use the ctx object to construct a new symint that\n            >>>     # represents the data-dependent size.\n            >>>     ctx = torch.library.get_ctx()\n            >>>     nnz = ctx.new_dynamic_size()\n            >>>     shape = [nnz, x.dim()]\n            >>>     result = x.new_empty(shape, dtype=torch.int64)\n            >>>     return result\n            >>>\n            >>> @torch.library.impl(lib, "custom_nonzero", "CPU")\n            >>> def custom_nonzero_cpu(x):\n            >>>     x_np = x.numpy()\n            >>>     res = np.stack(np.nonzero(x_np), axis=1)\n            >>>     return torch.tensor(res, device=x.device)\n\n        '
        if self._shape_env is None or not self._shape_env.allow_dynamic_output_shape_ops:
            raise torch._subclasses.fake_tensor.DynamicOutputShapeException(self._op)
        if isinstance(min, torch.SymInt) or isinstance(max, torch.SymInt):
            raise ValueError(f'ctx.new_dynamic_size(min={min}, max={max}): expected min and max to be statically known ints but got SymInt. This is not supported.')
        if min < 0:
            raise ValueError(f'ctx.new_dynamic_size(min={min}, ...): expected min to be greater than or equal to 0: this API can only create non-negative sizes.')
        result = self._shape_env.create_unbacked_symint()
        torch.fx.experimental.symbolic_shapes._constrain_range_for_size(result, min=min, max=max)
        return result