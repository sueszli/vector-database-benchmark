import collections
import functools
import torch
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
from typing import Any
__all__ = ['autocast', 'custom_fwd', 'custom_bwd']

class autocast(torch.amp.autocast_mode.autocast):
    """See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is equivalent to ``torch.autocast("cuda", args...)``
    """

    def __init__(self, enabled: bool=True, dtype: torch.dtype=torch.float16, cache_enabled: bool=True):
        if False:
            while True:
                i = 10
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = 'cuda'
            self.fast_dtype = dtype
            return
        super().__init__('cuda', enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def __enter__(self):
        if False:
            while True:
                i = 10
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        if False:
            return 10
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if False:
            print('Hello World!')
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)

def _cast(value, dtype):
    if False:
        i = 10
        return i + 15
    if isinstance(value, torch.Tensor):
        is_eligible = value.is_floating_point() and value.is_cuda and (value.dtype is not torch.float64)
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for (k, v) in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value

def custom_fwd(fwd=None, *, cast_inputs=None):
    if False:
        i = 10
        return i + 15
    "\n    Create a helper decorator for ``forward`` methods of custom autograd functions.\n\n    Autograd functions are subclasses of :class:`torch.autograd.Function`.\n    See the :ref:`example page<amp-custom-examples>` for more detail.\n\n    Args:\n        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,\n            when ``forward`` runs in an autocast-enabled region, casts incoming\n            floating-point CUDA Tensors to the target dtype (non-floating-point Tensors are not affected),\n            then executes ``forward`` with autocast disabled.\n            If ``None``, ``forward``'s internal ops execute with the current autocast state.\n\n    .. note::\n        If the decorated ``forward`` is called outside an autocast-enabled region,\n        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.\n    "
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if False:
            while True:
                i = 10
        args[0]._dtype = torch.get_autocast_gpu_dtype()
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled()
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(enabled=False):
                    return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
            else:
                return fwd(*args, **kwargs)
    return decorate_fwd

def custom_bwd(bwd):
    if False:
        i = 10
        return i + 15
    'Create a helper decorator for backward methods of custom autograd functions.\n\n    Autograd functions are subclasses of :class:`torch.autograd.Function`.\n    Ensures that ``backward`` executes with the same autocast state as ``forward``.\n    See the :ref:`example page<amp-custom-examples>` for more detail.\n    '

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)
    return decorate_bwd