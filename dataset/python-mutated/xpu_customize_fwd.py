import collections
import functools
import torch
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
from typing import Any

def _cast(value, dtype):
    if False:
        while True:
            i = 10
    if isinstance(value, torch.Tensor):
        is_eligible = value.is_floating_point() and value.is_xpu and (value.dtype is not torch.float64)
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
    "\n    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of\n    :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>`\n    for more detail.\n\n    Args:\n        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,\n            when ``forward`` runs in an autocast-enabled region, casts incoming\n            floating-point CUDA Tensors to the target dtype (non-floating-point Tensors\n            are not affected),\n            then executes ``forward`` with autocast disabled.\n            If ``None``, ``forward``'s internal ops execute with the current autocast state.\n\n    .. note::\n        If the decorated ``forward`` is called outside an autocast-enabled region,\n        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.\n    "
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if False:
            print('Hello World!')
        args[0]._dtype = torch.xpu.get_autocast_xpu_dtype()
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.xpu.is_autocast_xpu_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.xpu.is_autocast_xpu_enabled()
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with torch.xpu.autocast(enabled=False):
                    return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
            else:
                return fwd(*args, **kwargs)
    return decorate_fwd

def custom_bwd(bwd):
    if False:
        i = 10
        return i + 15
    '\n    Helper decorator for backward methods of custom autograd functions (subclasses of\n    :class:`torch.autograd.Function`).\n    Ensures that ``backward`` executes with the same autocast state as ``forward``.\n    See the :ref:`example page<amp-custom-examples>` for more detail.\n    '

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        if False:
            return 10
        with torch.xpu.autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)
    return decorate_bwd