"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""
import functools

def _disable_dynamo(fn=None, recursive=True):
    if False:
        i = 10
        return i + 15
    '\n    This API should be only used inside torch, external users should still use\n    torch._dynamo.disable. The main goal of this API is to avoid circular\n    imports issues that is common while using _dynamo.disable inside torch\n    itself.\n\n    This API avoids it by lazily importing torch._dynamo from the import time to\n    the invocation of the decorated function.\n    '
    if fn is not None:

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            import torch._dynamo
            return torch._dynamo.disable(fn, recursive)(*args, **kwargs)
        return inner
    else:
        return functools.partial(_disable_dynamo, recursive=recursive)