"""Async API.

This module contains the API for parallelism in TorchScript, notably:
    * torch.jit.fork
    * torch.jit.wait

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import torch
from torch._jit_internal import Future
from torch.jit._builtins import _register_builtin
from torch.utils import set_module
set_module(Future, 'torch.jit')

def fork(func, *args, **kwargs):
    if False:
        return 10
    '\n    Create an asynchronous task executing `func` and a reference to the value of the result of this execution.\n\n    `fork` will return immediately, so the return value of `func` may not have been computed yet. To force completion\n    of the task and access the return value invoke `torch.jit.wait` on the Future. `fork` invoked\n    with a `func` which returns `T` is typed as `torch.jit.Future[T]`. `fork` calls can be arbitrarily\n    nested, and may be invoked with positional and keyword arguments.\n    Asynchronous execution will only occur when run in TorchScript. If run in pure python,\n    `fork` will not execute in parallel. `fork` will also not execute in parallel when invoked\n    while tracing, however the `fork` and `wait` calls will be captured in the exported IR Graph.\n\n    .. warning::\n        `fork` tasks will execute non-deterministically. We recommend only spawning\n        parallel fork tasks for pure functions that do not modify their inputs,\n        module attributes, or global state.\n\n    Args:\n        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`\n            that will be invoked. If executed in TorchScript, it will execute asynchronously,\n            otherwise it will not. Traced invocations of fork will be captured in the IR.\n        ``*args``, ``**kwargs``: arguments to invoke `func` with.\n    Returns:\n        `torch.jit.Future[T]`: a reference to the execution of `func`. The value `T`\n        can only be accessed by forcing completion of `func` through `torch.jit.wait`.\n\n    Example (fork a free function):\n\n    .. code-block:: python\n\n        import torch\n        from torch import Tensor\n        def foo(a : Tensor, b : int) -> Tensor:\n            return a + b\n        def bar(a):\n            fut : torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)\n            return torch.jit.wait(fut)\n        script_bar = torch.jit.script(bar)\n        input = torch.tensor(2)\n        # only the scripted version executes asynchronously\n        assert script_bar(input) == bar(input)\n        # trace is not run asynchronously, but fork is captured in IR\n        graph = torch.jit.trace(bar, (input,)).graph\n        assert "fork" in str(graph)\n\n    Example (fork a module method):\n\n    .. code-block:: python\n\n        import torch\n        from torch import Tensor\n        class AddMod(torch.nn.Module):\n            def forward(self, a: Tensor, b : int):\n                return a + b\n        class Mod(torch.nn.Module):\n            def __init__(self):\n                super(self).__init__()\n                self.mod = AddMod()\n            def forward(self, input):\n                fut = torch.jit.fork(self.mod, a, b=2)\n                return torch.jit.wait(fut)\n        input = torch.tensor(2)\n        mod = Mod()\n        assert mod(input) == torch.jit.script(mod).forward(input)\n    '
    return torch._C.fork(func, *args, **kwargs)

def wait(future):
    if False:
        while True:
            i = 10
    '\n    Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task.\n\n    See :func:`~fork` for docs and examples.\n    Args:\n        future (torch.jit.Future[T]): an asynchronous task reference, created through `torch.jit.fork`\n    Returns:\n        `T`: the return value of the completed task\n    '
    return torch._C.wait(future)
_register_builtin(wait, 'aten::wait')