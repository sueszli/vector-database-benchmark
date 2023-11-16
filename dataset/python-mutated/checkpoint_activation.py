from contextlib import contextmanager, nullcontext
from typing import Any, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import _checkpoint_without_reentrant_generator, _DEFAULT_DETERMINISM_MODE
from .contract import contract

@contextmanager
def _no_hook(module: nn.Module):
    if False:
        return 10
    '\n    Disable hooks installed by checkpoint to avoid unintentional recursion\n    during backward recomputation.\n    '
    orig_enable_hook = checkpoint.state(module).enable_hook
    checkpoint.state(module).enable_hook = False
    try:
        yield
    finally:
        checkpoint.state(module).enable_hook = orig_enable_hook

@contract()
def checkpoint(module: nn.Module) -> nn.Module:
    if False:
        return 10
    '\n    This is a composable activation checkpointing API. Unlike functional\n    activation checkpointing APIs, this one does not require changing model\n    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,\n    this one does not modify model structure or fully-qualified names either.\n    Under the hood, it registers activation checkpointing logic as pre- and\n    post-forward hooks. Hence, this API can be easily applied to any model or\n    sub-modules in the model.\n\n    Args:\n        module (nn.Module): the target model or sub-module to apply activation\n            checkpointing.\n\n    Example::\n        >>> # xdoctest: +SKIP\n        >>> import torch.nn as nn\n        >>>\n        >>> class MyModel(nn.Module):\n        >>>     def __init__(self):\n        >>>         super().__init__()\n        >>>         self.l1 = nn.Linear(10, 10)\n        >>>         self.l2 = nn.Linear(10, 10)\n        >>>\n        >>>     def forward(self, x):\n        >>>         return self.l2(self.l1(x))\n        >>>\n        >>> model = MyModel()\n        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1\n        >>> model(torch.zeros(2, 10)).sum().backward()\n\n    '
    torch._C._log_api_usage_once('torch.distributed.checkpoint')

    def forward_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if False:
            return 10
        if checkpoint.state(module).enable_hook:

            def context_fns():
                if False:
                    for i in range(10):
                        print('nop')
                return (nullcontext(), _no_hook(module))
            checkpoint.state(module)._ac_generator = _checkpoint_without_reentrant_generator(module, True, context_fns, _DEFAULT_DETERMINISM_MODE, False, *inputs)
            next(checkpoint.state(module)._ac_generator)

    def forward_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if False:
            i = 10
            return i + 15
        if checkpoint.state(module).enable_hook:
            try:
                next(checkpoint.state(module)._ac_generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError('Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!')
        checkpoint.state(module)._ac_generator = None
    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module