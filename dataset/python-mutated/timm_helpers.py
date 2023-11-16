import math
from itertools import chain
from typing import Callable
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if False:
        for i in range(10):
            print('nop')
    if not depth_first and include_root:
        fn(module=module, name=name)
    for (child_name, child_module) in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def adapt_input_conv(in_chans, conv_weight):
    if False:
        return 10
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    (O, I, J, K) = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight

def checkpoint_seq(functions, x, every=1, flatten=False, skip_last=False, preserve_rng_state=True):
    if False:
        while True:
            i = 10
    "A helper function for checkpointing sequential models.\n\n    Sequential models execute a list of modules/functions in order\n    (sequentially). Therefore, we can divide such a sequence into segments\n    and checkpoint each segment. All segments except run in :func:`torch.no_grad`\n    manner, i.e., not storing the intermediate activations. The inputs of each\n    checkpointed segment will be saved for re-running the segment in the backward pass.\n\n    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.\n\n    .. warning::\n        Checkpointing currently only supports :func:`torch.autograd.backward`\n        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`\n        is not supported.\n\n    .. warning:\n        At least one of the inputs needs to have :code:`requires_grad=True` if\n        grads are needed for model inputs, otherwise the checkpointed part of the\n        model won't have gradients.\n\n    Args:\n        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.\n        x: A Tensor that is input to :attr:`functions`\n        every: checkpoint every-n functions (default: 1)\n        flatten (bool): flatten nn.Sequential of nn.Sequentials\n        skip_last (bool): skip checkpointing the last function in the sequence if True\n        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring\n            the RNG state during each checkpoint.\n\n    Returns:\n        Output of running :attr:`functions` sequentially on :attr:`*inputs`\n\n    Example:\n        >>> model = nn.Sequential(...)\n        >>> input_var = checkpoint_seq(model, input_var, every=2)\n    "

    def run_function(start, end, functions):
        if False:
            while True:
                i = 10

        def forward(_x):
            if False:
                i = 10
                return i + 15
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward
    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)
    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x