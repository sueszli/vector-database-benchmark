import functools
import torch
import torch.distributed as dist
from typing import Optional

class DefaultState:
    """
    Stores state needed to perform the default communication algorithm within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used.
    """
    __slots__ = ['process_group', 'world_size', 'gradient_predivide_factor', 'gradient_postdivide_factor']

    def __init__(self, process_group: dist.ProcessGroup):
        if False:
            i = 10
            return i + 15
        if process_group is None:
            raise ValueError(f'Expected to pass in an explicit ProcessGroup to {self}.')
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(self.world_size)
        self.gradient_postdivide_factor = self.world_size / self.gradient_predivide_factor

    @staticmethod
    def _get_gradient_predivide_factor(world_size: int) -> float:
        if False:
            return 10
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)

class LowPrecisionState(DefaultState):
    """
    Stores state needed to perform gradient communication in a lower precision within a communication hook.

    Communication hook will cast gradients back to the original
    parameter precision specified by ``parameter_type`` (default: torch.float32).
    Builds on top of the :class:`DefaultState`.

    Args:
        parameter_type (torch.dtype): The precision of model's parameters.
        Required for a hook to cast gradients back to a parameter's precision.
    """
    __slots__ = ['parameter_type']

    def __init__(self, process_group, parameter_type=torch.float32):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(process_group)
        self.parameter_type = parameter_type

def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Casts gradients back to full parameter precision so that further computation happens in full precision.\n    '
    orig_grad_data = grad.data
    grad.data = grad.data.to(state.parameter_type)
    orig_grad_data.record_stream(torch.cuda.current_stream())

def allreduce_hook(state: DefaultState, grad: torch.Tensor):
    if False:
        while True:
            i = 10
    '\n    Implement the  FSDP communication hook for ``all_reduce`` algorithm and a necessary pre- and post-division of gradients.\n\n    Args:\n        state (DefaultState): State information, configures pre- and post-division factors.\n        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.\n    '
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    if state.gradient_postdivide_factor > 1:
        grad.div_(state.gradient_postdivide_factor)

def reduce_scatter_hook(state: DefaultState, grad: torch.Tensor, output: torch.Tensor):
    if False:
        return 10
    '\n    Implement the  FSDP communication hook for ``reduce_scatter`` algorithm.\n\n    For sharded FSDP strategies and a necessary pre- and post-division of gradients.\n\n    Args:\n        state (DefaultState): State information, configures pre- and post-division factors.\n        grad (torch.Tensor): An unsharded gradient for the local batch that needs to be\n        communicated across ranks.\n        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.\n    '
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.reduce_scatter_tensor(output, grad, group=state.process_group)
    if state.gradient_postdivide_factor > 1:
        output.div_(state.gradient_postdivide_factor)

def _low_precision_hook(prec: torch.dtype, state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor):
    if False:
        while True:
            i = 10
    if grad.dtype != prec:
        grad.data = grad.data.to(prec)
    if output is not None:
        if output.dtype != prec:
            output.data = output.data.to(prec)
        reduce_scatter_hook(state, grad, output)
        _decompress(state, output)
    else:
        allreduce_hook(state, grad)
        _decompress(state, grad)

def fp16_compress_hook(state: LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor]=None):
    if False:
        i = 10
        return i + 15
    "\n    Implement FSDP communication hook for a simple gradient compression approach.\n    Casts ``grad`` to half-precision floating-point format (``torch.float16``).\n\n    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a\n    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)\n    gradients are averaged by a ``state.gradient_postdivide_factor``.\n    Once post-division is done, compressed gradients are casted back to parameters' precision.\n\n    Args:\n        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.\n        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.\n        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.\n    "
    fp16_hook = functools.partial(_low_precision_hook, torch.float16)
    return fp16_hook(state, grad, output)

def bf16_compress_hook(state: LowPrecisionState, grad: torch.Tensor, output: Optional[torch.Tensor]=None):
    if False:
        i = 10
        return i + 15
    "\n    Implement FSDP communication hook for a simple gradient compression approach .\n    Casts ``grad`` to half-precision floating-point format.\n\n    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a\n    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)\n    gradients are averaged by a ``state.gradient_postdivide_factor``.\n    Once post-division is done, compressed gradients are casted back to parameters' precision.\n\n    Args:\n        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.\n        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.\n        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.\n    "
    bf16_hook = functools.partial(_low_precision_hook, torch.bfloat16)
    return bf16_hook(state, grad, output)