from typing import Optional
import torch
from .expanded_weights_impl import ExpandedWeight

def is_batch_first(expanded_args_and_kwargs):
    if False:
        while True:
            i = 10
    batch_first = None
    for arg in expanded_args_and_kwargs:
        if not isinstance(arg, ExpandedWeight):
            continue
        if not batch_first:
            batch_first = arg.batch_first
        elif arg.batch_first != batch_first:
            raise RuntimeError('Got conflicting batch_first arguments in the same layer')
    return batch_first

def standard_kwargs(kwarg_names, expanded_args):
    if False:
        while True:
            i = 10
    "Separate args and kwargs from `__torch_function__`s that standardize kwargs.\n\n    Most `__torch_function__`s standardize the kwargs that they give, so this will separate\n    the args and kwargs they pass. Functions that don't are linear and convND.\n    "
    kwarg_values = expanded_args[len(expanded_args) - len(kwarg_names):]
    expanded_args_without_kwargs = expanded_args[:len(expanded_args) - len(kwarg_names)]
    expanded_kwargs = dict(zip(kwarg_names, kwarg_values))
    return (expanded_args_without_kwargs, expanded_kwargs)

def forward_helper(func, expanded_args, expanded_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Compute the forward pass for a function that has expanded weight(s) passed to it.\n\n    It will run the forward pass where all ExpandedWeights are their original\n    weight. It runs checks on the given arguments and detaches the outputs.\n\n    .. note:: First argument in :attr:`expanded_args` must be the input with the batch\n    dimension as the first element of the shape\n\n    .. note:: :attr:`func` must return a Tensor or tuple of Tensors\n\n    Args:\n        func: The function to be called\n        expanded_args: Arguments to be passed to :attr:`func`. Will include arguments\n          that need to be unpacked because they are ExpandedWeights\n        expanded_kwargs: Keyword arguments to be passed to :attr:`func`.\n          Similar to :attr:`expanded_args`.\n    '
    (unexpanded_args, unexpanded_kwargs) = _check_and_unexpand_args(func, expanded_args, expanded_kwargs)
    return func(*unexpanded_args, **unexpanded_kwargs)

def _check_and_unexpand_args(func, expanded_args, expanded_kwargs):
    if False:
        for i in range(10):
            print('nop')
    input = expanded_args[0]
    if isinstance(input, ExpandedWeight):
        raise RuntimeError(f'Expanded Weights do not support inputs that are also ExpandedWeights. Input must be a Tensor, got {type(input).__name__} in function {func.__name__}')
    if not isinstance(input, torch.Tensor):
        raise RuntimeError(f'Expanded Weights requires a Tensor as the first input to get the batch dimension, got {type(input).__name__} in function {func.__name__}')
    if len(input.shape) == 0:
        raise RuntimeError(f'Expanded Weights requires a batch dimension but got an input of size 0 in function {func.__name__}')
    if input.shape[0] == 0:
        raise RuntimeError(f'0 is not a valid batch size for Expanded Weights but got input tensor of {input} in function {func.__name__}')
    for arg in expanded_args + tuple(expanded_kwargs.values()):
        if not isinstance(arg, ExpandedWeight):
            continue
        batch_size = input.shape[0] if arg.batch_first else input.shape[1]
        if arg.allow_smaller_batches and batch_size > arg.batch_size or (not arg.allow_smaller_batches and arg.batch_size != batch_size):
            raise RuntimeError(f'Expected ExpandedWeights to have batch size matching input but got input batch size of {batch_size} with ExpandedWeight of batch size {arg.batch_size}')
    loss_reduction: Optional[str] = None
    for arg in expanded_args + tuple(expanded_kwargs.values()):
        if isinstance(arg, ExpandedWeight):
            if loss_reduction is None:
                loss_reduction = arg.loss_reduction
            elif loss_reduction != arg.loss_reduction:
                raise RuntimeError(f'Expected ExpandedWeights to all have the same loss_reduction argument but got onewith {loss_reduction} and one with {arg.loss_reduction}')
    unexpanded_args = tuple((arg.orig_weight if isinstance(arg, ExpandedWeight) else arg for arg in expanded_args))
    unexpanded_kwargs = {name: arg.orig_weight if isinstance(arg, ExpandedWeight) else arg for (name, arg) in expanded_kwargs.items()}
    return (unexpanded_args, unexpanded_kwargs)

def maybe_scale_by_batch_size(grad_sample, expanded_weight):
    if False:
        print('Hello World!')
    if expanded_weight.loss_reduction == 'mean':
        return grad_sample * expanded_weight.batch_size
    else:
        return grad_sample

def set_grad_sample_if_exists(maybe_expanded_weight, per_sample_grad_fn):
    if False:
        return 10
    unpacked = unpack_expanded_weight_or_tensor(maybe_expanded_weight)
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        grad_sample_contribution = maybe_scale_by_batch_size(per_sample_grad_fn(unpacked), maybe_expanded_weight)
        if maybe_expanded_weight.batch_size > grad_sample_contribution.shape[0]:
            intermediate = torch.zeros(maybe_expanded_weight.batch_size, *grad_sample_contribution.shape[1:], dtype=grad_sample_contribution.dtype, device=grad_sample_contribution.device)
            intermediate[:grad_sample_contribution.shape[0]] = grad_sample_contribution
            grad_sample_contribution = intermediate
        if hasattr(unpacked, 'grad_sample') and unpacked.grad_sample is not None:
            unpacked.grad_sample = unpacked.grad_sample + grad_sample_contribution
        else:
            unpacked.grad_sample = grad_sample_contribution

def unpack_expanded_weight_or_tensor(maybe_expanded_weight, func=lambda x: x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(maybe_expanded_weight, ExpandedWeight):
        orig_weight = maybe_expanded_weight.orig_weight
        return func(orig_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor) and (not maybe_expanded_weight.requires_grad):
        return func(maybe_expanded_weight)
    elif isinstance(maybe_expanded_weight, torch.Tensor):
        raise RuntimeError('ExpandedWeights currently does not support a mixture of ExpandedWeight parameters and normal Parameters. Please file and issue with pytorch/pytorch')

def sum_over_all_but_batch_and_last_n(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    if False:
        print('Hello World!')
    '\n    Calculate the sum over all dimensions, except the first (batch dimension), and excluding the last n_dims.\n\n    This function will ignore the first dimension and it will\n    not aggregate over the last n_dims dimensions.\n    Args:\n        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.\n        n_dims: Number of dimensions to keep.\n    Example:\n        >>> tensor = torch.ones(1, 2, 3, 4, 5)\n        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape\n        torch.Size([1, 4, 5])\n    Returns:\n        A tensor of shape ``(B, ..., X[n_dims-1])``\n    '
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)