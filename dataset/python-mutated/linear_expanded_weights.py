import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import forward_helper, set_grad_sample_if_exists, unpack_expanded_weight_or_tensor, is_batch_first
from typing import List, Optional

@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _, __, *expanded_args_and_kwargs):
        if False:
            i = 10
            return i + 15
        if len(expanded_args_and_kwargs[0].shape) <= 1:
            raise RuntimeError(f'Input does not have a batch dimension. Expanded Weights expected input of at least rank 2, got of rank {len(expanded_args_and_kwargs[0].shape)}')
        expanded_kwargs = {'bias': expanded_args_and_kwargs[2] if len(expanded_args_and_kwargs) == 3 else None}
        expanded_args = expanded_args_and_kwargs[:2]
        ctx.batch_first = is_batch_first(expanded_args_and_kwargs)
        output = forward_helper(F.linear, expanded_args, expanded_kwargs)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            i = 10
            return i + 15
        (input, weight) = ctx.args
        bias = ctx.kwargs['bias']
        results: List[Optional[torch.Tensor]] = []
        results.append(None)
        results.append(None)
        if input.requires_grad:
            results.append(grad_output.matmul(unpack_expanded_weight_or_tensor(weight)))
        else:
            results.append(None)
        results.extend([None] * 2)
        if not ctx.batch_first:
            grad_output = grad_output.transpose(0, 1)
            input = input.transpose(0, 1)
        set_grad_sample_if_exists(weight, lambda _: torch.einsum('n...i,n...j->nij', grad_output, input))
        set_grad_sample_if_exists(bias, lambda _: torch.einsum('n...k->nk', grad_output))
        return tuple(results)