from functools import reduce
import operator
import torch
import torch.nn.functional as F
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import standard_kwargs, forward_helper, set_grad_sample_if_exists, unpack_expanded_weight_or_tensor
from typing import List, Optional

@implements_per_sample_grads(F.group_norm)
class GroupNormPerSampleGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        if False:
            return 10
        (expanded_args, expanded_kwargs) = standard_kwargs(kwarg_names, expanded_args_and_kwargs)
        (input, num_groups) = expanded_args
        N = input.shape[0]
        C = input.shape[1]
        HxW = reduce(operator.mul, input.shape[2:], 1)
        (weight, bias, eps) = (expanded_kwargs['weight'], expanded_kwargs['bias'], expanded_kwargs['eps'])
        (output, mean, rstd) = forward_helper(torch.native_group_norm, (input, weight, bias, N, C, HxW, num_groups, eps), {})
        (ctx.input, ctx.num_groups) = (input, num_groups)
        (ctx.weight, ctx.eps) = (weight, eps)
        (ctx.mean, ctx.rstd) = (mean, rstd)
        if isinstance(bias, ExpandedWeight):
            ctx.bias = bias
        if input.requires_grad and isinstance(weight, ExpandedWeight):
            ctx.weight = weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            return 10
        (input, num_groups) = (ctx.input, ctx.num_groups)
        (weight, bias, eps) = (ctx.weight, ctx.bias, ctx.eps)
        (mean, rstd) = (ctx.mean, ctx.rstd)
        results: List[Optional[torch.Tensor]] = []
        results.append(None)
        results.append(None)
        if input.requires_grad:
            weight_c = unpack_expanded_weight_or_tensor(weight, lambda t: t.contiguous())
            input_c = input.contiguous()
            grad_output_c = grad_output.contiguous() if grad_output is not None else None
            N = input.shape[0]
            C = input.shape[1]
            HxW = 1
            for s in input.shape[2:]:
                HxW *= s
            bw_fn = torch.ops.aten.native_group_norm_backward
            results.append(bw_fn(grad_output_c, input_c, mean, rstd, weight_c, N, C, HxW, num_groups, (True, False, False))[0])
        else:
            results.append(None)
        results = results + [None] * 4
        if hasattr(ctx, 'weight'):
            set_grad_sample_if_exists(weight, lambda _: torch.einsum('ni...->ni', F.group_norm(input, num_groups, eps=eps) * grad_output))
        if hasattr(ctx, 'bias'):
            set_grad_sample_if_exists(bias, lambda _: torch.einsum('ni...->ni', grad_output))
        return tuple(results)