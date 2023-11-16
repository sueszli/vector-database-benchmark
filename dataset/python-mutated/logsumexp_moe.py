import torch

class LogSumExpMoE(torch.autograd.Function):
    """Standard LogSumExp forward pass, but use *posterior* for the backward.

    See `"Mixture Models for Diverse Machine Translation: Tricks of the Trade"
    (Shen et al., 2019) <https://arxiv.org/abs/1902.07816>`_.
    """

    @staticmethod
    def forward(ctx, logp, posterior, dim=-1):
        if False:
            for i in range(10):
                print('nop')
        ctx.save_for_backward(posterior)
        ctx.dim = dim
        return torch.logsumexp(logp, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            while True:
                i = 10
        (posterior,) = ctx.saved_tensors
        grad_logp = grad_output.unsqueeze(ctx.dim) * posterior
        return (grad_logp, None, None)