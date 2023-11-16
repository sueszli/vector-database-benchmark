"""PyTorch BERT model."""
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
logger = logging.getLogger(__name__)

def gelu(x):
    if False:
        while True:
            i = 10
    "Implementation of the gelu activation function.\n        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):\n        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))\n    "
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    if False:
        i = 10
        return i + 15
    return x * torch.sigmoid(x)
ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}

class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        if False:
            for i in range(10):
                print('nop')
        'Construct a layernorm module in the TF style (epsilon inside the square root).\n        '
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class CrossEn(nn.Module):

    def __init__(self, config=None):
        if False:
            return 10
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        if False:
            while True:
                i = 10
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if False:
            print('Hello World!')
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            i = 10
            return i + 15
        return (grad_output[ctx.batch_size * ctx.rank:ctx.batch_size * (ctx.rank + 1)], None)

class AllGather2(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if False:
            for i in range(10):
                print('nop')
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            for i in range(10):
                print('nop')
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (grad_input[ctx.rank * ctx.batch_size:(ctx.rank + 1) * ctx.batch_size], None)