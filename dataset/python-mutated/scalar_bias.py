import torch

class ScalarBias(torch.autograd.Function):
    """
    Adds a vector of scalars, used in self-attention mechanism to allow
    the model to optionally attend to this vector instead of the past
    """

    @staticmethod
    def forward(ctx, input, dim, bias_init):
        if False:
            for i in range(10):
                print('nop')
        size = list(input.size())
        size[dim] += 1
        output = input.new(*size).fill_(bias_init)
        output.narrow(dim, 1, size[dim] - 1).copy_(input)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad):
        if False:
            i = 10
            return i + 15
        return (grad.narrow(ctx.dim, 1, grad.size(ctx.dim) - 1), None, None)

def scalar_bias(input, dim, bias_init=0):
    if False:
        print('Hello World!')
    return ScalarBias.apply(input, dim, bias_init)