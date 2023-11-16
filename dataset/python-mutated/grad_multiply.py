import torch

class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        if False:
            while True:
                i = 10
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        if False:
            return 10
        return (grad * ctx.scale, None)