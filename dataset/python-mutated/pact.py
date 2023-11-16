import torch
import torch.nn as nn

class PACTClip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if False:
            return 10
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, 0, alpha.data)

    @staticmethod
    def backward(ctx, dy):
        if False:
            return 10
        (x, alpha) = ctx.saved_tensors
        dx = dy.clone()
        dx[x < 0] = 0
        dx[x > alpha] = 0
        dalpha = dy.clone()
        dalpha[x <= alpha] = 0
        return (dx, torch.sum(dalpha))

class PACTReLU(nn.Module):

    def __init__(self, alpha=6.0):
        if False:
            print('Hello World!')
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return PACTClip.apply(x, self.alpha)