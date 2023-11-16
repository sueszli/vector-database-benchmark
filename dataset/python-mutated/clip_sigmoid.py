import torch

def clip_sigmoid(x, eps=0.0001):
    if False:
        for i in range(10):
            print('nop')
    'Sigmoid function for input feature.\n\n    Args:\n        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].\n        eps (float, optional): Lower bound of the range to be clamped to.\n            Defaults to 1e-4.\n\n    Returns:\n        torch.Tensor: Feature map after sigmoid.\n    '
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y