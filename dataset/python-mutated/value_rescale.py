import torch

def value_transform(x: torch.Tensor, eps: float=0.01) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        A function to reduce the scale of the action-value function.\n        :math: `h(x) = sign(x)(\\sqrt{(abs(x)+1)} - 1) + \\eps * x` .\n    Arguments:\n        - x: (:obj:`torch.Tensor`) The input tensor to be normalized.\n        - eps: (:obj:`float`) The coefficient of the additive regularization term \\\n            to ensure h^{-1} is Lipschitz continuous\n    Returns:\n        - (:obj:`torch.Tensor`) Normalized tensor.\n\n    .. note::\n        Observe and Look Further: Achieving Consistent Performance on Atari\n         (https://arxiv.org/abs/1805.11593)\n    '
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x

def value_inv_transform(x: torch.Tensor, eps: float=0.01) -> torch.Tensor:
    if False:
        return 10
    '\n    Overview:\n        The inverse form of value rescale.\n        :math: `h^{-1}(x) = sign(x)({(\\frac{\\sqrt{1+4\\eps(|x|+1+\\eps)}-1}{2\\eps})}^2-1)` .\n    Arguments:\n        - x: (:obj:`torch.Tensor`) The input tensor to be unnormalized.\n        - eps: (:obj:`float`) The coefficient of the additive regularization term \\\n            to ensure h^{-1} is Lipschitz continuous\n    Returns:\n        - (:obj:`torch.Tensor`) Unnormalized tensor.\n    '
    return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)

def symlog(x: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        A function to normalize the targets.\n        :math: `symlog(x) = sign(x)(\\ln{|x|+1})` .\n    Arguments:\n        - x: (:obj:`torch.Tensor`) The input tensor to be normalized.\n    Returns:\n        - (:obj:`torch.Tensor`) Normalized tensor.\n\n    .. note::\n        Mastering Diverse Domains through World Models\n         (https://arxiv.org/abs/2301.04104)\n    '
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def inv_symlog(x: torch.Tensor) -> torch.Tensor:
    if False:
        while True:
            i = 10
    '\n    Overview:\n        The inverse form of symlog.\n        :math: `symexp(x) = sign(x)(\\exp{|x|}-1)` .\n    Arguments:\n        - x: (:obj:`torch.Tensor`) The input tensor to be unnormalized.\n    Returns:\n        - (:obj:`torch.Tensor`) Unnormalized tensor.\n    '
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)