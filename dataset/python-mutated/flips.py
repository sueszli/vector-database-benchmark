import torch
from kornia.core import Module, Tensor
__all__ = ['Vflip', 'Hflip', 'Rot180', 'rot180', 'hflip', 'vflip']

class Vflip(Module):
    """Vertically flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The vertically flipped image tensor.

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        tensor([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return vflip(input)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__

class Hflip(Module):
    """Horizontally flip a tensor image or a batch of tensor images.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Returns:
        The horizontally flipped image tensor.

    Examples:
        >>> hflip = Hflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> hflip(input)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return hflip(input)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return self.__class__.__name__

class Rot180(Module):
    """Rotate a tensor image or a batch of tensor images 180 degrees.

    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input: input tensor.

    Examples:
        >>> rot180 = Rot180()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> rot180(input)
        tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def forward(self, input: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        return rot180(input)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

def rot180(input: Tensor) -> Tensor:
    if False:
        i = 10
        return i + 15
    'Rotate a tensor image or a batch of tensor images 180 degrees.\n\n    .. image:: _static/img/rot180.png\n\n    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.\n\n    Args:\n        input: input tensor.\n\n    Returns:\n        The rotated image tensor.\n    '
    return torch.flip(input, [-2, -1])

def hflip(input: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Horizontally flip a tensor image or a batch of tensor images.\n\n    .. image:: _static/img/hflip.png\n\n    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.\n\n    Args:\n        input: input tensor.\n\n    Returns:\n        The horizontally flipped image tensor.\n    '
    return input.flip(-1).contiguous()

def vflip(input: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Vertically flip a tensor image or a batch of tensor images.\n\n    .. image:: _static/img/vflip.png\n\n    Input must be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.\n\n    Args:\n        input: input tensor.\n\n    Returns:\n        The vertically flipped image tensor.\n    '
    return input.flip(-2).contiguous()