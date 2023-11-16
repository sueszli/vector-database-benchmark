from typing import Tuple, Union
from torch import Tensor, Size

def fold_batch(x: Tensor, nonbatch_ndims: int=1) -> Tuple[Tensor, Size]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Overview:\n        :math:`(T, B, X) \\leftarrow (T*B, X)`\\\n        Fold the first (ndim - nonbatch_ndims) dimensions of a tensor as batch dimension.\\\n        This operation is similar to `torch.flatten` but provides an inverse function\n        `unfold_batch` to restore the folded dimensions.\n\n    Arguments:\n        - x (:obj:`torch.Tensor`): the tensor to fold\n        - nonbatch_ndims (:obj:`int`): the number of dimensions that is not folded as\n            batch dimension.\n\n    Returns:\n        - x (:obj:`torch.Tensor`): the folded tensor\n        - batch_dims: the folded dimensions of the original tensor, which can be used to\n             reverse the operation\n\n    Examples:\n        >>> x = torch.ones(10, 20, 5, 4, 8)\n        >>> x, batch_dim = fold_batch(x, 2)\n        >>> x.shape == (1000, 4, 8)\n        >>> batch_dim == (10, 20, 5)\n\n    '
    if nonbatch_ndims > 0:
        batch_dims = x.shape[:-nonbatch_ndims]
        x = x.view(-1, *x.shape[-nonbatch_ndims:])
        return (x, batch_dims)
    else:
        batch_dims = x.shape
        x = x.view(-1)
        return (x, batch_dims)

def unfold_batch(x: Tensor, batch_dims: Union[Size, Tuple]) -> Tensor:
    if False:
        return 10
    '\n    Overview:\n        Unfold the batch dimension of a tensor.\n\n    Arguments:\n        - x (:obj:`torch.Tensor`): the tensor to unfold\n        - batch_dims (:obj:`torch.Size`): the dimensions that are folded\n\n    Returns:\n        - x (:obj:`torch.Tensor`): the original unfolded tensor\n\n    Examples:\n        >>> x = torch.ones(10, 20, 5, 4, 8)\n        >>> x, batch_dim = fold_batch(x, 2)\n        >>> x.shape == (1000, 4, 8)\n        >>> batch_dim == (10, 20, 5)\n        >>> x = unfold_batch(x, batch_dim)\n        >>> x.shape == (10, 20, 5, 4, 8)\n    '
    return x.view(*batch_dims, *x.shape[1:])

def unsqueeze_repeat(x: Tensor, repeat_times: int, unsqueeze_dim: int=0) -> Tensor:
    if False:
        print('Hello World!')
    '\n    Overview:\n        Squeeze the tensor on `unsqueeze_dim` and then repeat in this dimension for `repeat_times` times.\\\n        This is useful for preproprocessing the input to an model ensemble.\n\n    Arguments:\n        - x (:obj:`torch.Tensor`): the tensor to squeeze and repeat\n        - repeat_times (:obj:`int`): the times that the tensor is repeatd\n        - unsqueeze_dim (:obj:`int`): the unsqueezed dimension\n\n    Returns:\n        - x (:obj:`torch.Tensor`): the unsqueezed and repeated tensor\n\n    Examples:\n        >>> x = torch.ones(64, 6)\n        >>> x = unsqueeze_repeat(x, 4)\n        >>> x.shape == (4, 64, 6)\n\n        >>> x = torch.ones(64, 6)\n        >>> x = unsqueeze_repeat(x, 4, -1)\n        >>> x.shape == (64, 6, 4)\n    '
    assert -1 <= unsqueeze_dim <= len(x.shape), f'unsqueeze_dim should be from {-1} to {len(x.shape)}'
    x = x.unsqueeze(unsqueeze_dim)
    repeats = [1] * len(x.shape)
    repeats[unsqueeze_dim] *= repeat_times
    return x.repeat(*repeats)