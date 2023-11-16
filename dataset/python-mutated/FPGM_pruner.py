from typing import Callable, Optional, Union
import torch
from .base_structured_sparsifier import BaseStructuredSparsifier
__all__ = ['FPGMPruner']

class FPGMPruner(BaseStructuredSparsifier):
    """Filter Pruning via Geometric Median (FPGM) Structured Pruner
    This sparsifier prune fliter (row) in a tensor according to distances among filters according to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.
    2. `dist` defines the distance measurement type. Default: 3 (L2 distance).
    Available options are: [1, 2, (custom callable distance function)].

    Note::
        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).
            - N: output channels size
            - C: input channels size
            - H: height of kernel
            - W: width of kernel
    """

    def __init__(self, sparsity_level: float=0.5, dist: Optional[Union[Callable, int]]=None):
        if False:
            print('Hello World!')
        defaults = {'sparsity_level': sparsity_level}
        if dist is None:
            dist = 2
        if callable(dist):
            self.dist_fn = dist
        elif dist == 1:
            self.dist_fn = lambda x: torch.cdist(x, x, p=1)
        elif dist == 2:
            self.dist_fn = lambda x: torch.cdist(x, x, p=2)
        else:
            raise NotImplementedError('Distance function is not yet implemented.')
        super().__init__(defaults=defaults)

    def _compute_distance(self, t):
        if False:
            print('Hello World!')
        'Compute distance across all entries in tensor `t` along all dimension\n        except for the one identified by dim.\n        Args:\n            t (torch.Tensor): tensor representing the parameter to prune\n        Returns:\n            distance (torch.Tensor): distance computed across filtters\n        '
        dim = 0
        size = t.size(dim)
        slc = [slice(None)] * t.dim()
        t_flatten = [t[tuple(slc[:dim] + [slice(i, i + 1)] + slc[dim + 1:])].reshape(-1) for i in range(size)]
        t_flatten = torch.stack(t_flatten)
        dist_matrix = self.dist_fn(t_flatten)
        distance = torch.sum(torch.abs(dist_matrix), 1)
        return distance

    def update_mask(self, module, tensor_name, sparsity_level, **kwargs):
        if False:
            i = 10
            return i + 15
        tensor_weight = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        if sparsity_level <= 0:
            mask.data = torch.ones_like(mask).bool()
        elif sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask).bool()
        else:
            distance = self._compute_distance(tensor_weight)
            tensor_size = tensor_weight.shape[0]
            nparams_toprune = round(sparsity_level * tensor_size)
            nparams_toprune = min(max(nparams_toprune, 0), tensor_size)
            topk = torch.topk(distance, k=nparams_toprune, largest=False)
            mask[topk.indices] = False