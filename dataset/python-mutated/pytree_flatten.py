import torch
from torch._export.db.case import export_case, SupportLevel
from torch.utils import _pytree as pytree

@export_case(example_inputs=({1: torch.randn(3, 2), 2: torch.randn(3, 2)},), support_level=SupportLevel.SUPPORTED)
def pytree_flatten(x):
    if False:
        i = 10
        return i + 15
    '\n    Pytree from PyTorch cannot be captured by TorchDynamo.\n    '
    (y, spec) = pytree.tree_flatten(x)
    return y[0] + 1