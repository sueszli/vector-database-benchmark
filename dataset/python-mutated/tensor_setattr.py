import torch
from torch._export.db.case import export_case, SupportLevel

@export_case(example_inputs=(torch.randn(3, 2), 'attr'), tags={'python.builtin'}, support_level=SupportLevel.SUPPORTED)
def tensor_setattr(x, attr):
    if False:
        for i in range(10):
            print('nop')
    '\n    setattr() call onto tensors is not supported.\n    '
    setattr(x, attr, torch.randn(3, 2))
    return x + 4