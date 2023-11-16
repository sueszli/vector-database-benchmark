import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 2),), tags={'torch.dynamic-shape'})
def dynamic_shape_constructor(x):
    if False:
        return 10
    '\n    Tensor constructors should be captured with dynamic shape inputs rather\n    than being baked in with static shape.\n    '
    return torch.ones(x.shape[0] * 2)