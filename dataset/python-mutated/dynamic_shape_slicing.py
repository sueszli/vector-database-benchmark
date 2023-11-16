import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 2),), tags={'torch.dynamic-shape'})
def dynamic_shape_slicing(x):
    if False:
        while True:
            i = 10
    '\n    Slices with dynamic shape arguments should be captured into the graph\n    rather than being baked in.\n    '
    return x[:x.shape[0] - 2, x.shape[1] - 1::2]