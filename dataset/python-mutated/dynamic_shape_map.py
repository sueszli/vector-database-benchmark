import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import map

@export_case(example_inputs=(torch.ones(3, 2), torch.ones(2)), tags={'torch.dynamic-shape', 'torch.map'})
def dynamic_shape_map(xs, y):
    if False:
        i = 10
        return i + 15
    '\n    functorch map() maps a function over the first tensor dimension.\n    '

    def body(x, y):
        if False:
            return 10
        return x + y
    return map(body, xs, y)