import torch
from torch._export.db.case import export_case
from torch.export import Dim
from functorch.experimental.control_flow import cond
x = torch.randn(3, 2)
y = torch.ones(2)
dim0_x = Dim('dim0_x')

@export_case(example_inputs=(x, y), tags={'torch.cond', 'torch.dynamic-shape'}, extra_inputs=(torch.randn(2, 2), torch.ones(2)), dynamic_shapes={'x': {0: dim0_x}, 'y': None})
def cond_operands(x, y):
    if False:
        i = 10
        return i + 15
    '\n    The operands passed to cond() must be:\n      - a list of tensors\n      - match arguments of `true_fn` and `false_fn`\n\n    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.\n    '

    def true_fn(x, y):
        if False:
            while True:
                i = 10
        return x + y

    def false_fn(x, y):
        if False:
            while True:
                i = 10
        return x - y
    return cond(x.shape[0] > 2, true_fn, false_fn, [x, y])