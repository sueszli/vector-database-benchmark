import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 2),), tags={'python.assert'})
def dynamic_shape_assert(x):
    if False:
        while True:
            i = 10
    '\n    A basic usage of python assertion.\n    '
    assert x.shape[0] > 2, f'{x.shape[0]} is greater than 2'
    assert x.shape[0] > 1
    return x