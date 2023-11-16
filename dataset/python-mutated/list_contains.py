import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 2),), tags={'torch.dynamic-shape', 'python.data-structure', 'python.assert'})
def list_contains(x):
    if False:
        while True:
            i = 10
    '\n    List containment relation can be checked on a dynamic shape or constants.\n    '
    assert x.size(-1) in [6, 2]
    assert x.size(0) not in [4, 5, 6]
    assert 'monkey' not in ['cow', 'pig']
    return x + x