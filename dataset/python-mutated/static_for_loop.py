import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 2),), tags={'python.control-flow'})
class StaticForLoop(torch.nn.Module):
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        for i in range(10):
            ret.append(i + x)
        return ret