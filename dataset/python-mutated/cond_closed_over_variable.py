import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond

@export_case(example_inputs=(torch.tensor(True), torch.ones(3, 2)), tags={'torch.cond', 'python.closure'})
class CondClosedOverVariable(torch.nn.Module):
    """
    torch.cond() supports branches closed over arbitrary variables.
    """

    def forward(self, pred, x):
        if False:
            print('Hello World!')

        def true_fn(val):
            if False:
                while True:
                    i = 10
            return x * 2

        def false_fn(val):
            if False:
                print('Hello World!')
            return x - 2
        return cond(pred, true_fn, false_fn, [x + 1])