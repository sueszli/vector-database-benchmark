import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 4),))
class ClassMethod(torch.nn.Module):
    """
    Class methods are inlined during tracing.
    """

    @classmethod
    def method(cls, x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.linear(x)
        return self.method(x) * self.__class__.method(x) * type(self).method(x)