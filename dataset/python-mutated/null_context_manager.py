import contextlib
import torch
from torch._export.db.case import export_case

@export_case(example_inputs=(torch.ones(3, 2),), tags={'python.context-manager'})
def null_context_manager(x):
    if False:
        return 10
    '\n    Null context manager in Python will be traced out.\n    '
    ctx = contextlib.nullcontext()
    with ctx:
        return x.sin() + x.cos()