import torch
from typing import TypeVar
T = TypeVar('T')

def all_same_mode(modes):
    if False:
        return 10
    return all(tuple((mode == modes[0] for mode in modes)))
no_dispatch = torch._C._DisableTorchDispatch