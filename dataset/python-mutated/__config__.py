import torch

def show():
    if False:
        i = 10
        return i + 15
    '\n    Return a human-readable string with descriptions of the\n    configuration of PyTorch.\n    '
    return torch._C._show_config()

def _cxx_flags():
    if False:
        return 10
    'Returns the CXX_FLAGS used when building PyTorch.'
    return torch._C._cxx_flags()

def parallel_info():
    if False:
        for i in range(10):
            print('nop')
    'Returns detailed string with parallelization settings'
    return torch._C._parallel_info()