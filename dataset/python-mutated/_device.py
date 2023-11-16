import torch
from torch.overrides import TorchFunctionMode
from torch.utils._contextlib import context_decorator
import functools
CURRENT_DEVICE = None

@functools.lru_cache(1)
def _device_constructors():
    if False:
        while True:
            i = 10
    return {torch.empty, torch.empty_permuted, torch.empty_strided, torch.empty_quantized, torch.ones, torch.arange, torch.bartlett_window, torch.blackman_window, torch.eye, torch.fft.fftfreq, torch.fft.rfftfreq, torch.full, torch.fill, torch.hamming_window, torch.hann_window, torch.kaiser_window, torch.linspace, torch.logspace, torch.nested.nested_tensor, torch.ones, torch.rand, torch.randn, torch.randint, torch.randperm, torch.range, torch.sparse_coo_tensor, torch.sparse_compressed_tensor, torch.sparse_csr_tensor, torch.sparse_csc_tensor, torch.sparse_bsr_tensor, torch.sparse_bsc_tensor, torch.tril_indices, torch.triu_indices, torch.vander, torch.zeros, torch.asarray, torch.tensor, torch.as_tensor, torch.scalar_tensor, torch.asarray}

class DeviceContext(TorchFunctionMode):

    def __init__(self, device):
        if False:
            print('Hello World!')
        self.device = torch.device(device)

    def __enter__(self):
        if False:
            return 10
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if False:
            print('Hello World!')
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get('device') is None:
            kwargs['device'] = self.device
        return func(*args, **kwargs)

def device_decorator(device, func):
    if False:
        print('Hello World!')
    return context_decorator(lambda : device, func)

def set_device(device):
    if False:
        print('Hello World!')
    '\n    Set the default device inside of the wrapped function by decorating it with this function.\n\n    If you would like to use this as a context manager, use device as a\n    context manager directly, e.g., ``with torch.device(device)``.\n    '
    return lambda func: device_decorator(torch.device(device), func)