import os
import torch
from logging import warning
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_10
CPU_DEVICE = torch.device('cpu')
TORCH_CUDA_NO_OP_LIST = ['set_device', 'synchronize', 'reset_peak_memory_stats', 'reset_accumulated_memory_stats']
COMMON_TENSOR_TYPE = ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half', 'Bool', 'BFloat16']
CREATE_TENSOR_FUNC = ['rand', 'randint', 'randn', 'zeros', 'ones', 'empty', 'full', 'rand_like', 'randint_like', 'randn_like', 'zeros_like', 'ones_like', 'empty_like', 'full_like', 'tensor', 'scalar_tensor', 'sparse_coo_tensor', 'sparse_csr_tensor', 'sparse_csc_tensor', 'sparse_bsc_tensor', 'sparse_bsr_tensor', 'sparse_compressed_tensor', 'nested_tensorrandperm', 'normal', 'range', 'arange', 'eye', 'as_tensor', 'asarray', 'linspace', 'logspace', 'tril_indices', 'triu_indices', 'bartlett_window', 'blackman_window', 'hamming_window', 'hann_window', 'kaiser_window', 'empty_quantized', 'empty_strided', 'frombuffer', 'from_file']
attrs = []
is_cuda_patched = False

def replace_attr(obj, name: str, value):
    if False:
        for i in range(10):
            print('nop')
    torch_attr = getattr(obj, name)
    setattr(obj, name, value)
    attrs.append((obj, name, torch_attr))

def np_op_func(*args, **kwargs):
    if False:
        while True:
            i = 10
    pass

def cuda(self, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return self

def is_gpu_device(device):
    if False:
        i = 10
        return i + 15
    return isinstance(device, int) or (isinstance(device, str) and 'cuda' in device) or (isinstance(device, torch.device) and device.type == 'cuda')

def to(torch_to):
    if False:
        return 10

    def new_to(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if is_gpu_device(kwargs.get('device')):
            kwargs['device'] = 'cpu'
            return torch_to(self, *args, **kwargs)
        elif len(args) > 0 and is_gpu_device(args[0]):
            return torch_to(self, 'cpu', *args[1:], **kwargs)
        else:
            return torch_to(self, *args, **kwargs)
    return new_to

def load(torch_load):
    if False:
        print('Hello World!')

    def new_load(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'map_location' in kwargs:
            kwargs['map_location'] = 'cpu'
            return torch_load(*args, **kwargs)
        elif len(args) > 1:
            return torch_load(args[0], 'cpu', *args[2:], **kwargs)
        else:
            return torch_load(*args, **kwargs)
    return new_load

class DeviceClass:

    def __new__(cls, string):
        if False:
            return 10
        return CPU_DEVICE

def GradScalerClass_wrapper(GradScaler):
    if False:
        return 10

    class GradScalerClass:

        def __new__(cls, *args, **kwargs):
            if False:
                return 10
            kwargs['enabled'] = False
            return GradScaler(*args, **kwargs)
    return GradScalerClass
if not TORCH_VERSION_LESS_1_10:

    class new_autocast(torch.autocast):

        def __init__(self, device_type, dtype=None, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            device_type = 'cpu' if device_type == 'cuda' else device_type
            dtype = torch.bfloat16 if dtype == torch.float16 else dtype
            super().__init__(device_type, dtype, *args, **kwargs)

class no_op_context:

    def __init__(self, *args, **kargs):
        if False:
            return 10
        pass

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __exit__(self):
        if False:
            print('Hello World!')
        pass

    def wait_stream(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        pass

def current_stream():
    if False:
        print('Hello World!')
    return no_op_context()

def init_process_group(torch_init_process_group):
    if False:
        i = 10
        return i + 15

    def new_init_process_group(backend, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        if backend == 'nccl':
            torch_init_process_group('gloo', *args, **kargs)
        else:
            torch_init_process_group(backend, *args, **kargs)
    return new_init_process_group

def create_tensor_func(torch_create_tensor_func):
    if False:
        while True:
            i = 10

    def new_create_tensor_func(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if is_gpu_device(kwargs.get('device')):
            kwargs['device'] = 'cpu'
        return torch_create_tensor_func(*args, **kwargs)
    return new_create_tensor_func

def patch_cuda(disable_jit: bool=True):
    if False:
        print('Hello World!')
    "\n    patch_cuda is used to make users' application that is written for cuda only\n    runnable on a CPU device by one-line patching.\n\n    e.g.\n        >>> from bigdl.nano.pytorch.patching import patch_cuda\n        >>> patch_cuda()  # be sure it is used at the header of the application\n        >>> # all other cuda only codes will be avilable for cpu\n\n    :param disable_jit: bool, if to disable jit compile. This is a known issue\n           for patch_cuda function. jit compile has not been supported for some\n           of the patching. Users may change it to False to check if their application\n           is affected by this issue.\n    "
    global is_cuda_patched
    if is_cuda_patched:
        return
    if disable_jit:
        warning('This CUDA patch is incompatible with JIT, JIT will be disabled!')
        torch.jit._state.disable()
    replace_attr(torch.Tensor, 'cuda', cuda)
    replace_attr(torch.Tensor, 'to', to(torch.Tensor.to))
    replace_attr(torch.nn.Module, 'cuda', cuda)
    replace_attr(torch.nn.Module, 'to', to(torch.nn.Module.to))
    replace_attr(torch, 'device', DeviceClass)
    replace_attr(torch, 'load', load(torch.load))
    replace_attr(torch.cuda, 'Stream', no_op_context)
    replace_attr(torch.cuda, 'current_stream', current_stream)
    replace_attr(torch.Tensor, 'record_stream', np_op_func)
    if not TORCH_VERSION_LESS_1_10:
        replace_attr(torch, 'autocast', new_autocast)
        replace_attr(torch.cuda.amp, 'autocast', torch.cpu.amp.autocast)
    replace_attr(torch.cuda.amp, 'GradScaler', GradScalerClass_wrapper(torch.cuda.amp.GradScaler))
    replace_attr(torch.distributed, 'init_process_group', init_process_group(torch.distributed.init_process_group))
    for no_op_cand in TORCH_CUDA_NO_OP_LIST:
        replace_attr(torch.cuda, no_op_cand, np_op_func)
    for t in COMMON_TENSOR_TYPE:
        replace_attr(torch.cuda, f'{t}Tensor', getattr(torch, f'{t}Tensor'))
    for f in CREATE_TENSOR_FUNC:
        try:
            replace_attr(torch, f, create_tensor_func(getattr(torch, f)))
        except AttributeError:
            pass
    is_cuda_patched = True

def unpatch_cuda():
    if False:
        i = 10
        return i + 15
    '\n    unpatch_cuda is an reverse function to patch_cuda. It will change the application\n    back to be available on cuda.\n\n    e.g.\n        >>> from bigdl.nano.pytorch.patching import unpatch_cuda\n        >>> unpatch_cuda()  # be sure it is used after patch_cuda\n        >>> # all other codes will be avilable for cuda\n\n    :param disable_jit: bool, if to disable jit compile. This is a known issue\n           for patch_cuda function. jit compile has not been supported for some\n           of the patching. Users may change it to False to check if their application\n           is affected by this issue.\n    '
    global is_cuda_patched
    if not is_cuda_patched:
        return
    torch.jit._state.enable()
    for (obj, name, torch_attr) in attrs:
        setattr(obj, name, torch_attr)
    is_cuda_patched = False

def get_cuda_status():
    if False:
        for i in range(10):
            print('nop')
    return is_cuda_patched