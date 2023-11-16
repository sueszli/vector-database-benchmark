from typing import Any
import torch
import enum
from torch._C import _from_dlpack
from torch._C import _to_dlpack as to_dlpack

class DLDeviceType(enum.IntEnum):
    kDLCPU = (1,)
    kDLGPU = (2,)
    kDLCPUPinned = (3,)
    kDLOpenCL = (4,)
    kDLVulkan = (7,)
    kDLMetal = (8,)
    kDLVPI = (9,)
    kDLROCM = (10,)
    kDLExtDev = (12,)
    kDLOneAPI = (14,)
torch._C._add_docstr(to_dlpack, 'to_dlpack(tensor) -> PyCapsule\n\nReturns an opaque object (a "DLPack capsule") representing the tensor.\n\n.. note::\n  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns\n  cannot be used for anything in Python other than use it as input to\n  ``from_dlpack``. The more idiomatic use of DLPack is to call\n  ``from_dlpack`` directly on the tensor object - this works when that\n  object has a ``__dlpack__`` method, which PyTorch and most other\n  libraries indeed have now.\n\n.. warning::\n  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.\n  Behavior when a capsule is consumed multiple times is undefined.\n\nArgs:\n    tensor: a tensor to be exported\n\nThe DLPack capsule shares the tensor\'s memory.\n')

def from_dlpack(ext_tensor: Any) -> 'torch.Tensor':
    if False:
        for i in range(10):
            print('nop')
    'from_dlpack(ext_tensor) -> Tensor\n\n    Converts a tensor from an external library into a ``torch.Tensor``.\n\n    The returned PyTorch tensor will share the memory with the input tensor\n    (which may have come from another library). Note that in-place operations\n    will therefore also affect the data of the input tensor. This may lead to\n    unexpected issues (e.g., other libraries may have read-only flags or\n    immutable data structures), so the user should only do this if they know\n    for sure that this is fine.\n\n    Args:\n        ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):\n            The tensor or DLPack capsule to convert.\n\n            If ``ext_tensor`` is a tensor (or ndarray) object, it must support\n            the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``\n            method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is\n            an opaque ``PyCapsule`` instance, typically produced by a\n            ``to_dlpack`` function or method.\n\n    Examples::\n\n        >>> import torch.utils.dlpack\n        >>> t = torch.arange(4)\n\n        # Convert a tensor directly (supported in PyTorch >= 1.10)\n        >>> t2 = torch.from_dlpack(t)\n        >>> t2[:2] = -1  # show that memory is shared\n        >>> t2\n        tensor([-1, -1,  2,  3])\n        >>> t\n        tensor([-1, -1,  2,  3])\n\n        # The old-style DLPack usage, with an intermediate capsule object\n        >>> capsule = torch.utils.dlpack.to_dlpack(t)\n        >>> capsule\n        <capsule object "dltensor" at ...>\n        >>> t3 = torch.from_dlpack(capsule)\n        >>> t3\n        tensor([-1, -1,  2,  3])\n        >>> t3[0] = -9  # now we\'re sharing memory between 3 tensors\n        >>> t3\n        tensor([-9, -1,  2,  3])\n        >>> t2\n        tensor([-9, -1,  2,  3])\n        >>> t\n        tensor([-9, -1,  2,  3])\n\n    '
    if hasattr(ext_tensor, '__dlpack__'):
        device = ext_tensor.__dlpack_device__()
        if device[0] in (DLDeviceType.kDLGPU, DLDeviceType.kDLROCM):
            stream = torch.cuda.current_stream(f'cuda:{device[1]}')
            is_cuda = device[0] == DLDeviceType.kDLGPU
            stream_ptr = 1 if is_cuda and stream.cuda_stream == 0 else stream.cuda_stream
            dlpack = ext_tensor.__dlpack__(stream=stream_ptr)
        else:
            dlpack = ext_tensor.__dlpack__()
    else:
        dlpack = ext_tensor
    return _from_dlpack(dlpack)