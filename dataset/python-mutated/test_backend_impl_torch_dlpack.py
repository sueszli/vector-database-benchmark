from nvidia.dali.tensors import TensorCPU, TensorGPU, TensorListCPU, TensorListGPU
import nvidia.dali.tensors as tensors
import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import ctypes
from nvidia.dali.backend import CheckDLPackCapsule

def convert_to_torch(tensor, device='cuda', dtype=None, size=None):
    if False:
        i = 10
        return i + 15
    if size is None:
        if isinstance(tensor, TensorListCPU) or isinstance(tensor, TensorListGPU):
            t = tensor.as_tensor()
        else:
            t = tensor
        size = t.shape()
    dali_torch_tensor = torch.empty(size=size, device=device, dtype=dtype)
    c_type_pointer = ctypes.c_void_p(dali_torch_tensor.data_ptr())
    tensor.copy_to_external(c_type_pointer)
    return dali_torch_tensor

def test_dlpack_tensor_gpu_direct_creation():
    if False:
        for i in range(10):
            print('nop')
    arr = torch.rand(size=[3, 5, 6], device='cuda')
    tensor = TensorGPU(to_dlpack(arr))
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))

def test_dlpack_tensor_gpu_to_cpu():
    if False:
        print('Hello World!')
    arr = torch.rand(size=[3, 5, 6], device='cuda')
    tensor = TensorGPU(to_dlpack(arr))
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.cpu().eq(dali_torch_tensor.cpu()))

def test_dlpack_tensor_list_gpu_direct_creation():
    if False:
        while True:
            i = 10
    arr = torch.rand(size=[3, 5, 6], device='cuda')
    tensor_list = TensorListGPU(to_dlpack(arr), 'NHWC')
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))

def test_dlpack_tensor_list_gpu_to_cpu():
    if False:
        for i in range(10):
            print('nop')
    arr = torch.rand(size=[3, 5, 6], device='cuda')
    tensor_list = TensorListGPU(to_dlpack(arr), 'NHWC')
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.cpu().eq(dali_torch_tensor.cpu()))

def check_dlpack_types_gpu(t):
    if False:
        return 10
    arr = torch.tensor([[0.39, 1.5], [1.5, 0.33]], device='cuda', dtype=t)
    tensor = TensorGPU(to_dlpack(arr), 'NHWC')
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype, size=tensor.shape())
    assert torch.all(arr.eq(dali_torch_tensor))

def test_dlpack_interface_types():
    if False:
        while True:
            i = 10
    for t in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float64, torch.float32, torch.float16]:
        yield (check_dlpack_types_gpu, t)

def test_dlpack_tensor_cpu_direct_creation():
    if False:
        print('Hello World!')
    arr = torch.rand(size=[3, 5, 6], device='cpu')
    tensor = TensorCPU(to_dlpack(arr))
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))

def test_dlpack_tensor_list_cpu_direct_creation():
    if False:
        while True:
            i = 10
    arr = torch.rand(size=[3, 5, 6], device='cpu')
    tensor_list = TensorListCPU(to_dlpack(arr), 'NHWC')
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))

def test_dlpack_tensor_list_cpu_direct_creation_list():
    if False:
        return 10
    arr = torch.rand(size=[3, 5, 6], device='cpu')
    tensor_list = TensorListCPU([to_dlpack(arr)], 'NHWC')
    dali_torch_tensor = convert_to_torch(tensor_list, device=arr.device, dtype=arr.dtype)
    assert torch.all(arr.eq(dali_torch_tensor))

def test_tensor_cpu_from_dlpack():
    if False:
        i = 10
        return i + 15

    def create_tmp(idx):
        if False:
            while True:
                i = 10
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a))
        return tensors.TensorCPU(dlt, '')
    out = [create_tmp(i) for i in range(4)]
    for (i, t) in enumerate(out):
        np.testing.assert_array_equal(np.array(t), np.full((4, 4), i))

def test_tensor_list_cpu_from_dlpack():
    if False:
        print('Hello World!')

    def create_tmp(idx):
        if False:
            i = 10
            return i + 15
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a))
        return tensors.TensorListCPU(dlt, '')
    out = [create_tmp(i) for i in range(4)]
    for (i, tl) in enumerate(out):
        np.testing.assert_array_equal(tl.as_array(), np.full((4, 4), i))

def test_tensor_gpu_from_dlpack():
    if False:
        for i in range(10):
            print('nop')

    def create_tmp(idx):
        if False:
            print('Hello World!')
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a).cuda())
        return tensors.TensorGPU(dlt, '')
    out = [create_tmp(i) for i in range(4)]
    for (i, t) in enumerate(out):
        np.testing.assert_array_equal(np.array(t.as_cpu()), np.full((4, 4), i))

def test_tensor_list_gpu_from_dlpack():
    if False:
        for i in range(10):
            print('nop')

    def create_tmp(idx):
        if False:
            i = 10
            return i + 15
        a = np.full((4, 4), idx)
        dlt = to_dlpack(torch.from_numpy(a).cuda())
        return tensors.TensorListGPU(dlt, '')
    out = [create_tmp(i) for i in range(4)]
    for (i, tl) in enumerate(out):
        np.testing.assert_array_equal(tl.as_cpu().as_array(), np.full((4, 4), i))

def check_dlpack_types_cpu(t):
    if False:
        for i in range(10):
            print('nop')
    arr = torch.tensor([[0.39, 1.5], [1.5, 0.33]], device='cpu', dtype=t)
    tensor = TensorCPU(to_dlpack(arr), 'NHWC')
    dali_torch_tensor = convert_to_torch(tensor, device=arr.device, dtype=arr.dtype, size=tensor.shape())
    assert torch.all(arr.eq(dali_torch_tensor))

def test_dlpack_interface_types_cpu():
    if False:
        i = 10
        return i + 15
    for t in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float64, torch.float32]:
        yield (check_dlpack_types_cpu, t)

def test_CheckDLPackCapsuleNone():
    if False:
        while True:
            i = 10
    info = CheckDLPackCapsule(None)
    assert info == (False, False)

def test_CheckDLPackCapsuleCpu():
    if False:
        i = 10
        return i + 15
    arr = torch.rand(size=[3, 5, 6], device='cpu')
    info = CheckDLPackCapsule(to_dlpack(arr))
    assert info == (True, False)

def test_CheckDLPackCapsuleGpu():
    if False:
        i = 10
        return i + 15
    arr = torch.rand(size=[3, 5, 6], device='cuda')
    info = CheckDLPackCapsule(to_dlpack(arr))
    assert info == (True, True)