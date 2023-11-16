import os
import ctypes
_cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
_cudart = ctypes.CDLL(os.path.join(_cuda_home, 'lib64/libcudart.so'))

def cuda_profile_start():
    if False:
        while True:
            i = 10
    _cudart.cudaProfilerStart()

def cuda_profile_stop():
    if False:
        i = 10
        return i + 15
    _cudart.cudaProfilerStop()
_nvtx = ctypes.CDLL(os.path.join(_cuda_home, 'lib64/libnvToolsExt.so'))

def cuda_nvtx_range_push(name):
    if False:
        return 10
    _nvtx.nvtxRangePushW(ctypes.c_wchar_p(name))

def cuda_nvtx_range_pop():
    if False:
        print('Hello World!')
    _nvtx.nvtxRangePop()