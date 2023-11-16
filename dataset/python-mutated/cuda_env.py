import functools
import logging
from typing import Optional
import torch
from ... import config
log = logging.getLogger(__name__)

def get_cuda_arch() -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            (major, minor) = torch.cuda.get_device_capability(0)
            cuda_arch = major * 10 + minor
        return str(cuda_arch)
    except Exception as e:
        log.error('Error getting cuda arch: %s', e)
        return None

def get_cuda_version() -> Optional[str]:
    if False:
        i = 10
        return i + 15
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        return cuda_version
    except Exception as e:
        log.error('Error getting cuda version: %s', e)
        return None

@functools.lru_cache(None)
def nvcc_exist(nvcc_path: str='nvcc') -> bool:
    if False:
        while True:
            i = 10
    if nvcc_path is None:
        return False
    import subprocess
    res = subprocess.call(['which', nvcc_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res == 0