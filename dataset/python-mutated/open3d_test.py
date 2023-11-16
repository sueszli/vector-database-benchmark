import os
import sys
import urllib.request
import zipfile
import numpy as np
import pytest

def torch_available():
    if False:
        return 10
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        return False
    return True

def list_devices():
    if False:
        return 10
    '\n    If Open3D is built with CUDA support:\n    - If cuda device is available, returns [Device("CPU:0"), Device("CUDA:0")].\n    - If cuda device is not available, returns [Device("CPU:0")].\n\n    If Open3D is built without CUDA support:\n    - returns [Device("CPU:0")].\n    '
    import open3d as o3d
    if o3d.core.cuda.device_count() > 0:
        return [o3d.core.Device('CPU:0'), o3d.core.Device('CUDA:0')]
    else:
        return [o3d.core.Device('CPU:0')]

def list_devices_with_torch():
    if False:
        i = 10
        return i + 15
    '\n    Similar to list_devices(), but take PyTorch available devices into account.\n    The returned devices are compatible on both PyTorch and Open3D.\n\n    If PyTorch is not available at all, empty list will be returned, thus the\n    test is effectively skipped.\n    '
    if torch_available():
        import open3d as o3d
        import torch
        if o3d.core.cuda.device_count() > 0 and torch.cuda.is_available() and (torch.cuda.device_count() > 0):
            return [o3d.core.Device('CPU:0'), o3d.core.Device('CUDA:0')]
        else:
            return [o3d.core.Device('CPU:0')]
    else:
        return []