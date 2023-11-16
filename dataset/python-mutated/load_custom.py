""" Loading of Deformable DETR's CUDA kernels"""
import os
from pathlib import Path

def load_cuda_kernels():
    if False:
        i = 10
        return i + 15
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent.parent.parent / 'kernels' / 'deformable_detr'
    src_files = [root / filename for filename in ['vision.cpp', os.path.join('cpu', 'ms_deform_attn_cpu.cpp'), os.path.join('cuda', 'ms_deform_attn_cuda.cu')]]
    load('MultiScaleDeformableAttention', src_files, with_cuda=True, extra_include_paths=[str(root)], extra_cflags=['-DWITH_CUDA=1'], extra_cuda_cflags=['-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'])
    import MultiScaleDeformableAttention as MSDA
    return MSDA