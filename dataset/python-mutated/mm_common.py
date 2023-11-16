import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
log = logging.getLogger(__name__)

def triton_config(num_stages, num_warps, **kwargs):
    if False:
        while True:
            i = 10
    from triton import Config
    return Config(kwargs, num_stages=num_stages, num_warps=num_warps)

def filtered_configs(m: int, n: int, k: int, configs: List[Tuple[int, int, int, int, int]], has_int8_tensor=False):
    if False:
        for i in range(10):
            print('nop')
    'Heuristic to shrink configs when they are bigger than the input size'
    min_block_size = 32 if has_int8_tensor else 16
    m = max(next_power_of_2(V.graph.sizevars.size_hint(m, fallback=torch._inductor.config.unbacked_symint_fallback)), min_block_size)
    n = max(next_power_of_2(V.graph.sizevars.size_hint(n, fallback=torch._inductor.config.unbacked_symint_fallback)), min_block_size)
    k = max(next_power_of_2(V.graph.sizevars.size_hint(k, fallback=torch._inductor.config.unbacked_symint_fallback)), min_block_size)
    used = set()
    for (block_m, block_n, block_k, num_stages, num_warps) in configs:
        block_m = max(min(block_m, m), min_block_size)
        block_n = max(min(block_n, n), min_block_size)
        block_k = max(min(block_k, k), min_block_size)
        num_warps = min(num_warps, block_m * block_n // 256)
        if (block_m, block_n, block_k, num_stages, num_warps) not in used:
            used.add((block_m, block_n, block_k, num_stages, num_warps))
            yield triton_config(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_stages=num_stages, num_warps=num_warps)
mm_kernel_configs = [{'config': (64, 64, 32, 2, 4), 'cond': True}, {'config': (64, 128, 32, 3, 4), 'cond': True}, {'config': (128, 64, 32, 3, 4), 'cond': True}, {'config': (64, 128, 32, 4, 8), 'cond': True}, {'config': (128, 64, 32, 4, 8), 'cond': True}, {'config': (64, 32, 32, 5, 8), 'cond': True}, {'config': (32, 64, 32, 5, 8), 'cond': True}, {'config': (128, 128, 32, 2, 8), 'cond': True}, {'config': (64, 64, 64, 3, 8), 'cond': True}, {'config': (32, 32, 128, 2, 4), 'cond': torch.version.hip is None}, {'config': (64, 64, 16, 2, 4), 'cond': True}, {'config': (32, 32, 16, 1, 2), 'cond': True}]
int8_mm_kernel_configs = [{'config': (64, 64, 32, 2, 4), 'cond': True}, {'config': (64, 128, 32, 3, 4), 'cond': True}, {'config': (128, 64, 32, 3, 4), 'cond': True}, {'config': (64, 128, 32, 4, 8), 'cond': True}, {'config': (128, 64, 32, 4, 8), 'cond': True}, {'config': (64, 32, 32, 5, 8), 'cond': True}, {'config': (32, 64, 32, 5, 8), 'cond': True}, {'config': (128, 128, 32, 2, 8), 'cond': True}, {'config': (64, 64, 64, 3, 8), 'cond': True}, {'config': (128, 256, 128, 3, 8), 'cond': torch.version.hip is None}, {'config': (256, 128, 128, 3, 8), 'cond': torch.version.hip is None}]
mm_platform_configs = tuple((cast(Tuple[int, int, int, int, int], config['config']) for config in mm_kernel_configs if config['cond']))
int8_platform_configs = tuple((cast(Tuple[int, int, int, int, int], config['config']) for config in int8_mm_kernel_configs if config['cond']))
if torch.version.hip:
    mm_platform_configs = tuple(((config[0], config[1], config[2], 1, config[4]) for config in mm_platform_configs))
    int8_platform_configs = tuple(((config[0], config[1], config[2], 1, config[4]) for config in mm_platform_configs))
mm_configs = functools.partial(filtered_configs, configs=mm_platform_configs)
int8_mm_configs = functools.partial(filtered_configs, configs=int8_platform_configs)

def mm_grid(m, n, meta):
    if False:
        return 10
    '\n    The CUDA grid size for matmul triton templates.\n    '
    return (cdiv(m, meta['BLOCK_M']) * cdiv(n, meta['BLOCK_N']), 1, 1)

def acc_type(dtype):
    if False:
        for i in range(10):
            print('nop')
    if dtype in (torch.float16, torch.bfloat16):
        return 'tl.float32'
    return f'tl.{dtype}'.replace('torch.', '')

def mm_options(config, sym_k, layout, b_prologue_cast_type=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Common options to matmul triton templates.\n    '
    even_k_symbolic = sympy.gcd(sym_k, config.kwargs['BLOCK_K']) == config.kwargs['BLOCK_K']
    return dict(GROUP_M=8, EVEN_K=even_k_symbolic, ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, ACC_TYPE=acc_type(layout.dtype), B_PROLOGUE_CAST_TYPE=b_prologue_cast_type, num_stages=config.num_stages, num_warps=config.num_warps, **config.kwargs)

def mm_args(mat1, mat2, *others, layout=None, out_dtype=None, use_4x2_dim=False):
    if False:
        i = 10
        return i + 15
    '\n    Common arg processing for mm,bmm,addmm,etc\n    '
    (mat1, mat2) = realize_inputs(mat1, mat2)
    (*b1, m, k1) = mat1.get_size()
    (*b2, k2, n) = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for (a, b) in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout
        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        layout = FixedLayout(mat1.get_device(), out_dtype, [*b, m, n])
    else:
        assert out_dtype is None, 'out_dtype is ignored if layout is specified.'
    from ..lowering import expand
    others = [realize_inputs(expand(x, layout.size)) for x in others]
    return [m, n, k, layout, mat1, mat2, *others]

def addmm_epilogue(dtype, alpha, beta):
    if False:
        return 10

    def epilogue(acc, bias):
        if False:
            i = 10
            return i + 15
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)
    return epilogue