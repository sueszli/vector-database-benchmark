import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
aten = torch.ops.aten

def fetch_fake_tensors(match, kwarg_names) -> List[Tensor]:
    if False:
        i = 10
        return i + 15
    kwargs = match.kwargs
    return [kwargs[name].meta['val'] for name in kwarg_names]

def unwrap_fake_args(*arg_names):
    if False:
        i = 10
        return i + 15

    def decorator(func):
        if False:
            print('Hello World!')

        def wrapper(match):
            if False:
                while True:
                    i = 10
            fake_tensors = fetch_fake_tensors(match, arg_names)
            return func(*fake_tensors)
        return wrapper
    return decorator

def get_alignment_size(x: Tensor) -> int:
    if False:
        while True:
            i = 10
    if x.dtype == torch.float16 or x.dtype == torch.half or x.dtype == torch.bfloat16:
        return 8
    elif x.dtype == torch.float32 or x.dtype == torch.float:
        return 4
    else:
        return 0

def check_device(a: Tensor, b: Tensor) -> bool:
    if False:
        i = 10
        return i + 15
    return a.is_cuda and b.is_cuda

def check_dtype(a: Tensor, b: Tensor) -> bool:
    if False:
        i = 10
        return i + 15
    return a.is_floating_point() and b.is_floating_point()

def is_symbolic(a: Optional[Tensor]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return a is not None and any((isinstance(x, torch.SymInt) for x in chain(a.size(), a.stride())))

def any_is_symbolic(*args: Optional[Tensor]) -> bool:
    if False:
        i = 10
        return i + 15
    return any((is_symbolic(a) for a in args))

def should_pad_common(mat1: Tensor, mat2: Tensor, input: Optional[Tensor]=None) -> bool:
    if False:
        return 10
    return torch._inductor.config.shape_padding and check_device(mat1, mat2) and check_dtype(mat1, mat2) and (not any_is_symbolic(mat1, mat2, input))

def get_padded_length(x: int, alignment_size) -> int:
    if False:
        for i in range(10):
            print('nop')
    if alignment_size == 0 or x % alignment_size == 0:
        return 0
    return int((x // alignment_size + 1) * alignment_size) - x

def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
    if False:
        i = 10
        return i + 15
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1:])
    return torch.cat([x, pad], dim=dim)

def addmm_pattern(input: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float) -> Tensor:
    if False:
        while True:
            i = 10
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def should_pad_addmm(match: Match) -> bool:
    if False:
        return 10
    (mat1, mat2, input) = fetch_fake_tensors(match, ('mat1', 'mat2', 'input'))
    return should_pad_common(mat1, mat2, input) and should_pad_bench(mat1, mat2, torch.ops.aten.addmm, input=input)

def addmm_replace(input: Optional[Tensor], mat1: Tensor, mat2: Tensor, beta=1.0, alpha=1.0) -> Tensor:
    if False:
        i = 10
        return i + 15
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
        return pad_addmm(input, mat1, mat2, m_padded_length, k_padded_length, n_padded_length, beta, alpha)
    return aten.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def pad_addmm(input: Optional[Tensor], mat1: Tensor, mat2: Tensor, m_padded_length: int, k_padded_length: int, n_padded_length: int, beta=1.0, alpha=1.0):
    if False:
        for i in range(10):
            print('nop')
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
    elif m_padded_length != 0:
        mat1 = pad_dim(mat1, m_padded_length, 0)
    if input is not None and k_padded_length == 0:
        if n_padded_length != 0:
            if input.dim() == 2:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1:
                input = pad_dim(input, n_padded_length, 0)
        elif m_padded_length != 0 and input.dim() == 2:
            input = pad_dim(input, m_padded_length, 0)
    if k_padded_length != 0:
        return addmm_replace(input, mat1, mat2, beta=beta, alpha=alpha)
    elif n_padded_length != 0:
        return addmm_replace(input, mat1, mat2, beta=beta, alpha=alpha)[:, :-n_padded_length]
    else:
        return addmm_replace(input, mat1, mat2, beta=beta, alpha=alpha)[:-m_padded_length, :]

def is_mm_compute_bound(M: int, K: int, N: int, dtype: torch.dtype) -> bool:
    if False:
        i = 10
        return i + 15
    denominator = M * K + N * K + M * N
    if denominator == 0:
        return False
    arithmetic_intensity = M * N * K / denominator
    try:
        machine_balance = 1000 * utils.get_device_tflops(dtype) / utils.get_gpu_dram_gbps()
    except Exception:
        return True
    machine_balance = machine_balance * 0.5
    return arithmetic_intensity > machine_balance

@functools.lru_cache(None)
def get_pad_cache():
    if False:
        return 10
    return torch._inductor.codecache.LocalCache()

def get_cached_should_pad(key):
    if False:
        while True:
            i = 10
    return get_pad_cache().lookup(key)

def set_cached_should_pad(key, value):
    if False:
        i = 10
        return i + 15
    return get_pad_cache().set_value(key, value=value)

def should_pad_bench_key(mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor]=None) -> str:
    if False:
        for i in range(10):
            print('nop')

    def tensor_key(t):
        if False:
            return 10
        return (t.shape, t.stride(), t.dtype)
    tf32_key = None if mat1.dtype != torch.float32 else torch.backends.cuda.matmul.allow_tf32
    key = (tensor_key(mat1), tensor_key(mat2), op, input if input is None else tensor_key(input), tf32_key)
    return str(key)

def should_pad_bench(mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor]=None) -> bool:
    if False:
        print('Hello World!')
    if not has_triton():
        return False
    do_bench = functools.partial(utils.do_bench, warmup=5)
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat2.shape[2]
            n = mat2.shape[2]
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False
        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False
        if not is_mm_compute_bound(m, k, n, mat1.dtype):
            return False
        key = should_pad_bench_key(mat1, mat2, op, input)
        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad
        mat1 = torch.randn_like(mat1)
        mat2 = torch.randn_like(mat2)
        if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
            ori_time = do_bench(lambda : op(mat1, mat2))
        else:
            if input is not None:
                input = torch.randn_like(input)
            ori_time = do_bench(lambda : op(input, mat1, mat2))
        mat1_pad = torch.randn_like(mat1)
        mat2_pad = torch.randn_like(mat2)
        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and input.is_cuda:
                input_pad = torch.randn_like(input)
            pad_time = do_bench(lambda : pad_addmm(input_pad, mat1_pad, mat2_pad, m_padded_length, k_padded_length, n_padded_length))
        elif op is torch.ops.aten.mm:
            pad_time = do_bench(lambda : pad_mm(mat1_pad, mat2_pad, m_padded_length, k_padded_length, n_padded_length))
        else:
            pad_time = do_bench(lambda : pad_bmm(mat1_pad, mat2_pad, m_padded_length, k_padded_length, n_padded_length))
        should_pad = ori_time > pad_time * 1.1
        set_cached_should_pad(key, should_pad)
        return should_pad

def mm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    if False:
        i = 10
        return i + 15
    return aten.mm(mat1, mat2)

def should_pad_mm(match: Match) -> bool:
    if False:
        while True:
            i = 10
    (mat1, mat2) = fetch_fake_tensors(match, ('mat1', 'mat2'))
    return should_pad_common(mat1, mat2) and should_pad_bench(mat1, mat2, torch.ops.aten.mm)

def mm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
    k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
    return pad_mm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)

def pad_mm(mat1: Tensor, mat2: Tensor, m_padded_length: int, k_padded_length: int, n_padded_length: int) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
        return torch.ops.aten.mm(mat1, mat2)[:, :-n_padded_length]
    else:
        mat1 = pad_dim(mat1, m_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)[:-m_padded_length, :]

def bmm_pattern(mat1: Tensor, mat2: Tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    return aten.bmm(mat1, mat2)

def should_pad_bmm(match: Match) -> bool:
    if False:
        print('Hello World!')
    (mat1, mat2) = fetch_fake_tensors(match, ('mat1', 'mat2'))
    return should_pad_common(mat1, mat2) and should_pad_bench(mat1, mat2, torch.ops.aten.bmm)

def bmm_replace(mat1: Tensor, mat2: Tensor) -> Tensor:
    if False:
        print('Hello World!')
    m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
    k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
    n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
    if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
        return pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)
    return aten.bmm(mat1, mat2)

def pad_bmm(mat1: Tensor, mat2: Tensor, m_padded_length: int, k_padded_length: int, n_padded_length: int) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 2)
        mat2 = pad_dim(mat2, k_padded_length, 1)
        return aten.bmm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 2)
        return aten.bmm(mat1, mat2)[:, :, :-n_padded_length].contiguous()
    else:
        mat1 = pad_dim(mat1, m_padded_length, 1)
        return aten.bmm(mat1, mat2)[:, :-m_padded_length, :].contiguous()

@functools.lru_cache(None)
def _pad_mm_init():
    if False:
        print('Hello World!')
    from .joint_graph import patterns
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    dim2a = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim2b = functools.partial(torch.empty, (4, 4), device=device, requires_grad=True)
    dim3a = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim3b = functools.partial(torch.empty, (4, 4, 4), device=device, requires_grad=True)
    dim1a = functools.partial(torch.empty, 4, device=device, requires_grad=True)
    rep = {'beta': 0.213377, 'alpha': 0.113377}
    for (pattern, replacement, args, workaround, extra_check) in [(mm_pattern, mm_replace, [dim2a(), dim2b()], {}, should_pad_mm), (bmm_pattern, bmm_replace, [dim3a(), dim3b()], {}, should_pad_bmm), (addmm_pattern, addmm_replace, [dim1a(), dim2a(), dim2b()], rep, should_pad_addmm)]:
        assert isinstance(workaround, dict)
        register_replacement(pattern, replacement, args, joint_fwd_bwd, patterns, extra_check=extra_check, scalar_workaround=workaround)
        register_replacement(pattern, replacement, args, fwd_only, patterns, extra_check=extra_check, scalar_workaround=workaround)