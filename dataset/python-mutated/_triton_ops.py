import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE = int(os.getenv('TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE', 2))

def check(cond, msg):
    if False:
        return 10
    if not cond:
        raise ValueError(msg)

def check_bsr_layout(f_name, t):
    if False:
        for i in range(10):
            print('nop')
    check(t.layout == torch.sparse_bsr, f'{f_name}(): only BSR sparse format is supported for the sparse argument.')

def check_device(f_name, t, device):
    if False:
        return 10
    check(t.device == device and t.device.type == 'cuda', f'{f_name}(): all inputs are expected to be on the same GPU device.')

def check_mm_compatible_shapes(f_name, lhs, rhs):
    if False:
        while True:
            i = 10
    check(lhs.dim() >= 2 and rhs.dim() >= 2, f'{f_name}(): all inputs involved in the matrix product are expected to be at least 2D, but got lhs.dim() == {lhs.dim()} and rhs.dim() == {rhs.dim()}.')
    (m, kl) = lhs.shape[-2:]
    (kr, n) = rhs.shape[-2:]
    check(kl == kr, f"{f_name}(): arguments' sizes involved in the matrix product are not compatible for matrix multiplication, got lhs.shape[-1] == {kl} which is not equal to rhs.shape[-2] == {kr}.")

def check_dtype(f_name, t, dtype, *additional_dtypes):
    if False:
        while True:
            i = 10
    check(t.dtype == dtype and t.dtype in (torch.half, torch.bfloat16, torch.float) + tuple(*additional_dtypes), f'{f_name}(): all inputs are expected to be of the same dtype and one of (half, bfloat16, float32) or {additional_dtypes}, but got dtype == {t.dtype}.')

def check_blocksize(f_name, blocksize):
    if False:
        i = 10
        return i + 15
    assert len(blocksize) == 2

    def is_power_of_two(v):
        if False:
            return 10
        return not v & v - 1

    def is_compatible_blocksize(b):
        if False:
            while True:
                i = 10
        res = True
        for blocksize in b:
            res = (blocksize >= 16 and is_power_of_two(blocksize)) and res
        return res
    check(is_compatible_blocksize(blocksize), f"{f_name}(): sparse inputs' blocksize ({blocksize[0]}, {blocksize[1]}) should be at least 16 and a power of 2 in each dimension.")

def make_triton_contiguous(t):
    if False:
        return 10
    if (t.stride(-2) > 1 or t.dtype is torch.float32) and t.stride(-1) > 1:
        return t.contiguous()
    else:
        return t

def broadcast_batch_dims(f_name, *tensors):
    if False:
        print('Hello World!')
    try:
        return torch.broadcast_shapes(*(t.shape[:-2] for t in tensors))
    except Exception:
        check(False, f"{f_name}(): inputs' batch dimensions are not broadcastable!")

def slicer(dim, slice_range, *tensors):
    if False:
        return 10
    for t in tensors:
        slices = [slice(None)] * t.dim()
        slices[dim] = slice_range
        yield t[slices]

def multidim_slicer(dims, slices, *tensors):
    if False:
        for i in range(10):
            print('nop')
    for t in tensors:
        s = [slice(None)] * t.dim()
        for (d, d_slice) in zip(dims, slices):
            if d is not None:
                s[d] = d_slice
        yield t[s]

def ptr_stride_extractor(*tensors):
    if False:
        for i in range(10):
            print('nop')
    for t in tensors:
        yield t
        yield from t.stride()

def grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
    if False:
        i = 10
        return i + 15
    assert 0 <= len(full_grid) <= 3
    assert 0 <= len(grid_blocks) <= 3
    import itertools

    def generate_grid_points():
        if False:
            print('Hello World!')
        for (fg, mg) in zip(full_grid, grid_blocks):
            yield range(0, fg, mg)

    def generate_sliced_tensors(slices):
        if False:
            print('Hello World!')
        for (t, t_dims) in tensor_dims_map.items():
            yield next(multidim_slicer(t_dims, slices, t))
    for grid_point in itertools.product(*generate_grid_points()):
        grid = [min(fg - gp, mg) for (fg, gp, mg) in zip(full_grid, grid_point, grid_blocks)]
        slices = [slice(gp, gp + g) for (gp, g) in zip(grid_point, grid)]
        yield (grid[::-1], *generate_sliced_tensors(slices))

def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=None):
    if False:
        return 10
    cuda_max_grid = (2147483647, 65535, 65535)[::-1]
    if grid_blocks is None:
        grid_blocks = cuda_max_grid
    else:

        def valid_grid_dim(g, mg):
            if False:
                return 10
            if g is None:
                return mg
            else:
                return max(1, min(g, mg))
        grid_blocks = tuple((valid_grid_dim(g, mg) for (g, mg) in zip(grid_blocks, cuda_max_grid)))
    for (grid, *sliced_tensors) in grid_partitioner(full_grid, grid_blocks, tensor_dims_map):
        kernel(grid, *sliced_tensors)

def prepare_inputs(bsr, *dense_tensors):
    if False:
        return 10
    crow_indices = bsr.crow_indices().unsqueeze(0)
    col_indices = bsr.col_indices().unsqueeze(0)
    values = make_triton_contiguous(bsr.values().unsqueeze(0))
    tensors = [make_triton_contiguous(t.unsqueeze(0)) for t in dense_tensors]
    batch_dims_broadcasted = torch.broadcast_shapes(values.shape[:-3], *(t.shape[:-2] for t in tensors))

    def batch_broadcast_and_squash(t, batch_dims, invariant_dims):
        if False:
            i = 10
            return i + 15
        return t.broadcast_to(batch_dims + invariant_dims).flatten(0, len(batch_dims) - 1)
    crow_indices = batch_broadcast_and_squash(crow_indices, batch_dims_broadcasted, (-1,))
    col_indices = batch_broadcast_and_squash(col_indices, batch_dims_broadcasted, (-1,))
    values = batch_broadcast_and_squash(values, batch_dims_broadcasted, values.shape[-3:])
    tensors = [batch_broadcast_and_squash(t, batch_dims_broadcasted, t.shape[-2:]) for t in tensors]
    return (crow_indices, col_indices, values, *tensors)

def broadcast_batch_dims_bsr(f_name, bsr, *tensors):
    if False:
        for i in range(10):
            print('nop')
    batch_shape = broadcast_batch_dims(f_name, bsr, *tensors)
    crow_indices = bsr.crow_indices().broadcast_to(batch_shape + (-1,))
    col_indices = bsr.col_indices().broadcast_to(batch_shape + (-1,))
    values = bsr.values().broadcast_to(batch_shape + bsr.values().shape[-3:])
    size = batch_shape + bsr.shape[-2:]
    return torch.sparse_compressed_tensor(crow_indices, col_indices, values, size=size, layout=bsr.layout)

def tile_to_blocksize(t, blocksize):
    if False:
        while True:
            i = 10
    (*rest, m, n) = t.shape
    new_shape = rest + [m // blocksize[0], blocksize[0], n // blocksize[1], blocksize[1]]
    return t.view(new_shape).transpose(-3, -2)

def as1Dbatch(tensor):
    if False:
        while True:
            i = 10
    'Return tensor as 3D tensor by either prepending new dimensions to\n    the tensor shape (when ``tensor.ndim < 3``), or by collapsing\n    starting dimensions into the first dimension (when ``tensor.ndim >\n    3``).\n    '
    while tensor.ndim < 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 3:
        tensor = tensor.flatten(0, tensor.ndim - 3)
    assert tensor.ndim == 3, tensor.shape
    return tensor

def scatter_mm(blocks, others, indices_data, *, accumulators=None):
    if False:
        return 10
    'Scattered matrix multiplication of tensors.\n\n    A scattered matrix multiplication is defined as a series of matrix\n    multiplications applied to input tensors according to the input\n    and output mappings specified by indices data.\n\n    The following indices data formats are supported for defining a\n    scattered matrix multiplication operation (:attr:`indices_data[0]`\n    holds the name of the indices data format as specified below):\n\n    - ``"scatter_mm"`` - matrix multiplications scattered in batches\n      of tensors.\n\n      If :attr:`blocks` is a :math:`(* \times M \times K) tensor,\n      :attr:`others` is a :math:`(* \times K \times N)` tensor,\n      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor,\n      and :attr:`indices = indices_data[\'indices\']` is a :math:`(*\n      \times 3)` tensor, then the operation is equivalent to the\n      following code::\n\n        c_offsets, pq = indices_data[1:]\n        for r in range(len(c_offsets) - 1):\n            for g in range(c_offsets[r], c_offsets[r + 1]):\n                p, q = pq[g]\n                accumulators[r] += blocks[p] @ others[q]\n\n    - ``"bsr_strided_mm"`` - matrix multiplications scattered in\n      batches of tensors and a tensor.\n\n      If :attr:`blocks` is a :math:`(Ms \times Ks) tensor,\n      :attr:`others` is a :math:`(* \times K \times N)` tensor,\n      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor, then\n      the operation is equivalent to the following code::\n\n        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]\n        for b in range(nbatches):\n            for i, r in enumerate(r_offsets):\n                r0, r1 = divmod(r, N)\n                acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]\n                for g in range(c_indices[i], c_indices[i+1]):\n                    p = p_offsets[g]\n                    q0, q1 = divmod(q_offsets[g], N)\n                    acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]\n\n      where ``Ns = N // meta[\'SPLIT_N\']``, and ``M`` and ``K`` are\n      integer multiples of ``Ms`` and ``Ks``, respectively.\n\n    - ``"bsr_strided_mm_compressed"`` - matrix multiplications\n      scattered in batches of tensors and a tensor. A memory and\n      processor efficient version of ``"bsr_strided_mm"`` format.  If\n      :attr:`blocks` is a :math:`(Ms \times Ks) tensor, :attr:`others`\n      is a :math:`(* \times K \times N)` tensor, :attr:`accumulators`\n      is a :math:`(* \times M \times N)` tensor, then the operation is\n      equivalent to the following code::\n\n        c_indices, r_offsets, q_offsets, meta = indices_data[1:]\n        for b in range(nbatches):\n            for r in r_offsets:\n                m = (r // N) // Ms\n                n = (r % N) // Ns\n                r0, r1 = divmod(r, N)\n                c0, c1 = c_indices[m], c_indices[m + 1]\n                acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]\n                for i, p in enumerate(range(c0, c1)):\n                    q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i]\n                    q0, q1 = divmod(q, N)\n                    acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]\n\n      where ``Ns = N // meta[\'SPLIT_N\']``, and ``M`` and ``K`` are\n      integer multiples of ``Ms`` and ``Ks``, respectively.\n\n      Notice that the order of ``r_offsets`` items can be arbitrary;\n      this property enables defining swizzle operators via\n      rearrangements of ``r_offsets`` items..\n\n    Auxilary functions are provided for pre-computing\n    :attr:`indices_data`. For example,\n    :func:`bsr_scatter_mm_indices_data` is used to define indices data\n    for matrix multiplication of BSR and strided tensors.\n\n    Parameters\n    ----------\n    blocks (Tensor): a 3-D tensor of first matrices to be multiplied\n\n    others (Tensor): a tensor of second matrices to be multiplied. If\n      ``indices_data[0]=="scatter_mm"``, the tensor is a 1-D batch\n      tensor of second input matrices to be multiplied. Otherwise, the\n      second input matrices are slices of the :attr:`others` tensor.\n    indices_data (tuple): a format data that defines the inputs and\n      outputs of scattered matrix multiplications.\n\n    Keyword arguments\n    -----------------\n\n    accumulators (Tensor, optional): a tensor of matrix product\n      accumulators. If ``indices_data[0]=="scatter_mm"``, the tensor\n      is a 1-D batch tensor of output matrices. Otherwise, output\n      matrices are slices of the :attr:`accumulators` tensor.\n    '
    indices_format = indices_data[0]
    assert blocks.ndim == 3
    (P, Ms, Ks) = blocks.shape
    if indices_format == 'scatter_mm':
        (c_offsets, pq) = indices_data[1:]
        assert others.ndim == 3
        (Q, Ks_, Ns) = others.shape
        assert Ks == Ks_
        if accumulators is None:
            R = c_offsets.shape[0] - 1
            accumulators = torch.zeros((R, Ms, Ns), dtype=blocks.dtype, device=blocks.device)
        else:
            (R, Ms_, Ns_) = accumulators.shape
            assert Ms_ == Ms
            assert Ns_ == Ns
        if Ms % 16 or Ks % 16 or Ns % 16 or (_scatter_mm2 is None):
            for r in range(c_offsets.shape[0] - 1):
                g0 = c_offsets[r]
                g1 = c_offsets[r + 1]
                for g in range(g0, g1):
                    (p, q) = pq[g]
                    accumulators[r] += blocks[p] @ others[q]
        else:
            _scatter_mm2(blocks, others, c_offsets, pq, accumulators)
        return accumulators
    elif indices_format == 'bsr_strided_mm':
        others_shape = others.shape
        others = as1Dbatch(others)
        (B, K, N) = others.shape
        assert K % Ks == 0
        (c_indices, r_offsets, p_offsets, q_offsets, meta) = indices_data[1:]
        SPLIT_N = meta['SPLIT_N']
        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros((*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device)
        else:
            (M, N_) = accumulators.shape[-2:]
            assert N_ == N
        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)
        Ns = N // SPLIT_N
        if Ms % 16 or Ks % 16 or Ns % 16 or (_scatter_mm6 is None):
            accumulators.zero_()
            for b in range(B):
                for r in range(r_offsets.shape[0]):
                    r_ = r_offsets[r].item()
                    g0 = c_indices[r].item()
                    g1 = c_indices[r + 1].item()
                    (r0, r1) = divmod(r_, N)
                    acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
                    for g in range(g0, g1):
                        (p, q) = (p_offsets[g], q_offsets[g])
                        (q0, q1) = divmod(q.item(), N)
                        acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]
        else:
            _scatter_mm6(blocks, others, c_indices, r_offsets, p_offsets, q_offsets, meta, accumulators)
        return accumulators.view(accumulators_shape)
    elif indices_format == 'bsr_strided_mm_compressed':
        others_shape = others.shape
        others = as1Dbatch(others)
        (B, K, N) = others.shape
        assert K % Ks == 0
        (c_indices, r_offsets, q_offsets, meta) = indices_data[1:]
        SPLIT_N = meta['SPLIT_N']
        if accumulators is None:
            M = Ms + (r_offsets.max().item() + 1) // N
            accumulators = torch.zeros((*others_shape[:-2], M, N), dtype=blocks.dtype, device=blocks.device)
        else:
            (M, N_) = accumulators.shape[-2:]
            assert N_ == N
        accumulators_shape = accumulators.shape
        accumulators = as1Dbatch(accumulators)
        Ns = N // SPLIT_N
        if Ms % 16 or Ks % 16 or Ns % 16 or (_scatter_mm6 is None):
            for b in range(B):
                for j in range(len(r_offsets)):
                    (r0, r1) = divmod(r_offsets[j].item(), N)
                    m = r0 // Ms
                    n = r1 // Ns
                    c0 = c_indices[m].item()
                    c1 = c_indices[m + 1].item()
                    acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
                    for (i, p) in enumerate(range(c0, c1)):
                        q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i].item()
                        (q0, q1) = divmod(q, N)
                        acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]
        else:
            p_offsets = torch.empty((0,), dtype=q_offsets.dtype, device=q_offsets.device)
            _scatter_mm6(blocks, others, c_indices, r_offsets, p_offsets, q_offsets, meta, accumulators)
        return accumulators.view(accumulators_shape)
    else:
        raise NotImplementedError(indices_format)

def scatter_mm_meta(M, K, N, Ms, Ks, GROUP_SIZE=None, TILE_M=None, TILE_N=None, SPLIT_N=None, num_warps=None, num_stages=None, **extra):
    if False:
        while True:
            i = 10
    if {TILE_M, TILE_N, SPLIT_N, num_warps, num_stages, GROUP_SIZE} == {None}:
        device_name = torch.cuda.get_device_name()
        meta = get_meta('scatter_mm', (M, K, N, Ms, Ks), device_name, version=(0, torch.float16, 0.5))
        if meta is not None:
            meta.update(**extra)
            return meta
        if (M, K, N) == (256,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 1
                TILE_M = 16
                TILE_N = 16
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 2
                TILE_M = 32
                TILE_N = 16
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 1
                TILE_M = 32
                TILE_N = 32
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 1
                TILE_M = 32
                TILE_N = 32
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4
        elif (M, K, N) == (512,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 8
                TILE_M = 16
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 2
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 8
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 2
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 4
                TILE_M = 32
                TILE_N = 128
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 8
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
        elif (M, K, N) == (1024,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 4
                TILE_M = 16
                TILE_N = 128
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 8
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 2
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (256, 256):
                SPLIT_N = 16
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4
        elif (M, K, N) == (2048,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 4
                TILE_M = 16
                TILE_N = 128
                GROUP_SIZE = 8
                num_stages = 1
                num_warps = 1
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 4
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 1
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 4
                TILE_M = 64
                TILE_N = 128
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (128, 128):
                SPLIT_N = 8
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 4
                num_stages = 1
                num_warps = 4
            elif (Ms, Ks) == (256, 256):
                SPLIT_N = 4
                TILE_M = 64
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4
        elif (M, K, N) == (4096,) * 3:
            if (Ms, Ks) == (16, 16):
                SPLIT_N = 2
                TILE_M = 16
                TILE_N = 256
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 2
            elif (Ms, Ks) == (32, 32):
                SPLIT_N = 2
                TILE_M = 32
                TILE_N = 64
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 1
            elif (Ms, Ks) == (64, 64):
                SPLIT_N = 2
                TILE_M = 64
                TILE_N = 128
                GROUP_SIZE = 2
                num_stages = 1
                num_warps = 4
    if SPLIT_N is None:
        SPLIT_N = {16: 1, 32: 2, 64: 4, 128: 8, 256: 16, 512: 8, 1024: 16, 4096: 32, 8192: 64}.get(N, 16)
        if Ms >= 512 and N >= 2048:
            SPLIT_N = 1
    Ns = N // SPLIT_N
    if TILE_M is None:
        TILE_M = min(64 if Ns < 512 else 32, Ms)
    if TILE_N is None:
        TILE_N = min(64 if Ns < 512 else 32, Ns)
    num_stages = num_stages or 1
    if num_warps is None:
        if min(M, N) > 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 1024:
            num_warps = {16: 1, 32: 1, 64: 2}.get(Ms, 4)
        elif min(M, N) == 256:
            num_warps = {16: 1, 32: 4}.get(Ms, 4)
        else:
            num_warps = {16: 1, 32: 2}.get(Ms, 4)
    GROUP_SIZE = GROUP_SIZE or 4
    assert TILE_M <= Ms, dict(TILE_M=TILE_M, Ms=Ms)
    assert TILE_N <= Ns, dict(TILE_N=TILE_N, Ns=Ns)
    assert Ms <= M, dict(M=M, Ms=Ms)
    assert Ns <= N, dict(N=N, Ns=Ns)
    assert Ks <= K, dict(K=K, Ks=Ks)
    return dict(TILE_M=TILE_M, TILE_N=TILE_N, GROUP_SIZE=GROUP_SIZE, num_stages=num_stages, num_warps=num_warps, SPLIT_N=SPLIT_N, **extra)

def bsr_dense_mm_meta(M, K, N, Ms, Ks, GROUP_SIZE_ROW=None, num_warps=None, num_stages=None, **extra):
    if False:
        for i in range(10):
            print('nop')
    if {num_warps, num_stages, GROUP_SIZE_ROW} == {None}:
        device_name = torch.cuda.get_device_name()
        meta = get_meta('bsr_dense_mm', (M, K, N, Ms, Ks), device_name, version=(0, torch.float16, 0.5))
        if meta is not None:
            meta.update(**extra)
            return meta
    GROUP_SIZE_ROW = GROUP_SIZE_ROW or 4
    num_stages = num_stages or 1
    num_warps = num_warps or 4
    return dict(GROUP_SIZE_ROW=GROUP_SIZE_ROW, num_stages=num_stages, num_warps=num_warps, **extra)

class TensorAsKey:
    """A light-weight wrapper of a tensor that enables storing tensors as
    keys with efficient memory reference based comparision as an
    approximation to data equality based keys.

    Motivation: the hash value of a torch tensor is tensor instance
    based that does not use data equality and makes the usage of
    tensors as keys less useful. For instance, the result of
    ``len({a.crow_indices(), a.crow_indices()})`` is `2`, although,
    the tensor results from `crow_indices` method call are equal, in
    fact, these share the same data storage.
    On the other hand, for efficient caching of tensors we want to
    avoid calling torch.equal that compares tensors item-wise.

    TensorAsKey offers a compromise in that it guarantees key equality
    of tensors that references data in the same storage in the same
    manner and without accessing underlying data. However, this
    approach does not always guarantee correctness. For instance, for
    a complex tensor ``x``, we have ``TensorAsKey(x) ==
    TensorAsKey(x.conj())`` while ``torch.equal(x, x.conj())`` would
    return False.
    """

    def __init__(self, obj):
        if False:
            print('Hello World!')

        def get_tensor_key(obj):
            if False:
                while True:
                    i = 10
            assert not (obj.dtype.is_floating_point or obj.dtype.is_complex), obj.dtype
            return (obj.data_ptr(), obj.storage_offset(), obj.shape, obj.stride(), obj.dtype)
        self._obj_ref = weakref.ref(obj)
        if obj.layout is torch.strided:
            self.key = get_tensor_key(obj)
        elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
            self.key = (get_tensor_key(obj.crow_indices()), get_tensor_key(obj.col_indices()))
        elif obj.layout in {torch.sparse_csc, torch.sparse_bsc}:
            self.key = (get_tensor_key(obj.ccol_indices()), get_tensor_key(obj.row_indices()))
        else:
            raise NotImplementedError(obj.layout)
        self._hash = hash(self.key)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._hash

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, TensorAsKey):
            return False
        if self.obj is None or other.obj is None:
            return self is other
        return self.key == other.key

    @property
    def obj(self):
        if False:
            print('Hello World!')
        'Return object if alive, otherwise None.'
        return self._obj_ref()

@lru_cache(maxsize=TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE)
def _bsr_scatter_mm_indices_data(indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, compressed_sparse_tensor_as_key):
    if False:
        for i in range(10):
            print('nop')
    bsr = compressed_sparse_tensor_as_key.obj
    assert bsr is not None
    (crow_indices, col_indices) = (bsr.crow_indices(), bsr.col_indices())
    device = crow_indices.device
    indices_dtype = torch.int32
    if indices_format == 'bsr_strided_mm_compressed':
        Ns = N // SPLIT_N
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            q_offsets_lst.append((col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N) + b.repeat_interleave(r1 - r0))
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = crow_indices
        nnz_per_row = crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N)
        (nnz_per_row, indices) = nnz_per_row.sort(descending=True, stable=True)
        r_offsets = r_offsets[indices]
        return (indices_format, c_indices, r_offsets, q_offsets)
    elif indices_format == 'bsr_strided_mm':
        Ns = N // SPLIT_N
        p_offsets_lst = []
        q_offsets_lst = []
        b = torch.arange(SPLIT_N, dtype=indices_dtype, device=device) * Ns
        for m in range(M // Ms):
            r0 = crow_indices[m].item()
            r1 = crow_indices[m + 1].item()
            if r1 == r0:
                continue
            p_offsets_lst.append(torch.arange(r0, r1, dtype=indices_dtype, device=device).repeat(SPLIT_N))
            q_offsets_lst.append((col_indices[r0:r1] * (Ks * N)).repeat(SPLIT_N) + b.repeat_interleave(r1 - r0))
        q_offsets = torch.cat(q_offsets_lst)
        crow_indices_diff = crow_indices.diff()
        non_zero_row_indices = crow_indices_diff.nonzero()
        a = non_zero_row_indices * (Ms * N)
        r_offsets = (a + b).view(-1)
        c_indices = torch.cat((crow_indices[:1], torch.cumsum(crow_indices_diff[non_zero_row_indices].repeat_interleave(SPLIT_N), 0)))
        p_offsets = torch.cat(p_offsets_lst)
        return (indices_format, c_indices, r_offsets, p_offsets, q_offsets)
    elif indices_format == 'scatter_mm':
        Ns = Ms
        c_indices = [0]
        pq_offsets = []
        for b in range(nbatches):
            for m in range(M // Ms):
                r0 = crow_indices[m].item()
                r1 = crow_indices[m + 1].item()
                for n in range(N // Ns):
                    c_indices.append(c_indices[-1] + r1 - r0)
                    for t in range(r1 - r0):
                        p = r0 + t
                        q = (col_indices[p].item() + b * (K // Ks)) * (N // Ns) + n
                        pq_offsets.append([p, q])
        return (indices_format, torch.tensor(c_indices, dtype=indices_dtype, device=device), torch.tensor(pq_offsets, dtype=indices_dtype, device=device))
    else:
        raise ValueError(f'Invalid indices_format={indices_format!r}. Expected bsr_strided_mm_compressed|bsr_strided_mm|scatter_mm')

def bsr_scatter_mm_indices_data(bsr, other, indices_format='bsr_strided_mm_compressed', **meta_input):
    if False:
        print('Hello World!')
    'Computes indices data for :func:`scatter_mm` used in BSR and\n    strided tensor matrix multiplication.\n    '
    assert bsr.dense_dim() == 0
    assert bsr.ndim == 2
    crow_indices = bsr.crow_indices()
    col_indices = bsr.col_indices()
    blocksize = bsr.values().shape[-2:]
    (M, K) = bsr.shape
    (Ms, Ks) = blocksize
    (K_, N) = other.shape[-2:]
    assert K_ == K
    nbatches = other.shape[:-2].numel()
    meta = scatter_mm_meta(M, K, N, Ms, Ks, **meta_input)
    if 'allow_tf32' not in meta_input:
        meta.update(allow_tf32=bsr.dtype in {torch.float16, torch.bfloat16})
    SPLIT_N = meta['SPLIT_N']
    indices_data = _bsr_scatter_mm_indices_data(indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, TensorAsKey(bsr))
    if indices_format == 'bsr_strided_mm_compressed':
        meta.update(is_compressed=True)
        return indices_data + (meta,)
    elif indices_format == 'bsr_strided_mm':
        meta.update(is_compressed=False)
        return indices_data + (meta,)
    else:
        return indices_data

def bsr_scatter_mm(bsr, other, indices_data=None, out=None):
    if False:
        return 10
    'BSR @ strided -> strided\n    '
    assert bsr.ndim == 2
    assert other.ndim >= 2
    (Ms, Ks, Ns) = (bsr.shape[-2], bsr.shape[-1], other.shape[-1])
    blocksize = bsr.values().shape[-2:]
    if indices_data is None:
        indices_data = bsr_scatter_mm_indices_data(bsr, other, indices_format='bsr_strided_mm_compressed')
    indices_format = indices_data[0]
    if out is None:
        out = torch.empty((*other.shape[:-2], Ms, Ns), dtype=bsr.dtype, device=bsr.device)
    out_shape = out.shape
    out = as1Dbatch(out)
    if bsr._nnz() == 0:
        out.zero_()
    elif indices_format in {'bsr_strided_mm_compressed', 'bsr_strided_mm'}:
        out.zero_()
        scatter_mm(bsr.values(), other, indices_data, accumulators=out)
    elif indices_format == 'scatter_mm':
        nbatches = other.shape[:-2].numel()
        accumulators = torch.zeros((nbatches * Ms // blocksize[0] * Ns // blocksize[0], blocksize[0], blocksize[0]), dtype=bsr.dtype, device=bsr.device)
        others = as1Dbatch(other).transpose(-2, -1).view(nbatches, Ns // blocksize[0], blocksize[0], Ks // blocksize[1], blocksize[1]).movedim((3, 1, 4, 2), (1, 2, 3, 4)).flatten(0, 2)
        scatter_mm(bsr.values(), others, indices_data, accumulators=accumulators)
        out.copy_(accumulators.unflatten(0, (nbatches, Ms // blocksize[0], Ns // blocksize[0])).movedim((1, 2, 3, 4), (3, 1, 4, 2)).reshape(nbatches, Ns, Ms).transpose(-2, -1))
    else:
        raise NotImplementedError(indices_format)
    return out.view(out_shape)
if has_triton():
    import triton
    import triton.language as tl
    from typing import Optional, Tuple

    @triton.jit
    def _sampled_addmm_kernel(alpha, beta, IS_BETA_ZERO: tl.constexpr, BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr, k, TILE_K: tl.constexpr, values_ptr, values_batch_stride, values_nnz_stride, values_row_block_stride, values_col_block_stride, crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, col_indices_ptr, col_indices_batch_stride, col_indices_stride, mat1_ptr, mat1_batch_stride, mat1_tiled_row_stride, mat1_tiled_col_stride, mat1_row_block_stride, mat1_col_block_stride, mat2_ptr, mat2_batch_stride, mat2_tiled_row_stride, mat2_tiled_col_stride, mat2_row_block_stride, mat2_col_block_stride, acc_dtype: tl.constexpr, allow_tf32: tl.constexpr):
        if False:
            while True:
                i = 10
        batch_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)
        crow_indices_offset_ptr = crow_indices_ptr + crow_indices_batch_stride * batch_pid + crow_indices_stride * row_block_pid
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return
        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)
        values_block_ptrs = values_ptr + values_batch_stride * batch_pid + values_nnz_stride * nnz_offset + values_row_block_stride * row_block_arange[:, None] + values_col_block_stride * col_block_arange[None, :]
        col_index_nnz_ptr = col_indices_ptr + col_indices_batch_stride * batch_pid + col_indices_stride * nnz_offset
        mat1_block_ptrs = mat1_ptr + mat1_batch_stride * batch_pid + mat1_tiled_row_stride * row_block_pid + mat1_row_block_stride * row_block_arange[:, None]
        mat2_block_ptrs = mat2_ptr + mat2_batch_stride * batch_pid + mat2_col_block_stride * col_block_arange[None, :]
        k_tile_arange = tl.arange(0, TILE_K)
        for _ in range(row_nnz):
            acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
            col_block = tl.load(col_index_nnz_ptr)
            for k_tile in range(0, k, TILE_K):
                k_offsets = k_tile + k_tile_arange
                mask_k = k_offsets < k
                mat1_block = tl.load(mat1_block_ptrs + mat1_col_block_stride * k_offsets[None, :], mask=mask_k[None, :], other=0.0)
                mat2_block = tl.load(mat2_block_ptrs + mat2_tiled_col_stride * col_block + mat2_row_block_stride * k_offsets[:, None], mask=mask_k[:, None], other=0.0)
                acc_block += tl.dot(mat1_block, mat2_block, allow_tf32=allow_tf32, out_dtype=acc_dtype)
            if IS_BETA_ZERO:
                acc_block *= alpha
            else:
                acc_block = alpha * acc_block + beta * tl.load(values_block_ptrs)
            tl.store(values_block_ptrs, acc_block.to(values_ptr.dtype.element_ty))
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride

    @triton.jit
    def _bsr_strided_dense_rowspace_kernel(BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr, values_ptr, values_batch_stride, values_nnz_stride, values_row_block_stride, values_col_block_stride, crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, col_indices_ptr, col_indices_batch_stride, col_indices_stride, dense_ptr, dense_batch_stride, dense_tiled_row_stride, dense_tiled_col_stride, dense_row_block_stride, dense_col_block_stride, output_ptr, output_batch_stride, output_tiled_row_stride, output_tiled_col_stride, output_row_block_stride, output_col_block_stride, acc_dtype: tl.constexpr, allow_tf32: tl.constexpr, GROUP_SIZE_ROW: tl.constexpr):
        if False:
            i = 10
            return i + 15
        batch_pid = tl.program_id(axis=2)
        row_block_pid = tl.program_id(axis=0)
        col_block_pid = tl.program_id(axis=1)
        n_block_rows = tl.num_programs(axis=0)
        n_block_cols = tl.num_programs(axis=1)
        (row_block_pid, col_block_pid) = tl.swizzle2d(row_block_pid, col_block_pid, n_block_rows, n_block_cols, GROUP_SIZE_ROW)
        crow_indices_offset_ptr = crow_indices_ptr + crow_indices_batch_stride * batch_pid + crow_indices_stride * row_block_pid
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return
        row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
        col_block_arange = tl.arange(0, BLOCKSIZE_COL)
        values_block_ptrs = values_ptr + values_batch_stride * batch_pid + values_nnz_stride * nnz_offset + values_row_block_stride * row_block_arange[:, None] + values_col_block_stride * col_block_arange[None, :]
        dense_block_ptrs = dense_ptr + dense_batch_stride * batch_pid + dense_tiled_col_stride * col_block_pid + dense_row_block_stride * col_block_arange[:, None] + dense_col_block_stride * row_block_arange[None, :]
        output_ptrs = output_ptr + output_batch_stride * batch_pid + output_tiled_row_stride * row_block_pid + output_tiled_col_stride * col_block_pid + output_row_block_stride * row_block_arange[:, None] + output_col_block_stride * row_block_arange[None, :]
        col_index_nnz_ptr = col_indices_ptr + col_indices_batch_stride * batch_pid + col_indices_stride * nnz_offset
        output_acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_ROW), dtype=acc_dtype)
        for _ in range(row_nnz):
            values_block = tl.load(values_block_ptrs)
            dense_row_idx = tl.load(col_index_nnz_ptr)
            dense_block = tl.load(dense_block_ptrs + dense_tiled_row_stride * dense_row_idx)
            output_acc_block += tl.dot(values_block, dense_block, allow_tf32=allow_tf32, out_dtype=acc_dtype)
            values_block_ptrs += values_nnz_stride
            col_index_nnz_ptr += col_indices_stride
        tl.store(output_ptrs, output_acc_block.to(output_ptr.dtype.element_ty))

    def _run_dense_rowspace_kernel(blocksize, values, crow_indices, col_indices, dense, output, max_grid, meta):
        if False:
            while True:
                i = 10
        n_batches = dense.size(0)
        n_block_rows = crow_indices.size(-1) - 1
        n_block_cols = dense.size(-3)
        full_grid = (n_batches, n_block_cols, n_block_rows)
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:3][::-1]) + (None,) * (3 - len(max_grid[:3]))
        else:
            grid_blocks = None
        tensor_dims_map = {values: (0, None, None), crow_indices: (0, None, -1), col_indices: (0, None, None), dense: (0, -3, None), output: (0, -3, -4)}
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        def kernel(grid, *sliced_tensors):
            if False:
                while True:
                    i = 10
            _bsr_strided_dense_rowspace_kernel[grid](*blocksize, *ptr_stride_extractor(*sliced_tensors), acc_dtype=acc_dtype, allow_tf32=allow_tf32, **meta)
        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    def _run_sampled_addmm_kernel(alpha, beta, is_beta_zero, blocksize, k, tile_k, values, crow_indices, col_indices, mat1, mat2, max_grid):
        if False:
            while True:
                i = 10
        n_batches = values.size(0)
        n_block_rows = crow_indices.size(-1) - 1
        full_grid = (n_batches, n_block_rows)
        if max_grid is not None:
            grid_blocks = tuple(max_grid[:2][::-1]) + (None,) * (2 - len(max_grid[:2]))
        else:
            grid_blocks = None
        tensor_dims_map = {values: (0, None), crow_indices: (0, -1), col_indices: (0, None), mat1: (0, -4), mat2: (0, None)}
        if values.dtype in (torch.half, torch.bfloat16):
            acc_dtype = tl.float32
            allow_tf32 = True
        else:
            acc_dtype = tl.float64
            allow_tf32 = False

        def kernel(grid, *sliced_tensors):
            if False:
                print('Hello World!')
            _sampled_addmm_kernel[grid](alpha, beta, is_beta_zero, *blocksize, k, tile_k, *ptr_stride_extractor(*sliced_tensors), acc_dtype=acc_dtype, allow_tf32=allow_tf32, num_stages=1, num_warps=4)
        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)

    def sampled_addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta=1.0, alpha=1.0, out: Optional[torch.Tensor]=None, skip_checks: bool=False, max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None):
        if False:
            return 10
        f_name = 'sampled_addmm'
        check_bsr_layout(f_name, input)
        input_broadcasted = broadcast_batch_dims_bsr(f_name, input, mat1, mat2)
        if not skip_checks:
            check_device(f_name, mat1, input.device)
            check_device(f_name, mat2, input.device)
            if beta != 0.0 and input.dtype is torch.bool:
                check(False, f'{f_name}(): having beta == {beta} not equal to 0.0 with boolean mask is not allowed.')
            if input.dtype is not torch.bool:
                check_dtype(f_name, mat1, input.dtype)
                check_dtype(f_name, mat2, input.dtype)
            else:
                check_dtype(f_name, mat1, mat2.dtype)
            check_mm_compatible_shapes(f_name, mat1, mat2)
            if out is not None:
                check_bsr_layout(f_name, out)
                check_device(f_name, out, mat1.device)
                check_dtype(f_name, out, input.dtype)
                check(out.shape == input_broadcasted.shape and out._nnz() == input._nnz(), f'{f_name}(): Expects `out` to be of shape {input_broadcasted.shape} and with nnz equal to {input_broadcasted._nnz()} but got out.shape = {out.shape} and out.nnz = {out._nnz()}')
        if out is None:
            out = input_broadcasted.to(mat1.dtype, copy=True)
        else:
            out.copy_(input_broadcasted)
        if out.numel() == 0 or out._nnz() == 0:
            return out
        blocksize = out.values().shape[-2:]
        m = mat1.size(-2)
        n = mat2.size(-1)
        k = mat1.size(-1)
        if alpha == 0.0 or k == 0:
            out.values().mul_(beta)
            return out
        out_backup = out
        (crow_indices, col_indices, values, mat1, mat2) = prepare_inputs(out, mat1, mat2)
        mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
        mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
        tile_k = max(*blocksize)
        _run_sampled_addmm_kernel(alpha, beta, beta == 0.0, blocksize, k, tile_k, values, crow_indices, col_indices, mat1, mat2, max_grid)
        if out_backup.values().stride()[-3:] != values.stride()[-3:]:
            out_backup.values().copy_(values.reshape(out_backup.values().shape))
        return out_backup

    def bsr_dense_mm(bsr: torch.Tensor, dense: torch.Tensor, *, out: Optional[torch.Tensor]=None, skip_checks: bool=False, max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]]=None, meta: Optional[dict]=None, enable_bsr_scatter_mm: bool=True):
        if False:
            for i in range(10):
                print('nop')
        f_name = 'bsr_dense_mm'
        (m, kl) = bsr.shape[-2:]
        if not skip_checks:
            check_bsr_layout(f_name, bsr)
            check_device(f_name, bsr, dense.device)
            check_dtype(f_name, bsr, dense.dtype)
            check_mm_compatible_shapes(f_name, bsr, dense)
            n = dense.size(-1)
            (row_block, col_block) = bsr.values().shape[-2:]
            check(not n % row_block, f'bsr_dense_mm(): dense.size(-1) == {n} should be divisible by blocksize[0] == {row_block}.')
            check_blocksize(f_name, (row_block, col_block))
        else:
            (kr, n) = dense.shape[-2:]
        original_batch_dims_broadcasted = broadcast_batch_dims(f_name, bsr, dense)
        if out is not None and (not skip_checks):
            expected_out_shape = original_batch_dims_broadcasted + (m, n)
            check(out.shape == expected_out_shape, f'bsr_dense_mm(): `out` argument has wrong shape, expected {expected_out_shape}, but got {out.shape}.')
            check(out.is_contiguous() or out.transpose(-2, -1).is_contiguous(), 'bsr_dense_mm(): only row-major/col-major `out` arguments are supported, i.e. (out.is_contiguous() or out.transpose(-2, -1).is_contiguous()) should be True.')
        if out is None:
            out = dense.new_empty(original_batch_dims_broadcasted + (m, n))
        if bsr._nnz() == 0:
            return out.zero_()
        blocksize = bsr.values().shape[-2:]
        if enable_bsr_scatter_mm and max(blocksize) == 16 and (bsr.dense_dim() == 0) and (bsr.ndim == 2):
            dtype = bsr.dtype
            if dtype in {torch.float16, torch.bfloat16} and (m >= 4096 and n >= 8192 or (m == 2048 and n >= 32768) or n >= 131072) or (dtype == torch.float32 and (m >= 1024 or (m == 512 and n >= 512) or (m == 256 and n >= 2048))):
                return bsr_scatter_mm(bsr, dense, out=out)
        if meta is None:
            meta = bsr_dense_mm_meta(m, kl, n, blocksize[0], blocksize[1])
        else:
            meta = bsr_dense_mm_meta(m, kl, n, blocksize[0], blocksize[1], **meta)
        out_backup = out
        (crow_indices, col_indices, values, dense, out) = prepare_inputs(bsr, dense, out)
        dense = tile_to_blocksize(dense, blocksize[::-1])
        out = tile_to_blocksize(out, (blocksize[0], blocksize[0]))
        _run_dense_rowspace_kernel(blocksize, values, crow_indices, col_indices, dense, out, max_grid, meta)
        return out_backup

    @triton.jit
    def _bsr_softmax_kernel(crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, values_ptr, values_batch_stride, values_row_block_stride, values_nnz_col_block_stride, row_block, col_block, MAX_ROW_NNZ: tl.constexpr, TILE: tl.constexpr):
        if False:
            print('Hello World!')
        batch_pid = tl.program_id(axis=2)
        row_block_offset_pid = tl.program_id(axis=1)
        row_block_pid = tl.program_id(axis=0)
        crow_indices_offset_ptr = crow_indices_ptr + crow_indices_batch_stride * batch_pid + crow_indices_stride * row_block_pid
        nnz_offset = tl.load(crow_indices_offset_ptr)
        nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)
        row_nnz = nnz_offset_next - nnz_offset
        if row_nnz == 0:
            return
        row_arange = tl.arange(0, TILE)
        mask = row_arange < row_nnz * col_block
        curr_row_values_ptrs = values_ptr + values_batch_stride * batch_pid + values_row_block_stride * row_block_offset_pid + nnz_offset * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
        max_row_value = tl.max(row_tile, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            curr_max_row_value = tl.max(row_tile, axis=0)
            max_row_value = tl.where(max_row_value > curr_max_row_value, max_row_value, curr_max_row_value)
        num = tl.exp(row_tile - max_row_value)
        denom = tl.sum(num, axis=0)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange -= TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            denom += tl.sum(num, axis=0)
        tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)
        for _ in range(TILE, MAX_ROW_NNZ, TILE):
            row_arange += TILE
            mask = row_arange < row_nnz * col_block
            row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float('inf')).to(tl.float32)
            num = tl.exp(row_tile - max_row_value)
            tl.store(curr_row_values_ptrs + row_arange, (num / denom).to(values_ptr.dtype.element_ty), mask=mask)

    def bsr_softmax(input, max_row_nnz=None):
        if False:
            print('Hello World!')
        f_name = 'bsr_softmax'
        check_bsr_layout(f_name, input)
        check_dtype(f_name, input, input.dtype)
        if input._nnz() == 0 or input.numel() == 0:
            return input.clone()
        (m, n) = input.shape[-2:]
        nnz = input._nnz()
        (row_block, col_block) = input.values().shape[-2:]
        if max_row_nnz is None:
            max_row_nnz = triton.next_power_of_2(n)
        else:
            max_row_nnz = triton.next_power_of_2(max_row_nnz)
        crow_indices = input.crow_indices().unsqueeze(0).flatten(0, -2)
        if input.values().transpose(-3, -2).is_contiguous():
            values = input.values().clone()
        else:
            values = input.values()
        values = values.transpose(-3, -2).contiguous().unsqueeze(0).flatten(0, -4).reshape(-1, row_block, nnz * col_block)
        full_grid = (values.shape[0], row_block, m // row_block)
        grid_blocks = None
        tensor_dims_map = {crow_indices[..., :-1]: (0, None, -1), values: (0, None, None)}

        def kernel(grid, *sliced_tensors):
            if False:
                print('Hello World!')
            _bsr_softmax_kernel[grid](*ptr_stride_extractor(*sliced_tensors), row_block, col_block, max_row_nnz, min(2 ** 17, max_row_nnz))
        launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)
        values = values.reshape(-1, row_block, nnz, col_block).transpose(-3, -2).reshape(*input.values().shape)
        return torch.sparse_compressed_tensor(input.crow_indices().clone(), input.col_indices().clone(), values, size=input.shape, layout=input.layout)

    def _scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor], dropout_p: float=0.0, is_causal: bool=False, scale: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        f_name = '_scaled_dot_product_attention'
        check(not is_causal, f'{f_name}(): is_causal == True is not supported.')
        check(attn_mask is not None, f'{f_name}(): attn_mask == None is not supported.')
        assert attn_mask is not None
        check(attn_mask.layout == torch.sparse_bsr, f'{f_name}(): attn_mask.layout must be {torch.sparse_bsr}, but got attn_mask.layout == {attn_mask.layout}.')
        check_device(f_name, key, query.device)
        check_device(f_name, value, query.device)
        check_device(f_name, attn_mask, query.device)
        check_dtype(f_name, key, query.dtype)
        check_dtype(f_name, value, query.dtype)
        if attn_mask.dtype is not torch.bool:
            check_dtype(f_name, attn_mask, query.dtype)
        sdpa = sampled_addmm(attn_mask, query, key.transpose(-2, -1), beta=0.0, skip_checks=False)
        if scale is None and query.size(-1) == 0 or scale == 0.0:
            check(False, f'{f_name}(): current value of scale == {scale} results in division by zero.')
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        sdpa.values().mul_(scale_factor)
        sdpa = bsr_softmax(sdpa)
        torch.nn.functional.dropout(sdpa.values(), p=dropout_p, inplace=True)
        sdpa = bsr_dense_mm(sdpa, value)
        return sdpa

    @triton.jit
    def _scatter_mm2_kernel(M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K, others_ptr, others_stride_Q, others_stride_K, others_stride_N, accumulators_ptr, accumulators_stride_R, accumulators_stride_M, accumulators_stride_N, pq_offsets_ptr, pq_offsets_stride, pq_ptr, pq_stride_T, pq_stride_1, dot_out_dtype: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, allow_tf32: tl.constexpr):
        if False:
            return 10
        Ms = M // TILE_M
        Ns = N // TILE_N
        pid_t = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        pid_m = pid // Ms
        pid_n = pid % Ms
        rm = pid_m * TILE_M + tl.arange(0, TILE_M)
        rn = pid_n * TILE_N + tl.arange(0, TILE_N)
        rk = tl.arange(0, K)
        A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
        B_ptr = others_ptr + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
        g0 = tl.load(pq_offsets_ptr + pid_t * pq_offsets_stride)
        g1 = tl.load(pq_offsets_ptr + (pid_t + 1) * pq_offsets_stride)
        if g0 == g1:
            return
        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)
        for i in range(g0, g1):
            p = tl.load(pq_ptr + i * pq_stride_T)
            q = tl.load(pq_ptr + i * pq_stride_T + pq_stride_1)
            A = tl.load(A_ptr + p * blocks_stride_P)
            B = tl.load(B_ptr + q * others_stride_Q)
            acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        C_ptr = accumulators_ptr + pid_t * accumulators_stride_R + (rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm2(blocks: torch.Tensor, others: torch.Tensor, pq_offsets: torch.Tensor, pq_indices: torch.Tensor, accumulators: torch.Tensor):
        if False:
            i = 10
            return i + 15
        (P, M, K) = blocks.shape
        (Q, _, N) = others.shape
        (R, _, _) = accumulators.shape
        meta = dict(TILE_M=max(16, M // 4), TILE_N=max(16, N // 4), num_stages=1, num_warps=2)

        def grid(META):
            if False:
                while True:
                    i = 10
            return (pq_offsets.shape[0] - 1, triton.cdiv(M, META['TILE_M']) * triton.cdiv(N, META['TILE_N']), 1)
        dot_out_dtype = {torch.float16: tl.float32, torch.bfloat16: tl.float32, torch.float32: tl.float64, torch.float64: tl.float64}[accumulators.dtype]
        if 'allow_tf32' not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)
        _scatter_mm2_kernel[grid](M, K, N, blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2), others, others.stride(0), others.stride(1), others.stride(2), accumulators, accumulators.stride(0), accumulators.stride(1), accumulators.stride(2), pq_offsets, pq_offsets.stride(0), pq_indices, pq_indices.stride(0), pq_indices.stride(1), dot_out_dtype=dot_out_dtype, **meta)

    @triton.jit
    def _scatter_mm6_kernel(nbatches, Ms, Ks: tl.constexpr, N, blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K, others_ptr, others_stride_B, others_stride_K, others_stride_N, accumulators_ptr, accumulators_stride_B, accumulators_stride_M, accumulators_stride_N, c_indices_ptr, r_offsets_ptr, p_offsets_ptr, q_offsets_ptr, is_compressed: tl.constexpr, dot_out_dtype: tl.constexpr, SPLIT_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, GROUP_SIZE: tl.constexpr, allow_tf32: tl.constexpr):
        if False:
            print('Hello World!')
        Ns = N // SPLIT_N
        BLOCKS_M = Ms // TILE_M
        BLOCKS_N = Ns // TILE_N
        pid_t_ = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        pid_b = pid_t_ % nbatches
        pid_t = pid_t_ // nbatches
        num_pid_in_group = GROUP_SIZE * BLOCKS_N
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE
        group_size_m = min(BLOCKS_M - first_pid_m, GROUP_SIZE)
        pid_m = first_pid_m + pid % group_size_m
        pid_n = pid % num_pid_in_group // group_size_m
        rm = pid_m * TILE_M + tl.arange(0, TILE_M)
        rn = pid_n * TILE_N + tl.arange(0, TILE_N)
        rk = tl.arange(0, Ks)
        A_ptr = blocks_ptr + (rm[:, None] * blocks_stride_M + rk[None, :] * blocks_stride_K)
        B_ptr = others_ptr + pid_b * others_stride_B + (rk[:, None] * others_stride_K + rn[None, :] * others_stride_N)
        r = tl.load(r_offsets_ptr + pid_t)
        if is_compressed:
            m = r // N // Ms
            n = r % N // Ns
            r0 = tl.load(c_indices_ptr + m)
            r1 = tl.load(c_indices_ptr + m + 1)
            g0 = n * r1 + (SPLIT_N - n) * r0
            nnz = r1 - r0
        else:
            g0 = tl.load(c_indices_ptr + pid_t)
            g1 = tl.load(c_indices_ptr + pid_t + 1)
            nnz = g1 - g0
        q_ptr = q_offsets_ptr + g0
        acc_block = tl.zeros((TILE_M, TILE_N), dtype=dot_out_dtype)
        if is_compressed:
            A_ptr += r0 * blocks_stride_P
            for _ in range(nnz):
                q = tl.load(q_ptr)
                B = tl.load(B_ptr + q)
                A = tl.load(A_ptr)
                acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
                A_ptr += blocks_stride_P
                q_ptr += 1
        else:
            p_ptr = p_offsets_ptr + g0
            for _ in range(nnz):
                q = tl.load(q_ptr)
                B = tl.load(B_ptr + q)
                p = tl.load(p_ptr)
                A = tl.load(A_ptr + p * blocks_stride_P)
                p_ptr += 1
                q_ptr += 1
                acc_block += tl.dot(A, B, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        C_ptr = accumulators_ptr + r + pid_b * accumulators_stride_B + (rm[:, None] * accumulators_stride_M + rn[None, :] * accumulators_stride_N)
        tl.store(C_ptr, acc_block.to(accumulators_ptr.dtype.element_ty))

    def _scatter_mm6(blocks: torch.Tensor, others: torch.Tensor, c_indices: torch.Tensor, r_offsets: torch.Tensor, p_offsets: torch.Tensor, q_offsets: torch.Tensor, meta: dict, accumulators: torch.Tensor, force_contiguous: bool=True):
        if False:
            for i in range(10):
                print('nop')
        SPLIT_N = meta['SPLIT_N']
        (P, Ms, Ks) = blocks.shape
        (B, K_, N) = others.shape
        (B_, M, N_) = accumulators.shape
        assert N_ == N
        Ns = N // SPLIT_N
        assert B_ == B

        def grid(META):
            if False:
                while True:
                    i = 10
            return (r_offsets.shape[0] * B, triton.cdiv(Ms, META['TILE_M']) * triton.cdiv(Ns, META['TILE_N']))
        dot_out_dtype = {torch.float16: tl.float32, torch.bfloat16: tl.float32, torch.float32: tl.float64, torch.float64: tl.float64}[accumulators.dtype]
        if 'allow_tf32' not in meta:
            meta.update(allow_tf32=dot_out_dtype == tl.float32)
        assert c_indices.stride(0) == 1
        assert r_offsets.stride(0) == 1
        assert p_offsets.stride(0) == 1
        assert q_offsets.stride(0) == 1
        if force_contiguous:
            blocks = blocks.contiguous()
            others = others.contiguous()
            if not accumulators.is_contiguous():
                accumulators_ = accumulators.contiguous()
            else:
                accumulators_ = accumulators
        else:
            accumulators_ = accumulators
        _scatter_mm6_kernel[grid](B, Ms, Ks, N, blocks, blocks.stride(0), blocks.stride(1), blocks.stride(2), others, others.stride(0), others.stride(1), others.stride(2), accumulators_, accumulators_.stride(0), accumulators_.stride(1), accumulators_.stride(2), c_indices, r_offsets, p_offsets, q_offsets, dot_out_dtype=dot_out_dtype, **meta)
        if force_contiguous and (not accumulators.is_contiguous()):
            accumulators.copy_(accumulators_)
else:
    bsr_softmax = None
    bsr_dense_mm = None
    sampled_addmm = None
    _scaled_dot_product_attention = None
    _scatter_mm2 = None
    _scatter_mm6 = None