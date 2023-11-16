import torch

def _sparse_semi_structured_from_dense_cutlass(dense):
    if False:
        return 10
    if dense.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor')
    (m, k) = dense.shape
    device = dense.device
    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f'Invalid datatype {dense.dtype} of dense matrix')
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError('Invalid number of elements per meta element calculated')
    if m % 32 != 0:
        raise RuntimeError(f'Number rows columns of dense matrix {m} must be divisible by 32')
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(f'Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}')
    meta_ncols = k // (4 * quadbits_per_meta_elem)
    dense_4 = dense.view(-1, k // 4, 4)
    (m0, m1, m2, m3) = (dense_4 != 0).unbind(-1)
    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | bit1.to(torch.int64) << 1
    idxs1 = bit2 | bit3.to(torch.int64) << 1
    sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
    sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
    sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    meta_4 = idxs0 | idxs1 << 2
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)
    if quadbits_per_meta_elem == 4:
        meta = meta_n[:, :, 0] | meta_n[:, :, 1] << 4 | meta_n[:, :, 2] << 8 | meta_n[:, :, 3] << 12
    elif quadbits_per_meta_elem == 8:
        meta = meta_n[:, :, 0] | meta_n[:, :, 1] << 4 | meta_n[:, :, 2] << 8 | meta_n[:, :, 3] << 12 | meta_n[:, :, 4] << 16 | meta_n[:, :, 5] << 20 | meta_n[:, :, 6] << 24 | meta_n[:, :, 7] << 28
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = 32
        magic2 = 16
        magic3 = k // 2
        magic4 = [0, k // 4, 1, k // 4 + 1]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = 64
        magic2 = 32
        magic3 = 2 * k
        magic4 = [0, k // 2, 1, k // 2 + 1, k, 3 * k // 2, k + 1, 3 * k // 2 + 1]
    tmp0 = torch.zeros(m * meta_ncols, dtype=torch.int64, device=device)
    tmp1 = (tmp0.view(meta_ncols // 2, -1) + torch.arange(0, meta_ncols, 2, device=device).view(meta_ncols // 2, 1)).view(-1, magic1)
    tmp2 = (torch.arange(0, 8, device=device).view(-1, 1) * torch.ones((magic0,), dtype=torch.int64, device=device) * meta_ncols).view(-1).repeat(m * meta_ncols // magic1).view(-1, magic1)
    tmp3 = (torch.arange(0, m // magic2, device=device).view(-1, 1) * magic3).repeat(meta_ncols // 2, magic1)
    tmp4 = torch.tensor(magic4, device=device).repeat(tmp3.shape[0], 8)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4
    meta_reordered = torch.gather(meta.view(-1), 0, meta_offsets.view(-1)).view(m, meta_ncols)
    return (sparse, meta_reordered)

def _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    if False:
        print('Hello World!')
    if sparse.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor')
    (m, k) = sparse.shape
    device = sparse.device
    if meta_reordered.dim() != 2:
        raise RuntimeError(f'Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor')
    if meta_reordered.device != device:
        raise RuntimeError(f'Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device')
    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f'Invalid datatype {meta_dtype} of meta matrix')
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    (meta_nrows, meta_ncols) = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(f'Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}')
    if meta_ncols * 4 * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(f'Number of columns of sparse matrix {k} different from the {meta_ncols * 4 * quadbits_per_meta_elem // 2}, expected according to the number of columns of meta matrix')
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = [0, 1, 32, 33]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = [0, 1, 4, 5]
    tmp1 = torch.tensor([0, 2], dtype=torch.int64, device=device).repeat(meta_nrows, meta_ncols // 2)
    tmp2 = (torch.arange(0, meta_ncols // 2, device=device) * 2 * meta_nrows).view(-1, 1).repeat(1, 2).view(-1).repeat(m, 1)
    tmp3 = (torch.arange(0, 8, device=device) * magic0).view(-1, 1).repeat(m // 8, meta_ncols)
    tmp4 = torch.tensor(magic1, device=device).view(-1, 1).repeat(1, 8 * meta_ncols).repeat(meta_nrows // 32, 1).view(meta_nrows, meta_ncols)
    tmp5 = (torch.arange(0, meta_nrows // 32, device=device) * 64).view(-1, 1).repeat(1, 32 * meta_ncols).view(meta_nrows, meta_ncols)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4 + tmp5
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets.view(-1)).view(m, meta_ncols)
    meta_2 = torch.empty((m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device)
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 3
        meta_2[:, :, 1] = meta >> 2 & 3
        meta_2[:, :, 2] = meta >> 4 & 3
        meta_2[:, :, 3] = meta >> 6 & 3
        meta_2[:, :, 4] = meta >> 8 & 3
        meta_2[:, :, 5] = meta >> 10 & 3
        meta_2[:, :, 6] = meta >> 12 & 3
        meta_2[:, :, 7] = meta >> 14 & 3
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 3
        meta_2[:, :, 1] = meta >> 2 & 3
        meta_2[:, :, 2] = meta >> 4 & 3
        meta_2[:, :, 3] = meta >> 6 & 3
        meta_2[:, :, 4] = meta >> 8 & 3
        meta_2[:, :, 5] = meta >> 10 & 3
        meta_2[:, :, 6] = meta >> 12 & 3
        meta_2[:, :, 7] = meta >> 14 & 3
        meta_2[:, :, 8] = meta >> 16 & 3
        meta_2[:, :, 9] = meta >> 18 & 3
        meta_2[:, :, 10] = meta >> 20 & 3
        meta_2[:, :, 11] = meta >> 22 & 3
        meta_2[:, :, 12] = meta >> 24 & 3
        meta_2[:, :, 13] = meta >> 26 & 3
        meta_2[:, :, 14] = meta >> 28 & 3
        meta_2[:, :, 15] = meta >> 30 & 3
    dense_offsets = meta_2.view(-1) + (torch.arange(0, m * k // 2, device=device) * 4).view(-1, 1).repeat(1, 2).view(-1)
    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    dense.scatter_(0, dense_offsets, sparse.view(-1))
    return dense.view(m, 2 * k)

def sparse_semi_structured_from_dense_cutlass(dense, compile=False):
    if False:
        i = 10
        return i + 15
    if compile:
        from torch._dynamo.utils import is_compile_supported
        if is_compile_supported(dense.device.type):
            kernel = torch.compile(_sparse_semi_structured_from_dense_cutlass)
            return kernel(dense)
    return _sparse_semi_structured_from_dense_cutlass(dense)

def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered, compile=False):
    if False:
        i = 10
        return i + 15
    if compile:
        from torch._dynamo.utils import is_compile_supported
        if is_compile_supported(sparse.device.type):
            kernel = torch.compile(_sparse_semi_structured_to_dense_cutlass)
            return kernel(sparse, meta_reordered)
    return _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered)