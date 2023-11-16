import functools
import torch
from ..lowering import lowerings
from ..select_algorithm import autotune_select_algorithm, ExternKernelChoice, TritonTemplate
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid, mm_options
aten = torch.ops.aten
aten_mm_plus_mm = ExternKernelChoice(torch.ops.inductor._mm_plus_mm, 'torch::inductor::_mm_plus_mm')
mm_plus_mm_template = TritonTemplate(name='mm_plus_mm', grid=mm_grid, debug=False, source='\n{{def_kernel("A", "B", "C", "D")}}\n    M = {{size("A", 0)}}\n    N = {{size("B", 1)}}\n    K1 = {{size("A", 1)}}\n    if M * N == 0:\n        # early exit due to zero-size input(s)\n        return\n    # K2 = {{size("C", 1)}}\n    stride_am = {{stride("A", 0)}}\n    stride_ak = {{stride("A", 1)}}\n    stride_bk = {{stride("B", 0)}}\n    stride_bn = {{stride("B", 1)}}\n    stride_cm = {{stride("C", 0)}}\n    stride_ck = {{stride("C", 1)}}\n    stride_dk = {{stride("D", 0)}}\n    stride_dn = {{stride("D", 1)}}\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    rk = tl.arange(0, BLOCK_K)\n    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)\n    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)\n    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)\n    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    for k1 in range(K1, 0, -BLOCK_K):\n        # First matmul with A @ B\n        if EVEN_K:\n            a = tl.load(A)\n            b = tl.load(B)\n        else:\n            a = tl.load(A, mask=rk[None, :] < k1, other=0.)\n            b = tl.load(B, mask=rk[:, None] < k1, other=0.)\n        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)\n        A += BLOCK_K * stride_ak\n        B += BLOCK_K * stride_bk\n\n    for k2 in range(K1, 0, -BLOCK_K):\n\n        # Second matmul with C @ D\n        if EVEN_K:\n            c = tl.load(C)\n            d = tl.load(D)\n        else:\n            c = tl.load(C, mask=rk[None, :] < k2, other=0.)\n            d = tl.load(D, mask=rk[:, None] < k2, other=0.)\n        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)\n        C += BLOCK_K * stride_ck\n        D += BLOCK_K * stride_dk\n\n\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}\n')

@functools.lru_cache(None)
def mm_configs():
    if False:
        return 10
    import triton
    mm_triton_configs = [{'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 2, 'num_warps': 4, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 3, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 4, 'num_warps': 16, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, 'num_stages': 4, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 4, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, 'num_stages': 1, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, 'num_stages': 1, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, 'num_stages': 1, 'num_warps': 8, 'cond': torch.version.hip is None}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, 'num_stages': 2, 'num_warps': 4, 'cond': True}, {'config': {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, 'num_stages': 1, 'num_warps': 2, 'cond': True}]
    if torch.version.hip:
        filtered_configs = [triton.Config(c['config'], num_stages=1, num_warps=c['num_warps']) for c in mm_triton_configs if c['cond']]
    else:
        filtered_configs = [triton.Config(c['config'], num_stages=c['num_stages'], num_warps=c['num_warps']) for c in mm_triton_configs if c['cond']]
    return filtered_configs

def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    if False:
        print('Hello World!')
    '\n    Computes mm(mat1, mat2) + mm(mat3, mat4)\n    '
    (m1, n1, k1, layout1, mat1, mat2) = mm_args(mat1, mat2, layout=layout)
    (m2, n2, _, layout2, mat3, mat4) = mm_args(mat3, mat4, layout=layout)
    if m1 * n1 == 0 or m2 * n2 == 0 or (not V.graph.sizevars.statically_known_list_equals(mat1.get_size(), mat3.get_size())) or (not V.graph.sizevars.statically_known_list_equals(mat2.get_size(), mat4.get_size())):
        if m1 == m2 and n1 == n2:
            V.graph.sizevars.guard_equals(m1, m2)
            V.graph.sizevars.guard_equals(n1, n2)
            return lowerings[aten.addmm](lowerings[aten.mm](mat3, mat4), mat1, mat2)
        return lowerings[aten.add](lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4))
    assert layout1 == layout2
    choices = [aten_mm_plus_mm.bind((mat1, mat2, mat3, mat4), layout1)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout1):
        for config in mm_configs():
            if config.kwargs['BLOCK_K'] < k1:
                mm_plus_mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2, mat3, mat4), layout=layout1, **mm_options(config, k1, layout1))
    return autotune_select_algorithm('mm_plus_mm', choices, [mat1, mat2, mat3, mat4], layout1)