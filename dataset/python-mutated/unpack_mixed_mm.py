import logging
from typing import List
from ..select_algorithm import autotune_select_algorithm, ChoiceCaller, TritonTemplate
from .mm_common import mm_args, mm_configs, mm_grid, mm_options
log = logging.getLogger(__name__)
uint4x2_mixed_mm_template = TritonTemplate(name='uint4x2_mixed_mm', grid=mm_grid, source='\n{{def_kernel("A", "B")}}\n    M = {{size("A", 0)}}\n    N = {{size("B", 1)}}\n    K = {{size("A", 1)}}\n    stride_am = {{stride("A", 0)}}\n    stride_ak = {{stride("A", 1)}}\n    stride_bk = {{stride("B", 0)}}\n    stride_bn = {{stride("B", 1)}}\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    rk = tl.arange(0, BLOCK_K)\n    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)\n    B = B + (rk[:, None]//2 * stride_bk + rbn[None, :] * stride_bn)\n    b_shifts = 4*(rk%2)\n    b_subs = 8*(1-(rk%2))\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    for k in range(K, 0, -BLOCK_K):\n        if EVEN_K:\n            a = tl.load(A)\n            b = tl.load(B)\n        else:\n            a = tl.load(A, mask=rk[None, :] < k, other=0.)\n            b = tl.load(B, mask=rk[:, None] < k, other=0.)\n        b = ((b >> b_shifts[:, None]) & 0xF) - 8\n        b = b.to(B_PROLOGUE_CAST_TYPE)\n        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)\n        A += BLOCK_K * stride_ak\n        B += BLOCK_K//2 * stride_bk\n\n    # rematerialize rm and rn to save registers\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}\n')

def tuned_uint4x2_mixed_mm(mat1, mat2, mat2_mm_shape, mat2_dtype):
    if False:
        for i in range(10):
            print('nop')
    (m, n, k, layout, mat1, mat2) = mm_args(mat1, mat2, layout=None, use_4x2_dim=True)
    choices: List[ChoiceCaller] = []
    b_prologue_cast_type = f'tl.{mat2_dtype}'.replace('torch.', '')
    for config in mm_configs(m, n, k):
        uint4x2_mixed_mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout, b_prologue_cast_type))
    return autotune_select_algorithm('uint4x2_mixed_mm', choices, [mat1, mat2], layout)