import torch
from ..lowering import register_lowering
from ..select_algorithm import autotune_select_algorithm, ExternKernelChoice, TritonTemplate
from ..utils import ceildiv as cdiv, use_aten_gemm_kernels, use_triton_template
from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options
aten = torch.ops.aten

def bmm_grid(b, m, n, meta):
    if False:
        for i in range(10):
            print('nop')
    return (cdiv(m, meta['BLOCK_M']) * cdiv(n, meta['BLOCK_N']), b, 1)
bmm_template = TritonTemplate(name='bmm', grid=bmm_grid, source='\n{{def_kernel("A", "B")}}\n    M = {{size("A", -2)}}\n    N = {{size("B", -1)}}\n    K = {{size("A", -1)}}\n\n    stride_aq = {{stride("A", 0)}}\n    stride_am = {{stride("A", 1)}}\n    stride_ak = {{stride("A", 2)}}\n\n    stride_bq = {{stride("B", 0)}}\n    stride_bk = {{stride("B", 1)}}\n    stride_bn = {{stride("B", 2)}}\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    rk = tl.arange(0, BLOCK_K)\n\n    idx_q = tl.program_id(1)  # batch dimension for BMM\n    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)\n    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    for k in range(K, 0, -BLOCK_K):\n        if EVEN_K:\n            a = tl.load(A)\n            b = tl.load(B)\n        else:\n            a = tl.load(A, mask=rk[None, :] < k, other=0.)\n            b = tl.load(B, mask=rk[:, None] < k, other=0.)\n        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)\n        A += BLOCK_K * stride_ak\n        B += BLOCK_K * stride_bk\n\n    # rematerialize rm and rn to save registers\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    idx_q = tl.program_id(1)  # batch dimension for BMM\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask")}}\n')
aten_bmm = ExternKernelChoice(torch.bmm, 'at::bmm_out')
aten_baddbmm = ExternKernelChoice(torch.baddbmm, 'at::baddbmm_out')

@register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, *, layout=None):
    if False:
        print('Hello World!')
    (m, n, k, layout, mat1, mat2) = mm_args(mat1, mat2, layout=layout)
    choices = [aten_bmm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout))
    return autotune_select_algorithm('bmm', choices, [mat1, mat2], layout)

def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    if False:
        for i in range(10):
            print('nop')
    (m, n, k, layout, mat1, mat2, inp) = mm_args(mat1, mat2, inp, layout=layout)
    choices = [aten_baddbmm.bind((inp, mat1, mat2), layout, alpha=alpha, beta=beta)] if use_aten_gemm_kernels() else []
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(choices, input_nodes=(inp, mat1, mat2), layout=layout, **mm_options(config, k, layout), prefix_args=1, epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta))
    return autotune_select_algorithm('baddbmm', choices, [inp, mat1, mat2], layout)