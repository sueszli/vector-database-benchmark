import logging
from typing import Any, Dict, List
import torch
from torch._inductor.virtualized import V
from .. import config as inductor_config
from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate
from ..lowering import register_lowering
from ..select_algorithm import autotune_select_algorithm, ExternKernelChoice, TritonTemplate
from ..utils import use_aten_gemm_kernels, use_cutlass_template, use_max_autotune, use_triton_template
from .mm_common import addmm_epilogue, int8_mm_configs, mm_args, mm_configs, mm_grid, mm_options
log = logging.getLogger(__name__)
aten = torch.ops.aten
mm_template = TritonTemplate(name='mm', grid=mm_grid, source='\n{{def_kernel("A", "B")}}\n    M = {{size("A", 0)}}\n    N = {{size("B", 1)}}\n    K = {{size("A", 1)}}\n    if M * N == 0:\n        # early exit due to zero-size input(s)\n        return\n    stride_am = {{stride("A", 0)}}\n    stride_ak = {{stride("A", 1)}}\n    stride_bk = {{stride("B", 0)}}\n    stride_bn = {{stride("B", 1)}}\n\n    # based on triton.ops.matmul\n    pid = tl.program_id(0)\n    grid_m = (M + BLOCK_M - 1) // BLOCK_M\n    grid_n = (N + BLOCK_N - 1) // BLOCK_N\n\n    # re-order program ID for better L2 performance\n    width = GROUP_M * grid_n\n    group_id = pid // width\n    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)\n    pid_m = group_id * GROUP_M + (pid % group_size)\n    pid_n = (pid % width) // (group_size)\n\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)\n    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)\n    rk = tl.arange(0, BLOCK_K)\n    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)\n    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)\n\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)\n    for k in range(K, 0, -BLOCK_K):\n        if EVEN_K:\n            a = tl.load(A)\n            b = tl.load(B)\n        else:\n            a = tl.load(A, mask=rk[None, :] < k, other=0.)\n            b = tl.load(B, mask=rk[:, None] < k, other=0.)\n        if B_PROLOGUE_CAST_TYPE is not None:\n            b = b.to(B_PROLOGUE_CAST_TYPE)\n        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)\n        A += BLOCK_K * stride_ak\n        B += BLOCK_K * stride_bk\n\n    # rematerialize rm and rn to save registers\n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    idx_m = rm[:, None]\n    idx_n = rn[None, :]\n    mask = (idx_m < M) & (idx_n < N)\n\n    # inductor generates a suffix\n    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}\n')
aten_mm = ExternKernelChoice(torch.mm, 'at::mm_out')
aten_addmm = ExternKernelChoice(torch.addmm, 'at::addmm_out')
aten__int_mm = ExternKernelChoice(torch._int_mm, 'at::_int_mm')

def _is_int8_mat(mat):
    if False:
        return 10
    return mat.get_dtype() in (torch.int8, torch.uint8)

def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    if False:
        return 10
    '\n    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt\n    kernel under the hood.  There are a few shapes where this is slower,\n    but they are rare.\n    '
    if inp.stride(0) == 0 or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)
aten_bias_addmm = ExternKernelChoice(bias_addmm, None)

@register_lowering(aten.mm)
def tuned_mm(mat1, mat2, *, layout=None):
    if False:
        for i in range(10):
            print('nop')
    (m, n, k, layout, mat1, mat2) = mm_args(mat1, mat2, layout=layout)
    choices = [aten_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if m * n != 0 and use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout))
    if m * n != 0 and use_cutlass_template(layout):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True)
    from torch._inductor.ir import FixedLayout, FlexibleLayout
    if len(choices) == 1 and use_aten_gemm_kernels() and isinstance(layout, FixedLayout):
        layout = FlexibleLayout(device=layout.device, dtype=layout.dtype, size=layout.size)
        choices = [aten_mm.bind((mat1, mat2), layout)]
    return autotune_select_algorithm('mm', choices, [mat1, mat2], layout)

@register_lowering(aten._int_mm)
def tuned_int_mm(mat1, mat2, *, layout=None):
    if False:
        return 10
    (m, n, k, layout, mat1, mat2) = mm_args(mat1, mat2, layout=layout, out_dtype=torch.int32)
    choices = [aten__int_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    if m * n != 0 and use_triton_template(layout, enable_int32=True):
        choices = []
        for config in int8_mm_configs(m, n, k):
            mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout))
    return autotune_select_algorithm('int_mm', choices, [mat1, mat2], layout)

@register_lowering(aten.addmm)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    if False:
        print('Hello World!')
    ordered_kwargs_for_cpp_kernel = ('beta', 'alpha')
    (m, n, k, layout, mat1, mat2, inp_expanded) = mm_args(mat1, mat2, inp, layout=layout)
    if m * n == 0 or not use_max_autotune():
        choices = [aten_addmm.bind((inp, mat1, mat2), layout, ordered_kwargs_for_cpp_kernel, alpha=alpha, beta=beta)] if use_aten_gemm_kernels() else []
        return autotune_select_algorithm('addmm', choices, [inp, mat1, mat2], layout)
    choices = [aten_addmm.bind((inp_expanded, mat1, mat2), layout, ordered_kwargs_for_cpp_kernel, alpha=alpha, beta=beta)] if use_aten_gemm_kernels() else []
    if use_aten_gemm_kernels() and inp_expanded.get_stride()[0] == 0 and (inp_expanded.get_device().type == 'cuda') and inductor_config.triton.autotune_cublasLt:
        choices.insert(0, aten_bias_addmm.bind((inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta))
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(choices, input_nodes=(inp_expanded, mat1, mat2), layout=layout, **mm_options(config, k, layout), prefix_args=1, epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta))
    if use_cutlass_template(layout):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2, inp_expanded], alpha=alpha, beta=beta, input_reorder=[2, 0, 1], fuseable=False)
    return autotune_select_algorithm('addmm', choices, [inp_expanded, mat1, mat2], layout)

def fallback_mixed_mm(mat1, mat2, *, out):
    if False:
        for i in range(10):
            print('nop')
    return torch.mm(mat1, mat2.to(mat1.dtype), out=out)
aten_fallback_mixed_mm = ExternKernelChoice(fallback_mixed_mm, None)

def tuned_mixed_mm(mat1, mat2, mat2_dtype):
    if False:
        while True:
            i = 10
    (m, n, k, layout, mat1, mat2) = mm_args(mat1, mat2, layout=None)
    choices = [aten_fallback_mixed_mm.bind((mat1, mat2), layout)]
    if mat1.layout.dtype != torch.float32 and (not mat2.layout.is_contiguous()):
        return autotune_select_algorithm('mixed_mm', choices, [mat1, mat2], layout)
    if inductor_config.force_mixed_mm:
        choices = []
    b_prologue_cast_type = f'tl.{mat2_dtype}'.replace('torch.', '')
    has_int8_tensor = _is_int8_mat(mat1) or _is_int8_mat(mat2)
    for config in mm_configs(m, n, k, has_int8_tensor=has_int8_tensor):
        mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2), layout=layout, **mm_options(config, k, layout, b_prologue_cast_type))
    return autotune_select_algorithm('mixed_mm', choices, [mat1, mat2], layout)

def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    if False:
        while True:
            i = 10
    out_dtype = torch.promote_types(mat3.get_dtype(), torch.int32) if out_dtype is None else out_dtype
    (m, n, k, layout, mat1, mat2, mat3) = mm_args(mat1, mat2, mat3, layout=layout, out_dtype=out_dtype)
    choices: List[Dict[Any, Any]] = []
    for config in int8_mm_configs(m, n, k):
        mm_template.maybe_append_choice(choices, input_nodes=(mat1, mat2, mat3), layout=layout, **dict(mm_options(config, k, layout), **{'ACC_TYPE': 'tl.int32'}), suffix_args=1, epilogue_fn=V.ops.mul)
    return autotune_select_algorithm('int_mm', choices, [mat1, mat2, mat3], layout)