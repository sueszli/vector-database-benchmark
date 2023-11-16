import argparse
import collections
import itertools
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar
DEFAULT_ARCH = [50, 70, 75, 80]
MAX_ARCH = 90
ENABLE_MACRO = 'PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION'
assert sorted(DEFAULT_ARCH) == DEFAULT_ARCH

def find_arch_range(min_arch, max_arch):
    if False:
        print('Hello World!')
    assert min_arch >= DEFAULT_ARCH[0] and min_arch <= MAX_ARCH
    assert max_arch >= DEFAULT_ARCH[0] and max_arch <= MAX_ARCH
    assert min_arch <= max_arch
    n = len(DEFAULT_ARCH)
    start_idx = n - 1
    for i in range(n - 1):
        if DEFAULT_ARCH[i] <= min_arch and min_arch < DEFAULT_ARCH[i + 1]:
            start_idx = i
            break
    end_idx = n
    for i in range(n - 1):
        if DEFAULT_ARCH[i] <= max_arch and max_arch < DEFAULT_ARCH[i + 1]:
            end_idx = i + 1
    return DEFAULT_ARCH[start_idx:end_idx]

def find_max_arch(arch):
    if False:
        for i in range(10):
            print('nop')
    arch = sorted(arch)
    idx = DEFAULT_ARCH.index(arch[-1])
    if idx == len(DEFAULT_ARCH) - 1:
        return MAX_ARCH
    else:
        return DEFAULT_ARCH[idx + 1]

def convert_to_arch_list(arch):
    if False:
        for i in range(10):
            print('nop')
    arch = arch.lower().strip()
    if arch == 'all':
        return DEFAULT_ARCH
    arch = [int(s.strip()) for s in arch.split(';') if s.strip()]
    arch = list(set(arch))
    arch.sort()
    return find_arch_range(arch[0], arch[-1])

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='The argument for generating the memory efficient kernels.')
    parser.add_argument('--dst_path', type=str, default=str(Path(__file__).parent), help='The destination path to save the generated files.')
    parser.add_argument('--cuda_arch', type=convert_to_arch_list, default=convert_to_arch_list('All'), help='The CUDA architecture to be generated.')
    args = parser.parse_args()
    args.max_arch = find_max_arch(args.cuda_arch)
    return args
args = parse_args()
DTYPES = {'f32': 'float', 'f16': 'cutlass::half_t', 'bf16': 'cutlass::bfloat16_t'}
SM = args.cuda_arch
KERNEL_IMPL_TEMPLATE = '\n\nvoid  {NAME}({CPP_CLASS} default_fmha, Params &params, const phi::GPUContext& ctx) {{\n  using AttentionKernel = typename decltype(default_fmha)::FMHAKernel;\n  using FMHA = cutlass::gemm::device::GemmGrouped<AttentionKernel>;\n  using scalar_t = typename FMHA::GemmKernel::scalar_t;\n  using accum_t = typename FMHA::GemmKernel::accum_t;\n  using output_t = typename FMHA::GemmKernel::output_t;\n  using output_accum_t = typename FMHA::GemmKernel::output_accum_t;\n  using ElementQ = scalar_t;\n  using ElementK = scalar_t;\n  using ElementP = accum_t;\n  using ElementM = scalar_t;\n  using ElementAccumulator = accum_t;\n  using ElementV = scalar_t;\n  using ElementO = output_t;\n  using ElementOAccum = output_accum_t;\n\n  int problem_count = params.num_batches * params.num_heads;\n\n  std::vector<GemmCoord> problem_sizes1;\n  problem_sizes1.reserve(problem_count);\n\n  phi::Allocator::AllocationPtr problem_sizes_device0{{nullptr}};\n  phi::Allocator::AllocationPtr problem_sizes_device1{{nullptr}};\n  problem_sizes_device0 = phi::memory_utils::Alloc(\n      ctx.GetPlace(),\n      problem_count * sizeof(GemmCoord),\n      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));\n  problem_sizes_device1 = phi::memory_utils::Alloc(\n      ctx.GetPlace(),\n      problem_count * sizeof(GemmCoord),\n      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));\n  GemmCoord* problem0_device =\n      reinterpret_cast<GemmCoord*>(problem_sizes_device0->ptr());\n  GemmCoord* problem1_device =\n      reinterpret_cast<GemmCoord*>(problem_sizes_device1->ptr());\n  get_problem_sizes<<<params.num_batches, params.num_heads, 0, ctx.stream()>>>(\n      params.seq_lens,\n      params.kv_seq_lens,\n      problem0_device,\n      problem1_device,\n      params.num_batches,\n      params.num_heads,\n      params.head_size,\n      params.value_head_size);\n  phi::memory_utils::Copy(phi::CPUPlace(),\n                       problem_sizes1.data(),\n                       ctx.GetPlace(),\n                       problem1_device,\n                       sizeof(GemmCoord) * problem_count,\n                       ctx.stream());\n  if (AttentionKernel::kNeedsOutputAccumulatorBuffer) {{\n    const int64_t output_size = params.num_batches * params.num_heads *\n                                params.query_seq_len * params.value_head_size;\n    phi::Allocator::AllocationPtr tmp_output_accum_buffer_ptr{{nullptr}};\n    tmp_output_accum_buffer_ptr = phi::memory_utils::Alloc(\n        ctx.GetPlace(),\n        output_size * sizeof(ElementOAccum),\n        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));\n    params.output_accum_ptr = tmp_output_accum_buffer_ptr->ptr();\n  }}\n  int threadblock_count =\n      FMHA::sufficient(problem_sizes1.data(), problem_count);\n  typename FMHA::Arguments args(\n      problem0_device,\n      problem1_device,\n      problem_count,\n      threadblock_count,\n      params.num_heads,\n      params.kv_num_heads,\n      const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.query_ptr)),\n      const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.key_ptr)),\n      params.mask_ptr\n          ? const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.mask_ptr))\n          : nullptr,\n      const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(params.value_ptr)),\n      reinterpret_cast<scalar_t*>(params.output_ptr),\n      AttentionKernel::kNeedsOutputAccumulatorBuffer\n          ? reinterpret_cast<output_accum_t*>(params.output_accum_ptr)\n          : nullptr,\n      params.ldq,\n      params.ldk,\n      params.ldm,\n      params.ldv,\n      params.ldo,\n      params.ElementQ,\n      params.ElementK,\n      params.ElementM,\n      params.ElementV,\n      params.ElementO,\n      params.causal,\n      params.mask_broadcast_head,\n      params.scale,\n      problem_sizes1.data());\n\n  FMHA fmha;\n  cutlass::Status status;\n  size_t workspace_size = fmha.get_workspace_size(args);\n  phi::DenseTensor workspace;\n  workspace.Resize(phi::make_ddim({{static_cast<int64_t>(workspace_size)}}));\n  ctx.template Alloc<uint8_t>(&workspace);\n  status = fmha.initialize(args, workspace.data<uint8_t>());\n  if (status != cutlass::Status::kSuccess) {{\n    PADDLE_THROW(phi::errors::Unimplemented(\n        "Failed to initialize CUTLASS Grouped FMHA kernel."));\n  }}\n  status = fmha.run(ctx.stream());\n  if (status != cutlass::Status::kSuccess) {{\n    PADDLE_THROW(phi::errors::Unimplemented(\n        "Failed to run CUTLASS Grouped FMHA kernel."));\n  }}\n}}\n'

@dataclass(order=True)
class FwdKernel:
    sort_index: Tuple[int, ...] = field(init=False, repr=False)
    aligned: bool
    mask_aligned: bool
    dtype: str
    sm_range: Tuple[int, int]
    q: int
    k: int
    single_value_iter: bool
    support_mask: bool = True
    dispatch_cond: Optional[str] = None

    def __post_init__(self) -> None:
        if False:
            print('Hello World!')
        self.sort_index = (0 if self.aligned else 1, 0 if self.support_mask else 1, 0 if self.single_value_iter else 1, self.q, 0 if self.mask_aligned else 1)

    @property
    def _aligned_suffix(self) -> str:
        if False:
            print('Hello World!')
        return 'aligned' if self.aligned else 'notaligned'

    @property
    def _mask_aligned_suffix(self) -> str:
        if False:
            return 10
        return 'ma' if self.mask_aligned else 'mua'

    @property
    def _mask_support_suffix(self) -> str:
        if False:
            return 10
        return 'sm' if self.support_mask else 'usm'

    @property
    def _single_value_suffix(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'rf' if self.single_value_iter else 'urf'

    @property
    def name(self) -> str:
        if False:
            print('Hello World!')
        return f'fmha_cutlassF_variable_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_{self._single_value_suffix}_{self._mask_support_suffix}_{self._mask_aligned_suffix}_sm{self.sm_range[0]}'

    @property
    def cpp_class(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        template_args = ', '.join([DTYPES[self.dtype], f'cutlass::arch::Sm{self.sm_range[0]}', 'true' if self.aligned else 'false', 'true' if self.mask_aligned else 'false', str(self.q), str(self.k), 'true' if self.single_value_iter else 'false', 'cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly', 'true' if self.support_mask else 'false'])
        return f'cutlass::gemm::kernel::DefaultFMHAGrouped<{template_args}>'

    @property
    def impl_group(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'{self.dtype}_{self._aligned_suffix}_{self._mask_support_suffix}_{self._mask_aligned_suffix}_{self._single_value_suffix}_{self.q}x{self.k}'

    @property
    def cpp_impl(self) -> str:
        if False:
            i = 10
            return i + 15
        return KERNEL_IMPL_TEMPLATE.format(CPP_CLASS=self.cpp_class, NAME=self.name)

    @classmethod
    def get_all(cls) -> List['FwdKernel']:
        if False:
            while True:
                i = 10
        kernels: List[FwdKernel] = []
        for (aligned, dtype, (sm, sm_max)) in itertools.product([True, False], DTYPES.keys(), zip(SM, SM[1:] + [args.max_arch])):
            if dtype == 'bf16' and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            for (q, k, single_value_iter) in [(32, 128, True), (32, 128, False), (64, 64, True)]:
                for (support_mask, mask_aligned) in [(False, False), (True, False), (True, True)]:
                    kernels.append(cls(aligned=aligned, dtype=dtype, sm_range=(sm, sm_max), q=q, k=k, single_value_iter=single_value_iter, support_mask=support_mask, mask_aligned=mask_aligned))
        return kernels
T = TypeVar('T', bound=FwdKernel)

def write_decl_impl(kernels: List[T], family_name: str, impl_file: str, enable_def: str) -> None:
    if False:
        i = 10
        return i + 15
    cpp_file_header = '// This file is auto-generated. See "generate_variable_forward_kernels.py"\n'
    kernels.sort()
    implfile_to_kernels: Dict[str, List[T]] = collections.defaultdict(list)
    cat_to_kernels: Dict[Tuple[str, int, int], List[T]] = collections.defaultdict(list)
    dispatch_all = ''
    declarations = cpp_file_header + '#pragma once\n'
    declarations += f'#ifdef {enable_def}\n'
    declarations += f'#include "{impl_file}"\n'
    declarations += 'namespace phi {\n'
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[k.dtype, k.sm_range[0], k.sm_range[1]].append(k)
    for ((cat_dt, cat_sm, cat_sm_max), kernels) in cat_to_kernels.items():
        declarations += f'// ======== {cat_dt} / sm{cat_sm} ========\n'
        declarations += '\n'.join((k.cpp_impl.split('{')[0].rstrip() + ';' for k in kernels))
        dispatch_category_fn = f'dispatch_{family_name}_{cat_dt}_sm{cat_sm}'
        declarations += f'\n\ntemplate <typename T> void {dispatch_category_fn}(T cb) {{\n'
        for k in kernels:
            _call = f'cb({k.cpp_class}(), {k.name});\n'
            if k.dispatch_cond is not None:
                _call = f'if ({k.dispatch_cond}) {_call}'
            declarations += f'    {_call}'
        declarations += '}\n\n'
        dispatch_all += f'\n    if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{\n        {dispatch_category_fn}(cb);\n    }}'
    declarations += f"""\ntemplate <typename PaddleT, typename T>\nvoid dispatch_{family_name}(const ::phi::GPUContext &ctx, T cb) {{\n    auto cc = ctx.GetComputeCapability();\n    PADDLE_ENFORCE_GE(\n        cc,\n        70,\n        phi::errors::InvalidArgument("the Nvidia GPU's Compute Capability must be greater or equal than 70"));\n\n    using DT = typename ::phi::CutlassTrait<PaddleT>::Type;\n{dispatch_all}\n}}\n"""
    declarations += '} // namespace phi\n'
    declarations += f'#endif // {enable_def}\n'
    autogen_dir = Path(args.dst_path) / 'autogen_variable'
    os.makedirs(autogen_dir, exist_ok=True)
    declaration_path = autogen_dir / f'{family_name}.h'
    declaration_path.write_text(declarations)
    for (f, f_kernels) in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        impl_cu += f'#ifdef {enable_def}\n'
        impl_cu += f'#include "{impl_file}"\n'
        impl_cu += 'namespace phi {\n'
        for k in f_kernels:
            impl_cu += k.cpp_impl
        impl_cu += '} // namespace phi\n'
        impl_cu += f'#endif // {enable_def}\n'
        impl_path = autogen_dir / 'impl'
        os.makedirs(impl_path, exist_ok=True)
        (impl_path / f'{family_name}_{f}.cu').write_text(impl_cu)

def write_main_header():
    if False:
        i = 10
        return i + 15
    main_header_content = f'\n#pragma once\n\n#ifdef {ENABLE_MACRO}\n\n#include "paddle/phi/common/data_type.h"\n#include "paddle/phi/core/dense_tensor.h"\n#include "paddle/phi/backends/gpu/gpu_context.h"\n#include "paddle/phi/common/memory_utils.h"\n#include "paddle/phi/common/place.h"\n#include "paddle/phi/core/dense_tensor.h"\n#include "paddle/phi/core/kernel_registry.h"\n\n#include "cutlass/util/device_memory.h"\n#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/default_fmha_grouped.h"\n#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/gemm/gemm_grouped.h"\n\nnamespace phi {{\n\nusing GemmCoord = cutlass::gemm::GemmCoord;\n\nstruct Params {{\n  // meta params\n  phi::DataType datatype;\n\n  // [bs, nh, seq_len, dh]\n  const void* query_ptr;\n  const void* key_ptr;\n  const void* value_ptr;\n\n  // and it can be broadcasted in axis0, 1, 2.\n  const void* mask_ptr = nullptr;\n\n  const int* seq_lens = nullptr;\n  const int* kv_seq_lens = nullptr;\n\n  // Output tensors\n  void* output_ptr;  // [num_batches, num_heads, query_seq_len, head_size]\n  void* output_accum_ptr =\n      nullptr;  // [num_batches, num_heads, query_seq_len, head_size]\n\n  // Scale\n  float scale;\n\n  // Dimensions/strides\n  int32_t num_batches;\n  int32_t num_heads;\n  int32_t kv_num_heads;\n  int32_t query_seq_len;\n  int32_t key_value_seq_len;\n  int32_t head_size;\n  int32_t value_head_size;\n\n  int64_t ldq;\n  int64_t ldk;\n  int64_t ldm;\n  int64_t ldv;\n  int64_t ldo;\n\n  int64_t ElementQ;\n  int64_t ElementK;\n  int64_t ElementM;\n  int64_t ElementV;\n  int64_t ElementO;\n\n  bool causal;\n  bool mask_broadcast_head;\n}};\n\n__global__ static void get_problem_sizes(const int* seq_lens,\n                                         const int* kv_seq_lens,\n                                         GemmCoord* problem_sizes0,\n                                         GemmCoord* problem_sizes1,\n                                         const int bs,\n                                         const int num_head,\n                                         const int head_size,\n                                         const int value_head_size) {{\n  int bi = blockIdx.x;\n  int hi = threadIdx.x;\n  if (bi < bs && hi < num_head) {{\n    int id = bi * num_head + hi;\n    int m = seq_lens[bi];\n    int mkv = kv_seq_lens[bi];\n    int k0 = head_size;\n    int k1 = value_head_size;\n    GemmCoord problem0(m, mkv, k0);\n    GemmCoord problem1(m, k1, mkv);\n    problem_sizes0[id] = problem0;\n    problem_sizes1[id] = problem1;\n  }}\n}}\n\ntemplate <typename T>\nstruct CutlassTrait {{\n  using Type = T;\n}};\n\ntemplate <>\nstruct CutlassTrait<dtype::float16> {{\n  using Type = cutlass::half_t;\n}};\n\ntemplate <>\nstruct CutlassTrait<dtype::bfloat16> {{\n  using Type = cutlass::bfloat16_t;\n}};\n\n\ntemplate <typename T>\nstruct ToPhiDTypeTrait {{\n private:\n  using NonConstT = typename std::remove_const<T>::type;\n  static constexpr bool kIsFP16 = std::is_same<NonConstT, cutlass::half_t>::value;\n  static constexpr bool kIsBF16 = std::is_same<NonConstT, cutlass::bfloat16_t>::value;\n\n public:\n  using Type = typename std::conditional<kIsFP16, dtype::float16,\n      typename std::conditional<kIsBF16, dtype::bfloat16, NonConstT>::type>::type;\n}};\n\n}} // namespace phi\n\n#include "./cutlass_forward.h"\n\n#endif\n'
    path = Path(args.dst_path) / 'autogen_variable'
    os.makedirs(path, exist_ok=True)
    path = Path(path) / 'memory_efficient_variable_attention.h'
    path.write_text(main_header_content)
if os.path.exists(Path(args.dst_path) / 'autogen_variable'):
    shutil.rmtree(Path(args.dst_path) / 'autogen_variable')
forward_impl = 'paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/memory_efficient_variable_attention.h'
write_main_header()
write_decl_impl(FwdKernel.get_all(), 'cutlass_forward', impl_file=forward_impl, enable_def=ENABLE_MACRO)