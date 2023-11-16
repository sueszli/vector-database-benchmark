import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer
from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .cutlass_epilogue_gen import CutlassEVTEpilogueArgumentFormatter, CutlassEVTEpilogueTypeFormatter
log = logging.getLogger(__name__)
GEMM_TEMPLATE = '\n{{template.header().getvalue()}}\n{{template.globals().getvalue()}}\n{{instance_definition}}\n// When workspace_size is not a nullptr, populates requested workspace_size and returns.\n// Otherwise, computes the Gemm kernel using the given workspace ptr.\nextern "C" {\n{{kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=input_reorder)}} {\n  try {\n  {{kernel.check_not_null(X)}}\n  {{kernel.check_not_null(W)}}\n  {{kernel.check_not_null(Bias)}}\n  {{kernel.check_not_null(Y)}}\n  int64_t B = {{kernel.size(Y, 0, -3, default_value=1)}};\n  int64_t M = {{kernel.size(X, -2)}};\n  int64_t K = {{kernel.size(X, -1)}};\n  int64_t N = {{kernel.size(W, -1)}};\n  using ElementComputeEpilogue = {{instance_type}}::ElementAccumulator;\n  using coord_t = cutlass::gemm::GemmCoord::Index;\n  {{instance_type}}::Arguments arguments;\n  {{template.render_gemm_arguments(argument_template, epilogue_template, should_swap_xw,\n                                    X, W, Bias, Y, alpha, beta, kernel, epilogue_args)}}\n  {{instance_type}} gemm_op;\n  if (workspace_size) {\n    *workspace_size = gemm_op.get_workspace_size(arguments);\n    return 0;\n  }\n  {\n    auto status = gemm_op.can_implement(arguments);\n    CUTLASS_CHECK(status);\n  }\n  {\n    auto status = gemm_op.initialize(arguments, workspace, stream);\n    CUTLASS_CHECK(status);\n  }\n  {\n    auto status = gemm_op(stream);\n    CUTLASS_CHECK(status);\n  }\n  }\n  catch (std::exception& e) {\n    std::cerr << "Runtime error: " << e.what() << std::endl;\n    return -1;\n  }\n  catch (...) {\n    return -1;\n  }\n  return 0;\n}\n}\n'
GEMM_ARGS_CUTLASS_2X = "\n  int64_t batch_stride_x = {{kernel.stride(X, -3)}};\n  int64_t row_stride_x = {{kernel.row_or_column_stride(X)}};\n  int64_t batch_stride_w = {{kernel.stride(W, -3)}};\n  int64_t row_stride_w = {{kernel.row_or_column_stride(W)}};\n  int64_t batch_stride_bias = {{kernel.stride(Bias, -3)}};\n  int64_t row_stride_bias = {{kernel.row_or_column_stride(Bias)}};\n  int64_t batch_stride_y = {{kernel.stride(Y, -3)}};\n  int64_t row_stride_y = {{kernel.row_or_column_stride(Y)}};\n  // Initialize GemmUniversalInstance arguments.\n  arguments = {\n    {{template.gemm_mode()}},  // GemmUniversalMode mode\n    {\n      static_cast<coord_t>(M),\n      static_cast<coord_t>(N),\n      static_cast<coord_t>(K)\n    },  // GemmCoord problem_size\n    {{split_k if split_k > 1 else 'B'}},  // int batch_count\n    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue\n    {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // void const * ptr_A\n    {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // void const * ptr_B\n    {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // void const * ptr_C\n    {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // void * ptr_D\n    batch_stride_x,  // int64_t batch_stride_A\n    batch_stride_w,  // int64_t batch_stride_B\n    batch_stride_bias,  // int64_t batch_stride_C\n    batch_stride_y,  // int64_t batch_stride_D\n    row_stride_x,  // typename LayoutA::Stride::LongIndex lda\n    row_stride_w,  // typename LayoutB::Stride::LongIndex ldb\n    row_stride_bias,  // typename LayoutC::Stride::LongIndex ldc\n    row_stride_y,  // typename LayoutC::Stride::LongIndex ldd\n  };\n"
GEMM_ARGS_CUTLASS_3X = '\n  // Initialize GemmUniversal3xInstance arguments.\n  arguments = {\n    {{template.gemm_mode()}},  // GemmUniversalMode mode\n    {\n      static_cast<coord_t>({{M}}),\n      static_cast<coord_t>({{N}}),\n      static_cast<coord_t>(K),\n      static_cast<coord_t>(B)\n    }, // ProblemShape problem_shape\n    {\n      {{template.cutlass_type_cast(X, kernel.ptr(X))}},  // ElementA const* ptr_A\n      {\n        {{template.cute_int(kernel.stride(X, -2), "stride_x0")}},\n        {{template.cute_int(kernel.stride(X, -1), "stride_x1")}},\n        {{template.cute_int(kernel.stride(X, -3), "batch_stride_x")}}\n      },  // StrideA dA\n      {{template.cutlass_type_cast(W, kernel.ptr(W))}},  // ElementB const* ptr_B\n      {\n        {{template.cute_int(kernel.stride(W, -1), "stride_w1")}},\n        {{template.cute_int(kernel.stride(W, -2), "stride_w0")}},\n        {{template.cute_int(kernel.stride(W, -3), "batch_stride_w")}}\n      },  // StrideB dB\n    },  // MainloopArguments mainloop\n    {{epilogue_arguments}}\n  };\n'
GEMM_ARGS_CUTLASS_3X_EPILOGUE = '\n    // see https://tinyurl.com/4rk89z48\n    {\n      {{epilogue_args}},  // thread, typename FusionCallbacks::Arguments ( EVT ) or ThreadEpilogueOp::Params (non-EVT )\n      {{template.cutlass_type_cast(Bias, kernel.ptr(Bias))}},  // ElementC const* ptr_C\n      {\n        {{template.cute_int(kernel.stride(Bias, -2, 1), "stride_bias0")}},\n        {{template.cute_int(kernel.stride(Bias, -1, 1), "stride_bias1")}},\n        {{template.cute_int(kernel.stride(Bias, -3), "batch_stride_bias")}}\n      },  // StrideC dC\n      {{template.cutlass_type_cast(Y, kernel.ptr(Y))}},  // ElementD const* ptr_D\n      {\n        {{template.cute_int(kernel.stride(Y, -2), "stride_y0")}},\n        {{template.cute_int(kernel.stride(Y, -1), "stride_y1")}},\n        {{template.cute_int(kernel.stride(Y, -3), "batch_stride_y")}}\n      },  // StrideD dD\n    },  // EpilogueArguments epilogue\n'

class CUTLASSGemmTemplate(CUTLASSTemplate):
    """
    CUTLASS GEMM template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """

    def __init__(self, input_nodes: List[Buffer], layout: Layout, alpha: float, beta: float, input_reorder: Optional[List[int]]=None, can_fuse_epilogue: Optional[bool]=None):
        if False:
            print('Hello World!')
        '\n        Args:\n            input_nodes: input nodes of the kernel\n            layout: layout of the output node\n            alpha: alpha value of the GEMM operation\n            beta: beta value of the GEMM operation\n            input_reorder: reorder of the input nodes\n            can_fuse_epilogue: If set to True, will only list and use operators capable of flexible epilogue fusions.\n                               If False, it will not use those. If None, both may be listed, but it will not allow fusions.\n                               Defaults to None\n        '
        super().__init__('cutlass_gemm', input_nodes, layout, input_reorder)
        self.alpha = alpha
        self.beta = beta
        self.can_fuse_epilogue = can_fuse_epilogue

    @staticmethod
    def add_cutlass_gemm_choices(choices, layout, input_nodes, alpha=1, beta=0, input_reorder=None, fuseable=True, non_fuseable=True):
        if False:
            for i in range(10):
                print('nop')
        if non_fuseable:
            if fuseable:
                can_fuse_epilogue = False
            else:
                can_fuse_epilogue = None
            cutlass_template = CUTLASSGemmTemplate(input_nodes, layout, alpha=alpha, beta=beta, input_reorder=input_reorder, can_fuse_epilogue=can_fuse_epilogue)
            ops = cutlass_template.gen_ops()
            for op in ops:
                cutlass_template.maybe_append_choice(choices, op=op)
        else:
            ops = []
        if fuseable:
            cutlass_template_evt = CUTLASSGemmTemplate(input_nodes, layout, alpha=alpha, beta=beta, input_reorder=input_reorder, can_fuse_epilogue=True)
            ops_evt = cutlass_template_evt.gen_ops()
            for op in ops_evt:
                cutlass_template_evt.maybe_append_choice(choices, op=op)
        else:
            ops_evt = []
        log.debug('Added %d cutlass gemm configs and %d fuseable gemm configs.', len(ops), len(ops_evt))

    def header(self) -> IndentedBuffer:
        if False:
            print('Hello World!')
        res = super().header()
        res.splice('\n                #include "cutlass/gemm/gemm.h"\n                #include "cutlass/gemm/device/gemm_universal.h"\n                #include "cutlass/gemm/device/gemm_universal_adapter.h"\n                #include "cutlass/gemm/kernel/gemm_universal.hpp"\n                #include "cutlass/gemm/collective/collective_builder.hpp"\n                #include "cutlass/epilogue/collective/collective_builder.hpp"\n                #include "cutlass/epilogue/collective/default_epilogue.hpp"\n                #include "cutlass/epilogue/thread/linear_combination.h"\n                #include "cutlass/gemm/dispatch_policy.hpp"\n                #include "cutlass/gemm/kernel/tile_scheduler.hpp"\n                #include "cutlass/util/distribution.h"\n                #include "cutlass/util/packed_stride.hpp"\n                #include "cutlass/util/tensor_view_io.h"\n            ')
        return res

    @staticmethod
    def cutlass_layout(torch_layout) -> 'Optional[cutlass_lib.LayoutType]':
        if False:
            for i in range(10):
                print('nop')
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib
        if torch_layout.stride[-1] == 1:
            return cutlass_lib.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return None

    @staticmethod
    def flip_cutlass_layout(cutlass_layout: 'cutlass_lib.LayoutType') -> 'cutlass_lib.LayoutType':
        if False:
            while True:
                i = 10
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib
        if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
            return cutlass_lib.LayoutType.ColumnMajor
        else:
            return cutlass_lib.LayoutType.RowMajor

    @staticmethod
    def layout_match(torch_layout, cutlass_layout) -> bool:
        if False:
            i = 10
            return i + 15
        return CUTLASSGemmTemplate.cutlass_layout(torch_layout) == cutlass_layout

    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        if False:
            for i in range(10):
                print('nop')
        alignment = cutlass_utils.get_max_alignment(torch_layout)
        if alignment < op_element.alignment:
            return False
        else:
            op_element.alignment = alignment
            return True

    @staticmethod
    def has_tma_epilogue(op) -> bool:
        if False:
            for i in range(10):
                print('nop')
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib
        result = False
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split('.')[-1]
            result = epilogue_schedule_str.lower().startswith('tma')
        return result

    @staticmethod
    def supports_evt(op: 'cutlass_library.gemm_op.GemmOperation') -> bool:
        if False:
            while True:
                i = 10
        '\n        returns True if the op is capable of flexible epilogue fusions\n        using epilogue visitor trees.\n\n        See https://github.com/NVIDIA/cutlass/blob/e01b9b5029b7caca5a43c29f7d2714d7cf1dcae8/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu#L283-L285 # noqa: B950\n        '
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib
        if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
            return False
        if op.epilogue_schedule not in (cutlass_lib.EpilogueScheduleType.TmaWarpSpecialized, cutlass_lib.EpilogueScheduleType.TmaWarpSpecializedCooperative):
            return False
        return True

    def render_evt_epilogue_declaration(self, template_output_node_name: str, evt_type_name: str, epilogue_nodes: List[IRNode]) -> str:
        if False:
            print('Hello World!')
        'Generates the epilogue for the EVT epilogue fusion'
        return CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(template_output_node_name, evt_type_name, epilogue_nodes)

    def define_gemm_instance(self, op: 'cutlass_library.gemm_op.GemmOperation', output_buffer_name: str, epilogue_nodes: Optional[List[IRNode]]=None) -> Tuple[str, str]:
        if False:
            return 10
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib
        from torch._inductor.codegen.cuda.cutlass_lib_extensions.gemm_operation_extensions import EmitGemmUniversal3xInstanceWithEVT
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            if epilogue_nodes is not None and len(epilogue_nodes) > 0:
                emitter = EmitGemmUniversal3xInstanceWithEVT()
                op.epilogue_functor = lambda epilogue_functor_type_name: self.render_evt_epilogue_declaration(output_buffer_name, epilogue_functor_type_name, epilogue_nodes)
            else:
                emitter = cutlass_gemm_op.EmitGemmUniversal3xInstance()
            op_def = emitter.emit(op)
            pattern = re.compile('\\s*struct\\s(.*?)\\s:')
            decl = [line for line in op_def.split('\n') if 'struct ' in line][-1]
        else:
            if epilogue_nodes is not None and len(epilogue_nodes) > 0:
                raise RuntimeError('EVT epilogue fusion is not supported for Cutlass 2.x ops.')
            emitter = cutlass_gemm_op.EmitGemmInstance()
            op_def = emitter.emit(op)
            op_def = op_def.replace('cutlass::gemm::device::Gemm', 'cutlass::gemm::device::GemmUniversal')
            op_def = op_def.replace('false,', '')
            pattern = re.compile('\\s*using\\s(.*?)\\s=')
            decl = op_def.split('\n')[2]
        match = pattern.match(decl)
        if match is None:
            raise RuntimeError('Invalid Gemm config: \n' + op_def)
        op_type = match.groups()[0]
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            op_def += f'\n  using {op_type}_device_type = cutlass::gemm::device::GemmUniversalAdapter<{op_type}>;\n'
            op_type = f'{op_type}_device_type'
        return (op_def, op_type)

    @staticmethod
    def should_swap_XW(bias: IRNode, beta: float) -> bool:
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def swap_XW(op: 'cutlass_library.gemm_op.GemmOperation') -> 'cutlass_library.gemm_op.GemmOperation':
        if False:
            i = 10
            return i + 15
        new_op = copy.deepcopy(op)
        new_op.A.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.A.layout)
        new_op.B.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.B.layout)
        (new_op.A, new_op.B) = (new_op.B, new_op.A)
        new_op.C.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.C.layout)
        new_op.D.layout = CUTLASSGemmTemplate.flip_cutlass_layout(new_op.D.layout)
        return new_op

    def filter_op(self, op: 'cutlass_library.gemm_op.GemmOperation') -> 'cutlass_library.gemm_op.GemmOperation':
        if False:
            i = 10
            return i + 15
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.library as cutlass_lib
        if op.tile_description.math_instruction.opcode_class == cutlass_lib.OpcodeClass.Simt:
            return None
        if op.gemm_kind not in {cutlass_lib.GemmKind.Universal, cutlass_lib.GemmKind.Universal3x}:
            return None
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        accumulator_torch_dtype = cutlass_utils.get_accumulator_dtype([X.get_dtype(), W.get_dtype()])
        if not (cutlass_utils.dtype_match(X.get_dtype(), op.A.element) and cutlass_utils.dtype_match(W.get_dtype(), op.B.element) and cutlass_utils.dtype_match(self.output_node.get_layout().dtype, op.C.element) and cutlass_utils.dtype_match(accumulator_torch_dtype, op.accumulator_type())):
            return None
        if not (self.layout_match(X.get_layout(), op.A.layout) and self.layout_match(W.get_layout(), op.B.layout)):
            return None
        op = copy.deepcopy(op)
        op.D.layout = CUTLASSGemmTemplate.cutlass_layout(self.output_node.get_layout())
        if not (self.set_alignment(X.get_layout(), op.A) and self.set_alignment(W.get_layout(), op.B) and self.set_alignment(self.output_node.get_layout(), op.D)):
            return None
        op.element_epilogue = op.accumulator_type()
        if len(self.input_nodes) >= 3 and self.input_nodes[2] is not None:
            Bias = self.input_nodes[2]
            bias_layout = CUTLASSGemmTemplate.cutlass_layout(Bias.get_layout())
            if op.gemm_kind != cutlass_lib.GemmKind.Universal3x:
                if bias_layout != op.D.layout:
                    return None
            else:
                op.C.layout = bias_layout
            if not self.set_alignment(Bias.get_layout(), op.C):
                return None
        elif op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            op.C.element = cutlass_lib.DataType.void
        else:
            op.C.layout = op.D.layout
        supports_evt: bool = self.supports_evt(op)
        if self.can_fuse_epilogue is not None and self.can_fuse_epilogue != supports_evt:
            return None
        if inductor_cuda_config.cutlass_only_evt_capable_ops and (not supports_evt):
            return None
        return op

    def gen_ops(self) -> 'List[cutlass_gemm_op.GemmOperation]':
        if False:
            return 10
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib
        ops = cutlass_utils.gen_ops()[cutlass_lib.OperationKind.Gemm]
        res: Dict[str, cutlass_gemm_op.GemmOperation] = dict()
        num_3x_ops = 0
        num_2x_ops = 0
        for op_dict in ops.values():
            for op_list in op_dict.values():
                for op in op_list:
                    assert isinstance(op, cutlass_gemm_op.GemmOperation)
                    filter_res = self.filter_op(op)
                    if filter_res is not None and res.get(filter_res.configuration_name(), None) is None:
                        res[filter_res.configuration_name()] = filter_res
        for op in res.values():
            if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
                num_3x_ops += 1
            else:
                num_2x_ops += 1
        log.debug('Got cutlass configs: total number of ops: %d, total number of 3x ops: %d, total number of 2x ops: %d', len(res), num_3x_ops, num_2x_ops)
        return list(res.values())[:inductor_cuda_config.cutlass_max_profiling_configs]

    def gemm_mode(self) -> str:
        if False:
            print('Hello World!')
        sizes = self.output_node.get_size()
        if len(sizes) > 2:
            return 'cutlass::gemm::GemmUniversalMode::kBatched'
        else:
            return 'cutlass::gemm::GemmUniversalMode::kGemm'

    def render_gemm_arguments(self, argument_template: str, epilogue_template: str, should_swap_xw: bool, X: IRNode, W: IRNode, Bias: IRNode, Y: IRNode, alpha: float, beta: float, kernel: CUDATemplateKernel, epilogue_args) -> str:
        if False:
            while True:
                i = 10
        options = dict(alpha=self.alpha, beta=self.beta, X=X, W=W, Y=Y, Bias=Bias, template=self, kernel=kernel, M='M', N='N', epilogue_args=epilogue_args)
        if epilogue_template is not None:
            if should_swap_xw:

                def clone_with_transposed_stride(node: IRNode) -> IRNode:
                    if False:
                        for i in range(10):
                            print('nop')
                    old_layout = node.get_layout()
                    new_stride = list(old_layout.stride)
                    (new_stride[-2], new_stride[-1]) = (new_stride[-1], new_stride[-2])
                    new_layout = FixedLayout(old_layout.device, old_layout.dtype, list(old_layout.size), new_stride, old_layout.offset)
                    return Buffer(node.get_name(), new_layout)
                new_X = clone_with_transposed_stride(X)
                new_W = clone_with_transposed_stride(W)
                new_Bias = clone_with_transposed_stride(Bias)
                new_Y = clone_with_transposed_stride(Y)
                (options['X'], options['W'], options['Bias'], options['Y']) = (new_W, new_X, new_Bias, new_Y)
                (options['M'], options['N']) = ('N', 'M')
            epilogue_arguments = self._template_from_string(epilogue_template).render(**options)
            arguments = self._template_from_string(argument_template).render(epilogue_arguments=epilogue_arguments, **options)
        else:
            arguments = self._template_from_string(GEMM_ARGS_CUTLASS_2X).render(split_k=1, **options)
        return arguments

    def render(self, kernel: CUDATemplateKernel, op: 'cutlass_gemm_op.GemmOperation'=None, template_buffer_node: Optional[CUDATemplateBuffer]=None, epilogue_nodes: Optional[List[IRNode]]=None, **kwargs) -> str:
        if False:
            print('Hello World!')
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            assert self.can_fuse_epilogue and CUTLASSGemmTemplate.supports_evt(op), 'op does not support EVT epilogue fusion'
            assert template_buffer_node is not None, 'Template node is required for epilogue fusion'
            assert isinstance(template_buffer_node, CUDATemplateBuffer), f'Template node has to be a CUDATemplateBuffer, is type {type(template_buffer_node)}'
            assert template_buffer_node.name is not None, 'Output node has to be a Buffer with a name'
        template_output_node_name = template_buffer_node.name if template_buffer_node is not None else None
        assert cutlass_utils.try_import_cutlass()
        import cutlass_library.gemm_operation as cutlass_gemm_op
        import cutlass_library.library as cutlass_lib
        assert isinstance(op, cutlass_gemm_op.GemmOperation), 'op argument is required and has to be an instance of GemmOperation'
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])
        assert len(self.input_nodes) >= 2 and self.output_node is not None
        (X, W) = (self.input_nodes[0], self.input_nodes[1])
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]
        epilogue_template: Optional[str] = None
        should_swap_xw: bool = False
        epilogue_args = f'{{ElementComputeEpilogue({self.alpha}), ElementComputeEpilogue({self.beta})}}'
        if op.gemm_kind == cutlass_lib.GemmKind.Universal3x:
            if Bias is not None and self.has_tma_epilogue(op):
                if self.should_swap_XW(Bias, self.beta):
                    op = self.swap_XW(op)
                    should_swap_xw = True
            if epilogue_nodes is not None and len(epilogue_nodes) > 0:
                epilogue_args = CutlassEVTEpilogueArgumentFormatter.ir_to_evt_argument_string(cast(str, template_output_node_name), epilogue_nodes)
            epilogue_template = GEMM_ARGS_CUTLASS_3X_EPILOGUE
            argument_template = GEMM_ARGS_CUTLASS_3X
        else:
            argument_template = GEMM_ARGS_CUTLASS_2X
        (instance_definition, instance_type) = self.define_gemm_instance(op, cast(str, template_output_node_name), epilogue_nodes)
        options = dict(alpha=self.alpha, beta=self.beta, X=X, W=W, Y=Y, Bias=Bias, epilogue_template=epilogue_template, argument_template=argument_template, should_swap_xw=should_swap_xw, template=self, kernel=kernel, instance_definition=instance_definition, instance_type=instance_type, input_reorder=self.input_reorder, epilogue_args=epilogue_args)
        res = self._template_from_string(GEMM_TEMPLATE).render(**options)
        return res