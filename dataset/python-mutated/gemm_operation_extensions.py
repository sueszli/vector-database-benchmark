from ..cutlass_utils import try_import_cutlass
if try_import_cutlass():
    import enum
    from cutlass_library.library import *
    from cutlass_library.gemm_operation import *

    class EmitGemmUniversal3xInstanceWithEVT:
        """Responsible for emitting a CUTLASS 3.x template definition"""

        def __init__(self, operation_suffix=''):
            if False:
                while True:
                    i = 10
            self.operation_suffix = operation_suffix
            self.includes = ['cutlass/cutlass.h', 'cutlass/gemm/gemm.h', 'cutlass/numeric_types.h', 'cutlass/gemm/kernel/gemm_universal.hpp', 'cutlass/gemm/collective/collective_builder.hpp', 'cutlass/epilogue/collective/collective_builder.hpp']
            self.builtin_epilogue_functor_template = '\n            ${epilogue_functor}<\n              ${element_c},\n              ${epilogue_vector_length},\n              ${element_accumulator},\n              ${element_epilogue}\n            >\n        '
            self.gemm_template = '\n        using EpilogueScheduleType = ${epilogue_schedule};\n        static_assert(cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecialized> ||\n                 cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecializedCooperative>,\n                "Epilogue visitor trees are currently only supported by the TMA warp-specialized epilogue");\n        static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;\n        using ElementAcc = ${element_accumulator};\n        using ElementD = ${element_d};\n        ${epilogue_functor};\n        using ${operation_name}_epilogue =\n          typename cutlass::epilogue::collective::CollectiveBuilder<\n            ${arch}, ${opcode_class},\n            cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,\n            cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,\n            cutlass::epilogue::collective::EpilogueTileAuto,\n            ${element_accumulator}, ${element_epilogue},\n            ${element_c}, ${layout_c}, ${align_c},\n            ${element_d}, ${layout_d}, ${align_d},\n            EpilogueScheduleType,\n            ${operation_name}_epilogue_functor\n          >::CollectiveOp;\n\n        using ${operation_name}_mainloop =\n          typename cutlass::gemm::collective::CollectiveBuilder<\n            ${arch}, ${opcode_class},\n            ${element_a}, ${layout_a}, ${align_a},\n            ${element_b}, ${layout_b}, ${align_b},\n            ${element_accumulator},\n            cute::Shape<cute::_${tile_shape_m}, cute::_${tile_shape_n}, cute::_${tile_shape_k}>,\n            cute::Shape<cute::_${cluster_m},cute::_${cluster_n},cute::_${cluster_k}>,\n            ${stages},\n          ${kernel_schedule}\n          >::CollectiveOp;\n\n        // Gemm operator ${operation_name}\n        using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<\n            cute::Shape<int,int,int,int>,\n            ${operation_name}_mainloop,\n            ${operation_name}_epilogue,\n            ${tile_scheduler}>;\n\n        // Define named type\n        struct ${operation_name} :\n          public ${operation_name}_base { };\n\n        '

        def instance_template(self):
            if False:
                for i in range(10):
                    print('nop')
            return '\n        ${compile_guard_start}\n          using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;\n          manifest.append(\n            new ${gemm_kind}<GemmKernel>("${operation_name}"));\n        ${compile_guard_end}\n        '

        def emit(self, operation):
            if False:
                for i in range(10):
                    print('nop')
            tile_shape = operation.tile_description.tile_shape
            warp_count = operation.tile_description.warp_count
            if operation.tile_description.stages > 0:
                stage_count_string = f'cutlass::gemm::collective::StageCount<{str(operation.tile_description.stages)}>'
            else:
                stage_count_string = f'cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage)>'
            warp_shape = [tile_shape[idx] // warp_count[idx] for idx in range(3)]
            (instance_layout_A, instance_layout_B, instance_layout_C, instance_layout_D) = (operation.A.layout, operation.B.layout, operation.C.layout, operation.D.layout)
            epilogue_vector_length = 1
            if isinstance(operation.epilogue_functor, enum.Enum):
                values = {'epilogue_vector_length': str(epilogue_vector_length), 'element_epilogue': str(DataTypeTag[operation.element_epilogue]), 'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor]}
                epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
            elif callable(operation.epilogue_functor):
                epilogue_functor = operation.epilogue_functor(operation.procedural_name() + '_epilogue_functor')
            else:
                epilogue_functor = str(operation.epilogue_functor)
            values = {'operation_name': operation.procedural_name(), 'operation_suffix': self.operation_suffix, 'element_a': DataTypeTag[operation.A.element], 'layout_a': LayoutTag[instance_layout_A], 'element_b': DataTypeTag[operation.B.element], 'layout_b': LayoutTag[instance_layout_B], 'element_c': DataTypeTag[operation.C.element], 'layout_c': LayoutTag[instance_layout_C], 'element_d': DataTypeTag[operation.D.element], 'layout_d': LayoutTag[instance_layout_D], 'element_accumulator': DataTypeTag[operation.accumulator_type()], 'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class], 'arch': 'cutlass::arch::Sm%d' % operation.arch, 'tile_shape_m': str(operation.tile_description.tile_shape[0]), 'tile_shape_n': str(operation.tile_description.tile_shape[1]), 'tile_shape_k': str(operation.tile_description.tile_shape[2]), 'cluster_m': str(operation.tile_description.cluster_shape[0]), 'cluster_n': str(operation.tile_description.cluster_shape[1]), 'cluster_k': str(operation.tile_description.cluster_shape[2]), 'warp_shape_m': str(warp_shape[0]), 'warp_shape_n': str(warp_shape[1]), 'warp_shape_k': str(warp_shape[2]), 'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]), 'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]), 'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]), 'kernel_schedule': str(KernelScheduleTag[operation.kernel_schedule]), 'epilogue_schedule': str(EpilogueScheduleTag[operation.epilogue_schedule]), 'epilogue_functor': epilogue_functor, 'stages': stage_count_string, 'align_a': str(operation.A.alignment), 'align_b': str(operation.B.alignment), 'align_c': str(operation.C.alignment), 'align_d': str(operation.C.alignment), 'transform_a': ComplexTransformTag[operation.A.complex_transform], 'transform_b': ComplexTransformTag[operation.B.complex_transform], 'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation], 'epilogue_vector_length': str(epilogue_vector_length), 'element_epilogue': str(DataTypeTag[operation.element_epilogue]), 'tile_scheduler': str(TileSchedulerTag[operation.tile_scheduler])}
            return SubstituteTemplate(self.gemm_template, values)