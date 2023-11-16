import collections
import enum
import os.path
from gemm_operation import EmitGemmConfigurationLibrary, EmitGemmInstance, EpilogueFunctor, GemmOperation, SwizzlingFunctor
from library import ComplexTransformTag, DataTypeSize, DataTypeTag, EpilogueFunctorTag, GemmKind, LayoutTag, LayoutType, MathOperationTag, OpcodeClassTag, SubstituteTemplate, SwizzlingFunctorTag

class EmitGatherGemmScatterInstance(EmitGemmInstance):

    def __init__(self, operation_suffix=''):
        if False:
            while True:
                i = 10
        self.operation_suffix = operation_suffix
        self.includes = []
        self.builtin_epilogue_functor_template = '\n    ${epilogue_functor}<\n      ${element_c},\n      ${epilogue_vector_length},\n      ${element_accumulator},\n      ${element_epilogue}\n    >\n'
        self.gemm_template = '\n// Gemm operator ${operation_name}\ntemplate<cutlass::gemm::GemmUniversalMode Mode_ =\n             cutlass::gemm::GemmUniversalMode::kGemm>\nstruct ${operation_name} {\n  using Gemm =\n    cutlass::gemm::device::GemmUniversal<\n      ${element_a},\n      ${layout_a},\n      ${element_b},\n      ${layout_b},\n      ${element_c},\n      ${layout_c},\n      ${element_accumulator},\n      ${opcode_class},\n      ${arch},\n      cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,\n      cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,\n      cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,\n      ${epilogue_functor},\n      ${swizzling_functor},\n      ${stages},\n      ${align_a},\n      ${align_b},\n      ${math_operation},\n      ${transform_a},\n      ${transform_b},\n      ${gather_a}, // gather a\n      ${gather_b}, // gather b\n      ${scatter_d} // scatter d\n    >;\n  static const cutlass::gemm::GemmUniversalMode Mode = Mode_;\n};\n'

    def instance_template(self):
        if False:
            i = 10
            return i + 15
        return ''

    def emit(self, operation):
        if False:
            for i in range(10):
                print('nop')
        threadblock_shape = operation.tile_description.threadblock_shape
        warp_count = operation.tile_description.warp_count
        warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]
        transpose_layouts = {LayoutType.ColumnMajor: LayoutType.ColumnMajor, LayoutType.RowMajor: LayoutType.RowMajor}
        if operation.A.layout in transpose_layouts.keys() and operation.B.layout in transpose_layouts.keys() and (operation.C.layout in transpose_layouts.keys()):
            instance_layout_A = transpose_layouts[operation.A.layout]
            instance_layout_B = transpose_layouts[operation.B.layout]
            instance_layout_C = transpose_layouts[operation.C.layout]
            gemm_template = self.gemm_template
        else:
            (instance_layout_A, instance_layout_B, instance_layout_C) = (operation.A.layout, operation.B.layout, operation.C.layout)
            gemm_template = self.gemm_template_interleaved
        if isinstance(operation.epilogue_functor, enum.Enum):
            epilogue_vector_length = min(operation.C.alignment * DataTypeSize[operation.C.element], 128) // DataTypeSize[operation.C.element]
            values = {'epilogue_vector_length': str(epilogue_vector_length), 'element_epilogue': str(DataTypeTag[operation.element_epilogue]), 'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor]}
            epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
        else:
            epilogue_functor = self.epilogue_functor.emit_declaration()
        values = {'operation_name': operation.procedural_name(), 'operation_suffix': self.operation_suffix, 'element_a': DataTypeTag[operation.A.element], 'layout_a': LayoutTag[instance_layout_A], 'element_b': DataTypeTag[operation.B.element], 'layout_b': LayoutTag[instance_layout_B], 'element_c': DataTypeTag[operation.C.element], 'layout_c': LayoutTag[instance_layout_C], 'element_accumulator': DataTypeTag[operation.accumulator_type()], 'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class], 'arch': 'cutlass::arch::Sm%d' % operation.arch, 'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]), 'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]), 'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]), 'warp_shape_m': str(warp_shape[0]), 'warp_shape_n': str(warp_shape[1]), 'warp_shape_k': str(warp_shape[2]), 'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]), 'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]), 'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]), 'epilogue_functor': epilogue_functor, 'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor], 'stages': str(operation.tile_description.stages), 'align_a': str(operation.A.alignment), 'align_b': str(operation.B.alignment), 'transform_a': ComplexTransformTag[operation.A.complex_transform], 'transform_b': ComplexTransformTag[operation.B.complex_transform], 'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation], 'gather_a': 'true', 'gather_b': str(operation.layout_name() == 'tn').lower(), 'scatter_d': str(operation.layout_name() != 'tn').lower()}
        return SubstituteTemplate(gemm_template, values)

class EmitGatherGemmScatterConfigurationLibrary(EmitGemmConfigurationLibrary):

    def __init__(self, operation_path, configuration_name):
        if False:
            i = 10
            return i + 15
        self.configuration_name = configuration_name
        self.configuration_path = os.path.join(operation_path, 'configurations.h.tmp').replace('\\', '/')
        self.instance_emitter = {GemmKind.Universal: EmitGatherGemmScatterInstance}
        self.gemm_kind_wrappers = {GemmKind.Universal: 'GemmUniversalOperation'}
        self.wmma_guard_start = '#if defined(CUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)'
        self.separator = '\n///////////////////////////////////////////////////////////////////////////////////////////////////\n\n'
        self.header_template = '\n/*\n  Generated by gemm_operation.py - Do not edit.\n*/\n#pragma once\n#if defined(PADDLE_WITH_CUTLASS) && SPCONV_WITH_CUTLASS\n'
        self.namespace_template = '\nnamespace phi {\nnamespace sparse {\n'
        self.epilogue_template = '\n}  // namespace sparse\n}  // namespace phi\n#endif\n'

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.configuration_file = open(self.configuration_path, 'a')
        self.includes = collections.OrderedDict([])
        self.instance_definitions = []
        self.instance_wrappers = []
        self.operations = []
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if False:
            i = 10
            return i + 15
        for instance_definition in self.instance_definitions:
            self.configuration_file.write(instance_definition)
        self.configuration_file.close()

class GatherGemmScatterOperation(GemmOperation):

    def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, epilogue_functor=EpilogueFunctor.LinearCombination, swizzling_functor=SwizzlingFunctor.Identity8):
        if False:
            i = 10
            return i + 15
        super().__init__(gemm_kind, arch, tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)
        self.ShortLayoutTypeNames = {LayoutType.ColumnMajor: 't', LayoutType.ColumnMajorInterleaved2: 't2', LayoutType.ColumnMajorInterleaved32: 't32', LayoutType.ColumnMajorInterleaved64: 't64', LayoutType.RowMajor: 'n', LayoutType.RowMajorInterleaved2: 'n2', LayoutType.RowMajorInterleaved32: 'n32', LayoutType.RowMajorInterleaved64: 'n64', LayoutType.TensorNHWC: 'nhwc', LayoutType.TensorNDHWC: 'ndhwc', LayoutType.TensorNCHW: 'nchw', LayoutType.TensorNGHWC: 'nghwc', LayoutType.TensorNC32HW32: 'nc32hw32', LayoutType.TensorNC64HW64: 'nc64hw64', LayoutType.TensorC32RSK32: 'c32rsk32', LayoutType.TensorC64RSK64: 'c64rsk64'}

    def layout_name(self):
        if False:
            while True:
                i = 10
        return '{}{}'.format(self.ShortLayoutTypeNames[self.A.layout], self.ShortLayoutTypeNames[self.B.layout])