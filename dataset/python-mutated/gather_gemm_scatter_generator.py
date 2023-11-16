import sys
sys.path.append(sys.argv[1])
from gather_gemm_scatter_manifest import GatherGemmScatterManifest
from gather_gemm_scatter_operation import GatherGemmScatterOperation
from generator import ComplexTransform, CudaToolkitVersionSatisfies, EpilogueFunctor, GemmKind, SwizzlingFunctor, TensorDescription
from library import DataType, LayoutType, MathInstruction, MathOperation, OpcodeClass, TileDescription
from manifest import GeneratorTarget

def CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type, complex_transforms=None, epilogue_functor=EpilogueFunctor.LinearCombination, swizzling_functor=SwizzlingFunctor.Identity8):
    if False:
        i = 10
        return i + 15
    if complex_transforms is None:
        complex_transforms = [(ComplexTransform.none, ComplexTransform.none)]
    (element_a, element_b, element_c, element_epilogue) = data_type
    alignment_constraints = [0]
    if 'f16' == element_a.name or 'bf16' == element_a.name:
        alignment_constraints = [8]
    elif 'f32' == element_a.name or 'tf32' == element_a.name:
        alignment_constraints = [4]
    elif 'f64' == element_a.name:
        alignment_constraints = [1]
    operations = []
    for layout in layouts:
        for tile_description in tile_descriptions:
            for alignment in alignment_constraints:
                for complex_transform in complex_transforms:
                    alignment_c = min(8, alignment)
                    A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
                    B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
                    C = TensorDescription(element_c, layout[2], alignment_c)
                    new_operation = GatherGemmScatterOperation(GemmKind.Universal, tile_description.minimum_compute_capability, tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)
                    manifest.append(new_operation)
                    operations.append(new_operation)
    return operations

def GenerateSM80_TensorOp_16816(manifest, cuda_version, debug=False):
    if False:
        print('Hello World!')
    if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
        return
    layouts = [(LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor)]
    math_instructions = [MathInstruction([16, 8, 16], DataType.f16, DataType.f16, DataType.f16, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    min_cc = 80
    max_cc = 1024
    alignment_constraints = [8]
    for math_inst in math_instructions:
        tile_descriptions = [TileDescription([256, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 32], 3, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 32], 4, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 32], 4, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 5, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 32], 6, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 32], 6, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([256, 128, 64], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 64], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 64], 4, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 64], 4, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 64], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 64], 3, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 64], 3, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 64], 5, [2, 2, 1], math_inst, min_cc, max_cc)]
        if debug:
            tile_descriptions = [TileDescription([256, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc)]
        data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, math_inst.element_accumulator]
        CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type)
        if math_inst.element_a != math_inst.element_accumulator:
            data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, math_inst.element_accumulator]
            CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type_mixed)

def GenerateSM80_TensorOp_1688(manifest, cuda_version, debug=False):
    if False:
        while True:
            i = 10
    if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
        return
    layouts = [(LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor), (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor), (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor)]
    math_instructions = [MathInstruction([16, 8, 8], DataType.tf32, DataType.tf32, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    min_cc = 80
    max_cc = 1024
    for math_inst in math_instructions:
        tile_descriptions = [TileDescription([256, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 16], 4, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 16], 4, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 16], 6, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 16], 6, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([256, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 32], 4, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 32], 4, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 32], 5, [2, 2, 1], math_inst, min_cc, max_cc)]
        if debug:
            tile_descriptions = [TileDescription([256, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc)]
        data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, math_inst.element_accumulator]
        data_type_mixed = [math_inst.element_a, math_inst.element_b, math_inst.element_a, math_inst.element_accumulator]
        CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type)
        CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type_mixed)

def GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version, debug=False):
    if False:
        for i in range(10):
            print('nop')
    if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
        return
    layouts = [(LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor), (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor), (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor)]
    math_instructions = [MathInstruction([16, 8, 8], DataType.tf32, DataType.tf32, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    min_cc = 80
    max_cc = 1024
    for math_inst in math_instructions:
        tile_descriptions = [TileDescription([256, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 16], 4, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 16], 4, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 5, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 16], 6, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 16], 6, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 16], 10, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([256, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 32], 4, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 32], 4, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 32], 5, [2, 2, 1], math_inst, min_cc, max_cc)]
        if debug:
            tile_descriptions = [TileDescription([256, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc)]
        data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]
        CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type)

def GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version, debug=False):
    if False:
        print('Hello World!')
    if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
        return
    layouts = [(LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor), (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.RowMajor), (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.RowMajor)]
    math_instructions = [MathInstruction([16, 8, 8], DataType.f32, DataType.f32, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add_fast_f32)]
    min_cc = 80
    max_cc = 1024
    for math_inst in math_instructions:
        tile_descriptions = [TileDescription([128, 128, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 16], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 16], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 16], 4, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 16], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc)]
        if debug:
            tile_descriptions = [TileDescription([128, 128, 16], 4, [4, 2, 1], math_inst, min_cc, max_cc)]
        data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]
        CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type)

def GenerateSM75_TensorOp_1688(manifest, cuda_version, debug=False):
    if False:
        print('Hello World!')
    if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
        return
    layouts = [(LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.RowMajor)]
    math_instructions = [MathInstruction([16, 8, 8], DataType.f16, DataType.f16, DataType.f32, OpcodeClass.TensorOp, MathOperation.multiply_add), MathInstruction([16, 8, 8], DataType.f16, DataType.f16, DataType.f16, OpcodeClass.TensorOp, MathOperation.multiply_add)]
    min_cc = 75
    max_cc = 1024
    for math_inst in math_instructions:
        tile_descriptions = [TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc), TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc), TileDescription([256, 64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc), TileDescription([64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc)]
        if debug:
            tile_descriptions = [TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc)]
        data_type = [math_inst.element_a, math_inst.element_b, math_inst.element_accumulator, math_inst.element_accumulator]
        CreateGatherGemmScatterOperator(manifest, layouts, tile_descriptions, data_type)

def GenerateSM75(manifest, cuda_version, debug=False):
    if False:
        return 10
    GenerateSM75_TensorOp_1688(manifest, cuda_version, debug)

def GenerateSM80(manifest, cuda_version, debug=False):
    if False:
        return 10
    GenerateSM80_TensorOp_16816(manifest, cuda_version, debug)
    GenerateSM80_TensorOp_1688(manifest, cuda_version, debug)
    GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version, debug)
    GenerateSM80_TensorOp_1688_fast_fp32_math(manifest, cuda_version, debug)

class KernelCfg:

    def __init__(self, architectures, build_dir, cuda_version, curr_build_dir, disable_full_archs_compilation, filter_by_cc, generator_target, ignore_kernels, interface_dir, kernel_filter_file, kernels, operations, selected_kernel_list):
        if False:
            i = 10
            return i + 15
        self.architectures = architectures
        self.build_dir = build_dir
        self.cuda_version = cuda_version
        self.curr_build_dir = curr_build_dir
        self.disable_full_archs_compilation = disable_full_archs_compilation
        self.filter_by_cc = filter_by_cc
        self.generator_target = generator_target
        self.ignore_kernels = ignore_kernels
        self.interface_dir = interface_dir
        self.kernel_filter_file = kernel_filter_file
        self.kernels = kernels
        self.operations = operations
        self.selected_kernel_list = selected_kernel_list
if __name__ == '__main__':
    args = KernelCfg(architectures='80', build_dir=sys.argv[2], cuda_version=sys.argv[3], curr_build_dir=sys.argv[2], disable_full_archs_compilation=False, filter_by_cc='True', generator_target='library', ignore_kernels='', interface_dir=None, kernel_filter_file=None, kernels='', operations='all', selected_kernel_list=None)
    manifest = GatherGemmScatterManifest(args)
    debug = False
    GenerateSM75(manifest, args.cuda_version, debug)
    GenerateSM80(manifest, args.cuda_version, debug)
    manifest.emit(GeneratorTarget.Library)