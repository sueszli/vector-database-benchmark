from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.check_kernel_launches import check_cuda_kernel_launches, check_code_for_cuda_kernel_launches

class AlwaysCheckCudaLaunchTest(TestCase):

    def test_check_code(self):
        if False:
            return 10
        'Verifies that the regex works for a few different situations'
        self.assertEqual(2, check_code_for_cuda_kernel_launches('\nsome_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\nsome_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);\n\nsome_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\nsome_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);\nsome_other_stuff;\nsome_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\nsome_function_call<TemplateArg><<<1,2,0,stream>>> (arg1,arg2,arg3);\nC10_CUDA_KERNEL_LAUNCH_CHECK();\nsome_function_call<TemplateArg><<<1,2,0,stream>>> ( arg1 , arg2 , arg3 ) ;\n\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n        '))
        self.assertEqual(0, check_code_for_cuda_kernel_launches('\n#define SOME_MACRO(x) some_function_call<<<1,2>>> ( x ) ;  \\\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n\n#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)  \\\n  indexAddSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM> \\\n    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                \\\n      selfInfo, sourceInfo, indexInfo,                                               \\\n      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize);                          \\\n  C10_CUDA_KERNEL_LAUNCH_CHECK();\n        '))
        self.assertEqual(1, check_code_for_cuda_kernel_launches('\n            rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(\n                    numel,\n                    rng_engine_inputs,\n                    output_data,\n                    input_data,\n                    noise_data,\n                    lower,\n                    upper,\n                    [] __device__ (curandStatePhilox4_32_10_t* state) {\n                    return curand_uniform2_double(state);\n                    });\n                    C10_CUDA_KERNEL_LAUNCH_CHECK();\n\n            rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(\n                    numel,\n                    rng_engine_inputs,\n                    output_data,\n                    input_data,\n                    noise_data,\n                    lower,\n                    upper,\n                    [] __device__ (curandStatePhilox4_32_10_t* state) {\n                    return curand_uniform2_double(state);\n                    });\n                    uh oh;\n                    C10_CUDA_KERNEL_LAUNCH_CHECK();\n        '))

    def test_check_cuda_launches(self):
        if False:
            i = 10
            return i + 15
        unsafeLaunchesCount = check_cuda_kernel_launches()
        self.assertTrue(unsafeLaunchesCount == 0)
if __name__ == '__main__':
    run_tests()