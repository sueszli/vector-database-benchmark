import io
import sys
import torch
import yaml
from android_api_module import AndroidAPIModule
from builtin_ops import TSBuiltinOpsModule, TSCollectionOpsModule
from math_ops import PointwiseOpsModule, ReductionOpsModule, ComparisonOpsModule, OtherMathOpsModule, SpectralOpsModule, BlasLapackOpsModule
from nn_ops import NNConvolutionModule, NNPoolingModule, NNPaddingModule, NNNormalizationModule, NNActivationModule, NNRecurrentModule, NNTransformerModule, NNLinearModule, NNDropoutModule, NNSparseModule, NNDistanceModule, NNLossFunctionModule, NNVisionModule, NNShuffleModule, NNUtilsModule
from quantization_ops import GeneralQuantModule, StaticQuantModule, FusedQuantModule
from sampling_ops import SamplingOpsModule
from tensor_ops import TensorOpsModule, TensorCreationOpsModule, TensorIndexingOpsModule, TensorTypingOpsModule, TensorViewOpsModule
from torch.jit.mobile import _load_for_lite_interpreter
from torchvision_models import MobileNetV2Module, MobileNetV2VulkanModule, Resnet18Module
test_path_ios = 'ios/TestApp/models/'
test_path_android = 'android/pytorch_android/src/androidTest/assets/'
production_ops_path = 'test/mobile/model_test/model_ops.yaml'
coverage_out_path = 'test/mobile/model_test/coverage.yaml'
all_modules = {'pointwise_ops': PointwiseOpsModule(), 'reduction_ops': ReductionOpsModule(), 'comparison_ops': ComparisonOpsModule(), 'spectral_ops': SpectralOpsModule(), 'other_math_ops': OtherMathOpsModule(), 'blas_lapack_ops': BlasLapackOpsModule(), 'sampling_ops': SamplingOpsModule(), 'tensor_general_ops': TensorOpsModule(), 'tensor_creation_ops': TensorCreationOpsModule(), 'tensor_indexing_ops': TensorIndexingOpsModule(), 'tensor_typing_ops': TensorTypingOpsModule(), 'tensor_view_ops': TensorViewOpsModule(), 'convolution_ops': NNConvolutionModule(), 'pooling_ops': NNPoolingModule(), 'padding_ops': NNPaddingModule(), 'activation_ops': NNActivationModule(), 'normalization_ops': NNNormalizationModule(), 'recurrent_ops': NNRecurrentModule(), 'transformer_ops': NNTransformerModule(), 'linear_ops': NNLinearModule(), 'dropout_ops': NNDropoutModule(), 'sparse_ops': NNSparseModule(), 'distance_function_ops': NNDistanceModule(), 'loss_function_ops': NNLossFunctionModule(), 'vision_function_ops': NNVisionModule(), 'shuffle_ops': NNShuffleModule(), 'nn_utils_ops': NNUtilsModule(), 'general_quant_ops': GeneralQuantModule(), 'static_quant_ops': StaticQuantModule(), 'fused_quant_ops': FusedQuantModule(), 'torchscript_builtin_ops': TSBuiltinOpsModule(), 'torchscript_collection_ops': TSCollectionOpsModule(), 'mobilenet_v2': MobileNetV2Module(), 'mobilenet_v2_vulkan': MobileNetV2VulkanModule(), 'resnet18': Resnet18Module(), 'android_api_module': AndroidAPIModule()}
models_need_trace = ['static_quant_ops']

def calcOpsCoverage(ops):
    if False:
        while True:
            i = 10
    with open(production_ops_path) as input_yaml_file:
        production_ops_dict = yaml.safe_load(input_yaml_file)
    production_ops = set(production_ops_dict['root_operators'].keys())
    all_generated_ops = set(ops)
    covered_ops = production_ops.intersection(all_generated_ops)
    uncovered_ops = production_ops - covered_ops
    coverage = round(100 * len(covered_ops) / len(production_ops), 2)
    total_occurances = sum(production_ops_dict['root_operators'].values())
    covered_ops_dict = {op: production_ops_dict['root_operators'][op] for op in covered_ops}
    uncovered_ops_dict = {op: production_ops_dict['root_operators'][op] for op in uncovered_ops}
    covered_occurances = sum(covered_ops_dict.values())
    occurances_coverage = round(100 * covered_occurances / total_occurances, 2)
    print(f'\n{len(uncovered_ops)} uncovered ops: {uncovered_ops}\n')
    print(f'Generated {len(all_generated_ops)} ops')
    print(f'Covered {len(covered_ops)}/{len(production_ops)} ({coverage}%) production ops')
    print(f'Covered {covered_occurances}/{total_occurances} ({occurances_coverage}%) occurances')
    print(f'pytorch ver {torch.__version__}\n')
    with open(coverage_out_path, 'w') as f:
        yaml.safe_dump({'_covered_ops': len(covered_ops), '_production_ops': len(production_ops), '_generated_ops': len(all_generated_ops), '_uncovered_ops': len(uncovered_ops), '_coverage': round(coverage, 2), 'uncovered_ops': uncovered_ops_dict, 'covered_ops': covered_ops_dict, 'all_generated_ops': sorted(all_generated_ops)}, f)

def getModuleFromName(model_name):
    if False:
        i = 10
        return i + 15
    if model_name not in all_modules:
        print('Cannot find test model for ' + model_name)
        return (None, [])
    module = all_modules[model_name]
    if not isinstance(module, torch.nn.Module):
        module = module.getModule()
    has_bundled_inputs = False
    if model_name in models_need_trace:
        module = torch.jit.trace(module, [])
    else:
        module = torch.jit.script(module)
    ops = torch.jit.export_opnames(module)
    print(ops)
    runModule(module)
    return (module, ops)

def runModule(module):
    if False:
        print('Hello World!')
    buffer = io.BytesIO(module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    lite_module = _load_for_lite_interpreter(buffer)
    if lite_module.find_method('get_all_bundled_inputs'):
        input = lite_module.run_method('get_all_bundled_inputs')[0]
        lite_module.forward(*input)
    else:
        lite_module()

def generateAllModels(folder, on_the_fly=False):
    if False:
        while True:
            i = 10
    all_ops = []
    for name in all_modules:
        (module, ops) = getModuleFromName(name)
        all_ops = all_ops + ops
        path = folder + name + ('_temp.ptl' if on_the_fly else '.ptl')
        module._save_for_lite_interpreter(path)
        print('model saved to ' + path)
    calcOpsCoverage(all_ops)

def generateModel(name):
    if False:
        for i in range(10):
            print('nop')
    (module, ops) = getModuleFromName(name)
    if module is None:
        return
    path_ios = test_path_ios + name + '.ptl'
    path_android = test_path_android + name + '.ptl'
    module._save_for_lite_interpreter(path_ios)
    module._save_for_lite_interpreter(path_android)
    print('model saved to ' + path_ios + ' and ' + path_android)

def main(argv):
    if False:
        print('Hello World!')
    if argv is None or len(argv) != 1:
        print('\nThis script generate models for mobile test. For each model we have a "storage" version\nand an "on-the-fly" version. The "on-the-fly" version will be generated during test,and\nshould not be committed to the repo.\nThe "storage" version is for back compatibility # test (a model generated today should\nrun on master branch in the next 6 months). We can use this script to update a model that\nis no longer supported.\n- use \'python gen_test_model.py android-test\' to generate on-the-fly models for android\n- use \'python gen_test_model.py ios-test\' to generate on-the-fly models for ios\n- use \'python gen_test_model.py android\' to generate checked-in models for android\n- use \'python gen_test_model.py ios\' to generate on-the-fly models for ios\n- use \'python gen_test_model.py <model_name_no_suffix>\' to update the given storage model\n')
        return
    if argv[0] == 'android':
        generateAllModels(test_path_android, on_the_fly=False)
    elif argv[0] == 'ios':
        generateAllModels(test_path_ios, on_the_fly=False)
    elif argv[0] == 'android-test':
        generateAllModels(test_path_android, on_the_fly=True)
    elif argv[0] == 'ios-test':
        generateAllModels(test_path_ios, on_the_fly=True)
    else:
        generateModel(argv[0])
if __name__ == '__main__':
    main(sys.argv[1:])