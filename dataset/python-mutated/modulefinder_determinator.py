import modulefinder
import os
import pathlib
import sys
import warnings
from typing import Any, Dict, List, Set
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TARGET_DET_LIST = ['test_binary_ufuncs', 'test_cpp_extensions_aot_ninja', 'test_cpp_extensions_aot_no_ninja', 'test_cpp_extensions_jit', 'test_cpp_extensions_open_device_registration', 'test_cuda', 'test_cuda_primary_ctx', 'test_dataloader', 'test_determination', 'test_futures', 'test_jit', 'test_jit_legacy', 'test_jit_profiling', 'test_linalg', 'test_multiprocessing', 'test_nn', 'test_numpy_interop', 'test_optim', 'test_overrides', 'test_pruning_op', 'test_quantization', 'test_reductions', 'test_serialization', 'test_shape_ops', 'test_sort_and_select', 'test_tensorboard', 'test_testing', 'test_torch', 'test_utils', 'test_view_ops']
_DEP_MODULES_CACHE: Dict[str, Set[str]] = {}

def should_run_test(target_det_list: List[str], test: str, touched_files: List[str], options: Any) -> bool:
    if False:
        print('Hello World!')
    test = parse_test_module(test)
    if test not in target_det_list:
        if options.verbose:
            print_to_stderr(f'Running {test} without determination')
        return True
    if test.endswith('_no_ninja'):
        test = test[:-1 * len('_no_ninja')]
    if test.endswith('_ninja'):
        test = test[:-1 * len('_ninja')]
    dep_modules = get_dep_modules(test)
    for touched_file in touched_files:
        file_type = test_impact_of_file(touched_file)
        if file_type == 'NONE':
            continue
        elif file_type == 'CI':
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type == 'UNKNOWN':
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type in ['TORCH', 'CAFFE2', 'TEST']:
            parts = os.path.splitext(touched_file)[0].split(os.sep)
            touched_module = '.'.join(parts)
            if touched_module.startswith('test.'):
                touched_module = touched_module.split('test.')[1]
            if touched_module in dep_modules or touched_module == test.replace('/', '.'):
                log_test_reason(file_type, touched_file, test, options)
                return True
    if options.verbose:
        print_to_stderr(f'Determination is skipping {test}')
    return False

def test_impact_of_file(filename: str) -> str:
    if False:
        print('Hello World!')
    'Determine what class of impact this file has on test runs.\n\n    Possible values:\n        TORCH - torch python code\n        CAFFE2 - caffe2 python code\n        TEST - torch test code\n        UNKNOWN - may affect all tests\n        NONE - known to have no effect on test outcome\n        CI - CI configuration files\n    '
    parts = filename.split(os.sep)
    if parts[0] in ['.jenkins', '.circleci', '.ci']:
        return 'CI'
    if parts[0] in ['docs', 'scripts', 'CODEOWNERS', 'README.md']:
        return 'NONE'
    elif parts[0] == 'torch':
        if parts[-1].endswith('.py') or parts[-1].endswith('.pyi'):
            return 'TORCH'
    elif parts[0] == 'caffe2':
        if parts[-1].endswith('.py') or parts[-1].endswith('.pyi'):
            return 'CAFFE2'
    elif parts[0] == 'test':
        if parts[-1].endswith('.py') or parts[-1].endswith('.pyi'):
            return 'TEST'
    return 'UNKNOWN'

def log_test_reason(file_type: str, filename: str, test: str, options: Any) -> None:
    if False:
        while True:
            i = 10
    if options.verbose:
        print_to_stderr(f'Determination found {file_type} file {filename} -- running {test}')

def get_dep_modules(test: str) -> Set[str]:
    if False:
        print('Hello World!')
    if test in _DEP_MODULES_CACHE:
        return _DEP_MODULES_CACHE[test]
    test_location = REPO_ROOT / 'test' / f'{test}.py'
    finder = modulefinder.ModuleFinder(excludes=['scipy', 'numpy', 'numba', 'multiprocessing', 'sklearn', 'setuptools', 'hypothesis', 'llvmlite', 'joblib', 'email', 'importlib', 'unittest', 'urllib', 'json', 'collections', 'mpl_toolkits', 'google', 'onnx', 'mypy'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        finder.run_script(str(test_location))
    dep_modules = set(finder.modules.keys())
    _DEP_MODULES_CACHE[test] = dep_modules
    return dep_modules

def parse_test_module(test: str) -> str:
    if False:
        print('Hello World!')
    return test.split('.')[0]

def print_to_stderr(message: str) -> None:
    if False:
        i = 10
        return i + 15
    print(message, file=sys.stderr)