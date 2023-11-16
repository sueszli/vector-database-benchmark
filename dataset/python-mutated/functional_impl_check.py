import tempfile
from string import Template
import re
import pprint
import os
import torch
from cpp_api_parity.utils import TorchNNFunctionalTestParams, TORCH_NN_COMMON_TEST_HARNESS, compile_cpp_code_inline, set_python_tensors_requires_grad, move_python_tensors_to_device, add_test, compute_cpp_args_construction_stmts_and_forward_arg_symbols, serialize_arg_dict_as_script_module, compute_arg_dict, decorate_test_fn, compute_temp_file_path, generate_error_msg, is_torch_nn_functional_test, try_remove_folder
from cpp_api_parity.sample_functional import SAMPLE_FUNCTIONAL_CPP_SOURCE
TORCH_NN_FUNCTIONAL_TEST_FORWARD = Template('\nvoid ${functional_variant_name}_test_forward(\n    const std::string& arg_dict_file_path,\n    const std::string& forward_output_file_path) {\n  pybind11::gil_scoped_release no_gil;\n\n  namespace F = torch::nn::functional;\n\n  // Declare arguments\n  auto arg_dict = load_dict_from_file(arg_dict_file_path);\n  ${cpp_args_construction_stmts};\n\n  // Some functionals (such as `F::rrelu`) create random tensors in their call path.\n  // To make sure the random tensors created are the same in Python/C++, we need\n  // to set the RNG seed manually.\n  torch::manual_seed(0);\n\n  // Run function with arguments\n  auto cpp_output = ${cpp_function_call};\n\n  // Save the output into a file to be compared in Python later\n  write_ivalue_to_file(torch::IValue(cpp_output), forward_output_file_path);\n}\n')

def run_forward(unit_test_class, test_params):
    if False:
        while True:
            i = 10
    device = test_params.device
    inputs = set_python_tensors_requires_grad(move_python_tensors_to_device([arg_value for (_, arg_value) in test_params.arg_dict['input']], device))
    inputs += move_python_tensors_to_device([arg_value for (_, arg_value) in test_params.arg_dict['target']], device)
    inputs += move_python_tensors_to_device([arg_value for (_, arg_value) in test_params.arg_dict['extra_args']], device)
    torch.manual_seed(0)
    python_output = test_params.test_instance.constructor()(*inputs)
    return python_output

def test_forward(unit_test_class, test_params):
    if False:
        print('Hello World!')
    functional_variant_name = test_params.functional_variant_name
    cpp_tmp_folder = test_params.cpp_tmp_folder
    try_remove_folder(cpp_tmp_folder)
    os.mkdir(cpp_tmp_folder)
    python_output = run_forward(unit_test_class, test_params)
    arg_dict_file_path = compute_temp_file_path(cpp_tmp_folder, functional_variant_name, 'arg_dict')
    serialize_arg_dict_as_script_module(test_params.arg_dict).save(arg_dict_file_path)
    cpp_test_name = f'{test_params.functional_variant_name}_test_forward'
    cpp_test_fn = getattr(unit_test_class.functional_impl_check_cpp_module, cpp_test_name)

    def run_cpp_test_fn_and_check_output():
        if False:
            while True:
                i = 10
        forward_output_file_path = compute_temp_file_path(cpp_tmp_folder, functional_variant_name, 'forward_output')
        cpp_test_fn(arg_dict_file_path, forward_output_file_path)
        cpp_output = torch.load(forward_output_file_path)
        unit_test_class.assertEqual(python_output, cpp_output, msg=generate_error_msg('forward output', cpp_output, python_output))
    run_cpp_test_fn_and_check_output()
    try_remove_folder(cpp_tmp_folder)

def compute_functional_name(test_params_dict):
    if False:
        i = 10
        return i + 15

    def camel_case_to_snake_case(camel_case_str):
        if False:
            print('Hello World!')
        return re.sub('(?<!^)(?=[A-Z])', '_', camel_case_str).lower()
    if 'cpp_options_args' in test_params_dict:
        return camel_case_to_snake_case(test_params_dict['cpp_options_args'].split('(')[0].replace('F::', '').replace('FuncOptions', ''))
    elif 'cpp_function_call' in test_params_dict:
        return test_params_dict['cpp_function_call'].split('(')[0].replace('F::', '')
    else:
        raise RuntimeError('`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n{}'.format(pprint.pformat(test_params_dict)))

def compute_cpp_function_call(test_params_dict, arg_dict, functional_name):
    if False:
        for i in range(10):
            print('nop')
    if 'cpp_function_call' in test_params_dict:
        return test_params_dict['cpp_function_call']
    elif 'cpp_options_args' in test_params_dict:
        cpp_forward_args_symbols = [arg_name for (arg_name, _) in arg_dict['input'] + arg_dict['target'] + arg_dict['extra_args']]
        return 'F::{}({}, {})'.format(functional_name, ', '.join(cpp_forward_args_symbols), test_params_dict['cpp_options_args'])
    else:
        raise RuntimeError('`cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n{}'.format(pprint.pformat(test_params_dict)))

def process_test_params_for_functional(test_params_dict, device, test_instance_class):
    if False:
        print('Hello World!')
    test_instance = test_instance_class(**test_params_dict)
    functional_name = compute_functional_name(test_params_dict)
    assert test_instance.get_name().startswith('test_')
    functional_variant_name = test_instance.get_name()[5:] + ('_' + device if device != 'cpu' else '')
    arg_dict = compute_arg_dict(test_params_dict, test_instance)
    return TorchNNFunctionalTestParams(functional_name=functional_name, functional_variant_name=functional_variant_name, test_instance=test_instance, cpp_function_call=compute_cpp_function_call(test_params_dict, arg_dict, functional_name), arg_dict=arg_dict, has_parity=test_params_dict.get('has_parity', True), device=device, cpp_tmp_folder=tempfile.mkdtemp())

def write_test_to_test_class(unit_test_class, test_params_dict, test_instance_class, parity_table, devices):
    if False:
        i = 10
        return i + 15
    assert is_torch_nn_functional_test(test_params_dict)
    assert 'cpp_options_args' in test_params_dict or 'cpp_function_call' in test_params_dict, 'To enable C++ API parity test, `cpp_options_args` or `cpp_function_call` entry must be present in test params dict:\n{}. \nIf you are interested in adding the C++ API parity test, please see:\nNOTE [How to check NN module / functional API parity between Python and C++ frontends]. \nIf not, please add `test_cpp_api_parity=False` to the test params dict and file an issue about this.'.format(pprint.pformat(test_params_dict))
    assert not ('cpp_options_args' in test_params_dict and 'cpp_function_call' in test_params_dict), f'Only one of `cpp_options_args` and `cpp_function_call` entries should be present in test params dict:\n{pprint.pformat(test_params_dict)}'
    functional_name = compute_functional_name(test_params_dict)
    assert hasattr(torch.nn.functional, functional_name), "`torch.nn.functional` doesn't have function `{}`. (Discovered while processing\n{}.)".format(functional_name, pprint.pformat(test_params_dict))
    functional_full_name = 'F::' + functional_name
    assert functional_full_name in parity_table['torch::nn::functional'], 'Please add `{}` entry to `torch::nn::functional` section of `test/cpp_api_parity/parity-tracker.md`. (Discovered while processing\n{}.)'.format(functional_full_name, pprint.pformat(test_params_dict))
    for device in devices:
        test_params = process_test_params_for_functional(test_params_dict=test_params_dict, device=device, test_instance_class=test_instance_class)
        try_remove_folder(test_params.cpp_tmp_folder)
        unit_test_name = f'test_torch_nn_functional_{test_params.functional_variant_name}'
        unit_test_class.functional_test_params_map[unit_test_name] = test_params

        def test_fn(self):
            if False:
                while True:
                    i = 10
            test_forward(unit_test_class=self, test_params=unit_test_class.functional_test_params_map[self._testMethodName])
        test_fn = decorate_test_fn(test_fn=test_fn, test_cuda=test_params_dict.get('test_cuda', True), has_impl_parity=parity_table['torch::nn::functional'][functional_full_name][0] and test_params_dict.get('has_parity', True), device=device)
        add_test(unit_test_class, unit_test_name, test_fn)

def generate_test_cpp_sources(test_params, template):
    if False:
        print('Hello World!')
    (cpp_args_construction_stmts, _) = compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params)
    test_cpp_sources = template.substitute(functional_variant_name=test_params.functional_variant_name, cpp_args_construction_stmts=';\n  '.join(cpp_args_construction_stmts), cpp_function_call=test_params.cpp_function_call)
    return test_cpp_sources

def build_cpp_tests(unit_test_class, print_cpp_source=False):
    if False:
        for i in range(10):
            print('nop')
    assert len(unit_test_class.functional_test_params_map) > 0
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_FUNCTIONAL_CPP_SOURCE
    functions = []
    for test_params in unit_test_class.functional_test_params_map.values():
        cpp_sources += generate_test_cpp_sources(test_params=test_params, template=TORCH_NN_FUNCTIONAL_TEST_FORWARD)
        functions.append(f'{test_params.functional_variant_name}_test_forward')
    if print_cpp_source:
        print(cpp_sources)
    cpp_module = compile_cpp_code_inline(name='functional_impl_check', cpp_sources=cpp_sources, functions=functions)
    unit_test_class.functional_impl_check_cpp_module = cpp_module