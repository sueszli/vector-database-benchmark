import importlib.util
import os
import sys
from itertools import chain
from pathlib import Path
file_path = Path.cwd() / 'torch' / 'jit' / '_shape_functions.py'
module_name = 'torch.jit._shape_functions'
err_msg = 'Could not find shape functions file, please make sure\nyou are in the root directory of the Pytorch git repo'
if not file_path.exists():
    raise Exception(err_msg)
spec = importlib.util.spec_from_file_location(module_name, file_path)
assert spec is not None
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
assert spec.loader is not None
assert module is not None
spec.loader.exec_module(module)
bounded_compute_graph_mapping = module.bounded_compute_graph_mapping
shape_compute_graph_mapping = module.shape_compute_graph_mapping
SHAPE_HEADER = '\n/**\n * @generated\n * This is an auto-generated file. Please do not modify it by hand.\n * To re-generate, please run:\n * cd ~/pytorch && python\n * torchgen/shape_functions/gen_jit_shape_functions.py\n */\n#include <torch/csrc/jit/jit_log.h>\n#include <torch/csrc/jit/passes/inliner.h>\n#include <torch/csrc/jit/runtime/operator.h>\n#include <torch/csrc/jit/runtime/serialized_shape_function_registry.h>\n\n// clang-format off\n\nnamespace torch {\nnamespace jit {\n\n\nstd::string shape_funcs = ""\n'
DECOMP_CENTER = '\n\n\nconst std::string& GetSerializedShapeFunctions() {\n  return shape_funcs;\n}\n\n'
DECOMP_END = '\n// clang-format on\n\n} // namespace jit\n} // namespace torch\n'
SERIALIZED_SHAPE_UTIL_FILE_NAME = 'serialized_shape_function_registry.cpp'

def gen_serialized_decompisitions() -> str:
    if False:
        print('Hello World!')
    already_serialized_names = set()
    unique_funcs = []
    all_funcs = chain(shape_compute_graph_mapping.values(), *bounded_compute_graph_mapping.values())
    for scripted_func in all_funcs:
        if scripted_func.name in already_serialized_names:
            continue
        already_serialized_names.add(scripted_func.name)
        unique_funcs.append(scripted_func)
    output_strs = []
    curr_str = ''
    for scripted_func in unique_funcs:
        serialized_code = scripted_func.code
        MAX_MSFT_STR_LEN = 2000
        if len(curr_str) + len(serialized_code) <= MAX_MSFT_STR_LEN:
            curr_str += '\n' + serialized_code
        else:
            output_strs.append(curr_str)
            curr_str = scripted_func.code
    output_strs.append(curr_str)
    final_output = ''
    for output_str in output_strs:
        start = '+ std::string(R"=====('
        end = '\n)=====")\n'
        final_output += start + output_str + end
    final_output += ';'
    return final_output
SHAPE_SCHEMA_START = '\nconst OperatorMap<std::string>& GetShapeFunctionMappings() {\n static const OperatorMap<std::string> shape_mappings {\n'
SHAPE_SCHEMA_END = '\n  };\n\n  return shape_mappings;\n}\n'

def gen_shape_mappings() -> str:
    if False:
        while True:
            i = 10
    shape_mappings = []
    for (schema, scripted_func) in shape_compute_graph_mapping.items():
        shape_mappings.append('    {"' + schema + '", "' + scripted_func.name + '"},')
    return SHAPE_SCHEMA_START + '\n'.join(shape_mappings) + SHAPE_SCHEMA_END
BOUNDED_SCHEMA_START = '\nconst OperatorMap<std::pair<std::string, std::string>>& GetBoundedShapeMappings() {\n static const OperatorMap<std::pair<std::string, std::string>> shape_mappings {\n'

def gen_bounded_mappings() -> str:
    if False:
        for i in range(10):
            print('nop')
    bounded_mappings = []
    for (schema, (lower_func, upper_func)) in bounded_compute_graph_mapping.items():
        map_str = '    {"' + schema + '", {"' + lower_func.name + '", "' + upper_func.name + '"}},'
        bounded_mappings.append(map_str)
    return BOUNDED_SCHEMA_START + '\n'.join(bounded_mappings) + SHAPE_SCHEMA_END

def write_decomposition_util_file(path: str) -> None:
    if False:
        return 10
    decomposition_str = gen_serialized_decompisitions()
    shape_mappings = gen_shape_mappings()
    bounded_mappings = gen_bounded_mappings()
    file_components = [SHAPE_HEADER, decomposition_str, DECOMP_CENTER, shape_mappings, bounded_mappings, DECOMP_END]
    print('writing file to : ', path + '/' + SERIALIZED_SHAPE_UTIL_FILE_NAME)
    with open(os.path.join(path, SERIALIZED_SHAPE_UTIL_FILE_NAME), 'wb') as out_file:
        final_output = ''.join(file_components)
        out_file.write(final_output.encode('utf-8'))

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    pytorch_dir = Path(__file__).resolve().parents[2]
    upgrader_path = pytorch_dir / 'torch' / 'csrc' / 'jit' / 'runtime'
    write_decomposition_util_file(str(upgrader_path))
if __name__ == '__main__':
    main()