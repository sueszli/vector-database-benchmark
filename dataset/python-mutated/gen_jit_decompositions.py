import os
from pathlib import Path
from torch.jit._decompositions import decomposition_table
DECOMP_HEADER = '\n/**\n * @generated\n * This is an auto-generated file. Please do not modify it by hand.\n * To re-generate, please run:\n * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py\n */\n#include <torch/csrc/jit/jit_log.h>\n#include <torch/csrc/jit/passes/inliner.h>\n#include <torch/csrc/jit/runtime/operator.h>\n#include <torch/csrc/jit/runtime/decomposition_registry_util.h>\n\nnamespace torch {\nnamespace jit {\n\n\nconst std::string decomp_funcs =\nR"('
DECOMP_CENTER = '\n)";\n\nconst std::string& GetSerializedDecompositions() {\n  return decomp_funcs;\n}\n\nconst OperatorMap<std::string>& GetDecompositionMapping() {\n  // clang-format off\n static const OperatorMap<std::string> decomposition_mapping {\n'
DECOMP_END = '\n  };\n  // clang-format on\n\n  return decomposition_mapping;\n}\n\n} // namespace jit\n} // namespace torch\n'
DECOMPOSITION_UTIL_FILE_NAME = 'decomposition_registry_util.cpp'

def gen_serialized_decompisitions() -> str:
    if False:
        for i in range(10):
            print('nop')
    return '\n'.join([scripted_func.code for scripted_func in decomposition_table.values()])

def gen_decomposition_mappings() -> str:
    if False:
        while True:
            i = 10
    decomposition_mappings = []
    for (schema, scripted_func) in decomposition_table.items():
        decomposition_mappings.append('    {"' + schema + '", "' + scripted_func.name + '"},')
    return '\n'.join(decomposition_mappings)

def write_decomposition_util_file(path: str) -> None:
    if False:
        return 10
    decomposition_str = gen_serialized_decompisitions()
    decomposition_mappings = gen_decomposition_mappings()
    file_components = [DECOMP_HEADER, decomposition_str, DECOMP_CENTER, decomposition_mappings, DECOMP_END]
    print('writing file to : ', path + '/' + DECOMPOSITION_UTIL_FILE_NAME)
    with open(os.path.join(path, DECOMPOSITION_UTIL_FILE_NAME), 'wb') as out_file:
        final_output = ''.join(file_components)
        out_file.write(final_output.encode('utf-8'))

def main() -> None:
    if False:
        i = 10
        return i + 15
    pytorch_dir = Path(__file__).resolve().parents[3]
    upgrader_path = pytorch_dir / 'torch' / 'csrc' / 'jit' / 'runtime'
    write_decomposition_util_file(str(upgrader_path))
if __name__ == '__main__':
    main()