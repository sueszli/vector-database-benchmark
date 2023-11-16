import argparse
import os
import pathlib
import re
from collections import Counter, namedtuple
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type, Union
import yaml
import torchgen.dest as dest
from torchgen.api.lazy import setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, FileManager, NamespaceHelper
from torchgen.yaml_utils import YamlLoader
from .gen_backend_stubs import error_on_missing_kernels, gen_dispatcher_registrations, gen_dispatchkey_nativefunc_headers, parse_backend_yaml
ParsedExternalYaml = namedtuple('ParsedExternalYaml', ['backend_key', 'autograd_key', 'cpp_namespace', 'backend_indices', 'full_codegen'])

def parse_native_functions_keys(backend_yaml_path: str, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]]) -> Tuple[List[OperatorName], List[Any], List[OperatorName]]:
    if False:
        while True:
            i = 10
    native_functions_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions)}
    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)
    full_codegen = yaml_values.pop('full_codegen', [])
    non_native = yaml_values.pop('non_native', [])
    ir_gen = yaml_values.pop('ir_gen', [])
    assert isinstance(full_codegen, list)
    assert isinstance(non_native, list)
    assert isinstance(ir_gen, list)
    full_codegen_opnames = [OperatorName.parse(name) for name in full_codegen]
    ir_gen_opnames = [OperatorName.parse(name) for name in ir_gen]
    return (full_codegen_opnames, non_native, ir_gen_opnames)

def validate_shape_inference_header(shape_inference_hdr: str, expected_shape_infr_decls: List[str]) -> None:
    if False:
        while True:
            i = 10
    try:
        with open(shape_inference_hdr) as f:
            shape_infr_decls = f.read()
            shape_infr_decl_lines = set(shape_infr_decls.split('\n'))
    except OSError as e:
        raise AssertionError(f'Unable to read from the specified shape_inference_hdr file: {shape_inference_hdr}') from e
    shape_infr_regex = 'compute_shape_(\\w+)'
    actual_shape_infr_name_counts = Counter(re.findall(shape_infr_regex, shape_infr_decls))
    missing_decls = [decl for decl in expected_shape_infr_decls if decl not in shape_infr_decl_lines]
    if missing_decls:
        raise Exception(f'Missing shape inference function.\n\nPlease add declare this function in {shape_inference_hdr}:\n\nand implement it in the corresponding shape_inference.cpp file.\n\n{os.linesep.join(missing_decls)}')

def get_ltc_helper_fns() -> str:
    if False:
        for i in range(10):
            print('nop')
    return "at::Tensor to_meta(const at::Tensor& tensor) {\n  // undefined tensors can't be converted to the meta device, since they don't have sizes/strides\n  if (!tensor.defined()) return tensor;\n  auto out = at::native::empty_strided_meta_symint(tensor.sym_sizes(), tensor.sym_strides(), /*dtype=*/c10::make_optional(tensor.scalar_type()), /*layout=*/c10::make_optional(tensor.layout()), /*device=*/c10::make_optional(c10::Device(c10::kMeta)), /*pin_memory=*/c10::nullopt);\n  // needs to handle wrapped numbers, so dtype promotion works properly.\n  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {\n    out.unsafeGetTensorImpl()->set_wrapped_number(true);\n  }\n  return out;\n}\nc10::optional<at::Tensor> to_meta(const c10::optional<at::Tensor>& tensor) {\n  if (tensor.has_value()) {\n    return to_meta(*tensor);\n  }\n  return c10::nullopt;\n}\n\nstd::vector<at::Tensor> to_meta(at::ITensorListRef t_list) {\n  std::vector<at::Tensor> outs;\n  outs.reserve(t_list.size());\n  for (const auto& tensor : t_list) {\n    outs.push_back(to_meta(tensor));\n  }\n  return outs;\n}\n"

class default_args:
    node_base: str = 'Node'
    node_base_hdr: Optional[str] = None
    shape_inference_hdr: str = 'torch/csrc/lazy/core/shape_inference.h'
    tensor_class: str = 'torch::lazy::LazyTensor'
    tensor_class_hdr: str = 'torch/csrc/lazy/core/tensor.h'
    lazy_ir_generator: Type[GenLazyIR] = GenLazyIR
    native_func_definition_generator: Type[GenLazyNativeFuncDefinition] = GenLazyNativeFuncDefinition
    backend_name: str = 'TorchScript'

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Generate Lazy Tensor backend files')
    parser.add_argument('-s', '--source-yaml', '--source_yaml', help='path to source yaml file containing operator external definitions')
    parser.add_argument('-o', '--output-dir', '--output_dir', help='output directory')
    parser.add_argument('--dry-run', '--dry_run', type=bool, default=False, help='output directory')
    parser.add_argument('--impl-path', '--impl_path', type=str, default=None, help='path to the source C++ file containing kernel definitions')
    parser.add_argument('--gen-ts-lowerings', '--gen_ts_lowerings', action='store_true', help='Generate TorchScript lowerings in addition to Lazy IR and NativeFunctions')
    parser.add_argument('--node-base', '--node_base', type=str, default=default_args.node_base, help='Name of backend specific custom Lazy IR Node base class')
    parser.add_argument('--node-base-hdr', '--node_base_hdr', type=str, default=default_args.node_base_hdr, help='Path to header file defining custom Lazy IR Node base class')
    parser.add_argument('--shape-inference-hdr', '--shape_inference_hdr', type=str, default=default_args.shape_inference_hdr, help='Path to header file defining custom Lazy shape inference functions')
    parser.add_argument('--tensor-class', '--tensor_class', type=str, default=default_args.tensor_class, help='Name of backend specific custom Lazy Tensor class')
    parser.add_argument('--tensor-class-hdr', '--tensor_class_hdr', type=str, default=default_args.tensor_class_hdr, help='Path to header file defining custom Lazy Tensor class')
    parser.add_argument('--backend-name', '--backend_name', type=str, default=default_args.backend_name, help='Name of the backend to generate')
    options = parser.parse_args()
    torch_root = pathlib.Path(__file__).parent.parent.parent.absolute()
    aten_path = str(torch_root / 'aten' / 'src' / 'ATen')
    lazy_ir_generator: Type[GenLazyIR] = default_args.lazy_ir_generator
    if options.gen_ts_lowerings:
        lazy_ir_generator = GenTSLazyIR
    native_func_definition_generator: Type[GenLazyNativeFuncDefinition] = default_args.native_func_definition_generator
    run_gen_lazy_tensor(aten_path, options.source_yaml, options.output_dir, options.dry_run, options.impl_path, options.node_base, options.node_base_hdr, options.tensor_class, options.tensor_class_hdr, options.shape_inference_hdr, lazy_ir_generator, native_func_definition_generator, options.backend_name)

def run_gen_lazy_tensor(aten_path: str, source_yaml: str, output_dir: str, dry_run: bool, impl_path: Optional[str], node_base: str=default_args.node_base, node_base_hdr: Optional[str]=default_args.node_base_hdr, tensor_class: str=default_args.tensor_class, tensor_class_hdr: str=default_args.tensor_class_hdr, shape_inference_hdr: str=default_args.shape_inference_hdr, lazy_ir_generator: Type[GenLazyIR]=default_args.lazy_ir_generator, native_func_definition_generator: Type[GenLazyNativeFuncDefinition]=default_args.native_func_definition_generator, build_in_tree: bool=False, per_operator_headers: bool=False, backend_name: str=default_args.backend_name, gen_forced_fallback_code: bool=False, use_lazy_shape: bool=True, backend_namespace: str='torch::lazy', get_tensorlist: str='GetTensorList', get_tensor_or_wrap_number: str='GetLtcTensorOrCreateForWrappedNumber', try_get_tensor: str='TryGetLtcTensor', metrics_counter: str='TORCH_LAZY_FN_COUNTER("lazy::")', create_tensor: str='LazyTensor::Create', create_from_first_tensor: bool=False, create_aten_from_ltc_tensor: str='torch::lazy::CreateAtenFromLtcTensor', tuple_aten_from_ltc_tensors: str='torch::lazy::TupleAtenFromLtcTensors', lazy_value_class: str='torch::lazy::Value', lazy_tensor_ptr: str='LazyTensorPtr', get_device_fn: str='torch::lazy::GetBackendDevice') -> None:
    if False:
        i = 10
        return i + 15
    lv_tokens = lazy_value_class.split('::')
    lv_class = lv_tokens[-1]
    lv_ns = '::'.join(lv_tokens[:-1])
    setValueT(BaseCppType(lv_ns, lv_class))
    template_dir = os.path.join(aten_path, 'templates')

    def make_file_manager(install_dir: str) -> FileManager:
        if False:
            i = 10
            return i + 15
        return FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=dry_run)
    fm = make_file_manager(output_dir)
    native_yaml_path = os.path.join(aten_path, 'native/native_functions.yaml')
    tags_yaml_path = os.path.join(aten_path, 'native/tags.yaml')
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    (native_functions, backend_indices) = (parsed_yaml.native_functions, parsed_yaml.backend_indices)
    grouped_native_functions = get_grouped_native_functions(native_functions)

    def sort_native_function(f: Union[NativeFunctionsGroup, NativeFunction]) -> str:
        if False:
            return 10
        '\n        We sort the native function because of the note in concat_map_codegen.\n        TODO(alanwaketan): Remove this sorting hack once all ops are grouped properly.\n        '
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        return str(func.name.name)
    grouped_native_functions = sorted(grouped_native_functions, key=sort_native_function)
    parsed_backend_yaml = parse_backend_yaml(source_yaml, grouped_native_functions, backend_indices)
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    backend_indices = parsed_backend_yaml.backend_indices
    (full_codegen, non_native, ir_gen) = parse_native_functions_keys(source_yaml, grouped_native_functions)

    def concat_map_codegen(func: Callable[[NativeFunction], Sequence[str]], xs: Iterable[Union[NativeFunctionsGroup, NativeFunction]], ops_list: List[OperatorName]=full_codegen) -> Iterator[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        We code-gen for the functional variant, which is all we need for IR classes/lowerings/shape inferences, but we\n        only code-gen additional entries for the inplace variant for the native functions.\n        '
        for x in xs:
            fs = list(x.functions()) if isinstance(x, NativeFunctionsGroup) else [x]
            for f in fs:
                if f.func.name in ops_list:
                    yield from func(f)
    selector = SelectiveBuilder.get_nop_selector()
    assert backend_key is not None
    class_name = backend_indices[backend_key].native_function_class_name()
    if impl_path is not None:
        error_on_missing_kernels(native_functions, backend_indices, backend_key, autograd_key, class_name, impl_path, full_codegen)
    " Validate Shape Inference Definitions\n\n    Generated lazy native functions all perform shape inference, by first using a meta:: kernel\n    if available for that op, and otherwise using a 'compute_shape_{op}' function instead.  The generator\n    knows the call signature for compute_shape_{op} because it matches the nativefunction (and meta::) signature,\n    so it just has to check whether the op is structured and generate a call for one or the other.  It's up to the dev\n    to supply the missing compute_shape_{op} function, but the codegen at least warns you about this and provides\n    the expected signature which can be copy-pasted into shape_inference.h.\n\n    compute_shape_{op} functions are handwritten and should be replaced over time as ops get ported\n    to structured kernels.\n\n    See torch/csrc/lazy/core/shape_inference.cpp #READ THIS! for more information.\n    "
    if shape_inference_hdr is not None:
        expected_shape_infr_decls = list(concat_map_codegen(dest.GenLazyShapeInferenceDefinition(backend_indices[backend_key], tensor_class), grouped_native_functions))
        validate_shape_inference_header(shape_inference_hdr, expected_shape_infr_decls)
    assert class_name is not None
    gen_dispatchkey_nativefunc_headers(fm, class_name, cpp_namespace, backend_indices, grouped_native_functions, backend_key, autograd_key, backend_name)
    for dispatch_key in [backend_key] if autograd_key is None else [backend_key, autograd_key]:
        gen_dispatcher_registrations(fm, output_dir, class_name, backend_indices, grouped_native_functions, backend_key, dispatch_key, selector, build_in_tree=build_in_tree, per_operator_headers=per_operator_headers, backend_name=backend_name, eager_registration=False)
    ns_helper = NamespaceHelper(cpp_namespace)
    fm.write_with_template(f'{backend_key}NativeFunctions.cpp', 'DispatchKeyNativeFunctions.cpp', lambda : {'includes': [f'#include <{path}>' for path in [tensor_class_hdr, shape_inference_hdr, 'ATen/Functions.h', 'ATen/native/TensorConversions.h', 'ATen/NativeFunctions.h', 'ATen/CompositeExplicitAutogradNonFunctionalFunctions.h', 'ATen/MetaFunctions.h', 'ATen/Operators.h', 'ATen/native/CPUFallback.h', 'torch/csrc/lazy/core/ir_builder.h', 'torch/csrc/lazy/core/lazy_graph_executor.h', 'torch/csrc/lazy/core/metrics.h', 'torch/csrc/lazy/core/shape.h', f'{output_dir}/{backend_key}NativeFunctions.h', f'{output_dir}/LazyIr.h'] + (['torch/csrc/lazy/ts_backend/ts_eager_fallback.h'] if gen_forced_fallback_code else [])], 'helper_fns': get_ltc_helper_fns(), 'native_functions_include': '', 'namespace_prologue': ns_helper.prologue, 'namespace_epilogue': ns_helper.epilogue, 'native_function_definitions': list(concat_map_codegen(native_func_definition_generator(f'{backend_key}NativeFunctions', backend_indices[backend_key], tensor_class, gen_forced_fallback_code, backend_namespace, get_tensorlist, get_tensor_or_wrap_number, try_get_tensor, metrics_counter, create_tensor, create_from_first_tensor, create_aten_from_ltc_tensor, tuple_aten_from_ltc_tensors, lazy_tensor_ptr, get_device_fn), grouped_native_functions))})
    lazy_ir_obj = lazy_ir_generator(backend_indices[backend_key], backend_name, node_base, use_lazy_shape)
    fm.write_with_template('LazyIr.h', 'LazyIr.h', lambda : {'lazy_ir_sysinc': [f'#include <{path}>' for path in ['ATen/core/Formatting.h', 'c10/core/ScalarType.h', 'c10/util/Optional.h', 'torch/csrc/lazy/core/hash.h', 'torch/csrc/lazy/core/ir.h', 'torch/csrc/lazy/core/shape.h', 'vector']], 'lazy_ir_inc': [f'#include "{node_base_hdr}"'] if node_base_hdr is not None else [], 'ir_declarations': list(concat_map_codegen(lazy_ir_obj, grouped_native_functions, full_codegen + ir_gen)), 'namespace_prologue': ns_helper.prologue, 'namespace_epilogue': ns_helper.epilogue})
    fm.write_with_template('LazyNonNativeIr.h', 'LazyNonNativeIr.h', lambda : {'lazy_non_native_ir_inc': [f'#include <{path}>' for path in ['torch/csrc/lazy/core/ir.h', 'torch/csrc/lazy/core/ir_builder.h', 'torch/csrc/lazy/core/internal_ops/ltc_ops.h', 'torch/csrc/lazy/core/shape_inference.h'] + ([node_base_hdr] if node_base_hdr else []) if path], 'non_native_ir_nodes': dest.generate_non_native_lazy_ir_nodes(non_native, lazy_ir_obj), 'namespace_prologue': ns_helper.prologue, 'namespace_epilogue': ns_helper.epilogue})
if __name__ == '__main__':
    main()