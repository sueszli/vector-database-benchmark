import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import arg_parser_output_exprs, cpp_dispatch_exprs, cpp_dispatch_target, dispatch_lambda_args, dispatch_lambda_exprs, dispatch_lambda_return_str, has_tensor_options, namedtuple_fieldnames, PythonSignature, PythonSignatureDeprecated, PythonSignatureGroup, PythonSignatureNativeFunctionPair, signature, signature_from_schema
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import Argument, BaseOperatorName, FunctionSchema, NativeFunction, Type, Variant
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
_SKIP_PYTHON_BINDINGS = ['alias', 'contiguous', 'is_cuda', 'is_sparse', 'is_sparse_csr', 'size', 'stride', 'sym_size', 'sym_stride', 'sym_storage_offset', 'sym_numel', '.*_backward', '.*_backward_(out|input|weight|bias)', '.*_forward', '.*_forward_out', '.*_jvp', '_unsafe_view', 'tensor', '_?sparse_(coo|compressed|csr|csc|bsr|bsc)_tensor.*', '_range.*', '_sparse_add_out', '_sparse_div.*', '_sparse_mul.*', '_sparse_sub.*', '_sparse_dense_add_out', 'index', 'index_out', 'unique_dim_consecutive', '_cumsum.*', '_cumprod.*', '_sum.*', '_prod.*', '_th_.*', '_thnn_.*', 'range.*', '_solve.*', '_inverse.*', '_cholesky.*', '_triangular_solve.*', '_qr.*', '_svd.*', 'slice', 'item', '_local_scalar_dense', 'to', '_to_copy', '_to_copy_out', '_reshape_copy', '_reshape_copy_out', 'copy_sparse_to_sparse_', 'copy_', 'numpy_T', 'matrix_H', 'mT', 'mH', 'nonzero(_(out|numpy))?', 'set_data', '.*_overrideable', 'data', 'is_leaf', 'output_nr', '_version', 'requires_grad_', 'retains_grad', 'set_', '_fw_primal', 'fake_quantize_per_tensor_affine_cachemask', 'fake_quantize_per_channel_affine_cachemask', '_new_zeros_with_same_feature_meta', '_has_same_storage_numel', '_reshape_alias', 'replace_', 'copy', 'fill.Tensor', 'fill.Scalar', 'lift.*', 'normal_functional', 'nbytes', 'itemsize']
SKIP_PYTHON_BINDINGS = [re.compile(f'^{pattern}$') for pattern in _SKIP_PYTHON_BINDINGS]
SKIP_PYTHON_BINDINGS_SIGNATURES = ['add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor', 'add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)', 'sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor', 'sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)', 'mul.Scalar(Tensor self, Scalar other) -> Tensor', 'mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)', 'div.Scalar(Tensor self, Scalar other) -> Tensor', 'div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)']

@with_native_function
def should_generate_py_binding(f: NativeFunction) -> bool:
    if False:
        return 10
    if 'generated' in f.tags and 'view_copy' not in f.tags:
        return False
    name = cpp.name(f.func)
    for skip_regex in SKIP_PYTHON_BINDINGS:
        if skip_regex.match(name):
            return False
    signature = str(f.func)
    for pattern in SKIP_PYTHON_BINDINGS_SIGNATURES:
        if pattern == signature:
            return False
    return True

def get_pycname(name: BaseOperatorName) -> str:
    if False:
        while True:
            i = 10
    return f'THPVariable_{name}'

def is_noarg(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> bool:
    if False:
        return 10
    return len(overloads) == 1 and overloads[0].signature.arguments_count() == 0

def is_py_variable_method(f: NativeFunction) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return f.python_module is None and Variant.method in f.variants

def is_py_torch_function(f: NativeFunction) -> bool:
    if False:
        i = 10
        return i + 15
    return f.python_module is None and Variant.function in f.variants

def is_py_nn_function(f: NativeFunction) -> bool:
    if False:
        print('Hello World!')
    return f.python_module == 'nn'

def is_py_fft_function(f: NativeFunction) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return f.python_module == 'fft'

def is_py_linalg_function(f: NativeFunction) -> bool:
    if False:
        while True:
            i = 10
    return f.python_module == 'linalg'

def is_py_nested_function(f: NativeFunction) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return f.python_module == 'nested'

def is_py_sparse_function(f: NativeFunction) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return f.python_module == 'sparse'

def is_py_special_function(f: NativeFunction) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return f.python_module == 'special'

def gen(out: str, native_yaml_path: str, tags_yaml_path: str, deprecated_yaml_path: str, template_path: str, *, symint: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    native_functions = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
    native_functions = list(filter(should_generate_py_binding, native_functions))
    methods = load_signatures(native_functions, deprecated_yaml_path, method=True)
    create_python_bindings(fm, methods, is_py_variable_method, None, 'python_variable_methods.cpp', method=True, symint=symint)
    functions = load_signatures(native_functions, deprecated_yaml_path, method=False)
    create_python_bindings_sharded(fm, functions, is_py_torch_function, 'torch', 'python_torch_functions.cpp', method=False, num_shards=3, symint=symint)
    create_python_bindings(fm, functions, is_py_nn_function, 'torch.nn', 'python_nn_functions.cpp', method=False, symint=symint)
    create_python_bindings(fm, functions, is_py_fft_function, 'torch.fft', 'python_fft_functions.cpp', method=False, symint=symint)
    create_python_bindings(fm, functions, is_py_linalg_function, 'torch.linalg', 'python_linalg_functions.cpp', method=False, symint=symint)
    create_python_bindings(fm, functions, is_py_nested_function, 'torch.nested', 'python_nested_functions.cpp', method=False)
    create_python_bindings(fm, functions, is_py_sparse_function, 'torch.sparse', 'python_sparse_functions.cpp', method=False, symint=symint)
    create_python_bindings(fm, functions, is_py_special_function, 'torch.special', 'python_special_functions.cpp', method=False, symint=symint)
    create_python_return_type_bindings(fm, functions, lambda fn: True, 'python_return_types.cpp')
    create_python_return_type_bindings_header(fm, functions, lambda fn: True, 'python_return_types.h')
    valid_tags = parse_tags_yaml(tags_yaml_path)

    def gen_tags_enum() -> Dict[str, str]:
        if False:
            return 10
        return {'enum_of_valid_tags': ''.join([f'\n.value("{tag}", at::Tag::{tag})' for tag in sorted(valid_tags)])}
    fm.write('python_enum_tag.cpp', gen_tags_enum)

def group_filter_overloads(pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool]) -> Dict[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]:
    if False:
        for i in range(10):
            print('nop')
    grouped: Dict[BaseOperatorName, List[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        if pred(pair.function):
            grouped[pair.function.func.name.name].append(pair)
    return grouped

def create_python_bindings(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], module: Optional[str], filename: str, *, method: bool, symint: bool=True) -> None:
    if False:
        print('Hello World!')
    'Generates Python bindings to ATen functions'
    py_methods: List[str] = []
    ops_headers: List[str] = []
    py_method_defs: List[str] = []
    py_forwards: List[str] = []
    grouped = group_filter_overloads(pairs, pred)
    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        py_methods.append(method_impl(name, module, overloads, method=method, symint=symint))
        py_method_defs.append(method_def(name, module, overloads, method=method))
        py_forwards.extend(forward_decls(name, overloads, method=method))
        ops_headers.append(f'#include <ATen/ops/{name.base}.h>')
    fm.write_with_template(filename, filename, lambda : {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}', 'ops_headers': ops_headers, 'py_forwards': py_forwards, 'py_methods': py_methods, 'py_method_defs': py_method_defs})

def create_python_return_type_bindings(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], filename: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Generate function to initialize and return named tuple for native functions\n    which returns named tuple and registration invocations in `python_return_types.cpp`.\n    '
    py_return_types_definition: List[str] = []
    py_return_types_registrations: List[str] = []
    grouped = group_filter_overloads(pairs, pred)
    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        (definitions, registrations) = generate_return_type_definition_and_registrations(overloads)
        py_return_types_definition.append('' if not definitions else '\n'.join(definitions))
        py_return_types_registrations.append('' if not registrations else '\n'.join(registrations))
    fm.write_with_template(filename, filename, lambda : {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}', 'py_return_types': py_return_types_definition, 'py_return_types_registrations': py_return_types_registrations})

def create_python_return_type_bindings_header(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], filename: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Generate function to initialize and return named tuple for native functions\n    which returns named tuple and relevant entry for the map in `python_return_types.cpp`.\n    '
    py_return_types_declarations: List[str] = []
    grouped = group_filter_overloads(pairs, pred)
    for name in sorted(grouped.keys(), key=str):
        overloads = grouped[name]
        declarations = generate_return_type_declarations(overloads)
        py_return_types_declarations.append('' if not declarations else '\n'.join(declarations))
    fm.write_with_template(filename, filename, lambda : {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}', 'py_return_types_declarations': py_return_types_declarations})

def create_python_bindings_sharded(fm: FileManager, pairs: Sequence[PythonSignatureNativeFunctionPair], pred: Callable[[NativeFunction], bool], module: Optional[str], filename: str, *, method: bool, num_shards: int, symint: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Generates Python bindings to ATen functions'
    grouped = group_filter_overloads(pairs, pred)

    def key_func(kv: Tuple[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return kv[0].base

    def env_func(kv: Tuple[BaseOperatorName, List[PythonSignatureNativeFunctionPair]]) -> Dict[str, List[str]]:
        if False:
            for i in range(10):
                print('nop')
        (name, fn_pairs) = kv
        return {'ops_headers': [f'#include <ATen/ops/{name.base}.h>'], 'py_forwards': list(forward_decls(name, fn_pairs, method=method)), 'py_methods': [method_impl(name, module, fn_pairs, method=method, symint=symint)], 'py_method_defs': [method_def(name, module, fn_pairs, method=method)]}
    fm.write_sharded(filename, grouped.items(), base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/{filename}'}, key_fn=key_func, env_callable=env_func, num_shards=num_shards, sharded_keys={'ops_headers', 'py_forwards', 'py_methods', 'py_method_defs'})

def load_signatures(native_functions: List[NativeFunction], deprecated_yaml_path: str, *, method: bool, skip_deprecated: bool=False, pyi: bool=False) -> Sequence[PythonSignatureNativeFunctionPair]:
    if False:
        i = 10
        return i + 15

    @with_native_function
    def gen_signature_pairs(f: NativeFunction) -> PythonSignatureNativeFunctionPair:
        if False:
            return 10
        return PythonSignatureNativeFunctionPair(signature=signature(f, method=method, pyi=pyi), function=f)
    pairs = list(map(gen_signature_pairs, native_functions))
    deprecated = load_deprecated_signatures(pairs, deprecated_yaml_path, method=method, pyi=pyi)
    return pairs if skip_deprecated else pairs + deprecated

def load_deprecated_signatures(pairs: Sequence[PythonSignatureNativeFunctionPair], deprecated_yaml_path: str, *, method: bool, pyi: bool) -> List[PythonSignatureNativeFunctionPair]:
    if False:
        for i in range(10):
            print('nop')
    grouped: Dict[str, List[PythonSignatureNativeFunctionPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.signature.name].append(pair)
    results: List[PythonSignatureNativeFunctionPair] = []
    with open(deprecated_yaml_path) as f:
        deprecated_defs = yaml.load(f, Loader=YamlLoader)
    for deprecated in deprecated_defs:
        schema = FunctionSchema.parse(deprecated['name'])
        (aten_name, call_args) = split_name_params(deprecated['aten'])
        is_out = aten_name.endswith('_out')
        if is_out:
            aten_name = aten_name.replace('_out', '')
        known_constants = {'1': Type.parse('Scalar')}
        schema_args_by_name = {a.name: a for a in schema.arguments.flat_all}
        for name in call_args:
            assert name in schema_args_by_name or name in known_constants, f'deprecation definiton: Unrecognized value {name}'

        def is_schema_compatible(aten_schema: FunctionSchema) -> bool:
            if False:
                i = 10
                return i + 15
            arguments: Iterable[Argument]
            if is_out:
                arguments = itertools.chain(aten_schema.arguments.out, aten_schema.arguments.flat_non_out)
            else:
                arguments = aten_schema.arguments.flat_all
            for (i, arg) in enumerate(arguments):
                if i < len(call_args):
                    arg_name = call_args[i]
                    if arg_name in known_constants:
                        schema_type = known_constants[arg_name]
                        schema_annotation = None
                    else:
                        schema_arg = schema_args_by_name[arg_name]
                        schema_type = schema_arg.type
                        schema_annotation = schema_arg.annotation
                    if schema_type != arg.type or schema_annotation != arg.annotation:
                        return False
                elif arg.default is None:
                    return False
            return len(schema.returns) == len(aten_schema.returns) and all((a == b for (a, b) in zip(schema.returns, aten_schema.returns)))
        any_schema_found = False
        for pair in grouped[aten_name]:
            if not is_schema_compatible(pair.function.func):
                continue
            any_schema_found = True
            python_sig = signature_from_schema(schema, category_override=pair.function.category_override, method=method, pyi=pyi)
            results.append(PythonSignatureNativeFunctionPair(signature=PythonSignatureDeprecated(name=python_sig.name, input_args=python_sig.input_args, input_kwargs=python_sig.input_kwargs, output_args=python_sig.output_args, tensor_options_args=python_sig.tensor_options_args, method=python_sig.method, deprecated_schema=schema, deprecated_args_exprs=tuple(call_args), returns=python_sig.returns), function=pair.function))
        assert any_schema_found, f'No native function with name {aten_name} matched signature:\n  {str(schema)}'
    return results

@with_native_function
def gen_namedtuple_typename_key(f: NativeFunction) -> str:
    if False:
        return 10
    name = cpp.name(f.func)
    fieldnames = namedtuple_fieldnames(f.func.returns)
    return '_'.join([name] + fieldnames)

def emit_namedtuple_call(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> Tuple[List[str], Dict[str, str]]:
    if False:
        i = 10
        return i + 15
    '\n    Generate block of named tuple type def inits, and add typeref snippets\n    to declarations that use them\n    '
    typenames: Dict[str, str] = {}
    typedefs: List[str] = []
    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue
        name = cpp.name(overload.function.func)
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f"NamedTuple{('' if not typedefs else len(typedefs))}"
            typenames[tn_key] = typename
            typedefs.append(f'static PyTypeObject* {typename} = generated::get_{name}_namedtuple();')
    return (typedefs, typenames)

def generate_return_type_definition_and_registrations(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> Tuple[List[str], List[str]]:
    if False:
        i = 10
        return i + 15
    '\n    Generate block of function in `python_return_types.cpp` to initialize\n    and return named tuple for a native function which returns named tuple\n    and registration invocations in same file.\n    '
    typenames: Dict[str, str] = {}
    definitions: List[str] = []
    registrations: List[str] = []
    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue
        fields = ', '.join((f'{{"{fn}", ""}}' for fn in fieldnames))
        name = cpp.name(overload.function.func)
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f"{name}NamedTuple{('' if not definitions else len(definitions))}"
            typenames[tn_key] = typename
            definitions.append(f'PyTypeObject* get_{name}_namedtuple() {{\n    static PyStructSequence_Field NamedTuple_fields[] = {{ {fields},  {{nullptr}} }};\n    static PyTypeObject {typename};\n    static bool is_initialized = false;\n    static PyStructSequence_Desc desc = {{ "torch.return_types.{name}", nullptr, NamedTuple_fields, {len(fieldnames)} }};\n    if (!is_initialized) {{\n        PyStructSequence_InitType(&{typename}, &desc);\n        {typename}.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;\n        is_initialized = true;\n    }}\n    return &{typename};\n}}\n')
            registrations.append(f'addReturnType(return_types_module, "{name}", generated::get_{name}_namedtuple());')
    return (definitions, registrations)

def generate_return_type_declarations(overloads: Sequence[PythonSignatureNativeFunctionPair]) -> List[str]:
    if False:
        print('Hello World!')
    '\n    Generate block of function declarations in `python_return_types.h` to initialize\n    and return named tuple for a native function.\n    '
    typenames: Dict[str, str] = {}
    declarations: List[str] = []
    for overload in overloads:
        fieldnames = namedtuple_fieldnames(overload.function.func.returns)
        if not fieldnames:
            continue
        name = cpp.name(overload.function.func)
        tn_key = gen_namedtuple_typename_key(overload.function)
        typename = typenames.get(tn_key)
        if typename is None:
            typename = f"{name}NamedTuple{('' if not declarations else len(declarations))}"
            typenames[tn_key] = typename
            declarations.append(f'PyTypeObject* get_{name}_namedtuple();')
    return declarations
PY_VARIABLE_METHOD_VARARGS = CodeTemplate('\\\n// ${name}\nstatic PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)\n{\n  ${method_header}\n  static PythonArgParser parser({\n    ${signatures}\n  }, /*traceable=*/${traceable});\n\n  ParsedArgs<${max_args}> parsed_args;\n  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);\n  ${check_has_torch_function}\n  switch (_r.idx) {\n    ${dispatch}\n  }\n  ${method_footer}\n}\n\n')
PY_VARIABLE_CASE = CodeTemplate('case ${overload_index}: {\n  ${body}\n}\n')
PY_VARIABLE_METHOD_VARARGS_SINGLETON = CodeTemplate('// ${name}\nstatic PyObject * ${pycname}(PyObject* self_, PyObject* args, PyObject* kwargs)\n{\n  ${method_header}\n  static PythonArgParser parser({\n    ${signatures}\n  }, /*traceable=*/${traceable});\n\n  ParsedArgs<${max_args}> parsed_args;\n  auto _r = parser.parse(${self_}, args, kwargs, parsed_args);\n  ${check_has_torch_function}\n  ${dispatch}\n  ${method_footer}\n}\n\n')
PY_VARIABLE_METHOD_NOARGS = CodeTemplate('// ${name}\nstatic PyObject * ${pycname}(PyObject* self_, PyObject* args)\n{\n  ${method_header}\n  ${check_has_torch_function}\n  ${dispatch}\n  ${method_footer}\n}\n\n')

def method_impl(name: BaseOperatorName, module: Optional[str], overloads: Sequence[PythonSignatureNativeFunctionPair], *, method: bool, symint: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a python binding for all overloads of an op.\n    '
    pycname = get_pycname(name)
    noarg = is_noarg(overloads)
    (namedtuple_inits, namedtuple_typenames) = emit_namedtuple_call(overloads)
    method_header = ['HANDLE_TH_ERRORS']
    method_header += namedtuple_inits
    method_header += ['const Tensor& self = THPVariable_Unpack(self_);'] if method else []
    method_footer = ([] if noarg else ['Py_RETURN_NONE;']) + ['END_HANDLE_TH_ERRORS']
    traceable = 'true' if all((should_trace(o.function) for o in overloads)) else 'false'
    grouped_overloads: Sequence[PythonSignatureGroup] = group_overloads(overloads, symint=symint)
    is_singleton = len(grouped_overloads) == 1
    signatures: List[str] = []
    dispatch: List[str] = []
    for (overload_index, overload) in enumerate(grouped_overloads):
        signature = overload.signature.signature_str(symint=symint)
        signatures.append(f'{cpp_string(str(signature))},')
        dispatch_body = emit_dispatch_case(overload, namedtuple_typenames, symint=symint)
        dispatch.append(PY_VARIABLE_CASE.substitute(overload_index=overload_index, body=dispatch_body) if not is_singleton else dispatch_body)
    if noarg:
        template = PY_VARIABLE_METHOD_NOARGS
    elif is_singleton:
        template = PY_VARIABLE_METHOD_VARARGS_SINGLETON
    else:
        template = PY_VARIABLE_METHOD_VARARGS
    return template.substitute(name=name, pycname=pycname, method_header=method_header, max_args=max((o.signature.arguments_count() for o in overloads)), signatures=signatures, traceable=traceable, check_has_torch_function=gen_has_torch_function_check(name=name, module=module, noarg=noarg, method=method), dispatch=dispatch, method_footer=method_footer, self_='self_' if method else 'nullptr')

def gen_has_torch_function_check(name: BaseOperatorName, module: Optional[str], *, noarg: bool, method: bool) -> str:
    if False:
        return 10
    if noarg:
        if method:
            return f'if(check_has_torch_function(self_)) {{\n  return handle_torch_function(self_, "{name}");\n}}\n'
        else:
            return ''
    self_ = 'self_' if method else 'nullptr'
    namespace = {'torch': 'THPVariableFunctionsModule', 'torch.nn': 'THPNNVariableFunctionsModule', 'torch.fft': 'THPFFTVariableFunctionsModule', 'torch.linalg': 'THPLinalgVariableFunctionsModule', 'torch.nested': 'THPNestedVariableFunctionsModule', 'torch.sparse': 'THPSparseVariableFunctionsModule', 'torch.special': 'THPSpecialVariableFunctionsModule'}[module] if module else 'THPVariableClass'
    return f'''if(_r.has_torch_function()) {{\n  return handle_torch_function(_r, {self_}, args, kwargs, {namespace}, "{module or 'torch.Tensor'}");\n}}\n'''
PY_VARIABLE_OUT = CodeTemplate('if (_r.isNone(${out_idx})) {\n  ${call_dispatch}\n} else {\n  ${call_dispatch_out}\n}\n')

def emit_dispatch_case(overload: PythonSignatureGroup, namedtuple_typenames: Dict[str, str], *, symint: bool=True) -> str:
    if False:
        while True:
            i = 10
    '\n    Emit dispatch code for a single parsed signature. This corresponds to either\n    a single native function, or a pair that differ only in output params. In the\n    latter case, a single python signature is used for both and dispatching\n    switches on the presence/absence of passed output args.\n    '
    if overload.outplace is not None:
        return PY_VARIABLE_OUT.substitute(out_idx=overload.signature.output_idx(), call_dispatch=emit_single_dispatch(overload.signature, overload.base, namedtuple_typenames, symint=symint), call_dispatch_out=emit_single_dispatch(overload.signature, overload.outplace, namedtuple_typenames, symint=symint))
    else:
        return emit_single_dispatch(overload.signature, overload.base, namedtuple_typenames, symint=symint)

def forward_decls(name: BaseOperatorName, overloads: Sequence[PythonSignatureNativeFunctionPair], *, method: bool) -> Tuple[str, ...]:
    if False:
        print('Hello World!')
    if method:
        return ()
    pycname = get_pycname(name)
    if is_noarg(overloads):
        return (f'static PyObject * {pycname}(PyObject* self_, PyObject* args);\n',)
    else:
        return (f'static PyObject * {pycname}(PyObject* self_, PyObject* args, PyObject* kwargs);\n',)

def method_def(name: BaseOperatorName, module: Optional[str], overloads: Sequence[PythonSignatureNativeFunctionPair], *, method: bool) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Generate method def entry.\n    '
    pycname = get_pycname(name)
    if name.dunder_method:
        pycname = f'TypeError_to_NotImplemented_<{pycname}>'
    if is_noarg(overloads):
        flags = 'METH_NOARGS' if method else 'METH_VARARGS | METH_KEYWORDS'
    else:
        pycname = f'castPyCFunctionWithKeywords({pycname})'
        flags = 'METH_VARARGS | METH_KEYWORDS'
    if module == 'torch':
        flags += ' | METH_STATIC'
    return f'{{"{name}", {pycname}, {flags}, NULL}},'

def group_overloads(overloads: Sequence[PythonSignatureNativeFunctionPair], *, symint: bool=True) -> Sequence[PythonSignatureGroup]:
    if False:
        i = 10
        return i + 15
    bases: Dict[str, PythonSignatureNativeFunctionPair] = {}
    outplaces: Dict[str, PythonSignatureNativeFunctionPair] = {}
    for overload in overloads:
        sig = overload.signature.signature_str(skip_outputs=True, symint=symint)
        if overload.function.func.is_out_fn():
            if sig in outplaces:
                raise RuntimeError(f'Found duplicated function definition:\n- {overload.function.func}.\nExisting definition:\n- {outplaces[sig].function.func}.')
            outplaces[sig] = overload
        else:
            if sig in bases:
                raise RuntimeError(f'Found duplicated function definition:\n- {overload.function.func}.\nExisting definition:\n- {bases[sig].function.func}.')
            bases[sig] = overload
    for (sig, out) in outplaces.items():
        if sig not in bases:
            candidates: List[str] = []
            for overload in overloads:
                if str(overload.function.func.name.name) == str(out.function.func.name.name) and (not overload.function.func.is_out_fn()) and (not overload.signature.deprecated):
                    candidates.append(overload.signature.signature_str(skip_outputs=True, symint=symint))
            out_sig = out.signature.signature_str(symint=symint)
            raise RuntimeError(f'While identifying overloads, we found an out schema {out_sig} without a corresponding non-out variant. We expected the non-out variant to have schema: \n- {sig}\nPlease check that you spelled the schema correctly in native_functions.yaml. We discovered the following candidate(s): \n' + '\n'.join((f'- {candidate}' for candidate in candidates)))
    grouped = [PythonSignatureGroup.from_pairs(functional=base, out=outplaces.get(sig)) for (sig, base) in bases.items()]
    return sort_overloads(grouped, symint=symint)

def sort_overloads(grouped_overloads: Sequence[PythonSignatureGroup], *, symint: bool=True) -> Sequence[PythonSignatureGroup]:
    if False:
        for i in range(10):
            print('nop')

    def is_arg_smaller(t1: Type, t2: Type) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return str(t1) == 'Scalar' and str(t2) == 'Tensor' or (str(t1) == 'Scalar?' and str(t2) == 'Tensor?') or ('Dimname' in str(t1) and 'Dimname' not in str(t2)) or (str(t1) == 'int[]' and (str(t2) == 'int' or str(t2) == 'int?')) or (str(t1) == 'Tensor[]' and str(t2).find('[]') != -1) or (str(t1) == 'SymInt[]' and str(t2) == 'int[]') or ((str(t1) == 'SymInt' or str(t1) == 'int') and str(t2) == 'Tensor')

    def is_smaller(s1: PythonSignature, s2: PythonSignature) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Returns True if s1 < s2 in the partial order.'
        (args1, args2) = (s1.arguments(skip_outputs=True), s2.arguments(skip_outputs=True))
        if len(args1) != len(args2):
            return False
        equal = all((arg1.type == arg2.type for (arg1, arg2) in zip(args1, args2)))
        smaller_or_equal = all((str(arg1.type) == str(arg2.type) or is_arg_smaller(arg1.type, arg2.type) for (arg1, arg2) in zip(args1, args2)))
        return smaller_or_equal and (not equal)
    grouped_overloads = sorted(grouped_overloads, key=lambda x: x.signature.signature_str(symint=symint))
    larger_than: Dict[int, Set[int]] = defaultdict(set)
    for (i1, overload1) in enumerate(grouped_overloads):
        for (i2, overload2) in enumerate(grouped_overloads):
            if is_smaller(overload1.signature, overload2.signature):
                larger_than[i1].add(i2)
    if not larger_than:
        return list(grouped_overloads)
    N = len(grouped_overloads)
    sorted_ids: List[int] = list(filter(lambda x: x not in larger_than, range(N)))
    for idx in range(N):
        i = sorted_ids[idx]
        for j in sorted(larger_than.keys()):
            larger = larger_than[j]
            larger.discard(i)
            if not larger:
                del larger_than[j]
                sorted_ids.append(j)
    return [grouped_overloads[x] for x in sorted_ids]

def emit_single_dispatch(ps: PythonSignature, f: NativeFunction, namedtuple_typenames: Dict[str, str], *, symint: bool=True) -> str:
    if False:
        print('Hello World!')
    '\n    Emit dispatch code for a single native function.\n    '

    @with_native_function
    def go(f: NativeFunction) -> str:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(ps, PythonSignatureDeprecated):
            schema_comment = f'// [deprecated] aten::{ps.deprecated_schema}'
        else:
            schema_comment = f'// aten::{f.func}'
        deprecated = '[deprecated] ' if ps.deprecated else ''
        name = cpp.name(f.func)
        lambda_formals = ', '.join((f'{a.type_str} {a.name}' for a in dispatch_lambda_args(ps, f, symint=symint)))
        lambda_return = dispatch_lambda_return_str(f)
        dispatch_callee = cpp_dispatch_target(f)
        dispatch_args = ', '.join(cpp_dispatch_exprs(f, python_signature=ps))
        parser_outputs = arg_parser_output_exprs(ps, f, symint=symint)
        lambda_arg_exprs = dispatch_lambda_exprs(ps, f, symint=symint)
        inits = '\n'.join(lambda_arg_exprs.inits)
        lambda_args = ', '.join(lambda_arg_exprs.exprs)
        need_set_requires_grad = ps.tensor_options_args and (not has_tensor_options(f) or (ps.method and 'requires_grad' in parser_outputs))
        set_requires_grad = f".set_requires_grad({parser_outputs['requires_grad'].expr})" if need_set_requires_grad else ''
        if lambda_return == 'void':
            return f'{schema_comment}\n{inits}\nauto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{\n  pybind11::gil_scoped_release no_gil;\n  {dispatch_callee}({dispatch_args});\n}};\ndispatch_{name}({lambda_args}){set_requires_grad};\nPy_RETURN_NONE;\n'
        else:
            typename = namedtuple_typenames.get(gen_namedtuple_typename_key(f))
            namedtuple_typeref = f'{typename}, ' if typename is not None else ''
            return f'{schema_comment}\n{inits}\nauto dispatch_{name} = []({lambda_formals}) -> {lambda_return} {{\n  pybind11::gil_scoped_release no_gil;\n  return {dispatch_callee}({dispatch_args});\n}};\nreturn wrap({namedtuple_typeref}dispatch_{name}({lambda_args}){set_requires_grad});\n'
    return go(f)