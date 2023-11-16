import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypeVar, Union
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup, DispatcherSignature, NamedCType, NativeSignature, SpecialArgName
from torchgen.context import method_with_native_function, native_function_manager, with_native_function, with_native_function_and_indices
from torchgen.gen_functionalization_type import gen_functionalization_definition, gen_functionalization_registration, gen_functionalization_view_inverse_declaration, GenCompositeViewCopyKernel
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import Argument, BackendIndex, BackendMetadata, BaseOperatorName, DEFAULT_KERNEL_NAMESPACE, DispatchKey, FRAGMENT_NAMESPACES, FunctionSchema, is_cuda_dispatch_key, is_generic_dispatch_key, is_ufunc_dispatch_key, Location, NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup, OperatorName, OptionalType, SchemaKind, SelfArgument, STRUCTURED_DISPATCH_KEYS, TensorOptionsArguments, Type, Variant, ViewSchemaKind
from torchgen.native_function_generation import add_generated_native_functions, gen_composite_functional_kernel, gen_composite_out_kernel, pre_group_native_functions
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, concatMap, context, FileManager, make_file_manager, mapMaybe, NamespaceHelper, Target
from torchgen.yaml_utils import YamlDumper, YamlLoader
T = TypeVar('T')

class LineLoader(YamlLoader):

    def construct_mapping(self, node, deep=False):
        if False:
            return 10
        mapping = super().construct_mapping(node, deep=deep)
        mapping['__line__'] = node.start_mark.line + 1
        return mapping
_GLOBAL_PARSE_NATIVE_YAML_CACHE = {}
_GLOBAL_PARSE_TAGS_YAML_CACHE = {}
ParsedYaml = namedtuple('ParsedYaml', ['native_functions', 'backend_indices'])

def parse_native_yaml_struct(es: object, valid_tags: Set[str], ignore_keys: Optional[Set[DispatchKey]]=None, path: str='<stdin>', skip_native_fns_gen: bool=False) -> ParsedYaml:
    if False:
        print('Hello World!')
    assert isinstance(es, list)
    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        funcs = e.get('func')
        with context(lambda : f'in {loc}:\n  {funcs}'):
            (func, m) = NativeFunction.from_yaml(e, loc, valid_tags, ignore_keys)
            rs.append(func)
            BackendIndex.grow_index(bs, m)
    error_check_native_functions(rs)
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(lambda : BackendIndex(dispatch_key=DispatchKey.Undefined, use_out_as_primary=True, external=False, device_guard=False, index={}))
    if not skip_native_fns_gen:
        add_generated_native_functions(rs, bs)
    for (k, v) in bs.items():
        indices[k] = BackendIndex(dispatch_key=k, use_out_as_primary=True, external=False, device_guard=is_cuda_dispatch_key(k), index=v)
    return ParsedYaml(rs, indices)

def parse_tags_yaml_struct(es: object, path: str='<stdin>') -> Set[str]:
    if False:
        i = 10
        return i + 15
    assert isinstance(es, list)
    rs: Set[str] = set()
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        tags = e.get('tag')
        with context(lambda : f'in {loc}:\n  {tags}'):
            e_i = e.copy()
            name = e_i.pop('tag')
            desc = e_i.pop('desc', '')
            assert desc != ''
            rs.add(name)
    return rs

@functools.lru_cache(maxsize=None)
def parse_tags_yaml(path: str) -> Set[str]:
    if False:
        while True:
            i = 10
    global _GLOBAL_PARSE_TAGS_YAML_CACHE
    if path not in _GLOBAL_PARSE_TAGS_YAML_CACHE:
        with open(path) as f:
            es = yaml.load(f, Loader=LineLoader)
            _GLOBAL_PARSE_TAGS_YAML_CACHE[path] = parse_tags_yaml_struct(es, path=path)
    return _GLOBAL_PARSE_TAGS_YAML_CACHE[path]

def parse_native_yaml(path: str, tags_yaml_path: str, ignore_keys: Optional[Set[DispatchKey]]=None, *, skip_native_fns_gen: bool=False, loaded_yaml: Optional[object]=None) -> ParsedYaml:
    if False:
        for i in range(10):
            print('nop')
    global _GLOBAL_PARSE_NATIVE_YAML_CACHE
    if path not in _GLOBAL_PARSE_NATIVE_YAML_CACHE:
        valid_tags = parse_tags_yaml(tags_yaml_path)
        if loaded_yaml is None:
            with open(path) as f:
                es = yaml.load(f, Loader=LineLoader)
        else:
            es = loaded_yaml
        _GLOBAL_PARSE_NATIVE_YAML_CACHE[path] = parse_native_yaml_struct(es, valid_tags, ignore_keys, path=path, skip_native_fns_gen=skip_native_fns_gen)
    return _GLOBAL_PARSE_NATIVE_YAML_CACHE[path]

def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    if False:
        return 10
    func_map: Dict[OperatorName, NativeFunction] = {}
    base_func_map: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map[f.structured_delegate]
            assert delegate_func.structured, f"{f.func.name} is marked as a structured_delegate pointing to {f.structured_delegate}, but {f.structured_delegate} is not marked as structured. Consider adding 'structured=True' to the delegated operator"
        if 'inplace_view' in f.tags and str(f.func.name) != 'resize_' and (str(f.func.name) != 'resize_as_'):
            base_name = f.func.name.name
            overload_name = f.func.name.overload_name
            assert base_name.inplace, f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            out_of_place_base_name = BaseOperatorName(base_name.base, False, base_name.dunder_method)
            assert len(base_func_map[out_of_place_base_name]) > 0, f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "

def cpp_string(s: str) -> str:
    if False:
        return 10
    'Convert a python string into a c++ string literal'
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\x07', '\\a')
    s = s.replace('\x08', '\\b')
    s = s.replace('\x0c', '\\f')
    s = s.replace('\n', '\\n')
    s = s.replace('\x0b', '\\v')
    s = s.replace('\t', '\\t')
    return f'"{s}"'

def static_dispatch_keys(backends: List[BackendIndex]) -> List[DispatchKey]:
    if False:
        i = 10
        return i + 15
    if len(backends) == 0:
        return []
    else:
        return [backend.dispatch_key for backend in backends] + [DispatchKey.CompositeImplicitAutograd, DispatchKey.CompositeImplicitAutogradNestedTensor, DispatchKey.CompositeExplicitAutograd, DispatchKey.CompositeExplicitAutogradNonFunctional]

def get_static_dispatch_backend(f: NativeFunction, backend_index: BackendIndex) -> Optional[DispatchKey]:
    if False:
        while True:
            i = 10
    if f.structured_delegate is not None or backend_index.has_kernel(f):
        return backend_index.dispatch_key
    elif f.has_composite_explicit_autograd_kernel:
        return DispatchKey.CompositeExplicitAutograd
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return DispatchKey.CompositeExplicitAutogradNonFunctional
    elif f.has_composite_implicit_autograd_kernel:
        return DispatchKey.CompositeImplicitAutograd
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return DispatchKey.CompositeImplicitAutogradNestedTensor
    return None

def static_dispatch_ops_header(f: NativeFunction, backend_index: List[BackendIndex]) -> Optional[str]:
    if False:
        return 10
    if backend_index is None or f.manual_kernel_registration:
        return None
    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(f'#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>')
    return '\n'.join(output)

def static_dispatch_extra_headers(backends: List[BackendIndex]) -> List[str]:
    if False:
        while True:
            i = 10
    return [f'#include <ATen/{dispatch_key}Functions.h>' for dispatch_key in static_dispatch_keys(backends)]

def translate_args(sig: Union[CppSignature, DispatcherSignature], cpp_sig: CppSignature) -> str:
    if False:
        while True:
            i = 10

    def add_spl_memory_format_binding(input_bindings: List[Binding]) -> List[Binding]:
        if False:
            for i in range(10):
                print('nop')
        output_bindings: List[Binding] = []
        for binding in input_bindings:
            if binding.name == 'memory_format':
                spl_mem_format_binding = Binding(nctype=NamedCType(SpecialArgName.possibly_redundant_memory_format, binding.nctype.type), name=binding.name, default=binding.default, argument=binding.argument)
                output_bindings.append(spl_mem_format_binding)
            else:
                output_bindings.append(binding)
        return output_bindings
    src_bindings = list(sig.arguments())
    goal_bindings = list(cpp_sig.arguments())
    for arg in goal_bindings:
        if arg.nctype.name == SpecialArgName.possibly_redundant_memory_format:
            src_bindings = add_spl_memory_format_binding(src_bindings)
            break
    exprs = translate(src_bindings, goal_bindings)
    return ', '.join((a.expr for a in exprs))

def generate_static_dispatch_backend_call(sig: Union[CppSignature, DispatcherSignature], f: NativeFunction, backend_index: BackendIndex) -> str:
    if False:
        print('Hello World!')
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    backend_metadata = backend_index.get_kernel(f)
    kernel_ns = backend_metadata.cpp_namespace if backend_metadata and backend_metadata.cpp_namespace else DEFAULT_KERNEL_NAMESPACE
    ns = kernel_ns.replace('::native', '')
    return f'return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});'

def generate_static_dispatch_fallback_call(sig: Union[CppSignature, DispatcherSignature], f: NativeFunction, backend_indices: List[BackendIndex]) -> str:
    if False:
        i = 10
        return i + 15
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    ns = DEFAULT_KERNEL_NAMESPACE.replace('::native', '')
    if f.has_composite_explicit_autograd_kernel:
        return f'return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});'
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return f'return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});'
    elif f.has_composite_implicit_autograd_kernel:
        return f'return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});'
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return f'return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});'
    else:
        return f"""TORCH_CHECK(false, "Static dispatch does not support {name} for{', '.join([str(index.dispatch_key) for index in backend_indices])} ");"""

def static_dispatch(sig: Union[CppSignature, DispatcherSignature], f: NativeFunction, backend_indices: List[BackendIndex]) -> str:
    if False:
        return 10
    '\n    For a given `NativeFunction`, find out the corresponding backend and dispatch to it. If more than one\n    backends exsit, fallback to static dispatch by determining dispatch key from inputs.\n    Arguments:\n        sig: A CppSignature or DispatcherSignature for this native function we want to use.\n        f: NativeFunction to generate static dispatch.\n        backend_indices: All available backends.\n    Return:\n        C++ code to call backend-specific functions, e.g., "return at::cpu::add(self, other, scale);"\n    '
    if len(backend_indices) == 0 or f.manual_kernel_registration:
        return ''
    keys = [b for b in backend_indices if b.has_kernel(f) or (f.structured_delegate is not None and b.dispatch_key in STRUCTURED_DISPATCH_KEYS)]
    if len(keys) == 1:
        return generate_static_dispatch_backend_call(sig, f, keys[0])
    elif len(keys) == 0:
        return generate_static_dispatch_fallback_call(sig, f, backend_indices)
    native_tensor_args = [a.name for a in sig.arguments() if isinstance(a.argument, SelfArgument) or (isinstance(a.argument, Argument) and a.argument.type.is_tensor_like())]
    tensor_args = ', '.join(native_tensor_args)
    tensor_opts = f.func.arguments.tensor_options
    stmts = []
    subexprs: List[str] = []
    if tensor_opts is not None:
        subexprs.append('DispatchKeySet(c10::computeDispatchKey(dtype, layout, device))')
    if tensor_args != '':
        subexprs.append(f'c10::detail::multi_dispatch_key_set({tensor_args})')
    stmts.append(f"DispatchKeySet _dk_set = {' | '.join(subexprs)};")
    stmts.append('DispatchKey _dk = c10::highestPriorityBackendTypeId(_dk_set);')
    dispatch_code = []
    for index in keys:
        dispatch_code.append(f'case DispatchKey::{index.dispatch_key}:')
        dispatch_code.append(f'\t{generate_static_dispatch_backend_call(sig, f, index)};')
    fallback = generate_static_dispatch_fallback_call(sig, f, backend_indices)
    connector = '\n\t\t'
    return f'\n    {connector.join(stmts)}\n    switch (_dk) {{\n        {connector.join(dispatch_code)}\n        default:\n            {fallback}\n    }}\n    '

@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if not self.selector.is_native_function_selected(f):
            return None
        tags = '{' + ', '.join((f'at::Tag::{tag}' for tag in sorted(f.tags))) + '}'
        return f'm.def({cpp_string(str(f.func))}, {tags});\n'

@dataclass(frozen=True)
class ComputeOperators:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        if False:
            for i in range(10):
                print('nop')
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()
        if self.target is Target.DECLARATION:
            return f'''\nstruct TORCH_API {name} {{\n  using schema = {sig.type()};\n  using ptr_schema = schema*;\n  // See Note [static constexpr char* members for windows NVCC]\n  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")\n  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")\n  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})\n  static {sig.defn(name='call', is_redispatching_fn=False)};\n  static {sig.defn(name='redispatch', is_redispatching_fn=True)};\n}};'''
        elif self.target is Target.DEFINITION:
            defns = f'\nSTATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{f.func.name.name}")\nSTATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")\nSTATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})\n\n// aten::{f.func}\nstatic C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{\n  return c10::Dispatcher::singleton()\n      .findSchemaOrThrow({name}::name, {name}::overload_name)\n      .typed<{name}::schema>();\n}}\n'
            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ', '.join(['dispatchKeySet'] + [a.name for a in sig.arguments()])
                    method_base = 'redispatch'
                else:
                    dispatcher_exprs_str = ', '.join([a.name for a in sig.arguments()])
                    method_base = 'call'
                dispatcher_call = method_base
                method_name = f'{name}::{method_base}'
                fn_body = f'\n    static auto op = create_{name}_typed_handle();\n    return op.{dispatcher_call}({dispatcher_exprs_str});'
                if not is_redispatching_fn and len(self.static_dispatch_backend_indices) > 0:
                    fn_body = static_dispatch(sig, f, backend_indices=self.static_dispatch_backend_indices)
                defns += f'\n// aten::{f.func}\n{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{\n    {fn_body}\n}}\n'
            return defns
        else:
            assert_never(self.target)

@dataclass(frozen=True)
class ComputeFunction:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if False:
            while True:
                i = 10
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
        has_symint = f.func.has_symint()
        result = ''
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ', '.join([e.expr for e in exprs])
            if sig.symint:
                intlike_t = 'c10::SymInt'
            else:
                intlike_t = 'int64_t'
            if Variant.function in f.variants:
                result += f'\n// aten::{f.func}\ninline {sig.decl()} {{\n    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});\n}}'
            if has_symint:
                result += f'\nnamespace symint {{\n  template <typename T, typename = std::enable_if_t<std::is_same<T, {intlike_t}>::value>>\n  {sig.decl(suppress_symint_suffix=True)} {{\n    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});\n  }}\n}}\n'
        return result

@dataclass(frozen=True)
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if False:
            return 10
        if Variant.method not in f.variants:
            return None
        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None
        sig_group = CppSignatureGroup.from_native_function(f, method=True, fallback_binding=f.manual_cpp_binding)
        if self.target is Target.DECLARATION:
            result = ''
            for sig in sig_group.signatures():
                result += f'{sig.decl()} const;\n'
            return result
        if self.target is not Target.DEFINITION:
            assert_never(self.target)
        result = ''
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ', '.join([e.expr for e in exprs])
            result += f"\n// aten::{f.func}\ninline {sig.defn(prefix='Tensor::')} const {{\n    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});\n}}\n"
        return result

@dataclass(frozen=True)
class ComputeRedispatchFunction:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding)
        result = ''
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments())
            exprs_str = ', '.join(['dispatchKeySet'] + [a.expr for a in exprs])
            result += f'\n// aten::{f.func}\ninline {sig.decl(is_redispatching_fn=True)} {{\n    return at::_ops::{f.func.name.unambiguous_name()}::redispatch({exprs_str});\n}}\n'
        return result

@with_native_function
def compute_aten_op(f: NativeFunction) -> str:
    if False:
        print('Hello World!')
    return f'{{"aten::{f.func.name.name}", "{f.func.name.overload_name}"}},'

def compute_meta_function_declaration(g: NativeFunctionsGroup) -> Optional[str]:
    if False:
        print('Hello World!')
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ', '.join((a.decl() for a in args))
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = 'at::impl::MetaBase'
        meta_return = 'void'
        precomputed = g.out.precomputed if g.structured else None
        if precomputed:
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [elem for replace_list in precomputed_values for elem in replace_list]
            precomputed_template_parameters = [elem.name.upper() for elem in precomputed_elements]
            precomputed_template_params_str = ', '.join((f'bool {param} = false' for param in precomputed_template_parameters))
            precompute_template_decl = f'template <{precomputed_template_params_str}>'
            precomputed_elements_with_cpp_types = [structured.argument_type(elem, binds=elem.name) for elem in precomputed_elements]
            precomputed_elements_decl = ';\n'.join((f'{elem.cpp_type(strip_ref=True)} {elem.name}' for elem in precomputed_elements_with_cpp_types))
            setter_methods = []
            for (i, elem) in enumerate(precomputed_elements):
                return_ty_templates = ', '.join(precomputed_template_parameters[:i] + ['true'] + precomputed_template_parameters[i + 1:])
                return_ty = f'precompute_out<{return_ty_templates}>'
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(strip_ref=True)
                signature = f'{return_ty} set_{elem.name}({elem_cpp_ty} value)'
                assert_msg = f'"{precomputed_elements[i].name} already set"'
                assert_stmt = f'static_assert({precomputed_template_parameters[i]} == false, {assert_msg});'
                construction_stmts = []
                construction_stmts.append(f'{return_ty} ret;')
                for (j, elem) in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f'ret.{elem.name} = value;')
                    else:
                        construction_stmts.append(f'ret.{elem.name} = this->{elem.name};')
                construction_stmts.append('return ret;')
                construction_block = '\n'.join(construction_stmts)
                setter_methods.append(f'\n                    {signature} {{\n                        {assert_stmt}\n                        {construction_block}\n                    }}\n                ')
            setter_methods_decl = '\n'.join(setter_methods)
            meta_return_template_params = ', '.join(['true'] * len(precomputed_template_parameters))
            meta_return_typedef = f'using meta_return_ty = precompute_out <{meta_return_template_params}>;'
            meta_return = 'meta_return_ty'
            precomputed_decl = f'\n                {precompute_template_decl}\n                struct TORCH_API precompute_out {{\n                    {setter_methods_decl}\n                    {precomputed_elements_decl};\n            }};'
        else:
            meta_return_typedef = ''
            precomputed_decl = ''
        return f'struct TORCH_API structured_{name} : public {parent_class} {{\n    {precomputed_decl}\n    {meta_return_typedef}\n    {meta_return} meta({args_str});\n}};\n'

def needs_backend_select(f: NativeFunction, selector: SelectiveBuilder) -> bool:
    if False:
        print('Hello World!')
    name = str(f.func.name.name)
    if name.endswith('_like') or name.startswith('new_'):
        return False
    if f.func.arguments.tensor_options is None:
        return False
    return selector.is_native_function_selected(f)

@dataclass(frozen=True)
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if False:
            while True:
                i = 10
        if not needs_backend_select(f, self.selector):
            return None
        name = native.name(f.func)
        native_sig = NativeSignature(f.func, symint=True)
        native_tensor_args = [a for a in native_sig.arguments() if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()]
        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        sig: Union[NativeSignature, DispatcherSignature]
        sig = dispatcher_sig
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key = 'c10::computeDispatchKey(dtype, layout, device)'
        if self.target is Target.DEFINITION:
            if native_tensor_args:
                assert f.func.arguments.has_tensor_arg()
                tensor_args = ', '.join((a.name for a in native_tensor_args))
                compute_dk = f'DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});\nDispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);\nDispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);'
            else:
                assert not f.func.arguments.has_tensor_arg()
                compute_dk = f'DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});'
            return f"// aten::{f.func}\nC10_ALWAYS_INLINE\n{sig.defn(name)} {{\n  {compute_dk}\n  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(\n      _dk, {', '.join((a.expr for a in dispatcher_exprs))});\n}}\n"
        elif self.target is Target.REGISTRATION:
            return f'm.impl("aten::{f.func.name}", TORCH_FN({name}));'
        else:
            assert_never(self.target)

def format_yaml(data: object) -> str:
    if False:
        i = 10
        return i + 15
    YamlDumper.ignore_aliases = lambda self, data: True

    def dict_representer(dumper: Any, data: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return dumper.represent_dict(data.items())
    YamlDumper.add_representer(OrderedDict, dict_representer)
    return yaml.dump(data, default_flow_style=False, Dumper=YamlDumper, width=1000000000.0)

def pythonify_default(s: str) -> object:
    if False:
        for i in range(10):
            print('nop')
    if s == 'true':
        return True
    elif s == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def dynamic_type(t: Type) -> str:
    if False:
        return 10
    if isinstance(t, OptionalType):
        return dynamic_type(t.elem)
    if str(t) == 'Tensor':
        return 'at::Tensor'
    return cpp.argumenttype_type(t, mutable=False, binds='__placeholder__', symint=False).cpp_type()

def compute_method_of_yaml(variants: Set[Variant]) -> List[str]:
    if False:
        i = 10
        return i + 15
    method_of = ['Type']
    if Variant.method in variants:
        method_of.append('Tensor')
    if Variant.function in variants:
        method_of.append('namespace')
    return method_of

def compute_returns_yaml(f: NativeFunction) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    if False:
        return 10
    name_to_field_name: Dict[str, str] = {}
    names = cpp.return_names(f)
    returns = []
    for (i, (r, name)) in enumerate(zip(f.func.returns, names)):
        ret = {'dynamic_type': dynamic_type(r.type), 'name': name, 'type': cpp.return_type(r, symint=False).cpp_type()}
        if r.name:
            ret['field_name'] = r.name
            if f.func.is_out_fn():
                name_to_field_name[f.func.arguments.out[i].name] = r.name
        returns.append(ret)
    return (returns, name_to_field_name)

def compute_cpp_argument_yaml(cpp_a: Binding, *, schema_order: bool, kwarg_only_set: Set[str], out_arg_set: Set[str], name_to_field_name: Dict[str, str]) -> object:
    if False:
        i = 10
        return i + 15
    if isinstance(cpp_a.argument, TensorOptionsArguments):
        arg: Dict[str, object] = {'annotation': None, 'dynamic_type': 'at::TensorOptions', 'is_nullable': False, 'name': cpp_a.name, 'type': cpp_a.type, 'kwarg_only': True}
        if cpp_a.default is not None:
            arg['default'] = cpp_a.default
        return arg
    elif isinstance(cpp_a.argument, SelfArgument):
        raise AssertionError()
    elif isinstance(cpp_a.argument, Argument):
        return compute_argument_yaml(cpp_a.argument, schema_order=schema_order, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name)

def compute_argument_yaml(a: Argument, *, schema_order: bool, kwarg_only_set: Set[str], out_arg_set: Set[str], name_to_field_name: Dict[str, str]) -> object:
    if False:
        while True:
            i = 10
    arg: Dict[str, object] = {'annotation': str(a.annotation) if a.annotation else None, 'dynamic_type': dynamic_type(a.type), 'is_nullable': a.type.is_nullable(), 'name': a.name, 'type': cpp.argument_type(a, binds='__placeholder__', symint=False).cpp_type()}
    if a.default is not None:
        arg['default'] = pythonify_default(cpp.default_expr(a.default, a.type, symint=False))
    if a.name in kwarg_only_set:
        arg['kwarg_only'] = True
    if a.name in out_arg_set:
        arg['output'] = True
        arg['allocate'] = True
        if a.name in name_to_field_name:
            arg['field_name'] = name_to_field_name[a.name]
    l = a.type.is_list_like()
    if l is not None and l.size is not None and (str(l.elem) != 'bool'):
        arg['size'] = l.size
    return arg

@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    if False:
        for i in range(10):
            print('nop')
    (returns, name_to_field_name) = compute_returns_yaml(f)
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}
    sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    cpp_args = sig_group.signature.arguments()
    arguments = [compute_cpp_argument_yaml(cpp_a, schema_order=False, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name) for cpp_a in cpp_args]
    schema_order_jit_arguments = list(f.func.schema_order_arguments())
    schema_order_arguments = [compute_argument_yaml(a, schema_order=True, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name) for a in schema_order_jit_arguments]
    cpp_schema_order_types = [r.type for a in schema_order_jit_arguments for r in cpp.argument(a, method=False, cpp_no_default_args=set(), faithful=False, symint=False, has_tensor_options=False)]
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f"{cpp_returns} ({', '.join(cpp_schema_order_types)})"
    is_factory_method = any((isinstance(a.argument, TensorOptionsArguments) for a in cpp_args)) and Variant.method not in f.variants
    return OrderedDict([('name', cpp.name(f.func)), ('operator_name', str(f.func.name.name)), ('overload_name', str(f.func.name.overload_name)), ('manual_kernel_registration', f.manual_kernel_registration), ('category_override', f.category_override if f.category_override is not None else ''), ('schema_string', f'aten::{f.func}'), ('arguments', arguments), ('schema_order_cpp_signature', schema_order_cpp_signature), ('schema_order_arguments', schema_order_arguments), ('method_of', compute_method_of_yaml(f.variants)), ('mode', 'native'), ('python_module', '' if f.python_module is None else f.python_module), ('returns', returns), ('inplace', f.func.name.name.inplace), ('is_factory_method', is_factory_method), ('abstract', f.is_abstract), ('device_guard', f.device_guard), ('with_gil', False), ('deprecated', False), ('has_math_kernel', f.has_composite_implicit_autograd_kernel)])

def has_autogenerated_composite_kernel(f: NativeFunction) -> bool:
    if False:
        return 10
    return (f.structured or f.structured_delegate is not None) and (f.func.kind() == SchemaKind.functional or f.func.kind() == SchemaKind.inplace)

@with_native_function_and_indices
def compute_registration_declarations(f: NativeFunction, backend_indices: Dict[DispatchKey, BackendIndex]) -> str:
    if False:
        print('Hello World!')
    name = dispatcher.name(f.func)
    returns_type = dispatcher.returns_type(f.func.returns).cpp_type_registration_declarations()
    args = dispatcher.arguments(f.func)
    args_str = ', '.join((a.no_default().decl_registration_declarations() for a in args))
    comment_data: Dict[str, str] = {'schema': f'aten::{f.func}', 'dispatch': str({k for (k, v) in backend_indices.items() if v.has_kernel(f)} != {DispatchKey.CompositeImplicitAutograd} and {k for (k, v) in backend_indices.items() if v.has_kernel(f)} != {DispatchKey.CompositeImplicitAutograd, DispatchKey.CompositeImplicitAutogradNestedTensor}), 'default': str(f.has_composite_kernel or has_autogenerated_composite_kernel(f))}
    return f'{returns_type} {name}({args_str}); // {json.dumps(comment_data)}\n'

def get_custom_build_selector(provided_op_registration_allowlist: Optional[List[str]], op_selection_yaml_path: Optional[str]) -> SelectiveBuilder:
    if False:
        while True:
            i = 10
    assert not (provided_op_registration_allowlist is not None and op_selection_yaml_path is not None), 'Both provided_op_registration_allowlist and ' + 'op_selection_yaml_path can NOT be provided at the ' + 'same time.'
    op_registration_allowlist: Optional[Set[str]] = None
    if provided_op_registration_allowlist is not None:
        op_registration_allowlist = set(provided_op_registration_allowlist)
    if op_registration_allowlist is not None:
        selector = SelectiveBuilder.from_legacy_op_registration_allow_list(op_registration_allowlist, True, False)
    elif op_selection_yaml_path is not None:
        selector = SelectiveBuilder.from_yaml_path(op_selection_yaml_path)
    else:
        selector = SelectiveBuilder.get_nop_selector()
    return selector

def get_grouped_by_view_native_functions(native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsViewGroup]]:
    if False:
        for i in range(10):
            print('nop')

    def maybe_create_view_group(d: Dict[Union[ViewSchemaKind, SchemaKind], NativeFunction]) -> List[Union[NativeFunction, NativeFunctionsViewGroup]]:
        if False:
            while True:
                i = 10
        funcs: List[Union[NativeFunction, NativeFunctionsViewGroup]] = []
        if ViewSchemaKind.aliasing in d:
            view = d.pop(ViewSchemaKind.aliasing)
            view_inplace = d.pop(ViewSchemaKind.aliasing_inplace, None)
            view_copy = d.pop(SchemaKind.functional, None)
            funcs.append(NativeFunctionsViewGroup(view=view, view_copy=view_copy, view_inplace=view_inplace))
        for func in d.values():
            funcs.append(func)
        return funcs
    grouped_by_views: Dict[FunctionSchema, Dict[Union[SchemaKind, ViewSchemaKind], NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        schema = f.func.view_signature()
        view_kind: ViewSchemaKind = f.view_schema_kind
        if view_kind == ViewSchemaKind.non_aliasing:
            kind = f.func.kind()
            assert kind not in grouped_by_views[schema]
            grouped_by_views[schema][kind] = f
        else:
            assert view_kind not in grouped_by_views[schema]
            grouped_by_views[schema][view_kind] = f
    return list(concatMap(maybe_create_view_group, grouped_by_views.values()))

def get_grouped_native_functions(native_functions: Sequence[NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
    if False:
        while True:
            i = 10

    def flatten_pre_group(d: Dict[SchemaKind, NativeFunction]) -> Sequence[Union[NativeFunction, NativeFunctionsGroup]]:
        if False:
            i = 10
            return i + 15
        r = NativeFunctionsGroup.from_dict(d)
        if r is None:
            assert not any(('generated' in f.tags for f in d.values()))
            return list(d.values())
        else:
            return [r]
    pre_grouped_native_functions = pre_group_native_functions(native_functions)
    return list(concatMap(flatten_pre_group, list(pre_grouped_native_functions.values())))

def get_ns_grouped_kernels(*, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], backend_indices: Dict[DispatchKey, BackendIndex], native_function_decl_gen: Callable[[Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]]=dest.compute_native_function_declaration) -> Dict[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    for f in grouped_native_functions:
        native_function_namespaces = set()
        dispatch_keys = set()
        for (dispatch_key, backend_idx) in backend_indices.items():
            backend_metadata = backend_idx.get_kernel(f)
            if backend_metadata:
                namespace = backend_metadata.cpp_namespace
                dispatch_keys.add(dispatch_key)
                native_function_namespaces.add(namespace)
            else:
                namespace = DEFAULT_KERNEL_NAMESPACE
            assert len(native_function_namespaces) <= 1, f'Codegen only supports one namespace per operator, got {native_function_namespaces} from {dispatch_keys}'
            ns_grouped_kernels[namespace].extend(native_function_decl_gen(f, backend_idx))
    return ns_grouped_kernels

def get_native_function_declarations_from_ns_grouped_kernels(*, ns_grouped_kernels: Dict[str, List[str]]) -> List[str]:
    if False:
        while True:
            i = 10
    declarations: List[str] = []
    newline = '\n'
    for (namespace, kernels) in ns_grouped_kernels.items():
        ns_helper = NamespaceHelper(namespace_str=namespace, entity_name='', max_level=4)
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(f'\n{ns_helper.prologue}\n{newline.join(ordered_kernels)}\n{ns_helper.epilogue}\n        '.split(newline))
    return declarations

def get_native_function_declarations(*, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], backend_indices: Dict[DispatchKey, BackendIndex], native_function_decl_gen: Callable[[Union[NativeFunctionsGroup, NativeFunction], BackendIndex], List[str]]=dest.compute_native_function_declaration) -> List[str]:
    if False:
        print('Hello World!')
    '\n    Generate kernel declarations, in `NativeFunction(s).h`.\n    :param grouped_native_functions: a sequence of `NativeFunction` or `NativeFunctionGroup`.\n    :param backend_indices: kernel collections grouped by dispatch key.\n    :param native_function_decl_gen: callable to generate kernel declaration for each `NativeFunction`.\n    :return: a list of string, from the string with all declarations, grouped by namespaces, split by newline.\n    '
    ns_grouped_kernels = get_ns_grouped_kernels(grouped_native_functions=grouped_native_functions, backend_indices=backend_indices, native_function_decl_gen=native_function_decl_gen)
    return get_native_function_declarations_from_ns_grouped_kernels(ns_grouped_kernels=ns_grouped_kernels)

def get_kernel_namespace(*, f: Union[NativeFunction, NativeFunctionsGroup], backend_idx: BackendIndex) -> str:
    if False:
        return 10
    backend_metadata = backend_idx.get_kernel(f)
    assert not backend_metadata or '::native' in backend_metadata.cpp_namespace, f"The kernel for function {(f.func.name if isinstance(f, NativeFunction) else f.functional.func.name)} with dispatch key {backend_idx.dispatch_key} has a namespace {backend_metadata.cpp_namespace} and it's not ending with '::native'."
    return backend_metadata.cpp_namespace if backend_metadata else DEFAULT_KERNEL_NAMESPACE

def get_native_function_definitions(*, fm: FileManager, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], dispatch_key: DispatchKey, backend_idx: BackendIndex, selector: SelectiveBuilder, rocm: bool, symint: bool, skip_dispatcher_op_registration: bool, gen_dispatch_helpers: bool) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    definitions: List[str] = []
    ns_definitions: Dict[str, List[str]] = defaultdict(list)
    anonymous_definitions: Dict[str, List[str]] = defaultdict(list)
    registrations: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    newline = '\n'
    ns_gen = dest.RegisterDispatchKey(backend_idx, Target.NAMESPACED_DEFINITION, selector, rocm=rocm, symint=symint, class_method_name=None, skip_dispatcher_op_registration=skip_dispatcher_op_registration)
    anonymous_gen = dest.RegisterDispatchKey(backend_idx, Target.ANONYMOUS_DEFINITION, selector, rocm=rocm, symint=symint, class_method_name=None, skip_dispatcher_op_registration=skip_dispatcher_op_registration)
    reg_gen = dest.RegisterDispatchKey(backend_idx, Target.REGISTRATION, selector, rocm=rocm, symint=symint, class_method_name=None, skip_dispatcher_op_registration=skip_dispatcher_op_registration)
    for f in grouped_native_functions:
        kernel_namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace('::native', '')
        ns_definitions[kernel_namespace].extend(ns_gen(f))
        anonymous_definitions[kernel_namespace].extend(anonymous_gen(f))
        namespace = f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace] = defaultdict(list)
        registrations[kernel_namespace][namespace].extend(reg_gen(f))
    for kernel_namespace in ns_definitions:
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace)
        registration_body = ''
        for namespace in registrations[kernel_namespace]:
            if not registrations[kernel_namespace][namespace]:
                continue
            registration_body += f'\nTORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{\n    {newline.join(registrations[kernel_namespace][namespace])}\n}};'
        definitions.extend(fm.substitute_with_template('RegisterDispatchDefinitions.ini', lambda : {'ns_prologue': ns_helper.prologue, 'ns_epilogue': ns_helper.epilogue, 'dispatch_helpers': dest.gen_registration_helpers(backend_idx) if gen_dispatch_helpers else [], 'dispatch_anonymous_definitions': anonymous_definitions[kernel_namespace], 'static_init_dispatch_registrations': '' if skip_dispatcher_op_registration else registration_body, 'deferred_dispatch_registrations': '', 'dispatch_namespace': dispatch_key.lower(), 'dispatch_namespaced_definitions': ns_definitions[kernel_namespace]}).split(newline))
    return definitions

def get_namespaced_declaration(*, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], dispatch_key: DispatchKey, backend_idx: BackendIndex, selector: SelectiveBuilder, rocm: bool, symint: bool) -> List[str]:
    if False:
        i = 10
        return i + 15
    declarations: List[str] = []
    ns_grouped_kernels: Dict[str, List[str]] = defaultdict(list)
    newline = '\n'
    func = dest.RegisterDispatchKey(backend_idx, Target.NAMESPACED_DECLARATION, selector, rocm=rocm, class_method_name=None, skip_dispatcher_op_registration=False, symint=symint)
    for f in grouped_native_functions:
        namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace('native', dispatch_key.lower())
        ns_grouped_kernels[namespace].extend(func(f))
    for (namespace, kernels) in ns_grouped_kernels.items():
        if len(kernels) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=namespace, entity_name='', max_level=3)
        ordered_kernels = list(OrderedDict.fromkeys(kernels))
        declarations.extend(f'\n{ns_helper.prologue}\n{newline.join(ordered_kernels)}\n{ns_helper.epilogue}\n        '.split(newline))
    return declarations

def get_native_function_schema_registrations(*, native_functions: Sequence[NativeFunction], schema_selector: SelectiveBuilder) -> Tuple[List[str], str]:
    if False:
        return 10
    ns_native_functions: Dict[str, List[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_native_functions[native_function.namespace].append(native_function)
    schema_registrations = ''
    aten_schema_registrations = []
    custom_namespace = None
    for (namespace, funcs) in ns_native_functions.items():
        schema_registrations_body = list(mapMaybe(RegisterSchema(schema_selector), funcs))
        if namespace == 'aten':
            aten_schema_registrations = schema_registrations_body
        else:
            custom_namespace = namespace
            tab = '\t'
            torch_library_macro = 'TORCH_LIBRARY_FRAGMENT' if namespace in FRAGMENT_NAMESPACES else 'TORCH_LIBRARY'
            schema_registrations += f'\n{torch_library_macro}({custom_namespace}, m) {{\n  {tab.join(schema_registrations_body)}\n}};'
    return (aten_schema_registrations, schema_registrations)

def gen_aggregated_headers(*, native_functions: Sequence[NativeFunction], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], structured_native_functions: Sequence[NativeFunctionsGroup], static_dispatch_idx: List[BackendIndex], selector: SelectiveBuilder, backend_indices: Dict[DispatchKey, BackendIndex], cpu_fm: FileManager, cuda_fm: FileManager, functions_keys: Set[DispatchKey], dispatch_keys: Sequence[DispatchKey], rocm: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    cpu_fm.write('NativeMetaFunctions.h', lambda : {'NativeMetaFunctions_includes': [], 'NativeMetaFunctions_declarations': list(mapMaybe(compute_meta_function_declaration, structured_native_functions))})
    method_native_functions = [fn for fn in native_functions if Variant.method in fn.variants]
    non_method_native_functions = [fn for fn in native_functions if fn not in method_native_functions]
    cpu_fm.write('MethodOperators.h', lambda : {'MethodOperators_includes': [], 'MethodOperators_declarations': list(mapMaybe(ComputeOperators(Target.DECLARATION, static_dispatch_backend_indices=static_dispatch_idx), method_native_functions))})
    cpu_fm.write('Operators.h', lambda : {'Operators_includes': ['#include <ATen/MethodOperators.h>'], 'Operators_declarations': list(mapMaybe(ComputeOperators(Target.DECLARATION, static_dispatch_backend_indices=static_dispatch_idx), non_method_native_functions))})
    cpu_fm.write('Functions.h', lambda : {'static_dispatch_extra_headers': static_dispatch_extra_headers(static_dispatch_idx), 'Functions_includes': ['#include <ATen/Operators.h>'], 'Functions_declarations': list(mapMaybe(ComputeFunction(), native_functions))})
    declarations = get_native_function_declarations(grouped_native_functions=grouped_native_functions, backend_indices=backend_indices)
    cpu_fm.write('NativeFunctions.h', lambda : {'NativeFunctions_includes': ['#include <ATen/NativeMetaFunctions.h>'], 'NativeFunctions_declarations': declarations})
    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        if dispatch_key in functions_keys:
            inl_headers = f'#include <ATen/{dispatch_key}Functions_inl.h>'
            fm.write_with_template(f'{dispatch_key}Functions.h', 'DispatchKeyFunctions.h', lambda : {'dispatch_key': str(dispatch_key), 'inline_headers': inl_headers})
            fm.write_with_template(f'{dispatch_key}Functions_inl.h', 'DispatchKeyFunctions_inl.h', lambda : {'DispatchKeyFunctions_inl_includes': [], 'dispatch_namespace': dispatch_key.lower(), 'dispatch_namespaced_declarations': get_namespaced_declaration(grouped_native_functions=grouped_native_functions, dispatch_key=dispatch_key, backend_idx=backend_indices[dispatch_key], selector=selector, rocm=rocm, symint=True)})
        del fm

def gen_per_operator_headers(*, native_functions: Sequence[NativeFunction], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], static_dispatch_idx: List[BackendIndex], selector: SelectiveBuilder, backend_indices: Dict[DispatchKey, BackendIndex], cpu_fm: FileManager, cuda_fm: FileManager, ops_fm: FileManager, functions_keys: Set[DispatchKey], dispatch_keys: Sequence[DispatchKey], rocm: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    functions_by_root_name: Dict[str, List[NativeFunction]] = defaultdict(list)
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)
    grouped_functions_by_root_name: Dict[str, List[Union[NativeFunction, NativeFunctionsGroup]]] = defaultdict(list)
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)
    for (name, functions) in functions_by_root_name.items():
        ops_fm.write_with_template(f'{name}_ops.h', 'Operator.h', lambda : {'declarations': list(mapMaybe(ComputeOperators(Target.DECLARATION, static_dispatch_backend_indices=static_dispatch_idx), functions))})
        ops_fm.write_with_template(f'{name}.h', 'Function.h', lambda : {'static_dispatch_ops_headers': list(mapMaybe(lambda fn: static_dispatch_ops_header(fn, backend_index=static_dispatch_idx), functions)), 'operator_includes': f'#include <ATen/ops/{name}_ops.h>', 'function_definitions': list(mapMaybe(ComputeFunction(), functions))})
        grouped_functions = grouped_functions_by_root_name.get(name, [])
        structured_functions = [fn for fn in grouped_functions if isinstance(fn, NativeFunctionsGroup) and fn.structured]
        is_structured = len(structured_functions) > 0
        if is_structured:
            ops_fm.write_with_template(f'{name}_meta.h', 'NativeMetaFunction.h', lambda : {'meta_function_declarations': list(mapMaybe(compute_meta_function_declaration, structured_functions))})
        declarations = get_native_function_declarations(grouped_native_functions=grouped_functions, backend_indices=backend_indices, native_function_decl_gen=dest.compute_native_function_declaration)
        ops_fm.write_with_template(f'{name}_native.h', 'NativeFunction.h', lambda : {'extra_includes': f'#include <ATen/ops/{name}_meta.h>' if is_structured else [], 'native_function_declarations': declarations})
    for (category, suffix) in [('Functions', ''), ('Operators', '_ops'), ('NativeMetaFunctions', '_meta'), ('NativeFunctions', '_native')]:
        cpu_fm.write(f'{category}.h', lambda : {f'{category}_includes': [f'#include <ATen/ops/{name}{suffix}.h>' for name in sorted(functions_by_root_name.keys())], f'{category}_declarations': []})
    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue
        dispatch_namespace = dispatch_key.lower()
        dispatch_names = []
        for (name, functions) in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(concatMap(dest.RegisterDispatchKey(backend_indices[dispatch_key], Target.NAMESPACED_DECLARATION, selector, rocm=rocm, symint=True, class_method_name=None, skip_dispatcher_op_registration=False), grouped_functions))
            if len(declarations) == 0:
                continue
            dispatch_names.append(name)
            ops_fm.write_with_template(f'{name}_{dispatch_namespace}_dispatch.h', 'DispatchKeyFunction.h', lambda : {'dispatch_namespace': dispatch_namespace, 'dispatch_namespaced_declarations': declarations})
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        inl_headers = f'#include <ATen/{dispatch_key}Functions_inl.h>'
        fm.write_with_template(f'{dispatch_key}Functions.h', 'DispatchKeyFunctions.h', lambda : {'dispatch_key': str(dispatch_key), 'inline_headers': inl_headers})
        fm.write_with_template(f'{dispatch_key}Functions_inl.h', 'DispatchKeyFunctions_inl.h', lambda : {'dispatch_namespace': dispatch_namespace, 'DispatchKeyFunctions_inl_includes': [f'#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>' for name in sorted(dispatch_names)], 'dispatch_namespaced_declarations': []})
        del fm
    cpu_fm.write('MethodOperators.h', lambda : {'MethodOperators_includes': sorted((f'#include <ATen/ops/{name}_ops.h>' for (name, functions) in functions_by_root_name.items() if any((Variant.method in fn.variants for fn in functions)))), 'MethodOperators_declarations': []})

def gen_headers(*, native_functions: Sequence[NativeFunction], valid_tags: Set[str], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], structured_native_functions: Sequence[NativeFunctionsGroup], static_dispatch_idx: List[BackendIndex], selector: SelectiveBuilder, backend_indices: Dict[DispatchKey, BackendIndex], core_fm: FileManager, cpu_fm: FileManager, cuda_fm: FileManager, ops_fm: FileManager, dispatch_keys: Sequence[DispatchKey], functions_keys: Set[DispatchKey], rocm: bool, per_operator_headers: bool) -> None:
    if False:
        while True:
            i = 10
    if per_operator_headers:
        gen_per_operator_headers(native_functions=native_functions, grouped_native_functions=grouped_native_functions, static_dispatch_idx=static_dispatch_idx, selector=selector, backend_indices=backend_indices, cpu_fm=cpu_fm, cuda_fm=cuda_fm, ops_fm=ops_fm, dispatch_keys=dispatch_keys, functions_keys=functions_keys, rocm=rocm)
    else:
        gen_aggregated_headers(native_functions=native_functions, grouped_native_functions=grouped_native_functions, structured_native_functions=structured_native_functions, static_dispatch_idx=static_dispatch_idx, selector=selector, backend_indices=backend_indices, cpu_fm=cpu_fm, cuda_fm=cuda_fm, dispatch_keys=dispatch_keys, functions_keys=functions_keys, rocm=rocm)
    core_fm.write('TensorBody.h', lambda : {'tensor_method_declarations': list(mapMaybe(ComputeTensorMethod(target=Target.DECLARATION, static_dispatch_backend_indices=static_dispatch_idx), native_functions)), 'tensor_method_definitions': list(mapMaybe(ComputeTensorMethod(target=Target.DEFINITION, static_dispatch_backend_indices=static_dispatch_idx), native_functions))})
    cpu_fm.write('RedispatchFunctions.h', lambda : {'function_redispatch_definitions': list(mapMaybe(ComputeRedispatchFunction(), native_functions))})
    cpu_fm.write('RegistrationDeclarations.h', lambda : {'registration_declarations': [compute_registration_declarations(f, backend_indices) for f in native_functions]})
    cpu_fm.write('VmapGeneratedPlumbing.h', lambda : gen_all_vmap_plumbing(native_functions))

    def gen_aten_interned_strings() -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        attrs = set()
        names = set()
        for func in native_functions:
            names.add(str(func.func.name.name))
            names.add(func.func.name.name.base)
            for arg in func.func.schema_order_arguments():
                attrs.add(arg.name)
        names -= {'and', 'and_eq', 'bitand', 'bitor', 'compl', 'not', 'not_eq', 'or', 'or_eq', 'xor', 'xor_eq'}
        return {'aten_symbols': ' \\\n'.join([f'_(aten, {name})' for name in sorted(names)]), 'attr_symbols': ' \\\n'.join([f'_(attr, {name})' for name in sorted(attrs)])}
    core_fm.write('aten_interned_strings.h', gen_aten_interned_strings)

    def gen_tags_enum() -> Dict[str, str]:
        if False:
            return 10
        return {'enum_of_valid_tags': ',\n'.join(sorted(valid_tags))}
    core_fm.write('enum_tag.h', gen_tags_enum)

def gen_source_files(*, native_functions: Sequence[NativeFunction], grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], structured_native_functions: Sequence[NativeFunctionsGroup], view_groups: Sequence[NativeFunctionsViewGroup], selector: SelectiveBuilder, static_dispatch_idx: List[BackendIndex], backend_indices: Dict[DispatchKey, BackendIndex], core_fm: FileManager, cpu_fm: FileManager, cpu_vec_fm: FileManager, cuda_fm: FileManager, dispatch_keys: Sequence[DispatchKey], functions_keys: Set[DispatchKey], rocm: bool, force_schema_registration: bool, per_operator_headers: bool, skip_dispatcher_op_registration: bool) -> None:
    if False:
        return 10
    extra_cuda_headers = '#include <c10/cuda/CUDAGuard.h>\n#include <ATen/cuda/ATenCUDAGeneral.h>\n#include <ATen/cuda/CUDADevice.h>\n#include <ATen/cuda/CUDAContext.h>'
    if rocm:
        extra_cuda_headers = '#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n#include <ATen/hip/ATenHIPGeneral.h>\n#include <ATen/hip/HIPDevice.h>\n#include <ATen/hip/HIPContext.h>'
    for dispatch_key in dispatch_keys:
        fm = cuda_fm if is_cuda_dispatch_key(dispatch_key) else cpu_fm
        if per_operator_headers:

            def operator_headers() -> List[str]:
                if False:
                    return 10
                headers = []
                for g in grouped_native_functions:
                    is_registered = False
                    if backend_index.has_kernel(g):
                        is_registered = True
                    elif isinstance(g, NativeFunctionsGroup) and any((backend_index.has_kernel(fn) for fn in g.functions())):
                        is_registered = True
                    elif g.structured and dispatch_key in (DispatchKey.Meta, DispatchKey.CompositeExplicitAutogradNonFunctional):
                        is_registered = True
                    if not is_registered:
                        continue
                    headers.append(f'#include <ATen/ops/{g.root_name}_native.h>')
                    if dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                        headers.append(f'#include <ATen/ops/{g.root_name}.h>')
                    if dispatch_key in functions_keys:
                        headers.append(f'#include <ATen/ops/{g.root_name}_{dispatch_namespace}_dispatch.h>')
                return sorted(set(headers))
        else:

            def operator_headers() -> List[str]:
                if False:
                    print('Hello World!')
                headers = ['#include <ATen/NativeFunctions.h>']
                if dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                    headers.append('#include <ATen/Functions.h>')
                if dispatch_key in functions_keys:
                    headers.append(f'#include <ATen/{dispatch_key!s}Functions.h>')
                return headers
        backend_index = backend_indices[dispatch_key]
        ns_grouped_native_functions = defaultdict(list)
        for grouped_native_function in grouped_native_functions:
            namespace = grouped_native_function.namespace if isinstance(grouped_native_function, NativeFunction) else grouped_native_function.functional.namespace
            ns_grouped_native_functions[namespace].append(grouped_native_function)
        dispatch_namespace = str(dispatch_key).lower()
        gen_dispatch_helpers: bool = dispatch_key != DispatchKey.CompositeImplicitAutogradNestedTensor
        dispatch_definitions = get_native_function_definitions(fm=fm, grouped_native_functions=grouped_native_functions, dispatch_key=dispatch_key, backend_idx=backend_index, selector=selector, rocm=rocm, symint=True, skip_dispatcher_op_registration=skip_dispatcher_op_registration, gen_dispatch_helpers=gen_dispatch_helpers)
        fm.write_with_template(f'Register{dispatch_key}.cpp', 'RegisterDispatchKey.cpp', lambda : {'extra_cuda_headers': extra_cuda_headers if is_cuda_dispatch_key(dispatch_key) else '', 'external_backend_headers': '', 'dispatch_headers': dest.gen_registration_headers(backend_index, per_operator_headers, rocm), 'ops_headers': operator_headers(), 'dispatch_helpers': '', 'dispatch_definitions': dispatch_definitions})
        for g in structured_native_functions:
            if not g.out.ufunc_inner_loop or not is_ufunc_dispatch_key(dispatch_key):
                continue
            name = g.functional.func.name.name
            if dispatch_key is DispatchKey.CPU:
                assert fm is cpu_fm
                fm.write_with_template(f'UfuncCPU_{name}.cpp', 'UfuncCPU.cpp', lambda : {'meta_declaration': compute_meta_function_declaration(g), 'native_declaration': dest.compute_native_function_declaration(g, backend_indices[dispatch_key]), 'native_definitions': dest.compute_ufunc_cpu(g)})
                cpu_vec_fm.write_with_template(f'UfuncCPUKernel_{name}.cpp', 'UfuncCPUKernel.cpp', lambda : {'name': name, 'native_definitions': dest.compute_ufunc_cpu_kernel(g)})
            elif dispatch_key is DispatchKey.CUDA:
                cuda_headers = '#include <ATen/native/cuda/Loops.cuh>'
                if rocm:
                    cuda_headers = '#include <ATen/native/hip/Loops.cuh>'
                fm.write_with_template(f'UfuncCUDA_{name}.cu', 'UfuncCUDA.cu', lambda : {'name': name, 'cuda_headers': cuda_headers, 'meta_declaration': compute_meta_function_declaration(g), 'native_declaration': dest.compute_native_function_declaration(g, backend_indices[dispatch_key]), 'native_definitions': dest.compute_ufunc_cuda(g)})
            else:
                raise AssertionError(f'unrecognized {dispatch_key} for ufunc')
        del fm

    def gen_backend_select() -> Dict[str, List[str]]:
        if False:
            while True:
                i = 10
        relevant_fns = [fn for fn in native_functions if needs_backend_select(fn, selector)]
        return {'ops_headers': [f'#include <ATen/ops/{fn.root_name}_ops.h>' for fn in relevant_fns], 'backend_select_method_definitions': list(mapMaybe(ComputeBackendSelect(Target.DEFINITION, selector), relevant_fns)), 'backend_select_function_registrations': list(mapMaybe(ComputeBackendSelect(Target.REGISTRATION, selector), relevant_fns))}
    cpu_fm.write('RegisterBackendSelect.cpp', gen_backend_select)
    schema_selector = selector
    if force_schema_registration:
        schema_selector = SelectiveBuilder.get_nop_selector()
    (aten_schema_registrations, schema_registrations) = get_native_function_schema_registrations(native_functions=native_functions, schema_selector=schema_selector)
    cpu_fm.write('RegisterSchema.cpp', lambda : {'aten_schema_registrations': [] if skip_dispatcher_op_registration else aten_schema_registrations, 'schema_registrations': [] if skip_dispatcher_op_registration else schema_registrations})

    def key_func(fn: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> str:
        if False:
            return 10
        return fn.root_name
    cpu_fm.write_sharded('Operators.cpp', native_functions, key_fn=key_func, env_callable=lambda fn: {'operator_headers': [f'#include <ATen/ops/{fn.root_name}.h>'], 'definitions': [ComputeOperators(Target.DEFINITION, static_dispatch_backend_indices=static_dispatch_idx)(fn)]}, base_env={'static_dispatch_extra_headers': static_dispatch_extra_headers(static_dispatch_idx)}, num_shards=5, sharded_keys={'operator_headers', 'definitions', 'static_dispatch_extra_headers'})
    cpu_fm.write('Functions.cpp', lambda : {})
    core_fm.write('TensorMethods.cpp', lambda : {})
    core_fm.write('ATenOpList.cpp', lambda : {'aten_ops': list(mapMaybe(compute_aten_op, native_functions))})

    def functionalization_env_callable(g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> Dict[str, List[str]]:
        if False:
            i = 10
            return i + 15

        def gen_op_headers(g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> List[str]:
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(g, NativeFunctionsViewGroup):
                headers = [f'#include <ATen/ops/{g.view.root_name}_native.h>', f'#include <ATen/ops/{g.view.root_name}_ops.h>']
                if g.view_copy is not None:
                    headers += [f'#include <ATen/ops/{g.view_copy.root_name}_native.h>', f'#include <ATen/ops/{g.view_copy.root_name}_ops.h>']
                return headers
            elif isinstance(g, NativeFunctionsGroup):
                headers = [f'#include <ATen/ops/{g.functional.root_name}_native.h>', f'#include <ATen/ops/{g.functional.root_name}_ops.h>', f'#include <ATen/ops/{g.out.root_name}_native.h>', f'#include <ATen/ops/{g.out.root_name}_ops.h>']
                if g.inplace is not None:
                    headers += [f'#include <ATen/ops/{g.inplace.root_name}_native.h>', f'#include <ATen/ops/{g.inplace.root_name}_ops.h>']
                if g.mutable is not None:
                    headers += [f'#include <ATen/ops/{g.mutable.root_name}_native.h>', f'#include <ATen/ops/{g.mutable.root_name}_ops.h>']
                return headers
            else:
                return [f'#include <ATen/ops/{g.root_name}_native.h>', f'#include <ATen/ops/{g.root_name}_ops.h>']
        return {'ops_headers': gen_op_headers(g), 'func_definitions': gen_functionalization_definition(selector, g), 'func_registrations': gen_functionalization_registration(selector, g, backend_indices[DispatchKey.CompositeImplicitAutograd])}
    all_groups: List[Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]] = list(structured_native_functions) + list(view_groups)
    structured_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda g: list(g.functions()), structured_native_functions)}
    view_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda g: list(g.functions()), view_groups)}
    for f in native_functions:
        if f.func.name not in structured_map and f.func.name not in view_map:
            all_groups.append(f)
    cpu_fm.write_sharded('RegisterFunctionalization.cpp', all_groups, key_fn=key_func, env_callable=functionalization_env_callable, num_shards=4, sharded_keys={'ops_headers', 'func_definitions', 'func_registrations', 'func_add_back_views_definitions', 'func_add_back_views_registrations'})
    cpu_fm.write('FunctionalInverses.h', lambda : {'view_inverse_declarations': list(mapMaybe(lambda g: gen_functionalization_view_inverse_declaration(selector, g), view_groups))})
    cpu_fm.write('CompositeViewCopyKernels.cpp', lambda : {'ops_headers': ['\n'.join((f'#include <ATen/ops/{f.root_name}_ops.h>\n#include <ATen/ops/{f.root_name}_native.h>' for f in ([g.view] if g.view_copy is None else [g.view, g.view_copy]))) for g in view_groups] + ['\n'.join((f'#include <ATen/ops/{f.root_name}_ops.h>' for f in [g.inplace, g.mutable, g.functional] if f is not None and 'generated' not in f.tags)) for g in structured_native_functions], 'CompositeViewCopyKernel_Definitions': list(mapMaybe(GenCompositeViewCopyKernel(backend_indices[DispatchKey.CompositeExplicitAutogradNonFunctional]), view_groups)), 'GeneratedCompositeFunctional_Definitions': list(mapMaybe(gen_composite_functional_kernel, structured_native_functions)), 'GeneratedCompositeOut_Definitions': list(mapMaybe(gen_composite_out_kernel, structured_native_functions))})

def gen_declarations_yaml(cpu_fm: FileManager, native_functions: Sequence[NativeFunction]) -> None:
    if False:
        for i in range(10):
            print('nop')
    cpu_fm.write('Declarations.yaml', lambda : format_yaml([compute_declaration_yaml(f) for f in native_functions]))

def get_torchgen_root() -> pathlib.Path:
    if False:
        i = 10
        return i + 15
    "\n    If you're depending on torchgen out-of-tree, you can use the root to figure\n    out the path to native_functions.yaml\n    "
    return pathlib.Path(__file__).parent.resolve()

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Generate ATen source files')
    parser.add_argument('-s', '--source-path', help='path to source directory for ATen', default='aten/src/ATen')
    parser.add_argument('-o', '--output-dependencies', help='output a list of dependencies into the given file and exit')
    parser.add_argument('--dry-run', action='store_true', help='run without writing any files (still updates outputs)')
    parser.add_argument('--per-operator-headers', action='store_true', help='generate separate headers per operator in ATen/ops')
    parser.add_argument('-d', '--install-dir', '--install_dir', help='output directory', default='build/aten/src/ATen')
    parser.add_argument('--rocm', action='store_true', help='reinterpret CUDA as ROCm/HIP and adjust filepaths accordingly')
    parser.add_argument('--mps', action='store_true', help='Generate MPS registration code when set')
    parser.add_argument('--op-registration-whitelist', '--op_registration_whitelist', nargs='*', help='filter op registrations by the whitelist (if set); each item is `namespace`::`operator name` without overload name; e.g.: aten::empty aten::conv2d ...')
    parser.add_argument('--op-selection-yaml-path', '--op_selection_yaml_path', help='Provide a path to the operator selection (for custom build) YAML that contains the information about the set of selected operators and their categories (training, ...). Each operator is either a full operator name with overload or just a bare operator name. The operator names also contain the namespace prefix (e.g. aten::)')
    parser.add_argument('--backend-whitelist', '--backend_whitelist', nargs='*', help='filter dispatch backend by the whitelist (if set), e.g.: CPU CUDA QuantizedCPU ...')
    parser.add_argument('--static-dispatch-backend', '--static_dispatch_backend', nargs='*', help='generate static dispatch code for the specific backend (if set)')
    parser.add_argument('--skip-dispatcher-op-registration', '--skip_dispatcher_op_registration', action='store_true', help='Avoid registering operators into the dispatcher.')
    parser.add_argument('--force-schema-registration', '--force_schema_registration', action='store_true', help='force it to generate schema-only registrations for all ops, includingthose that are not listed on --op-registration-whitelist')
    parser.add_argument('--generate', type=str, nargs='*', choices=['headers', 'sources', 'declarations_yaml'], default=['headers', 'sources', 'declarations_yaml'], help='Generate only a subset of files')
    options = parser.parse_args()
    selector = get_custom_build_selector(options.op_registration_whitelist, options.op_selection_yaml_path)
    native_yaml_path = os.path.join(options.source_path, 'native/native_functions.yaml')
    tags_yaml_path = os.path.join(options.source_path, 'native/tags.yaml')
    from torchgen.model import dispatch_keys
    ignore_keys = set()
    if not options.mps:
        ignore_keys.add(DispatchKey.MPS)
        if DispatchKey.MPS in dispatch_keys:
            del dispatch_keys[dispatch_keys.index(DispatchKey.MPS)]
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path, ignore_keys)
    valid_tags = _GLOBAL_PARSE_TAGS_YAML_CACHE[tags_yaml_path]
    (native_functions, backend_indices) = (parsed_yaml.native_functions, parsed_yaml.backend_indices)
    grouped_native_functions = get_grouped_native_functions(native_functions)
    structured_native_functions = [g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)]
    native_functions_with_view_groups = get_grouped_by_view_native_functions(native_functions)
    view_groups = [g for g in native_functions_with_view_groups if isinstance(g, NativeFunctionsViewGroup)]
    core_install_dir = f'{options.install_dir}/core'
    pathlib.Path(core_install_dir).mkdir(parents=True, exist_ok=True)
    ops_install_dir = f'{options.install_dir}/ops'
    pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)
    core_fm = make_file_manager(options=options, install_dir=core_install_dir)
    cpu_fm = make_file_manager(options=options)
    cpu_vec_fm = make_file_manager(options=options)
    cuda_fm = make_file_manager(options=options)
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)
    functions_keys = {DispatchKey.CPU, DispatchKey.CUDA, DispatchKey.CompositeImplicitAutograd, DispatchKey.CompositeImplicitAutogradNestedTensor, DispatchKey.CompositeExplicitAutograd, DispatchKey.CompositeExplicitAutogradNonFunctional, DispatchKey.Meta}
    if options.mps:
        functions_keys.add(DispatchKey.MPS)
    if options.backend_whitelist:
        dispatch_keys = [k for k in dispatch_keys if is_generic_dispatch_key(k) or str(k) in options.backend_whitelist]
    static_dispatch_idx: List[BackendIndex] = []
    if options.static_dispatch_backend:
        static_dispatch_idx = [backend_indices[DispatchKey.parse(key)] for key in options.static_dispatch_backend]
        for key in options.static_dispatch_backend:
            dp_key = DispatchKey.parse(key)
            if dp_key not in functions_keys:
                functions_keys.add(dp_key)
    if 'sources' in options.generate:
        gen_source_files(native_functions=native_functions, grouped_native_functions=grouped_native_functions, structured_native_functions=structured_native_functions, view_groups=view_groups, selector=selector, static_dispatch_idx=static_dispatch_idx, backend_indices=backend_indices, core_fm=core_fm, cpu_fm=cpu_fm, cpu_vec_fm=cpu_vec_fm, cuda_fm=cuda_fm, dispatch_keys=dispatch_keys, functions_keys=functions_keys, rocm=options.rocm, force_schema_registration=options.force_schema_registration, per_operator_headers=options.per_operator_headers, skip_dispatcher_op_registration=options.skip_dispatcher_op_registration)
    if 'headers' in options.generate:
        gen_headers(native_functions=native_functions, valid_tags=valid_tags, grouped_native_functions=grouped_native_functions, structured_native_functions=structured_native_functions, static_dispatch_idx=static_dispatch_idx, selector=selector, backend_indices=backend_indices, core_fm=core_fm, cpu_fm=cpu_fm, cuda_fm=cuda_fm, ops_fm=ops_fm, dispatch_keys=dispatch_keys, functions_keys=functions_keys, rocm=options.rocm, per_operator_headers=options.per_operator_headers)
    if 'declarations_yaml' in options.generate:
        gen_declarations_yaml(native_functions=native_functions, cpu_fm=cpu_fm)
    if options.output_dependencies:
        depfile_path = pathlib.Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem
        for (fm, prefix) in [(cpu_fm, ''), (cpu_vec_fm, 'cpu_vec_'), (core_fm, 'core_'), (cuda_fm, 'cuda_'), (ops_fm, 'ops_')]:
            varname = prefix + depfile_stem
            path = depfile_path.parent / (prefix + depfile_name)
            fm.write_outputs(varname, str(path))
if __name__ == '__main__':
    main()