from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import dispatch_strategy, gen_differentiable_outputs, NativeFunctionWithDifferentiabilityInfo
from torchgen.api.types import BaseCType, Binding, boolT, ConstRefCType, CType, DispatcherSignature, intArrayRefT, longT, OptionalCType, symIntArrayRefT, SymIntT, tensorT
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import NativeFunction, SchemaKind, SelfArgument, TensorOptionsArguments, Type
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import get_return_value, MANUAL_AUTOGRAD, tie_return_values, type_wrapper_name
VIEW_FUNCTIONS_WITH_METADATA_CHANGE = ['view_as_complex', 'view_as_real', '_conj', '_neg_view', '_nested_view_from_buffer']
VIEW_FUNCTIONS = {'numpy_T': 'self', 'alias': 'self', 'as_strided': 'self', 'diagonal': 'self', 'expand': 'self', 'permute': 'self', 'select': 'self', 'slice': 'self', 'split': 'self', 'split_with_sizes': 'self', 'squeeze': 'self', 't': 'self', 'transpose': 'self', 'unfold': 'self', 'unsqueeze': 'self', 'flatten': 'self', 'view': 'self', 'unbind': 'self', '_indices': 'self', '_values': 'self', 'indices': 'self', 'values': 'self', 'crow_indices': 'self', 'col_indices': 'self', 'ccol_indices': 'self', 'row_indices': 'self', 'sparse_coo_tensor_with_dims_and_tensors': 'values', '_reshape_alias': 'self', '_test_autograd_multiple_dispatch_view': 'self'}
for key in VIEW_FUNCTIONS_WITH_METADATA_CHANGE:
    VIEW_FUNCTIONS[key] = 'self'
RETURNS_VIEWS_OF_INPUT = set(VIEW_FUNCTIONS.keys()).union({'chunk', 'detach', 'contiguous', 'reshape', 'reshape_as', 'expand_as', 'view_as', 'real', 'imag', 'narrow', 'movedim', 'tensor_split', 'swapdims', 'swapaxes', 'mT', 'mH', 'adjoint', 'matrix_H'})
ALL_VIEW_FUNCTIONS = {**VIEW_FUNCTIONS, '_unsafe_view': 'self'}
ARRAYREF_TO_VEC = CodeTemplate('auto ${vec} = ${arg}.vec();\n')
OPTIONAL_TO_VAL = CodeTemplate('auto ${val} = ${arg}.value_or(${default});\n')
CALL_DISPATCH = CodeTemplate('at::_ops::${unambiguous_name}::call(${unpacked_args})')
SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE = CodeTemplate('std::function<at::Tensor(const at::Tensor&)> func=nullptr;\nif (${is_view_with_metadata_change} || !self.unsafeGetTensorImpl()->support_as_strided() ||\n    c10::AutogradState::get_tls_state().get_view_replay_enabled()) {\n  ${replay_view_func}\n}\n')
REPLAY_VIEW_LAMBDA_FUNC = CodeTemplate('func = [=](const at::Tensor& ${input_base}) {\n  return ${replay_view_call};\n};\n')
METHOD_DEFINITION = CodeTemplate('${return_type} ${type_wrapper_name}(${formals}) {\n  ${type_definition_body}\n}\n')
WRAPPER_REGISTRATION = CodeTemplate('m.impl("${unqual_operator_name_with_overload}",\n       TORCH_FN(${class_type}::${type_wrapper_name})\n);\n')
AUTOGRAD_NOT_IMPLEMENTED_REGISTRATION = CodeTemplate('m.impl("${unqual_operator_name_with_overload}", torch::autograd::autogradNotImplementedFallback());\n')
INPLACE_REDISPATCH = CodeTemplate('{\n  at::AutoDispatchBelowADInplaceOrView guard;\n  at::_ops::${unambiguous_name}::redispatch(${unpacked_args});\n}\n')
ASSIGN_RETURN_VALUE = CodeTemplate('${return_values} = ${rhs_value};\n')
VIEW_REDISPATCH = CodeTemplate('${assign_return_values} ([&]() {\n  at::AutoDispatchBelowADInplaceOrView guard;\n  return at::_ops::${unambiguous_name}::redispatch(${unpacked_args});\n})();\n')
TMP_VAR = '_tmp'

def is_tensor_type(t: Type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return t.is_tensor_like() and t.is_list_like() is None

def is_tensor_list_type(t: Type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return t.is_tensor_like() and t.is_list_like() is not None
UNPACK_TENSOR = CodeTemplate('auto${ref} ${arg_name}_ = unpack${suffix}(${arg_name}, "${arg_name}", ${arg_pos});')

def unpacked_name(arg_name: str) -> str:
    if False:
        return 10
    return arg_name + '_'

@with_native_function
def unpack_args(f: NativeFunction) -> Tuple[List[str], List[Binding]]:
    if False:
        i = 10
        return i + 15
    body: List[str] = []
    unpacked_bindings: List[Binding] = []
    bindings = [r for a in f.func.schema_order_arguments() for r in cpp.argument(a, method=False, symint=True, cpp_no_default_args=set(), faithful=False, has_tensor_options=False)]
    for (i, binding) in enumerate(bindings):
        assert not isinstance(binding.argument, SelfArgument)
        if isinstance(binding.argument, TensorOptionsArguments):
            raise RuntimeError("VariableKernel shouldn't take TensorOptions")
        is_nullable = binding.argument.type.is_nullable()
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue
        is_tensor_list = is_tensor_list_type(binding.argument.type)
        ref = not is_nullable and (not is_tensor_list)
        suffix = '_opt' if is_nullable and (not is_tensor_list) else ''
        body.append(UNPACK_TENSOR.substitute(arg_name=binding.name, arg_pos=i, suffix=suffix, ref='&' if ref else ''))
        unpacked_bindings.append(Binding(name=unpacked_name(binding.name), nctype=binding.nctype, argument=binding.argument, default=binding.default))
    return (body, unpacked_bindings)

def get_base_name(f: NativeFunction) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f.func.name.name.base

def get_view_info(f: NativeFunction) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    base_name = get_base_name(f)
    view_info = VIEW_FUNCTIONS.get(base_name, None)
    if view_info is None and base_name in RETURNS_VIEWS_OF_INPUT:
        view_info = 'self'
    return view_info

def emit_view_call(f: NativeFunction, input_base: str, unpacked_args: Sequence[str]) -> str:
    if False:
        i = 10
        return i + 15
    return CALL_DISPATCH.substitute(unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=unpacked_args)

def emit_view_lambda(f: NativeFunction, unpacked_bindings: List[Binding]) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate an additional lambda function to recover views in backward when as_strided is not supported.\n    See Note [View + Inplace update for base tensor] and [View + Inplace update for view tensor] for more details.\n    '
    input_base = 'input_base'
    replay_view_func = ''
    updated_unpacked_args: List[str] = []
    known_view_arg_simple_types: List[CType] = [BaseCType(longT), OptionalCType(BaseCType(longT)), BaseCType(SymIntT), OptionalCType(BaseCType(SymIntT)), BaseCType(boolT), BaseCType(intArrayRefT), BaseCType(symIntArrayRefT), ConstRefCType(BaseCType(tensorT))]
    for unpacked_binding in unpacked_bindings:
        (arg, arg_type) = (unpacked_binding.name, unpacked_binding.nctype.type)
        if arg == 'self_':
            updated_unpacked_args.append(input_base)
            continue
        if arg_type not in known_view_arg_simple_types:
            known_types_str = ', '.join([str(t) for t in known_view_arg_simple_types])
            raise TypeError(f'You are adding an {arg_type} {arg} argument to op {cpp.name(f.func)} in addition to known types: {known_types_str}. Please update the list or materialize it so that it can be closed over by value, also add a test in pytorch/xla/test/test_operations.py where this code is exercised.')
        if arg_type == BaseCType(intArrayRefT) or arg_type == BaseCType(symIntArrayRefT):
            arg_vec = arg + '_vec'
            replay_view_func += ARRAYREF_TO_VEC.substitute(arg=arg, vec=arg_vec)
            updated_unpacked_args.append(arg_vec)
        elif arg_type == OptionalCType(BaseCType(longT)):
            arg_value = arg + '_val'
            replay_view_func += OPTIONAL_TO_VAL.substitute(arg=arg, val=arg_value, default='0')
            updated_unpacked_args.append(arg_value)
        elif (arg == 'nested_size_' or arg == 'nested_strides_' or arg == 'offsets_') and arg_type == ConstRefCType(BaseCType(tensorT)):
            updated_unpacked_args.append(arg[:-1])
        else:
            updated_unpacked_args.append(arg)
    replay_view_call = emit_view_call(f, input_base, updated_unpacked_args)
    replay_view_func += REPLAY_VIEW_LAMBDA_FUNC.substitute(input_base=input_base, replay_view_call=replay_view_call)
    is_view_with_metadata_change = 'true' if cpp.name(f.func) in VIEW_FUNCTIONS_WITH_METADATA_CHANGE else 'false'
    return SETUP_REPLAY_VIEW_IF_NOT_SUPPORT_AS_STRIDED_OR_VIEW_WITH_METADATA_CHANGE.substitute(is_view_with_metadata_change=is_view_with_metadata_change, replay_view_func=replay_view_func)

def emit_view_body(fn: NativeFunctionWithDifferentiabilityInfo, var: str) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    f = fn.func
    base_name = get_base_name(f)
    view_info = get_view_info(f)
    call = ''
    differentiable_outputs = gen_differentiable_outputs(fn)
    differentiable_output_vars = {r.name for r in differentiable_outputs}
    if not isinstance(view_info, str):
        raise TypeError(f'The view info should be a string for {base_name}, but it is: {view_info}')
    if len(differentiable_output_vars) == 0:
        rhs_value = f'as_view({view_info}, {var}, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false)'
    elif len(differentiable_output_vars) == 1:
        return_info = differentiable_outputs[0]
        if not is_tensor_type(return_info.type) and (not is_tensor_list_type(return_info.type)):
            raise RuntimeError(f'{base_name} that return differentiable views can only return Tensor or Tensor[]')

        def get_creation_meta_in_mode(original: str) -> str:
            if False:
                while True:
                    i = 10
            creation_meta_with_grad_mode = f'(at::GradMode::is_enabled() ? {original} : CreationMeta::NO_GRAD_MODE)'
            return f'InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : {creation_meta_with_grad_mode}'
        if is_tensor_list_type(return_info.type):
            creation_meta = get_creation_meta_in_mode('CreationMeta::MULTI_OUTPUT_NODE')
            call += f'as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ {creation_meta});'
            rhs_value = f'std::move({var})'
        else:
            (_, unpacked_bindings) = unpack_args(f)
            call += emit_view_lambda(f, unpacked_bindings)
            creation_meta = get_creation_meta_in_mode('CreationMeta::DEFAULT')
            rhs_value = f'as_view(/* base */ {view_info}, /* output */ {var}, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ {creation_meta})'
    else:
        raise RuntimeError('Function that return multiple differentiable output when at least one of them is view is not supported.')
    return (call, rhs_value)

def modifies_arguments(f: NativeFunction) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return f.func.kind() in [SchemaKind.inplace, SchemaKind.out]

@with_native_function_with_differentiability_info
def emit_inplace_or_view_body(fn: NativeFunctionWithDifferentiabilityInfo) -> List[str]:
    if False:
        while True:
            i = 10
    f = fn.func
    inplace_view_body: List[str] = []
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    dispatcher_exprs = dispatcher_sig.exprs()
    dispatch_key_set = 'ks & c10::after_ADInplaceOrView_keyset'
    redispatch_args = ', '.join([dispatch_key_set] + [a.expr for a in dispatcher_exprs])
    if modifies_arguments(f):
        inplace_view_body.append(INPLACE_REDISPATCH.substitute(unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=redispatch_args))
        for r in cpp.return_names(f):
            inplace_view_body.append(f'increment_version({r});')
    else:
        assert get_view_info(f) is not None
        inplace_view_body.append(VIEW_REDISPATCH.substitute(assign_return_values='auto ' + TMP_VAR + ' = ', unambiguous_name=f.func.name.unambiguous_name(), unpacked_args=redispatch_args))
        (call, rhs_value) = emit_view_body(fn, TMP_VAR)
        inplace_view_body.append(call)
        assert rhs_value is not None
        inplace_view_body.append(ASSIGN_RETURN_VALUE.substitute(return_values=tie_return_values(f), rhs_value=rhs_value))
    if f.func.returns:
        inplace_view_body.append(f'return {get_return_value(f)};')
    return inplace_view_body

@with_native_function
def gen_formals(f: NativeFunction) -> str:
    if False:
        i = 10
        return i + 15
    return ', '.join(['c10::DispatchKeySet ks'] + [f"{cpp.argument_type(a, binds='__placeholder__', symint=True).cpp_type()} {a.name}" for a in f.func.schema_order_arguments()])

@with_native_function_with_differentiability_info
def inplace_or_view_method_definition(fn: NativeFunctionWithDifferentiabilityInfo) -> Optional[str]:
    if False:
        while True:
            i = 10
    f = fn.func
    if get_view_info(f) is None and (not modifies_arguments(f) or len(f.func.returns) == 0):
        return None
    return METHOD_DEFINITION.substitute(return_type=cpp.returns_type(f.func.returns, symint=True).cpp_type(), type_wrapper_name=type_wrapper_name(f), formals=gen_formals(f), type_definition_body=emit_inplace_or_view_body(fn))

@with_native_function_with_differentiability_info
def inplace_or_view_method_registration(fn: NativeFunctionWithDifferentiabilityInfo) -> Optional[str]:
    if False:
        return 10
    f = fn.func
    if get_view_info(f) is None and (not modifies_arguments(f) or len(f.func.returns) == 0):
        return None
    return WRAPPER_REGISTRATION.substitute(unqual_operator_name_with_overload=f.func.name, type_wrapper_name=type_wrapper_name(f), class_type='ADInplaceOrView')

def use_derived(fn: NativeFunctionWithDifferentiabilityInfo) -> bool:
    if False:
        for i in range(10):
            print('nop')
    f = fn.func
    name = cpp.name(f.func)
    return name not in MANUAL_AUTOGRAD and dispatch_strategy(fn) == 'use_derived'

def gen_inplace_or_view_type_env(fn: NativeFunctionWithDifferentiabilityInfo) -> Dict[str, List[str]]:
    if False:
        i = 10
        return i + 15
    definition = inplace_or_view_method_definition(fn)
    registration = inplace_or_view_method_registration(fn)
    return {'ops_headers': [f'#include <ATen/ops/{fn.func.root_name}_ops.h>'] if definition is not None else [], 'inplace_or_view_method_definitions': [definition] if definition is not None else [], 'inplace_or_view_wrapper_registrations': [registration] if registration is not None else []}

def gen_inplace_or_view_type(out: str, native_yaml_path: str, tags_yaml_path: str, fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo], template_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    num_shards = 2
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded('ADInplaceOrViewType.cpp', [fn for fn in fns_with_infos if use_derived(fn)], key_fn=lambda fn: fn.func.root_name, base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/ADInplaceOrViewType.cpp'}, env_callable=gen_inplace_or_view_type_env, num_shards=2, sharded_keys={'ops_headers', 'inplace_or_view_method_definitions', 'inplace_or_view_wrapper_registrations'})