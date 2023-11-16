import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import Argument, BaseTy, BaseType, FunctionSchema, ListType, NativeFunction, OptionalType, Return, SchemaKind, Type
from torchgen.utils import mapMaybe

def is_tensor(typ: Type) -> bool:
    if False:
        while True:
            i = 10
    return isinstance(typ, BaseType) and typ.name == BaseTy.Tensor

def is_optional_tensor(typ: Type) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(typ, OptionalType) and is_tensor(typ.elem)

def is_tensor_list(typ: Type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return isinstance(typ, ListType) and is_tensor(typ.elem)

def unwrap_tensor(name: str, cur_level_var: str) -> List[str]:
    if False:
        print('Hello World!')
    result = f'    Tensor {name}_value;\n    optional<int64_t> {name}_bdim;\n    std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}, {cur_level_var});'
    return textwrap.dedent(result).split('\n')

def unwrap_optional_tensor(name: str, cur_level_var: str) -> List[str]:
    if False:
        print('Hello World!')
    result = f'    optional<Tensor> {name}_value;\n    optional<int64_t> {name}_bdim;\n    if ({name}) {{\n        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});\n    }}'
    return textwrap.dedent(result).split('\n')

def gen_unwraps(flat_arguments: Sequence[Argument], cur_level_var: str) -> Tuple[str, List[str]]:
    if False:
        while True:
            i = 10
    arg_names = [a.name for a in flat_arguments]
    arg_types = [a.type for a in flat_arguments]
    tensors = [name for (typ, name) in zip(arg_types, arg_names) if is_tensor(typ)]
    optional_tensors = [name for (typ, name) in zip(arg_types, arg_names) if is_optional_tensor(typ)]
    unwraps = []
    for tensor in tensors:
        unwraps += unwrap_tensor(tensor, cur_level_var)
    for opt_tensor in optional_tensors:
        unwraps += unwrap_optional_tensor(opt_tensor, cur_level_var)
    unwrap_code = '\n'.join(unwraps)
    unwrapped_arg_list = []
    for arg in arg_names:
        if arg in tensors or arg in optional_tensors:
            unwrapped_arg_list += [f'{arg}_value', f'{arg}_bdim']
        else:
            unwrapped_arg_list.append(arg)
    return (unwrap_code, unwrapped_arg_list)

def gen_case_where_all_bdims_are_none(outer_sig: DispatcherSignature, schema: FunctionSchema, cur_level_var: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    conditions = []
    flat_args = schema.arguments.flat_all
    for arg in flat_args:
        if not arg.type.is_tensor_like():
            continue
        conditions.append(f'!isBatchedAtLevel({arg.name}, {cur_level_var})')
    sig = DispatcherSignature.from_schema(schema)
    translated_args = ', '.join((e.expr for e in translate(outer_sig.arguments(), sig.arguments())))
    return f"if ({' && '.join(conditions)}) {{\n  return at::_ops::{sig.func.name.unambiguous_name()}::call({translated_args});\n}}"

def gen_returns(returns: Tuple[Return, ...], cur_level_var: str, results_var: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    idx = 0
    wrapped_returns = []
    for ret in returns:
        if is_tensor(ret.type):
            wrapped_returns.append(f'makeBatched(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})')
            idx += 2
        elif is_tensor_list(ret.type):
            wrapped_returns.append(f'makeBatchedVector(std::get<{idx}>({results_var}), std::get<{idx + 1}>({results_var}), {cur_level_var})')
            idx += 2
        else:
            wrapped_returns.append(f'std::get<{idx}>({results_var})')
            idx += 1
    if len(wrapped_returns) == 1:
        result = f'return {wrapped_returns[0]};'
    else:
        result = f"return std::make_tuple({', '.join(wrapped_returns)});"
    return result

def accepts_at_least_one_tensor_input(schema: FunctionSchema) -> bool:
    if False:
        return 10
    return any((a.type.is_tensor_like() for a in schema.arguments.flat_all))

def is_mutated_arg(argument: Argument) -> bool:
    if False:
        i = 10
        return i + 15
    return argument.annotation is not None and argument.annotation.is_write

def gen_vmap_inplace_plumbing(native_function: NativeFunction) -> Optional[str]:
    if False:
        print('Hello World!')
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns
    assert schema.kind() == SchemaKind.inplace
    if not is_mutated_arg(schema.arguments.flat_all[0]):
        return None
    if not len([arg for arg in schema.arguments.flat_all if is_mutated_arg(arg)]) == 1:
        return None
    if len(returns) == 0:
        return None
    if not all((is_tensor(ret.type) or is_tensor_list(ret.type) for ret in returns)):
        return None
    if not accepts_at_least_one_tensor_input(schema):
        return None
    cur_level_var = 'cur_level'
    (unwraps, unwrapped_arg_list) = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)
    return f"""template <typename batch_rule_t, batch_rule_t batch_rule>\n{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{\n  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);\n  auto maybe_layer = maybeCurrentDynamicLayer();\n  vmap_check_escaped(maybe_layer, "gen_vmap_inplace_plumbing");\n  int64_t {cur_level_var} = maybe_layer->layerId();\n{textwrap.indent(bdims_all_none_case, '  ')}\n{textwrap.indent(unwraps, '  ')}\n  batch_rule({', '.join(unwrapped_arg_list)});\n  return {schema.arguments.flat_all[0].name};\n}}"""

def gen_vmap_plumbing_no_returns(native_function: NativeFunction) -> str:
    if False:
        print('Hello World!')
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    cur_level_var = 'cur_level'
    (unwraps, unwrapped_arg_list) = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)
    return f"""template <typename batch_rule_t, batch_rule_t batch_rule>\n{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{\n  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);\n  auto maybe_layer = maybeCurrentDynamicLayer();\n  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing_no_returns");\n  int64_t {cur_level_var} = maybe_layer->layerId();\n{textwrap.indent(bdims_all_none_case, '  ')}\n{textwrap.indent(unwraps, '  ')}\n  batch_rule({', '.join(unwrapped_arg_list)});\n}}"""

def gen_vmap_plumbing(native_function: NativeFunction) -> Optional[str]:
    if False:
        while True:
            i = 10
    schema = native_function.func
    sig = DispatcherSignature.from_schema(schema)
    returns = schema.returns
    if not accepts_at_least_one_tensor_input(schema):
        return None
    if len(returns) == 0:
        return gen_vmap_plumbing_no_returns(native_function)
    if not all((ret.type.is_tensor_like() for ret in returns)):
        return None
    if 'inplace_view' in native_function.tags:
        return None
    if schema.kind() == SchemaKind.inplace:
        return gen_vmap_inplace_plumbing(native_function)
    if schema.kind() != SchemaKind.functional:
        return None
    results_var = 'results'
    cur_level_var = 'cur_level'
    (unwraps, unwrapped_arg_list) = gen_unwraps(schema.arguments.flat_all, cur_level_var)
    bdims_all_none_case = gen_case_where_all_bdims_are_none(sig, schema, cur_level_var)
    wrapped_returns = gen_returns(returns, cur_level_var, results_var)
    return f"""template <typename batch_rule_t, batch_rule_t batch_rule>\n{sig.decl(name=schema.name.unambiguous_name() + '_generated_plumbing')} {{\n  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);\n  auto maybe_layer = maybeCurrentDynamicLayer();\n  vmap_check_escaped(maybe_layer, "gen_vmap_plumbing");\n  int64_t {cur_level_var} = maybe_layer->layerId();\n{textwrap.indent(bdims_all_none_case, '  ')}\n{textwrap.indent(unwraps, '  ')}\n  auto {results_var} = batch_rule({', '.join(unwrapped_arg_list)});\n  {wrapped_returns}\n}}"""

@dataclass(frozen=True)
class ComputeBatchRulePlumbing:

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if False:
            print('Hello World!')
        opname = str(f.func.name)
        result = gen_vmap_plumbing(f)
        return result

def gen_all_vmap_plumbing(native_functions: Sequence[NativeFunction]) -> str:
    if False:
        i = 10
        return i + 15
    body = '\n'.join(list(mapMaybe(ComputeBatchRulePlumbing(), native_functions)))
    return f'\n#pragma once\n#include <ATen/Operators.h>\n#include <ATen/functorch/PlumbingHelper.h>\n\nnamespace at {{ namespace functorch {{\n\n{body}\n\n}}}} // namespace at::functorch\n'