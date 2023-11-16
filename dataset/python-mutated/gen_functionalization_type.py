from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import BaseCType, Binding, CType, DispatcherSignature, FunctionalizationLambda, iTensorListRefT, NativeSignature, tensorListT, tensorT, VectorCType, ViewInverseSignature
from torchgen.context import method_with_native_function, native_function_manager, with_native_function, with_native_function_and
from torchgen.model import Argument, BackendIndex, BaseTy, BaseType, FunctionSchema, ListType, NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup, Return, SchemaKind, SelfArgument, TensorOptionsArguments
from torchgen.native_function_generation import INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY, MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT, OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
from torchgen.selective_build.selector import SelectiveBuilder
MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION = OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY + MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT + INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY + ['record_stream', 'resize_', 'resize_as_', '_fill_mem_eff_dropout_mask_']

@dataclass(frozen=True)
class GenCompositeViewCopyKernel:
    backend_index: BackendIndex

    @method_with_native_function
    def __call__(self, g: NativeFunctionsViewGroup) -> Optional[str]:
        if False:
            return 10
        if g.view_copy is None:
            return None
        metadata = self.backend_index.get_kernel(g.view_copy)
        assert metadata is not None
        if str(g.view_copy.func.name) == 'view_copy':
            assert metadata.kernel == 'view_copy_symint'
            return 'at::Tensor view_copy_symint(const at::Tensor & self, at::SymIntArrayRef size) {\n  c10::SymDimVector shape = infer_size_dv(size, self.sym_numel());\n  if (!at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape).has_value()) {\n    return self.reshape_symint(size);\n  } else {\n    auto output = at::_ops::view::call(self, size);\n    return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);\n  }\n}\n'
        view_copy_sig = NativeSignature(g.view_copy.func, symint=metadata.supports_symint())
        view_sig = DispatcherSignature(g.view.func)
        view_api_name = g.view.func.name.unambiguous_name()
        exprs = ', '.join([e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())])
        assert len(g.view.func.returns) == 1
        assert g.view.func.returns[0].type == BaseType(BaseTy.Tensor) or g.view.func.returns[0].type == ListType(BaseType(BaseTy.Tensor), None)
        if g.view.func.returns[0].type == BaseType(BaseTy.Tensor):
            return_cloned_output = '  return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);'
        else:
            return_cloned_output = f'  {view_copy_sig.returns_type().cpp_type()} out_clone;\n  for (const auto i : c10::irange(output.size())) {{\n    out_clone.push_back(output[i].clone(/*memory_format=*/at::MemoryFormat::Contiguous));\n  }}\n  return out_clone;'
        return f'\n{view_copy_sig.defn(name=metadata.kernel)} {{\n  auto output = at::_ops::{view_api_name}::call({exprs});\n  {return_cloned_output}\n}}\n'

def return_str(rets: Tuple[Return, ...], names: List[str]) -> str:
    if False:
        return 10
    assert len(rets) == len(names)
    if len(rets) == 0:
        return ''
    elif len(rets) == 1:
        return f'return {names[0]};'
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"

def modifies_arguments(f: NativeFunction) -> bool:
    if False:
        i = 10
        return i + 15
    return any((a.annotation is not None and a.annotation.is_write for a in f.func.arguments.flat_all))

def wrapper_name(func: FunctionSchema) -> str:
    if False:
        return 10
    if func.name.overload_name:
        return f'{cpp.name(func)}_{func.name.overload_name}'
    else:
        return cpp.name(func)

def is_tensor_like(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> bool:
    if False:
        print('Hello World!')
    return isinstance(a, SelfArgument) or (isinstance(a, Argument) and a.type.is_tensor_like())

def get_owning_type(t: CType) -> Tuple[CType, Callable[[str], str]]:
    if False:
        return 10
    if t == BaseCType(tensorListT):
        return (VectorCType(BaseCType(tensorT)), lambda x: f'{x}.vec()')
    if t == BaseCType(iTensorListRefT):
        return (VectorCType(BaseCType(tensorT)), lambda x: f'{{{x}.begin(), {x}.end()}}')
    return (t, lambda x: x)

def unwrap_tensor_args(sig: DispatcherSignature, *, is_view_op: bool) -> Tuple[str, List[Binding]]:
    if False:
        return 10
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            unwrapped_name = f'{arg.name}_'
            maybe_sync_input = '' if is_view_op else f'at::functionalization::impl::sync({arg.name});'
            (unwrapped_type, conversion_fn) = get_owning_type(arg.nctype.remove_const_ref().type)
            unwrapped_tensor_args.append(f'\n      {unwrapped_type.cpp_type()} {unwrapped_name};\n      if (at::functionalization::impl::isFunctionalTensor({arg.name})) {{\n        {maybe_sync_input}\n        {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});\n      }} else {{\n        {unwrapped_name} = {conversion_fn(arg.name)};\n      }}')
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    unwrap_tensor_args_str = '\n      '.join(unwrapped_tensor_args)
    return (unwrap_tensor_args_str, context)

def convert_to_meta_tensors(sig: DispatcherSignature) -> Tuple[str, List[Binding]]:
    if False:
        i = 10
        return i + 15
    context: List[Binding] = []
    unwrapped_tensor_args: List[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            a_ = arg.name
            unwrapped_name = f'{arg.name}_meta'
            unwrapped_tensor_args.append(f'auto {unwrapped_name} = to_meta({a_});')
            context.append(arg.with_name(unwrapped_name))
        else:
            context.append(arg)
    unwrap_tensor_args_str = '\n        '.join(unwrapped_tensor_args)
    return (unwrap_tensor_args_str, context)

def assert_view_op_properties(func: FunctionSchema) -> None:
    if False:
        while True:
            i = 10

    def is_alias(a: Argument) -> bool:
        if False:
            i = 10
            return i + 15
        return a.annotation is not None
    args = func.arguments.flat_non_out
    assert len(args) > 0 and args[0].type == BaseType(BaseTy.Tensor), f'In the functionalization codegen, we expect the first argument of every view operator to be a tensor,\nbut found an argument of type {str(args[0].type)} for operator: {str(func.name)}.'
    assert is_alias(args[0]) and (not any((is_alias(a) for a in args[1:]))), "In the functionalization codegen, we expect the first argument of every view operator to alias the output.\nView operators with multiple aliasing inputs aren't supported yet. Found an operator that doesn't satisfy this constraint"

def emit_view_functionalization_body(g: NativeFunctionsViewGroup, *, view_inplace: bool) -> str:
    if False:
        print('Hello World!')
    if view_inplace:
        assert g.view_inplace is not None
        f = g.view_inplace
    else:
        f = g.view
    assert g.view_copy is not None
    with native_function_manager(f):
        call_sig = DispatcherSignature.from_schema(g.view_copy.func)
        api_name = g.view_copy.func.name.unambiguous_name()
        noop_api_name = f.func.name.unambiguous_name()
        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        assert_view_op_properties(f.func)
        view_tensor_name = dispatcher_sig.arguments()[0].name
        return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()
        (unwrap_tensor_args_str, unwrapped_args_ctx) = unwrap_tensor_args(dispatcher_sig, is_view_op=True)
        view_redispatch_args = [e.expr for e in translate(unwrapped_args_ctx, call_sig.arguments(), method=False)]
        forward_lambda = FunctionalizationLambda.from_func(g, is_reverse=False)
        reverse_lambda = FunctionalizationLambda.from_func(g, is_reverse=True)
        (meta_conversion_str, meta_call_ctx) = convert_to_meta_tensors(dispatcher_sig)
        meta_call_args = [e.expr for e in translate(meta_call_ctx, call_sig.arguments(), method=False)]
        if 'inplace_view' in f.tags:
            return f"\n    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{\n      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{\n        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.\n        {unwrap_tensor_args_str}\n        at::AutoDispatchSkipFunctionalize guard;\n        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});\n      }}\n      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();\n      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(\n        {forward_lambda.decl()} {{\n          if (reapply_views) {{\n            return {forward_lambda.inner_call(reapply_views=True)}\n          }} else {{\n            return {forward_lambda.inner_call(reapply_views=False)}\n          }}\n        }},\n        {reverse_lambda.decl()} {{\n          return {reverse_lambda.inner_call()}\n        }}\n      );\n      auto compute_reference_meta =\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);\n      {return_type} reference_tensor_output;\n      if (compute_reference_meta) {{\n        {meta_conversion_str}\n        at::AutoDispatchSkipFunctionalize func_guard;\n        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});\n      }}\n      // This function adds the above view meta to the current tensor and replays them off the base,\n      // mutating the size/stride info of the current FunctionalTensorWrapper.\n      // Because of this, we need to make sure to run the reference shape function above,\n      // BEFORE doing this (otherwise we'll end up runnin the reference function using the wrong sizes/strides)\n      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);\n      // See  Note [Propagating strides in the functionalization pass]\n      // XLA/LTC don't implement the logic to propagate strides correctly, so we need to rely\n      // on a reference implementation here (instead of relying on the output from the forward lambda\n      // having the correct stride info)\n      if (compute_reference_meta) {{\n        at::functionalization::impl::set_sizes_strides_offset({view_tensor_name}, reference_tensor_output);\n      }}\n      return {view_tensor_name};\n    }}\n"
        else:
            is_multi_output_view = isinstance(f.func.returns[0].type, ListType)
            return f"\n    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{\n      {unwrap_tensor_args_str}\n      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{\n        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.\n        at::AutoDispatchSkipFunctionalize guard;\n        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});\n      }}\n      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();\n      auto compute_reference_meta =\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);\n      {return_type} reference_tensor_output;\n      if (compute_reference_meta) {{\n        {meta_conversion_str}\n        at::AutoDispatchSkipFunctionalize func_guard;\n        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});\n      }}\n      {return_type} tmp_output;\n      {{\n        at::AutoDispatchSkipFunctionalize guard;\n        if (reapply_views) {{\n          tmp_output = at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});\n        }} else {{\n          tmp_output = at::_ops::{api_name}::call({', '.join(view_redispatch_args)});\n        }}\n      }}\n      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(\n        {forward_lambda.decl()} {{\n          if (reapply_views) {{\n            return {forward_lambda.inner_call(reapply_views=True)}\n          }} else {{\n            return {forward_lambda.inner_call(reapply_views=False)}\n          }}\n        }},\n        {reverse_lambda.decl()} {{\n          return {reverse_lambda.inner_call()}\n        }},\n        /*is_multi_output=*/{str(is_multi_output_view).lower()}\n      );\n      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, {view_tensor_name}, view_meta);\n      // See  Note [Propagating strides in the functionalization pass]\n      if (compute_reference_meta) {{\n        at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);\n      }}\n      return out;\n    }}\n"

def maybe_create_output(f: NativeFunction, var_name: str) -> str:
    if False:
        while True:
            i = 10
    if len(f.func.returns) == 0:
        return ''
    return_type = dispatcher.returns_type(f.func.returns).remove_const_ref().cpp_type()
    return f'{return_type} {var_name} = '

def get_mutable_redispatch_return_names(f: NativeFunction, inner_return_var: str) -> Tuple[List[str], List[str]]:
    if False:
        for i in range(10):
            print('nop')
    aliased_returns = []
    non_aliased_returns = []
    for (i, name) in enumerate(f.func.aliased_return_names()):
        if name is not None:
            aliased_returns.append(name)
        else:
            non_aliased_returns.append(inner_return_var if len(f.func.returns) == 1 else f'std::get<{i}>({inner_return_var})')
    return (aliased_returns, non_aliased_returns)

def return_from_mutable_noop_redispatch(f: NativeFunction, inner_return_var: str) -> str:
    if False:
        i = 10
        return i + 15
    (aliased, non_aliased) = get_mutable_redispatch_return_names(f, inner_return_var)
    return return_str(f.func.returns, aliased + non_aliased)

def wrap_propagate_mutations_and_return(f: NativeFunction, functional_op: NativeFunction, inner_return_var: str) -> str:
    if False:
        print('Hello World!')
    mutable_arg_names = f.func.arguments.mutable_arg_names()
    (aliased_outer_rets, non_aliased_outer_rets) = get_mutable_redispatch_return_names(f, inner_return_var)
    (_, non_aliased_inner_rets) = get_mutable_redispatch_return_names(functional_op, inner_return_var)
    assert len(mutable_arg_names) + len(non_aliased_outer_rets) == len(non_aliased_inner_rets)
    updates = []
    non_aliased_wrapped_ret_names = []
    for (i, inner_ret) in enumerate(non_aliased_inner_rets[:len(non_aliased_outer_rets)]):
        ret_name = f'output_{i}'
        updates.append(f'  auto output_{i} = at::functionalization::impl::to_functional_tensor({inner_ret});')
        non_aliased_wrapped_ret_names.append(ret_name)
    for (outer_arg, inner_ret) in zip(mutable_arg_names, non_aliased_inner_rets[len(non_aliased_outer_rets):]):
        updates.append(f'  at::functionalization::impl::propagate_xla_data({outer_arg}, {inner_ret});\n  at::functionalization::impl::replace_({outer_arg}, {inner_ret});\n  at::functionalization::impl::commit_update({outer_arg});\n  at::functionalization::impl::sync({outer_arg});')
    returns_str = return_str(f.func.returns, aliased_outer_rets + non_aliased_wrapped_ret_names)
    updates_str = '\n'.join(updates)
    return f'{updates_str}\n    {returns_str}'

@with_native_function_and
def emit_inplace_functionalization_body(f: NativeFunction, g: NativeFunctionsGroup) -> str:
    if False:
        for i in range(10):
            print('nop')
    assert modifies_arguments(f)
    dispatcher_sig = DispatcherSignature.from_schema(f.func)
    (unwrap_tensor_args_str, unwrapped_args_ctx) = unwrap_tensor_args(dispatcher_sig, is_view_op=False)
    mutated_names = [a.name for a in f.func.arguments.flat_all if a.type.is_tensor_like() and a.annotation is not None]
    non_mutated_names = [a.name for a in f.func.arguments.flat_all if a.type.is_tensor_like() and a.annotation is None]
    non_mutated_tensor_names = [a.name for a in f.func.arguments.flat_all if a.type == BaseType(BaseTy.Tensor) and a.annotation is None]
    check_all_mutated_args_are_functional = ' && '.join(['true'] + [f'at::functionalization::impl::isFunctionalTensor({a})' for a in mutated_names])
    check_any_non_mutated_args_are_functional = ' || '.join(['false'] + [f'at::functionalization::impl::isFunctionalTensor({a})' for a in non_mutated_names])
    check_any_non_mutated_tensors_are_xla = ' || '.join(['false'] + [f'{a}.device().type() == c10::DeviceType::XLA' for a in non_mutated_tensor_names])
    inplace_exprs = [e.expr for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)]
    return_type = dispatcher.returns_type(g.functional.func.returns).remove_const_ref().cpp_type()
    functional_sig = DispatcherSignature.from_schema(g.functional.func)
    functional_exprs = [e.expr for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)]
    if f.func.is_out_fn():
        mutable_input_post_processing = '\n'.join([f"\n      at::functionalization::impl::replace_(\n        {a.name}, {('std::get<' + str(i) + '>(tmp_output)' if len(f.func.returns) > 1 else 'tmp_output')});\n      at::functionalization::impl::commit_update({a.name});" for (i, a) in enumerate(f.func.arguments.out) if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])
    else:
        mutable_input_post_processing = '\n'.join([f'\n      at::functionalization::impl::replace_({a.name}, tmp_output);\n      at::functionalization::impl::commit_update({a.name});' for a in f.func.arguments.flat_all if a.annotation and a.annotation.is_write and a.type.is_tensor_like()])
    (meta_conversion_str, meta_call_ctx) = convert_to_meta_tensors(dispatcher_sig)
    any_storage_args = any((a.type == BaseType(BaseTy.Storage) for a in f.func.arguments.flat_all))
    return f"""\n    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{\n      if ({str(not any_storage_args and f.func.kind() == SchemaKind.inplace).lower()}) {{\n        // Before converting the mutable op to its functional variant, run meta tensors through the original op.\n        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional variants.\n        // (We can only do this for inplace ops today though, because they technically all support meta tensors).\n        {meta_conversion_str}\n        at::AutoDispatchSkipFunctionalize func_guard;\n        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n        at::_ops::{f.func.name.unambiguous_name()}::call({', '.join((a.name for a in meta_call_ctx))});\n      }}\n      {unwrap_tensor_args_str}\n      if (!({check_all_mutated_args_are_functional})) {{\n        // We want to disable this check if there are any XLA tensors.\n        // cpu_tensor.copy_(xla_tensor) is valid code.\n        if (!({check_any_non_mutated_tensors_are_xla}) && ({check_any_non_mutated_args_are_functional})) {{\n         // case 1: trying to mutate a non functional tensor with a functional tensor is an error\n         TORCH_INTERNAL_ASSERT(false,\n           "mutating a non-functional tensor with a functional tensor is not allowed.",\n           " Please ensure that all of your inputs are wrapped inside of a functionalize() call.");\n        }} else {{\n         // case 2: arguments are not functional tensors, so we no-op and redispatch.\n         at::AutoDispatchSkipFunctionalize guard;\n         {maybe_create_output(f, 'tmp_output')}at::_ops::{f.func.name.unambiguous_name()}::call({', '.join(inplace_exprs)});\n         {return_from_mutable_noop_redispatch(f, 'tmp_output')};\n        }}\n      }} else {{\n        {return_type} tmp_output;\n        {{\n          at::AutoDispatchSkipFunctionalize guard;\n          tmp_output = at::_ops::{g.functional.func.name.unambiguous_name()}::call({', '.join(functional_exprs)});\n        }}\n        {wrap_propagate_mutations_and_return(f, g.functional, 'tmp_output')}\n      }}\n    }}"""

def gen_functionalization_view_inverse_declaration(selector: SelectiveBuilder, g: NativeFunctionsViewGroup) -> Optional[str]:
    if False:
        while True:
            i = 10

    @with_native_function
    def emit_decl_helper(g: NativeFunctionsViewGroup) -> Optional[str]:
        if False:
            while True:
                i = 10
        if g.view.has_composite_implicit_autograd_kernel:
            return None
        view_copy_inverse_sig = ViewInverseSignature(g)
        return view_copy_inverse_sig.decl()
    return emit_decl_helper(g)

def gen_functionalization_registration(selector: SelectiveBuilder, g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup], composite_implicit_autograd_index: BackendIndex) -> List[str]:
    if False:
        for i in range(10):
            print('nop')

    @with_native_function
    def emit_registration_helper(f: NativeFunction) -> str:
        if False:
            for i in range(10):
                print('nop')
        assert not f.has_composite_implicit_autograd_kernel
        registration_str = f'TORCH_FN(functionalization::{wrapper_name(f.func)})'
        return f'm.impl("{f.func.name}", {registration_str});'
    if not selector.include_all_operators:
        return []
    if isinstance(g, NativeFunctionsViewGroup):
        if str(g.view.func.name) == 'lift_fresh':
            return []
        view_str = []
        if not g.view.has_composite_implicit_autograd_kernel:
            view_str.append(emit_registration_helper(g.view))
        if g.view_inplace is not None and (not g.view_inplace.has_composite_implicit_autograd_kernel):
            assert g.view_inplace.is_view_op
            view_str.append(emit_registration_helper(g.view_inplace))
        return view_str
    elif isinstance(g, NativeFunctionsGroup):
        fns = list(g.functions())
    else:
        if str(g.func.name) in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION:
            return []
        fns = [g]
    registrations = []
    for f in fns:
        if f.has_composite_implicit_autograd_kernel:
            continue
        if str(f.func.name) == 'lift':
            return []
        if str(f.func.name) == 'resize_':
            return []
        assert not f.is_view_op
        if modifies_arguments(f):
            registrations.append(emit_registration_helper(f))
    return registrations

def gen_functionalization_definition(selector: SelectiveBuilder, g: Union[NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup]) -> List[str]:
    if False:
        while True:
            i = 10
    if not selector.include_all_operators:
        return []
    if isinstance(g, NativeFunctionsViewGroup):
        view_defs = []
        if not g.composite:
            assert g.view_copy is not None
            view_defs.append(emit_view_functionalization_body(g, view_inplace=False))
            if g.view_inplace is not None:
                view_defs.append(emit_view_functionalization_body(g, view_inplace=True))
        return view_defs
    elif isinstance(g, NativeFunction):
        if str(g.func.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION:
            assert g.has_composite_implicit_autograd_kernel or not modifies_arguments(g)
        return []
    else:
        mutation_defs = []
        mutation_defs.append(emit_inplace_functionalization_body(g.out, g))
        if g.inplace is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.inplace, g))
        if g.mutable is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.mutable, g))
        return mutation_defs
    return []