from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import Annotation, Argument, BackendIndex, BackendMetadata, BaseOperatorName, BaseTy, BaseType, DEFAULT_KERNEL_NAMESPACE, DeviceCheckType, DispatchKey, FunctionSchema, NativeFunction, NativeFunctionsGroup, OperatorName, Return, SchemaKind, Variant
from torchgen.utils import concatMap
OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY = ['adaptive_avg_pool3d_backward.grad_input', '_slow_conv2d_backward.grad_input']
MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = ['_cummax_helper', '_cummin_helper']
FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT = ['_assert_async', '_assert_async.msg', '_dimI', '_dimV', '_has_same_storage_numel', '_linalg_check_errors', '_local_scalar_dense', '_nested_tensor_from_mask_left_aligned', '_nnz', '_use_cudnn_ctc_loss', '_use_cudnn_ctc_loss.Tensor', '_validate_compressed_sparse_indices', 'allclose', 'dense_dim', 'equal', 'is_coalesced', 'is_pinned', 'is_same_size', 'is_set_to', 'q_per_channel_axis', 'q_scale', 'q_zero_point', 'qscheme', 'record_stream', 'sparse_dim', 'sym_constrain_range', 'sym_constrain_range_for_size', '_nested_tensor_storage_offsets', '_chunk_grad_outputs_efficient_attention', '_fused_sdp_choice']
INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY = ['polygamma_']

def pre_group_native_functions(native_functions: Sequence[NativeFunction]) -> Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]]:
    if False:
        while True:
            i = 10
    pre_grouped_native_functions: Dict[FunctionSchema, Dict[SchemaKind, NativeFunction]] = defaultdict(dict)
    for f in native_functions:
        d = pre_grouped_native_functions[f.func.signature()]
        assert f.func.kind() not in d
        d[f.func.kind()] = f
    return pre_grouped_native_functions

def get_expected_out_variant_overload_name(overload_name: Optional[str]) -> str:
    if False:
        print('Hello World!')
    return 'out' if not overload_name else f'{overload_name}_out'

def self_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    if False:
        print('Hello World!')
    assert func.kind() == SchemaKind.inplace
    assert func.arguments.self_arg is not None
    return FunctionSchema(name=func.name.remove_inplace().with_overload(get_expected_out_variant_overload_name(func.name.overload_name)), arguments=func.arguments.remove_self_annotation().with_out_args([Argument(name='out', type=func.arguments.self_arg.argument.type, default=None, annotation=func.arguments.self_arg.argument.annotation)]), returns=func.returns)

def functional_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    if False:
        return 10
    assert func.kind() == SchemaKind.functional
    (new_returns, new_out_args) = generate_out_args_from_schema(func)
    return FunctionSchema(name=func.name.with_overload(get_expected_out_variant_overload_name(func.name.overload_name)), arguments=func.arguments.signature().with_out_args(new_out_args), returns=tuple(new_returns))

def generate_out_args_from_schema(func: FunctionSchema) -> Tuple[List[Return], List[Argument]]:
    if False:
        for i in range(10):
            print('nop')
    assert not any((r.annotation is not None and r.annotation.is_write for r in func.returns))
    tensorlike_rets = [r for r in func.returns if r.type.is_tensor_like()]
    assert len(tensorlike_rets) > 0
    used_annotations = concatMap(lambda a: [] if a.annotation is None else a.annotation.alias_set, func.arguments.flat_all)
    valid_annotations = [x for x in 'abcdefghijklmnopqrstuvwxyz' if x not in used_annotations]
    all_rets_are_tensors = all((r.type == BaseType(BaseTy.Tensor) for r in func.returns))
    new_out_args: List[Argument] = []
    new_returns: List[Return] = []
    for (i, r) in enumerate(func.returns):
        if r.type.is_tensor_like():
            new_out = Argument(name='out' if len(func.returns) == 1 else f'out{i}', type=r.type, default=None, annotation=Annotation.parse(f'{valid_annotations[i]}!'))
            new_out_args.append(new_out)
            if all_rets_are_tensors:
                new_ret = Return(name=None, type=new_out.type, annotation=new_out.annotation)
                new_returns.append(new_ret)
        else:
            new_returns.append(r)
    return (new_returns, new_out_args)

def mutable_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    if False:
        print('Hello World!')
    assert func.kind() == SchemaKind.mutable
    (new_returns, new_out_args) = generate_out_args_from_schema(func)
    return FunctionSchema(name=func.name.remove_inplace().with_overload(get_expected_out_variant_overload_name(func.name.overload_name)), arguments=func.arguments.with_out_args(new_out_args), returns=tuple(new_returns))

def generate_function(f: NativeFunction, k: SchemaKind) -> Tuple[NativeFunction, Dict[DispatchKey, Dict['OperatorName', 'BackendMetadata']]]:
    if False:
        for i in range(10):
            print('nop')
    from torchgen.api import cpp
    if k == SchemaKind.functional:
        assert f.func.kind() != SchemaKind.functional
        func = f.func.signature(keep_return_names=True).with_name(OperatorName(name=BaseOperatorName(base=f.func.name.name.base, inplace=False, dunder_method=f.func.name.name.dunder_method, functional_overload=f.func.kind() == SchemaKind.mutable), overload_name=f.func.name.overload_name))
    elif k == SchemaKind.out:
        if f.func.kind() == SchemaKind.inplace:
            func = self_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.mutable:
            func = mutable_to_out_signature(f.func)
        elif f.func.kind() == SchemaKind.functional:
            func = functional_to_out_signature(f.func)
        else:
            raise AssertionError('We only bother generating out= functions from either inplace or mutable or functional variants')
    else:
        raise AssertionError('We currently only generate either functional or out= NativeFunctions')
    kernel_name = func.name.unambiguous_name() if func.kind() == SchemaKind.out else cpp.name(func)
    if f.func.has_symint():
        kernel_name += '_symint'
    backend_metadata = {DispatchKey.CompositeExplicitAutograd: {func.name: BackendMetadata(kernel=kernel_name, structured=False, cpp_namespace=DEFAULT_KERNEL_NAMESPACE)}}
    tags = {'generated'} | set(f.tags & {'nondeterministic_seeded', 'view_copy', 'pt2_compliant_tag'})
    return (NativeFunction(func=func, use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors, variants={Variant.function}, structured=False, structured_delegate=None, structured_inherits=None, precomputed=None, autogen=[], ufunc_inner_loop={}, manual_kernel_registration=False, manual_cpp_binding=False, python_module=None, category_override=None, device_guard=False, device_check=DeviceCheckType.NoCheck, loc=f.loc, cpp_no_default_args=set(), is_abstract=f.is_abstract, has_composite_implicit_autograd_kernel=False, has_composite_implicit_autograd_nested_tensor_kernel=False, has_composite_explicit_autograd_kernel=True, has_composite_explicit_autograd_non_functional_kernel=False, tags=tags, namespace=f.namespace), backend_metadata)

def add_generated_native_functions(rs: List[NativeFunction], indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]) -> None:
    if False:
        print('Hello World!')
    pre_grouped_native_functions = pre_group_native_functions(rs)
    for d in pre_grouped_native_functions.values():
        has_functional = SchemaKind.functional in d
        has_inplace = SchemaKind.inplace in d
        has_mutable = SchemaKind.mutable in d
        has_out = SchemaKind.out in d
        if has_mutable or has_inplace or has_out or has_functional:
            are_manual = all((f.manual_cpp_binding for f in d.values()))
            has_view_ops = any((f.is_view_op for f in d.values()))
            are_composite_implicit = all((f.has_composite_implicit_autograd_kernel for f in d.values()))
            if are_manual or has_view_ops or are_composite_implicit:
                continue
            if has_out and len(d.values()) == 1:
                if str(d[SchemaKind.out].func.name) not in OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY:
                    raise AssertionError(f'Found an out= operator that we could not find any other variants of: {str(d[SchemaKind.out].func)}')
                continue
            if has_inplace and str(d[SchemaKind.inplace].func.name) in INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY:
                continue
            base_fn = d[SchemaKind.inplace] if has_inplace else d[SchemaKind.mutable] if has_mutable else d[SchemaKind.out] if has_out else d[SchemaKind.functional]
            base_fn_valid = base_fn.func.kind() == SchemaKind.inplace or any((r.type.is_tensor_like() for r in base_fn.func.returns))
            needs_out = any(('out' in str(op_name) for op_name in base_fn.autogen))
            gets_out_variant = not has_out and base_fn_valid and needs_out
            if not has_out and (not base_fn_valid):
                if str(base_fn.func.name) not in MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT and str(base_fn.func.name) not in FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT:
                    raise AssertionError(f"Found an operator that we could not generate an out= variant for: {str(base_fn.func)}.\nThis type of operators don't have tensor-like return, making it difficult to generate a proper out= variant. If\nout= variant is not needed, please add the function name into FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT list.")
            if gets_out_variant:
                (fn, metadata) = generate_function(base_fn, SchemaKind.out)
                d[SchemaKind.out] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)
            if not has_functional and (has_out or gets_out_variant):
                (fn, metadata) = generate_function(base_fn, SchemaKind.functional)
                d[SchemaKind.functional] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)

def return_str(rets: Tuple[Return, ...], names: List[str]) -> str:
    if False:
        i = 10
        return i + 15
    assert len(rets) == len(names)
    if len(rets) == 0:
        return ''
    elif len(rets) == 1:
        return f'return {names[0]};'
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"

def gather_nonaliased_inner_rets(func: FunctionSchema, out_var: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    aliased_rets = func.aliased_return_names()
    non_aliased_names = []
    is_out_var_a_tuple = len(func.returns) > 1
    for (i, r) in enumerate(aliased_rets):
        if r is None:
            non_aliased_names.append(f'std::get<{i}>({out_var})' if is_out_var_a_tuple else out_var)
    return non_aliased_names

@with_native_function
def gen_composite_functional_kernel(g: NativeFunctionsGroup) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    if 'generated' not in g.functional.tags:
        return None
    if g.inplace is not None and 'generated' not in g.inplace.tags:
        target_f = g.inplace
    elif g.mutable is not None and 'generated' not in g.mutable.tags:
        target_f = g.mutable
    else:
        raise AssertionError(str(g.functional.func))
    sig = DispatcherSignature(g.functional.func)
    target_sig = DispatcherSignature(target_f.func)
    context: List[Union[Binding, Expr]] = []
    clone_mutable_inputs = []
    cloned_return_names = []
    for (a_curr, a_tgt) in zip(dispatcher.jit_arguments(g.functional.func), dispatcher.jit_arguments(target_f.func)):
        if a_tgt.annotation is not None and a_tgt.annotation.is_write:
            clone_mutable_inputs.append(f'auto {a_curr.name}_clone = clone_arg({a_curr.name});')
            context.append(Expr(expr=f'{a_curr.name}_clone', type=dispatcher.argument_type(a_curr, binds=a_curr.name)))
            cloned_return_names.append(f'{a_curr.name}_clone')
        else:
            context.append(dispatcher.argument(a_curr))
    exprs = ', '.join([e.expr for e in translate(context, target_sig.arguments())])
    out_name = 'output'
    maybe_assign = f'auto {out_name} = ' if len(target_f.func.returns) > 0 else ''
    inner_return_names = gather_nonaliased_inner_rets(target_f.func, out_name)
    ret_str = return_str(g.functional.func.returns, inner_return_names + cloned_return_names)
    clone_mutable_inputs_str = '\n'.join(clone_mutable_inputs)
    return f"\n{sig.defn(name=sig.name() + ('_symint' if g.out.func.has_symint() else ''))} {{\n  {clone_mutable_inputs_str}\n  {maybe_assign}at::_ops::{target_f.func.name.unambiguous_name()}::call({exprs});\n  {ret_str}\n}}\n"

@with_native_function
def gen_composite_out_kernel(g: NativeFunctionsGroup) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    if 'generated' not in g.out.tags:
        return None
    sig = DispatcherSignature(g.out.func)
    target_sig = DispatcherSignature(g.functional.func)
    exprs = ', '.join([e.expr for e in translate(sig.arguments(), target_sig.arguments())])
    copy_outs = []
    out_name = 'tmp_output'
    for (i, out_arg) in enumerate(g.out.func.arguments.out):
        functional_return_name = out_name if len(g.functional.func.returns) == 1 else f'std::get<{i}>({out_name})'
        copy_outs.append(f'  resize_out_helper({out_arg.name}, {functional_return_name});\n  copy_arg({out_arg.name}, {functional_return_name});')
    rets = []
    for (i, ret_name) in enumerate(g.out.func.aliased_return_names()):
        if ret_name is not None:
            rets.append(ret_name)
        else:
            functional_return_name = out_name if len(g.functional.func.returns) == 1 else f'std::get<{i}>({out_name})'
            rets.append(functional_return_name)
    copy_outs_str = '\n'.join(copy_outs)
    return f"\n{sig.defn(name=g.out.func.name.unambiguous_name() + ('_symint' if g.out.func.has_symint() else ''))} {{\n  auto {out_name} = at::_ops::{g.functional.func.name.unambiguous_name()}::call({exprs});\n  {copy_outs_str}\n  {return_str(g.out.func.returns, rets)}\n}}\n"