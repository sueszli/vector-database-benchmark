from vyper.codegen.core import _freshname, eval_once_check, make_setter
from vyper.codegen.ir_node import IRnode
from vyper.evm.address_space import MEMORY
from vyper.exceptions import StateAccessViolation
from vyper.semantics.types.subscriptable import TupleT
_label_counter = 0

def _generate_label(name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    global _label_counter
    _label_counter += 1
    return f'label{_label_counter}'

def _align_kwargs(func_t, args_ir):
    if False:
        i = 10
        return i + 15
    '\n    Using a list of args, find the kwargs which need to be filled in by\n    the compiler\n    '
    assert func_t.n_positional_args <= len(args_ir) <= func_t.n_total_args
    num_provided_kwargs = len(args_ir) - func_t.n_positional_args
    unprovided_kwargs = func_t.keyword_args[num_provided_kwargs:]
    return [i.default_value for i in unprovided_kwargs]

def ir_for_self_call(stmt_expr, context):
    if False:
        i = 10
        return i + 15
    from vyper.codegen.expr import Expr
    method_name = stmt_expr.func.attr
    func_t = stmt_expr.func._metadata['type']
    pos_args_ir = [Expr(x, context).ir_node for x in stmt_expr.args]
    default_vals = _align_kwargs(func_t, pos_args_ir)
    default_vals_ir = [Expr(x, context).ir_node for x in default_vals]
    args_ir = pos_args_ir + default_vals_ir
    assert len(args_ir) == len(func_t.arguments)
    args_tuple_t = TupleT([x.typ for x in args_ir])
    args_as_tuple = IRnode.from_list(['multi'] + [x for x in args_ir], typ=args_tuple_t)
    if context.is_constant() and func_t.is_mutable:
        raise StateAccessViolation(f"May not call state modifying function '{method_name}' within {context.pp_constancy()}.", stmt_expr)
    _label = func_t._ir_info.internal_function_label(context.is_ctor_context)
    return_label = _generate_label(f'{_label}_call')
    if func_t.return_type is not None:
        return_buffer = IRnode.from_list(context.new_internal_variable(func_t.return_type), annotation=f'{return_label}_return_buf')
    else:
        return_buffer = None
    dst_tuple_t = TupleT(tuple(func_t.argument_types))
    args_dst = IRnode(func_t._ir_info.frame_info.frame_start, typ=dst_tuple_t, location=MEMORY)
    if args_as_tuple.contains_self_call:
        copy_args = ['seq']
        tmp_args_buf = IRnode(context.new_internal_variable(dst_tuple_t), typ=dst_tuple_t, location=MEMORY)
        copy_args.append(make_setter(tmp_args_buf, args_as_tuple))
        copy_args.append(make_setter(args_dst, tmp_args_buf))
    else:
        copy_args = make_setter(args_dst, args_as_tuple)
    goto_op = ['goto', func_t._ir_info.internal_function_label(context.is_ctor_context)]
    if return_buffer is not None:
        goto_op += [return_buffer]
    goto_op.append(['symbol', return_label])
    call_sequence = ['seq']
    call_sequence.append(eval_once_check(_freshname(stmt_expr.node_source_code)))
    call_sequence.extend([copy_args, goto_op, ['label', return_label, ['var_list'], 'pass']])
    if return_buffer is not None:
        call_sequence += [return_buffer]
    o = IRnode.from_list(call_sequence, typ=func_t.return_type, location=MEMORY, annotation=stmt_expr.get('node_source_code'), add_gas_estimate=func_t._ir_info.gas_estimate)
    o.is_self_call = True
    return o