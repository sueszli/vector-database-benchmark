from vyper.codegen.abi_encoder import abi_encoding_matches_vyper
from vyper.codegen.context import Context, VariableRecord
from vyper.codegen.core import get_element_ptr, getpos, make_setter, needs_clamp
from vyper.codegen.expr import Expr
from vyper.codegen.function_definitions.utils import get_nonreentrant_lock
from vyper.codegen.ir_node import Encoding, IRnode
from vyper.codegen.stmt import parse_body
from vyper.evm.address_space import CALLDATA, DATA, MEMORY
from vyper.semantics.types import TupleT
from vyper.semantics.types.function import ContractFunctionT

def _register_function_args(func_t: ContractFunctionT, context: Context) -> list[IRnode]:
    if False:
        i = 10
        return i + 15
    ret = []
    base_args_t = TupleT(tuple((arg.typ for arg in func_t.positional_args)))
    if func_t.is_constructor:
        base_args_ofst = IRnode(0, location=DATA, typ=base_args_t, encoding=Encoding.ABI)
    else:
        base_args_ofst = IRnode(4, location=CALLDATA, typ=base_args_t, encoding=Encoding.ABI)
    for (i, arg) in enumerate(func_t.positional_args):
        arg_ir = get_element_ptr(base_args_ofst, i)
        if needs_clamp(arg.typ, Encoding.ABI):
            p = context.new_variable(arg.name, arg.typ, is_mutable=False)
            dst = IRnode(p, typ=arg.typ, location=MEMORY)
            copy_arg = make_setter(dst, arg_ir)
            copy_arg.source_pos = getpos(arg.ast_source)
            ret.append(copy_arg)
        else:
            assert abi_encoding_matches_vyper(arg.typ)
            context.vars[arg.name] = VariableRecord(name=arg.name, pos=arg_ir, typ=arg.typ, mutable=False, location=arg_ir.location, encoding=Encoding.ABI)
    return ret

def _generate_kwarg_handlers(func_t: ContractFunctionT, context: Context) -> dict[str, tuple[int, IRnode]]:
    if False:
        print('Hello World!')

    def handler_for(calldata_kwargs, default_kwargs):
        if False:
            while True:
                i = 10
        calldata_args = func_t.positional_args + calldata_kwargs
        calldata_args_t = TupleT(list((arg.typ for arg in calldata_args)))
        abi_sig = func_t.abi_signature_for_kwargs(calldata_kwargs)
        calldata_kwargs_ofst = IRnode(4, location=CALLDATA, typ=calldata_args_t, encoding=Encoding.ABI)
        ret = ['seq']
        args_abi_t = calldata_args_t.abi_type
        calldata_min_size = args_abi_t.min_size() + 4
        for (i, arg_meta) in enumerate(calldata_kwargs):
            k = func_t.n_positional_args + i
            dst = context.lookup_var(arg_meta.name).pos
            lhs = IRnode(dst, location=MEMORY, typ=arg_meta.typ)
            rhs = get_element_ptr(calldata_kwargs_ofst, k, array_bounds_check=False)
            copy_arg = make_setter(lhs, rhs)
            copy_arg.source_pos = getpos(arg_meta.ast_source)
            ret.append(copy_arg)
        for x in default_kwargs:
            dst = context.lookup_var(x.name).pos
            lhs = IRnode(dst, location=MEMORY, typ=x.typ)
            lhs.source_pos = getpos(x.ast_source)
            kw_ast_val = func_t.default_values[x.name]
            rhs = Expr(kw_ast_val, context).ir_node
            copy_arg = make_setter(lhs, rhs)
            copy_arg.source_pos = getpos(x.ast_source)
            ret.append(copy_arg)
        ret.append(['goto', func_t._ir_info.external_function_base_entry_label])
        return (abi_sig, calldata_min_size, ret)
    ret = {}
    keyword_args = func_t.keyword_args
    for arg in keyword_args:
        context.new_variable(arg.name, arg.typ, is_mutable=False)
    for (i, _) in enumerate(keyword_args):
        calldata_kwargs = keyword_args[:i]
        default_kwargs = keyword_args[i:]
        (sig, calldata_min_size, ir_node) = handler_for(calldata_kwargs, default_kwargs)
        ret[sig] = (calldata_min_size, ir_node)
    (sig, calldata_min_size, ir_node) = handler_for(keyword_args, [])
    ret[sig] = (calldata_min_size, ir_node)
    return ret

def generate_ir_for_external_function(code, func_t, context):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the IR for an external function. Returns IR for the body\n    of the function, handle kwargs and exit the function. Also returns\n    metadata required for `module.py` to construct the selector table.\n    '
    (nonreentrant_pre, nonreentrant_post) = get_nonreentrant_lock(func_t)
    handle_base_args = _register_function_args(func_t, context)
    kwarg_handlers = _generate_kwarg_handlers(func_t, context)
    body = ['seq']
    body += handle_base_args
    body += nonreentrant_pre
    body += [parse_body(code.body, context, ensure_terminated=True)]
    body = ['label', func_t._ir_info.external_function_base_entry_label, ['var_list'], body]
    exit_sequence = ['seq'] + nonreentrant_post
    if func_t.is_constructor:
        pass
    elif context.return_type is None:
        exit_sequence += [['stop']]
    else:
        exit_sequence += [['return', 'ret_ofst', 'ret_len']]
    exit_sequence_args = ['var_list']
    if context.return_type is not None:
        exit_sequence_args += ['ret_ofst', 'ret_len']
    exit_ = ['label', func_t._ir_info.exit_sequence_label, exit_sequence_args, exit_sequence]
    func_common_ir = IRnode.from_list(['seq', body, exit_], source_pos=getpos(code))
    return (kwarg_handlers, func_common_ir)