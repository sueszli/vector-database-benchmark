from vyper import ast as vy_ast
from vyper.codegen.context import Context
from vyper.codegen.function_definitions.utils import get_nonreentrant_lock
from vyper.codegen.ir_node import IRnode
from vyper.codegen.stmt import parse_body
from vyper.semantics.types.function import ContractFunctionT

def generate_ir_for_internal_function(code: vy_ast.FunctionDef, func_t: ContractFunctionT, context: Context) -> IRnode:
    if False:
        while True:
            i = 10
    '\n    Parse a internal function (FuncDef), and produce full function body.\n\n    :param func_t: the ContractFunctionT\n    :param code: ast of function\n    :param context: current calling context\n    :return: function body in IR\n    '
    for arg in func_t.arguments:
        context.new_variable(arg.name, arg.typ, is_mutable=True)
    (nonreentrant_pre, nonreentrant_post) = get_nonreentrant_lock(func_t)
    function_entry_label = func_t._ir_info.internal_function_label(context.is_ctor_context)
    cleanup_label = func_t._ir_info.exit_sequence_label
    stack_args = ['var_list']
    if func_t.return_type:
        stack_args += ['return_buffer']
    stack_args += ['return_pc']
    body = ['label', function_entry_label, stack_args, ['seq'] + nonreentrant_pre + [parse_body(code.body, context, ensure_terminated=True)]]
    cleanup_routine = ['label', cleanup_label, ['var_list', 'return_pc'], ['seq'] + nonreentrant_post + [['exit_to', 'return_pc']]]
    return IRnode.from_list(['seq', body, cleanup_routine])