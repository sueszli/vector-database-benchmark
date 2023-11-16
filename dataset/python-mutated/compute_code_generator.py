import ast
import contextlib
from cinn import ir
from .expr_executor import ExprExecutor, exec_assign
from .utils import VariableTable, is_node_parsed_in_schedule

class ComputeCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the compute part
    """

    def __init__(self, fn, function_name, inputs_signature):
        if False:
            for i in range(10):
                print('nop')
        self.fn = fn
        self.function_name = function_name
        self.inputs_signature = inputs_signature
        self.cinn_llir_func = None
        self.variables_table = VariableTable()
        self.extra_scope = {'range': ir.sequential}

    def parse(self):
        if False:
            return 10
        ast_node = self.fn.parse()
        with ir.IRBuilder() as builder, self.variables_table:
            for (k, v) in self.fn.scope.items():
                self.variables_table.add(k, v)
            for (k, v) in self.extra_scope.items():
                self.variables_table.add(k, v)
            self.visit(ast_node)
        return builder.get()

    def visit_FunctionDef(self, node) -> None:
        if False:
            print('Hello World!')
        '\n        Parse CINN Low Level IR FunctionDef.\n\n        Args:\n            node(ast.FunctionDef): The ast FunctionDef Node\n        '
        with ir.LowerFuncContext(self.function_name) as func_ctx:
            arg_names = self.visit(node.args)
            assert len(node.args.defaults) == 0, 'Not support default args'
            for (i, arg_name) in enumerate(arg_names):
                if hasattr(self.inputs_signature[i], 'dtype'):
                    tensor_shape = [ir.Expr(dim) for dim in self.inputs_signature[i].shape]
                    llir_value = ir._Buffer_.make(arg_name, self.inputs_signature[i].dtype)
                    ir.Arg(arg_name, llir_value)
                    llir_value = ir._Tensor_.make(arg_name, self.inputs_signature[i].dtype, tensor_shape, tensor_shape)
                    self.variables_table.add(arg_name, llir_value)
                else:
                    llir_value = ir.Var(arg_name)
                    ir.Arg(arg_name, llir_value)
                    llir_value = ir.Expr(llir_value)
                    self.variables_table.add(arg_name, llir_value)
            body = self.visit_compound_statement(node.body)

    def visit_compound_statement(self, stmts):
        if False:
            print('Hello World!')
        for stmt in stmts:
            self.visit(stmt)

    def visit_arguments(self, node):
        if False:
            while True:
                i = 10
        '\n        Parse CINN Low Level IR Argument.\n        If it is not jit mode, it will get information from arg.annoatation.\n\n        Args:\n            node(ast.arguments): The ast argument Node\n\n        Returns:\n            list[string]: A list of parameter names\n        '
        arg_names = [arg.arg for arg in node.args]
        if len(self.inputs_signature) != len(arg_names):
            self.inputs_signature = []
            for arg in node.args:
                arg_annotation = arg.annotation
                if isinstance(arg_annotation, ast.Call):
                    self.inputs_signature.append(ExprExecutor(self.variables_table.get()).exec(arg_annotation))
                elif isinstance(arg_annotation, int):
                    if -2 ** 21 <= arg_annotation and arg_annotation <= 2 ** 31 - 1:
                        self.inputs_signature.append('i32')
                    elif 2 ** 63 <= arg_annotation and arg_annotation <= 2 ** 64 - 1:
                        self.inputs_signature.append('u64')
                    else:
                        self.inputs_signature.append('i64')
                elif isinstance(arg_annotation, float):
                    return self.inputs_signature.append('fp32')
                else:
                    raise TypeError(f'Unsupported type {type(arg_annotation)} for {arg_annotation}')
        return arg_names

    def visit_For(self, node) -> ir.Expr:
        if False:
            print('Hello World!')
        '\n        parse CINN Low Level IR For.\n\n        Args:\n            node(ast.For): The ast For node\n        '
        for_ctx = ExprExecutor(self.variables_table.get()).exec(node.iter)
        with self.variables_table:
            with for_ctx as loop_var:
                local_var_table = exec_assign(target=node.target, source=loop_var)
                for (k, v) in local_var_table.items():
                    loop_var.rename(k)
                    self.variables_table.add(k, ir.Expr(v))
                self.visit_compound_statement(node.body)

    def visit_Assign(self, node):
        if False:
            return 10
        '\n        parse CINN Low Level IR Store.\n\n        Args:\n            node(ast.Assign): The ast Assign node\n\n        Returns:\n            ir.Expr, Points to the Expr of ir::ExprNode<Store>\n        '
        if isinstance(node.value, ast.Call) and is_node_parsed_in_schedule(node.value):
            return 'no compute'
        assert len(node.targets) == 1, "Unsupport targets is a                list of nodes, like 'a = b = c'"
        lhs = node.targets[0]
        rhs_expr = ExprExecutor(self.variables_table.get()).exec(node.value)
        if isinstance(lhs, ast.Subscript):
            expr_tensor = ExprExecutor(self.variables_table.get()).exec(lhs.value)
            if isinstance(lhs.slice, ast.Tuple):
                expr_indices = []
                for idx in lhs.slice.elts:
                    expr_indices.append(ExprExecutor(self.variables_table.get()).exec(idx))
            else:
                expr_indices = [ExprExecutor(self.variables_table.get()).exec(lhs.slice)]
            if not isinstance(rhs_expr, ir.Expr):
                rhs_expr = ir.Expr(rhs_expr)
            ir.TensorStore(expr_tensor.Expr(), rhs_expr, expr_indices)
        else:
            local_var_table = exec_assign(target=lhs, source=rhs_expr)
            if isinstance(lhs, ast.Tuple):
                for (k, v) in local_var_table.items():
                    v.as_var_ref().rename(k)
                    self.variables_table.add(k, v)
            else:
                for (k, v) in local_var_table.items():
                    v[0].as_var_ref().rename(k)
                    self.variables_table.add(k, v[0])

    def visit_If(self, node):
        if False:
            i = 10
            return i + 15
        with self.variables_table:
            with ir.IfContext(ExprExecutor(self.variables_table.get()).exec(node.test)):
                with ir.ThenContext():
                    with self.variables_table:
                        self.visit_compound_statement(node.body)
                if node.orelse:
                    with ir.ElseContext():
                        with self.variables_table:
                            self.visit_compound_statement(node.body)

    def visit_With(self, node):
        if False:
            for i in range(10):
                print('nop')
        with self.variables_table:
            with contextlib.ExitStack() as context_stack:
                for item in node.items:
                    cur_ctx = ExprExecutor(self.variables_table.get()).exec(item.context_expr)
                    cur_ctx = context_stack.enter_context(cur_ctx)
                    if item.optional_vars is not None:
                        local_var_table = exec_assign(target=item.optional_vars, source=cur_ctx)
                        for (k, v) in local_var_table.items():
                            self.variables_table.add(k, v)
                body = self.visit_compound_statement(node.body)

    def visit_Expr(self, node):
        if False:
            print('Hello World!')
        if is_node_parsed_in_schedule(node.value):
            return
        res = ExprExecutor(self.variables_table.get()).exec(node.value)
        if isinstance(res, ir.Expr):
            ir.link_to_parent_context(res)