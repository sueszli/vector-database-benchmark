import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc

def _is_ufunc(func):
    if False:
        print('Hello World!')
    return isinstance(func, (np.ufunc, DUFunc))

@rewrites.register_rewrite('after-inference')
class RewriteArrayExprs(rewrites.Rewrite):
    """The RewriteArrayExprs class is responsible for finding array
    expressions in Numba intermediate representation code, and
    rewriting those expressions to a single operation that will expand
    into something similar to a ufunc call.
    """

    def __init__(self, state, *args, **kws):
        if False:
            print('Hello World!')
        super(RewriteArrayExprs, self).__init__(state, *args, **kws)
        special_ops = state.targetctx.special_ops
        if 'arrayexpr' not in special_ops:
            special_ops['arrayexpr'] = _lower_array_expr

    def match(self, func_ir, block, typemap, calltypes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Using typing and a basic block, search the basic block for array\n        expressions.\n        Return True when one or more matches were found, False otherwise.\n        '
        if len(calltypes) == 0:
            return False
        self.crnt_block = block
        self.typemap = typemap
        self.array_assigns = OrderedDict()
        self.const_assigns = {}
        assignments = block.find_insts(ir.Assign)
        for instr in assignments:
            target_name = instr.target.name
            expr = instr.value
            if isinstance(expr, ir.Expr) and isinstance(typemap.get(target_name, None), types.Array):
                self._match_array_expr(instr, expr, target_name)
            elif isinstance(expr, ir.Const):
                self.const_assigns[target_name] = expr
        return len(self.array_assigns) > 0

    def _match_array_expr(self, instr, expr, target_name):
        if False:
            print('Hello World!')
        '\n        Find whether the given assignment (*instr*) of an expression (*expr*)\n        to variable *target_name* is an array expression.\n        '
        expr_op = expr.op
        array_assigns = self.array_assigns
        if expr_op in ('unary', 'binop') and expr.fn in npydecl.supported_array_operators:
            if all((self.typemap[var.name].is_internal for var in expr.list_vars())):
                array_assigns[target_name] = instr
        elif expr_op == 'call' and expr.func.name in self.typemap:
            func_type = self.typemap[expr.func.name]
            if isinstance(func_type, types.Function):
                func_key = func_type.typing_key
                if _is_ufunc(func_key):
                    if not self._has_explicit_output(expr, func_key):
                        array_assigns[target_name] = instr

    def _has_explicit_output(self, expr, func):
        if False:
            return 10
        '\n        Return whether the *expr* call to *func* (a ufunc) features an\n        explicit output argument.\n        '
        nargs = len(expr.args) + len(expr.kws)
        if expr.vararg is not None:
            return True
        return nargs > func.nin

    def _get_array_operator(self, ir_expr):
        if False:
            while True:
                i = 10
        ir_op = ir_expr.op
        if ir_op in ('unary', 'binop'):
            return ir_expr.fn
        elif ir_op == 'call':
            return self.typemap[ir_expr.func.name].typing_key
        raise NotImplementedError("Don't know how to find the operator for '{0}' expressions.".format(ir_op))

    def _get_operands(self, ir_expr):
        if False:
            print('Hello World!')
        'Given a Numba IR expression, return the operands to the expression\n        in order they appear in the expression.\n        '
        ir_op = ir_expr.op
        if ir_op == 'binop':
            return (ir_expr.lhs, ir_expr.rhs)
        elif ir_op == 'unary':
            return ir_expr.list_vars()
        elif ir_op == 'call':
            return ir_expr.args
        raise NotImplementedError("Don't know how to find the operands for '{0}' expressions.".format(ir_op))

    def _translate_expr(self, ir_expr):
        if False:
            print('Hello World!')
        'Translate the given expression from Numba IR to an array expression\n        tree.\n        '
        ir_op = ir_expr.op
        if ir_op == 'arrayexpr':
            return ir_expr.expr
        operands_or_args = [self.const_assigns.get(op_var.name, op_var) for op_var in self._get_operands(ir_expr)]
        return (self._get_array_operator(ir_expr), operands_or_args)

    def _handle_matches(self):
        if False:
            return 10
        'Iterate over the matches, trying to find which instructions should\n        be rewritten, deleted, or moved.\n        '
        replace_map = {}
        dead_vars = set()
        used_vars = defaultdict(int)
        for instr in self.array_assigns.values():
            expr = instr.value
            arr_inps = []
            arr_expr = (self._get_array_operator(expr), arr_inps)
            new_expr = ir.Expr(op='arrayexpr', loc=expr.loc, expr=arr_expr, ty=self.typemap[instr.target.name])
            new_instr = ir.Assign(new_expr, instr.target, instr.loc)
            replace_map[instr] = new_instr
            self.array_assigns[instr.target.name] = new_instr
            for operand in self._get_operands(expr):
                operand_name = operand.name
                if operand.is_temp and operand_name in self.array_assigns:
                    child_assign = self.array_assigns[operand_name]
                    child_expr = child_assign.value
                    child_operands = child_expr.list_vars()
                    for operand in child_operands:
                        used_vars[operand.name] += 1
                    arr_inps.append(self._translate_expr(child_expr))
                    if child_assign.target.is_temp:
                        dead_vars.add(child_assign.target.name)
                        replace_map[child_assign] = None
                elif operand_name in self.const_assigns:
                    arr_inps.append(self.const_assigns[operand_name])
                else:
                    used_vars[operand.name] += 1
                    arr_inps.append(operand)
        return (replace_map, dead_vars, used_vars)

    def _get_final_replacement(self, replacement_map, instr):
        if False:
            i = 10
            return i + 15
        'Find the final replacement instruction for a given initial\n        instruction by chasing instructions in a map from instructions\n        to replacement instructions.\n        '
        replacement = replacement_map[instr]
        while replacement in replacement_map:
            replacement = replacement_map[replacement]
        return replacement

    def apply(self):
        if False:
            return 10
        "When we've found array expressions in a basic block, rewrite that\n        block, returning a new, transformed block.\n        "
        (replace_map, dead_vars, used_vars) = self._handle_matches()
        result = self.crnt_block.copy()
        result.clear()
        delete_map = {}
        for instr in self.crnt_block.body:
            if isinstance(instr, ir.Assign):
                if instr in replace_map:
                    replacement = self._get_final_replacement(replace_map, instr)
                    if replacement:
                        result.append(replacement)
                        for var in replacement.value.list_vars():
                            var_name = var.name
                            if var_name in delete_map:
                                result.append(delete_map.pop(var_name))
                            if used_vars[var_name] > 0:
                                used_vars[var_name] -= 1
                else:
                    result.append(instr)
            elif isinstance(instr, ir.Del):
                instr_value = instr.value
                if used_vars[instr_value] > 0:
                    used_vars[instr_value] -= 1
                    delete_map[instr_value] = instr
                elif instr_value not in dead_vars:
                    result.append(instr)
            else:
                result.append(instr)
        if delete_map:
            for instr in delete_map.values():
                result.insert_before_terminator(instr)
        return result
_unaryops = {operator.pos: ast.UAdd, operator.neg: ast.USub, operator.invert: ast.Invert}
_binops = {operator.add: ast.Add, operator.sub: ast.Sub, operator.mul: ast.Mult, operator.truediv: ast.Div, operator.mod: ast.Mod, operator.or_: ast.BitOr, operator.rshift: ast.RShift, operator.xor: ast.BitXor, operator.lshift: ast.LShift, operator.and_: ast.BitAnd, operator.pow: ast.Pow, operator.floordiv: ast.FloorDiv}
_cmpops = {operator.eq: ast.Eq, operator.ne: ast.NotEq, operator.lt: ast.Lt, operator.le: ast.LtE, operator.gt: ast.Gt, operator.ge: ast.GtE}

def _arr_expr_to_ast(expr):
    if False:
        return 10
    'Build a Python expression AST from an array expression built by\n    RewriteArrayExprs.\n    '
    if isinstance(expr, tuple):
        (op, arr_expr_args) = expr
        ast_args = []
        env = {}
        for arg in arr_expr_args:
            (ast_arg, child_env) = _arr_expr_to_ast(arg)
            ast_args.append(ast_arg)
            env.update(child_env)
        if op in npydecl.supported_array_operators:
            if len(ast_args) == 2:
                if op in _binops:
                    return (ast.BinOp(ast_args[0], _binops[op](), ast_args[1]), env)
                if op in _cmpops:
                    return (ast.Compare(ast_args[0], [_cmpops[op]()], [ast_args[1]]), env)
            else:
                assert op in _unaryops
                return (ast.UnaryOp(_unaryops[op](), ast_args[0]), env)
        elif _is_ufunc(op):
            fn_name = '__ufunc_or_dufunc_{0}'.format(hex(hash(op)).replace('-', '_'))
            fn_ast_name = ast.Name(fn_name, ast.Load())
            env[fn_name] = op
            ast_call = ast.Call(fn_ast_name, ast_args, [])
            return (ast_call, env)
    elif isinstance(expr, ir.Var):
        return (ast.Name(expr.name, ast.Load(), lineno=expr.loc.line, col_offset=expr.loc.col if expr.loc.col else 0), {})
    elif isinstance(expr, ir.Const):
        return (ast.Num(expr.value), {})
    raise NotImplementedError("Don't know how to translate array expression '%r'" % (expr,))

@contextlib.contextmanager
def _legalize_parameter_names(var_list):
    if False:
        return 10
    "\n    Legalize names in the variable list for use as a Python function's\n    parameter names.\n    "
    var_map = OrderedDict()
    for var in var_list:
        old_name = var.name
        new_name = var.scope.redefine(old_name, loc=var.loc).name
        new_name = new_name.replace('$', '_').replace('.', '_')
        if new_name in var_map:
            raise AssertionError(f'{new_name!r} not unique')
        var_map[new_name] = (var, old_name)
        var.name = new_name
    param_names = list(var_map)
    try:
        yield param_names
    finally:
        for (var, old_name) in var_map.values():
            var.name = old_name

class _EraseInvalidLineRanges(ast.NodeTransformer):

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if False:
            for i in range(10):
                print('nop')
        node = super().generic_visit(node)
        if hasattr(node, 'lineno'):
            if getattr(node, 'end_lineno', None) is not None:
                if node.lineno > node.end_lineno:
                    del node.lineno
                    del node.end_lineno
        return node

def _fix_invalid_lineno_ranges(astree: ast.AST):
    if False:
        while True:
            i = 10
    'Inplace fixes invalid lineno ranges.\n    '
    ast.fix_missing_locations(astree)
    _EraseInvalidLineRanges().visit(astree)
    ast.fix_missing_locations(astree)

def _lower_array_expr(lowerer, expr):
    if False:
        while True:
            i = 10
    'Lower an array expression built by RewriteArrayExprs.\n    '
    expr_name = '__numba_array_expr_%s' % hex(hash(expr)).replace('-', '_')
    expr_filename = expr.loc.filename
    expr_var_list = expr.list_vars()
    expr_var_unique = sorted(set(expr_var_list), key=lambda var: var.name)
    expr_args = [var.name for var in expr_var_unique]
    with _legalize_parameter_names(expr_var_unique) as expr_params:
        ast_args = [ast.arg(param_name, None) for param_name in expr_params]
        ast_module = ast.parse('def {0}(): return'.format(expr_name), expr_filename, 'exec')
        assert hasattr(ast_module, 'body') and len(ast_module.body) == 1
        ast_fn = ast_module.body[0]
        ast_fn.args.args = ast_args
        (ast_fn.body[0].value, namespace) = _arr_expr_to_ast(expr.expr)
        _fix_invalid_lineno_ranges(ast_module)
    code_obj = compile(ast_module, expr_filename, 'exec')
    exec(code_obj, namespace)
    impl = namespace[expr_name]
    context = lowerer.context
    builder = lowerer.builder
    outer_sig = expr.ty(*(lowerer.typeof(name) for name in expr_args))
    inner_sig_args = []
    for argty in outer_sig.args:
        if isinstance(argty, types.Optional):
            argty = argty.type
        if isinstance(argty, types.Array):
            inner_sig_args.append(argty.dtype)
        else:
            inner_sig_args.append(argty)
    inner_sig = outer_sig.return_type.dtype(*inner_sig_args)
    flags = targetconfig.ConfigStack().top_or_none()
    flags = compiler.Flags() if flags is None else flags.copy()
    flags.error_model = 'numpy'
    cres = context.compile_subroutine(builder, impl, inner_sig, flags=flags, caching=False)
    from numba.np import npyimpl

    class ExprKernel(npyimpl._Kernel):

        def generate(self, *args):
            if False:
                while True:
                    i = 10
            arg_zip = zip(args, self.outer_sig.args, inner_sig.args)
            cast_args = [self.cast(val, inty, outty) for (val, inty, outty) in arg_zip]
            result = self.context.call_internal(builder, cres.fndesc, inner_sig, cast_args)
            return self.cast(result, inner_sig.return_type, self.outer_sig.return_type)
    ufunc = SimpleNamespace(nin=len(expr_args), nout=1, __name__=expr_name)
    ufunc.nargs = ufunc.nin + ufunc.nout
    args = [lowerer.loadvar(name) for name in expr_args]
    return npyimpl.numpy_ufunc_kernel(context, builder, outer_sig, args, ufunc, ExprKernel)