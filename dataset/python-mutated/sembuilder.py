"""Helper to quickly build instruction's semantic side effects"""
import inspect
import ast
import re
from future.utils import PY3
import miasm.expression.expression as m2_expr
from miasm.ir.ir import IRBlock, AssignBlock

class MiasmTransformer(ast.NodeTransformer):
    """AST visitor translating DSL to Miasm expression

    memX[Y]       -> ExprMem(Y, X)
    iX(Y)         -> ExprIntX(Y)
    X if Y else Z -> ExprCond(Y, X, Z)
    'X'(Y)        -> ExprOp('X', Y)
    ('X' % Y)(Z)  -> ExprOp('X' % Y, Z)
    {a, b}        -> ExprCompose(((a, 0, a.size), (b, a.size, a.size + b.size)))
    """
    parse_integer = re.compile('^i([0-9]+)$')
    parse_mem = re.compile('^mem([0-9]+)$')

    def visit_Call(self, node):
        if False:
            i = 10
            return i + 15
        "iX(Y) -> ExprIntX(Y),\n        'X'(Y) -> ExprOp('X', Y), ('X' % Y)(Z) -> ExprOp('X' % Y, Z)"
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            fc_name = node.func.id
            new_name = fc_name
            integer = self.parse_integer.search(fc_name)
            if integer is not None:
                size = int(integer.groups()[0])
                new_name = 'ExprInt'
                node.func.id = new_name
                node.args.append(ast.Num(n=size))
        elif isinstance(node.func, ast.Str) or (isinstance(node.func, ast.BinOp) and isinstance(node.func.op, ast.Mod) and isinstance(node.func.left, ast.Str)):
            op_name = node.func
            node.func = ast.Name(id='ExprOp', ctx=ast.Load())
            node.args[0:0] = [op_name]
        return node

    def visit_IfExp(self, node):
        if False:
            return 10
        'X if Y else Z -> ExprCond(Y, X, Z)'
        node = self.generic_visit(node)
        call = ast.Call(func=ast.Name(id='ExprCond', ctx=ast.Load()), args=[self.visit(node.test), self.visit(node.body), self.visit(node.orelse)], keywords=[], starargs=None, kwargs=None)
        return call

    def visit_Set(self, node):
        if False:
            while True:
                i = 10
        '{a, b} -> ExprCompose(a, b)'
        if len(node.elts) == 0:
            return node
        node = self.generic_visit(node)
        return ast.Call(func=ast.Name(id='ExprCompose', ctx=ast.Load()), args=node.elts, keywords=[], starargs=None, kwargs=None)
if PY3:

    def get_arg_name(name):
        if False:
            while True:
                i = 10
        return name.arg

    def gen_arg(name, ctx):
        if False:
            i = 10
            return i + 15
        return ast.arg(arg=name, ctx=ctx)
else:

    def get_arg_name(name):
        if False:
            print('Hello World!')
        return name.id

    def gen_arg(name, ctx):
        if False:
            return 10
        return ast.Name(id=name, ctx=ctx)

class SemBuilder(object):
    """Helper for building instruction's semantic side effects method

    This class provides a decorator @parse to use on them.
    The context in which the function will be parsed must be supplied on
    instantiation
    """

    def __init__(self, ctx):
        if False:
            print('Hello World!')
        'Create a SemBuilder\n        @ctx: context dictionary used during parsing\n        '
        self.transformer = MiasmTransformer()
        self._ctx = dict(m2_expr.__dict__)
        self._ctx['IRBlock'] = IRBlock
        self._ctx['AssignBlock'] = AssignBlock
        self._functions = {}
        self._ctx.update(ctx)

    @property
    def functions(self):
        if False:
            print('Hello World!')
        'Return a dictionary name -> func of parsed functions'
        return self._functions.copy()

    @staticmethod
    def _create_labels(loc_else=False):
        if False:
            return 10
        "Return the AST standing for label creations\n        @loc_else (optional): if set, create a label 'loc_else'"
        loc_end = 'loc_end = ir.get_next_loc_key(instr)'
        loc_end_expr = 'loc_end_expr = ExprLoc(loc_end, ir.IRDst.size)'
        out = ast.parse(loc_end).body
        out += ast.parse(loc_end_expr).body
        loc_if = 'loc_if = ir.loc_db.add_location()'
        loc_if_expr = 'loc_if_expr = ExprLoc(loc_if, ir.IRDst.size)'
        out += ast.parse(loc_if).body
        out += ast.parse(loc_if_expr).body
        if loc_else:
            loc_else = 'loc_else = ir.loc_db.add_location()'
            loc_else_expr = 'loc_else_expr = ExprLoc(loc_else, ir.IRDst.size)'
            out += ast.parse(loc_else).body
            out += ast.parse(loc_else_expr).body
        return out

    def _parse_body(self, body, argument_names):
        if False:
            return 10
        'Recursive function transforming a @body to a block expression\n        Return:\n         - AST to append to body (real python statements)\n         - a list of blocks, ie list of affblock, ie list of ExprAssign (AST)'
        real_body = []
        blocks = [[[]]]
        for statement in body:
            if isinstance(statement, ast.Assign):
                src = self.transformer.visit(statement.value)
                dst = self.transformer.visit(statement.targets[0])
                if isinstance(dst, ast.Name) and dst.id not in argument_names and (dst.id not in self._ctx) and (dst.id not in self._local_ctx):
                    statement.value = src
                    real_body.append(statement)
                    self._local_ctx[dst.id] = src
                    continue
                dst.ctx = ast.Load()
                res = ast.Call(func=ast.Name(id='ExprAssign', ctx=ast.Load()), args=[dst, src], keywords=[], starargs=None, kwargs=None)
                blocks[-1][-1].append(res)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Str):
                real_body.append(statement)
            elif isinstance(statement, ast.If):
                cond = statement.test
                real_body += self._create_labels(loc_else=True)
                loc_end = ast.Name(id='loc_end_expr', ctx=ast.Load())
                loc_if = ast.Name(id='loc_if_expr', ctx=ast.Load())
                loc_else = ast.Name(id='loc_else_expr', ctx=ast.Load()) if statement.orelse else loc_end
                dst = ast.Call(func=ast.Name(id='ExprCond', ctx=ast.Load()), args=[cond, loc_if, loc_else], keywords=[], starargs=None, kwargs=None)
                if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not):
                    dst.args[1:] = dst.args[1:][::-1]
                    dst.args[0] = cond.operand
                IRDst = ast.Attribute(value=ast.Name(id='ir', ctx=ast.Load()), attr='IRDst', ctx=ast.Load())
                loc_db = ast.Attribute(value=ast.Name(id='ir', ctx=ast.Load()), attr='loc_db', ctx=ast.Load())
                blocks[-1][-1].append(ast.Call(func=ast.Name(id='ExprAssign', ctx=ast.Load()), args=[IRDst, dst], keywords=[], starargs=None, kwargs=None))
                elements = [(statement.body, 'loc_if')]
                if statement.orelse:
                    elements.append((statement.orelse, 'loc_else'))
                for (content, loc_name) in elements:
                    (sub_blocks, sub_body) = self._parse_body(content, argument_names)
                    if len(sub_blocks) > 1:
                        raise RuntimeError('Imbricated if unimplemented')
                    jmp_end = ast.Call(func=ast.Name(id='ExprAssign', ctx=ast.Load()), args=[IRDst, loc_end], keywords=[], starargs=None, kwargs=None)
                    sub_blocks[-1][-1].append(jmp_end)
                    instr = ast.Name(id='instr', ctx=ast.Load())
                    effects = ast.List(elts=sub_blocks[-1][-1], ctx=ast.Load())
                    assignblk = ast.Call(func=ast.Name(id='AssignBlock', ctx=ast.Load()), args=[effects, instr], keywords=[], starargs=None, kwargs=None)
                    loc_if_name = ast.Name(id=loc_name, ctx=ast.Load())
                    assignblks = ast.List(elts=[assignblk], ctx=ast.Load())
                    sub_blocks[-1] = ast.Call(func=ast.Name(id='IRBlock', ctx=ast.Load()), args=[loc_db, loc_if_name, assignblks], keywords=[], starargs=None, kwargs=None)
                    blocks += sub_blocks
                    real_body += sub_body
                blocks.append([[]])
            else:
                raise RuntimeError('Unimplemented %s' % statement)
        return (blocks, real_body)

    def parse(self, func):
        if False:
            return 10
        'Function decorator, returning a correct method from a pseudo-Python\n        one'
        parsed = ast.parse(inspect.getsource(func))
        fc_ast = parsed.body[0]
        argument_names = [get_arg_name(name) for name in fc_ast.args.args]
        self._local_ctx = {}
        (blocks, body) = self._parse_body(fc_ast.body, argument_names)
        fc_ast.args.args[0:0] = [gen_arg('ir', ast.Param()), gen_arg('instr', ast.Param())]
        cur_instr = blocks[0][0]
        if len(blocks[-1][0]) == 0:
            blocks.pop()
        other_blocks = blocks[1:]
        body.append(ast.Return(value=ast.Tuple(elts=[ast.List(elts=cur_instr, ctx=ast.Load()), ast.List(elts=other_blocks, ctx=ast.Load())], ctx=ast.Load())))
        ret = ast.parse('')
        ret.body = [ast.FunctionDef(name=fc_ast.name, args=fc_ast.args, body=body, decorator_list=[])]
        fixed = ast.fix_missing_locations(ret)
        codeobj = compile(fixed, '<string>', 'exec')
        ctx = self._ctx.copy()
        eval(codeobj, ctx)
        self._functions[fc_ast.name] = ctx[fc_ast.name]
        return ctx[fc_ast.name]