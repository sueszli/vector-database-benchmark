"""This file contains Hy's core macros that are written in Python and
return compiler Result objects (or Python AST objects) rather than Hy
model trees. These macros serve the role of special forms in other
Lisps: all ordinary macros should eventually compile down to one of
these, or to one of the model builders in hy.compiler."""
import ast
import textwrap
from contextlib import nullcontext
from itertools import dropwhile
from funcparserlib.parser import finished, forward_decl, many, maybe, oneplus, some
from hy._compat import PY3_11, PY3_12
from hy.compiler import Result, asty, mkexpr
from hy.errors import HyEvalError, HyInternalError, HyTypeError
from hy.macros import pattern_macro, require, require_reader, local_macro_name
from hy.model_patterns import FORM, KEYWORD, LITERAL, STR, SYM, Tag, braces, brackets, dolike, in_tuple, keepsym, notpexpr, parse_if, pexpr, sym, tag, times, unpack
from hy.models import Bytes, Complex, Dict, Expression, FComponent, Float, FString, Integer, Keyword, List, Sequence, String, Symbol, Tuple, as_model, is_unpack
from hy.reader import mangle
from hy.scoping import OuterVar, ScopeFn, ScopeGen, ScopeLet, is_function_scope, is_inside_function_scope, nearest_python_scope
Inf = float('inf')

def pvalue(root, wanted):
    if False:
        for i in range(10):
            print('nop')
    return pexpr(sym(root) + wanted) >> (lambda x: x[0])

def maybe_annotated(target):
    if False:
        print('Hello World!')
    return pexpr(sym('annotate') + target + FORM).named('`annotate` form') | target >> (lambda x: (x, None))

def dotted(name):
    if False:
        while True:
            i = 10
    return Expression(map(Symbol, ['.', *name.split('.')]))
type_params = sym(':tp') + brackets(many(maybe_annotated(SYM) | unpack('either', Symbol)))

def digest_type_params(compiler, tp):
    if False:
        return 10
    'Return a `type_params` attribute for `FunctionDef` etc.'
    if tp:
        if not PY3_12:
            compiler._syntax_error(tp, '`:tp` requires Python 3.12 or later')
        (tp,) = tp
    elif not PY3_12:
        return {}
    return dict(type_params=[asty.TypeVarTuple(x[1], name=mangle(x[1])) if is_unpack('iterable', x) else asty.ParamSpec(x[1], name=mangle(x[1])) if is_unpack('mapping', x) else asty.TypeVar(x[0], name=mangle(x[0]), bound=x[1] and compiler.compile(x[1]).force_expr) for x in tp or []])

@pattern_macro('do', [many(FORM)])
def compile_do(compiler, expr, root, body):
    if False:
        while True:
            i = 10
    return compiler._compile_branch(body)

@pattern_macro(['eval-and-compile', 'eval-when-compile', 'do-mac'], [many(FORM)])
def compile_eval_foo_compile(compiler, expr, root, body):
    if False:
        return 10
    new_expr = Expression([Symbol('do').replace(expr[0])]).replace(expr)
    try:
        value = compiler.eval(new_expr + body)
    except HyInternalError:
        raise
    except Exception as e:
        raise HyEvalError(str(e), compiler.filename, body, compiler.source)
    return compiler.compile(as_model(value)) if root == 'do-mac' else compiler._compile_branch(body) if root == 'eval-and-compile' else Result()

@pattern_macro(['py', 'pys'], [STR])
def compile_inline_python(compiler, expr, root, code):
    if False:
        return 10
    exec_mode = root == 'pys'
    try:
        o = asty.parse(expr, textwrap.dedent(code) if exec_mode else '(' + code + '\n)', compiler.filename, 'exec' if exec_mode else 'eval').body
    except (SyntaxError, ValueError) as e:
        raise compiler._syntax_error(expr, "Python parse error in '{}': {}".format(root, e))
    return Result(stmts=o) if exec_mode else o

@pattern_macro(['quote', 'quasiquote'], [FORM])
def compile_quote(compiler, expr, root, arg):
    if False:
        i = 10
        return i + 15
    return compiler.compile(render_quoted_form(compiler, arg, level=Inf if root == 'quote' else 0)[0])

def render_quoted_form(compiler, form, level):
    if False:
        for i in range(10):
            print('nop')
    '\n    Render a quoted form as a new hy Expression.\n\n    `level` is the level of quasiquoting of the current form. We can\n    unquote if level is 0.\n\n    Returns a two-tuple (`expression`, `splice`).\n\n    The `splice` return value is used to mark `unquote-splice`d forms.\n    We need to distinguish them as want to concatenate them instead of\n    just nesting them.\n    '
    op = None
    if isinstance(form, Expression) and form and isinstance(form[0], Symbol):
        op = mangle(form[0]).replace('_', '-')
        if op in ('unquote', 'unquote-splice', 'quasiquote'):
            if level == 0 and op != 'quasiquote':
                if len(form) != 2:
                    raise HyTypeError("`%s' needs 1 argument, got %s" % op, len(form) - 1, compiler.filename, form, compiler.source)
                return (form[1], op == 'unquote-splice')
            level += 1 if op == 'quasiquote' else -1
    name = form.__class__.__name__
    body = [form]
    if isinstance(form, Sequence):
        contents = []
        for x in form:
            (f_contents, splice) = render_quoted_form(compiler, x, level)
            if splice:
                if is_unpack('iterable', f_contents):
                    raise compiler._syntax_error(f_contents, '`unpack-iterable` is not allowed here')
                f_contents = Expression([Symbol('unpack-iterable'), Expression([Symbol('or'), f_contents, List()])])
            contents.append(f_contents)
        body = [List(contents)]
        if isinstance(form, FString) and form.brackets is not None:
            body.extend([Keyword('brackets'), String(form.brackets)])
        elif isinstance(form, FComponent) and form.conversion is not None:
            body.extend([Keyword('conversion'), String(form.conversion)])
    elif isinstance(form, Symbol):
        body = [String(form), Keyword('from_parser'), Symbol('True')]
    elif isinstance(form, Keyword):
        body = [String(form.name), Keyword('from_parser'), Symbol('True')]
    elif isinstance(form, String):
        if form.brackets is not None:
            body.extend([Keyword('brackets'), String(form.brackets)])
    return (Expression([dotted('hy.models.' + name), *body]).replace(form), False)

@pattern_macro(['not', 'bnot'], [FORM], shadow=True)
def compile_unary_operator(compiler, expr, root, arg):
    if False:
        print('Hello World!')
    ops = {'not': ast.Not, 'bnot': ast.Invert}
    operand = compiler.compile(arg)
    return operand + asty.UnaryOp(expr, op=ops[root](), operand=operand.force_expr)

@pattern_macro(['and', 'or'], [many(FORM)], shadow=True)
def compile_logical_or_and_and_operator(compiler, expr, operator, args):
    if False:
        print('Hello World!')
    ops = {'and': (ast.And, True), 'or': (ast.Or, None)}
    (opnode, default) = ops[operator]
    if len(args) == 0:
        return asty.Constant(expr[0], value=default)
    ret = None
    var = None
    assignment = None
    stmts = None
    can_append = False

    def put(node, value):
        if False:
            i = 10
            return i + 15
        nonlocal var, assignment, can_append
        if var is None:
            var = compiler.get_anon_var()
        name = asty.Name(node, id=var, ctx=ast.Store())
        ret.temp_variables.append(name)
        can_append = False
        return (assignment := asty.Assign(node, targets=[name], value=value))

    def get(node):
        if False:
            return 10
        if var is None:
            stmts.append(put(node, ret.force_expr))
        name = asty.Name(node, id=var, ctx=ast.Load())
        ret.temp_variables.append(name)
        return name
    for value in map(compiler.compile, args):
        if ret is None:
            ret = value
            stmts = ret.stmts
            can_append = False
        elif value.stmts:
            node = value.stmts[0]
            cond = get(node)
            if operator == 'or':
                cond = asty.UnaryOp(node, op=ast.Not(), operand=cond)
            branch = asty.If(node, test=cond, body=value.stmts, orelse=[])
            stmts.append(branch)
            stmts = branch.body
            stmts.append(put(node, value.force_expr))
        else:
            value = value.force_expr

            def enbool(expr):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal can_append
                if can_append:
                    expr.values.append(value)
                    return expr
                can_append = True
                return asty.BoolOp(expr, op=opnode(), values=[expr, value])
            if assignment:
                assignment.value = enbool(assignment.value)
            else:
                ret.expr = enbool(ret.expr)
    if var:
        ret.expr = get(expr)
    return ret
c_ops = {'=': ast.Eq, '!=': ast.NotEq, '<': ast.Lt, '<=': ast.LtE, '>': ast.Gt, '>=': ast.GtE, 'is': ast.Is, 'is-not': ast.IsNot, 'in': ast.In, 'not-in': ast.NotIn}
c_ops = {mangle(k): v for (k, v) in c_ops.items()}

def get_c_op(compiler, sym):
    if False:
        return 10
    k = mangle(sym)
    if k not in c_ops:
        raise compiler._syntax_error(sym, 'Illegal comparison operator: ' + str(sym))
    return c_ops[k]()

@pattern_macro(['=', 'is', '<', '<=', '>', '>='], [oneplus(FORM)], shadow=True)
@pattern_macro(['!=', 'is-not', 'in', 'not-in'], [times(2, Inf, FORM)], shadow=True)
def compile_compare_op_expression(compiler, expr, root, args):
    if False:
        for i in range(10):
            print('nop')
    if len(args) == 1:
        return compiler.compile(args[0]) + asty.Constant(expr, value=True)
    ops = [get_c_op(compiler, root) for _ in args[1:]]
    (exprs, ret, _) = compiler._compile_collect(args)
    return ret + asty.Compare(expr, left=exprs[0], ops=ops, comparators=exprs[1:])

@pattern_macro('chainc', [FORM, many(SYM + FORM)])
def compile_chained_comparison(compiler, expr, root, arg1, args):
    if False:
        i = 10
        return i + 15
    ret = compiler.compile(arg1)
    arg1 = ret.force_expr
    ops = [get_c_op(compiler, op) for (op, _) in args]
    (args, ret2, _) = compiler._compile_collect([x for (_, x) in args])
    return ret + ret2 + asty.Compare(expr, left=arg1, ops=ops, comparators=args)
m_ops = {'+': (ast.Add, '+'), '/': (ast.Div, '*'), '//': (ast.FloorDiv, '*'), '*': (ast.Mult, '*'), '-': (ast.Sub, '+'), '%': (ast.Mod, None), '**': (ast.Pow, '**'), '<<': (ast.LShift, '+'), '>>': (ast.RShift, '+'), '|': (ast.BitOr, '|'), '^': (ast.BitXor, None), '&': (ast.BitAnd, '&'), '@': (ast.MatMult, '@')}

@pattern_macro(['+', '*', '|'], [many(FORM)], shadow=True)
@pattern_macro(['-', '/', '&', '@'], [oneplus(FORM)], shadow=True)
@pattern_macro(['**', '//', '<<', '>>'], [times(2, Inf, FORM)], shadow=True)
@pattern_macro(['%', '^'], [times(2, 2, FORM)], shadow=True)
def compile_maths_expression(compiler, expr, root, args):
    if False:
        return 10
    if len(args) == 0:
        return asty.Constant(expr, value={'+': 0, '|': 0, '*': 1}[root])
    if len(args) == 1:
        if root == '/':
            args = [Integer(1).replace(expr), args[0]]
        elif root in ('+', '-'):
            op = {'+': ast.UAdd, '-': ast.USub}[root]()
            ret = compiler.compile(args[0])
            return ret + asty.UnaryOp(expr, op=op, operand=ret.force_expr)
        else:
            return compiler.compile(args[0])
    op = m_ops[root][0]
    right_associative = root == '**'
    ret = compiler.compile(args[-1 if right_associative else 0])
    for child in args[-2 if right_associative else 1::-1 if right_associative else 1]:
        left_expr = ret.force_expr
        ret += compiler.compile(child)
        right_expr = ret.force_expr
        if right_associative:
            (left_expr, right_expr) = (right_expr, left_expr)
        ret += asty.BinOp(expr, left=left_expr, op=op(), right=right_expr)
    return ret
a_ops = {x + '=': v for (x, v) in m_ops.items()}

@pattern_macro([x for (x, (_, v)) in a_ops.items() if v is not None], [FORM, oneplus(FORM)])
@pattern_macro([x for (x, (_, v)) in a_ops.items() if v is None], [FORM, times(1, 1, FORM)])
def compile_augassign_expression(compiler, expr, root, target, values):
    if False:
        return 10
    if len(values) > 1:
        return compiler.compile(mkexpr(root, [target], mkexpr(a_ops[root][1], rest=values)).replace(expr))
    op = a_ops[root][0]
    target = compiler._storeize(target, compiler.compile(target))
    ret = compiler.compile(values[0])
    return ret + asty.AugAssign(expr, target=target, value=ret.force_expr, op=op())

@pattern_macro('setv', [many(maybe_annotated(FORM) + FORM)])
@pattern_macro(((3, 8), 'setx'), [times(1, 1, SYM + FORM)])
def compile_def_expression(compiler, expr, root, decls):
    if False:
        return 10
    if not decls:
        return asty.Constant(expr, value=None)
    result = Result()
    is_assignment_expr = root == 'setx'
    for decl in decls:
        if is_assignment_expr:
            ann = None
            (name, value) = decl
        else:
            ((name, ann), value) = decl
        result += compile_assign(compiler, ann, name, value, is_assignment_expr=is_assignment_expr)
    return result

@pattern_macro(['annotate'], [FORM, FORM])
def compile_basic_annotation(compiler, expr, root, target, ann):
    if False:
        i = 10
        return i + 15
    return compile_assign(compiler, ann, target, None)

def compile_assign(compiler, ann, name, value, *, is_assignment_expr=False, let_scope=None):
    if False:
        while True:
            i = 10
    assert not is_assignment_expr or (value is not None and ann is None)
    annotate_only = value is None
    if annotate_only:
        result = Result()
    else:
        with let_scope or nullcontext():
            result = compiler.compile(value)
        if let_scope:
            name = let_scope.add(name)
    ld_name = compiler.compile(name)
    if result.temp_variables and isinstance(name, Symbol):
        result.rename(compiler, compiler._nonconst(name))
        if not is_assignment_expr:
            result.expr = None
    else:
        st_name = compiler._storeize(name, ld_name)
        if ann is not None:
            ann_result = compiler.compile(ann)
            result = ann_result + result
        if is_assignment_expr:
            node = asty.NamedExpr
        elif ann is not None:
            node = lambda x, **kw: asty.AnnAssign(x, annotation=ann_result.force_expr, simple=int(isinstance(name, Symbol)), **kw)
        else:
            node = asty.Assign
        result += node(name if hasattr(name, 'start_line') else result, value=result.force_expr if not annotate_only else None, target=st_name, targets=[st_name])
    return result

@pattern_macro(['global', 'nonlocal'], [many(SYM)])
def compile_global_or_nonlocal(compiler, expr, root, syms):
    if False:
        print('Hello World!')
    if not syms:
        return asty.Pass(expr)
    names = [mangle(s) for s in syms]
    if root == 'global':
        ret = asty.Global(expr, names=names)
    else:
        ret = OuterVar(expr, compiler.scope, names)
    try:
        compiler.scope.define_nonlocal(ret, root)
    except SyntaxError as e:
        raise compiler._syntax_error(expr, e.msg)
    return ret if syms else Result()

@pattern_macro('del', [many(FORM)])
def compile_del_expression(compiler, expr, name, args):
    if False:
        i = 10
        return i + 15
    if not args:
        return asty.Pass(expr)
    del_targets = []
    ret = Result()
    for target in args:
        compiled_target = compiler.compile(target)
        ret += compiled_target
        del_targets.append(compiler._storeize(target, compiled_target, ast.Del))
    return ret + asty.Delete(expr, targets=del_targets)

@pattern_macro('get', [FORM, oneplus(FORM)], shadow=True)
def compile_index_expression(compiler, expr, name, obj, indices):
    if False:
        i = 10
        return i + 15
    (indices, ret, _) = compiler._compile_collect(indices)
    ret += compiler.compile(obj)
    for ix in indices:
        ret += asty.Subscript(expr, value=ret.force_expr, slice=ast.Index(value=ix), ctx=ast.Load())
    return ret
notsym = lambda *dissallowed: some(lambda x: isinstance(x, Symbol) and str(x) not in dissallowed)

@pattern_macro('.', [FORM, many(SYM | brackets(FORM) | pexpr(notsym('unpack-iterable', 'unpack-mapping'), many(FORM)))])
def compile_attribute_access(compiler, expr, name, invocant, keys):
    if False:
        while True:
            i = 10
    ret = compiler.compile(invocant)
    for attr in keys:
        if isinstance(attr, Symbol):
            ret += asty.Attribute(attr, value=ret.force_expr, attr=mangle(attr), ctx=ast.Load())
        elif isinstance(attr, Expression):
            (root, args) = attr
            func = asty.Attribute(root, value=ret.force_expr, attr=mangle(root), ctx=ast.Load())
            (args, funcret, keywords) = compiler._compile_collect(args, with_kwargs=True)
            ret += funcret + func + asty.Call(expr, func=func, args=args, keywords=keywords)
        else:
            compiled_attr = compiler.compile(attr[0])
            ret = compiled_attr + ret + asty.Subscript(attr, value=ret.force_expr, slice=ast.Index(value=compiled_attr.force_expr), ctx=ast.Load())
    return ret

@pattern_macro('cut', [FORM, maybe(FORM), maybe(FORM), maybe(FORM)])
def compile_cut_expression(compiler, expr, name, obj, lower, upper, step):
    if False:
        print('Hello World!')
    ret = [Result()]

    def c(e):
        if False:
            for i in range(10):
                print('nop')
        ret[0] += compiler.compile(e)
        return ret[0].force_expr
    if upper is None:
        upper = lower
        lower = Symbol('None')
    s = asty.Subscript(expr, value=c(obj), slice=asty.Slice(expr, lower=c(lower), upper=c(upper), step=c(step)), ctx=ast.Load())
    return ret[0] + s

@pattern_macro('unpack-iterable', [FORM])
def compile_unpack_iterable(compiler, expr, root, arg):
    if False:
        i = 10
        return i + 15
    ret = compiler.compile(arg)
    ret += asty.Starred(expr, value=ret.force_expr, ctx=ast.Load())
    return ret

@pattern_macro('if', [FORM, FORM, FORM])
def compile_if(compiler, expr, _, cond, body, orel_expr):
    if False:
        for i in range(10):
            print('nop')
    cond = compiler.compile(cond)
    body = compiler.compile(body)
    nested = root = False
    orel = Result()
    if isinstance(orel_expr, Expression) and isinstance(orel_expr[0], Symbol) and (orel_expr[0] == Symbol('if*')):
        root = compiler.temp_if is None
        nested = True
        compiler.temp_if = compiler.temp_if or compiler.get_anon_var()
    orel = compiler.compile(orel_expr)
    if not cond.stmts and isinstance(cond.force_expr, ast.Name):
        name = cond.force_expr.id
        branch = None
        if name == 'True':
            branch = body
        elif name in ('False', 'None'):
            branch = orel
        if branch is not None:
            if compiler.temp_if and branch.stmts:
                name = asty.Name(expr, id=mangle(compiler.temp_if), ctx=ast.Store())
                branch += asty.Assign(expr, targets=[name], value=body.force_expr)
            return branch
    ret = cond
    if body.stmts or orel.stmts:
        var = compiler.temp_if or compiler.get_anon_var()
        name = asty.Name(expr, id=mangle(var), ctx=ast.Store())
        body += asty.Assign(expr, targets=[name], value=body.force_expr)
        if not nested or not orel.stmts or (not root and var != compiler.temp_if):
            orel += asty.Assign(expr, targets=[name], value=orel.force_expr)
        ret += asty.If(expr, test=ret.force_expr, body=body.stmts, orelse=orel.stmts)
        expr_name = asty.Name(expr, id=mangle(var), ctx=ast.Load())
        ret += Result(expr=expr_name, temp_variables=[expr_name, name])
    else:
        ret += asty.IfExp(expr, test=ret.force_expr, body=body.force_expr, orelse=orel.force_expr)
    if root:
        compiler.temp_if = None
    return ret
loopers = many(tag('setv', sym(':setv') + FORM + FORM) | tag('if', sym(':if') + FORM) | tag('do', sym(':do') + FORM) | tag('afor', sym(':async') + FORM + FORM) | tag('for', FORM + FORM))

@pattern_macro(['for'], [brackets(loopers, name='square-bracketed loop clauses'), many(notpexpr('else')) + maybe(dolike('else'))])
@pattern_macro(['lfor', 'sfor', 'gfor'], [loopers, FORM])
@pattern_macro(['dfor'], [loopers, finished])
def compile_comprehension(compiler, expr, root, parts, final):
    if False:
        for i in range(10):
            print('nop')
    node_class = {'for': asty.For, 'lfor': asty.ListComp, 'dfor': asty.DictComp, 'sfor': asty.SetComp, 'gfor': asty.GeneratorExp}[root]
    is_for = root == 'for'
    ctx = nullcontext() if is_for else compiler.scope.create(ScopeGen)
    mac_con = nullcontext() if is_for else compiler.local_state()
    with mac_con, ctx as scope:
        if is_for:
            parts = parts[0]
        if node_class is asty.DictComp:
            if not (parts and parts[-1].tag == 'for'):
                raise compiler._syntax_error(parts[-1] if parts else parts, '`dfor` must end with key and value forms')
            final = parts.pop().value
        if not parts:
            return Result(expr=asty.parse(expr, {asty.For: 'None', asty.ListComp: '[]', asty.DictComp: '{}', asty.SetComp: '{1}.__class__()', asty.GeneratorExp: '(_ for _ in [])'}[node_class]).body[0].value)
        new_parts = []
        for p in parts:
            if p.tag in ('if', 'do'):
                tag_value = compiler.compile(p.value)
            else:
                tag_value = [compiler._storeize(p.value[0], compiler.compile(p.value[0])), compiler.compile(p.value[1])]
                if not is_for:
                    scope.iterator(tag_value[0])
            new_parts.append(Tag(p.tag, tag_value))
        parts = new_parts
        orel = []
        if is_for:
            (body, else_expr) = final
            if else_expr is not None:
                orel.append(compiler._compile_branch(else_expr))
                orel[0] += orel[0].expr_as_stmt()
        elif node_class is asty.DictComp:
            (key, elt) = map(compiler.compile, final)
        else:
            key = None
            elt = compiler.compile(final)
        if is_for or elt.stmts or (key is not None and key.stmts) or any((p.tag == 'do' or (p.value[1].stmts if p.tag in ('for', 'afor', 'setv') else p.value.stmts) for p in parts)):

            def f(parts):
                if False:
                    print('Hello World!')
                if not parts:
                    if is_for:
                        if body:
                            bd = compiler._compile_branch(body)
                            return bd + bd.expr_as_stmt()
                        return Result(stmts=[asty.Pass(expr)])
                    if node_class is asty.DictComp:
                        ret = key + elt
                        val = asty.Tuple(key, ctx=ast.Load(), elts=[key.force_expr, elt.force_expr])
                    else:
                        ret = elt
                        val = elt.force_expr
                    return ret + asty.Expr(elt, value=asty.Yield(elt, value=val))
                ((tagname, v), parts) = (parts[0], parts[1:])
                if tagname in ('for', 'afor'):
                    orelse = orel and orel.pop().stmts
                    node = asty.AsyncFor if tagname == 'afor' else asty.For
                    return v[1] + node(v[1], target=v[0], iter=v[1].force_expr, body=f(parts).stmts, orelse=orelse)
                elif tagname == 'setv':
                    return v[1] + asty.Assign(v[1], targets=[v[0]], value=v[1].force_expr) + f(parts)
                elif tagname == 'if':
                    return v + asty.If(v, test=v.force_expr, body=f(parts).stmts, orelse=[])
                elif tagname == 'do':
                    return v + v.expr_as_stmt() + f(parts)
                else:
                    raise ValueError("can't happen")
            if is_for:
                return f(parts)
            fname = compiler.get_anon_var()
            stmts = []
            ret = Result()
            assignment_names = scope.finalize()
            if scope.exposing_assignments and assignment_names:
                unlocal_type = asty.Nonlocal if is_inside_function_scope(scope.parent) else asty.Global
                stmts.append(unlocal_type(expr, names=assignment_names))
                if_body = []
                if scope.nonlocal_vars:
                    if_body.append(asty.Nonlocal(expr, names=list(sorted(scope.nonlocal_vars))))
                assignments = asty.Tuple(expr, elts=[asty.Name(expr, id=var, ctx=ast.Store()) for var in assignment_names], ctx=ast.Store())
                if_body.append(asty.Assign(expr, targets=[assignments], value=asty.Constant(expr, value=None)))
                ret += asty.If(expr, test=asty.Constant(expr, value=False), body=if_body, orelse=[])
            ret += asty.FunctionDef(expr, name=fname, args=ast.arguments(args=[], vararg=None, kwarg=None, posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=stmts + f(parts).stmts, decorator_list=[], **{'type_params': []} if PY3_12 else {})
            return ret + Result(expr=asty.parse(expr, '{}({}())'.format({asty.ListComp: '[].__class__', asty.DictComp: '{}.__class__', asty.SetComp: '{1}.__class__', asty.GeneratorExp: ''}[node_class], fname)).body[0].value)
        generators = []
        for (tagname, v) in parts:
            if tagname in ('for', 'afor'):
                generators.append(ast.comprehension(target=v[0], iter=v[1].expr, ifs=[], is_async=int(tagname == 'afor')))
            elif tagname == 'setv':
                generators.append(ast.comprehension(target=v[0], iter=asty.Tuple(v[1], elts=[v[1].expr], ctx=ast.Load()), ifs=[], is_async=0))
            elif tagname == 'if':
                generators[-1].ifs.append(v.expr)
            else:
                raise ValueError("can't happen")
        if node_class is asty.DictComp:
            return asty.DictComp(expr, key=key.expr, value=elt.expr, generators=generators)
        return node_class(expr, elt=elt.expr, generators=generators)

@pattern_macro(['while'], [FORM, many(notpexpr('else')), maybe(dolike('else'))])
def compile_while_expression(compiler, expr, root, cond, body, else_expr):
    if False:
        i = 10
        return i + 15
    cond_compiled = compiler.compile(cond)
    body = compiler._compile_branch(body)
    body += body.expr_as_stmt()
    body_stmts = body.stmts or [asty.Pass(expr)]
    if cond_compiled.stmts:
        cond_var = asty.Name(cond, id=compiler.get_anon_var(), ctx=ast.Load())

        def make_not(operand):
            if False:
                return 10
            return asty.UnaryOp(cond, op=ast.Not(), operand=operand)
        body_stmts = cond_compiled.stmts + [asty.Assign(cond, targets=[compiler._storeize(cond, cond_var)], value=make_not(make_not(cond_compiled.force_expr))), asty.If(cond, test=cond_var, body=body_stmts, orelse=[])]
        cond_compiled = Result() + asty.Assign(cond, targets=[compiler._storeize(cond, cond_var)], value=asty.Constant(cond, value=True)) + cond_var
    orel = Result()
    if else_expr is not None:
        orel = compiler._compile_branch(else_expr)
        orel += orel.expr_as_stmt()
    ret = cond_compiled + asty.While(expr, test=cond_compiled.force_expr, body=body_stmts, orelse=orel.stmts)
    return ret

@pattern_macro(['break', 'continue'], [])
def compile_break_or_continue_expression(compiler, expr, root):
    if False:
        for i in range(10):
            print('nop')
    return (asty.Break if root == 'break' else asty.Continue)(expr)

@pattern_macro(['with', 'with/a'], [brackets(oneplus(FORM + FORM)) | brackets(FORM >> (lambda x: [(Symbol('_'), x)])), many(FORM)])
def compile_with_expression(compiler, expr, root, args, body):
    if False:
        i = 10
        return i + 15
    body = compiler._compile_branch(body)
    temp_var = compiler.get_anon_var()
    name = asty.Name(expr, id=mangle(temp_var), ctx=ast.Store())
    body += asty.Assign(expr, targets=[name], value=body.force_expr)
    initial_assign = asty.Assign(expr, targets=[name], value=asty.Constant(expr, value=None))
    ret = Result(stmts=[initial_assign])
    items = []
    for (variable, ctx) in args[0]:
        ctx = compiler.compile(ctx)
        ret += ctx
        variable = None if variable == Symbol('_') else compiler._storeize(variable, compiler.compile(variable))
        items.append(asty.withitem(expr, context_expr=ctx.force_expr, optional_vars=variable))
    node = asty.With if root == 'with' else asty.AsyncWith
    ret += node(expr, body=body.stmts, items=items)
    expr_name = asty.Name(expr, id=mangle(temp_var), ctx=ast.Load())
    ret += Result(expr=expr_name)
    return ret
_pattern = forward_decl()
_pattern.define((SYM | KEYWORD | LITERAL | brackets(many(_pattern | unpack('iterable'))) | in_tuple(many(_pattern | unpack('iterable'))) | pexpr(keepsym('.'), many(SYM)) | pexpr(keepsym('|'), many(_pattern)) | braces(many(LITERAL + _pattern), maybe(pvalue('unpack-mapping', SYM))) | pexpr(pexpr(keepsym('.'), oneplus(SYM)) | notsym('.', '|', 'unpack-mapping', 'unpack-iterable'), many(parse_if(lambda x: not isinstance(x, Keyword), _pattern)), many(KEYWORD + _pattern))) + maybe(sym(':as') + SYM))
match_clause = _pattern + maybe(sym(':if') + FORM)

@pattern_macro(((3, 10), 'match'), [FORM, many(match_clause + FORM)])
def compile_match_expression(compiler, expr, root, subject, clauses):
    if False:
        while True:
            i = 10
    subject = compiler.compile(subject)
    return_var = asty.Name(expr, id=mangle(compiler.get_anon_var()), ctx=ast.Store())
    lifted_if_defs = []
    match_cases = []
    for (*pattern, guard, body) in clauses:
        if guard and body == Keyword('as'):
            raise compiler._syntax_error(body, ':as clause cannot come after :if guard')
        body = compiler._compile_branch([body])
        body += asty.Assign(pattern[0], targets=[return_var], value=body.force_expr)
        body += body.expr_as_stmt()
        body = body.stmts
        pattern = compile_pattern(compiler, pattern)
        if guard:
            guard = compiler.compile(guard)
            if guard.stmts:
                fname = compiler.get_anon_var()
                guardret = Result() + asty.FunctionDef(guard, name=fname, args=ast.arguments(args=[], varargs=None, kwarg=None, posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=guard.stmts + [asty.Return(guard.expr, value=guard.expr)], decorator_list=[], **{'type_params': []} if PY3_12 else {})
                lifted_if_defs.append(guardret)
                guard = Result(expr=asty.parse(guard, f'{fname}()').body[0].value)
        match_cases.append(ast.match_case(pattern=pattern, guard=guard.force_expr if guard else None, body=body))
    returnable = Result(expr=asty.Name(expr, id=return_var.id, ctx=ast.Load()), temp_variables=[return_var])
    ret = Result() + subject
    ret += asty.Assign(expr, targets=[return_var], value=asty.Constant(expr, value=None))
    if not match_cases:
        return ret + returnable
    for lifted_if in lifted_if_defs:
        ret += lifted_if
    ret += asty.Match(expr, subject=subject.force_expr, cases=match_cases)
    return ret + returnable

def compile_pattern(compiler, pattern):
    if False:
        i = 10
        return i + 15
    (value, assignment) = pattern
    if assignment is not None:
        return compiler.scope.assign(asty.MatchAs(value, pattern=compile_pattern(compiler, (value, None)), name=mangle(compiler._nonconst(assignment))))
    if str(value) in ('None', 'True', 'False'):
        return asty.MatchSingleton(value, value=compiler.compile(value).force_expr.value)
    elif isinstance(value, (String, Integer, Float, Complex, Bytes)):
        return asty.MatchValue(value, value=compiler.compile(value).expr)
    elif value == Symbol('_'):
        return asty.MatchAs(value)
    elif isinstance(value, Symbol):
        return compiler.scope.assign(asty.MatchAs(value, name=mangle(value)))
    elif isinstance(value, Expression) and value[0] == Symbol('|'):
        return asty.MatchOr(value, patterns=[compile_pattern(compiler, v) for v in value[1]])
    elif isinstance(value, Expression) and value[0] == Symbol('.'):
        (root, syms) = value
        dotform = mkexpr(root, *syms).replace(value)
        return asty.MatchValue(value, value=compiler.compile(dotform).expr)
    elif isinstance(value, (Tuple, List)):
        patterns = value[0]
        patterns = [compile_pattern(compiler, (v, None) if is_unpack('iterable', v) else v) for v in patterns]
        return asty.MatchSequence(value, patterns=patterns)
    elif is_unpack('iterable', value):
        return compiler.scope.assign(asty.MatchStar(value, name=mangle(value[1])))
    elif isinstance(value, Dict):
        (kvs, rest) = value
        (keys, values) = zip(*kvs) if kvs else ([], [])
        return compiler.scope.assign(asty.MatchMapping(value, keys=[compiler.compile(key).expr for key in keys], patterns=[compile_pattern(compiler, v) for v in values], rest=mangle(rest) if rest else None))
    elif isinstance(value, Expression):
        (head, args, kwargs) = value
        (keywords, values) = zip(*kwargs) if kwargs else ([], [])
        return asty.MatchClass(value, cls=compiler.compile((head[:1] + head[1]).replace(head) if type(head) is Expression else head).expr, patterns=[compile_pattern(compiler, v) for v in args], kwd_attrs=[kwd.name for kwd in keywords], kwd_patterns=[compile_pattern(compiler, value) for value in values])
    elif isinstance(value, Keyword):
        return asty.MatchClass(value, cls=compiler.compile(dotted('hy.models.Keyword')).expr, patterns=[asty.MatchValue(value, value=asty.Constant(value, value=value.name))], kwd_attrs=[], kwd_patterns=[])
    else:
        raise compiler._syntax_error(value, 'unsupported')

@pattern_macro('raise', [maybe(FORM), maybe(sym(':from') + FORM)])
def compile_raise_expression(compiler, expr, root, exc, cause):
    if False:
        while True:
            i = 10
    ret = Result()
    if exc is not None:
        exc = compiler.compile(exc)
        ret += exc
        exc = exc.force_expr
    if cause is not None:
        cause = compiler.compile(cause)
        ret += cause
        cause = cause.force_expr
    return ret + asty.Raise(expr, type=ret.expr, exc=exc, inst=None, tback=None, cause=cause)

@pattern_macro('try', [many(notpexpr('except', 'except*', 'else', 'finally')), many(pexpr(keepsym('except') | keepsym('except*'), brackets() | brackets(FORM) | brackets(SYM, FORM), many(FORM))), maybe(dolike('else')), maybe(dolike('finally'))])
def compile_try_expression(compiler, expr, root, body, catchers, orelse, finalbody):
    if False:
        for i in range(10):
            print('nop')
    if orelse is not None and (not catchers):
        body += list(orelse)
        orelse = None
    body = compiler._compile_branch(body)
    if not (catchers or finalbody):
        return body
    return_var = asty.Name(expr, id=mangle(compiler.get_anon_var()), ctx=ast.Store())
    handler_results = Result()
    handlers = []
    except_syms_seen = set()
    for catcher in catchers:
        (except_sym, exceptions, ebody) = catcher
        if not PY3_11 and except_sym == Symbol('except*'):
            hy_compiler._syntax_error(except_sym, '`{}` requires Python 3.11 or later')
        except_syms_seen.add(str(except_sym))
        if len(except_syms_seen) > 1:
            raise compiler._syntax_error(except_sym, 'cannot have both `except` and `except*` on the same `try`')
        name = None
        if len(exceptions) == 2:
            name = mangle(compiler._nonconst(exceptions[0]))
        exceptions_list = exceptions[-1] if exceptions else List()
        if isinstance(exceptions_list, List):
            if len(exceptions_list):
                (elts, types, _) = compiler._compile_collect(exceptions_list)
                types += asty.Tuple(exceptions_list, elts=elts, ctx=ast.Load())
            else:
                types = Result()
        else:
            types = compiler.compile(exceptions_list)
        with compiler.scope.create(ScopeLet) as scope:
            if name:
                scope.add(name, name)
            ebody = compiler._compile_branch(ebody)
        ebody += asty.Assign(catcher, targets=[return_var], value=ebody.force_expr)
        ebody += ebody.expr_as_stmt()
        handler_results += types + asty.ExceptHandler(catcher, type=types.expr, name=name, body=ebody.stmts or [asty.Pass(catcher)])
        handlers.append(handler_results.stmts.pop())
    if orelse is None:
        orelse = []
    else:
        orelse = compiler._compile_branch(orelse)
        orelse += asty.Assign(expr, targets=[return_var], value=orelse.force_expr)
        orelse += orelse.expr_as_stmt()
        orelse = orelse.stmts
    if finalbody is None:
        finalbody = []
    else:
        finalbody = compiler._compile_branch(finalbody)
        finalbody += finalbody.expr_as_stmt()
        finalbody = finalbody.stmts
    returnable = Result(expr=asty.Name(expr, id=return_var.id, ctx=ast.Load()), temp_variables=[return_var])
    body += body.expr_as_stmt() if orelse else asty.Assign(expr, targets=[return_var], value=body.force_expr)
    body = body.stmts or [asty.Pass(expr)]
    x = (asty.TryStar if 'except*' in except_syms_seen else asty.Try)(expr, body=body, handlers=handlers, orelse=orelse, finalbody=finalbody)
    return handler_results + x + returnable
NASYM = some(lambda x: isinstance(x, Symbol) and x not in (Symbol('/'), Symbol('*')))
argument = maybe_annotated(NASYM | brackets(NASYM, FORM))
varargs = lambda unpack_type, wanted: maybe_annotated(pvalue(unpack_type, wanted))
kwonly_delim = some(lambda x: x == Symbol('*'))
lambda_list = brackets(maybe(many(argument) + sym('/')), many(argument), maybe(kwonly_delim | varargs('unpack-iterable', NASYM)), many(argument), maybe(varargs('unpack-mapping', NASYM)))

@pattern_macro(['fn', 'fn/a'], [maybe(type_params), maybe_annotated(lambda_list), many(FORM)])
def compile_function_lambda(compiler, expr, root, tp, params, body):
    if False:
        return 10
    is_async = root == 'fn/a'
    (params, returns) = params
    (posonly, args, rest, kwonly, kwargs) = params
    has_annotations = returns is not None or any((isinstance(param, tuple) and param[1] is not None for param in (posonly or []) + args + kwonly + [rest, kwargs]))
    (args, ret) = compile_lambda_list(compiler, params)
    with compiler.local_state(), compiler.scope.create(ScopeFn, args, is_async) as scope:
        body = compiler._compile_branch(body)
    if not (has_annotations or tp or body.stmts or is_async):
        return ret + asty.Lambda(expr, args=args, body=body.force_expr)
    node = asty.AsyncFunctionDef if is_async else asty.FunctionDef
    name = compiler.get_anon_var()
    ret += compile_function_node(compiler, expr, node, [], tp, name, args, returns, body, scope)
    return ret + Result(expr=ret.temp_variables[0])

@pattern_macro(['defn', 'defn/a'], [maybe(brackets(many(FORM))), maybe(type_params), maybe_annotated(SYM), lambda_list, many(FORM)])
def compile_function_def(compiler, expr, root, decorators, tp, name, params, body):
    if False:
        for i in range(10):
            print('nop')
    is_async = root == 'defn/a'
    (name, returns) = name
    node = asty.AsyncFunctionDef if is_async else asty.FunctionDef
    (decorators, ret, _) = compiler._compile_collect(decorators[0] if decorators else [])
    (args, ret2) = compile_lambda_list(compiler, params)
    ret += ret2
    name = mangle(compiler._nonconst(name))
    compiler.scope.define(name)
    with compiler.local_state(), compiler.scope.create(ScopeFn, args, is_async) as scope:
        body = compiler._compile_branch(body)
    return ret + compile_function_node(compiler, expr, node, decorators, tp, name, args, returns, body, scope)

def compile_function_node(compiler, expr, node, decorators, tp, name, args, returns, body, scope):
    if False:
        for i in range(10):
            print('nop')
    ret = Result()
    if body.expr:
        enode = asty.Expr if scope.is_async and scope.has_yield else asty.Return
        body += enode(body.expr, value=body.expr)
    ret += node(expr, name=name, args=args, body=body.stmts or [asty.Pass(expr)], decorator_list=decorators, returns=compiler.compile(returns).force_expr if returns is not None else None, **digest_type_params(compiler, tp))
    ast_name = asty.Name(expr, id=name, ctx=ast.Load())
    return ret + Result(temp_variables=[ast_name, ret.stmts[-1]])

@pattern_macro('defmacro', [SYM, brackets(maybe(many(argument) + sym('/')), many(argument), maybe(varargs('unpack-iterable', NASYM))), many(FORM)])
def compile_macro_def(compiler, expr, root, name, params, body):
    if False:
        for i in range(10):
            print('nop')

    def E(*x):
        if False:
            return 10
        return Expression(x)
    S = Symbol
    compiler.warn_on_core_shadow(name)
    fn_def = E(S('fn'), List(expr[2]), *body).replace(expr)
    if compiler.is_in_local_state():
        state = compiler.local_state_stack[-1]
        ret = compiler.compile(E(S('setv'), S(local_macro_name(name)), fn_def).replace(expr))
        state['macros'][mangle(name)] = compiler.eval(fn_def)
        return ret + ret.expr_as_stmt()
    ret = compiler.compile(E(S('eval-and-compile'), E(E(dotted('hy.macros.macro'), str(name)), fn_def)).replace(expr))
    return ret + ret.expr_as_stmt()

def compile_lambda_list(compiler, params):
    if False:
        print('Hello World!')
    ret = Result()
    (posonly_parms, args_parms, rest_parms, kwonly_parms, kwargs_parms) = params
    if not (posonly_parms or posonly_parms is None):
        raise compiler._syntax_error(params, "positional only delimiter '/' must have an argument")
    posonly_parms = posonly_parms or []
    is_positional_arg = lambda x: isinstance(x[0], Symbol)
    invalid_non_default = next((arg for arg in dropwhile(is_positional_arg, posonly_parms + args_parms) if is_positional_arg(arg)), None)
    if invalid_non_default:
        raise compiler._syntax_error(invalid_non_default[0], 'non-default argument follows default argument')
    (posonly_ast, posonly_defaults, ret) = compile_arguments_set(compiler, posonly_parms, ret)
    (args_ast, args_defaults, ret) = compile_arguments_set(compiler, args_parms, ret)
    (kwonly_ast, kwonly_defaults, ret) = compile_arguments_set(compiler, kwonly_parms, ret, True)
    rest_ast = kwargs_ast = None
    if rest_parms == Symbol('*'):
        if not kwonly_parms:
            raise compiler._syntax_error(rest_parms, 'named arguments must follow bare *')
        rest_ast = None
    elif rest_parms:
        ([rest_ast], _, ret) = compile_arguments_set(compiler, [rest_parms], ret)
    if kwargs_parms:
        ([kwargs_ast], _, ret) = compile_arguments_set(compiler, [kwargs_parms], ret)
    return (ast.arguments(args=args_ast, defaults=[*posonly_defaults, *args_defaults], vararg=rest_ast, posonlyargs=posonly_ast, kwonlyargs=kwonly_ast, kw_defaults=kwonly_defaults, kwarg=kwargs_ast), ret)

def compile_arguments_set(compiler, decls, ret, is_kwonly=False):
    if False:
        for i in range(10):
            print('nop')
    args_ast = []
    args_defaults = []
    for (decl, ann) in decls:
        default = None
        if isinstance(decl, List):
            (sym, default) = decl
        else:
            sym = decl
        if ann is not None:
            ret += compiler.compile(ann)
            ann_ast = ret.force_expr
        else:
            ann_ast = None
        if default is not None:
            ret += compiler.compile(default)
            args_defaults.append(ret.force_expr)
        elif not isinstance(decl, List) and is_kwonly:
            args_defaults.append(None)
        elif isinstance(decl, List):
            args_defaults.append(None)
        args_ast.append(asty.arg(sym, arg=mangle(compiler._nonconst(sym)), annotation=ann_ast))
    return (args_ast, args_defaults, ret)

@pattern_macro('return', [maybe(FORM)])
def compile_return(compiler, expr, root, arg):
    if False:
        return 10
    ret = Result()
    if arg is None:
        return asty.Return(expr, value=None)
    ret += compiler.compile(arg)
    return ret + asty.Return(expr, value=ret.force_expr)

@pattern_macro('yield', [maybe(FORM)])
def compile_yield_expression(compiler, expr, root, arg):
    if False:
        i = 10
        return i + 15
    if is_inside_function_scope(compiler.scope):
        nearest_python_scope(compiler.scope).has_yield = True
    ret = Result()
    if arg is not None:
        ret += compiler.compile(arg)
    return ret + asty.Yield(expr, value=ret.force_expr)

@pattern_macro(['yield-from', 'await'], [FORM])
def compile_yield_from_or_await_expression(compiler, expr, root, arg):
    if False:
        while True:
            i = 10
    if root == 'yield-from' and is_inside_function_scope(compiler.scope):
        nearest_python_scope(compiler.scope).has_yield = True
    ret = Result() + compiler.compile(arg)
    node = asty.YieldFrom if root == 'yield-from' else asty.Await
    return ret + node(expr, value=ret.force_expr)

@pattern_macro('defclass', [maybe(brackets(many(FORM))), maybe(type_params), SYM, maybe(brackets(many(FORM)) + maybe(STR) + many(FORM))])
def compile_class_expression(compiler, expr, root, decorators, tp, name, rest):
    if False:
        for i in range(10):
            print('nop')
    (base_list, docstring, body) = rest or ([[]], None, [])
    (decorators, ret, _) = compiler._compile_collect(decorators[0] if decorators else [])
    (bases_expr, ret2, keywords) = compiler._compile_collect(base_list[0], with_kwargs=True)
    ret += ret2
    bodyr = Result()
    if docstring is not None:
        bodyr += compiler.compile(docstring).expr_as_stmt()
    name = mangle(compiler._nonconst(name))
    compiler.scope.define(name)
    with compiler.local_state(), compiler.scope.create(ScopeFn):
        e = compiler._compile_branch(body)
        bodyr += e + e.expr_as_stmt()
    return ret + asty.ClassDef(expr, decorator_list=decorators, name=name, keywords=keywords, starargs=None, kwargs=None, bases=bases_expr, body=bodyr.stmts or [asty.Pass(expr)], **digest_type_params(compiler, tp))
module_name_pattern = SYM | pexpr(some(lambda x: isinstance(x, Symbol) and (not str(x[0]).strip('.'))) + oneplus(SYM))

def module_name_str(x):
    if False:
        return 10
    return '.'.join(map(mangle, x[1][x[1][0] == Symbol('None'):])) if isinstance(x, Expression) else str(x) if isinstance(x, Symbol) and (not x.strip('.')) else mangle(x)
importlike = keepsym('*') | keepsym(':as') + SYM | brackets(many(SYM + maybe(sym(':as') + SYM)))

def assignment_shape(module, rest):
    if False:
        for i in range(10):
            print('nop')
    prefix = ''
    assignments = 'EXPORTS'
    if rest is None:
        prefix = module_name_str(module)
    elif rest == Symbol('*'):
        pass
    elif rest[0] == Keyword('as'):
        prefix = mangle(rest[1])
    else:
        assignments = [(k, v or k) for (k, v) in rest[0]]
    return (prefix, assignments)

@pattern_macro('require', [many(module_name_pattern + times(0, 2, maybe(sym(':macros')) + importlike | keepsym(':readers') + (keepsym('*') | brackets(many(SYM)))))])
def compile_require(compiler, expr, root, entries):
    if False:
        i = 10
        return i + 15
    ret = Result()
    for entry in entries:
        (module, assignments) = entry
        (readers, rest) = ([names for (key, names) in assignments if (key == Keyword('readers')) == flag] for flag in (True, False))
        if len(rest) > 1 or len(readers) > 1:
            raise compiler._syntax_error(entry, f"redefinition of ':{('macros' if len(rest) > 1 else 'readers')}' brackets.")
        rest = rest[0] if rest else None
        readers = readers and readers[0]
        (prefix, assignments) = assignment_shape(module, rest)
        module_name = module_name_str(module)
        if isinstance(module, Expression) and module[1][0] == Symbol('None'):
            module_name = str(module[0]) + module_name
        if (rest or not readers) and compiler.is_in_local_state():
            reqs = require(module_name, compiler.local_state_stack[-1]['macros'], assignments=assignments, prefix=prefix, compiler=compiler)
            ret += compiler.compile(Expression([Symbol('setv'), List([Symbol(local_macro_name(m)) for (m, _, _) in reqs]), Expression([dotted('hy.macros.require_vals'), String(module_name), Dict(), Keyword('assignments'), List([(String(m), String(m)) for (_, m, _) in reqs])])]).replace(expr))
            ret += ret.expr_as_stmt()
        elif (rest or not readers) and require(module_name, compiler.module, assignments=assignments, prefix=prefix, compiler=compiler):
            ret += compiler.compile(Expression([dotted('hy.macros.require'), String(module_name), Symbol('None'), Keyword('target_module_name'), String(compiler.module.__name__), Keyword('assignments'), String('EXPORTS') if assignments == 'EXPORTS' else List([List([String(k), String(v)]) for (k, v) in assignments]), Keyword('prefix'), String(prefix)]).replace(expr))
            ret += ret.expr_as_stmt()
        if readers:
            reader_assignments = 'ALL' if readers == Symbol('*') else [str(reader) for reader in readers[0]]
            if require_reader(module_name, compiler.module, reader_assignments):
                ret += compiler.compile(mkexpr('do', mkexpr(dotted('hy.macros.require-reader'), String(module_name), 'None', [reader_assignments]), mkexpr('eval-when-compile', mkexpr(dotted('hy.macros.enable-readers'), 'None', mkexpr(dotted('hy.reader.HyReader.current-reader')), [reader_assignments]))).replace(expr))
                ret += ret.expr_as_stmt()
    return ret

@pattern_macro('import', [many(module_name_pattern + maybe(importlike))])
def compile_import(compiler, expr, root, entries):
    if False:
        for i in range(10):
            print('nop')
    ret = Result()
    for entry in entries:
        (module, _) = entry
        (prefix, assignments) = assignment_shape(*entry)
        module_name = module_name_str(module)
        if assignments == 'EXPORTS' and prefix == '':
            node = asty.ImportFrom
            names = [asty.alias(module, name='*', asname=None)]
        elif assignments == 'EXPORTS':
            compiler.scope.define(prefix)
            node = asty.Import
            names = [asty.alias(module, name=module_name, asname=prefix if prefix != module_name else None)]
        else:
            node = asty.ImportFrom
            names = []
            for (k, v) in assignments:
                compiler.scope.define(mangle(v))
                names.append(asty.alias(module, name=mangle(k), asname=None if v == k else mangle(v)))
        ret += node(expr, module=module_name if module_name and module_name.strip('.') else None, names=names, level=len(module[0]) if isinstance(module, Expression) and module[1][0] == Symbol('None') else len(module) if isinstance(module, Symbol) and (not module.strip('.')) else 0)
    return ret

@pattern_macro('assert', [FORM, maybe(FORM)])
def compile_assert_expression(compiler, expr, root, test, msg):
    if False:
        i = 10
        return i + 15
    test = compiler.compile(test)
    if msg:
        msg = compiler.compile(msg)
    if not (test.stmts or (msg and msg.stmts)):
        return asty.Assert(expr, test=test.force_expr, msg=msg and msg.force_expr)
    return asty.If(expr, test=asty.Name(expr, id='__debug__', ctx=ast.Load()), orelse=[], body=test.stmts + [asty.If(test, test=asty.UnaryOp(test, op=ast.Not(), operand=test.force_expr), orelse=[], body=(msg.stmts if msg else []) + [asty.Assert(expr, test=asty.Constant(test, value=False), msg=msg and msg.force_expr)])])

@pattern_macro('let', [brackets(many(maybe_annotated(FORM) + FORM)), many(FORM)])
def compile_let(compiler, expr, root, bindings, body):
    if False:
        for i in range(10):
            print('nop')
    res = Result()
    bindings = bindings[0]
    scope = compiler.scope.create(ScopeLet)
    for ((target, ann), value) in bindings:
        res += compile_assign(compiler, ann, target, value, let_scope=scope)
    with scope:
        return res + compiler.compile(mkexpr('do', *body).replace(expr))

@pattern_macro(((3, 12), 'deftype'), [maybe(type_params), SYM, FORM])
def compile_deftype(compiler, expr, root, tp, name, value):
    if False:
        print('Hello World!')
    return asty.TypeAlias(expr, name=asty.Name(name, id=mangle(name), ctx=ast.Store()), value=compiler.compile(value).force_expr, **digest_type_params(compiler, tp))

@pattern_macro('pragma', [many(KEYWORD + FORM)])
def compile_pragma(compiler, expr, root, kwargs):
    if False:
        while True:
            i = 10
    for (kw, value) in kwargs:
        if kw == Keyword('warn-on-core-shadow'):
            compiler.local_state_stack[-1]['warn_on_core_shadow'] = bool(compiler.eval(value))
        else:
            raise compiler._syntax_error(kw, f"Unknown pragma `{kw}`. Perhaps it's implemented by a newer version of Hy.")
    return Result()

@pattern_macro('unquote unquote-splice unpack-mapping except except* finally else'.split(), [many(FORM)])
def compile_placeholder(compiler, expr, root, body):
    if False:
        i = 10
        return i + 15
    raise ValueError(f'`{root}` is not allowed here')