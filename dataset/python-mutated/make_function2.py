"""
All the crazy things we have to do to handle Python functions in Python before 3.0.
The saga of changes continues in 3.0 and above and in other files.
"""
from uncompyle6.scanner import Code
from uncompyle6.semantics.parser_error import ParserError
from uncompyle6.parser import ParserError as ParserError2
from uncompyle6.semantics.helper import print_docstring, find_all_globals, find_globals_and_nonlocals, find_none
from xdis import iscode, code_has_star_arg, code_has_star_star_arg
from itertools import zip_longest
from uncompyle6.show import maybe_show_tree_param_default

def make_function2(self, node, is_lambda, nested=1, code_node=None):
    if False:
        print('Hello World!')
    '\n    Dump function defintion, doc string, and function body.\n    This code is specialied for Python 2.\n    '

    def build_param(ast, name, default):
        if False:
            while True:
                i = 10
        'build parameters:\n            - handle defaults\n            - handle format tuple parameters\n        '
        if name.startswith('.'):
            name = self.get_tuple_parameter(ast, name)
            pass
        if default:
            value = self.traverse(default, indent='')
            maybe_show_tree_param_default(self.showast, name, value)
            result = '%s=%s' % (name, value)
            if result[-2:] == '= ':
                result += 'None'
            return result
        else:
            return name
    assert node[-1].kind.startswith('MAKE_')
    args_node = node[-1]
    if isinstance(args_node.attr, tuple):
        defparams = node[1:args_node.attr[0] + 1]
        (pos_args, kw_args, annotate_argc) = args_node.attr
    else:
        defparams = node[:args_node.attr]
        kw_args = 0
        pass
    lambda_index = None
    if lambda_index and is_lambda and iscode(node[lambda_index].attr):
        assert node[lambda_index].kind == 'LOAD_LAMBDA'
        code = node[lambda_index].attr
    else:
        code = code_node.attr
    assert iscode(code)
    code = Code(code, self.scanner, self.currentclass)
    argc = code.co_argcount
    paramnames = list(code.co_varnames[:argc])
    paramnames.reverse()
    defparams.reverse()
    try:
        ast = self.build_ast(code._tokens, code._customize, code, is_lambda=is_lambda, noneInNames='None' in code.co_names)
    except (ParserError, ParserError2) as p:
        self.write(str(p))
        if not self.tolerate_errors:
            self.ERROR = p
        return
    kw_pairs = 0
    indent = self.indent
    params = [build_param(ast, name, default) for (name, default) in zip_longest(paramnames, defparams, fillvalue=None)]
    params.reverse()
    if code_has_star_arg(code):
        params.append('*%s' % code.co_varnames[argc])
        argc += 1
    if is_lambda:
        self.write('lambda ', ', '.join(params))
        if len(ast) > 1 and self.traverse(ast[-1]) == 'None' and self.traverse(ast[-2]).strip().startswith('yield'):
            del ast[-1]
            ast_expr = ast[-1]
            while ast_expr.kind != 'expr':
                ast_expr = ast_expr[0]
            ast[-1] = ast_expr
            pass
    else:
        self.write('(', ', '.join(params))
    if kw_args > 0:
        if not 4 & code.co_flags:
            if argc > 0:
                self.write(', *, ')
            else:
                self.write('*, ')
            pass
        else:
            self.write(', ')
        for n in node:
            if n == 'pos_arg':
                continue
            else:
                self.preorder(n)
            break
        pass
    if code_has_star_star_arg(code):
        if argc > 0:
            self.write(', ')
        self.write('**%s' % code.co_varnames[argc + kw_pairs])
    if is_lambda:
        self.write(': ')
    else:
        self.println('):')
    if len(code.co_consts) > 0 and code.co_consts[0] is not None and (not is_lambda):
        print_docstring(self, indent, code.co_consts[0])
    if not is_lambda:
        assert ast == 'stmts'
    all_globals = find_all_globals(ast, set())
    (globals, nonlocals) = find_globals_and_nonlocals(ast, set(), set(), code, self.version)
    assert self.version >= (3, 0) or not nonlocals
    for g in sorted(all_globals & self.mod_globs | globals):
        self.println(self.indent, 'global ', g)
    self.mod_globs -= all_globals
    has_none = 'None' in code.co_names
    rn = has_none and (not find_none(ast))
    self.gen_source(ast, code.co_name, code._customize, is_lambda=is_lambda, returnNone=rn)
    code._tokens = None
    code._customize = None