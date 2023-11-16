"""
All the crazy things we have to do to handle Python functions in Python before 3.0.
The saga of changes continues in 3.0 and above and in other files.
"""
from typing import List, Tuple
from uncompyle6.scanner import Code
from uncompyle6.semantics.parser_error import ParserError
from uncompyle6.parser import ParserError as ParserError2
from uncompyle6.semantics.helper import print_docstring, find_all_globals, find_globals_and_nonlocals, find_none
from xdis import iscode

def make_function1(self, node, is_lambda, nested=1, code_node=None):
    if False:
        i = 10
        return i + 15
    '\n    Dump function defintion, doc string, and function body.\n    This code is specialied for Python 2.\n    '

    def build_param(tree, param_names: List[str]) -> Tuple[bool, List[str]]:
        if False:
            i = 10
            return i + 15
        'build parameters:\n            - handle defaults\n            - handle format tuple parameters\n        '
        args = tree[0]
        del tree[0]
        params = []
        assert args.kind in ('star_args', 'args', 'varargs')
        has_star_arg = args.kind in ('star_args', 'varargs')
        args_store = args[2]
        if args_store == 'args_store':
            for arg in args_store:
                params.append(param_names[arg.attr])
        return (has_star_arg, params)
    assert node[-1].kind.startswith('BUILD_')
    defparams = []
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
        tree = self.build_ast(code._tokens, code._customize, code, is_lambda=is_lambda, noneInNames='None' in code.co_names)
    except (ParserError, ParserError2) as p:
        self.write(str(p))
        if not self.tolerate_errors:
            self.ERROR = p
        return
    indent = self.indent
    (has_star_arg, params) = build_param(tree, code.co_names)
    if has_star_arg:
        params[-1] = '*' + params[-1]
    if is_lambda:
        self.write('lambda ', ', '.join(params))
        if len(tree) > 1 and self.traverse(tree[-1]) == 'None' and self.traverse(tree[-2]).strip().startswith('yield'):
            del tree[-1]
            tree_expr = tree[-1]
            while tree_expr.kind != 'expr':
                tree_expr = tree_expr[0]
            tree[-1] = tree_expr
            pass
    else:
        self.write('(', ', '.join(params))
    if is_lambda:
        self.write(': ')
    else:
        self.println('):')
    if len(code.co_consts) > 0 and code.co_consts[0] is not None and (not is_lambda):
        print_docstring(self, indent, code.co_consts[0])
    if not is_lambda:
        assert tree == 'stmts'
    all_globals = find_all_globals(tree, set())
    (globals, nonlocals) = find_globals_and_nonlocals(tree, set(), set(), code, self.version)
    for g in sorted(all_globals & self.mod_globs | globals):
        self.println(self.indent, 'global ', g)
    self.mod_globs -= all_globals
    has_none = 'None' in code.co_names
    rn = has_none and (not find_none(tree))
    tree.code = code
    self.gen_source(tree, code.co_name, code._customize, is_lambda=is_lambda, returnNone=rn)
    code._tokens = None
    code._customize = None