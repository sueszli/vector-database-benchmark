"""Implements the xonsh parser for Python v3.6."""
import xonsh.ast as ast
from xonsh.parsers.base import BaseParser, lopen_loc, store_ctx

class Parser(BaseParser):
    """A Python v3.6 compliant parser for the xonsh language."""

    def __init__(self, yacc_optimize=True, yacc_table='xonsh.parser_table', yacc_debug=False, outputdir=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        yacc_optimize : bool, optional\n            Set to false when unstable and true when parser is stable.\n        yacc_table : str, optional\n            Parser module used when optimized.\n        yacc_debug : debug, optional\n            Dumps extra debug info.\n        outputdir : str or None, optional\n            The directory to place generated tables within.\n        '
        tok_rules = ['await', 'async']
        for rule in tok_rules:
            self._tok_rule(rule)
        super().__init__(yacc_optimize=yacc_optimize, yacc_table=yacc_table, yacc_debug=yacc_debug, outputdir=outputdir)

    def p_classdef_or_funcdef(self, p):
        if False:
            print('Hello World!')
        '\n        classdef_or_funcdef : classdef\n                            | funcdef\n                            | async_funcdef\n        '
        p[0] = p[1]

    def p_async_funcdef(self, p):
        if False:
            i = 10
            return i + 15
        'async_funcdef : async_tok funcdef'
        (p1, f) = (p[1], p[2][0])
        p[0] = [ast.AsyncFunctionDef(**f.__dict__)]
        p[0][0]._async_tok = p1

    def p_async_compound_stmt(self, p):
        if False:
            for i in range(10):
                print('nop')
        'compound_stmt : async_stmt'
        p[0] = p[1]

    def p_async_for_stmt(self, p):
        if False:
            while True:
                i = 10
        'async_for_stmt : ASYNC for_stmt'
        f = p[2][0]
        p[0] = [ast.AsyncFor(**f.__dict__)]

    def p_async_with_stmt(self, p):
        if False:
            while True:
                i = 10
        'async_with_stmt : ASYNC with_stmt'
        w = p[2][0]
        p[0] = [ast.AsyncWith(**w.__dict__)]

    def p_atom_expr_await(self, p):
        if False:
            i = 10
            return i + 15
        'atom_expr : await_tok atom trailer_list_opt'
        p0 = self.apply_trailers(p[2], p[3])
        p1 = p[1]
        p0 = ast.Await(value=p0, ctx=ast.Load(), lineno=p1.lineno, col_offset=p1.lexpos)
        p[0] = p0

    def p_async_stmt(self, p):
        if False:
            i = 10
            return i + 15
        '\n        async_stmt : async_funcdef\n                   | async_with_stmt\n                   | async_for_stmt\n        '
        p[0] = p[1]

    def p_item_test(self, p):
        if False:
            for i in range(10):
                print('nop')
        'item : test COLON test'
        p[0] = [p[1], p[3]]

    def p_item_pow(self, p):
        if False:
            while True:
                i = 10
        'item : POW expr'
        p[0] = [None, p[2]]

    def _set_arg(self, args, arg, ensure_kw=False):
        if False:
            print('Hello World!')
        if isinstance(arg, ast.keyword):
            args['keywords'].append(arg)
        elif ensure_kw:
            args['keywords'].append(ast.keyword(arg=None, value=arg))
        else:
            args['args'].append(arg)

    def p_arglist_single(self, p):
        if False:
            while True:
                i = 10
        'arglist : argument comma_opt'
        p0 = {'args': [], 'keywords': []}
        self._set_arg(p0, p[1])
        p[0] = p0

    def p_arglist_many(self, p):
        if False:
            i = 10
            return i + 15
        'arglist : argument comma_argument_list comma_opt'
        p0 = {'args': [], 'keywords': []}
        self._set_arg(p0, p[1])
        for arg in p[2]:
            self._set_arg(p0, arg)
        p[0] = p0

    def p_argument_test_or_star(self, p):
        if False:
            return 10
        'argument : test_or_star_expr'
        p[0] = p[1]

    def p_argument_kwargs(self, p):
        if False:
            return 10
        'argument : POW test'
        p2 = p[2]
        p[0] = ast.keyword(arg=None, value=p2, lineno=p2.lineno, col_offset=p2.col_offset)

    def p_argument_args(self, p):
        if False:
            return 10
        'argument : TIMES test'
        p[0] = ast.Starred(value=p[2])

    def p_argument(self, p):
        if False:
            for i in range(10):
                print('nop')
        'argument : test comp_for'
        p1 = p[1]
        p[0] = ast.GeneratorExp(elt=p1, generators=p[2]['comps'], lineno=p1.lineno, col_offset=p1.col_offset)

    def p_argument_eq(self, p):
        if False:
            print('Hello World!')
        'argument : test EQUALS test'
        p3 = p[3]
        p[0] = ast.keyword(arg=p[1].id, value=p3, lineno=p3.lineno, col_offset=p3.col_offset)

    def p_comp_for(self, p):
        if False:
            return 10
        'comp_for : FOR exprlist IN or_test comp_iter_opt'
        super().p_comp_for(p)
        p[0]['comps'][0].is_async = 0

    def p_expr_stmt_annassign(self, p):
        if False:
            while True:
                i = 10
        'expr_stmt : testlist_star_expr COLON test EQUALS test\n        | testlist_star_expr COLON test\n        '
        p1 = p[1][0]
        (lineno, col) = lopen_loc(p1)
        if len(p[1]) > 1 or not isinstance(p1, (ast.Name, ast.Attribute, ast.Subscript)):
            loc = self.currloc(lineno, col)
            self._set_error('only single target can be annotated', loc)
        store_ctx(p1)
        p[0] = ast.AnnAssign(target=p1, annotation=p[3], value=p[5] if len(p) >= 6 else None, simple=1, lineno=lineno, col_offset=col)