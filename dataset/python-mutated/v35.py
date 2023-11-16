"""Implements the xonsh parser for Python v3.5."""
import xonsh.ast as ast
from xonsh.parsers.base import BaseParser

class Parser(BaseParser):
    """A Python v3.5 compliant parser for the xonsh language."""

    def __init__(self, lexer_optimize=True, lexer_table='xonsh.lexer_table', yacc_optimize=True, yacc_table='xonsh.parser_table', yacc_debug=False, outputdir=None):
        if False:
            for i in range(10):
                print('nop')
        'Parameters\n        ----------\n        lexer_optimize : bool, optional\n            Set to false when unstable and true when lexer is stable.\n        lexer_table : str, optional\n            Lexer module used when optimized.\n        yacc_optimize : bool, optional\n            Set to false when unstable and true when parser is stable.\n        yacc_table : str, optional\n            Parser module used when optimized.\n        yacc_debug : debug, optional\n            Dumps extra debug info.\n        outputdir : str or None, optional\n            The directory to place generated tables within.\n        '
        tok_rules = ['await', 'async']
        for rule in tok_rules:
            self._tok_rule(rule)
        super().__init__(lexer_optimize=lexer_optimize, lexer_table=lexer_table, yacc_optimize=yacc_optimize, yacc_table=yacc_table, yacc_debug=yacc_debug, outputdir=outputdir)

    def p_classdef_or_funcdef(self, p):
        if False:
            while True:
                i = 10
        'classdef_or_funcdef : classdef\n                               | funcdef\n                               | async_funcdef\n        '
        p[0] = p[1]

    def p_async_funcdef(self, p):
        if False:
            print('Hello World!')
        'async_funcdef : async_tok funcdef'
        (p1, f) = (p[1], p[2][0])
        p[0] = [ast.AsyncFunctionDef(**f.__dict__)]
        p[0][0]._async_tok = p1

    def p_async_compound_stmt(self, p):
        if False:
            print('Hello World!')
        'compound_stmt : async_stmt'
        p[0] = p[1]

    def p_async_for_stmt(self, p):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        'atom_expr : await_tok atom trailer_list_opt'
        p0 = self.apply_trailers(p[2], p[3])
        p1 = p[1]
        p0 = ast.Await(value=p0, ctx=ast.Load(), lineno=p1.lineno, col_offset=p1.lexpos)
        p[0] = p0

    def p_async_stmt(self, p):
        if False:
            for i in range(10):
                print('nop')
        'async_stmt : async_funcdef\n                      | async_with_stmt\n                      | async_for_stmt\n        '
        p[0] = p[1]

    def p_item_test(self, p):
        if False:
            while True:
                i = 10
        'item : test COLON test'
        p[0] = [p[1], p[3]]

    def p_item_pow(self, p):
        if False:
            return 10
        'item : POW expr'
        p[0] = [None, p[2]]

    def _set_arg(self, args, arg, ensure_kw=False):
        if False:
            i = 10
            return i + 15
        if isinstance(arg, ast.keyword):
            args['keywords'].append(arg)
        elif ensure_kw:
            args['keywords'].append(ast.keyword(arg=None, value=arg))
        else:
            args['args'].append(arg)

    def p_arglist_single(self, p):
        if False:
            return 10
        'arglist : argument comma_opt'
        p0 = {'args': [], 'keywords': []}
        self._set_arg(p0, p[1])
        p[0] = p0

    def p_arglist_many(self, p):
        if False:
            i = 10
            return i + 15
        'arglist : argument comma_argument_list comma_opt\n        '
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
            for i in range(10):
                print('nop')
        'argument : POW test'
        p[0] = ast.keyword(arg=None, value=p[2])

    def p_argument_args(self, p):
        if False:
            return 10
        'argument : TIMES test'
        p[0] = ast.Starred(value=p[2])

    def p_argument(self, p):
        if False:
            print('Hello World!')
        'argument : test comp_for'
        p1 = p[1]
        p[0] = ast.GeneratorExp(elt=p1, generators=p[2]['comps'], lineno=p1.lineno, col_offset=p1.col_offset)

    def p_argument_eq(self, p):
        if False:
            print('Hello World!')
        'argument : test EQUALS test'
        p[0] = ast.keyword(arg=p[1].id, value=p[3])