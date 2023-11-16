"""
spark grammar differences over Python 3.2 for Python 3.3.
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse32 import Python32Parser

class Python33Parser(Python32Parser):

    def p_33on(self, args):
        if False:
            print('Hello World!')
        '\n        # Python 3.3+ adds yield from.\n        expr          ::= yield_from\n        yield_from    ::= expr expr YIELD_FROM\n        stmt         ::= genexpr_func\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            print('Hello World!')
        self.remove_rules('\n        # 3.3+ adds POP_BLOCKS\n        whileTruestmt ::= SETUP_LOOP l_stmts_opt JUMP_BACK POP_BLOCK NOP COME_FROM_LOOP\n        whileTruestmt ::= SETUP_LOOP l_stmts_opt JUMP_BACK NOP COME_FROM_LOOP\n        ')
        super(Python33Parser, self).customize_grammar_rules(tokens, customize)
        return

class Python33ParserSingle(Python33Parser, PythonParserSingle):
    pass