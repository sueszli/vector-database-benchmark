"""
spark grammar differences over Python2.5 for Python 2.4.
"""
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse25 import Python25Parser

class Python24Parser(Python25Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            while True:
                i = 10
        super(Python24Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_misc24(self, args):
        if False:
            return 10
        '\n        # Python 2.4 only adds something like the below for if 1:\n        # However we will just treat it as a noop which messes up\n        # simple verify of bytecode.\n        # See also below in reduce_is_invalid where we check that the JUMP_FORWARD\n        # target matches the COME_FROM target\n        stmt     ::= nop_stmt\n        nop_stmt ::= JUMP_FORWARD POP_TOP COME_FROM\n\n        # 2.5+ has two LOAD_CONSTs, one for the number \'.\'s in a relative import\n        # keep positions similar to simplify semantic actions\n\n        import           ::= filler LOAD_CONST alias\n        import_from      ::= filler LOAD_CONST IMPORT_NAME importlist POP_TOP\n        import_from_star ::= filler LOAD_CONST IMPORT_NAME IMPORT_STAR\n\n        importmultiple ::= filler LOAD_CONST alias imports_cont\n        import_cont    ::= filler LOAD_CONST alias\n\n        # Handle "if true else: ..." in Python 2.4\n        stmt            ::= iftrue_stmt24\n        iftrue_stmt24   ::= _ifstmts_jump24 suite_stmts COME_FROM\n        _ifstmts_jump24 ::= c_stmts_opt JUMP_FORWARD POP_TOP\n\n        # Python 2.5+ omits POP_TOP POP_BLOCK\n        while1stmt ::= SETUP_LOOP l_stmts_opt JUMP_BACK\n                       POP_TOP POP_BLOCK COME_FROM\n        while1stmt ::= SETUP_LOOP l_stmts_opt JUMP_BACK\n                       POP_TOP POP_BLOCK\n\n        continue   ::= JUMP_BACK JUMP_ABSOLUTE\n\n        # Python 2.4\n        # The following has no "JUMP_BACK" after l_stmts because\n        # l_stmts ends in a "break", "return", or "continue"\n        while1stmt ::= SETUP_LOOP l_stmts\n                       POP_TOP POP_BLOCK\n\n        # The following has a "COME_FROM" at the end which comes from\n        # a "break" inside "l_stmts".\n        while1stmt ::= SETUP_LOOP l_stmts COME_FROM JUMP_BACK\n                       POP_TOP POP_BLOCK COME_FROM\n\n        # Python 2.5+:\n        #  call_stmt ::= expr POP_TOP\n        #  expr      ::= yield\n        call_stmt ::= yield\n\n        # Python 2.5+ adds POP_TOP at the end\n        gen_comp_body ::= expr YIELD_VALUE\n\n        # Python 2.4\n        # Python 2.6, 2.7 and 3.3+ use kv3\n        # Python 2.3- use kv\n        kvlist ::= kvlist kv2\n        kv2    ::= DUP_TOP expr expr ROT_THREE STORE_SUBSCR\n        '

    def remove_rules_24(self):
        if False:
            for i in range(10):
                print('nop')
        self.remove_rules('\n        expr ::= if_exp\n        ')

    def customize_grammar_rules(self, tokens, customize):
        if False:
            i = 10
            return i + 15
        self.remove_rules('\n        gen_comp_body ::= expr YIELD_VALUE POP_TOP\n        kvlist        ::= kvlist kv3\n        while1stmt    ::= SETUP_LOOP l_stmts JUMP_BACK COME_FROM\n        while1stmt    ::= SETUP_LOOP l_stmts_opt JUMP_BACK COME_FROM\n        while1stmt    ::= SETUP_LOOP returns COME_FROM\n        whilestmt     ::= SETUP_LOOP testexpr returns POP_BLOCK COME_FROM\n        with_cleanup  ::= LOAD_FAST DELETE_FAST WITH_CLEANUP END_FINALLY\n        with_cleanup  ::= LOAD_NAME DELETE_NAME WITH_CLEANUP END_FINALLY\n        withasstmt    ::= expr setupwithas store suite_stmts_opt POP_BLOCK LOAD_CONST COME_FROM with_cleanup\n        with          ::= expr setupwith SETUP_FINALLY suite_stmts_opt POP_BLOCK LOAD_CONST COME_FROM with_cleanup\n        stmt ::= with\n        stmt ::= withasstmt\n        ')
        super(Python24Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules_24()
        if self.version[:2] == (2, 4):
            self.check_reduce['nop_stmt'] = 'tokens'
        if self.version[:2] <= (2, 4):
            del self.reduce_check_table['ifelsestmt']

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            print('Hello World!')
        invalid = super(Python24Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        if invalid or tokens is None:
            return invalid
        lhs = rule[0]
        if lhs == 'nop_stmt':
            token_len = len(tokens)
            if 0 <= token_len < len(tokens):
                return not int(tokens[first].pattr) == tokens[last].offset
        elif lhs == 'try_except':
            if last == len(tokens):
                last -= 1
            if tokens[last] != 'COME_FROM' and tokens[last - 1] == 'COME_FROM':
                last -= 1
            return tokens[last] == 'COME_FROM' and tokens[last - 1] == 'END_FINALLY' and (tokens[last - 2] == 'POP_TOP') and (tokens[last - 3].kind != 'JUMP_FORWARD')
        return False

class Python24ParserSingle(Python24Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python24Parser()
    p.check_grammar()