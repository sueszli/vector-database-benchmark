from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse24 import Python24Parser

class Python23Parser(Python24Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            while True:
                i = 10
        super(Python23Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_misc23(self, args):
        if False:
            return 10
        '\n        # Python 2.4 only adds something like the below for if 1:\n        # However we will just treat it as a noop (which of course messes up\n        # simple verify of bytecode.\n        # See also below in reduce_is_invalid where we check that the JUMP_FORWARD\n        # target matches the COME_FROM target\n        stmt     ::= if1_stmt\n        if1_stmt ::= JUMP_FORWARD JUMP_IF_FALSE THEN POP_TOP COME_FROM\n                     stmts\n                     JUMP_FORWARD COME_FROM POP_TOP COME_FROM\n\n\n        # Used to keep semantic positions the same across later versions\n        # of Python\n        _while1test ::= SETUP_LOOP JUMP_FORWARD JUMP_IF_FALSE POP_TOP COME_FROM\n\n        while1stmt ::= _while1test l_stmts_opt JUMP_BACK\n                       POP_TOP POP_BLOCK COME_FROM\n\n        while1stmt ::= _while1test l_stmts_opt JUMP_BACK COME_FROM\n                       POP_TOP POP_BLOCK COME_FROM\n\n        # Python 2.3\n        # The following has no "JUMP_BACK" after l_stmts because\n        # l_stmts ends in a "break", "return", or "continue"\n        while1stmt ::= _while1test l_stmts\n                       POP_TOP POP_BLOCK\n\n        # The following has a "COME_FROM" at the end which comes from\n        # a "break" inside "l_stmts".\n        while1stmt ::= _while1test l_stmts COME_FROM JUMP_BACK\n                       POP_TOP POP_BLOCK COME_FROM\n        while1stmt ::= _while1test l_stmts JUMP_BACK\n                       POP_TOP POP_BLOCK\n\n        list_comp  ::= BUILD_LIST_0 DUP_TOP LOAD_ATTR store list_iter delete\n        list_for   ::= expr for_iter store list_iter JUMP_BACK come_froms POP_TOP JUMP_BACK\n\n        lc_body ::= LOAD_NAME expr CALL_FUNCTION_1 POP_TOP\n        lc_body ::= LOAD_FAST expr CALL_FUNCTION_1 POP_TOP\n        lc_body ::= LOAD_NAME expr LIST_APPEND\n        lc_body ::= LOAD_FAST expr LIST_APPEND\n\n        # "and" where the first part of the and is true,\n        # so there is only the 2nd part to evaluate\n        expr ::= and2\n        and2 ::= _jump jmp_false COME_FROM expr COME_FROM\n\n        alias       ::= IMPORT_NAME attributes store\n        if_exp      ::= expr jmp_false expr JUMP_FORWARD expr COME_FROM\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            print('Hello World!')
        super(Python23Parser, self).customize_grammar_rules(tokens, customize)

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            while True:
                i = 10
        invalid = super(Python24Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        if invalid:
            return invalid
        lhs = rule[0]
        if lhs == 'nop_stmt':
            return not int(tokens[first].pattr) == tokens[last].offset
        return False

class Python23ParserSingle(Python23Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python23Parser()
    p.check_grammar()
    p.dump_grammar()