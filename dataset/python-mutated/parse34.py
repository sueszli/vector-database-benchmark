"""
spark grammar differences over Python 3.3 for Python 3.4
"""
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse33 import Python33Parser

class Python34Parser(Python33Parser):

    def p_misc34(self, args):
        if False:
            return 10
        '\n        expr ::= LOAD_ASSERT\n\n\n        # passtmt is needed for semantic actions to add "pass"\n        suite_stmts_opt ::= pass\n\n        whilestmt     ::= SETUP_LOOP testexpr returns come_froms POP_BLOCK COME_FROM_LOOP\n\n        # Seems to be needed starting 3.4.4 or so\n        while1stmt    ::= SETUP_LOOP l_stmts\n                          COME_FROM JUMP_BACK POP_BLOCK COME_FROM_LOOP\n        while1stmt    ::= SETUP_LOOP l_stmts\n                          POP_BLOCK COME_FROM_LOOP\n\n        # FIXME the below masks a bug in not detecting COME_FROM_LOOP\n        # grammar rules with COME_FROM -> COME_FROM_LOOP already exist\n        whileelsestmt     ::= SETUP_LOOP testexpr l_stmts_opt JUMP_BACK POP_BLOCK\n                              else_suitel COME_FROM\n\n        while1elsestmt    ::= SETUP_LOOP l_stmts JUMP_BACK _come_froms POP_BLOCK else_suitel\n                              COME_FROM_LOOP\n\n        # Python 3.4+ optimizes the trailing two JUMPS away\n\n        # This is 3.4 only\n        yield_from ::= expr GET_ITER LOAD_CONST YIELD_FROM\n\n        _ifstmts_jump ::= c_stmts_opt JUMP_ABSOLUTE JUMP_FORWARD COME_FROM\n\n        genexpr_func ::= LOAD_ARG _come_froms FOR_ITER store comp_iter JUMP_BACK\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            for i in range(10):
                print('nop')
        self.remove_rules('\n        yield_from    ::= expr expr YIELD_FROM\n        # 3.4.2 has this. 3.4.4 may now\n        # while1stmt ::= SETUP_LOOP l_stmts COME_FROM JUMP_BACK COME_FROM_LOOP\n        ')
        super(Python34Parser, self).customize_grammar_rules(tokens, customize)
        return

class Python34ParserSingle(Python34Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python34Parser()
    p.check_grammar()
    from xdis.version_info import IS_PYPY, PYTHON_VERSION_TRIPLE
    if PYTHON_VERSION_TRIPLE[:2] == (3, 4):
        (lhs, rhs, tokens, right_recursive, dup_rhs) = p.check_sets()
        from uncompyle6.scanner import get_scanner
        s = get_scanner(PYTHON_VERSION_TRIPLE, IS_PYPY)
        opcode_set = set(s.opc.opname).union(set('JUMP_BACK CONTINUE RETURN_END_IF COME_FROM\n               LOAD_GENEXPR LOAD_ASSERT LOAD_SETCOMP LOAD_DICTCOMP LOAD_CLASSNAME\n               LAMBDA_MARKER RETURN_LAST\n            '.split()))
        remain_tokens = set(tokens) - opcode_set
        import re
        remain_tokens = set([re.sub('_\\d+$', '', t) for t in remain_tokens])
        remain_tokens = set([re.sub('_CONT$', '', t) for t in remain_tokens])
        remain_tokens = set(remain_tokens) - opcode_set
        print(remain_tokens)