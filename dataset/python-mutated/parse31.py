"""
spark grammar differences over Python 3.2 for Python 3.1.
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse32 import Python32Parser

class Python31Parser(Python32Parser):

    def p_31(self, args):
        if False:
            return 10
        '\n        subscript2     ::= expr expr DUP_TOPX BINARY_SUBSCR\n\n        setupwith      ::= DUP_TOP LOAD_ATTR store LOAD_ATTR CALL_FUNCTION_0 POP_TOP\n        setupwithas    ::= DUP_TOP LOAD_ATTR store LOAD_ATTR CALL_FUNCTION_0 store\n        with           ::= expr setupwith SETUP_FINALLY\n                           suite_stmts_opt\n                           POP_BLOCK LOAD_CONST COME_FROM_FINALLY\n                           load delete WITH_CLEANUP END_FINALLY\n\n        # Keeps Python 3.1 "with .. as" designator in the same position as it is in other version.\n        setupwithas31  ::= setupwithas SETUP_FINALLY load delete\n\n        withasstmt     ::= expr setupwithas31 store\n                           suite_stmts_opt\n                           POP_BLOCK LOAD_CONST COME_FROM_FINALLY\n                           load delete WITH_CLEANUP END_FINALLY\n\n        store ::= STORE_NAME\n        load  ::= LOAD_FAST\n        load  ::= LOAD_NAME\n        '

    def remove_rules_31(self):
        if False:
            i = 10
            return i + 15
        self.remove_rules('\n        # DUP_TOP_TWO is DUP_TOPX in 3.1 and earlier\n        subscript2 ::= expr expr DUP_TOP_TWO BINARY_SUBSCR\n\n        # The were found using grammar coverage\n        list_if     ::= expr jmp_false list_iter COME_FROM\n        list_if_not ::= expr jmp_true list_iter COME_FROM\n        ')

    def customize_grammar_rules(self, tokens, customize):
        if False:
            while True:
                i = 10
        super(Python31Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules_31()
        return
    pass

class Python31ParserSingle(Python31Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python31Parser()
    p.remove_rules_31()
    p.check_grammar()
    from xdis.version_info import IS_PYPY, PYTHON_VERSION_TRIPLE
    if PYTHON_VERSION_TRIPLE[:2] == (3, 1):
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
        import sys
        if len(sys.argv) > 1:
            from spark_parser.spark import rule2str
            for rule in sorted(p.rule2name.items()):
                print(rule2str(rule[0]))