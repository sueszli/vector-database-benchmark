"""
spark grammar differences over Python 3.4 for Python 3.5.
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle, nop_func
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parsers.parse34 import Python34Parser

class Python35Parser(Python34Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            i = 10
            return i + 15
        super(Python35Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_35on(self, args):
        if False:
            print('Hello World!')
        '\n\n        # FIXME! isolate this to only loops!\n        _ifstmts_jump  ::= c_stmts_opt come_froms\n        ifelsestmt ::= testexpr c_stmts_opt jump_forward_else else_suite _come_froms\n\n        pb_ja ::= POP_BLOCK JUMP_ABSOLUTE\n\n        # The number of canned instructions in new statements is mind boggling.\n        # I\'m sure by the time Python 4 comes around these will be turned\n        # into special opcodes\n\n        while1stmt     ::= SETUP_LOOP l_stmts COME_FROM JUMP_BACK\n                           POP_BLOCK COME_FROM_LOOP\n        while1stmt     ::= SETUP_LOOP l_stmts POP_BLOCK COME_FROM_LOOP\n        while1elsestmt ::= SETUP_LOOP l_stmts JUMP_BACK\n                           POP_BLOCK else_suite COME_FROM_LOOP\n\n        # The following rule is for Python 3.5+ where we can have stuff like\n        # while ..\n        #     if\n        #     ...\n        # the end of the if will jump back to the loop and there will be a COME_FROM\n        # after the jump\n        l_stmts ::= lastl_stmt come_froms l_stmts\n\n        # Python 3.5+ Await statement\n        expr       ::= await_expr\n        await_expr ::= expr GET_AWAITABLE LOAD_CONST YIELD_FROM\n\n        stmt       ::= await_stmt\n        await_stmt ::= await_expr POP_TOP\n\n        # Python 3.5+ has WITH_CLEANUP_START/FINISH\n\n        with       ::= expr\n                       SETUP_WITH POP_TOP suite_stmts_opt\n                       POP_BLOCK LOAD_CONST COME_FROM_WITH\n                       WITH_CLEANUP_START WITH_CLEANUP_FINISH END_FINALLY\n\n        withasstmt ::= expr\n                       SETUP_WITH store suite_stmts_opt\n                       POP_BLOCK LOAD_CONST COME_FROM_WITH\n                       WITH_CLEANUP_START WITH_CLEANUP_FINISH END_FINALLY\n\n        # Python 3.5+ async additions\n        stmt               ::= async_for_stmt\n        async_for_stmt     ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK jump_except COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                               JUMP_ABSOLUTE END_FINALLY COME_FROM\n                               for_block POP_BLOCK JUMP_ABSOLUTE\n                               COME_FROM_LOOP\n\n        async_for_stmt     ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK jump_except COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                               JUMP_ABSOLUTE END_FINALLY JUMP_BACK\n                               pass POP_BLOCK JUMP_ABSOLUTE\n                               COME_FROM_LOOP\n\n        stmt               ::= async_forelse_stmt\n        async_forelse_stmt ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                               JUMP_ABSOLUTE END_FINALLY COME_FROM\n                               for_block pb_ja\n                               else_suite COME_FROM_LOOP\n\n\n        inplace_op       ::= INPLACE_MATRIX_MULTIPLY\n        binary_operator  ::= BINARY_MATRIX_MULTIPLY\n\n        # Python 3.5+ does jump optimization\n        # In <.3.5 the below is a JUMP_FORWARD to a JUMP_ABSOLUTE.\n\n        return_if_stmt    ::= return_expr RETURN_END_IF POP_BLOCK\n        return_if_lambda  ::= RETURN_END_IF_LAMBDA COME_FROM\n\n        jb_else     ::= JUMP_BACK ELSE\n        ifelsestmtc ::= testexpr c_stmts_opt JUMP_FORWARD else_suitec\n        ifelsestmtl ::= testexpr c_stmts_opt jb_else else_suitel\n\n        # 3.5 Has jump optimization which can route the end of an\n        # "if/then" back to to a loop just before an else.\n        jump_absolute_else ::= jb_else\n        jump_absolute_else ::= CONTINUE ELSE\n\n        # Our hacky "ELSE" determination doesn\'t do a good job and really\n        # determine the start of an "else". It could also be the end of an\n        # "if-then" which ends in a "continue". Perhaps with real control-flow\n        # analysis we\'ll sort this out. Or call "ELSE" something more appropriate.\n        _ifstmts_jump ::= c_stmts_opt ELSE\n\n        # ifstmt ::= testexpr c_stmts_opt\n\n        iflaststmt ::= testexpr c_stmts_opt JUMP_FORWARD\n\n        # Python 3.3+ also has yield from. 3.5 does it\n        # differently than 3.3, 3.4\n\n        yield_from ::= expr GET_YIELD_FROM_ITER LOAD_CONST YIELD_FROM\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            for i in range(10):
                print('nop')
        self.remove_rules('\n          yield_from ::= expr GET_ITER LOAD_CONST YIELD_FROM\n          yield_from ::= expr expr YIELD_FROM\n          with       ::= expr SETUP_WITH POP_TOP suite_stmts_opt\n                         POP_BLOCK LOAD_CONST COME_FROM_WITH\n                         WITH_CLEANUP END_FINALLY\n          withasstmt ::= expr SETUP_WITH store suite_stmts_opt\n                         POP_BLOCK LOAD_CONST COME_FROM_WITH\n                         WITH_CLEANUP END_FINALLY\n        ')
        super(Python35Parser, self).customize_grammar_rules(tokens, customize)
        for (i, token) in enumerate(tokens):
            opname = token.kind
            if opname == 'LOAD_ASSERT':
                if 'PyPy' in customize:
                    rules_str = '\n                    stmt ::= JUMP_IF_NOT_DEBUG stmts COME_FROM\n                    '
                    self.add_unique_doc_rules(rules_str, customize)
            elif opname == 'BUILD_MAP_UNPACK_WITH_CALL':
                if self.version < (3, 7):
                    self.addRule('expr ::= unmapexpr', nop_func)
                    nargs = token.attr % 256
                    map_unpack_n = 'map_unpack_%s' % nargs
                    rule = map_unpack_n + ' ::= ' + 'expr ' * nargs
                    self.addRule(rule, nop_func)
                    rule = 'unmapexpr ::=  %s %s' % (map_unpack_n, opname)
                    self.addRule(rule, nop_func)
                    call_token = tokens[i + 1]
                    rule = 'call ::= expr unmapexpr ' + call_token.kind
                    self.addRule(rule, nop_func)
            elif opname == 'BEFORE_ASYNC_WITH' and self.version < (3, 8):
                rules_str = '\n                   stmt               ::= async_with_stmt\n                   async_with_pre     ::= BEFORE_ASYNC_WITH GET_AWAITABLE LOAD_CONST YIELD_FROM SETUP_ASYNC_WITH\n                   async_with_post    ::= COME_FROM_ASYNC_WITH\n                                          WITH_CLEANUP_START GET_AWAITABLE LOAD_CONST YIELD_FROM\n                                          WITH_CLEANUP_FINISH END_FINALLY\n\n                   async_with_stmt    ::= expr\n                                          async_with_pre\n                                          POP_TOP\n                                          suite_stmts_opt\n                                          POP_BLOCK LOAD_CONST\n                                          async_with_post\n                   async_with_stmt    ::= expr\n                                          async_with_pre\n                                          POP_TOP\n                                          suite_stmts_opt\n                                          async_with_post\n\n                   stmt               ::= async_with_as_stmt\n\n                   async_with_as_stmt ::= expr\n                                          async_with_pre\n                                          store\n                                          suite_stmts_opt\n                                          POP_BLOCK LOAD_CONST\n                                          async_with_post\n                '
                self.addRule(rules_str, nop_func)
            elif opname == 'BUILD_MAP_UNPACK':
                self.addRule('\n                   expr        ::= dict_unpack\n                   dict_unpack ::= dict_comp BUILD_MAP_UNPACK\n                   ', nop_func)
            elif opname == 'SETUP_WITH':
                rules_str = '\n                  with ::= expr\n                           SETUP_WITH POP_TOP suite_stmts_opt\n                           POP_BLOCK LOAD_CONST COME_FROM_WITH\n                           WITH_CLEANUP_START WITH_CLEANUP_FINISH END_FINALLY\n\n                  withasstmt ::= expr\n                       SETUP_WITH store suite_stmts_opt\n                       POP_BLOCK LOAD_CONST COME_FROM_WITH\n                       WITH_CLEANUP_START WITH_CLEANUP_FINISH END_FINALLY\n                '
                self.addRule(rules_str, nop_func)
            pass
        return

    def custom_classfunc_rule(self, opname, token, customize, *args):
        if False:
            i = 10
            return i + 15
        (args_pos, args_kw) = self.get_pos_kw(token)
        nak = (len(opname) - len('CALL_FUNCTION')) // 3
        uniq_param = args_kw + args_pos
        if frozenset(('GET_AWAITABLE', 'YIELD_FROM')).issubset(self.seen_ops):
            rule = 'async_call ::= expr ' + 'pos_arg ' * args_pos + 'kwarg ' * args_kw + 'expr ' * nak + token.kind + ' GET_AWAITABLE LOAD_CONST YIELD_FROM'
            self.add_unique_rule(rule, token.kind, uniq_param, customize)
            self.add_unique_rule('expr ::= async_call', token.kind, uniq_param, customize)
        if opname.startswith('CALL_FUNCTION_VAR'):
            token.kind = self.call_fn_name(token)
            if opname.endswith('KW'):
                kw = 'expr '
            else:
                kw = ''
            rule = 'call ::= expr expr ' + 'pos_arg ' * args_pos + 'kwarg ' * args_kw + kw + token.kind
            self.add_unique_rule(rule, token.kind, args_pos, customize)
        else:
            super(Python35Parser, self).custom_classfunc_rule(opname, token, customize, *args)

class Python35ParserSingle(Python35Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python35Parser()
    p.check_grammar()
    from xdis.version_info import PYTHON_VERSION_TRIPLE, IS_PYPY
    if PYTHON_VERSION_TRIPLE[:2] == (3, 5):
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