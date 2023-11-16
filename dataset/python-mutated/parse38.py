"""
spark grammar differences over Python 3.7 for Python 3.8
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle, nop_func
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parsers.parse37 import Python37Parser

class Python38Parser(Python37Parser):

    def p_38_stmt(self, args):
        if False:
            print('Hello World!')
        '\n        stmt               ::= async_for_stmt38\n        stmt               ::= async_forelse_stmt38\n        stmt               ::= call_stmt\n        stmt               ::= continue\n        stmt               ::= for38\n        stmt               ::= forelselaststmt38\n        stmt               ::= forelselaststmtl38\n        stmt               ::= forelsestmt38\n        stmt               ::= try_elsestmtl38\n        stmt               ::= try_except38\n        stmt               ::= try_except38r\n        stmt               ::= try_except38r2\n        stmt               ::= try_except38r3\n        stmt               ::= try_except38r4\n        stmt               ::= try_except_as\n        stmt               ::= try_except_ret38\n        stmt               ::= tryfinally38astmt\n        stmt               ::= tryfinally38rstmt\n        stmt               ::= tryfinally38rstmt2\n        stmt               ::= tryfinally38rstmt3\n        stmt               ::= tryfinally38stmt\n        stmt               ::= whileTruestmt38\n        stmt               ::= whilestmt38\n\n        call_stmt          ::= call\n        break ::= POP_BLOCK BREAK_LOOP\n        break ::= POP_BLOCK POP_TOP BREAK_LOOP\n        break ::= POP_TOP BREAK_LOOP\n        break ::= POP_EXCEPT BREAK_LOOP\n\n        # The "continue" rule is a weird one. In 3.8, CONTINUE_LOOP was removed.\n        # Inside an loop we can have this, which can only appear in side a try/except\n        # And it can also appear at the end of the try except.\n        continue           ::= POP_EXCEPT JUMP_BACK\n\n\n        # FIXME: this should be restricted to being inside a try block\n        stmt               ::= except_ret38\n        stmt               ::= except_ret38a\n\n        # FIXME: this should be added only when seeing GET_AITER or YIELD_FROM\n        async_for          ::= GET_AITER _come_froms\n                               SETUP_FINALLY GET_ANEXT LOAD_CONST YIELD_FROM POP_BLOCK\n        async_for_stmt38   ::= expr async_for\n                               store for_block\n                               COME_FROM_FINALLY\n                               END_ASYNC_FOR\n\n       genexpr_func_async  ::= LOAD_ARG func_async_prefix\n                               store comp_iter\n                               JUMP_BACK COME_FROM_FINALLY\n                               END_ASYNC_FOR\n\n        # FIXME: come froms after the else_suite or END_ASYNC_FOR distinguish which of\n        # for / forelse is used. Add come froms and check of add up control-flow detection phase.\n        async_forelse_stmt38 ::= expr\n                               GET_AITER\n                               SETUP_FINALLY\n                               GET_ANEXT\n                               LOAD_CONST\n                               YIELD_FROM\n                               POP_BLOCK\n                               store for_block\n                               COME_FROM_FINALLY\n                               END_ASYNC_FOR\n                               else_suite\n\n        # Seems to be used to discard values before a return in a "for" loop\n        discard_top        ::= ROT_TWO POP_TOP\n        discard_tops       ::= discard_top+\n\n        return             ::= return_expr\n                               discard_tops RETURN_VALUE\n\n        return             ::= popb_return\n        return             ::= pop_return\n        return             ::= pop_ex_return\n        except_stmt        ::= pop_ex_return\n        pop_return         ::= POP_TOP return_expr RETURN_VALUE\n        popb_return        ::= return_expr POP_BLOCK RETURN_VALUE\n        pop_ex_return      ::= return_expr ROT_FOUR POP_EXCEPT RETURN_VALUE\n\n        # 3.8 can push a looping JUMP_BACK into into a JUMP_ from a statement that jumps to it\n        lastl_stmt         ::= ifpoplaststmtl\n        ifpoplaststmtl     ::= testexpr POP_TOP c_stmts_opt\n        ifelsestmtl        ::= testexpr c_stmts_opt jb_cfs else_suitel JUMP_BACK come_froms\n\n        # Keep indices the same in ifelsestmtl\n        cf_pt              ::= COME_FROM POP_TOP\n        ifelsestmtl        ::= testexpr c_stmts cf_pt else_suite\n\n        for38              ::= expr get_iter store for_block JUMP_BACK\n        for38              ::= expr get_for_iter store for_block JUMP_BACK\n        for38              ::= expr get_for_iter store for_block JUMP_BACK POP_BLOCK\n        for38              ::= expr get_for_iter store for_block\n\n        forelsestmt38      ::= expr get_for_iter store for_block POP_BLOCK else_suite\n        forelsestmt38      ::= expr get_for_iter store for_block JUMP_BACK _come_froms\n                               else_suite\n\n        forelselaststmt38  ::= expr get_for_iter store for_block POP_BLOCK else_suitec\n        forelselaststmtl38 ::= expr get_for_iter store for_block POP_BLOCK else_suitel\n\n        returns_in_except   ::= _stmts except_return_value\n        except_return_value ::= POP_BLOCK return\n        except_return_value ::= expr POP_BLOCK RETURN_VALUE\n\n        whilestmt38        ::= _come_froms testexpr l_stmts_opt COME_FROM JUMP_BACK\n                                POP_BLOCK\n        whilestmt38        ::= _come_froms testexpr l_stmts_opt JUMP_BACK POP_BLOCK\n        whilestmt38        ::= _come_froms testexpr l_stmts_opt JUMP_BACK come_froms\n        whilestmt38        ::= _come_froms testexpr returns               POP_BLOCK\n        whilestmt38        ::= _come_froms testexpr l_stmts     JUMP_BACK\n        whilestmt38        ::= _come_froms testexpr l_stmts     come_froms\n\n        # while1elsestmt   ::=          l_stmts     JUMP_BACK\n        whileTruestmt      ::= _come_froms l_stmts              JUMP_BACK POP_BLOCK\n        while1stmt         ::= _come_froms l_stmts COME_FROM JUMP_BACK COME_FROM_LOOP\n        whileTruestmt38    ::= _come_froms l_stmts JUMP_BACK\n        whileTruestmt38    ::= _come_froms l_stmts JUMP_BACK COME_FROM_EXCEPT_CLAUSE\n        whileTruestmt38    ::= _come_froms pass JUMP_BACK\n\n        for_block          ::= _come_froms l_stmts_opt _come_from_loops JUMP_BACK\n\n        except_cond1       ::= DUP_TOP expr COMPARE_OP jmp_false\n                               POP_TOP POP_TOP POP_TOP\n                               POP_EXCEPT\n        except_cond_as     ::= DUP_TOP expr COMPARE_OP POP_JUMP_IF_FALSE\n                               POP_TOP STORE_FAST POP_TOP\n\n        try_elsestmtl38    ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                               except_handler38 COME_FROM\n                               else_suitel opt_come_from_except\n        try_except         ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                               except_handler38\n\n        try_except38       ::= SETUP_FINALLY POP_BLOCK POP_TOP suite_stmts_opt\n                               except_handler38a\n\n        # suite_stmts has a return\n        try_except38       ::= SETUP_FINALLY POP_BLOCK suite_stmts\n                               except_handler38b\n        try_except38r      ::= SETUP_FINALLY return_except\n                               except_handler38b\n        return_except      ::= stmts POP_BLOCK return\n\n\n        # In 3.8 there seems to be some sort of code fiddle with POP_EXCEPT when there\n        # is a final return in the "except" block.\n        # So we treat the "return" separate from the other statements\n        cond_except_stmt      ::= except_cond1 except_stmts\n        cond_except_stmts_opt ::= cond_except_stmt*\n\n        try_except38r2     ::= SETUP_FINALLY\n                               suite_stmts_opt\n                               POP_BLOCK JUMP_FORWARD\n                               COME_FROM_FINALLY POP_TOP POP_TOP POP_TOP\n                               cond_except_stmts_opt\n                               POP_EXCEPT return\n                               END_FINALLY\n                               COME_FROM\n\n        try_except38r3     ::= SETUP_FINALLY\n                               suite_stmts_opt\n                               POP_BLOCK JUMP_FORWARD\n                               COME_FROM_FINALLY\n                               cond_except_stmts_opt\n                               POP_EXCEPT return\n                               COME_FROM\n                               END_FINALLY\n                               COME_FROM\n\n\n        try_except38r4     ::= SETUP_FINALLY\n                               returns_in_except\n                               COME_FROM_FINALLY\n                               except_cond1\n                               return\n                               COME_FROM\n                               END_FINALLY\n\n\n        # suite_stmts has a return\n        try_except38       ::= SETUP_FINALLY POP_BLOCK suite_stmts\n                               except_handler38b\n        try_except_as      ::= SETUP_FINALLY POP_BLOCK suite_stmts\n                               except_handler_as END_FINALLY COME_FROM\n        try_except_as      ::= SETUP_FINALLY suite_stmts\n                               except_handler_as END_FINALLY COME_FROM\n\n        try_except_ret38   ::= SETUP_FINALLY returns except_ret38a\n        try_except_ret38a  ::= SETUP_FINALLY returns except_handler38c\n                               END_FINALLY come_from_opt\n\n        # Note: there is a suite_stmts_opt which seems\n        # to be bookkeeping which is not expressed in source code\n        except_ret38       ::= SETUP_FINALLY expr ROT_FOUR POP_BLOCK POP_EXCEPT\n                               CALL_FINALLY RETURN_VALUE COME_FROM\n                               COME_FROM_FINALLY\n                               suite_stmts_opt END_FINALLY\n        except_ret38a      ::= COME_FROM_FINALLY POP_TOP POP_TOP POP_TOP\n                               expr ROT_FOUR\n                               POP_EXCEPT RETURN_VALUE END_FINALLY\n\n        except_handler38   ::= _jump COME_FROM_FINALLY\n                               except_stmts END_FINALLY opt_come_from_except\n        except_handler38a  ::= COME_FROM_FINALLY POP_TOP POP_TOP POP_TOP\n                               POP_EXCEPT POP_TOP stmts END_FINALLY\n\n        except_handler38c  ::= COME_FROM_FINALLY except_cond1a except_stmts\n                               POP_EXCEPT JUMP_FORWARD COME_FROM\n        except_handler_as  ::= COME_FROM_FINALLY except_cond_as tryfinallystmt\n                               POP_EXCEPT JUMP_FORWARD COME_FROM\n\n        tryfinallystmt     ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                               BEGIN_FINALLY COME_FROM_FINALLY suite_stmts_opt\n                               END_FINALLY\n\n\n        lc_setup_finally   ::= LOAD_CONST SETUP_FINALLY\n        call_finally_pt    ::= CALL_FINALLY POP_TOP\n        cf_cf_finally      ::= come_from_opt COME_FROM_FINALLY\n        pop_finally_pt     ::= POP_FINALLY POP_TOP\n        ss_end_finally     ::= suite_stmts END_FINALLY\n        sf_pb_call_returns ::= SETUP_FINALLY POP_BLOCK CALL_FINALLY returns\n\n\n        # FIXME: DRY rules below\n        tryfinally38rstmt  ::= sf_pb_call_returns\n                               cf_cf_finally\n                               ss_end_finally\n        tryfinally38rstmt  ::= sf_pb_call_returns\n                               cf_cf_finally END_FINALLY\n                               suite_stmts\n        tryfinally38rstmt  ::= sf_pb_call_returns\n                               cf_cf_finally POP_FINALLY\n                               ss_end_finally\n        tryfinally38rstmt  ::= sf_bp_call_returns\n                               COME_FROM_FINALLY POP_FINALLY\n                               ss_end_finally\n\n        tryfinally38rstmt2 ::= lc_setup_finally POP_BLOCK call_finally_pt\n                               returns\n                               cf_cf_finally pop_finally_pt\n                               ss_end_finally POP_TOP\n        tryfinally38rstmt3 ::= SETUP_FINALLY expr POP_BLOCK CALL_FINALLY RETURN_VALUE\n                               COME_FROM COME_FROM_FINALLY\n                               ss_end_finally\n\n        tryfinally38stmt   ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                               BEGIN_FINALLY COME_FROM_FINALLY\n                               POP_FINALLY suite_stmts_opt END_FINALLY\n\n        tryfinally38astmt  ::= LOAD_CONST SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                               BEGIN_FINALLY COME_FROM_FINALLY\n                               POP_FINALLY POP_TOP suite_stmts_opt END_FINALLY POP_TOP\n        '

    def p_38walrus(self, args):
        if False:
            print('Hello World!')
        '\n        # named_expr is also known as the "walrus op" :=\n        expr              ::= named_expr\n        named_expr        ::= expr DUP_TOP store\n        '

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            for i in range(10):
                print('nop')
        super(Python38Parser, self).__init__(debug_parser)
        self.customized = {}

    def remove_rules_38(self):
        if False:
            while True:
                i = 10
        self.remove_rules('\n           stmt               ::= async_for_stmt37\n           stmt               ::= for\n           stmt               ::= forelsestmt\n           stmt               ::= try_except36\n           stmt               ::= async_forelse_stmt\n\n           async_for_stmt     ::= setup_loop expr\n                                  GET_AITER\n                                  SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                                  YIELD_FROM\n                                  store\n                                  POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                                  LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                                  END_FINALLY COME_FROM\n                                  for_block\n                                  COME_FROM\n                                  POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP POP_BLOCK\n                                  COME_FROM_LOOP\n           async_for_stmt37   ::= setup_loop expr\n                                  GET_AITER\n                                  SETUP_EXCEPT GET_ANEXT\n                                  LOAD_CONST YIELD_FROM\n                                  store\n                                  POP_BLOCK JUMP_BACK COME_FROM_EXCEPT DUP_TOP\n                                  LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                                  END_FINALLY for_block COME_FROM\n                                  POP_TOP POP_TOP POP_TOP POP_EXCEPT\n                                  POP_TOP POP_BLOCK\n                                  COME_FROM_LOOP\n\n          async_forelse_stmt ::= setup_loop expr\n                                 GET_AITER\n                                 SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                                 YIELD_FROM\n                                 store\n                                 POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                                 LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                                 END_FINALLY COME_FROM\n                                 for_block\n                                 COME_FROM\n                                 POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP POP_BLOCK\n                                 else_suite COME_FROM_LOOP\n\n           for                ::= setup_loop expr get_for_iter store for_block POP_BLOCK\n           for                ::= setup_loop expr get_for_iter store for_block POP_BLOCK NOP\n\n           for_block          ::= l_stmts_opt COME_FROM_LOOP JUMP_BACK\n           forelsestmt        ::= setup_loop expr get_for_iter store for_block POP_BLOCK else_suite\n           forelselaststmt    ::= setup_loop expr get_for_iter store for_block POP_BLOCK else_suitec\n           forelselaststmtl   ::= setup_loop expr get_for_iter store for_block POP_BLOCK else_suitel\n\n           tryelsestmtl3      ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                  except_handler COME_FROM else_suitel\n                                  opt_come_from_except\n           try_except         ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                  except_handler opt_come_from_except\n           tryfinallystmt     ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                                  LOAD_CONST COME_FROM_FINALLY suite_stmts_opt\n                                  END_FINALLY\n           tryfinally36       ::= SETUP_FINALLY returns\n                                  COME_FROM_FINALLY suite_stmts_opt END_FINALLY\n           tryfinally_return_stmt ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK\n                                      LOAD_CONST COME_FROM_FINALLY\n        ')

    def customize_grammar_rules(self, tokens, customize):
        if False:
            return 10
        super(Python37Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules_38()
        self.check_reduce['whileTruestmt38'] = 'tokens'
        self.check_reduce['whilestmt38'] = 'tokens'
        self.check_reduce['try_elsestmtl38'] = 'AST'
        customize_instruction_basenames = frozenset(('BEFORE', 'BUILD', 'CALL', 'DICT', 'GET', 'FORMAT', 'LIST', 'LOAD', 'MAKE', 'SETUP', 'UNPACK'))
        custom_ops_processed = frozenset()
        self.seen_ops = frozenset([t.kind for t in tokens])
        self.seen_op_basenames = frozenset([opname[:opname.rfind('_')] for opname in self.seen_ops])
        custom_ops_processed = set(['DICT_MERGE'])
        if 'PyPy' in customize:
            self.addRule('\n              stmt ::= assign3_pypy\n              stmt ::= assign2_pypy\n              assign3_pypy       ::= expr expr expr store store store\n              assign2_pypy       ::= expr expr store store\n              ', nop_func)
        n = len(tokens)
        has_get_iter_call_function1 = False
        for (i, token) in enumerate(tokens):
            if token == 'GET_ITER' and i < n - 2 and (tokens[i + 1] == 'CALL_FUNCTION_1'):
                has_get_iter_call_function1 = True
        for (i, token) in enumerate(tokens):
            opname = token.kind
            if opname[:opname.find('_')] not in customize_instruction_basenames or opname in custom_ops_processed:
                continue
            opname_base = opname[:opname.rfind('_')]
            if opname[:opname.find('_')] not in customize_instruction_basenames or opname in custom_ops_processed:
                continue
            if opname_base in ('BUILD_LIST', 'BUILD_SET', 'BUILD_SET_UNPACK', 'BUILD_TUPLE', 'BUILD_TUPLE_UNPACK'):
                v = token.attr
                is_LOAD_CLOSURE = False
                if opname_base == 'BUILD_TUPLE':
                    is_LOAD_CLOSURE = True
                    for j in range(v):
                        if tokens[i - j - 1].kind != 'LOAD_CLOSURE':
                            is_LOAD_CLOSURE = False
                            break
                    if is_LOAD_CLOSURE:
                        rule = 'load_closure ::= %s%s' % ('LOAD_CLOSURE ' * v, opname)
                        self.add_unique_rule(rule, opname, token.attr, customize)
                elif opname_base == 'BUILD_LIST':
                    v = token.attr
                    if v == 0:
                        rule_str = '\n                           list        ::= BUILD_LIST_0\n                           list_unpack ::= BUILD_LIST_0 expr LIST_EXTEND\n                           list        ::= list_unpack\n                        '
                        self.add_unique_doc_rules(rule_str, customize)
                elif opname == 'BUILD_TUPLE_UNPACK_WITH_CALL':
                    self.addRule('expr        ::= call_ex_kw3\n                           call_ex_kw3 ::= expr\n                                           build_tuple_unpack_with_call\n                                           expr\n                                           CALL_FUNCTION_EX_KW\n                        ', nop_func)
                if not is_LOAD_CLOSURE or v == 0:
                    build_count = token.attr
                    thousands = build_count // 1024
                    thirty32s = build_count // 32 % 32
                    if thirty32s > 0:
                        rule = 'expr32 ::=%s' % (' expr' * 32)
                        self.add_unique_rule(rule, opname_base, build_count, customize)
                        pass
                    if thousands > 0:
                        self.add_unique_rule('expr1024 ::=%s' % (' expr32' * 32), opname_base, build_count, customize)
                        pass
                    collection = opname_base[opname_base.find('_') + 1:].lower()
                    rule = '%s ::= ' % collection + 'expr1024 ' * thousands + 'expr32 ' * thirty32s + 'expr ' * (build_count % 32) + opname
                    self.add_unique_rules(['expr ::= %s' % collection, rule], customize)
                    continue
                continue
            elif opname == 'BUILD_STRING_2':
                self.addRule('\n                     expr                  ::= formatted_value_debug\n                     formatted_value_debug ::= LOAD_STR formatted_value2 BUILD_STRING_2\n                     formatted_value_debug ::= LOAD_STR formatted_value1 BUILD_STRING_2\n                   ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'BUILD_STRING_3':
                self.addRule('\n                     expr                  ::= formatted_value_debug\n                     formatted_value_debug ::= LOAD_STR formatted_value2 LOAD_STR BUILD_STRING_3\n                     formatted_value_debug ::= LOAD_STR formatted_value1 LOAD_STR BUILD_STRING_3\n                   ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_CLOSURE':
                self.addRule('load_closure ::= LOAD_CLOSURE+', nop_func)
            elif opname == 'LOOKUP_METHOD':
                self.addRule('\n                             expr      ::= attribute\n                             attribute ::= expr LOOKUP_METHOD\n                             ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'MAKE_FUNCTION_8':
                if 'LOAD_DICTCOMP' in self.seen_ops:
                    rule = '\n                       dict_comp ::= load_closure LOAD_DICTCOMP LOAD_STR\n                                     MAKE_FUNCTION_8 expr\n                                     GET_ITER CALL_FUNCTION_1\n                       '
                    self.addRule(rule, nop_func)
                elif 'LOAD_SETCOMP' in self.seen_ops:
                    rule = '\n                       set_comp ::= load_closure LOAD_SETCOMP LOAD_STR\n                                    MAKE_FUNCTION_CLOSURE expr\n                                    GET_ITER CALL_FUNCTION_1\n                       '
                    self.addRule(rule, nop_func)

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            i = 10
            return i + 15
        invalid = super(Python38Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        self.remove_rules_38()
        if invalid:
            return invalid
        lhs = rule[0]
        if lhs in ('whileTruestmt38', 'whilestmt38'):
            jb_index = last - 1
            while jb_index > 0 and tokens[jb_index].kind.startswith('COME_FROM'):
                jb_index -= 1
            t = tokens[jb_index]
            if t.kind != 'JUMP_BACK':
                return True
            return t.attr != tokens[first].off2int()
            pass
        return False

class Python38ParserSingle(Python38Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python38Parser()
    p.remove_rules_38()
    p.check_grammar()
    from xdis.version_info import PYTHON_VERSION_TRIPLE, IS_PYPY
    if PYTHON_VERSION_TRIPLE[:2] == (3, 8):
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