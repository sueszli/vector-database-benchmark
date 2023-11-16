"""
spark grammar differences over Python2 for Python 2.6.
"""
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse2 import Python2Parser
from uncompyle6.parsers.reducecheck import except_handler, ifelsestmt2, ifstmt2, tryelsestmt, tryexcept

class Python26Parser(Python2Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            for i in range(10):
                print('nop')
        super(Python26Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_try_except26(self, args):
        if False:
            return 10
        "\n        except_stmt    ::= except_cond3 except_suite\n        except_cond1   ::= DUP_TOP expr COMPARE_OP\n                           JUMP_IF_FALSE POP_TOP POP_TOP POP_TOP POP_TOP\n        except_cond3   ::= DUP_TOP expr COMPARE_OP\n                           JUMP_IF_FALSE POP_TOP POP_TOP store POP_TOP\n\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts\n                           come_froms_pop END_FINALLY come_froms\n\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts\n                           END_FINALLY\n\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts\n                           POP_TOP END_FINALLY\n                           come_froms\n\n        except_handler ::= jmp_abs COME_FROM except_stmts\n                           POP_TOP END_FINALLY\n\n        except_handler ::= jmp_abs COME_FROM except_stmts\n                           END_FINALLY JUMP_FORWARD\n\n\n        # Sometimes we don't put in COME_FROM to the next statement\n        # like we do in 2.7. Perhaps we should?\n        try_except     ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                            except_handler\n\n        tryelsestmt    ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                           except_handler else_suite come_froms\n        tryelsestmtl   ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                            except_handler else_suitel\n        tryelsestmtc   ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                           except_handler else_suitec\n\n        _ifstmts_jump  ::= c_stmts_opt JUMP_FORWARD COME_FROM POP_TOP\n\n        except_suite   ::= c_stmts_opt JUMP_FORWARD come_from_pop\n        except_suite   ::= c_stmts_opt jf_pop\n        except_suite   ::= c_stmts_opt jmp_abs come_from_pop\n\n        # This is what happens after a jump where\n        # we start a new block. For reasons that I don't fully\n        # understand, there is also a value on the top of the stack.\n        come_from_pop   ::=  COME_FROM POP_TOP\n        come_froms_pop  ::= come_froms POP_TOP\n        "

    def p_jumps26(self, args):
        if False:
            while True:
                i = 10
        "\n\n        # There are the equivalents of Python 2.7+'s\n        # POP_JUMP_IF_TRUE and POP_JUMP_IF_FALSE\n        jmp_true     ::= JUMP_IF_TRUE POP_TOP\n        jmp_false    ::= JUMP_IF_FALSE POP_TOP\n\n        jb_pop       ::= JUMP_BACK POP_TOP\n        jf_pop       ::= JUMP_FORWARD POP_TOP\n\n        jb_cont      ::= JUMP_BACK\n        jb_cont      ::= CONTINUE\n\n        jb_cf_pop ::= come_from_opt JUMP_BACK _come_froms POP_TOP\n        ja_cf_pop ::= JUMP_ABSOLUTE come_froms POP_TOP\n        jf_cf_pop ::= JUMP_FORWARD come_froms POP_TOP\n\n        # The first optional COME_FROM when it appears is really\n        # COME_FROM_LOOP, but in <= 2.6 we don't distinguish\n        # this\n\n        cf_jb_cf_pop ::= _come_froms JUMP_BACK come_froms POP_TOP\n\n        pb_come_from    ::= POP_BLOCK COME_FROM\n        jb_pb_come_from ::= JUMP_BACK pb_come_from\n\n        _ifstmts_jump ::= c_stmts_opt JUMP_FORWARD COME_FROM POP_TOP\n        _ifstmts_jump ::= c_stmts_opt JUMP_FORWARD come_froms POP_TOP COME_FROM\n\n        # This is what happens after a jump where\n        # we start a new block. For reasons that I don't fully\n        # understand, there is also a value on the top of the stack.\n        come_froms_pop  ::=  come_froms POP_TOP\n\n        "

    def p_stmt26(self, args):
        if False:
            return 10
        '\n        stmt ::= ifelsestmtr\n\n        # We use filler as a placeholder to keep nonterminal positions\n        # the same across different grammars so that the same semantic actions\n        # can be used\n        filler ::=\n\n        assert ::= assert_expr jmp_true LOAD_ASSERT RAISE_VARARGS_1 come_froms_pop\n        assert2 ::= assert_expr jmp_true LOAD_ASSERT expr RAISE_VARARGS_2 come_froms_pop\n\n        break ::= BREAK_LOOP JUMP_BACK\n\n        # Semantic actions want else_suitel to be at index 3\n        ifelsestmtl ::= testexpr c_stmts_opt cf_jb_cf_pop else_suitel\n        ifelsestmtc ::= testexpr c_stmts_opt ja_cf_pop    else_suitec\n\n        # Semantic actions want suite_stmts_opt to be at index 3\n        with        ::= expr setupwith SETUP_FINALLY suite_stmts_opt\n                        POP_BLOCK LOAD_CONST COME_FROM WITH_CLEANUP END_FINALLY\n\n        # Semantic actions want store to be at index 2\n        withasstmt ::= expr setupwithas store suite_stmts_opt\n                       POP_BLOCK LOAD_CONST COME_FROM WITH_CLEANUP END_FINALLY\n\n        # This is truly weird. 2.7 does this (not including POP_TOP) with\n        # opcode SETUP_WITH\n\n        setupwith     ::= DUP_TOP LOAD_ATTR ROT_TWO LOAD_ATTR CALL_FUNCTION_0 POP_TOP\n        setupwithas   ::= DUP_TOP LOAD_ATTR ROT_TWO LOAD_ATTR CALL_FUNCTION_0 setup_finally\n\n        setup_finally ::= STORE_FAST SETUP_FINALLY LOAD_FAST DELETE_FAST\n        setup_finally ::= STORE_NAME SETUP_FINALLY LOAD_NAME DELETE_NAME\n\n        while1stmt     ::= SETUP_LOOP l_stmts_opt come_from_opt JUMP_BACK _come_froms\n\n        # Sometimes JUMP_BACK is misclassified as CONTINUE.\n        # workaround until we have better control flow in place\n        while1stmt     ::= SETUP_LOOP l_stmts_opt CONTINUE _come_froms\n\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt jb_pop POP_BLOCK _come_froms\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt jb_cf_pop pb_come_from\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt jb_cf_pop POP_BLOCK\n        whilestmt      ::= SETUP_LOOP testexpr returns POP_BLOCK COME_FROM\n\n        # In the "whilestmt" below, there isn\'t a COME_FROM when the\n        # "while" is the last thing in the module or function.\n\n        whilestmt      ::= SETUP_LOOP testexpr returns POP_TOP POP_BLOCK\n\n        whileelsestmt  ::= SETUP_LOOP testexpr l_stmts_opt jb_pop POP_BLOCK\n                           else_suitel COME_FROM\n        while1elsestmt ::= SETUP_LOOP l_stmts JUMP_BACK else_suitel COME_FROM\n\n        return         ::= return_expr RETURN_END_IF POP_TOP\n        return         ::= return_expr RETURN_VALUE POP_TOP\n        return_if_stmt ::= return_expr RETURN_END_IF POP_TOP\n\n        iflaststmtl ::= testexpr c_stmts_opt jb_cf_pop\n        iflaststmt  ::= testexpr c_stmts_opt JUMP_ABSOLUTE come_from_pop\n\n        lastc_stmt ::= iflaststmt come_froms\n\n        ifstmt         ::= testexpr_then _ifstmts_jump\n\n        # Semantic actions want the else to be at position 3\n        ifelsestmt     ::= testexpr_then c_stmts_opt jf_cf_pop else_suite come_froms\n        ifelsestmt     ::= testexpr_then c_stmts_opt filler else_suitel come_froms POP_TOP\n        ifelsestmt     ::= testexpr c_stmts_opt jf_cf_pop else_suite\n\n        # We have no jumps to jumps, so no "come_froms" but a single "COME_FROM"\n        ifelsestmt     ::= testexpr      c_stmts_opt jf_cf_pop else_suite COME_FROM\n\n        # Semantic actions want else_suitel to be at index 3\n        ifelsestmtl    ::= testexpr_then c_stmts_opt jb_cf_pop else_suitel\n        ifelsestmtc    ::= testexpr_then c_stmts_opt ja_cf_pop else_suitec\n\n        iflaststmt     ::= testexpr_then c_stmts_opt JUMP_ABSOLUTE come_froms POP_TOP\n        iflaststmt     ::= testexpr      c_stmts_opt JUMP_ABSOLUTE come_froms POP_TOP\n\n        # "if"/"else" statement that ends in a RETURN\n        ifelsestmtr    ::= testexpr_then return_if_stmts returns\n\n        testexpr_then  ::= testtrue_then\n        testexpr_then  ::= testfalse_then\n        testtrue_then  ::= expr jmp_true_then\n        testfalse_then ::= expr jmp_false_then\n\n        jmp_false_then ::= JUMP_IF_FALSE THEN POP_TOP\n        jmp_true_then  ::= JUMP_IF_TRUE THEN POP_TOP\n\n        # In the "while1stmt" below, there sometimes isn\'t a\n        # "COME_FROM" when the "while1" is the last thing in the\n        # module or function.\n\n        while1stmt ::= SETUP_LOOP returns come_from_opt\n        for_block  ::= returns _come_froms\n        '

    def p_comp26(self, args):
        if False:
            while True:
                i = 10
        '\n        list_for ::= expr for_iter store list_iter JUMP_BACK come_froms POP_TOP\n\n        # The JUMP FORWARD below jumps to the JUMP BACK. It seems to happen\n        # in rare cases that may have to with length of code\n        # FIXME: we can add a reduction check for this\n\n        list_for ::= expr for_iter store list_iter JUMP_FORWARD come_froms POP_TOP\n                     COME_FROM JUMP_BACK\n\n        list_for ::= expr for_iter store list_iter jb_cont\n\n        # This is for a really funky:\n        #   [  x for x in range(10) if x % 2 if x % 3 ]\n        # the JUMP_ABSOLUTE is to the instruction after the last POP_TOP\n        #  we have a reduction check for this\n\n        list_for ::= expr for_iter store list_iter JUMP_ABSOLUTE come_froms\n                     POP_TOP jb_pop\n\n        list_iter  ::= list_if JUMP_BACK\n        list_iter  ::= list_if JUMP_BACK COME_FROM POP_TOP\n        list_comp  ::= BUILD_LIST_0 DUP_TOP\n                       store list_iter delete\n        list_comp  ::= BUILD_LIST_0 DUP_TOP\n                       store list_iter JUMP_BACK delete\n        lc_body    ::= LOAD_NAME expr LIST_APPEND\n        lc_body    ::= LOAD_FAST expr LIST_APPEND\n\n        comp_for ::= SETUP_LOOP expr for_iter store comp_iter jb_pb_come_from\n\n        comp_iter   ::= comp_if_not\n        comp_if_not ::= expr jmp_true comp_iter\n\n        comp_body   ::= gen_comp_body\n\n        for_block ::= l_stmts_opt _come_froms POP_TOP JUMP_BACK\n\n        # Make sure we keep indices the same as 2.7\n\n        setup_loop_lf ::= SETUP_LOOP LOAD_FAST\n        genexpr_func ::= setup_loop_lf FOR_ITER store comp_iter jb_pb_come_from\n        genexpr_func ::= setup_loop_lf FOR_ITER store comp_iter JUMP_BACK come_from_pop\n                         jb_pb_come_from\n\n        # This is for a really funky:\n        #   (x for x in range(10) if x % 2 if x % 3 )\n        # the JUMP_ABSOLUTE is to the instruction after the last POP_TOP.\n        # Add a reduction check for this?\n\n        genexpr_func ::= setup_loop_lf FOR_ITER store comp_iter JUMP_ABSOLUTE come_froms\n                         POP_TOP jb_pop jb_pb_come_from\n\n        genexpr_func ::= setup_loop_lf FOR_ITER store comp_iter JUMP_BACK come_froms\n                         POP_TOP jb_pb_come_from\n\n        generator_exp ::= LOAD_GENEXPR MAKE_FUNCTION_0 expr GET_ITER CALL_FUNCTION_1 COME_FROM\n        list_if ::= expr jmp_false_then list_iter\n        '

    def p_ret26(self, args):
        if False:
            i = 10
            return i + 15
        '\n        ret_and      ::= expr jmp_false return_expr_or_cond COME_FROM\n        ret_or       ::= expr jmp_true return_expr_or_cond COME_FROM\n        if_exp_ret   ::= expr jmp_false_then expr RETURN_END_IF POP_TOP return_expr_or_cond\n        if_exp_ret   ::= expr jmp_false_then expr return_expr_or_cond\n\n        return_if_stmt ::= return_expr RETURN_END_IF POP_TOP\n        return ::= return_expr RETURN_VALUE POP_TOP\n\n        # FIXME: split into Python 2.5\n        ret_or   ::= expr jmp_true return_expr_or_cond come_froms\n        '

    def p_except26(self, args):
        if False:
            return 10
        '\n        except_suite ::= c_stmts_opt jmp_abs POP_TOP\n        '

    def p_misc26(self, args):
        if False:
            return 10
        '\n        dict ::= BUILD_MAP kvlist\n        kvlist ::= kvlist kv3\n\n        # Note: preserve positions 0 2 and 4 for semantic actions\n        if_exp_not         ::= expr jmp_true  expr jf_cf_pop expr COME_FROM\n        if_exp             ::= expr jmp_false expr jf_cf_pop expr come_from_opt\n        if_exp             ::= expr jmp_false expr ja_cf_pop expr\n\n        expr               ::= if_exp_not\n\n        and                ::= expr JUMP_IF_FALSE POP_TOP expr JUMP_IF_FALSE POP_TOP\n\n        # A "compare_chained" is two comparisions like x <= y <= z\n        compare_chained          ::= expr compared_chained_middle ROT_TWO\n                                     COME_FROM POP_TOP _come_froms\n        compared_chained_middle  ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                     jmp_false compared_chained_middle _come_froms\n        compared_chained_middle   ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                     jmp_false compare_chained_right _come_froms\n\n        compared_chained_middle   ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                      jmp_false_then compared_chained_middle _come_froms\n        compared_chained_middle   ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                      jmp_false_then compare_chained_right _come_froms\n\n        compare_chained_right   ::= expr COMPARE_OP return_expr_lambda\n        compare_chained_right   ::= expr COMPARE_OP RETURN_END_IF_LAMBDA\n        compare_chained_right   ::= expr COMPARE_OP RETURN_END_IF COME_FROM\n\n        return_if_lambda   ::= RETURN_END_IF_LAMBDA POP_TOP\n        stmt               ::= if_exp_lambda\n        stmt               ::= if_exp_not_lambda\n        if_exp_lambda      ::= expr jmp_false_then expr return_if_lambda\n                               return_stmt_lambda LAMBDA_MARKER\n        if_exp_not_lambda ::=\n                               expr jmp_true_then expr return_if_lambda\n                               return_stmt_lambda LAMBDA_MARKER\n\n        # if_exp_true are for conditions which always evaluate true\n        # There is dead or non-optional remnants of the condition code though,\n        # and we use that to match on to reconstruct the source more accurately\n        expr               ::= if_exp_true\n        if_exp_true        ::= expr jf_pop expr COME_FROM\n\n        # This comes from\n        #   0 or max(5, 3) if 0 else 3\n        # where there seems to be an additional COME_FROM at the\n        # end. Not sure if this is appropriately named or\n        # is the best way to handle\n        expr               ::= if_exp_false\n        if_exp_false  ::= if_exp COME_FROM\n\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            while True:
                i = 10
        self.remove_rules('\n        withasstmt ::= expr SETUP_WITH store suite_stmts_opt\n                POP_BLOCK LOAD_CONST COME_FROM_WITH\n                WITH_CLEANUP END_FINALLY\n        ')
        super(Python26Parser, self).customize_grammar_rules(tokens, customize)
        self.reduce_check_table = {'except_handler': except_handler, 'ifstmt': ifstmt2, 'ifelsestmt': ifelsestmt2, 'tryelsestmt': tryelsestmt, 'try_except': tryexcept, 'tryelsestmtl': tryelsestmt}
        self.check_reduce['and'] = 'AST'
        self.check_reduce['assert_expr_and'] = 'AST'
        self.check_reduce['except_handler'] = 'tokens'
        self.check_reduce['ifstmt'] = 'AST'
        self.check_reduce['ifelsestmt'] = 'AST'
        self.check_reduce['forelselaststmtl'] = 'tokens'
        self.check_reduce['forelsestmt'] = 'tokens'
        self.check_reduce['list_for'] = 'AST'
        self.check_reduce['try_except'] = 'AST'
        self.check_reduce['tryelsestmt'] = 'AST'
        self.check_reduce['tryelsestmtl'] = 'AST'

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            while True:
                i = 10
        invalid = super(Python26Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        lhs = rule[0]
        if invalid or tokens is None:
            return invalid
        if rule in (('and', ('expr', 'jmp_false', 'expr', '\\e_come_from_opt')), ('and', ('expr', 'jmp_false', 'expr', 'come_from_opt')), ('assert_expr_and', ('assert_expr', 'jmp_false', 'expr'))):
            if ast[1] is None:
                return False
            if self.version >= (2, 6) and ast[2][0] == 'if_exp_not':
                return True
            test_index = last
            while tokens[test_index].kind == 'COME_FROM':
                test_index += 1
            if tokens[test_index].kind.startswith('JUMP_IF'):
                return False
            jmp_false = ast[1][0]
            jmp_target = jmp_false.offset + jmp_false.attr + 3
            return not (jmp_target == tokens[test_index].offset or tokens[last].pattr == jmp_false.pattr)
        elif lhs in ('forelselaststmtl', 'forelsestmt'):
            setup_inst = self.insts[self.offset2inst_index[tokens[first].offset]]
            last = min(len(tokens) - 1, last)
            if self.version <= (2, 2) and tokens[last] == 'COME_FROM':
                last += 1
            return tokens[last - 1].off2int() > setup_inst.argval
        elif rule == ('ifstmt', ('testexpr', '_ifstmts_jump')):
            for i in range(last - 1, last - 4, -1):
                t = tokens[i]
                if t == 'JUMP_FORWARD':
                    return t.attr > tokens[min(last, len(tokens) - 1)].off2int()
                elif t not in ('POP_TOP', 'COME_FROM'):
                    break
                pass
            pass
        elif rule == ('list_for', ('expr', 'for_iter', 'store', 'list_iter', 'JUMP_ABSOLUTE', 'come_froms', 'POP_TOP', 'jb_pop')):
            ja_attr = ast[4].attr
            return tokens[last].offset != ja_attr
        elif lhs == 'try_except':
            if last == len(tokens):
                last -= 1
            if tokens[last] != 'COME_FROM' and tokens[last - 1] == 'COME_FROM':
                last -= 1
            if tokens[last] == 'COME_FROM' and tokens[last - 1] == 'END_FINALLY' and (tokens[last - 2] == 'POP_TOP'):
                return tokens[last - 3].kind not in frozenset(('JUMP_FORWARD', 'RETURN_VALUE')) or (tokens[last - 3] == 'JUMP_FORWARD' and tokens[last - 3].attr != 2)
        elif lhs == 'tryelsestmt':
            if ast[3] == 'except_handler':
                except_handler = ast[3]
                if except_handler[0] == 'JUMP_FORWARD':
                    else_start = int(except_handler[0].pattr)
                    if last == len(tokens):
                        last -= 1
                    if tokens[last] == 'COME_FROM' and isinstance:
                        last_offset = int(tokens[last].offset.split('_')[0])
                        return else_start >= last_offset
            if last == len(tokens):
                last -= 1
            while tokens[last - 1] == 'COME_FROM' and tokens[last - 2] == 'COME_FROM':
                last -= 1
            if tokens[last] == 'COME_FROM' and tokens[last - 1] == 'COME_FROM':
                last -= 1
            if tokens[last] == 'COME_FROM' and tokens[last - 1] == 'END_FINALLY' and (tokens[last - 2] == 'POP_TOP'):
                return tokens[last - 3].kind in frozenset(('JUMP_FORWARD', 'RETURN_VALUE')) and (tokens[last - 3] != 'JUMP_FORWARD' or tokens[last - 3].attr == 2)
        return False

class Python26ParserSingle(Python2Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python26Parser()
    p.check_grammar()
    from xdis.version_info import IS_PYPY, PYTHON_VERSION_TRIPLE
    if PYTHON_VERSION_TRIPLE[:2] == (2, 6):
        (lhs, rhs, tokens, right_recursive, dup_rhs) = p.check_sets()
        from uncompyle6.scanner import get_scanner
        s = get_scanner(PYTHON_VERSION_TRIPLE, IS_PYPY)
        opcode_set = set(s.opc.opname).union(set('JUMP_BACK CONTINUE RETURN_END_IF COME_FROM\n               LOAD_GENEXPR LOAD_ASSERT LOAD_SETCOMP LOAD_DICTCOMP\n               LAMBDA_MARKER RETURN_LAST\n            '.split()))
        remain_tokens = set(tokens) - opcode_set
        import re
        remain_tokens = set([re.sub('_\\d+$', '', t) for t in remain_tokens])
        remain_tokens = set([re.sub('_CONT$', '', t) for t in remain_tokens])
        remain_tokens = set(remain_tokens) - opcode_set
        print(remain_tokens)