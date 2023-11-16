"""
spark grammar differences over Python 3.1 for Python 3.0.
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse31 import Python31Parser

class Python30Parser(Python31Parser):

    def p_30(self, args):
        if False:
            return 10
        '\n\n        pt_bp             ::= POP_TOP POP_BLOCK\n\n        assert            ::= assert_expr jmp_true LOAD_ASSERT RAISE_VARARGS_1\n                              COME_FROM POP_TOP\n        assert2           ::= assert_expr jmp_true LOAD_ASSERT expr CALL_FUNCTION_1\n                              RAISE_VARARGS_1 come_froms\n        call_stmt         ::= expr _come_froms POP_TOP\n\n        return_if_lambda       ::= RETURN_END_IF_LAMBDA COME_FROM POP_TOP\n        compare_chained_right  ::= expr COMPARE_OP RETURN_END_IF_LAMBDA\n\n        # FIXME: combine with parse3.2\n        whileTruestmt     ::= SETUP_LOOP l_stmts_opt\n                              jb_or_c COME_FROM_LOOP\n        whileTruestmt     ::= SETUP_LOOP returns\n                              COME_FROM_LOOP\n\n        # In many ways Python 3.0 code generation is more like Python 2.6 than\n        # it is 2.7 or 3.1. So we have a number of 2.6ish (and before) rules below\n        # Specifically POP_TOP is more prevelant since there is no POP_JUMP_IF_...\n        # instructions\n\n        _ifstmts_jump  ::= c_stmts JUMP_FORWARD _come_froms POP_TOP COME_FROM\n        _ifstmts_jump  ::= c_stmts COME_FROM POP_TOP\n\n        # Used to keep index order the same in semantic actions\n        jb_pop_top     ::= JUMP_BACK _come_froms POP_TOP\n\n        while1stmt     ::= SETUP_LOOP l_stmts COME_FROM_LOOP\n        whileelsestmt  ::= SETUP_LOOP testexpr l_stmts\n                           jb_pop_top POP_BLOCK\n                           else_suitel COME_FROM_LOOP\n        # while1elsestmt ::= SETUP_LOOP l_stmts\n        #                    jb_pop_top POP_BLOCK\n        #                    else_suitel COME_FROM_LOOP\n\n        else_suitel ::= l_stmts COME_FROM_LOOP JUMP_BACK\n\n        jump_absolute_else ::= COME_FROM JUMP_ABSOLUTE COME_FROM POP_TOP\n\n        jump_cf_pop   ::= _come_froms _jump  _come_froms POP_TOP\n\n        ifelsestmt  ::= testexpr c_stmts_opt jump_cf_pop else_suite COME_FROM\n        ifelsestmtl ::= testexpr c_stmts_opt jump_cf_pop else_suitel\n        ifelsestmtc ::= testexpr c_stmts_opt jump_absolute_else else_suitec\n        ifelsestmtc ::= testexpr c_stmts_opt jump_cf_pop else_suitec\n\n        iflaststmt  ::= testexpr c_stmts_opt JUMP_ABSOLUTE COME_FROM\n        iflaststmtl ::= testexpr c_stmts_opt jb_pop_top\n        iflaststmtl ::= testexpr c_stmts_opt come_froms JUMP_BACK COME_FROM POP_TOP\n\n        iflaststmt  ::= testexpr c_stmts_opt JUMP_ABSOLUTE COME_FROM POP_TOP\n\n\n        withasstmt    ::= expr setupwithas store suite_stmts_opt\n                          POP_BLOCK LOAD_CONST COME_FROM_FINALLY\n                          LOAD_FAST DELETE_FAST WITH_CLEANUP END_FINALLY\n        setupwithas   ::= DUP_TOP LOAD_ATTR STORE_FAST LOAD_ATTR CALL_FUNCTION_0 setup_finally\n        setup_finally ::= STORE_FAST SETUP_FINALLY LOAD_FAST DELETE_FAST\n\n        # Need to keep LOAD_FAST as index 1\n        set_comp_header  ::= BUILD_SET_0 DUP_TOP STORE_FAST\n\n        set_comp_func ::= set_comp_header\n                          LOAD_ARG FOR_ITER store comp_iter\n                          JUMP_BACK ending_return\n                          RETURN_VALUE RETURN_LAST\n\n        list_comp_header ::= BUILD_LIST_0 DUP_TOP STORE_FAST\n        list_comp        ::= list_comp_header\n                             LOAD_FAST FOR_ITER store comp_iter\n                             JUMP_BACK\n        list_comp        ::= list_comp_header\n                             LOAD_FAST FOR_ITER store comp_iter\n                             JUMP_BACK _come_froms POP_TOP JUMP_BACK\n\n        list_for         ::= DUP_TOP STORE_FAST\n                             expr_or_arg\n                             FOR_ITER\n                             store list_iter jb_or_c\n\n        set_comp         ::= set_comp_header\n                             LOAD_FAST FOR_ITER store comp_iter\n                             JUMP_BACK\n\n        dict_comp_header ::= BUILD_MAP_0 DUP_TOP STORE_FAST\n        dict_comp        ::= dict_comp_header\n                             LOAD_FAST FOR_ITER store dict_comp_iter\n                             JUMP_BACK\n        dict_comp        ::= dict_comp_header\n                             LOAD_FAST FOR_ITER store dict_comp_iter\n                             JUMP_BACK _come_froms POP_TOP JUMP_BACK\n\n        dict_comp_func   ::= BUILD_MAP_0\n                             DUP_TOP STORE_FAST\n                             LOAD_ARG FOR_ITER store\n                             dict_comp_iter JUMP_BACK ending_return\n\n        stmt         ::= try_except30\n        try_except30 ::= SETUP_EXCEPT suite_stmts_opt\n                        _come_froms pt_bp\n                         except_handler opt_come_from_except\n\n        # From Python 2.6\n\n\n\tlc_body     ::= LOAD_FAST expr LIST_APPEND\n        lc_body     ::= LOAD_NAME expr LIST_APPEND\n        list_if     ::= expr jmp_false_then list_iter\n        list_if_not ::= expr jmp_true list_iter JUMP_BACK come_froms POP_TOP\n        list_iter   ::= list_if JUMP_BACK\n        list_iter   ::= list_if JUMP_BACK _come_froms POP_TOP\n\n        #############\n\n        dict_comp_iter   ::= expr expr ROT_TWO expr STORE_SUBSCR\n\n        # JUMP_IF_TRUE POP_TOP as a replacement\n        comp_if       ::= expr jmp_false comp_iter\n        comp_if       ::= expr jmp_false comp_iter JUMP_BACK COME_FROM POP_TOP\n        comp_if_not   ::= expr jmp_true  comp_iter JUMP_BACK COME_FROM POP_TOP\n        comp_iter     ::= expr expr SET_ADD\n        comp_iter     ::= expr expr LIST_APPEND\n\n        jump_forward_else     ::= JUMP_FORWARD COME_FROM POP_TOP\n        jump_absolute_else    ::= JUMP_ABSOLUTE COME_FROM POP_TOP\n        except_suite          ::= c_stmts POP_EXCEPT jump_except POP_TOP\n        except_suite_finalize ::= SETUP_FINALLY c_stmts_opt except_var_finalize END_FINALLY\n                                  _jump COME_FROM POP_TOP\n\n        except_handler        ::= jmp_abs COME_FROM_EXCEPT except_stmts END_FINALLY\n\n        _ifstmts_jump         ::= c_stmts_opt JUMP_FORWARD COME_FROM POP_TOP\n        _ifstmts_jump         ::= c_stmts_opt come_froms POP_TOP JUMP_FORWARD _come_froms\n\n        jump_except           ::= _jump COME_FROM POP_TOP\n\n        expr_jt               ::= expr jmp_true\n        or                    ::= expr jmp_false expr jmp_true expr\n        or                    ::= expr_jt expr\n\n        import_from ::= LOAD_CONST LOAD_CONST IMPORT_NAME importlist _come_froms POP_TOP\n\n        ################################################################################\n        # In many ways 3.0 is like 2.6. One similarity is there is no JUMP_IF_TRUE and\n        # JUMP_IF_FALSE\n        # The below rules in fact are the same or similar.\n\n        jmp_true         ::= JUMP_IF_TRUE POP_TOP\n        jmp_true_then    ::= JUMP_IF_TRUE _come_froms POP_TOP\n        jmp_false        ::= JUMP_IF_FALSE _come_froms POP_TOP\n        jmp_false_then   ::= JUMP_IF_FALSE POP_TOP\n\n        # We don\'t have hacky THEN detection, so we do it\n        # in the grammar below which is also somewhat hacky.\n\n        stmt             ::= ifstmt30\n        stmt             ::= ifnotstmt30\n        ifstmt30         ::= testfalse_then _ifstmts_jump30\n        ifnotstmt30      ::= testtrue_then  _ifstmts_jump30\n\n        testfalse_then   ::= expr jmp_false_then\n        testtrue_then    ::= expr jmp_true_then\n        call_stmt        ::= expr COME_FROM\n        _ifstmts_jump30  ::= c_stmts POP_TOP\n\n        gen_comp_body    ::= expr YIELD_VALUE COME_FROM POP_TOP\n\n        except_handler   ::= jmp_abs COME_FROM_EXCEPT except_stmts\n                             COME_FROM POP_TOP END_FINALLY\n\n        or               ::= expr jmp_true_then expr come_from_opt\n        ret_or           ::= expr jmp_true_then expr come_from_opt\n        ret_and          ::= expr jump_false expr come_from_opt\n\n        ################################################################################\n        for_block      ::= l_stmts_opt _come_froms POP_TOP JUMP_BACK\n\n        except_handler ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts\n                           POP_TOP END_FINALLY come_froms\n        except_handler ::= jmp_abs COME_FROM_EXCEPT except_stmts\n                           POP_TOP END_FINALLY\n\n        return_if_stmt ::= return_expr RETURN_END_IF come_froms POP_TOP\n        return_if_stmt ::= return_expr RETURN_VALUE come_froms POP_TOP\n\n        and            ::= expr jmp_false_then expr come_from_opt\n\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt come_from_opt\n                           JUMP_BACK _come_froms POP_TOP POP_BLOCK COME_FROM_LOOP\n        whilestmt      ::= SETUP_LOOP testexpr returns\n                           POP_TOP POP_BLOCK COME_FROM_LOOP\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt come_from_opt\n                           come_froms POP_TOP POP_BLOCK COME_FROM_LOOP\n\n\n        # A "compare_chained" is two comparisions like x <= y <= z\n        compared_chained_middle  ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                     jmp_false compared_chained_middle _come_froms\n        compared_chained_middle  ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                     jmp_false compare_chained_right _come_froms\n        compare_chained_right ::= expr COMPARE_OP RETURN_END_IF\n        '

    def remove_rules_30(self):
        if False:
            return 10
        self.remove_rules('\n\n        # The were found using grammar coverage\n        while1stmt     ::= SETUP_LOOP l_stmts COME_FROM JUMP_BACK COME_FROM_LOOP\n        whileTruestmt  ::= SETUP_LOOP l_stmts_opt JUMP_BACK POP_BLOCK COME_FROM_LOOP\n        whileelsestmt  ::= SETUP_LOOP testexpr l_stmts_opt JUMP_BACK POP_BLOCK else_suitel COME_FROM_LOOP\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt JUMP_BACK POP_BLOCK COME_FROM_LOOP\n        whilestmt      ::= SETUP_LOOP testexpr l_stmts_opt JUMP_BACK POP_BLOCK JUMP_BACK COME_FROM_LOOP\n        whilestmt      ::= SETUP_LOOP testexpr returns POP_TOP POP_BLOCK COME_FROM_LOOP\n        withasstmt     ::= expr SETUP_WITH store suite_stmts_opt POP_BLOCK LOAD_CONST COME_FROM_WITH WITH_CLEANUP END_FINALLY\n        with           ::= expr SETUP_WITH POP_TOP suite_stmts_opt POP_BLOCK LOAD_CONST COME_FROM_WITH WITH_CLEANUP END_FINALLY\n\n        # lc_body ::= LOAD_FAST expr LIST_APPEND\n        # lc_body ::= LOAD_NAME expr LIST_APPEND\n        # lc_body ::= expr LIST_APPEND\n        # list_comp ::= BUILD_LIST_0 list_iter\n        list_for ::= expr FOR_ITER store list_iter jb_or_c\n        # list_if ::= expr jmp_false list_iter\n        # list_if ::= expr jmp_false_then list_iter\n        # list_if_not ::= expr jmp_true list_iter\n        # list_iter ::= list_if JUMP_BACK\n        # list_iter ::= list_if JUMP_BACK _come_froms POP_TOP\n        # list_iter ::= list_if_not\n        # load_closure ::= BUILD_TUPLE_0\n        # load_genexpr ::= BUILD_TUPLE_1 LOAD_GENEXPR LOAD_STR\n\n        ##########################################################################################\n\n        iflaststmtl        ::= testexpr c_stmts_opt JUMP_BACK COME_FROM_LOOP\n        ifelsestmtl        ::= testexpr c_stmts_opt JUMP_BACK else_suitel\n        iflaststmt         ::= testexpr c_stmts_opt JUMP_ABSOLUTE\n        _ifstmts_jump      ::= c_stmts_opt JUMP_FORWARD _come_froms\n\n        jump_forward_else  ::= JUMP_FORWARD ELSE\n        jump_absolute_else ::= JUMP_ABSOLUTE ELSE\n        whilestmt          ::= SETUP_LOOP testexpr l_stmts_opt COME_FROM JUMP_BACK POP_BLOCK\n                               COME_FROM_LOOP\n        whilestmt          ::= SETUP_LOOP testexpr returns\n                               POP_BLOCK COME_FROM_LOOP\n\n        assert             ::= assert_expr jmp_true LOAD_ASSERT RAISE_VARARGS_1\n\n        return_if_lambda   ::= RETURN_END_IF_LAMBDA\n        except_suite       ::= c_stmts POP_EXCEPT jump_except\n        whileelsestmt      ::= SETUP_LOOP testexpr l_stmts JUMP_BACK POP_BLOCK\n                               else_suitel COME_FROM_LOOP\n\n        ################################################################\n        # No JUMP_IF_FALSE_OR_POP, JUMP_IF_TRUE_OR_POP,\n        # POP_JUMP_IF_FALSE, or POP_JUMP_IF_TRUE\n\n        jmp_false        ::= POP_JUMP_IF_FALSE\n        jmp_true         ::= JUMP_IF_TRUE_OR_POP POP_TOP\n        jmp_true         ::= POP_JUMP_IF_TRUE\n\n        compared_chained_middle ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                    JUMP_IF_FALSE_OR_POP compared_chained_middle\n                                    COME_FROM\n        compared_chained_middle ::= expr DUP_TOP ROT_THREE COMPARE_OP\n                                    JUMP_IF_FALSE_OR_POP compare_chained_right COME_FROM\n        ret_or           ::= expr JUMP_IF_TRUE_OR_POP  return_expr_or_cond COME_FROM\n        ret_and          ::= expr JUMP_IF_FALSE_OR_POP return_expr_or_cond COME_FROM\n        if_exp_ret       ::= expr POP_JUMP_IF_FALSE expr RETURN_END_IF\n                             COME_FROM return_expr_or_cond\n        return_expr_or_cond ::= if_exp_ret\n        or               ::= expr JUMP_IF_TRUE_OR_POP expr COME_FROM\n        and              ::= expr JUMP_IF_TRUE_OR_POP expr COME_FROM\n        and              ::= expr JUMP_IF_FALSE_OR_POP expr COME_FROM\n        ')

    def customize_grammar_rules(self, tokens, customize):
        if False:
            i = 10
            return i + 15
        super(Python30Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules_30()
        self.check_reduce['iflaststmtl'] = 'AST'
        self.check_reduce['ifstmt'] = 'AST'
        self.check_reduce['ifelsestmtc'] = 'AST'
        self.check_reduce['ifelsestmt'] = 'AST'
        return

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            return 10
        invalid = super(Python30Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        if invalid:
            return invalid
        lhs = rule[0]
        if lhs in ('iflaststmtl', 'ifstmt', 'ifelsestmt', 'ifelsestmtc') and ast[0] == 'testexpr':
            testexpr = ast[0]
            if testexpr[0] == 'testfalse':
                testfalse = testexpr[0]
                if lhs == 'ifelsestmtc' and ast[2] == 'jump_absolute_else':
                    jump_absolute_else = ast[2]
                    come_from = jump_absolute_else[2]
                    return come_from == 'COME_FROM' and come_from.attr < tokens[first].offset
                    pass
                elif lhs in ('ifelsestmt', 'ifelsestmtc') and ast[2] == 'jump_cf_pop':
                    jump_cf_pop = ast[2]
                    come_froms = jump_cf_pop[0]
                    for come_from in come_froms:
                        if come_from.attr < tokens[first].offset:
                            return True
                    come_froms = jump_cf_pop[2]
                    if come_froms == 'COME_FROM':
                        if come_froms.attr < tokens[first].offset:
                            return True
                        pass
                    elif come_froms == '_come_froms':
                        for come_from in come_froms:
                            if come_from.attr < tokens[first].offset:
                                return True
                    return False
                elif testfalse[1] == 'jmp_false':
                    jmp_false = testfalse[1]
                    if last == len(tokens):
                        last -= 1
                    while isinstance(tokens[first].offset, str) and first < last:
                        first += 1
                    if first == last:
                        return True
                    while first < last and isinstance(tokens[last].offset, str):
                        last -= 1
                    if rule[0] == 'iflaststmtl':
                        return not jmp_false[0].attr <= tokens[last].offset
                    else:
                        jmp_false_target = jmp_false[0].attr
                        if tokens[first].offset > jmp_false_target:
                            return True
                        return jmp_false_target > tokens[last].offset and tokens[last] != 'JUMP_FORWARD'
                    pass
                pass
            pass
    pass

class Python30ParserSingle(Python30Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python30Parser()
    p.remove_rules_30()
    p.check_grammar()
    from xdis.version_info import PYTHON_VERSION_TRIPLE, IS_PYPY
    if PYTHON_VERSION_TRIPLE[:2] == (3, 0):
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