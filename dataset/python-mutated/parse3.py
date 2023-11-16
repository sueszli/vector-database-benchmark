"""
A spark grammar for Python 3.x.

However instead of terminal symbols being the usual ASCII text,
e.g. 5, myvariable, "for", etc.  they are CPython Bytecode tokens,
e.g. "LOAD_CONST 5", "STORE NAME myvariable", "SETUP_LOOP", etc.

If we succeed in creating a parse tree, then we have a Python program
that a later phase can turn into a sequence of ASCII text.
"""
import re
from uncompyle6.scanners.tok import Token
from uncompyle6.parser import PythonParser, PythonParserSingle, nop_func
from uncompyle6.parsers.reducecheck import and_invalid, except_handler_else, ifelsestmt, ifstmt, iflaststmt, or_check, testtrue, tryelsestmtl3, tryexcept, while1stmt
from uncompyle6.parsers.treenode import SyntaxTree
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG

class Python3Parser(PythonParser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            while True:
                i = 10
        self.added_rules = set()
        super(Python3Parser, self).__init__(SyntaxTree, 'stmts', debug=debug_parser)
        self.new_rules = set()

    def p_comprehension3(self, args):
        if False:
            return 10
        '\n        # Python3 scanner adds LOAD_LISTCOMP. Python3 does list comprehension like\n        # other comprehensions (set, dictionary).\n\n        # Our "continue" heuristic -  in two successive JUMP_BACKS, the first\n        # one may be a continue - sometimes classifies a JUMP_BACK\n        # as a CONTINUE. The two are kind of the same in a comprehension.\n\n        comp_for ::= expr for_iter store comp_iter CONTINUE\n        comp_for ::= expr for_iter store comp_iter JUMP_BACK\n\n        list_comp ::= BUILD_LIST_0 list_iter\n        lc_body   ::= expr LIST_APPEND\n        list_for  ::= expr_or_arg\n                      FOR_ITER\n                      store list_iter jb_or_c\n\n        # This is seen in PyPy, but possibly it appears on other Python 3?\n        list_if     ::= expr jmp_false list_iter COME_FROM\n        list_if_not ::= expr jmp_true list_iter COME_FROM\n\n        jb_or_c ::= JUMP_BACK\n        jb_or_c ::= CONTINUE\n        jb_cfs  ::= JUMP_BACK _come_froms\n\n        stmt ::= set_comp_func\n\n        # TODO this can be simplified\n        set_comp_func ::= BUILD_SET_0 LOAD_ARG FOR_ITER store comp_iter\n                          JUMP_BACK ending_return\n        set_comp_func ::= BUILD_SET_0 LOAD_FAST FOR_ITER store comp_iter\n                          JUMP_BACK ending_return\n        set_comp_func ::= BUILD_SET_0 LOAD_ARG FOR_ITER store comp_iter\n                          COME_FROM JUMP_BACK ending_return\n\n        comp_body ::= dict_comp_body\n        comp_body ::= set_comp_body\n        dict_comp_body ::= expr expr MAP_ADD\n        set_comp_body ::= expr SET_ADD\n\n        expr_or_arg     ::= LOAD_ARG\n        expr_or_arg     ::= expr\n        # See also common Python p_list_comprehension\n        '

    def p_dict_comp3(self, args):
        if False:
            return 10
        '"\n        expr ::= dict_comp\n        stmt ::= dict_comp_func\n        dict_comp_func ::= BUILD_MAP_0 LOAD_ARG FOR_ITER store\n                           comp_iter JUMP_BACK RETURN_VALUE RETURN_LAST\n        dict_comp_func ::= BUILD_MAP_0 LOAD_ARG FOR_ITER store\n                           comp_iter JUMP_BACK RETURN_VALUE_LAMBDA LAMBDA_MARKER\n        dict_comp_func ::= BUILD_MAP_0 LOAD_FAST FOR_ITER store\n                           comp_iter JUMP_BACK RETURN_VALUE RETURN_LAST\n        dict_comp_func ::= BUILD_MAP_0 LOAD_FAST FOR_ITER store\n                           comp_iter JUMP_BACK RETURN_VALUE_LAMBDA LAMBDA_MARKER\n\n        comp_iter     ::= comp_if_not\n        comp_if_not   ::= expr jmp_true comp_iter\n        '

    def p_grammar(self, args):
        if False:
            i = 10
            return i + 15
        '\n        sstmt ::= stmt\n        stmt  ::= ifelsestmtr\n        sstmt ::= return RETURN_LAST\n\n        return_if_stmts ::= return_if_stmt come_from_opt\n        return_if_stmts ::= _stmts return_if_stmt _come_froms\n        return_if_stmt  ::= return_expr RETURN_END_IF\n        returns         ::= _stmts return_if_stmt\n\n        stmt      ::= break\n        break     ::= BREAK_LOOP\n\n        stmt      ::= continue\n        continue  ::= CONTINUE\n        continues ::= _stmts lastl_stmt continue\n        continues ::= lastl_stmt continue\n        continues ::= continue\n\n\n        kwarg      ::= LOAD_STR expr\n        kwargs     ::= kwarg+\n\n        classdef ::= build_class store\n\n        # FIXME: we need to add these because don\'t detect this properly\n        # in custom rules. Specifically if one of the exprs is CALL_FUNCTION\n        # then we\'ll mistake that for the final CALL_FUNCTION.\n        # We can fix by triggering on the CALL_FUNCTION op\n        # Python3 introduced LOAD_BUILD_CLASS\n        # Other definitions are in a custom rule\n        build_class ::= LOAD_BUILD_CLASS mkfunc expr call CALL_FUNCTION_3\n        build_class ::= LOAD_BUILD_CLASS mkfunc expr call expr CALL_FUNCTION_4\n\n        stmt ::= classdefdeco\n        classdefdeco ::= classdefdeco1 store\n\n        expr    ::= LOAD_ASSERT\n        assert  ::= assert_expr jmp_true LOAD_ASSERT RAISE_VARARGS_1 COME_FROM\n        stmt    ::= assert2\n        assert2 ::= assert_expr jmp_true LOAD_ASSERT expr\n                    CALL_FUNCTION_1 RAISE_VARARGS_1 COME_FROM\n\n        assert_expr ::= expr\n        assert_expr ::= assert_expr_or\n        assert_expr ::= assert_expr_and\n        assert_expr_or ::= assert_expr jmp_true expr\n        assert_expr_and ::= assert_expr jmp_false expr\n\n        ifstmt  ::= testexpr _ifstmts_jump\n\n        testexpr ::= testfalse\n        testexpr ::= testtrue\n        testfalse ::= expr jmp_false\n        testtrue ::= expr jmp_true\n\n        _ifstmts_jump   ::= return_if_stmts\n        _ifstmts_jump   ::= stmts _come_froms\n        _ifstmts_jumpl  ::= c_stmts_opt come_froms\n\n        iflaststmt  ::= testexpr stmts_opt JUMP_ABSOLUTE\n        iflaststmt  ::= testexpr _ifstmts_jumpl\n\n        # ifstmts where we are in a loop\n        _ifstmts_jumpl     ::= _ifstmts_jump\n        iflaststmtl ::= testexpr c_stmts_opt JUMP_BACK\n        iflaststmtl ::= testexpr _ifstmts_jumpl\n\n        # These are used to keep parse tree indices the same\n        jump_forward_else  ::= JUMP_FORWARD ELSE\n        jump_absolute_else ::= JUMP_ABSOLUTE ELSE\n\n        # Note: in if/else kinds of statements, we err on the side\n        # of missing "else" clauses. Therefore we include grammar\n        # rules with and without ELSE.\n\n        ifelsestmt ::= testexpr stmts_opt JUMP_FORWARD\n                       else_suite opt_come_from_except\n        ifelsestmt ::= testexpr stmts_opt jump_forward_else\n                       else_suite _come_froms\n\n        # ifelsestmt ::= testexpr c_stmts_opt jump_forward_else\n        #                pass  _come_froms\n\n        # FIXME: remove this\n        stmt         ::= ifelsestmtc\n\n        c_stmts      ::= ifelsestmtc\n\n        ifelsestmtc ::= testexpr c_stmts_opt JUMP_ABSOLUTE else_suitec\n        ifelsestmtc ::= testexpr c_stmts_opt jump_absolute_else else_suitec\n        ifelsestmtc ::= testexpr c_stmts_opt jump_forward_else else_suitec _come_froms\n\n        # "if"/"else" statement that ends in a RETURN\n        ifelsestmtr ::= testexpr return_if_stmts returns\n\n        ifelsestmtl ::= testexpr c_stmts_opt JUMP_BACK else_suitel\n        ifelsestmtl ::= testexpr c_stmts_opt cf_jump_back else_suitel\n        ifelsestmtl ::= testexpr c_stmts_opt continue else_suitel\n\n\n        cf_jump_back ::= COME_FROM JUMP_BACK\n\n        # FIXME: this feels like a hack. Is it just 1 or two\n        # COME_FROMs?  the parsed tree for this and even with just the\n        # one COME_FROM for Python 2.7 seems to associate the\n        # COME_FROM targets from the wrong places\n\n        # this is nested inside a try_except\n        tryfinallystmt ::= SETUP_FINALLY suite_stmts_opt\n                           POP_BLOCK LOAD_CONST\n                           COME_FROM_FINALLY suite_stmts_opt END_FINALLY\n\n        except_handler_else ::= except_handler\n\n        except_handler ::= jmp_abs COME_FROM except_stmts\n                           END_FINALLY\n        except_handler ::= jmp_abs COME_FROM_EXCEPT except_stmts\n                           END_FINALLY\n\n        # FIXME: remove this\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts\n                           END_FINALLY COME_FROM\n\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts\n                           END_FINALLY COME_FROM_EXCEPT\n\n        except_stmts ::= except_stmt+\n\n        except_stmt ::= except_cond1 except_suite\n        except_stmt ::= except_cond2 except_suite\n        except_stmt ::= except_cond2 except_suite_finalize\n        except_stmt ::= except\n\n        ## FIXME: what\'s except_pop_except?\n        except_stmt ::= except_pop_except\n\n        # Python3 introduced POP_EXCEPT\n        except_suite ::= c_stmts_opt POP_EXCEPT jump_except\n        jump_except ::= JUMP_ABSOLUTE\n        jump_except ::= JUMP_BACK\n        jump_except ::= JUMP_FORWARD\n        jump_except ::= CONTINUE\n\n        # This is used in Python 3 in\n        # "except ... as e" to remove \'e\' after the c_stmts_opt finishes\n        except_suite_finalize ::= SETUP_FINALLY c_stmts_opt except_var_finalize\n                                  END_FINALLY _jump\n\n        except_var_finalize ::= POP_BLOCK POP_EXCEPT LOAD_CONST COME_FROM_FINALLY\n                                LOAD_CONST store delete\n\n        except_suite ::= returns\n\n        except_cond1 ::= DUP_TOP expr COMPARE_OP\n                         jmp_false POP_TOP POP_TOP POP_TOP\n\n        except_cond2 ::= DUP_TOP expr COMPARE_OP\n                         jmp_false POP_TOP store POP_TOP\n\n        except  ::=  POP_TOP POP_TOP POP_TOP c_stmts_opt POP_EXCEPT _jump\n        except  ::=  POP_TOP POP_TOP POP_TOP returns\n\n        jmp_abs ::= JUMP_ABSOLUTE\n        jmp_abs ::= JUMP_BACK\n\n        with    ::= expr SETUP_WITH POP_TOP suite_stmts_opt\n                    POP_BLOCK LOAD_CONST COME_FROM_WITH\n                    WITH_CLEANUP END_FINALLY\n\n        withasstmt ::= expr SETUP_WITH store suite_stmts_opt\n                POP_BLOCK LOAD_CONST COME_FROM_WITH\n                WITH_CLEANUP END_FINALLY\n\n        expr_jt     ::= expr jmp_true\n        expr_jitop  ::= expr JUMP_IF_TRUE_OR_POP\n\n        ## FIXME: Right now we have erroneous jump targets\n        ## This below is probably not correct when the COME_FROM is put in the right place\n        and  ::= expr jmp_false expr COME_FROM\n        or   ::= expr_jt  expr COME_FROM\n        or   ::= expr_jt expr\n        or   ::= expr_jitop expr COME_FROM\n        and  ::= expr JUMP_IF_FALSE_OR_POP expr COME_FROM\n\n        # # something like the below is needed when the jump targets are fixed\n        ## or  ::= expr JUMP_IF_TRUE_OR_POP COME_FROM expr\n        ## and ::= expr JUMP_IF_FALSE_OR_POP COME_FROM expr\n        '

    def p_misc3(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        except_handler ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts\n                           END_FINALLY COME_FROM\n        except_handler ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts\n                            END_FINALLY COME_FROM_EXCEPT_CLAUSE\n\n        for_block ::= l_stmts_opt COME_FROM_LOOP JUMP_BACK\n        for_block ::= l_stmts\n        iflaststmtl ::= testexpr c_stmts_opt\n        '

    def p_def_annotations3(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        # Annotated functions\n        stmt                  ::= function_def_annotate\n        function_def_annotate ::= mkfunc_annotate store\n\n        mkfuncdeco0 ::= mkfunc_annotate\n\n        # This has the annotation value.\n        # LOAD_NAME is used in an annotation type like\n        # int, float, str\n        annotate_arg    ::= LOAD_NAME\n        # LOAD_CONST is used in an annotation string\n        annotate_arg    ::= expr\n\n        # This stores the tuple of parameter names\n        # that have been annotated\n        annotate_tuple    ::= LOAD_CONST\n        '

    def p_come_from3(self, args):
        if False:
            print('Hello World!')
        '\n        opt_come_from_except ::= COME_FROM_EXCEPT\n        opt_come_from_except ::= _come_froms\n        opt_come_from_except ::= come_from_except_clauses\n\n        come_from_except_clauses ::= COME_FROM_EXCEPT_CLAUSE+\n        '

    def p_jump3(self, args):
        if False:
            print('Hello World!')
        '\n        jmp_false ::= POP_JUMP_IF_FALSE\n        jmp_true  ::= POP_JUMP_IF_TRUE\n\n        # FIXME: Common with 2.7\n        ret_and    ::= expr JUMP_IF_FALSE_OR_POP return_expr_or_cond COME_FROM\n        ret_or     ::= expr JUMP_IF_TRUE_OR_POP return_expr_or_cond COME_FROM\n        if_exp_ret ::= expr POP_JUMP_IF_FALSE expr RETURN_END_IF COME_FROM\n                       return_expr_or_cond\n\n\n        # compared_chained_middle is used exclusively in chained_compare\n        compared_chained_middle ::= expr DUP_TOP ROT_THREE COMPARE_OP JUMP_IF_FALSE_OR_POP\n                                    compared_chained_middle COME_FROM\n        compared_chained_middle ::= expr DUP_TOP ROT_THREE COMPARE_OP JUMP_IF_FALSE_OR_POP\n                                    compare_chained_right COME_FROM\n        '

    def p_stmt3(self, args):
        if False:
            print('Hello World!')
        '\n        stmt               ::= if_exp_lambda\n\n        stmt               ::= if_exp_not_lambda\n        if_exp_lambda      ::= expr jmp_false expr return_if_lambda\n                               return_stmt_lambda LAMBDA_MARKER\n        if_exp_not_lambda  ::= expr jmp_true expr return_if_lambda\n                               return_stmt_lambda LAMBDA_MARKER\n\n        return_stmt_lambda ::= return_expr RETURN_VALUE_LAMBDA\n        return_if_lambda   ::= RETURN_END_IF_LAMBDA\n\n        stmt ::= return_closure\n        return_closure ::= LOAD_CLOSURE RETURN_VALUE RETURN_LAST\n\n        stmt ::= whileTruestmt\n        ifelsestmt ::= testexpr c_stmts_opt JUMP_FORWARD else_suite _come_froms\n\n        # FIXME: go over this\n        _stmts ::= _stmts last_stmt\n        stmts ::= last_stmt\n        stmts_opt ::= stmts\n        last_stmt ::= iflaststmt\n        last_stmt ::= forelselaststmt\n        iflaststmt ::= testexpr last_stmt JUMP_ABSOLUTE\n        iflaststmt ::= testexpr stmts JUMP_ABSOLUTE\n\n        _iflaststmts_jump ::= stmts last_stmt\n        _ifstmts_jump ::= stmts_opt JUMP_FORWARD _come_froms\n\n        iflaststmt ::= testexpr _iflaststmts_jump\n        ifelsestmt ::= testexpr stmts_opt jump_absolute_else else_suite\n        ifelsestmt ::= testexpr stmts_opt jump_forward_else else_suite _come_froms\n        else_suite ::= stmts\n        else_suitel ::= stmts\n\n        # FIXME: remove this\n        _ifstmts_jump ::= c_stmts_opt JUMP_FORWARD _come_froms\n\n\n        # statements with continue and break\n        c_stmts ::= _stmts\n        c_stmts ::= _stmts lastc_stmt\n        c_stmts ::= lastc_stmt\n        c_stmts ::= continues\n\n        lastc_stmt ::= iflaststmtl\n        lastc_stmt ::= forelselaststmt\n        lastc_stmt ::= ifelsestmtc\n\n        # Statements in a loop\n        lstmt              ::= stmt\n        l_stmts            ::= lstmt+\n        '

    def p_loop_stmt3(self, args):
        if False:
            while True:
                i = 10
        '\n        stmt ::= whileelsestmt2\n\n        for               ::= SETUP_LOOP expr for_iter store for_block POP_BLOCK\n                              COME_FROM_LOOP\n\n        forelsestmt       ::= SETUP_LOOP expr for_iter store for_block POP_BLOCK\n                              else_suite COME_FROM_LOOP\n\n        forelselaststmt   ::= SETUP_LOOP expr for_iter store for_block POP_BLOCK\n                              else_suitec COME_FROM_LOOP\n\n        forelselaststmtl  ::= SETUP_LOOP expr for_iter store for_block POP_BLOCK\n                              else_suitel COME_FROM_LOOP\n\n        whilestmt         ::= SETUP_LOOP testexpr l_stmts_opt COME_FROM JUMP_BACK\n                              POP_BLOCK COME_FROM_LOOP\n\n        whilestmt         ::= SETUP_LOOP testexpr l_stmts_opt JUMP_BACK POP_BLOCK\n                              JUMP_BACK COME_FROM_LOOP\n        whilestmt         ::= SETUP_LOOP testexpr l_stmts_opt JUMP_BACK POP_BLOCK\n                              COME_FROM_LOOP\n\n        whilestmt         ::= SETUP_LOOP testexpr returns               POP_BLOCK\n                              COME_FROM_LOOP\n\n        while1elsestmt    ::= SETUP_LOOP          l_stmts     JUMP_BACK\n                              else_suitel\n\n        whileelsestmt     ::= SETUP_LOOP testexpr l_stmts_opt jb_cfs POP_BLOCK\n                              else_suitel COME_FROM_LOOP\n\n\n        whileelsestmt2    ::= SETUP_LOOP testexpr l_stmts_opt  JUMP_BACK POP_BLOCK\n                              else_suitel JUMP_BACK COME_FROM_LOOP\n\n        whileTruestmt     ::= SETUP_LOOP l_stmts_opt          JUMP_BACK POP_BLOCK\n                              COME_FROM_LOOP\n\n        # FIXME: Python 3.? starts adding branch optimization? Put this starting there.\n\n        while1stmt        ::= SETUP_LOOP l_stmts COME_FROM_LOOP\n        while1stmt        ::= SETUP_LOOP l_stmts COME_FROM JUMP_BACK COME_FROM_LOOP\n\n        while1elsestmt    ::= SETUP_LOOP l_stmts JUMP_BACK\n                              else_suite COME_FROM_LOOP\n\n        # FIXME: investigate - can code really produce a NOP?\n        whileTruestmt     ::= SETUP_LOOP l_stmts_opt JUMP_BACK NOP\n                              COME_FROM_LOOP\n        whileTruestmt     ::= SETUP_LOOP l_stmts_opt JUMP_BACK POP_BLOCK NOP\n                              COME_FROM_LOOP\n        for               ::= SETUP_LOOP expr for_iter store for_block POP_BLOCK NOP\n                              COME_FROM_LOOP\n        '

    def p_generator_exp3(self, args):
        if False:
            while True:
                i = 10
        '\n        load_genexpr ::= LOAD_GENEXPR\n        load_genexpr ::= BUILD_TUPLE_1 LOAD_GENEXPR LOAD_STR\n        '

    def p_expr3(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        expr           ::= LOAD_STR\n        expr           ::= if_exp_not\n        if_exp_not     ::= expr jmp_true  expr jump_forward_else expr COME_FROM\n\n        # a JUMP_FORWARD to another JUMP_FORWARD can get turned into\n        # a JUMP_ABSOLUTE with no COME_FROM\n        if_exp         ::= expr jmp_false expr jump_absolute_else expr\n\n        # if_exp_true are for conditions which always evaluate true\n        # There is dead or non-optional remnants of the condition code though,\n        # and we use that to match on to reconstruct the source more accurately\n        expr           ::= if_exp_true\n        if_exp_true    ::= expr JUMP_FORWARD expr COME_FROM\n        '

    @staticmethod
    def call_fn_name(token):
        if False:
            print('Hello World!')
        'Customize CALL_FUNCTION to add the number of positional arguments'
        if token.attr is not None:
            return '%s_%i' % (token.kind, token.attr)
        else:
            return '%s_0' % token.kind

    def custom_build_class_rule(self, opname, i, token, tokens, customize, is_pypy):
        if False:
            i = 10
            return i + 15
        '\n        # Should the first rule be somehow folded into the 2nd one?\n        build_class ::= LOAD_BUILD_CLASS mkfunc\n                        LOAD_CLASSNAME {expr}^n-1 CALL_FUNCTION_n\n                        LOAD_CONST CALL_FUNCTION_n\n        build_class ::= LOAD_BUILD_CLASS mkfunc\n                        expr\n                        call\n                        CALL_FUNCTION_3\n         '
        for i in range(i + 1, len(tokens)):
            if tokens[i].kind.startswith('MAKE_FUNCTION'):
                break
            elif tokens[i].kind.startswith('MAKE_CLOSURE'):
                break
            pass
        assert i < len(tokens), 'build_class needs to find MAKE_FUNCTION or MAKE_CLOSURE'
        assert tokens[i + 1].kind == 'LOAD_STR', 'build_class expecting CONST after MAKE_FUNCTION/MAKE_CLOSURE'
        call_fn_tok = None
        for i in range(i, len(tokens)):
            if tokens[i].kind.startswith('CALL_FUNCTION'):
                call_fn_tok = tokens[i]
                break
        if not call_fn_tok:
            raise RuntimeError('build_class custom rule for %s needs to find CALL_FUNCTION' % opname)
        if self.version < (3, 6):
            call_function = self.call_fn_name(call_fn_tok)
            (pos_args_count, kw_args_count) = self.get_pos_kw(call_fn_tok)
            rule = 'build_class ::= LOAD_BUILD_CLASS mkfunc %s%s' % ('expr ' * (pos_args_count - 1) + 'kwarg ' * kw_args_count, call_function)
        else:
            call_function = call_fn_tok.kind
            if call_function.startswith('CALL_FUNCTION_KW'):
                self.addRule('classdef ::= build_class_kw store', nop_func)
                if is_pypy:
                    (pos_args_count, kw_args_count) = self.get_pos_kw(call_fn_tok)
                    rule = 'build_class_kw ::= LOAD_BUILD_CLASS mkfunc %s%s%s' % ('expr ' * (pos_args_count - 1), 'kwarg ' * kw_args_count, call_function)
                else:
                    rule = 'build_class_kw ::= LOAD_BUILD_CLASS mkfunc %sLOAD_CONST %s' % ('expr ' * (call_fn_tok.attr - 1), call_function)
            else:
                call_function = self.call_fn_name(call_fn_tok)
                rule = 'build_class ::= LOAD_BUILD_CLASS mkfunc %s%s' % ('expr ' * (call_fn_tok.attr - 1), call_function)
        self.addRule(rule, nop_func)
        return

    def custom_classfunc_rule(self, opname, token, customize, next_token, is_pypy):
        if False:
            print('Hello World!')
        '\n        call ::= expr {expr}^n CALL_FUNCTION_n\n        call ::= expr {expr}^n CALL_FUNCTION_VAR_n\n        call ::= expr {expr}^n CALL_FUNCTION_VAR_KW_n\n        call ::= expr {expr}^n CALL_FUNCTION_KW_n\n\n        classdefdeco2 ::= LOAD_BUILD_CLASS mkfunc {expr}^n-1 CALL_FUNCTION_n\n        '
        (pos_args_count, kw_args_count) = self.get_pos_kw(token)
        nak = (len(opname) - len('CALL_FUNCTION')) // 3
        uniq_param = kw_args_count + pos_args_count
        if is_pypy and self.version >= (3, 6):
            if token == 'CALL_FUNCTION':
                token.kind = self.call_fn_name(token)
            rule = 'call ::= expr ' + 'pos_arg ' * pos_args_count + 'kwarg ' * kw_args_count + token.kind
        else:
            token.kind = self.call_fn_name(token)
            rule = 'call ::= expr ' + 'pos_arg ' * pos_args_count + 'kwarg ' * kw_args_count + 'expr ' * nak + token.kind
        self.add_unique_rule(rule, token.kind, uniq_param, customize)
        if 'LOAD_BUILD_CLASS' in self.seen_ops:
            if next_token == 'CALL_FUNCTION' and next_token.attr == 1 and (pos_args_count > 1):
                rule = 'classdefdeco2 ::= LOAD_BUILD_CLASS mkfunc %s%s_%d' % ('expr ' * (pos_args_count - 1), opname, pos_args_count)
                self.add_unique_rule(rule, token.kind, uniq_param, customize)

    def add_make_function_rule(self, rule, opname, attr, customize):
        if False:
            return 10
        'Python 3.3 added a an addtional LOAD_STR before MAKE_FUNCTION and\n        this has an effect on many rules.\n        '
        if self.version >= (3, 3):
            load_op = 'LOAD_STR '
            new_rule = rule % (load_op * 1)
        else:
            new_rule = rule % ('LOAD_STR ' * 0)
        self.add_unique_rule(new_rule, opname, attr, customize)

    def customize_grammar_rules(self, tokens, customize):
        if False:
            print('Hello World!')
        "The base grammar we start out for a Python version even with the\n        subclassing is, well, is pretty base.  And we want it that way: lean and\n        mean so that parsing will go faster.\n\n        Here, we add additional grammar rules based on specific instructions\n        that are in the instruction/token stream. In classes that\n        inherit from from here and other versions, grammar rules may\n        also be removed.\n\n        For example if we see a pretty rare DELETE_DEREF instruction we'll\n        add the grammar for that.\n\n        More importantly, here we add grammar rules for instructions\n        that may access a variable number of stack items. CALL_FUNCTION,\n        BUILD_LIST and so on are like this.\n\n        Without custom rules, there can be an super-exponential number of\n        derivations. See the deparsing paper for an elaboration of\n        this.\n\n        "
        self.is_pypy = False
        customize_instruction_basenames = frozenset(('BUILD', 'CALL', 'CONTINUE', 'DELETE', 'GET', 'JUMP', 'LOAD', 'LOOKUP', 'MAKE', 'RETURN', 'RAISE', 'SETUP', 'UNPACK', 'WITH'))
        custom_ops_processed = set(('BUILD_TUPLE_UNPACK_WITH_CALL',))
        self.seen_ops = frozenset([t.kind for t in tokens])
        self.seen_op_basenames = frozenset([opname[:opname.rfind('_')] for opname in self.seen_ops])
        if 'PyPy' in customize:
            self.is_pypy = True
            self.addRule('\n              stmt ::= assign3_pypy\n              stmt ::= assign2_pypy\n              assign3_pypy       ::= expr expr expr store store store\n              assign2_pypy       ::= expr expr store store\n              stmt               ::= if_exp_lambda\n              stmt               ::= if_exp_not_lambda\n              if_expr_lambda     ::= expr jmp_false expr return_if_lambda\n                                     return_expr_lambda LAMBDA_MARKER\n              if_exp_not_lambda  ::= expr jmp_true expr return_if_lambda\n                                     return_expr_lambda LAMBDA_MARKER\n              ', nop_func)
        n = len(tokens)
        has_get_iter_call_function1 = False
        for (i, token) in enumerate(tokens):
            if token == 'GET_ITER' and i < n - 2 and (self.call_fn_name(tokens[i + 1]) == 'CALL_FUNCTION_1'):
                has_get_iter_call_function1 = True
        for (i, token) in enumerate(tokens):
            opname = token.kind
            if opname[:opname.find('_')] not in customize_instruction_basenames or opname in custom_ops_processed:
                continue
            opname_base = opname[:opname.rfind('_')]
            if opname_base == 'BUILD_CONST_KEY_MAP':
                kvlist_n = 'expr ' * token.attr
                rule = 'dict ::= %sLOAD_CONST %s' % (kvlist_n, opname)
                self.addRule(rule, nop_func)
            elif opname in ('BUILD_CONST_LIST', 'BUILD_CONST_DICT', 'BUILD_CONST_SET'):
                if opname == 'BUILD_CONST_DICT':
                    rule = '\n                           add_consts          ::= ADD_VALUE*\n                           const_list          ::= COLLECTION_START add_consts %s\n                           dict                ::= const_list\n                           expr                ::= dict\n                           ' % opname
                else:
                    rule = '\n                           add_consts          ::= ADD_VALUE*\n                           const_list          ::= COLLECTION_START add_consts %s\n                           expr                ::= const_list\n                           ' % opname
                self.addRule(rule, nop_func)
            elif opname.startswith('BUILD_DICT_OLDER'):
                rule = 'dict ::= COLLECTION_START key_value_pairs BUILD_DICT_OLDER\n                          key_value_pairs ::= key_value_pair+\n                          key_value_pair  ::= ADD_KEY ADD_VALUE\n                       '
                self.addRule(rule, nop_func)
            elif opname.startswith('BUILD_LIST_UNPACK'):
                v = token.attr
                rule = 'build_list_unpack ::= %s%s' % ('expr ' * v, opname)
                self.addRule(rule, nop_func)
                rule = 'expr ::= build_list_unpack'
                self.addRule(rule, nop_func)
            elif opname_base in ('BUILD_MAP', 'BUILD_MAP_UNPACK'):
                kvlist_n = 'kvlist_%s' % token.attr
                if opname == 'BUILD_MAP_n':
                    rule = 'dict_comp_func ::= BUILD_MAP_n LOAD_FAST FOR_ITER store comp_iter JUMP_BACK RETURN_VALUE RETURN_LAST'
                    self.add_unique_rule(rule, 'dict_comp_func', 1, customize)
                    kvlist_n = 'kvlist_n'
                    rule = 'kvlist_n ::=  kvlist_n kv3'
                    self.add_unique_rule(rule, 'kvlist_n', 0, customize)
                    rule = 'kvlist_n ::='
                    self.add_unique_rule(rule, 'kvlist_n', 1, customize)
                    rule = 'dict ::=  BUILD_MAP_n kvlist_n'
                elif self.version >= (3, 5):
                    if not opname.startswith('BUILD_MAP_WITH_CALL'):
                        if opname.startswith('BUILD_MAP_UNPACK'):
                            if 'LOAD_DICTCOMP' in self.seen_ops:
                                rule = 'dict ::= %s%s' % ('dict_comp ' * token.attr, opname)
                                self.addRule(rule, nop_func)
                            rule = '\n                             expr        ::= dict_unpack\n                             dict_unpack ::= %s%s\n                             ' % ('expr ' * token.attr, opname)
                        else:
                            rule = '%s ::= %s %s' % (kvlist_n, 'expr ' * (token.attr * 2), opname)
                            self.add_unique_rule(rule, opname, token.attr, customize)
                            rule = 'dict ::=  %s' % kvlist_n
                else:
                    rule = kvlist_n + ' ::= ' + 'expr expr STORE_MAP ' * token.attr
                    self.add_unique_rule(rule, opname, token.attr, customize)
                    rule = 'dict ::=  %s %s' % (opname, kvlist_n)
                self.add_unique_rule(rule, opname, token.attr, customize)
            elif opname.startswith('BUILD_MAP_UNPACK_WITH_CALL'):
                v = token.attr
                rule = 'build_map_unpack_with_call ::= %s%s' % ('expr ' * v, opname)
                self.addRule(rule, nop_func)
            elif opname.startswith('BUILD_TUPLE_UNPACK_WITH_CALL'):
                v = token.attr
                rule = 'starred ::= %s %s' % ('expr ' * v, opname)
                self.addRule(rule, nop_func)
            elif opname in ('BUILD_CONST_LIST', 'BUILD_CONST_DICT', 'BUILD_CONST_SET'):
                if opname == 'BUILD_CONST_DICT':
                    rule = '\n                           add_consts          ::= ADD_VALUE*\n                           const_list          ::= COLLECTION_START add_consts %s\n                           dict                ::= const_list\n                           expr                ::= dict\n                           ' % opname
                else:
                    rule = '\n                           add_consts          ::= ADD_VALUE*\n                           const_list          ::= COLLECTION_START add_consts %s\n                           expr                ::= const_list\n                           ' % opname
                self.addRule(rule, nop_func)
            elif opname_base in ('BUILD_LIST', 'BUILD_SET', 'BUILD_TUPLE', 'BUILD_TUPLE_UNPACK'):
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
                if not is_LOAD_CLOSURE or v == 0:
                    build_count = token.attr
                    thousands = build_count // 1024
                    thirty32s = build_count // 32 % 32
                    if thirty32s > 0 or thousands > 0:
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
            elif opname_base == 'BUILD_SLICE':
                if token.attr == 2:
                    self.add_unique_rules(['expr ::= build_slice2', 'build_slice2 ::= expr expr BUILD_SLICE_2'], customize)
                else:
                    assert token.attr == 3, 'BUILD_SLICE value must be 2 or 3; is %s' % v
                    self.add_unique_rules(['expr ::= build_slice3', 'build_slice3 ::= expr expr expr BUILD_SLICE_3'], customize)
            elif opname in frozenset(('CALL_FUNCTION', 'CALL_FUNCTION_EX', 'CALL_FUNCTION_EX_KW', 'CALL_FUNCTION_VAR', 'CALL_FUNCTION_VAR_KW')) or opname.startswith('CALL_FUNCTION_KW'):
                if opname == 'CALL_FUNCTION' and token.attr == 1:
                    rule = '\n                     dict_comp    ::= LOAD_DICTCOMP LOAD_STR MAKE_FUNCTION_0 expr\n                                      GET_ITER CALL_FUNCTION_1\n                    classdefdeco1 ::= expr classdefdeco2 CALL_FUNCTION_1\n                    classdefdeco1 ::= expr classdefdeco1 CALL_FUNCTION_1\n                    '
                    self.addRule(rule, nop_func)
                self.custom_classfunc_rule(opname, token, customize, tokens[i + 1], self.is_pypy)
            elif opname_base == 'CALL_METHOD':
                (pos_args_count, kw_args_count) = self.get_pos_kw(token)
                nak = (len(opname_base) - len('CALL_METHOD')) // 3
                rule = 'call ::= expr ' + 'pos_arg ' * pos_args_count + 'kwarg ' * kw_args_count + 'expr ' * nak + opname
                self.add_unique_rule(rule, opname, token.attr, customize)
            elif opname == 'CONTINUE':
                self.addRule('continue ::= CONTINUE', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'CONTINUE_LOOP':
                self.addRule('continue ::= CONTINUE_LOOP', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'DELETE_ATTR':
                self.addRule('delete ::= expr DELETE_ATTR', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'DELETE_DEREF':
                self.addRule('\n                   stmt           ::= del_deref_stmt\n                   del_deref_stmt ::= DELETE_DEREF\n                   ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'DELETE_SUBSCR':
                self.addRule('\n                    delete ::= delete_subscript\n                    delete_subscript ::= expr expr DELETE_SUBSCR\n                   ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'GET_ITER':
                self.addRule('\n                    expr      ::= get_iter\n                    get_iter ::= expr GET_ITER\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'JUMP_IF_NOT_DEBUG':
                v = token.attr
                self.addRule('\n                    stmt        ::= assert_pypy\n                    stmt        ::= assert_not_pypy\n                    stmt        ::= assert2_pypy\n                    stmt        ::= assert2_not_pypy\n                    assert_pypy ::=  JUMP_IF_NOT_DEBUG assert_expr jmp_true\n                                     LOAD_ASSERT RAISE_VARARGS_1 COME_FROM\n                    assert_not_pypy ::=  JUMP_IF_NOT_DEBUG assert_expr jmp_false\n                                     LOAD_ASSERT RAISE_VARARGS_1 COME_FROM\n                    assert2_pypy ::= JUMP_IF_NOT_DEBUG assert_expr jmp_true\n                                     LOAD_ASSERT expr CALL_FUNCTION_1\n                                     RAISE_VARARGS_1 COME_FROM\n                    assert2_pypy ::= JUMP_IF_NOT_DEBUG assert_expr jmp_true\n                                     LOAD_ASSERT expr CALL_FUNCTION_1\n                                     RAISE_VARARGS_1 COME_FROM\n                    assert2_not_pypy ::= JUMP_IF_NOT_DEBUG assert_expr jmp_false\n                                     LOAD_ASSERT expr CALL_FUNCTION_1\n                                     RAISE_VARARGS_1 COME_FROM\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_BUILD_CLASS':
                self.custom_build_class_rule(opname, i, token, tokens, customize, self.is_pypy)
            elif opname == 'LOAD_CLASSDEREF':
                self.addRule('expr ::= LOAD_CLASSDEREF', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_CLASSNAME':
                self.addRule('expr ::= LOAD_CLASSNAME', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_DICTCOMP':
                if has_get_iter_call_function1:
                    rule_pat = 'dict_comp ::= LOAD_DICTCOMP %sMAKE_FUNCTION_0 expr GET_ITER CALL_FUNCTION_1'
                    self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                    pass
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_ATTR':
                self.addRule('\n                  expr      ::= attribute\n                  attribute ::= expr LOAD_ATTR\n                  ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_LISTCOMP':
                self.add_unique_rule('expr ::= listcomp', opname, token.attr, customize)
                custom_ops_processed.add(opname)
            elif opname == 'LOAD_SETCOMP':
                if has_get_iter_call_function1:
                    self.addRule('expr ::= set_comp', nop_func)
                    rule_pat = 'set_comp ::= LOAD_SETCOMP %sMAKE_FUNCTION_0 expr GET_ITER CALL_FUNCTION_1'
                    self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                    pass
                custom_ops_processed.add(opname)
            elif opname == 'LOOKUP_METHOD':
                self.addRule('\n                    attribute ::= expr LOOKUP_METHOD\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname.startswith('MAKE_CLOSURE'):
                if opname == 'MAKE_CLOSURE_0' and 'LOAD_DICTCOMP' in self.seen_ops:
                    rule = '\n                        dict_comp ::= load_closure LOAD_DICTCOMP LOAD_STR\n                                      MAKE_CLOSURE_0 expr\n                                      GET_ITER CALL_FUNCTION_1\n                    '
                    self.addRule(rule, nop_func)
                (pos_args_count, kw_args_count, annotate_args) = token.attr
                if self.version < (3, 3):
                    j = 1
                else:
                    j = 2
                if self.is_pypy or (i >= j and tokens[i - j] == 'LOAD_LAMBDA'):
                    rule_pat = 'lambda_body ::= %sload_closure LOAD_LAMBDA %%s%s' % ('pos_arg ' * pos_args_count, opname)
                    self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                if has_get_iter_call_function1:
                    rule_pat = 'generator_exp ::= %sload_closure load_genexpr %%s%s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname)
                    self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                    if has_get_iter_call_function1:
                        if self.is_pypy or (i >= j and tokens[i - j] == 'LOAD_LISTCOMP'):
                            rule_pat = 'listcomp ::= %sload_closure LOAD_LISTCOMP %%s%s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname)
                            self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                        if self.is_pypy or (i >= j and tokens[i - j] == 'LOAD_SETCOMP'):
                            rule_pat = 'set_comp ::= %sload_closure LOAD_SETCOMP %%s%s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname)
                            self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                        if self.is_pypy or (i >= j and tokens[i - j] == 'LOAD_DICTCOMP'):
                            self.add_unique_rule('dict_comp ::= %sload_closure LOAD_DICTCOMP %s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname), opname, token.attr, customize)
                if kw_args_count > 0:
                    kwargs_str = 'kwargs '
                else:
                    kwargs_str = ''
                if self.version <= (3, 2):
                    if annotate_args > 0:
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple load_closure LOAD_CODE %s' % (kwargs_str, 'pos_arg ' * pos_args_count, 'annotate_arg ' * annotate_args, opname)
                    else:
                        rule = 'mkfunc ::= %s%sload_closure LOAD_CODE %s' % (kwargs_str, 'pos_arg ' * pos_args_count, opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                elif (3, 3) <= self.version < (3, 6):
                    if annotate_args > 0:
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple load_closure LOAD_CODE LOAD_STR %s' % (kwargs_str, 'pos_arg ' * pos_args_count, 'annotate_arg ' * annotate_args, opname)
                    else:
                        rule = 'mkfunc ::= %s%sload_closure LOAD_CODE LOAD_STR %s' % (kwargs_str, 'pos_arg ' * pos_args_count, opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                if self.version >= (3, 4):
                    if not self.is_pypy:
                        load_op = 'LOAD_STR'
                    else:
                        load_op = 'LOAD_CONST'
                    if annotate_args > 0:
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple load_closure %s %s' % ('pos_arg ' * pos_args_count, kwargs_str, 'annotate_arg ' * annotate_args, load_op, opname)
                    else:
                        rule = 'mkfunc ::= %s%s load_closure LOAD_CODE %s %s' % ('pos_arg ' * pos_args_count, kwargs_str, load_op, opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                if kw_args_count == 0:
                    rule = 'mkfunc ::= %sload_closure load_genexpr %s' % ('pos_arg ' * pos_args_count, opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                if self.version < (3, 4):
                    rule = 'mkfunc ::= %sload_closure LOAD_CODE %s' % ('expr ' * pos_args_count, opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                pass
            elif opname_base.startswith('MAKE_FUNCTION'):
                if self.version >= (3, 6):
                    (pos_args_count, kw_args_count, annotate_args, closure) = token.attr
                    stack_count = pos_args_count + kw_args_count + annotate_args
                    if closure:
                        if pos_args_count:
                            rule = 'lambda_body ::= %s%s%s%s' % ('expr ' * stack_count, 'load_closure ' * closure, 'BUILD_TUPLE_1 LOAD_LAMBDA LOAD_STR ', opname)
                        else:
                            rule = 'lambda_body ::= %s%s%s' % ('load_closure ' * closure, 'LOAD_LAMBDA LOAD_STR ', opname)
                        self.add_unique_rule(rule, opname, token.attr, customize)
                    else:
                        rule = 'lambda_body ::= %sLOAD_LAMBDA LOAD_STR %s' % ('expr ' * stack_count, opname)
                        self.add_unique_rule(rule, opname, token.attr, customize)
                    rule = 'mkfunc ::= %s%s%s%s' % ('expr ' * stack_count, 'load_closure ' * closure, 'LOAD_CODE LOAD_STR ', opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                    if has_get_iter_call_function1:
                        rule_pat = 'generator_exp ::= %sload_genexpr %%s%s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname)
                        self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                        rule_pat = 'generator_exp ::= %sload_closure load_genexpr %%s%s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname)
                        self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                        if self.is_pypy or (i >= 2 and tokens[i - 2] == 'LOAD_LISTCOMP'):
                            if self.version >= (3, 6):
                                rule_pat = 'listcomp ::= load_closure LOAD_LISTCOMP %%s%s expr GET_ITER CALL_FUNCTION_1' % (opname,)
                                self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                            rule_pat = 'listcomp ::= %sLOAD_LISTCOMP %%s%s expr GET_ITER CALL_FUNCTION_1' % ('expr ' * pos_args_count, opname)
                            self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                    if self.is_pypy or (i >= 2 and tokens[i - 2] == 'LOAD_LAMBDA'):
                        rule_pat = 'lambda_body ::= %s%sLOAD_LAMBDA %%s%s' % ('pos_arg ' * pos_args_count, 'kwarg ' * kw_args_count, opname)
                        self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                    continue
                if self.version < (3, 6):
                    (pos_args_count, kw_args_count, annotate_args) = token.attr
                else:
                    (pos_args_count, kw_args_count, annotate_args, closure) = token.attr
                if self.version < (3, 3):
                    j = 1
                else:
                    j = 2
                if has_get_iter_call_function1:
                    rule_pat = 'generator_exp ::= %sload_genexpr %%s%s expr GET_ITER CALL_FUNCTION_1' % ('pos_arg ' * pos_args_count, opname)
                    self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                    if self.is_pypy or (i >= j and tokens[i - j] == 'LOAD_LISTCOMP'):
                        rule_pat = 'listcomp ::= %sLOAD_LISTCOMP %%s%s expr GET_ITER CALL_FUNCTION_1' % ('expr ' * pos_args_count, opname)
                        self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                if self.is_pypy or (i >= j and tokens[i - j] == 'LOAD_LAMBDA'):
                    rule_pat = 'lambda_body ::= %s%sLOAD_LAMBDA %%s%s' % ('pos_arg ' * pos_args_count, 'kwarg ' * kw_args_count, opname)
                    self.add_make_function_rule(rule_pat, opname, token.attr, customize)
                if kw_args_count == 0:
                    kwargs = 'no_kwargs'
                    self.add_unique_rule('no_kwargs ::=', opname, token.attr, customize)
                else:
                    kwargs = 'kwargs'
                if self.version < (3, 3):
                    rule = 'mkfunc ::= %s %s%s%s' % (kwargs, 'pos_arg ' * pos_args_count, 'LOAD_CODE ', opname)
                    self.add_unique_rule(rule, opname, token.attr, customize)
                    rule = 'mkfunc ::= %s%s%s' % ('pos_arg ' * pos_args_count, 'LOAD_CODE ', opname)
                elif self.version == (3, 3):
                    rule = 'mkfunc ::= %s %s%s%s' % (kwargs, 'pos_arg ' * pos_args_count, 'LOAD_CODE LOAD_STR ', opname)
                elif self.version >= (3, 6):
                    rule = 'mkfunc ::= %s%s %s%s' % ('pos_arg ' * pos_args_count, kwargs, 'LOAD_CODE LOAD_STR ', opname)
                elif self.version >= (3, 4):
                    rule = 'mkfunc ::= %s%s %s%s' % ('pos_arg ' * pos_args_count, kwargs, 'LOAD_CODE LOAD_STR ', opname)
                else:
                    rule = 'mkfunc ::= %s%sexpr %s' % (kwargs, 'pos_arg ' * pos_args_count, opname)
                self.add_unique_rule(rule, opname, token.attr, customize)
                if re.search('^MAKE_FUNCTION.*_A', opname):
                    if self.version >= (3, 6):
                        rule = 'mkfunc_annotate ::= %s%sannotate_tuple LOAD_CODE LOAD_STR %s' % ('pos_arg ' * pos_args_count, 'call ' * annotate_args, opname)
                        self.add_unique_rule(rule, opname, token.attr, customize)
                        rule = 'mkfunc_annotate ::= %s%sannotate_tuple LOAD_CODE LOAD_STR %s' % ('pos_arg ' * pos_args_count, 'annotate_arg ' * annotate_args, opname)
                    if self.version >= (3, 3):
                        if self.version == (3, 3):
                            pos_kw_tuple = ('kwargs ' * kw_args_count, 'pos_arg ' * pos_args_count)
                        else:
                            pos_kw_tuple = ('pos_arg ' * pos_args_count, 'kwargs ' * kw_args_count)
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple LOAD_CODE LOAD_STR EXTENDED_ARG %s' % (pos_kw_tuple[0], pos_kw_tuple[1], 'call ' * annotate_args, opname)
                        self.add_unique_rule(rule, opname, token.attr, customize)
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple LOAD_CODE LOAD_STR EXTENDED_ARG %s' % (pos_kw_tuple[0], pos_kw_tuple[1], 'annotate_arg ' * annotate_args, opname)
                    else:
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple LOAD_CODE EXTENDED_ARG %s' % ('kwargs ' * kw_args_count, 'pos_arg ' * pos_args_count, 'annotate_arg ' * annotate_args, opname)
                        self.add_unique_rule(rule, opname, token.attr, customize)
                        rule = 'mkfunc_annotate ::= %s%s%sannotate_tuple LOAD_CODE EXTENDED_ARG %s' % ('kwargs ' * kw_args_count, 'pos_arg ' * pos_args_count, 'call ' * annotate_args, opname)
                    self.addRule(rule, nop_func)
            elif opname == 'RETURN_VALUE_LAMBDA':
                self.addRule('\n                    return_expr_lambda ::= return_expr RETURN_VALUE_LAMBDA\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'RAISE_VARARGS_0':
                self.addRule('\n                    stmt        ::= raise_stmt0\n                    raise_stmt0 ::= RAISE_VARARGS_0\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'RAISE_VARARGS_1':
                self.addRule('\n                    stmt        ::= raise_stmt1\n                    raise_stmt1 ::= expr RAISE_VARARGS_1\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'RAISE_VARARGS_2':
                self.addRule('\n                    stmt        ::= raise_stmt2\n                    raise_stmt2 ::= expr expr RAISE_VARARGS_2\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'SETUP_EXCEPT':
                self.addRule('\n                    try_except     ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                       except_handler opt_come_from_except\n                    try_except     ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                       except_handler opt_come_from_except\n\n                    tryelsestmtl   ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                       except_handler else_suitel come_from_except_clauses\n\n                    stmt             ::= tryelsestmtl3\n\n                    tryelsestmtl3    ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                         except_handler_else COME_FROM else_suitel\n                                         opt_come_from_except\n                    tryelsestmt      ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                                         except_handler_else else_suite come_froms\n                    ', nop_func)
                custom_ops_processed.add(opname)
            elif opname_base in ('UNPACK_EX',):
                (before_count, after_count) = token.attr
                rule = 'unpack ::= ' + opname + ' store' * (before_count + after_count + 1)
                self.addRule(rule, nop_func)
            elif opname_base in ('UNPACK_TUPLE', 'UNPACK_SEQUENCE'):
                rule = 'unpack ::= ' + opname + ' store' * token.attr
                self.addRule(rule, nop_func)
            elif opname_base == 'UNPACK_LIST':
                rule = 'unpack_list ::= ' + opname + ' store' * token.attr
                self.addRule(rule, nop_func)
                custom_ops_processed.add(opname)
                pass
            pass
        self.reduce_check_table = {'except_handler_else': except_handler_else, 'ifstmtl': ifstmt, 'ifelsestmtc': ifelsestmt, 'ifelsestmt': ifelsestmt, 'or': or_check, 'testtrue': testtrue, 'tryelsestmtl3': tryelsestmtl3, 'try_except': tryexcept}
        if self.version == (3, 6):
            self.reduce_check_table['and'] = and_invalid
            self.check_reduce['and'] = 'AST'
        self.check_reduce['annotate_tuple'] = 'noAST'
        self.check_reduce['aug_assign1'] = 'AST'
        self.check_reduce['aug_assign2'] = 'AST'
        self.check_reduce['except_handler_else'] = 'tokens'
        self.check_reduce['ifelsestmt'] = 'AST'
        self.check_reduce['ifelsestmtc'] = 'AST'
        self.check_reduce['ifstmt'] = 'AST'
        self.check_reduce['ifstmtl'] = 'AST'
        if self.version == (3, 6):
            self.reduce_check_table['iflaststmtl'] = iflaststmt
            self.check_reduce['iflaststmt'] = 'AST'
            self.check_reduce['iflaststmtl'] = 'AST'
        self.check_reduce['or'] = 'AST'
        self.check_reduce['testtrue'] = 'tokens'
        if self.version < (3, 6) and (not self.is_pypy):
            self.check_reduce['try_except'] = 'AST'
        self.check_reduce['tryelsestmtl3'] = 'AST'
        self.check_reduce['while1stmt'] = 'noAST'
        self.check_reduce['while1elsestmt'] = 'noAST'
        return

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            return 10
        lhs = rule[0]
        n = len(tokens)
        last = min(last, n - 1)
        fn = self.reduce_check_table.get(lhs, None)
        if fn:
            if fn(self, lhs, n, rule, ast, tokens, first, last):
                return True
            pass
        if lhs in ('aug_assign1', 'aug_assign2') and ast[0][0] == 'and':
            return True
        elif lhs == 'annotate_tuple':
            return not isinstance(tokens[first].attr, tuple)
        elif lhs == 'kwarg':
            arg = tokens[first].attr
            return not (isinstance(arg, str) or isinstance(arg, unicode))
        elif rule == ('ifstmt', ('testexpr', '_ifstmts_jump')):
            if self.version <= (3, 0) or tokens[last] == 'RETURN_END_IF':
                return False
            if ifstmt(self, lhs, n, rule, ast, tokens, first, last):
                return True
            condition_jump = ast[0].last_child()
            if condition_jump.kind.startswith('POP_JUMP_IF'):
                condition_jump2 = tokens[min(last - 1, len(tokens) - 1)]
                if condition_jump2.kind.startswith('POP_JUMP_IF') and condition_jump != condition_jump2:
                    return condition_jump.attr == condition_jump2.attr
                if tokens[last] == 'COME_FROM' and tokens[last].off2int() != condition_jump.attr:
                    return False
                return condition_jump.attr < condition_jump2.off2int()
            return False
        elif rule == ('ifstmt', ('testexpr', '\\e__ifstmts_jump')):
            return True
        elif lhs == 'ifelsestmt' and rule[1][2] == 'jump_forward_else':
            last = min(last, len(tokens) - 1)
            if tokens[last].off2int() == -1:
                last -= 1
            jump_forward_else = ast[2]
            return tokens[first].off2int() <= jump_forward_else[0].attr < tokens[last].off2int()
        elif lhs == 'while1stmt':
            if while1stmt(self, lhs, n, rule, ast, tokens, first, last):
                return True
            if self.version == (3, 0):
                return False
            if 0 <= last < len(tokens) and tokens[last] in ('COME_FROM_LOOP', 'JUMP_BACK'):
                last += 1
            while last < len(tokens) and isinstance(tokens[last].offset, str):
                last += 1
            if last < len(tokens):
                offset = tokens[last].offset
                assert tokens[first] == 'SETUP_LOOP'
                if offset != tokens[first].attr:
                    return True
            return False
        elif lhs == 'while1elsestmt':
            n = len(tokens)
            if last == n:
                last -= 1
            if tokens[last] == 'COME_FROM_LOOP':
                last -= 1
            elif tokens[last - 1] == 'COME_FROM_LOOP':
                last -= 2
            if tokens[last] in ('JUMP_BACK', 'CONTINUE'):
                return True
            last += 1
            while last < n and isinstance(tokens[last].offset, str):
                last += 1
            if last == n:
                return False
            return self.version < (3, 8) and tokens[first].attr > tokens[last].offset
        elif rule == ('ifelsestmt', ('testexpr', 'c_stmts_opt', 'jump_forward_else', 'else_suite', '_come_froms')):
            come_froms = ast[-1]
            if not isinstance(come_froms, Token):
                return tokens[first].offset > come_froms[-1].attr
            return False
        return False

class Python30Parser(Python3Parser):

    def p_30(self, args):
        if False:
            return 10
        '\n        jmp_true ::= JUMP_IF_TRUE_OR_POP POP_TOP\n        _ifstmts_jump ::= c_stmts_opt JUMP_FORWARD POP_TOP COME_FROM\n        '

class Python3ParserSingle(Python3Parser, PythonParserSingle):
    pass

def info(args):
    if False:
        for i in range(10):
            print('nop')
    p = Python3Parser()
    if len(args) > 0:
        arg = args[0]
        if arg == '3.5':
            from uncompyle6.parser.parse35 import Python35Parser
            p = Python35Parser()
        elif arg == '3.3':
            from uncompyle6.parser.parse33 import Python33Parser
            p = Python33Parser()
        elif arg == '3.2':
            from uncompyle6.parser.parse32 import Python32Parser
            p = Python32Parser()
        elif arg == '3.0':
            p = Python30Parser()
    p.check_grammar()
    if len(sys.argv) > 1 and sys.argv[1] == 'dump':
        print('-' * 50)
        p.dump_grammar()
if __name__ == '__main__':
    import sys
    info(sys.argv)