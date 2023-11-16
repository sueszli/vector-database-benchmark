"""
spark grammar differences over Python 3.5 for Python 3.6.
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle, nop_func
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parsers.parse35 import Python35Parser
from uncompyle6.scanners.tok import Token

class Python36Parser(Python35Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            print('Hello World!')
        super(Python36Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_36_jump(self, args):
        if False:
            i = 10
            return i + 15
        '\n        # Zero or one COME_FROM\n        # And/or expressions have this\n        come_from_opt ::= COME_FROM?\n        '

    def p_36_misc(self, args):
        if False:
            print('Hello World!')
        'sstmt ::= sstmt RETURN_LAST\n\n        # long except clauses in a loop can sometimes cause a JUMP_BACK to turn into a\n        # JUMP_FORWARD to a JUMP_BACK. And when this happens there is an additional\n        # ELSE added to the except_suite. With better flow control perhaps we can\n        # sort this out better.\n        except_suite ::= c_stmts_opt POP_EXCEPT jump_except ELSE\n        except_suite_finalize ::= SETUP_FINALLY c_stmts_opt except_var_finalize END_FINALLY\n                                  _jump ELSE\n\n        # 3.6 redoes how return_closure works. FIXME: Isolate to LOAD_CLOSURE\n        return_closure   ::= LOAD_CLOSURE DUP_TOP STORE_NAME RETURN_VALUE RETURN_LAST\n\n        for_block       ::= l_stmts_opt come_from_loops JUMP_BACK\n        come_from_loops ::= COME_FROM_LOOP*\n\n        whilestmt       ::= SETUP_LOOP testexpr l_stmts_opt\n                            JUMP_BACK come_froms POP_BLOCK COME_FROM_LOOP\n        whilestmt       ::= SETUP_LOOP testexpr l_stmts_opt\n                            come_froms JUMP_BACK come_froms POP_BLOCK COME_FROM_LOOP\n\n        # 3.6 due to jump optimization, we sometimes add RETURN_END_IF where\n        # RETURN_VALUE is meant. Specifcally this can happen in\n        # ifelsestmt -> ...else_suite _. suite_stmts... (last) stmt\n        return             ::= return_expr RETURN_END_IF\n        return             ::= return_expr RETURN_VALUE COME_FROM\n        return_stmt_lambda ::= return_expr RETURN_VALUE_LAMBDA COME_FROM\n\n        # A COME_FROM is dropped off because of JUMP-to-JUMP optimization\n        and  ::= expr jmp_false expr\n        and  ::= expr jmp_false expr jmp_false\n\n        jf_cf       ::= JUMP_FORWARD COME_FROM\n        cf_jf_else  ::= come_froms JUMP_FORWARD ELSE\n\n        if_exp ::= expr jmp_false expr jf_cf expr COME_FROM\n\n        async_for_stmt36   ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM\n                               SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK JUMP_BACK COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                               END_FINALLY for_block\n                               COME_FROM\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP POP_BLOCK\n                               COME_FROM_LOOP\n\n        async_for_stmt36   ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                               END_FINALLY\n                               COME_FROM\n                               for_block\n                               COME_FROM\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP POP_BLOCK\n                               COME_FROM_LOOP\n\n        async_for_stmt     ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                               JUMP_ABSOLUTE END_FINALLY COME_FROM\n                               for_block POP_BLOCK\n                               COME_FROM_LOOP\n\n        stmt      ::= async_for_stmt36\n        stmt      ::= async_forelse_stmt36\n\n        async_forelse_stmt ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                               JUMP_ABSOLUTE END_FINALLY COME_FROM\n                               for_block POP_BLOCK\n                               else_suite COME_FROM_LOOP\n\n        async_forelse_stmt36 ::= SETUP_LOOP expr\n                               GET_AITER\n                               LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                               YIELD_FROM\n                               store\n                               POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                               LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                               END_FINALLY COME_FROM\n                               for_block _come_froms\n                               POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP\n                               POP_BLOCK\n                               else_suite COME_FROM_LOOP\n\n        # Adds a COME_FROM_ASYNC_WITH over 3.5\n        # FIXME: remove corresponding rule for 3.5?\n\n        except_suite ::= c_stmts_opt COME_FROM POP_EXCEPT jump_except COME_FROM\n\n        jb_cfs      ::= JUMP_BACK come_froms\n\n        # If statement inside a loop.\n        stmt                ::= ifstmtl\n        ifstmtl            ::= testexpr _ifstmts_jumpl\n        _ifstmts_jumpl     ::= c_stmts JUMP_BACK\n\n        ifelsestmtl ::= testexpr c_stmts_opt jb_cfs else_suitel\n        ifelsestmtl ::= testexpr c_stmts_opt cf_jf_else else_suitel\n        ifelsestmt  ::= testexpr c_stmts_opt cf_jf_else else_suite _come_froms\n        ifelsestmt  ::= testexpr c_stmts come_froms else_suite come_froms\n\n        # In 3.6+, A sequence of statements ending in a RETURN can cause\n        # JUMP_FORWARD END_FINALLY to be omitted from try middle\n\n        except_return    ::= POP_TOP POP_TOP POP_TOP returns\n        except_handler   ::= JUMP_FORWARD COME_FROM_EXCEPT except_return\n\n        # Try middle following a returns\n        except_handler36 ::= COME_FROM_EXCEPT except_stmts END_FINALLY\n\n        stmt             ::= try_except36\n        try_except36     ::= SETUP_EXCEPT returns except_handler36\n                             opt_come_from_except\n        try_except36     ::= SETUP_EXCEPT suite_stmts\n        try_except36     ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                             except_handler36 opt_come_from_except\n\n        # 3.6 omits END_FINALLY sometimes\n        except_handler36 ::= COME_FROM_EXCEPT except_stmts\n        except_handler36 ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts\n        except_handler   ::= jmp_abs COME_FROM_EXCEPT except_stmts\n\n        stmt             ::= tryfinally36\n        tryfinally36     ::= SETUP_FINALLY returns\n                             COME_FROM_FINALLY suite_stmts\n        tryfinally36     ::= SETUP_FINALLY returns\n                             COME_FROM_FINALLY suite_stmts_opt END_FINALLY\n        except_suite_finalize ::= SETUP_FINALLY returns\n                                  COME_FROM_FINALLY suite_stmts_opt END_FINALLY _jump\n\n        stmt ::= tryfinally_return_stmt\n        tryfinally_return_stmt ::= SETUP_FINALLY suite_stmts_opt POP_BLOCK LOAD_CONST\n                                   COME_FROM_FINALLY\n\n        compare_chained_right ::= expr COMPARE_OP come_froms JUMP_FORWARD\n        '

    def p_36_conditionals(self, args):
        if False:
            i = 10
            return i + 15
        '\n        expr                       ::= if_exp37\n        if_exp37                   ::= expr expr jf_cfs expr COME_FROM\n        jf_cfs                     ::= JUMP_FORWARD _come_froms\n        ifelsestmt                 ::= testexpr c_stmts_opt jf_cfs else_suite opt_come_from_except\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            return 10
        super(Python36Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules('\n           _ifstmts_jumpl     ::= c_stmts_opt\n           _ifstmts_jumpl     ::= _ifstmts_jump\n           except_handler     ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts END_FINALLY COME_FROM\n           async_for_stmt     ::= SETUP_LOOP expr\n                                  GET_AITER\n                                  LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                                  YIELD_FROM\n                                  store\n                                  POP_BLOCK jump_except COME_FROM_EXCEPT DUP_TOP\n                                  LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                                  POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                                  JUMP_ABSOLUTE END_FINALLY COME_FROM\n                                  for_block POP_BLOCK JUMP_ABSOLUTE\n                                  COME_FROM_LOOP\n           async_forelse_stmt ::= SETUP_LOOP expr\n                                  GET_AITER\n                                  LOAD_CONST YIELD_FROM SETUP_EXCEPT GET_ANEXT LOAD_CONST\n                                  YIELD_FROM\n                                  store\n                                  POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT DUP_TOP\n                                  LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_FALSE\n                                  POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_BLOCK\n                                  JUMP_ABSOLUTE END_FINALLY COME_FROM\n                                  for_block pb_ja\n                                  else_suite COME_FROM_LOOP\n\n        ')
        self.check_reduce['call_kw'] = 'AST'
        custom_ops_processed = set()
        for (i, token) in enumerate(tokens):
            opname = token.kind
            if opname == 'FORMAT_VALUE':
                rules_str = '\n                    expr              ::= formatted_value1\n                    formatted_value1  ::= expr FORMAT_VALUE\n                '
                self.add_unique_doc_rules(rules_str, customize)
            elif opname == 'FORMAT_VALUE_ATTR':
                rules_str = '\n                expr              ::= formatted_value2\n                formatted_value2  ::= expr expr FORMAT_VALUE_ATTR\n                '
                self.add_unique_doc_rules(rules_str, customize)
            elif opname == 'MAKE_FUNCTION_CLOSURE':
                if 'LOAD_DICTCOMP' in self.seen_ops:
                    rule = '\n                       dict_comp ::= load_closure LOAD_DICTCOMP LOAD_STR\n                                     MAKE_FUNCTION_CLOSURE expr\n                                     GET_ITER CALL_FUNCTION_1\n                       '
                    self.addRule(rule, nop_func)
                elif 'LOAD_SETCOMP' in self.seen_ops:
                    rule = '\n                       set_comp ::= load_closure LOAD_SETCOMP LOAD_STR\n                                    MAKE_FUNCTION_CLOSURE expr\n                                    GET_ITER CALL_FUNCTION_1\n                       '
                    self.addRule(rule, nop_func)
            elif opname == 'BEFORE_ASYNC_WITH':
                rules_str = '\n                  stmt ::= async_with_stmt\n                  async_with_pre     ::= BEFORE_ASYNC_WITH GET_AWAITABLE LOAD_CONST YIELD_FROM SETUP_ASYNC_WITH\n                  async_with_post    ::= COME_FROM_ASYNC_WITH\n                                         WITH_CLEANUP_START GET_AWAITABLE LOAD_CONST YIELD_FROM\n                                         WITH_CLEANUP_FINISH END_FINALLY\n                  async_with_as_stmt ::= expr\n                               async_with_pre\n                               store\n                               suite_stmts_opt\n                               POP_BLOCK LOAD_CONST\n                               async_with_post\n                 stmt ::= async_with_as_stmt\n                 async_with_stmt ::= expr\n                               POP_TOP\n                               suite_stmts_opt\n                               POP_BLOCK LOAD_CONST\n                               async_with_post\n                 async_with_stmt ::= expr\n                               POP_TOP\n                               suite_stmts_opt\n                               async_with_post\n                '
                self.addRule(rules_str, nop_func)
            elif opname.startswith('BUILD_STRING'):
                v = token.attr
                rules_str = '\n                    expr                 ::= joined_str\n                    joined_str           ::= %sBUILD_STRING_%d\n                ' % ('expr ' * v, v)
                self.add_unique_doc_rules(rules_str, customize)
                if 'FORMAT_VALUE_ATTR' in self.seen_ops:
                    rules_str = '\n                      formatted_value_attr ::= expr expr FORMAT_VALUE_ATTR expr BUILD_STRING\n                      expr                 ::= formatted_value_attr\n                    '
                    self.add_unique_doc_rules(rules_str, customize)
            elif opname.startswith('BUILD_MAP_UNPACK_WITH_CALL'):
                v = token.attr
                rule = 'build_map_unpack_with_call ::= %s%s' % ('expr ' * v, opname)
                self.addRule(rule, nop_func)
            elif opname.startswith('BUILD_TUPLE_UNPACK_WITH_CALL'):
                v = token.attr
                rule = 'build_tuple_unpack_with_call ::= ' + 'expr1024 ' * int(v // 1024) + 'expr32 ' * int(v // 32 % 32) + 'expr ' * (v % 32) + opname
                self.addRule(rule, nop_func)
                rule = 'starred ::= %s %s' % ('expr ' * v, opname)
                self.addRule(rule, nop_func)
            elif opname == 'GET_AITER':
                self.addRule('\n                    expr                ::= generator_exp_async\n\n                    generator_exp_async ::= load_genexpr LOAD_STR MAKE_FUNCTION_0 expr\n                                            GET_AITER LOAD_CONST YIELD_FROM CALL_FUNCTION_1\n                    stmt                ::= genexpr_func_async\n\n                    func_async_prefix   ::= _come_froms\n                                            LOAD_CONST YIELD_FROM\n                                            SETUP_EXCEPT GET_ANEXT LOAD_CONST YIELD_FROM\n                    func_async_middle   ::= POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT\n                                            DUP_TOP LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                                            END_FINALLY COME_FROM\n                    genexpr_func_async  ::= LOAD_ARG func_async_prefix\n                                            store func_async_middle comp_iter\n                                            JUMP_BACK\n                                            POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP\n\n                    expr                ::= list_comp_async\n                    list_comp_async     ::= LOAD_LISTCOMP LOAD_STR MAKE_FUNCTION_0\n                                            expr GET_AITER\n                                            LOAD_CONST YIELD_FROM CALL_FUNCTION_1\n                                            GET_AWAITABLE LOAD_CONST\n                                            YIELD_FROM\n\n                    expr                ::= list_comp_async\n                    list_afor2          ::= func_async_prefix\n                                            store func_async_middle list_iter\n                                            JUMP_BACK\n                                            POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP\n                    list_comp_async     ::= BUILD_LIST_0 LOAD_ARG list_afor2\n                    get_aiter           ::= expr GET_AITER\n                    list_afor           ::= get_aiter list_afor2\n                    list_iter           ::= list_afor\n                   ', nop_func)
            elif opname == 'GET_AITER':
                self.add_unique_doc_rules('get_aiter ::= expr GET_AITER', customize)
                if not {'MAKE_FUNCTION_0', 'MAKE_FUNCTION_CLOSURE'} in self.seen_ops:
                    self.addRule("\n                        expr                ::= dict_comp_async\n                        expr                ::= generator_exp_async\n                        expr                ::= list_comp_async\n\n                        dict_comp_async     ::= LOAD_DICTCOMP\n                                                LOAD_STR\n                                                MAKE_FUNCTION_0\n                                                get_aiter\n                                                CALL_FUNCTION_1\n\n                        dict_comp_async     ::= BUILD_MAP_0 LOAD_ARG\n                                                dict_comp_async\n\n                        func_async_middle   ::= POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT\n                                                DUP_TOP LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                                                END_FINALLY COME_FROM\n\n                        func_async_prefix   ::= _come_froms SETUP_EXCEPT GET_ANEXT LOAD_CONST YIELD_FROM\n\n                        generator_exp_async ::= load_genexpr LOAD_STR MAKE_FUNCTION_0\n                                                get_aiter CALL_FUNCTION_1\n\n                        genexpr_func_async  ::= LOAD_ARG func_async_prefix\n                                                store func_async_middle comp_iter\n                                                JUMP_LOOP COME_FROM\n                                                POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP\n\n                        # FIXME this is a workaround for probalby some bug in the Earley parser\n                        # if we use get_aiter, then list_comp_async doesn't match, and I don't\n                        # understand why.\n                        expr_get_aiter      ::= expr GET_AITER\n\n                        list_afor           ::= get_aiter list_afor2\n\n                        list_afor2          ::= func_async_prefix\n                                                store func_async_middle list_iter\n                                                JUMP_LOOP COME_FROM\n                                                POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP\n\n                        list_comp_async     ::= BUILD_LIST_0 LOAD_ARG list_afor2\n                        list_comp_async     ::= LOAD_LISTCOMP LOAD_STR MAKE_FUNCTION_0\n                                                expr_get_aiter CALL_FUNCTION_1\n                                                GET_AWAITABLE LOAD_CONST\n                                                YIELD_FROM\n\n                        list_iter           ::= list_afor\n\n                        set_comp_async       ::= LOAD_SETCOMP\n                                                 LOAD_STR\n                                                 MAKE_FUNCTION_0\n                                                 get_aiter\n                                                 CALL_FUNCTION_1\n\n                        set_comp_async       ::= LOAD_CLOSURE\n                                                 BUILD_TUPLE_1\n                                                 LOAD_SETCOMP\n                                                 LOAD_STR MAKE_FUNCTION_CLOSURE\n                                                 get_aiter CALL_FUNCTION_1\n                                                 await\n                       ", nop_func)
                    custom_ops_processed.add(opname)
                self.addRule('\n                    dict_comp_async      ::= BUILD_MAP_0 LOAD_ARG\n                                             dict_comp_async\n\n                    expr                 ::= dict_comp_async\n                    expr                 ::= generator_exp_async\n                    expr                 ::= list_comp_async\n                    expr                 ::= set_comp_async\n\n                    func_async_middle   ::= POP_BLOCK JUMP_FORWARD COME_FROM_EXCEPT\n                                            DUP_TOP LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                                            END_FINALLY _come_froms\n\n                    get_aiter            ::= expr GET_AITER\n\n                    list_afor            ::= get_aiter list_afor2\n\n                    list_comp_async      ::= BUILD_LIST_0 LOAD_ARG list_afor2\n                    list_iter            ::= list_afor\n\n\n                    set_afor             ::= get_aiter set_afor2\n                    set_iter             ::= set_afor\n                    set_iter             ::= set_for\n\n                    set_comp_async       ::= BUILD_SET_0 LOAD_ARG\n                                             set_comp_async\n\n                   ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'GET_ANEXT':
                self.addRule('\n                    func_async_prefix   ::= _come_froms SETUP_EXCEPT GET_ANEXT LOAD_CONST YIELD_FROM\n                    func_async_prefix   ::= _come_froms SETUP_FINALLY GET_ANEXT LOAD_CONST YIELD_FROM POP_BLOCK\n                    func_async_prefix   ::= _come_froms\n                                            LOAD_CONST YIELD_FROM\n                                            SETUP_EXCEPT GET_ANEXT LOAD_CONST YIELD_FROM\n                    func_async_middle   ::= JUMP_FORWARD COME_FROM_EXCEPT\n                                            DUP_TOP LOAD_GLOBAL COMPARE_OP POP_JUMP_IF_TRUE\n                    list_comp_async     ::= BUILD_LIST_0 LOAD_ARG list_afor2\n                    list_afor2          ::= func_async_prefix\n                                            store list_iter\n                                            JUMP_BACK COME_FROM_FINALLY\n                                            END_ASYNC_FOR\n                    list_afor2          ::= func_async_prefix\n                                            store func_async_middle list_iter\n                                            JUMP_LOOP COME_FROM\n                                            POP_TOP POP_TOP POP_TOP POP_EXCEPT POP_TOP\n                   ', nop_func)
                custom_ops_processed.add(opname)
            elif opname == 'SETUP_ANNOTATIONS':
                rule = '\n                    stmt ::= SETUP_ANNOTATIONS\n                    stmt ::= ann_assign_init_value\n                    stmt ::= ann_assign_no_init\n\n                    ann_assign_init_value ::= expr store store_annotation\n                    ann_assign_no_init    ::= store_annotation\n                    store_annotation      ::= LOAD_NAME STORE_ANNOTATION\n                    store_annotation      ::= subscript STORE_ANNOTATION\n                 '
                self.addRule(rule, nop_func)
                self.check_reduce['assign'] = 'token'
            elif opname == 'WITH_CLEANUP_START':
                rules_str = '\n                  stmt        ::= with_null\n                  with_null   ::= with_suffix\n                  with_suffix ::= WITH_CLEANUP_START WITH_CLEANUP_FINISH END_FINALLY\n                '
                self.addRule(rules_str, nop_func)
            elif opname == 'SETUP_WITH':
                rules_str = '\n                  with       ::= expr SETUP_WITH POP_TOP suite_stmts_opt COME_FROM_WITH\n                                 with_suffix\n\n                  # Removes POP_BLOCK LOAD_CONST from 3.6-\n                  withasstmt ::= expr SETUP_WITH store suite_stmts_opt COME_FROM_WITH\n                                 with_suffix\n                  with       ::= expr SETUP_WITH POP_TOP suite_stmts_opt POP_BLOCK\n                                 BEGIN_FINALLY COME_FROM_WITH\n                                 with_suffix\n                '
                self.addRule(rules_str, nop_func)
                pass
            pass
        return

    def custom_classfunc_rule(self, opname, token, customize, next_token, is_pypy):
        if False:
            for i in range(10):
                print('nop')
        (args_pos, args_kw) = self.get_pos_kw(token)
        nak = (len(opname) - len('CALL_FUNCTION')) // 3
        uniq_param = args_kw + args_pos
        if frozenset(('GET_AWAITABLE', 'YIELD_FROM')).issubset(self.seen_ops):
            rule = 'async_call ::= expr ' + 'pos_arg ' * args_pos + 'kwarg ' * args_kw + 'expr ' * nak + token.kind + ' GET_AWAITABLE LOAD_CONST YIELD_FROM'
            self.add_unique_rule(rule, token.kind, uniq_param, customize)
            self.add_unique_rule('expr ::= async_call', token.kind, uniq_param, customize)
        if opname.startswith('CALL_FUNCTION_KW'):
            if is_pypy:
                super(Python36Parser, self).custom_classfunc_rule(opname, token, customize, next_token, is_pypy)
            else:
                self.addRule('expr ::= call_kw36', nop_func)
                values = 'expr ' * token.attr
                rule = 'call_kw36 ::= expr {values} LOAD_CONST {opname}'.format(**locals())
                self.add_unique_rule(rule, token.kind, token.attr, customize)
        elif opname == 'CALL_FUNCTION_EX_KW':
            self.addRule('expr        ::= call_ex_kw4\n                            call_ex_kw4 ::= expr\n                                            expr\n                                            expr\n                                            CALL_FUNCTION_EX_KW\n                         ', nop_func)
            if 'BUILD_MAP_UNPACK_WITH_CALL' in self.seen_op_basenames:
                self.addRule('expr        ::= call_ex_kw\n                                call_ex_kw  ::= expr expr build_map_unpack_with_call\n                                                CALL_FUNCTION_EX_KW\n                             ', nop_func)
            if 'BUILD_TUPLE_UNPACK_WITH_CALL' in self.seen_op_basenames:
                self.addRule('expr        ::= call_ex_kw3\n                                call_ex_kw3 ::= expr\n                                                build_tuple_unpack_with_call\n                                                expr\n                                                CALL_FUNCTION_EX_KW\n                             ', nop_func)
                if 'BUILD_MAP_UNPACK_WITH_CALL' in self.seen_op_basenames:
                    self.addRule('expr        ::= call_ex_kw2\n                                    call_ex_kw2 ::= expr\n                                                    build_tuple_unpack_with_call\n                                                    build_map_unpack_with_call\n                                                    CALL_FUNCTION_EX_KW\n                             ', nop_func)
        elif opname == 'CALL_FUNCTION_EX':
            self.addRule('\n                         expr        ::= call_ex\n                         starred     ::= expr\n                         call_ex     ::= expr starred CALL_FUNCTION_EX\n                         ', nop_func)
            if self.version >= (3, 6):
                if 'BUILD_MAP_UNPACK_WITH_CALL' in self.seen_ops:
                    self.addRule('\n                            expr        ::= call_ex_kw\n                            call_ex_kw  ::= expr expr\n                                            build_map_unpack_with_call CALL_FUNCTION_EX\n                            ', nop_func)
                if 'BUILD_TUPLE_UNPACK_WITH_CALL' in self.seen_ops:
                    self.addRule('\n                            expr        ::= call_ex_kw3\n                            call_ex_kw3 ::= expr\n                                            build_tuple_unpack_with_call\n                                            %s\n                                            CALL_FUNCTION_EX\n                            ' % 'expr ' * token.attr, nop_func)
                    pass
                self.addRule('\n                            expr        ::= call_ex_kw4\n                            call_ex_kw4 ::= expr\n                                            expr\n                                            expr\n                                            CALL_FUNCTION_EX\n                            ', nop_func)
            pass
        else:
            super(Python36Parser, self).custom_classfunc_rule(opname, token, customize, next_token, is_pypy)

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            print('Hello World!')
        invalid = super(Python36Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        if invalid:
            return invalid
        if rule[0] == 'assign':
            if len(tokens) >= last + 1 and tokens[last] == 'LOAD_NAME' and (tokens[last + 1] == 'STORE_ANNOTATION') and (tokens[last - 1].pattr == tokens[last + 1].pattr):
                return True
            pass
        if rule[0] == 'call_kw':
            nt = ast[0]
            while not isinstance(nt, Token):
                if nt[0] == 'call_kw':
                    return True
                nt = nt[0]
                pass
            pass
        return False

class Python36ParserSingle(Python36Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python36Parser()
    p.check_grammar()
    from xdis.version_info import PYTHON_VERSION_TRIPLE, IS_PYPY
    if PYTHON_VERSION_TRIPLE[:2] == (3, 6):
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