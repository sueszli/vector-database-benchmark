"""
Base grammar for Python 2.x.

However instead of terminal symbols being the usual ASCII text,
e.g. 5, myvariable, "for", etc.  they are CPython Bytecode tokens,
e.g. "LOAD_CONST 5", "STORE NAME myvariable", "SETUP_LOOP", etc.

If we succeed in creating a parse tree, then we have a Python program
that a later phase can turn into a sequence of ASCII text.
"""
from __future__ import print_function
from uncompyle6.parsers.reducecheck import except_handler_else, ifelsestmt, tryelsestmt
from uncompyle6.parser import PythonParser, PythonParserSingle, nop_func
from uncompyle6.parsers.treenode import SyntaxTree
from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG

class Python2Parser(PythonParser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            return 10
        super(Python2Parser, self).__init__(SyntaxTree, 'stmts', debug=debug_parser)
        self.new_rules = set()

    def p_print2(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        stmt ::= print_items_stmt\n        stmt ::= print_nl\n        stmt ::= print_items_nl_stmt\n\n        print_items_stmt ::= expr PRINT_ITEM print_items_opt\n        print_items_nl_stmt ::= expr PRINT_ITEM print_items_opt PRINT_NEWLINE_CONT\n        print_items_opt ::= print_items?\n        print_items     ::= print_item+\n        print_item      ::= expr PRINT_ITEM_CONT\n        print_nl        ::= PRINT_NEWLINE\n        '

    def p_print_to(self, args):
        if False:
            print('Hello World!')
        '\n        stmt ::= print_to\n        stmt ::= print_to_nl\n        stmt ::= print_nl_to\n        print_to ::= expr print_to_items POP_TOP\n        print_to_nl ::= expr print_to_items PRINT_NEWLINE_TO\n        print_nl_to ::= expr PRINT_NEWLINE_TO\n        print_to_items ::= print_to_items print_to_item\n        print_to_items ::= print_to_item\n        print_to_item ::= DUP_TOP expr ROT_TWO PRINT_ITEM_TO\n        '

    def p_grammar(self, args):
        if False:
            while True:
                i = 10
        '\n        sstmt ::= stmt\n        sstmt ::= return RETURN_LAST\n\n        return_if_stmts ::= return_if_stmt\n        return_if_stmts ::= _stmts return_if_stmt\n        return_if_stmt ::= return_expr RETURN_END_IF\n\n        return_stmt_lambda ::= return_expr RETURN_VALUE_LAMBDA\n\n        stmt      ::= break\n        break     ::= BREAK_LOOP\n\n        stmt      ::= continue\n        continue  ::= CONTINUE\n        continues ::= _stmts lastl_stmt continue\n        continues ::= lastl_stmt continue\n        continues ::= continue\n\n        stmt ::= assert2\n        stmt ::= raise_stmt0\n        stmt ::= raise_stmt1\n        stmt ::= raise_stmt2\n        stmt ::= raise_stmt3\n\n        raise_stmt0 ::= RAISE_VARARGS_0\n        raise_stmt1 ::= expr RAISE_VARARGS_1\n        raise_stmt2 ::= expr expr RAISE_VARARGS_2\n        raise_stmt3 ::= expr expr expr RAISE_VARARGS_3\n\n        for         ::= SETUP_LOOP expr for_iter store\n                        for_block POP_BLOCK _come_froms\n\n        delete           ::= delete_subscript\n        delete_subscript ::= expr expr DELETE_SUBSCR\n        delete           ::= expr DELETE_ATTR\n\n        _lambda_body ::= load_closure lambda_body\n        kwarg     ::= LOAD_CONST expr\n\n        kv3 ::= expr expr STORE_MAP\n\n        classdef ::= buildclass store\n\n        buildclass ::= LOAD_CONST expr mkfunc\n                     CALL_FUNCTION_0 BUILD_CLASS\n\n        # Class decorators starting in 2.6\n        stmt ::= classdefdeco\n        classdefdeco ::= classdefdeco1 store\n        classdefdeco1 ::= expr classdefdeco1 CALL_FUNCTION_1\n        classdefdeco1 ::= expr classdefdeco2 CALL_FUNCTION_1\n        classdefdeco2 ::= LOAD_CONST expr mkfunc CALL_FUNCTION_0 BUILD_CLASS\n\n        assert_expr ::= expr\n        assert_expr ::= assert_expr_or\n        assert_expr ::= assert_expr_and\n        assert_expr_or ::= assert_expr jmp_true expr\n        assert_expr_and ::= assert_expr jmp_false expr\n\n        ifstmt ::= testexpr _ifstmts_jump\n\n        testexpr ::= testfalse\n        testexpr ::= testtrue\n        testfalse ::= expr jmp_false\n        testtrue ::= expr jmp_true\n\n        _ifstmts_jump ::= return_if_stmts\n\n        iflaststmt  ::= testexpr c_stmts_opt JUMP_ABSOLUTE\n        iflaststmtl ::= testexpr c_stmts_opt JUMP_BACK\n\n        # this is nested inside a try_except\n        tryfinallystmt  ::= SETUP_FINALLY suite_stmts_opt\n                            POP_BLOCK LOAD_CONST\n                            COME_FROM suite_stmts_opt END_FINALLY\n\n        lastc_stmt ::= tryelsestmtc\n\n        # Move to 2.7? 2.6 may use come_froms\n        tryelsestmtc    ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                            except_handler_else else_suitec COME_FROM\n\n        tryelsestmtl    ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                            except_handler_else else_suitel COME_FROM\n\n        try_except      ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                            except_handler COME_FROM\n\n        # Note: except_stmts may have many jumps after END_FINALLY\n        except_handler  ::= JUMP_FORWARD COME_FROM except_stmts\n                            END_FINALLY come_froms\n\n        except_handler  ::= jmp_abs COME_FROM except_stmts\n                             END_FINALLY\n\n        except_handler_else  ::= except_handler\n\n        except_stmts ::= except_stmt+\n\n        except_stmt ::= except_cond1 except_suite\n        except_stmt ::= except\n\n        except_suite ::= c_stmts_opt JUMP_FORWARD\n        except_suite ::= c_stmts_opt jmp_abs\n        except_suite ::= returns\n\n        except  ::=  POP_TOP POP_TOP POP_TOP c_stmts_opt _jump\n        except  ::=  POP_TOP POP_TOP POP_TOP returns\n\n        jmp_abs ::= JUMP_ABSOLUTE\n        jmp_abs ::= JUMP_BACK\n        jmp_abs ::= CONTINUE\n        '

    def p_generator_exp2(self, args):
        if False:
            return 10
        '\n        generator_exp ::= LOAD_GENEXPR MAKE_FUNCTION_0 expr GET_ITER CALL_FUNCTION_1\n        '

    def p_expr2(self, args):
        if False:
            i = 10
            return i + 15
        '\n        expr ::= LOAD_LOCALS\n        expr ::= LOAD_ASSERT\n        expr ::= slice0\n        expr ::= slice1\n        expr ::= slice2\n        expr ::= slice3\n        expr ::= unary_convert\n\n        expr_jt  ::= expr jmp_true\n        or       ::= expr_jt  expr come_from_opt\n        and      ::= expr jmp_false expr come_from_opt\n\n        unary_convert ::= expr UNARY_CONVERT\n\n        # In Python 3, DUP_TOPX_2 is DUP_TOP_TWO\n        subscript2 ::= expr expr DUP_TOPX_2 BINARY_SUBSCR\n        '

    def p_slice2(self, args):
        if False:
            while True:
                i = 10
        '\n        store ::= expr STORE_SLICE+0\n        store ::= expr expr STORE_SLICE+1\n        store ::= expr expr STORE_SLICE+2\n        store ::= expr expr expr STORE_SLICE+3\n\n        aug_assign1 ::= expr expr inplace_op ROT_FOUR  STORE_SLICE+3\n        aug_assign1 ::= expr expr inplace_op ROT_THREE STORE_SLICE+1\n        aug_assign1 ::= expr expr inplace_op ROT_THREE STORE_SLICE+2\n        aug_assign1 ::= expr expr inplace_op ROT_TWO   STORE_SLICE+0\n\n        slice0 ::= expr SLICE+0\n        slice0 ::= expr DUP_TOP SLICE+0\n        slice1 ::= expr expr SLICE+1\n        slice1 ::= expr expr DUP_TOPX_2 SLICE+1\n        slice2 ::= expr expr SLICE+2\n        slice2 ::= expr expr DUP_TOPX_2 SLICE+2\n        slice3 ::= expr expr expr SLICE+3\n        slice3 ::= expr expr expr DUP_TOPX_3 SLICE+3\n        '

    def p_op2(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        inplace_op ::= INPLACE_DIVIDE\n        binary_operator  ::= BINARY_DIVIDE\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            return 10
        "The base grammar we start out for a Python version even with the\n        subclassing is, well, is pretty base.  And we want it that way: lean and\n        mean so that parsing will go faster.\n\n        Here, we add additional grammar rules based on specific instructions\n        that are in the instruction/token stream. In classes that\n        inherit from from here and other versions, grammar rules may\n        also be removed.\n\n        For example if we see a pretty rare JUMP_IF_NOT_DEBUG\n        instruction we'll add the grammar for that.\n\n        More importantly, here we add grammar rules for instructions\n        that may access a variable number of stack items. CALL_FUNCTION,\n        BUILD_LIST and so on are like this.\n\n        Without custom rules, there can be an super-exponential number of\n        derivations. See the deparsing paper for an elaboration of\n        this.\n        "
        if 'PyPy' in customize:
            self.addRule('\n                        stmt ::= assign3_pypy\n                        stmt ::= assign2_pypy\n                        assign3_pypy ::= expr expr expr store store store\n                        assign2_pypy ::= expr expr store store\n                        list_comp    ::= expr  BUILD_LIST_FROM_ARG for_iter store list_iter\n                                         JUMP_BACK\n                        ', nop_func)
        customize_instruction_basenames = frozenset(('BUILD', 'CALL', 'CONTINUE', 'DELETE', 'DUP', 'EXEC', 'GET', 'JUMP', 'LOAD', 'LOOKUP', 'MAKE', 'SETUP', 'RAISE', 'UNPACK'))
        custom_seen_ops = set()
        for (i, token) in enumerate(tokens):
            opname = token.kind
            if opname[:opname.find('_')] not in customize_instruction_basenames or opname in custom_seen_ops:
                continue
            opname_base = opname[:opname.rfind('_')]
            if opname in ('BUILD_CONST_LIST', 'BUILD_CONST_SET'):
                rule = '\n                       add_consts          ::= add_value+\n                       add_value           ::= ADD_VALUE\n                       add_value           ::= ADD_VALUE_VAR\n                       const_list          ::= COLLECTION_START add_consts %s\n                       expr                ::= const_list\n                       ' % opname
                self.addRule(rule, nop_func)
            if opname_base in ('BUILD_LIST', 'BUILD_SET', 'BUILD_TUPLE'):
                build_count = token.attr
                thousands = build_count // 1024
                thirty32s = build_count // 32 % 32
                if thirty32s > 0 or thousands > 0:
                    rule = 'expr32 ::=%s' % (' expr' * 32)
                    self.add_unique_rule(rule, opname_base, build_count, customize)
                if thousands > 0:
                    self.add_unique_rule('expr1024 ::=%s' % (' expr32' * 32), opname_base, build_count, customize)
                collection = opname_base[opname_base.find('_') + 1:].lower()
                rule = '%s ::= ' % collection + 'expr1024 ' * thousands + 'expr32 ' * thirty32s + 'expr ' * (build_count % 32) + opname
                self.add_unique_rules(['expr ::= %s' % collection, rule], customize)
                continue
            elif opname_base == 'BUILD_MAP':
                if opname == 'BUILD_MAP_n':
                    self.add_unique_rules(['kvlist_n ::=  kvlist_n kv3', 'kvlist_n ::=', 'dict ::= BUILD_MAP_n kvlist_n'], customize)
                    if self.version >= (2, 7):
                        self.add_unique_rule('dict_comp_func ::= BUILD_MAP_n LOAD_FAST FOR_ITER store comp_iter JUMP_BACK RETURN_VALUE RETURN_LAST', 'dict_comp_func', 0, customize)
                else:
                    kvlist_n = ' kv3' * token.attr
                    rule = 'dict ::= %s%s' % (opname, kvlist_n)
                    self.addRule(rule, nop_func)
                continue
            elif opname_base == 'BUILD_SLICE':
                slice_num = token.attr
                if slice_num == 2:
                    self.add_unique_rules(['expr ::= build_slice2', 'build_slice2 ::= expr expr BUILD_SLICE_2'], customize)
                else:
                    assert slice_num == 3, 'BUILD_SLICE value must be 2 or 3; is %s' % slice_num
                    self.add_unique_rules(['expr ::= build_slice3', 'build_slice3 ::= expr expr expr BUILD_SLICE_3'], customize)
                continue
            elif opname_base in ('CALL_FUNCTION', 'CALL_FUNCTION_VAR', 'CALL_FUNCTION_VAR_KW', 'CALL_FUNCTION_KW'):
                (args_pos, args_kw) = self.get_pos_kw(token)
                nak = (len(opname_base) - len('CALL_FUNCTION')) // 3
                rule = 'call ::= expr ' + 'expr ' * args_pos + 'kwarg ' * args_kw + 'expr ' * nak + opname
            elif opname_base == 'CALL_METHOD':
                (args_pos, args_kw) = self.get_pos_kw(token)
                nak = (len(opname_base) - len('CALL_METHOD')) // 3
                rule = 'call ::= expr ' + 'expr ' * args_pos + 'kwarg ' * args_kw + 'expr ' * nak + opname
            elif opname == 'CONTINUE_LOOP':
                self.addRule('continue ::= CONTINUE_LOOP', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname == 'DELETE_ATTR':
                self.addRule('delete ::= expr DELETE_ATTR', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname.startswith('DELETE_SLICE'):
                self.addRule('\n                del_expr ::= expr\n                delete   ::= del_expr DELETE_SLICE+0\n                delete   ::= del_expr del_expr DELETE_SLICE+1\n                delete   ::= del_expr del_expr DELETE_SLICE+2\n                delete   ::= del_expr del_expr del_expr DELETE_SLICE+3\n                ', nop_func)
                custom_seen_ops.add(opname)
                self.check_reduce['del_expr'] = 'AST'
                continue
            elif opname == 'DELETE_DEREF':
                self.addRule('\n                   stmt           ::= del_deref_stmt\n                   del_deref_stmt ::= DELETE_DEREF\n                   ', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname == 'DELETE_SUBSCR':
                self.addRule('\n                    delete ::= delete_subscript\n                    delete_subscript ::= expr expr DELETE_SUBSCR\n                   ', nop_func)
                self.check_reduce['delete_subscript'] = 'AST'
                custom_seen_ops.add(opname)
                continue
            elif opname == 'GET_ITER':
                self.addRule('\n                    expr      ::= get_iter\n                    attribute ::= expr GET_ITER\n                    ', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname_base in ('DUP_TOPX', 'RAISE_VARARGS'):
                continue
            elif opname == 'EXEC_STMT':
                self.addRule('\n                    stmt      ::= exec_stmt\n                    exec_stmt ::= expr exprlist DUP_TOP EXEC_STMT\n                    exec_stmt ::= expr exprlist EXEC_STMT\n                    exprlist  ::= expr+\n                    ', nop_func)
                continue
            elif opname == 'JUMP_IF_NOT_DEBUG':
                self.addRule('\n                    jmp_true_false ::= POP_JUMP_IF_TRUE\n                    jmp_true_false ::= POP_JUMP_IF_FALSE\n                    stmt ::= assert_pypy\n                    stmt ::= assert2_pypy\n                    assert_pypy  ::= JUMP_IF_NOT_DEBUG assert_expr jmp_true_false\n                                     LOAD_ASSERT RAISE_VARARGS_1 COME_FROM\n                    assert2_pypy ::= JUMP_IF_NOT_DEBUG assert_expr jmp_true_false\n                                     LOAD_ASSERT expr CALL_FUNCTION_1\n                                     RAISE_VARARGS_1 COME_FROM\n                     ', nop_func)
                continue
            elif opname == 'LOAD_ATTR':
                self.addRule('\n                  expr      ::= attribute\n                  attribute ::= expr LOAD_ATTR\n                  ', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname == 'LOAD_LISTCOMP':
                self.addRule('expr ::= listcomp', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname == 'LOAD_SETCOMP':
                self.add_unique_rules(['expr ::= set_comp', 'set_comp ::= LOAD_SETCOMP MAKE_FUNCTION_0 expr GET_ITER CALL_FUNCTION_1'], customize)
                custom_seen_ops.add(opname)
                continue
            elif opname == 'LOOKUP_METHOD':
                self.addRule('\n                             expr      ::= attribute\n                             attribute ::= expr LOOKUP_METHOD\n                             ', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname_base == 'MAKE_FUNCTION':
                if i > 0 and tokens[i - 1] == 'LOAD_LAMBDA':
                    self.addRule('lambda_body ::= %s LOAD_LAMBDA %s' % ('pos_arg ' * token.attr, opname), nop_func)
                rule = 'mkfunc ::= %s LOAD_CODE %s' % ('expr ' * token.attr, opname)
            elif opname_base == 'MAKE_CLOSURE':
                if i > 0 and tokens[i - 1] == 'LOAD_LAMBDA':
                    self.addRule('lambda_body ::= %s load_closure LOAD_LAMBDA %s' % ('expr ' * token.attr, opname), nop_func)
                if i > 0:
                    prev_tok = tokens[i - 1]
                    if prev_tok == 'LOAD_GENEXPR':
                        self.add_unique_rules(['generator_exp ::= %s load_closure LOAD_GENEXPR %s expr GET_ITER CALL_FUNCTION_1' % ('expr ' * token.attr, opname)], customize)
                        pass
                self.add_unique_rules(['mkfunc ::= %s load_closure LOAD_CODE %s' % ('expr ' * token.attr, opname)], customize)
                if self.version >= (2, 7):
                    if i > 0:
                        prev_tok = tokens[i - 1]
                        if prev_tok == 'LOAD_DICTCOMP':
                            self.add_unique_rules(['dict_comp ::= %s load_closure LOAD_DICTCOMP %s expr GET_ITER CALL_FUNCTION_1' % ('expr ' * token.attr, opname)], customize)
                        elif prev_tok == 'LOAD_SETCOMP':
                            self.add_unique_rules(['expr ::= set_comp', 'set_comp ::= %s load_closure LOAD_SETCOMP %s expr GET_ITER CALL_FUNCTION_1' % ('expr ' * token.attr, opname)], customize)
                        pass
                    pass
                continue
            elif opname == 'SETUP_EXCEPT':
                if 'PyPy' in customize:
                    self.add_unique_rules(['stmt ::= try_except_pypy', 'try_except_pypy ::= SETUP_EXCEPT suite_stmts_opt except_handler_pypy', 'except_handler_pypy ::= COME_FROM except_stmts END_FINALLY COME_FROM'], customize)
                custom_seen_ops.add(opname)
                continue
            elif opname == 'SETUP_FINALLY':
                if 'PyPy' in customize:
                    self.addRule('\n                        stmt ::= tryfinallystmt_pypy\n                        tryfinallystmt_pypy ::= SETUP_FINALLY suite_stmts_opt COME_FROM_FINALLY\n                                                suite_stmts_opt END_FINALLY', nop_func)
                custom_seen_ops.add(opname)
                continue
            elif opname_base in ('UNPACK_TUPLE', 'UNPACK_SEQUENCE'):
                custom_seen_ops.add(opname)
                rule = 'unpack ::= ' + opname + ' store' * token.attr
            elif opname_base == 'UNPACK_LIST':
                custom_seen_ops.add(opname)
                rule = 'unpack_list ::= ' + opname + ' store' * token.attr
            else:
                continue
            self.addRule(rule, nop_func)
            pass
        self.reduce_check_table = {'except_handler_else': except_handler_else, 'ifelsestmt': ifelsestmt, 'tryelsestmt': tryelsestmt, 'tryelsestmtl': tryelsestmt}
        self.check_reduce['and'] = 'AST'
        self.check_reduce['assert_expr_and'] = 'AST'
        self.check_reduce['aug_assign2'] = 'AST'
        self.check_reduce['except_handler_else'] = 'tokens'
        self.check_reduce['ifelsestmt'] = 'AST'
        self.check_reduce['ifstmt'] = 'tokens'
        self.check_reduce['or'] = 'AST'
        self.check_reduce['raise_stmt1'] = 'tokens'
        self.check_reduce['tryelsestmt'] = 'AST'
        self.check_reduce['tryelsestmtl'] = 'AST'
        return

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            for i in range(10):
                print('nop')
        if tokens is None:
            return False
        lhs = rule[0]
        n = len(tokens)
        fn = self.reduce_check_table.get(lhs, None)
        if fn:
            if fn(self, lhs, n, rule, ast, tokens, first, last):
                return True
            pass
        if rule == ('and', ('expr', 'jmp_false', 'expr', '\\e_come_from_opt')):
            if tokens[last] == 'YIELD_VALUE':
                return True
            jmp_false = ast[1]
            if jmp_false[0] == 'POP_JUMP_IF_FALSE':
                while first < last and isinstance(tokens[last].offset, str):
                    last -= 1
                if jmp_false[0].attr < tokens[last].offset:
                    return True
            jmp_false = ast[1][0]
            jmp_target = jmp_false.offset + jmp_false.attr + 3
            return not (jmp_target == tokens[last].offset or tokens[last].pattr == jmp_false.pattr)
        elif lhs in ('aug_assign1', 'aug_assign2') and ast[0] and (ast[0][0] in ('and', 'or')):
            return True
        elif lhs == 'assert_expr_and':
            jmp_false = ast[1]
            jump_target = jmp_false[0].attr
            return jump_target > tokens[last].off2int()
        elif lhs in ('raise_stmt1',):
            return tokens[first] == 'LOAD_ASSERT' and last >= len(tokens)
        elif rule == ('or', ('expr', 'jmp_true', 'expr', '\\e_come_from_opt')):
            expr2 = ast[2]
            return expr2 == 'expr' and expr2[0] == 'LOAD_ASSERT'
        elif lhs in ('delete_subscript', 'del_expr'):
            op = ast[0][0]
            return op.kind in ('and', 'or')
        return False

class Python2ParserSingle(Python2Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python2Parser()
    p.check_grammar()