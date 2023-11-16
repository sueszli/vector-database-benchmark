"""
spark grammar differences over Python 3 for Python 3.2.
"""
from __future__ import print_function
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse3 import Python3Parser

class Python32Parser(Python3Parser):

    def p_30to33(self, args):
        if False:
            print('Hello World!')
        '\n        # Store locals is only in Python 3.0 to 3.3\n        stmt           ::= store_locals\n        store_locals   ::= LOAD_FAST STORE_LOCALS\n        '

    def p_gen_comp32(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        genexpr_func ::= LOAD_ARG FOR_ITER store comp_iter JUMP_BACK\n        '

    def p_32to35(self, args):
        if False:
            print('Hello World!')
        '\n        if_exp            ::= expr jmp_false expr jump_forward_else expr COME_FROM\n\n        # compare_chained_right is used in a "chained_compare": x <= y <= z\n        compare_chained_right ::= expr COMPARE_OP RETURN_VALUE\n        compare_chained_right ::= expr COMPARE_OP RETURN_VALUE_LAMBDA\n\n        # Python < 3.5 no POP BLOCK\n        whileTruestmt  ::= SETUP_LOOP l_stmts_opt JUMP_BACK COME_FROM_LOOP\n\n        # Python 3.5+ has jump optimization to remove the redundant\n        # jump_excepts. But in 3.3 we need them added\n\n        try_except     ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                           except_handler\n                           jump_excepts come_from_except_clauses\n\n        except_handler ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts\n                           END_FINALLY\n\n        tryelsestmt    ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK\n                           except_handler else_suite\n                           jump_excepts come_from_except_clauses\n\n        jump_excepts   ::= jump_except+\n\n        # Python 3.2+ has more loop optimization that removes\n        # JUMP_FORWARD in some cases, and hence we also don\'t\n        # see COME_FROM\n        _ifstmts_jump ::= stmts_opt\n        _ifstmts_jump ::= stmts_opt JUMP_FORWARD _come_froms\n        _ifstmts_jumpl ::= c_stmts_opt\n        _ifstmts_jumpl ::= c_stmts_opt JUMP_FORWARD _come_froms\n\n        kv3       ::= expr expr STORE_MAP\n        '
    pass

    def p_32on(self, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        # In Python 3.2+, DUP_TOPX is DUP_TOP_TWO\n        subscript2 ::= expr expr DUP_TOP_TWO BINARY_SUBSCR\n        '
        pass

    def customize_grammar_rules(self, tokens, customize):
        if False:
            print('Hello World!')
        self.remove_rules('\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts END_FINALLY COME_FROM\n        except_handler ::= JUMP_FORWARD COME_FROM except_stmts END_FINALLY COME_FROM_EXCEPT\n        except_handler ::= JUMP_FORWARD COME_FROM_EXCEPT except_stmts END_FINALLY COME_FROM_EXCEPT_CLAUSE\n        except_handler ::= jmp_abs COME_FROM except_stmts END_FINALLY\n        tryelsestmt    ::= SETUP_EXCEPT suite_stmts_opt POP_BLOCK except_handler else_suite come_from_except_clauses\n        whileTruestmt  ::= SETUP_LOOP l_stmts_opt JUMP_BACK NOP COME_FROM_LOOP\n        whileTruestmt  ::= SETUP_LOOP l_stmts_opt JUMP_BACK POP_BLOCK NOP COME_FROM_LOOP\n        ')
        super(Python32Parser, self).customize_grammar_rules(tokens, customize)
        for (i, token) in enumerate(tokens):
            opname = token.kind
            if opname.startswith('MAKE_FUNCTION_A'):
                (args_pos, args_kw, annotate_args) = token.attr
                rule = 'mkfunc_annotate ::= %s%sannotate_tuple LOAD_CONST LOAD_CODE EXTENDED_ARG %s' % ('pos_arg ' * args_pos, 'annotate_arg ' * annotate_args, opname)
                self.add_unique_rule(rule, opname, token.attr, customize)
                pass
            return
        pass

class Python32ParserSingle(Python32Parser, PythonParserSingle):
    pass