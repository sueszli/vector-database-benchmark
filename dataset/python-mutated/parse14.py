from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle, nop_func
from uncompyle6.parsers.parse15 import Python15Parser

class Python14Parser(Python15Parser):

    def p_misc14(self, args):
        if False:
            for i in range(10):
                print('nop')
        "\n        # Not much here yet, but will probably need to add UNARY_CALL,\n        # LOAD_LOCAL, SET_FUNC_ARGS\n\n        args            ::= RESERVE_FAST UNPACK_ARG args_store\n        args_store      ::= STORE_FAST*\n        call            ::= expr tuple BINARY_CALL\n        expr            ::= call\n        kv              ::= DUP_TOP expr ROT_TWO LOAD_CONST STORE_SUBSCR\n        mkfunc          ::= LOAD_CODE BUILD_FUNCTION\n        print_expr_stmt ::= expr PRINT_EXPR\n        raise_stmt2     ::= expr expr RAISE_EXCEPTION\n        star_args       ::= RESERVE_FAST UNPACK_VARARG_1 args_store\n        stmt            ::= args\n        stmt            ::= print_expr_stmt\n        stmt            ::= star_args\n        stmt            ::= varargs\n        varargs         ::= RESERVE_FAST UNPACK_VARARG_0 args_store\n\n        # Not strictly needed, but tidies up output\n\n        stmt     ::= doc_junk\n        doc_junk ::= LOAD_CONST POP_TOP\n\n        # Not sure why later Python's omit the COME_FROM\n        jb_pop14  ::= JUMP_BACK COME_FROM POP_TOP\n\n        whileelsestmt ::= SETUP_LOOP testexpr l_stmts_opt\n                          jb_pop14\n                          POP_BLOCK else_suitel COME_FROM\n\n        print_items_nl_stmt ::= expr PRINT_ITEM_CONT print_items_opt PRINT_NEWLINE_CONT\n\n\n        # 1.4 doesn't have linenotab, and although this shouldn't\n        # be a show stopper, our CONTINUE detection is off here.\n        continue ::= JUMP_BACK\n        "

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            i = 10
            return i + 15
        super(Python14Parser, self).__init__(debug_parser)
        self.customized = {}

    def customize_grammar_rules(self, tokens, customize):
        if False:
            for i in range(10):
                print('nop')
        super(Python14Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules('\n        whileelsestmt ::= SETUP_LOOP testexpr l_stmts_opt\n                          jb_pop\n                          POP_BLOCK else_suitel COME_FROM\n        ')
        self.check_reduce['doc_junk'] = 'tokens'
        for (i, token) in enumerate(tokens):
            opname = token.kind
            opname_base = opname[:opname.rfind('_')]
            if opname_base == 'UNPACK_VARARG':
                if token.attr > 1:
                    self.addRule(f'star_args ::= RESERVE_FAST {opname} args_store', nop_func)

    def reduce_is_invalid(self, rule, ast, tokens, first, last):
        if False:
            return 10
        invalid = super(Python14Parser, self).reduce_is_invalid(rule, ast, tokens, first, last)
        if invalid or tokens is None:
            return invalid
        if rule[0] == 'doc_junk':
            return not isinstance(tokens[first].pattr, str)

class Python14ParserSingle(Python14Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python14Parser()
    p.check_grammar()
    p.dump_grammar()