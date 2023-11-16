from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse23 import Python23Parser

class Python22Parser(Python23Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            for i in range(10):
                print('nop')
        super(Python22Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_misc22(self, args):
        if False:
            i = 10
            return i + 15
        '\n        for_iter  ::= LOAD_CONST FOR_LOOP\n        list_iter ::= list_if JUMP_FORWARD\n                      COME_FROM POP_TOP COME_FROM\n        list_for  ::= expr for_iter store list_iter CONTINUE JUMP_FORWARD\n                      COME_FROM POP_TOP COME_FROM\n\n        # Some versions of Python 2.2 have been found to generate\n        # PRINT_ITEM_CONT for PRINT_ITEM\n        print_items_stmt ::= expr PRINT_ITEM_CONT print_items_opt\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            print('Hello World!')
        super(Python22Parser, self).customize_grammar_rules(tokens, customize)
        self.remove_rules('\n        kvlist ::= kvlist kv2\n        ')
        if self.version[:2] <= (2, 2):
            del self.reduce_check_table['ifstmt']

class Python22ParserSingle(Python23Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python22Parser()
    p.check_grammar()
    p.dump_grammar()