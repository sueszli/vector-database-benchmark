from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse22 import Python22Parser

class Python21Parser(Python22Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            while True:
                i = 10
        super(Python21Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_forstmt21(self, args):
        if False:
            return 10
        '\n        for         ::= SETUP_LOOP expr for_iter store\n                        returns\n                        POP_BLOCK COME_FROM\n        for         ::= SETUP_LOOP expr for_iter store\n                        l_stmts_opt _jump_back\n                        POP_BLOCK COME_FROM\n        '

    def p_import21(self, args):
        if False:
            while True:
                i = 10
        '\n        alias ::= IMPORT_NAME_CONT store\n        '

class Python21ParserSingle(Python22Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python21Parser()
    p.check_grammar()
    p.dump_grammar()