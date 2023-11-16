from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle
from uncompyle6.parsers.parse14 import Python14Parser

class Python13Parser(Python14Parser):

    def p_misc13(self, args):
        if False:
            i = 10
            return i + 15
        '\n        # Nothing here yet, but will need to add LOAD_GLOBALS\n        '

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            i = 10
            return i + 15
        super(Python13Parser, self).__init__(debug_parser)
        self.customized = {}

class Python13ParserSingle(Python13Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python13Parser()
    p.check_grammar()
    p.dump_grammar()