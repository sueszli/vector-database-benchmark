from spark_parser import DEFAULT_DEBUG as PARSER_DEFAULT_DEBUG
from uncompyle6.parser import PythonParserSingle, nop_func
from uncompyle6.parsers.parse21 import Python21Parser

class Python15Parser(Python21Parser):

    def __init__(self, debug_parser=PARSER_DEFAULT_DEBUG):
        if False:
            for i in range(10):
                print('nop')
        super(Python15Parser, self).__init__(debug_parser)
        self.customized = {}

    def p_import15(self, args):
        if False:
            return 10
        '\n        import      ::= filler IMPORT_NAME STORE_FAST\n        import      ::= filler IMPORT_NAME STORE_NAME\n\n        import_from ::= filler IMPORT_NAME importlist\n        import_from ::= filler filler IMPORT_NAME importlist POP_TOP\n\n        importlist  ::= importlist IMPORT_FROM\n        importlist  ::= IMPORT_FROM\n        '

    def customize_grammar_rules(self, tokens, customize):
        if False:
            for i in range(10):
                print('nop')
        super(Python15Parser, self).customize_grammar_rules(tokens, customize)
        for (i, token) in enumerate(tokens):
            opname = token.kind
            opname_base = opname[:opname.rfind('_')]
            if opname_base == 'UNPACK_LIST':
                self.addRule('store ::= unpack_list', nop_func)

class Python15ParserSingle(Python15Parser, PythonParserSingle):
    pass
if __name__ == '__main__':
    p = Python15Parser()
    p.check_grammar()
    p.dump_grammar()