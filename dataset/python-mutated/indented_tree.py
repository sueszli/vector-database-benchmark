"""
Parsing Indentation
===================

A demonstration of parsing indentation (“whitespace significant” language)
and the usage of the Indenter class.

Since indentation is context-sensitive, a postlex stage is introduced to
manufacture INDENT/DEDENT tokens.

It is crucial for the indenter that the NL_type matches
the spaces (and tabs) after the newline.
"""
from lark import Lark
from lark.indenter import Indenter
tree_grammar = '\n    ?start: _NL* tree\n\n    tree: NAME _NL [_INDENT tree+ _DEDENT]\n\n    %import common.CNAME -> NAME\n    %import common.WS_INLINE\n    %declare _INDENT _DEDENT\n    %ignore WS_INLINE\n\n    _NL: /(\\r?\\n[\\t ]*)+/\n'

class TreeIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8
parser = Lark(tree_grammar, parser='lalr', postlex=TreeIndenter())
test_tree = '\na\n    b\n    c\n        d\n        e\n    f\n        g\n'

def test():
    if False:
        print('Hello World!')
    print(parser.parse(test_tree).pretty())
if __name__ == '__main__':
    test()