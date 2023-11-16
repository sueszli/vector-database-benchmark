"""
Custom lexer
============

Demonstrates using a custom lexer to parse a non-textual stream of data

You can use a custom lexer to tokenize text when the lexers offered by Lark
are too slow, or not flexible enough.

You can also use it (as shown in this example) to tokenize streams of objects.
"""
from lark import Lark, Transformer, v_args
from lark.lexer import Lexer, Token

class TypeLexer(Lexer):

    def __init__(self, lexer_conf):
        if False:
            i = 10
            return i + 15
        pass

    def lex(self, data):
        if False:
            i = 10
            return i + 15
        for obj in data:
            if isinstance(obj, int):
                yield Token('INT', obj)
            elif isinstance(obj, (type(''), type(u''))):
                yield Token('STR', obj)
            else:
                raise TypeError(obj)
parser = Lark('\n        start: data_item+\n        data_item: STR INT*\n\n        %declare STR INT\n        ', parser='lalr', lexer=TypeLexer)

class ParseToDict(Transformer):

    @v_args(inline=True)
    def data_item(self, name, *numbers):
        if False:
            print('Hello World!')
        return (name.value, [n.value for n in numbers])
    start = dict

def test():
    if False:
        while True:
            i = 10
    data = ['alice', 1, 27, 3, 'bob', 4, 'carrie', 'dan', 8, 6]
    print(data)
    tree = parser.parse(data)
    res = ParseToDict().transform(tree)
    print('-->')
    print(res)
if __name__ == '__main__':
    test()