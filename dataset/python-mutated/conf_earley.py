"""
Earley’s dynamic lexer
======================

Demonstrates the power of Earley’s dynamic lexer on a toy configuration language

Using a lexer for configuration files is tricky, because values don't
have to be surrounded by delimiters. Using a basic lexer for this just won't work.

In this example we use a dynamic lexer and let the Earley parser resolve the ambiguity.

Another approach is to use the contextual lexer with LALR. It is less powerful than Earley,
but it can handle some ambiguity when lexing and it's much faster.
See examples/conf_lalr.py for an example of that approach.

"""
from lark import Lark
parser = Lark('\n        start: _NL? section+\n        section: "[" NAME "]" _NL item+\n        item: NAME "=" VALUE? _NL\n\n        NAME: /\\w/+\n        VALUE: /./+\n\n        %import common.NEWLINE -> _NL\n        %import common.WS_INLINE\n        %ignore WS_INLINE\n    ', parser='earley')

def test():
    if False:
        for i in range(10):
            print('nop')
    sample_conf = '\n[bla]\n\na=Hello\nthis="that",4\nempty=\n'
    r = parser.parse(sample_conf)
    print(r.pretty())
if __name__ == '__main__':
    test()