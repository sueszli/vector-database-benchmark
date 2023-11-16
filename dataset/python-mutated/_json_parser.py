"""
Simple JSON Parser
==================

The code is short and clear, and outperforms every other parser (that's written in Python).
For an explanation, check out the JSON parser tutorial at /docs/json_tutorial.md

(this is here for use by the other examples)
"""
from lark import Lark, Transformer, v_args
json_grammar = '\n    ?start: value\n\n    ?value: object\n          | array\n          | string\n          | SIGNED_NUMBER      -> number\n          | "true"             -> true\n          | "false"            -> false\n          | "null"             -> null\n\n    array  : "[" [value ("," value)*] "]"\n    object : "{" [pair ("," pair)*] "}"\n    pair   : string ":" value\n\n    string : ESCAPED_STRING\n\n    %import common.ESCAPED_STRING\n    %import common.SIGNED_NUMBER\n    %import common.WS\n\n    %ignore WS\n'

class TreeToJson(Transformer):

    @v_args(inline=True)
    def string(self, s):
        if False:
            i = 10
            return i + 15
        return s[1:-1].replace('\\"', '"')
    array = list
    pair = tuple
    object = dict
    number = v_args(inline=True)(float)
    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False
json_parser = Lark(json_grammar, parser='lalr', lexer='basic', propagate_positions=False, maybe_placeholders=False, transformer=TreeToJson())