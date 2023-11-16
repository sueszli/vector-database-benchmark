"""
Creating an AST from the parse tree
===================================

    This example demonstrates how to transform a parse-tree into an AST using `lark.ast_utils`.

    create_transformer() collects every subclass of `Ast` subclass from the module,
    and creates a Lark transformer that builds the AST with no extra code.

    This example only works with Python 3.
"""
import sys
from typing import List
from dataclasses import dataclass
from lark import Lark, ast_utils, Transformer, v_args
from lark.tree import Meta
this_module = sys.modules[__name__]

class _Ast(ast_utils.Ast):
    pass

class _Statement(_Ast):
    pass

@dataclass
class Value(_Ast, ast_utils.WithMeta):
    """Uses WithMeta to include line-number metadata in the meta attribute"""
    meta: Meta
    value: object

@dataclass
class Name(_Ast):
    name: str

@dataclass
class CodeBlock(_Ast, ast_utils.AsList):
    statements: List[_Statement]

@dataclass
class If(_Statement):
    cond: Value
    then: CodeBlock

@dataclass
class SetVar(_Statement):
    name: str
    value: Value

@dataclass
class Print(_Statement):
    value: Value

class ToAst(Transformer):

    def STRING(self, s):
        if False:
            i = 10
            return i + 15
        return s[1:-1]

    def DEC_NUMBER(self, n):
        if False:
            print('Hello World!')
        return int(n)

    @v_args(inline=True)
    def start(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x
parser = Lark('\n    start: code_block\n\n    code_block: statement+\n\n    ?statement: if | set_var | print\n\n    if: "if" value "{" code_block "}"\n    set_var: NAME "=" value ";"\n    print: "print" value ";"\n\n    value: name | STRING | DEC_NUMBER\n    name: NAME\n\n    %import python (NAME, STRING, DEC_NUMBER)\n    %import common.WS\n    %ignore WS\n    ', parser='lalr')
transformer = ast_utils.create_transformer(this_module, ToAst())

def parse(text):
    if False:
        return 10
    tree = parser.parse(text)
    return transformer.transform(tree)
if __name__ == '__main__':
    print(parse('\n        a = 1;\n        if a {\n            print "a is 1";\n            a = 2;\n        }\n    '))