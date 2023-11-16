import os
import tempfile
import sympy
from sympy.testing.pytest import raises
from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark
from sympy.external import import_module
lark = import_module('lark')
disabled = lark is None
grammar_file = os.path.join(os.path.dirname(__file__), '../latex/lark/grammar/latex.lark')
modification1 = '\n%override DIV_SYMBOL: DIV\n%override MUL_SYMBOL: MUL | CMD_TIMES\n'
modification2 = '\n%override number: /\\d+(,\\d*)?/\n'

def init_custom_parser(modification, transformer=None):
    if False:
        for i in range(10):
            print('nop')
    with open(grammar_file, encoding='utf-8') as f:
        latex_grammar = f.read()
    latex_grammar += modification
    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(latex_grammar, encoding='utf8'))
        parser = LarkLaTeXParser(grammar_file=f.name, transformer=transformer)
    return parser

def test_custom1():
    if False:
        i = 10
        return i + 15
    parser = init_custom_parser(modification1)
    with raises(lark.exceptions.UnexpectedCharacters):
        parser.doparse('a \\cdot b')
        parser.doparse('x \\div y')

class CustomTransformer(TransformToSymPyExpr):

    def number(self, tokens):
        if False:
            i = 10
            return i + 15
        if ',' in tokens[0]:
            return sympy.core.numbers.Float(tokens[0].replace(',', '.'))
        else:
            return sympy.core.numbers.Integer(tokens[0])

def test_custom2():
    if False:
        while True:
            i = 10
    parser = init_custom_parser(modification2, CustomTransformer)
    with raises(lark.exceptions.UnexpectedCharacters):
        parse_latex_lark('100,1')
        parse_latex_lark('0,009')
    parser.doparse('100,1')
    parser.doparse('0,009')
    parser.doparse('2,71828')
    parser.doparse('3,14159')