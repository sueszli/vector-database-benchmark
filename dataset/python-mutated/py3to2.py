"""
Python 3 to Python 2 converter (tree templates)
===============================================

This example demonstrates how to translate between two trees using tree templates.
It parses Python 3, translates it to a Python 2 AST, and then outputs the result as Python 2 code.

Uses reconstruct_python.py for generating the final Python 2 code.
"""
from lark import Lark
from lark.tree_templates import TemplateConf, TemplateTranslator
from lark.indenter import PythonIndenter
from reconstruct_python import PythonReconstructor
TEMPLATED_PYTHON = '\n%import python (single_input, file_input, eval_input, atom, var, stmt, expr, testlist_star_expr, _NEWLINE, _INDENT, _DEDENT, COMMENT, NAME)\n\n%extend atom: TEMPLATE_NAME -> var\n\nTEMPLATE_NAME: "$" NAME\n\n?template_start: (stmt | testlist_star_expr _NEWLINE)\n\n%ignore /[\\t \\f]+/          // WS\n%ignore /\\\\[\\t \\f]*\\r?\\n/   // LINE_CONT\n%ignore COMMENT\n'
parser = Lark(TEMPLATED_PYTHON, parser='lalr', start=['single_input', 'file_input', 'eval_input', 'template_start'], postlex=PythonIndenter(), maybe_placeholders=False)

def parse_template(s):
    if False:
        i = 10
        return i + 15
    return parser.parse(s + '\n', start='template_start')

def parse_code(s):
    if False:
        return 10
    return parser.parse(s + '\n', start='file_input')
pytemplate = TemplateConf(parse=parse_template)
translations_3to2 = {'yield from $a': 'for _tmp in $a: yield _tmp', 'raise $e from $x': 'raise $e', '$a / $b': 'float($a) / $b'}
translations_3to2 = {pytemplate(k): pytemplate(v) for (k, v) in translations_3to2.items()}
python_reconstruct = PythonReconstructor(parser)

def translate_py3to2(code):
    if False:
        i = 10
        return i + 15
    tree = parse_code(code)
    tree = TemplateTranslator(translations_3to2).translate(tree)
    return python_reconstruct.reconstruct(tree)
_TEST_CODE = '\nif a / 2 > 1:\n    yield from [1,2,3]\nelse:\n    raise ValueError(a) from e\n\n'

def test():
    if False:
        i = 10
        return i + 15
    print(_TEST_CODE)
    print('   ----->    ')
    print(translate_py3to2(_TEST_CODE))
if __name__ == '__main__':
    test()