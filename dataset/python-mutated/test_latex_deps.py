from sympy.external import import_module
from sympy.testing.pytest import ignore_warnings, raises
antlr4 = import_module('antlr4', warn_not_installed=False)
if antlr4:
    disabled = True

def test_no_import():
    if False:
        print('Hello World!')
    from sympy.parsing.latex import parse_latex
    with ignore_warnings(UserWarning):
        with raises(ImportError):
            parse_latex('1 + 1')