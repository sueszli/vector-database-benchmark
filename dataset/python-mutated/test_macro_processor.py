import pytest
from hy.compiler import HyASTCompiler
from hy.errors import HyMacroExpansionError
from hy.macros import macro, macroexpand
from hy.models import Expression, Float, List, String, Symbol
from hy.reader import read

@macro('test')
def tmac(*tree):
    if False:
        print('Hello World!')
    'Turn an expression into a list'
    return List(tree)

def test_preprocessor_simple():
    if False:
        return 10
    'Test basic macro expansion'
    obj = macroexpand(read('(test "one" "two")'), __name__, HyASTCompiler(__name__))
    assert obj == List([String('one'), String('two')])
    assert type(obj) == List

def test_preprocessor_expression():
    if False:
        print('Hello World!')
    "Test that macro expansion doesn't recurse"
    obj = macroexpand(read('(test (test "one" "two"))'), __name__, HyASTCompiler(__name__))
    assert type(obj) == List
    assert type(obj[0]) == Expression
    assert obj[0] == Expression([Symbol('test'), String('one'), String('two')])
    obj = List([String('one'), String('two')])
    obj = read('(shill ["one" "two"])')[1]
    assert obj == macroexpand(obj, __name__, HyASTCompiler(__name__))

def test_preprocessor_exceptions():
    if False:
        i = 10
        return i + 15
    'Test that macro expansion raises appropriate exceptions'
    with pytest.raises(HyMacroExpansionError) as excinfo:
        macroexpand(read('(when)'), __name__, HyASTCompiler(__name__))
    assert 'TypeError: when()' in excinfo.value.msg

def test_macroexpand_nan():
    if False:
        while True:
            i = 10
    import math
    NaN = float('nan')
    x = macroexpand(Float(NaN), __name__, HyASTCompiler(__name__))
    assert type(x) is Float
    assert math.isnan(x)

def test_macroexpand_source_data():
    if False:
        i = 10
        return i + 15
    ast = Expression([Symbol('when'), String('a')])
    ast.start_line = 3
    ast.start_column = 5
    bad = macroexpand(ast, 'hy.core.macros', once=True)
    assert bad.start_line == 3
    assert bad.start_column == 5