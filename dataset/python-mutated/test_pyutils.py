from sympy.codegen.ast import Print
from sympy.codegen.pyutils import render_as_module

def test_standard():
    if False:
        print('Hello World!')
    ast = Print('x y'.split(), 'coordinate: %12.5g %12.5g\\n')
    assert render_as_module(ast, standard='python3') == '\n\nprint("coordinate: %12.5g %12.5g\\n" % (x, y), end="")'