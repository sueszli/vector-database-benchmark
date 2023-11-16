from sympy.functions.elementary.trigonometric import sin
from sympy.printing.gtk import print_gtk
from sympy.testing.pytest import XFAIL, raises

@XFAIL
def test_1():
    if False:
        for i in range(10):
            print('nop')
    from sympy.abc import x
    print_gtk(x ** 2, start_viewer=False)
    print_gtk(x ** 2 + sin(x) / 4, start_viewer=False)

def test_settings():
    if False:
        return 10
    from sympy.abc import x
    raises(TypeError, lambda : print_gtk(x, method='garbage'))