from sympy.core.symbol import symbols
from sympy.codegen.abstract_nodes import List

def test_List():
    if False:
        for i in range(10):
            print('nop')
    l = List(2, 3, 4)
    assert l == List(2, 3, 4)
    assert str(l) == '[2, 3, 4]'
    (x, y, z) = symbols('x y z')
    l = List(x ** 2, y ** 3, z ** 4)
    m = l.replace(lambda arg: arg.is_Pow and arg.exp > 2, lambda p: p.base - p.exp)
    assert m == [x ** 2, y - 3, z - 4]
    hash(m)