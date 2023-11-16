from sympy.core import Symbol, Integer
x = Symbol('x')
i3 = Integer(3)

def timeit_x_is_integer():
    if False:
        i = 10
        return i + 15
    x.is_integer

def timeit_Integer_is_irrational():
    if False:
        return 10
    i3.is_irrational