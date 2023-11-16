from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.trigonometry import trigintegrate
x = Symbol('x')

def timeit_trigintegrate_sin3x():
    if False:
        while True:
            i = 10
    trigintegrate(sin(x) ** 3, x)

def timeit_trigintegrate_x2():
    if False:
        for i in range(10):
            print('nop')
    trigintegrate(x ** 2, x)