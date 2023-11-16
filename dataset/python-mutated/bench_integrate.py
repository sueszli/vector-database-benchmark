from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
x = Symbol('x')

def bench_integrate_sin():
    if False:
        print('Hello World!')
    integrate(sin(x), x)

def bench_integrate_x1sin():
    if False:
        return 10
    integrate(x ** 1 * sin(x), x)

def bench_integrate_x2sin():
    if False:
        print('Hello World!')
    integrate(x ** 2 * sin(x), x)

def bench_integrate_x3sin():
    if False:
        print('Hello World!')
    integrate(x ** 3 * sin(x), x)