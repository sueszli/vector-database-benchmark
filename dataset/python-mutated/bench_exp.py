from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
(x, y) = symbols('x,y')
e = exp(2 * x)
q = exp(3 * x)

def timeit_exp_subs():
    if False:
        for i in range(10):
            print('nop')
    e.subs(q, y)