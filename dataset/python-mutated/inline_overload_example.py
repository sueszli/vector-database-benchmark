import numba
from numba.extending import overload
from numba import njit, types

def bar(x):
    if False:
        print('Hello World!')
    'A function stub to overload'
    pass

@overload(bar, inline='always')
def ol_bar_tuple(x):
    if False:
        while True:
            i = 10
    if isinstance(x, types.UniTuple):

        def impl(x):
            if False:
                i = 10
                return i + 15
            return x[0]
        return impl

def cost_model(expr, caller, callee):
    if False:
        i = 10
        return i + 15
    return isinstance(caller.typemap[expr.args[0].name], types.Integer)

@overload(bar, inline=cost_model)
def ol_bar_scalar(x):
    if False:
        while True:
            i = 10
    if isinstance(x, types.Number):

        def impl(x):
            if False:
                print('Hello World!')
            return x + 1
        return impl

@njit
def foo():
    if False:
        i = 10
        return i + 15
    a = bar((1, 2, 3))
    b = bar(100)
    c = bar(300j)
    return a + b + c
foo()