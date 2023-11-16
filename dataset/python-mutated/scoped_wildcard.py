foo(x)
foo(A.x)
foo(y)

def foo():
    if False:
        i = 10
        return i + 15
    from A import *
    foo(x)
    foo(A.x)
    foo(y)