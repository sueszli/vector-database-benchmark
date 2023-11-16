def foo():
    if False:
        return 10
    foo()
    bar()
    g()
    baz()

def bar():
    if False:
        i = 10
        return i + 15
    quux()
    bar()
    f()
    foo()