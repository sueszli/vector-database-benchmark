def foo():
    if False:
        while True:
            i = 10
    foo()
    bar()
    foo()
    x = bar()
    foo()
    foo2(bar())
    foo()
    return bar()