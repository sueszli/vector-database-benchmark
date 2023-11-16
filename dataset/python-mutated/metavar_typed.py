def foo():
    if False:
        return 10
    foo(1)
    foo(1.0)
    foo('')