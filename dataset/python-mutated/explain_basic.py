def foo():
    if False:
        print('Hello World!')
    foo()
    foo(bar())
    bar()