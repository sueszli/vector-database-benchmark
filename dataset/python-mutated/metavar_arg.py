def foo():
    if False:
        print('Hello World!')
    foo(1, 2)
    foo(a_very_long_constant_name, 2)
    foo(unsafe(), 2)
    foo(bar(1, 3), 2)
    foo(2, 1)