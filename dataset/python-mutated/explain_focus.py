def foo():
    if False:
        for i in range(10):
            print('nop')
    foo(1, 2, 3)
    foo(2, 2, 2)
    foo(0, 1)