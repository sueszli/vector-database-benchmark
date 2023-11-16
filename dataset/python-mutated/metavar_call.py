def foo():
    if False:
        for i in range(10):
            print('nop')
    foo(1, 2)
    return 1