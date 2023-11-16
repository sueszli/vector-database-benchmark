def foo():
    if False:
        for i in range(10):
            print('nop')
    a = 1

    def bar():
        if False:
            print('Hello World!')
        nonlocal a
        a = 2