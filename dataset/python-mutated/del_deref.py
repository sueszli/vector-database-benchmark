def f():
    if False:
        print('Hello World!')
    x = 1
    y = 2

    def g():
        if False:
            i = 10
            return i + 15
        nonlocal x
        print(y)
        try:
            print(x)
        except NameError:
            print('NameError')

    def h():
        if False:
            print('Hello World!')
        nonlocal x
        print(y)
        try:
            del x
        except NameError:
            print('NameError')
    print(x, y)
    del x
    g()
    h()
f()