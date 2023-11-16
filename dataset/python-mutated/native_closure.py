@micropython.native
def f():
    if False:
        while True:
            i = 10
    x = 1

    @micropython.native
    def g():
        if False:
            for i in range(10):
                print('nop')
        nonlocal x
        return x
    return g
print(f()())

@micropython.native
def f(x):
    if False:
        i = 10
        return i + 15

    @micropython.native
    def g():
        if False:
            for i in range(10):
                print('nop')
        nonlocal x
        return x
    return g
print(f(2)())

@micropython.native
def f(x):
    if False:
        i = 10
        return i + 15
    y = 2 * x

    @micropython.native
    def g(z):
        if False:
            while True:
                i = 10
        return x + y + z
    return g
print(f(2)(3))