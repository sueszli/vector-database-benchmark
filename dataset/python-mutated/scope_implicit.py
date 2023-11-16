def f():
    if False:
        i = 10
        return i + 15

    def g():
        if False:
            while True:
                i = 10
        return x
    x = 3
    return g
print(f()())

def f():
    if False:
        return 10

    def g():
        if False:
            while True:
                i = 10

        def h():
            if False:
                for i in range(10):
                    print('nop')
            return x
        return h
    x = 4
    return g
print(f()()())

def f():
    if False:
        i = 10
        return i + 15
    x = 0

    def g():
        if False:
            for i in range(10):
                print('nop')
        x
        x = 1
    g()
try:
    f()
except NameError:
    print('NameError')