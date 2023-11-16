def f():
    if False:
        for i in range(10):
            print('nop')
    foo(1)
    bar(2)
    baz(3)

def g():
    if False:
        i = 10
        return i + 15
    foo(3)
    meh(2)
    baz(1)

def h():
    if False:
        while True:
            i = 10
    baz(1)
    bar(2)
    foo(3)