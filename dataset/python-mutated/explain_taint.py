def foo():
    if False:
        print('Hello World!')
    a = source()
    b = a
    c = b
    sink(c)

def foo2():
    if False:
        while True:
            i = 10
    a = source()
    b = a
    c = b
    other(c)

def foo2():
    if False:
        i = 10
        return i + 15
    a = other()
    b = a
    c = b
    sink(c)