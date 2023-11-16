def foo():
    if False:
        while True:
            i = 10
    a = source1()
    b = sanitize(a)
    sink1(b)
    sink(b)

def bar():
    if False:
        while True:
            i = 10
    a = source1()
    sanitize()
    eval(a)
    sink(a)

def baz():
    if False:
        return 10
    a = source1()
    b = sanitize()
    eval(b)