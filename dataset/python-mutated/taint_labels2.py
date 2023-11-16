def foo():
    if False:
        print('Hello World!')
    a = source()
    if cond():
        b = a
        b = sanitize()
    else:
        b = a
    sink(b)

def bar():
    if False:
        i = 10
        return i + 15
    a = source()
    if cond():
        b = a
        b = sanitize()
    sink(b)

def baz():
    if False:
        return 10
    a = source()
    if cond():
        b = a
    sink(b)