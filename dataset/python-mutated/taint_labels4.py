def foo():
    if False:
        print('Hello World!')
    a = p()
    if cond():
        b = q(a)
    sink(b)

def bar():
    if False:
        i = 10
        return i + 15
    a = p()
    sink(a)

def baz():
    if False:
        print('Hello World!')
    a = q()
    sink(a)

def boo():
    if False:
        i = 10
        return i + 15
    if cond():
        a = p()
    else:
        a = q()
    sink(a)