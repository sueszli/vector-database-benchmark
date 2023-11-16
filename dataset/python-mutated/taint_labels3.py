def foo():
    if False:
        for i in range(10):
            print('nop')
    a = p()
    if cond():
        b = q(a)
    sink(b)

def bar():
    if False:
        print('Hello World!')
    a = p()
    sink(a)

def baz():
    if False:
        while True:
            i = 10
    a = q()
    sink(a)

def boo():
    if False:
        print('Hello World!')
    if cond():
        a = p()
    else:
        a = q()
    sink(a)