def test1():
    if False:
        while True:
            i = 10
    x = bar()
    if cond:
        y = 42
    foo(x)

def test2():
    if False:
        return 10
    x = bar()
    while cond:
        y = 42
    foo(x)

def test3():
    if False:
        print('Hello World!')
    x = bar()
    if cond1:
        x = baz()
    foo(x)