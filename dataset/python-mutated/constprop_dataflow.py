def test1():
    if False:
        for i in range(10):
            print('nop')
    x = 'x'
    foo(x)
    y = 'y'
    foo(y)
    t = x + y
    foo(t)

def test2(c):
    if False:
        i = 10
        return i + 15
    if c:
        a = 'a'
        foo(a)
    else:
        a = 'b'
        foo(a)
    foo(a)

def test3(c):
    if False:
        for i in range(10):
            print('nop')
    if c:
        x = 'hi'
    foo(x)

def test4(c):
    if False:
        for i in range(10):
            print('nop')
    x = 'hi'
    foo(x)
    while c:
        x = x + ' hi'
        foo(x)
    foo(x)