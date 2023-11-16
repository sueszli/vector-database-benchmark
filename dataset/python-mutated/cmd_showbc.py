def f():
    if False:
        print('Hello World!')
    a = None + False + True
    a = 0
    a = 1000
    a = -1000
    a = 1
    b = (1, 2)
    c = [1, 2]
    d = {1, 2}
    e = {}
    f = {1: 2}
    g = 'a'
    h = b'a'
    i = 1
    j = 2
    k = a + b
    l = -a
    m = not a
    m = a == b == c
    m = not (a == b and b == c)
    n = b.c
    b.c = n
    p = b[0]
    b[0] = p
    b[0] += p
    a = b[:]
    (a, b) = c
    (a, *a) = a
    (a, b) = (b, a)
    (a, b, c) = (c, b, a)
    del a
    global gl
    gl = a
    del gl
    a = (b for c in d if e)
    a = [b for c in d if e]
    a = {b: b for c in d if e}
    a()
    a(1)
    a(b=1)
    a(*b)
    a.b()
    a.b(1)
    a.b(c=1)
    a.b(*c)
    if a:
        x
    else:
        y
    while a:
        b
    while not a:
        b
    a = a or a
    for a in b:
        c
    try:
        while a:
            break
    except:
        b
    finally:
        c
    while a:
        try:
            break
        except:
            pass
    with a:
        b
    x = 1

    def closure():
        if False:
            i = 10
            return i + 15
        nonlocal x
        a = x + 1
        x = 1
        del x
    import a
    from a import b
    if a:
        raise
    if a:
        raise 1
    if a:
        return
    if a:
        return 1

def f():
    if False:
        i = 10
        return i + 15
    l1 = l2 = l3 = l4 = l5 = l6 = l7 = l8 = l9 = l10 = 1
    m1 = m2 = m3 = m4 = m5 = m6 = m7 = m8 = m9 = m10 = 2
    l10 + m10

def f(a=1):
    if False:
        print('Hello World!')
    pass

    def f(b=2):
        if False:
            for i in range(10):
                print('nop')
        return b + a

def f():
    if False:
        for i in range(10):
            print('nop')
    yield
    yield 1
    yield from 1

class Class:
    pass
del Class

def f(self):
    if False:
        for i in range(10):
            print('nop')
    super().f()
from sys import *