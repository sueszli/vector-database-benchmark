from contextlib import contextmanager
l = 0
I = 0
O = 0
l: int = 0
(a, l) = (0, 1)
[a, l] = (0, 1)
(a, *l) = (0, 1, 2)
a = l = 0
o = 0
i = 0
for l in range(3):
    pass
for (a, l) in zip(range(3), range(3)):
    pass

def f1():
    if False:
        for i in range(10):
            print('nop')
    global l
    l = 0

def f2():
    if False:
        print('Hello World!')
    l = 0

    def f3():
        if False:
            for i in range(10):
                print('nop')
        nonlocal l
        l = 1
    f3()
    return l

def f4(l, /, I):
    if False:
        return 10
    return (l, I, O)

def f5(l=0, *, I=1):
    if False:
        while True:
            i = 10
    return (l, I)

def f6(*l, **I):
    if False:
        while True:
            i = 10
    return (l, I)

@contextmanager
def ctx1():
    if False:
        return 10
    yield 0
with ctx1() as l:
    pass

@contextmanager
def ctx2():
    if False:
        print('Hello World!')
    yield (0, 1)
with ctx2() as (a, l):
    pass
try:
    pass
except ValueError as l:
    pass
if (l := 5) > 0:
    pass