def f(*, a):
    if False:
        return 10
    print(a)
f(a=1)

def f(*, a, b):
    if False:
        for i in range(10):
            print('nop')
    print(a, b)
f(a=1, b=2)
f(b=1, a=2)

def f(a, *, b, c):
    if False:
        return 10
    print(a, b, c)
f(1, b=3, c=4)
f(1, c=3, b=4)
f(1, **{'b': '3', 'c': 4})
try:
    f(1)
except TypeError:
    print('TypeError')
try:
    f(1, b=2)
except TypeError:
    print('TypeError')
try:
    f(1, c=2)
except TypeError:
    print('TypeError')

def f(a, *, b, **kw):
    if False:
        for i in range(10):
            print('nop')
    print(a, b, kw)
f(1, b=2)
f(1, b=2, c=3)

def f(*a, b, c):
    if False:
        return 10
    print(a, b, c)
f(b=1, c=2)
f(c=1, b=2)

def f(a, *b, c):
    if False:
        while True:
            i = 10
    print(a, b, c)
f(1, c=2)
f(1, 2, c=3)
f(a=1, c=3)

def f(*, x=lambda : 1):
    if False:
        while True:
            i = 10
    return x()
print(f())
print(f(x=f))
print(f(x=lambda : 2))