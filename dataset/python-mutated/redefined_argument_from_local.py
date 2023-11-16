def func(a):
    if False:
        for i in range(10):
            print('nop')
    for b in range(1):
        ...

def func(a):
    if False:
        i = 10
        return i + 15
    try:
        ...
    except ValueError:
        ...
    except KeyError:
        ...
if True:

    def func(a):
        if False:
            return 10
        ...
else:
    for a in range(1):
        print(a)

def func(a):
    if False:
        return 10
    for a in range(1):
        ...

def func(i):
    if False:
        while True:
            i = 10
    for i in range(10):
        print(i)

def func(e):
    if False:
        i = 10
        return i + 15
    try:
        ...
    except Exception as e:
        print(e)

def func(f):
    if False:
        while True:
            i = 10
    with open('') as f:
        print(f)

def func(a, b):
    if False:
        return 10
    with context() as (a, b, c):
        print(a, b, c)

def func(a, b):
    if False:
        i = 10
        return i + 15
    with context() as [a, b, c]:
        print(a, b, c)

def func(a):
    if False:
        print('Hello World!')
    with open('foo.py') as f, open('bar.py') as a:
        ...

def func(a):
    if False:
        i = 10
        return i + 15

    def bar(b):
        if False:
            for i in range(10):
                print('nop')
        for a in range(1):
            print(a)

def func(a):
    if False:
        return 10

    def bar(b):
        if False:
            print('Hello World!')
        for b in range(1):
            print(b)

def func(a=1):
    if False:
        for i in range(10):
            print('nop')

    def bar(b=2):
        if False:
            i = 10
            return i + 15
        for a in range(1):
            print(a)
        for b in range(1):
            print(b)