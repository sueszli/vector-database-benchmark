def f():
    if False:
        for i in range(10):
            print('nop')
    print(x)
    global x
    print(x)

def f():
    if False:
        i = 10
        return i + 15
    global x
    print(x)
    global x
    print(x)

def f():
    if False:
        while True:
            i = 10
    print(x)
    global x, y
    print(x)

def f():
    if False:
        print('Hello World!')
    global x, y
    print(x)
    global x, y
    print(x)

def f():
    if False:
        for i in range(10):
            print('nop')
    x = 1
    global x
    x = 1

def f():
    if False:
        i = 10
        return i + 15
    global x
    x = 1
    global x
    x = 1

def f():
    if False:
        for i in range(10):
            print('nop')
    del x
    global x, y
    del x

def f():
    if False:
        for i in range(10):
            print('nop')
    global x, y
    del x
    global x, y
    del x

def f():
    if False:
        while True:
            i = 10
    del x
    global x
    del x

def f():
    if False:
        while True:
            i = 10
    global x
    del x
    global x
    del x

def f():
    if False:
        for i in range(10):
            print('nop')
    del x
    global x, y
    del x

def f():
    if False:
        i = 10
        return i + 15
    global x, y
    del x
    global x, y
    del x

def f():
    if False:
        return 10
    print(f'x={x!r}')
    global x

def f():
    if False:
        i = 10
        return i + 15
    global x
    print(x)

def f():
    if False:
        while True:
            i = 10
    global x, y
    print(x)

def f():
    if False:
        print('Hello World!')
    global x
    x = 1

def f():
    if False:
        while True:
            i = 10
    global x, y
    x = 1

def f():
    if False:
        i = 10
        return i + 15
    global x
    del x

def f():
    if False:
        print('Hello World!')
    global x, y
    del x

def f():
    if False:
        return 10
    global x
    print(f'x={x!r}')