def f1(x):
    if False:
        i = 10
        return i + 15
    return 1

def f1(x):
    if False:
        i = 10
        return i + 15
    return 'foo'

def f2(x):
    if False:
        for i in range(10):
            print('nop')
    pass

def f2(x, y):
    if False:
        while True:
            i = 10
    pass

def f3(x):
    if False:
        i = 10
        return i + 15
    return 'asd' + x

def f3(x):
    if False:
        while True:
            i = 10
    return 1 + x