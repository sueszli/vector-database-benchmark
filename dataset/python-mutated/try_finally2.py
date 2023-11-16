def f1(a, b):
    if False:
        while True:
            i = 10
    pass

def test1():
    if False:
        print('Hello World!')
    val = 1
    try:
        raise ValueError()
    finally:
        f1(2, 2)
        print(val)
try:
    test1()
except ValueError:
    pass

def f2(a, b, c):
    if False:
        i = 10
        return i + 15
    pass

def test2():
    if False:
        while True:
            i = 10
    val = 1
    try:
        raise ValueError()
    finally:
        f2(2, 2, 2)
        print(val)
try:
    test2()
except ValueError:
    pass