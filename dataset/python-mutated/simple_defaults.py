def f1(x=12):
    if False:
        return 10
    pass

def f2(x=-12):
    if False:
        return 10
    pass

def f3(x=12):
    if False:
        for i in range(10):
            print('nop')
    pass

def f4(x=-12):
    if False:
        for i in range(10):
            print('nop')
    pass

def f5(x=12.3):
    if False:
        i = 10
        return i + 15
    pass

def f6(x=-12.3):
    if False:
        i = 10
        return i + 15
    pass

def f7(x='asd'):
    if False:
        i = 10
        return i + 15
    pass

def f8(x='asd'):
    if False:
        return 10
    pass

def f9(x='asd'):
    if False:
        for i in range(10):
            print('nop')
    pass

def f10(x=True):
    if False:
        print('Hello World!')
    pass

def f11(x=False):
    if False:
        for i in range(10):
            print('nop')
    pass

def f12(x=3j):
    if False:
        print('Hello World!')
    pass

def f13(x=1 + 2j):
    if False:
        return 10
    pass

def f14(x=1.3 + 2j):
    if False:
        print('Hello World!')
    pass