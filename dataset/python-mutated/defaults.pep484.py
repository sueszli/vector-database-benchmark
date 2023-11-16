def f1(x: e1='foo') -> r1:
    if False:
        print('Hello World!')
    pass

def f2(x: e2='foo') -> r2:
    if False:
        i = 10
        return i + 15
    pass

def f3(x: e3=123) -> r3:
    if False:
        i = 10
        return i + 15
    pass

def f4(x: e4=(1, 2)) -> r4:
    if False:
        print('Hello World!')
    pass

def f5(x: e5=(1,)) -> r5:
    if False:
        while True:
            i = 10
    pass

def f6(x: e6=int) -> r6:
    if False:
        return 10
    pass

def f7(x: int=int):
    if False:
        while True:
            i = 10
    pass

def f8(x: e8={1: 2}) -> r8:
    if False:
        return 10
    pass

def f9(x: e9=[1, 2][:1]) -> r9:
    if False:
        while True:
            i = 10
    pass