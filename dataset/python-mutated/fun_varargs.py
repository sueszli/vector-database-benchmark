def f1(*args):
    if False:
        i = 10
        return i + 15
    print(args)
f1()
f1(1)
f1(1, 2)

def f2(a, *args):
    if False:
        while True:
            i = 10
    print(a, args)
f2(1)
f2(1, 2)
f2(1, 2, 3)

def f3(a, b, *args):
    if False:
        while True:
            i = 10
    print(a, b, args)
f3(1, 2)
f3(1, 2, 3)
f3(1, 2, 3, 4)

def f4(a=0, *args):
    if False:
        return 10
    print(a, args)
f4()
f4(1)
f4(1, 2)
f4(1, 2, 3)

def f5(a, b=0, *args):
    if False:
        while True:
            i = 10
    print(a, b, args)
f5(1)
f5(1, 2)
f5(1, 2, 3)
f5(1, 2, 3, 4)