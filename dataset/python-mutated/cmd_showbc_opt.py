def f0():
    if False:
        print('Hello World!')
    return 0
    print(1)

def f1(x):
    if False:
        return 10
    if x:
        return
        print(1)
    print(2)

def f2(x):
    if False:
        return 10
    raise Exception
    print(1)

def f3(x):
    if False:
        i = 10
        return i + 15
    while x:
        break
        print(1)
    print(2)

def f4(x):
    if False:
        i = 10
        return i + 15
    while x:
        continue
        print(1)
    print(2)