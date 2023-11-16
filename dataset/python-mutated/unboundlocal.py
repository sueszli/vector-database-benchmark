def f1():
    if False:
        return 10
    print(x)
    x = 1

def f2():
    if False:
        while True:
            i = 10
    for i in range(0):
        print(i)
    print(i)

def check(f):
    if False:
        i = 10
        return i + 15
    try:
        f()
    except NameError:
        print('NameError')
check(f1)
check(f2)