def f():
    if False:
        while True:
            i = 10
    return
print(f())

def g():
    if False:
        i = 10
        return i + 15
    return 1
print(g())

def f(x):
    if False:
        i = 10
        return i + 15
    return 1 if x else 2
print(f(0), f(1))