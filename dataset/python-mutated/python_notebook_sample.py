a = 1
b = 2

def f(x):
    if False:
        i = 10
        return i + 15
    return x + 1
c = f(b)
a * b + c

def g(x):
    if False:
        i = 10
        return i + 15
    return x + 2
d = 4
a * b + g(c) + d
1 + 1