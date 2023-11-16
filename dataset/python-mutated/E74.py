lambda l: dict(zip(l, range(len(l))))

def f(l):
    if False:
        i = 10
        return i + 15
    print(l, l, l)
x = (lambda l: dict(zip(l, range(len(l)))),)
x = (lambda l: dict(zip(l, range(len(l)))), lambda l: dict(zip(l, range(len(l)))))