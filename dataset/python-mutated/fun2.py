def f(x):
    if False:
        return 10
    print(x + 1)

def g(x):
    if False:
        i = 10
        return i + 15
    f(2 * x)
    f(4 * x)
g(3)