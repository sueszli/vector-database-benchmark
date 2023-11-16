def f():
    if False:
        i = 10
        return i + 15
    x = 1

    def g(z):
        if False:
            return 10
        print(x, z)
    return g
f()(z=42)