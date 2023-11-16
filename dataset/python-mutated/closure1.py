def f(x):
    if False:
        for i in range(10):
            print('nop')
    y = 2 * x

    def g(z):
        if False:
            for i in range(10):
                print('nop')
        return y + z
    return g
print(f(1)(1))
x = f(2)
y = f(3)
print(x(1), x(2), x(3))
print(y(1), y(2), y(3))
print(x(1), x(2), x(3))
print(y(1), y(2), y(3))