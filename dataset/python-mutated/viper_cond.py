@micropython.viper
def f():
    if False:
        for i in range(10):
            print('nop')
    x = False
    if x:
        pass
    else:
        print('not x', x)
f()

@micropython.viper
def f():
    if False:
        i = 10
        return i + 15
    x = True
    if x:
        print('x', x)
f()

@micropython.viper
def g():
    if False:
        i = 10
        return i + 15
    y = 1
    if y:
        print('y', y)
g()

@micropython.viper
def h():
    if False:
        for i in range(10):
            print('nop')
    z = 65536
    if z:
        print('z', z)
h()