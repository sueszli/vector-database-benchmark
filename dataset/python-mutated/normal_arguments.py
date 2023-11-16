def simple(a):
    if False:
        while True:
            i = 10
    return a
simple(1)
simple()
simple(1, 2)
simple(1, 2, 3)
simple(a=1)
simple(b=1)
simple(1, a=1)

def two_params(x, y):
    if False:
        while True:
            i = 10
    return y
two_params(y=2, x=1)
two_params(1, y=2)
two_params(1, x=2)
two_params(1, 2, y=3)

def default(x, y=1, z=2):
    if False:
        return 10
    return x
default()
default(1)
default(1, 2)
default(1, 2, 3)
default(1, 2, 3, 4)
default(x=1)

class Instance:

    def __init__(self, foo):
        if False:
            print('Hello World!')
        self.foo = foo
Instance(1).foo
Instance(foo=1).foo
Instance(1, 2).foo
Instance().foo